import json
import time

import pandas as pd
import numpy as np
import scipy.optimize as optim

from .constants import *
from .recalibration import extract_xray_points
from .orchestration import NavigationData
from .utils import sliding_window, cluster_df


def preprocess_emt_data(df, convert_to_mm=True, reverse_xy=True):
    """
    Preprocess data recorded by EMT.
    """
    ret = df.copy()
    if convert_to_mm:
        ret[['x', 'y', 'z']] *= 25.4
    # eliminate noise by applying a sliding window filter
    ret = sliding_window(ret)
    # translate to image center (part of registration)
    image_center = np.array([
        DEFAULT_CROP[2] // 2 * PX2MM,
        DEFAULT_CROP[3] // 2 * PX2MM
    ])
    ret[['x', 'y']] += image_center
    if reverse_xy:
        A_reverse_xy = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        ret[['x', 'y', 'z']] = ret[['x', 'y', 'z']].values.dot(A_reverse_xy)
    return ret


def load_compensation_data(data_path):
    training_path = data_path / 'training'
    validation_path = data_path / 'validation'

    training_data = NavigationData()
    for experiment in training_path.glob('*'):
        d = NavigationData.load(experiment)
        training_data = NavigationData.merge(training_data, d)
    training_data.measurement_name = 'emt_compensation_training_data'

    validation_data = NavigationData()
    for experiment in validation_path.glob('*'):
        d = NavigationData.load(experiment)
        validation_data = NavigationData.merge(validation_data, d)
    validation_data.measurement_name = 'emt_compensation_validation_data'

    df_emt_training = pd.DataFrame(
        np.array(training_data.emt_data)[..., 1:], # cut stray index
        columns=['time', 'x', 'y', 'z']
    )
    df_emt_training = preprocess_emt_data(df_emt_training)

    df_emt_validation = pd.DataFrame(
        np.array(validation_data.emt_data)[..., 1:],
        columns=['time', 'x', 'y', 'z']
    )
    df_emt_validation = preprocess_emt_data(df_emt_validation)

    _, _, _, _, labels_training = cluster_df(df_emt_training, epsilon=1.0, minimum_samples=100)
    _, _, _, _, labels_validation = cluster_df(df_emt_validation, epsilon=1.0, minimum_samples=100)

    df_xray_training = extract_xray_points(training_data, training_path)
    df_xray_validation = extract_xray_points(validation_data, validation_path)

    X_emt_training = df_emt_training.groupby(labels_training).mean().iloc[1:] # remove "outlier" cluster
    X_emt_training['time'] -= X_emt_training['time'].min()
    X_emt_training.columns = X_emt_training.columns.droplevel(1)
    X_emt_training = X_emt_training.sort_values('time')
    X_emt_training = X_emt_training[['x', 'y', 'z']].values

    X_emt_validation = df_emt_validation.groupby(labels_validation).mean().iloc[1:] # remove "outlier" cluster
    X_emt_validation['time'] -= X_emt_validation['time'].min()
    X_emt_validation.columns = X_emt_validation.columns.droplevel(1)
    X_emt_validation = X_emt_validation.sort_values('time')
    X_emt_validation = X_emt_validation[['x', 'y', 'z']].values

    X_xray_training = df_xray_training
    X_xray_training['time'] -= X_xray_training['time'].min()
    X_xray_training = X_xray_training.sort_values('time')
    X_xray_training = X_xray_training[['x', 'y', 'z']].values

    X_xray_validation = df_xray_validation
    X_xray_validation['time'] -= X_xray_validation['time'].min()
    X_xray_validation = X_xray_validation.sort_values('time')
    X_xray_validation = X_xray_validation[['x', 'y', 'z']].values

    return X_emt_training, X_emt_validation, X_xray_training, X_xray_validation


class LinearCompensation:
    def __init__(self, dim=3):
        self.A = np.eye(dim)
        self.B = np.zeros(dim)
        self.dim = dim

    def fit(self, X_emt_training, X_xray_training, mask=None):
        if mask is None:
            mask = np.ones(len(X_emt_training)).astype(bool)

        self.loc = np.min(X_xray_training, axis=0)
        self.scale = np.max(X_xray_training, axis=0) - self.loc
        X_xray = (np.copy(X_xray_training) - self.loc) / self.scale
        X_emt = (np.copy(X_emt_training) - self.loc) / self.scale

        def __objective_linear(X, *args):
            nonlocal mask
            nonlocal X_xray
            nonlocal X_emt
            A = np.reshape(X[0:9], (3, 3))
            B = X[9:]
            E = (X_emt[mask].dot(A) + B) - X_xray[mask]
            return np.mean(np.square(E))

        res = optim.minimize(__objective_linear, np.array([
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                0, 0, 0
            ])
        )
        self.A = np.reshape(res.x[:9], (3, 3))
        self.B = res.x[9:]

    def apply(self, X):
        comp = ((X - self.loc) / self.scale).dot(self.A) + self.B
        return (comp * self.scale) + self.loc

    @staticmethod
    def load(filename):
        comp = LinearCompensation()
        with open(filename, 'r') as f:
            d = json.load(f)
        comp.A = d.get('A', np.eye(3))
        comp.B = d.get('B', np.zeros(3))
        comp.dim = len(comp.B)
        return comp

    def save(self, filename):
        d = {
            'time': time.time(),
            'A': self.A,
            'B': self.B
        }
        with open(filename, 'w') as f:
            json.dump(d, f)


class CubicCompensation:
    def __init__(self, dim=3):
        self.loc = None
        self.scale = None
        self.A = np.zeros((dim, dim))
        self.B = np.zeros((dim, dim))
        self.C = np.eye(dim)
        self.D = np.zeros(dim)
        self.dim = dim

    def fit(self, X_emt_training, X_xray_training, mask=None):
        if mask is None:
            mask = np.ones(len(X_emt_training)).astype(bool)

        linear_compensation = LinearCompensation()
        linear_compensation.fit(X_emt_training, X_xray_training, mask)

        self.loc = np.min(X_xray_training, axis=0)
        self.scale = np.max(X_xray_training, axis=0) - self.loc
        X_xray = (np.copy(X_xray_training) - self.loc) / self.scale
        X_emt = (np.copy(X_emt_training) - self.loc) / self.scale

        def __objective_cubic(X, *args):
            nonlocal mask
            nonlocal X_xray
            nonlocal X_emt
            A = np.reshape(X[0:9], (3, 3))
            B = np.reshape(X[9:18], (3, 3))
            C = np.reshape(X[18:27], (3, 3))
            D = X[27:]
            E = (
                np.power(X_emt[mask], 3).dot(A) + \
                np.square(X_emt[mask]).dot(B) + \
                X_emt[mask].dot(C) + \
                D
            ) - X_xray[mask]
            return np.mean(np.square(E))

        # initialize cubic compensation with linear compensation
        A = linear_compensation.A
        B = linear_compensation.B
        res = optim.minimize(__objective_cubic, np.array([
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                *(A.flatten()),
                *B
            ])
        )
        self.A = np.reshape(res.x[0:9], (3, 3))
        self.B = np.reshape(res.x[9:18], (3, 3))
        self.C = np.reshape(res.x[18:27], (3, 3))
        self.D = res.x[27:]

    def apply(self, X):
        x = (X - self.loc) / self.scale
        comp = np.power(x, 3).dot(self.A) + \
               np.power(x, 2).dot(self.B) + \
               x.dot(self.C) + \
               self.D
        return (comp * self.scale) + self.loc