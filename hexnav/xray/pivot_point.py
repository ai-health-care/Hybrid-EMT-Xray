import json
from pathlib import Path

import click

import cv2

import numpy as np
import pandas as pd
import scipy
from sympy.parsing.sympy_parser import parse_expr

from ..paths import WEIGHTS_PATH


class PivotPointDataRecord:
    def __init__(self, image_filename: Path, gantry_angle: float, gantry_height: float):
        self._image_filename = image_filename
        self._gantry_angle = gantry_angle
        self._gantry_height = gantry_height

    @staticmethod
    def from_dict(d: dict):
        for attribute in ('image_filename', 'gantry_angle', 'gantry_height'):
            if attribute not in d:
                raise RuntimeError(f'Attribute "{attribute}" missing.')
        return PivotPointDataRecord(
            d['image_filename'],
            d['gantry_angle'],
            d['gantry_height']
        )

    def to_dict(self):
        return {
            'image_filename': self._image_filename,
            'gantry_angle': self._gantry_angle,
            'gantry_height': self._gantry_height
        }


def infer_centerline_x(image_filename: Path, N_keypoints=6, min_threshold=127, max_threshold=255):
    """
    From a bright image with three dark dots on a straight line (in y-direction, same x),
    determine the centers of each blob and infer their common x-coordinate.
    """
    image = cv2.imread(str(image_filename), cv2.IMREAD_GRAYSCALE)
    detector_parameters = cv2.SimpleBlobDetector_Params()
    detector_parameters.blobColor = 0
    detector_parameters.filterByArea = True
    detector_parameters.minArea = 100
    detector_parameters.maxArea = 2000
    detector_parameters.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(detector_parameters)
    keypoints = detector.detect(image)
    if len(keypoints) > N_keypoints:
        raise RuntimeError('Detected more or less than 3 dark blobs.')
    X = [keypoint.pt[0] for keypoint in keypoints]
    if len(X) < 2:
        return np.nan
    X_mean = np.mean(X)
    # TODO: correct for skew
    common_x = X_mean
    return common_x


def create_pivot_dataframe(dataset_path: Path):
    """
    """
    json_path = dataset_path / 'data.json'
    records = []
    with open(json_path, 'r') as f:
        record_dicts = json.load(f)
        for record_dict in record_dicts:
            record = PivotPointDataRecord.from_dict(record_dict)
            records.append(record.to_dict())
    df = pd.DataFrame(records)

    def __catch_wrapper(image_filename):
        image_path = Path(dataset_path) / image_filename
        try:
            return infer_centerline_x(str(image_path))
        except RuntimeError as e:
            click.secho(f'Could not infer x-coordinate from "{image_path}"', fg='red')

    df['center_x'] = [
        __catch_wrapper(image_filename) for image_filename in df.image_filename
    ]
    return df


def target_sinusoidal(x, *args):
    A, w, p, c = args
    return A * np.sin(w * x + p) + c


def evaluate_pivot_function(target_str, height=None):
    """
    Evaluate a function that maps gantry angles to pivot x coordinates.

    :param height: gantry height (optional, if fitted on gantry height)
    """
    target_expr = parse_expr(target_str)
    if height is not None:
        target_expr = target_expr.subs('height', height)

    def target_func(X, *args):
        # substitute coefficients first
        symbols = sorted([
            str(s)
            for s in list(target_expr.free_symbols)
            if str(s) != 'x'
        ])
        subs = { symbol: args[i] for i, symbol in enumerate(symbols) }
        expr = target_expr.subs(subs)
        if np.isscalar(X):
            return float(expr.evalf(subs={'x': X}))
        else:
            return np.array([
                float(expr.evalf(subs={'x': x}))
                for x in X
            ])
    return target_func


def learn_pivot_function(df: pd.DataFrame):
    # fit sinusoidal function over gantry angle vs. inferred center x
    X = df[~np.isnan(df.center_x)].gantry_angle.values
    Y = df[~np.isnan(df.center_x)].center_x.values
    a = np.max(Y) - np.min(Y)
    d = np.min(Y) + a / 2

    target_init = np.array([0, 0, 0, d])
    target_str = 'a * x**3 + b * x**2 + c * x + d'
    #target_str = 'a * sin(b * x + c) + d'

    coefficients, _ = scipy.optimize.curve_fit(
        evaluate_pivot_function(target_str),
        X, Y,
        p0=target_init
    )
    return coefficients, target_str, target_init


def learn_pivot_family(df: pd.DataFrame):
    X = df[~np.isnan(df.center_x)][['gantry_height', 'gantry_angle']].values
    Y = df[~np.isnan(df.center_x)].center_x.values
    a = np.max(Y) - np.min(Y)
    d = np.min(Y) + a / 2

    target_init = np.array([1, 1, 1, d, 1, 1, 1])
    target_str = 'a * (x - ha * height)**3 + b * (x - ha * height)**2 + c * (x - ha * height) + d + hb * height + hc * height ** 2'

    #target_init = np.array([a, 0, 0, d, 1, 1, 1])
    #target_str = 'a * sin(b * (x - ha * height) + c) + d + hb * height + hc * height ** 2'

    def evaluate_error(args):
        heights = X[:, 0]
        angles = X[:, 1]
        E = np.array([])
        for height in np.unique(heights):
            target = evaluate_pivot_function(target_str, height)
            result = target(angles[heights == height], *args)
            error = result - Y[heights == height]
            E = np.append(E, error)
        return np.mean(np.square(E))

    result = scipy.optimize.minimize(
        evaluate_error,
        x0=target_init
    )
    return result.x, target_str, target_init


def save_pivot_functions(name, coefficients, target_str, target_init):
    path = WEIGHTS_PATH / 'pivot_point'
    path.mkdir(exist_ok=True)
    d = {
        'coefficients': coefficients,
        'target': target_str,
        'init': target_init
    }
    with open(path / name, 'w') as f:
        json.dump(d, f)


def load_pivot_functions(name):
    path = WEIGHTS_PATH / 'pivot_point'
    with open(path / name, 'r') as f:
        d = json.load(f)
    return d['coefficients'], d['target'], d['init']