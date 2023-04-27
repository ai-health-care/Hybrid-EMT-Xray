import os

import cv2
import numpy as np
import pandas as pd

from .constants import PX2MM
from .paths import MEASUREMENT_PATH
from .xray.sensordetection import detect_sensor


def extract_xray_points(navigation_data, measurement_root=MEASUREMENT_PATH):
    """
    Extract all sensor coordinates from a measurement.
    Store sensor coordinates together with timestamp, gantry angle and gantry height.
    """
    time = [ xray['time'] for xray in navigation_data.xray_data ]
    image_filenames = [
        os.path.join(measurement_root, xray['measurement'], 'xrays', xray['image'])
        for xray in navigation_data.xray_data
    ]
    images = [ cv2.imread(filename) for filename in image_filenames ]
    sensors = []
    for i, image in enumerate(images):
        box, _ = detect_sensor(image)
        if box is None or len(box) == 0:
            print(f'sensor not found in {image_filenames[i]}')
            box = np.array([-1, -1, -1, -1])
        sensors.append(box)
    sensors = np.array(sensors)
    gantry_angles = [ xray['angle'] for xray in navigation_data.xray_data ]
    gantry_heights = [ xray['height'] for xray in navigation_data.xray_data ]
    z = [ xray.get('z', 0) for xray in navigation_data.xray_data ]

    return pd.DataFrame({
        'time': time,
        'x': sensors[:, 0] * PX2MM,
        'y': sensors[:, 1] * PX2MM,
        'z': z,
        'gantry_angle': gantry_angles,
        'gantry_height': gantry_heights
    })


def infer_z(sensor_90, sensor_other, angle_deg_1, angle_deg_2):
    """
    Infer z-coordinate by calculating the intersection of two rays.
    """
    angle_rad_1 = np.deg2rad(angle_deg_1)
    angle_rad_2 = np.deg2rad(angle_deg_2)

    rot_y_1 = np.array([
        [np.cos(angle_rad_1), -np.sin(angle_rad_1)],
        [np.sin(angle_rad_1), np.cos(angle_rad_1)]
    ])
    
    rot_y_2 = np.array([
        [np.cos(angle_rad_2), -np.sin(angle_rad_2)],
        [np.sin(angle_rad_2), np.cos(angle_rad_2)]
    ])

    normal_90 = rot_y_1.dot(np.array([0, 1]))
    normal_other = rot_y_2.dot(np.array([0, 1]))

    A = np.vstack((normal_90, -normal_other)).T
    point_90 = np.copy(sensor_90)
    point_other = np.copy(sensor_other)
    B = point_other - point_90
    R = np.linalg.inv(A).dot(B)
    intersect = point_90 + normal_90 * R[0]
    return intersect[1]