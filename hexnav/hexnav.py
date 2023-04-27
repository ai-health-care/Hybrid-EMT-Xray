#!/usr/bin/env python3
import os
from pathlib import Path
import time

import cv2

import click

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')

import numpy as np

from .subsystems.carmservo import CarmServo, VirtualCarmServo
from .subsystems.framegrabber import crop_xray
from .paths import DATA_PATH
from .xray.pivot_point import create_pivot_dataframe, evaluate_pivot_function, learn_pivot_family
from .orchestration import Navigation, NavigationData
from .subsystems.linearguide import LinearGuide
from .subsystems.igttracker import IGTLinkTracker
from .subsystems.footpedal import Footpedal


@click.group()
def main():
    pass


@main.command()
@click.argument('filename', type=click.Path(exists=True, dir_okay=True, file_okay=True))
def crop(filename):
    # TODO: batch crop if input is directory
    filenames = []
    if filename.endswith('.png'):
        filenames = [filename]
    else:
        filenames = [
            filename + '/' + f
            for f in os.listdir(filename)
            if f.endswith('.png')
        ]
    for image_filename in filenames:
        print(image_filename)
        xray = cv2.imread(image_filename)
        old_height, old_width, _ = xray.shape
        xray_cropped, bounds = crop_xray(xray)
        click.secho(f'{bounds}')
        cv2.imwrite(image_filename, xray_cropped)

        bbox_filename = image_filename[:-len('.png')] + '.txt'
        #try:
        with open(bbox_filename, 'r') as f:
            text = f.read()
        lines = text.splitlines()
        if lines:
            klass, x0, y0, w0, h0 = [float(t) for t in lines[0].split(' ')]
            x = (x0 * old_width - bounds[0]) / bounds[2]
            y = (y0 * old_height - bounds[1]) / bounds[3]
            new_bbox = f'{int(klass)} {x} {y} {w0} {h0}'
            with open(bbox_filename, 'w') as f:
                f.write(new_bbox)


@main.command()
@click.argument('measurement')
def analyze(measurement):
    """ Analyze recorded data in retrospective study.
    """
    navigation_data = NavigationData.load(measurement)


@main.command()
@click.option('--steps', '-s', type=int, default=10)
@click.option('--height', '-h', type=int, default=10, help='C-arm gantry height')
@click.option('--stepsize', '-S', type=int, default=10)
def autonav(steps, height, stepsize):
    """ Automated navigation experiment.
    """
    navigation = Navigation(height)
    navigation.record_line(steps, step_size=stepsize)
    navigation.save_data()


@main.command()
@click.option('--steps', '-s', type=int, default=10, help='Number of points on the calibration phantom')
@click.option('--height', '-h', type=int, default=10, help='C-arm gantry height')
@click.option('--sensor_z', '-z', type=float, default=0, help='z-elevation of sensor')
def compensationdata(steps, height, sensor_z):
    """ Record data for EMT compensation.
    """
    navigation = Navigation(height, use_pedal=True)
    navigation.record_compensation(steps, sensor_z=sensor_z)
    navigation.save_data()


@main.command()
@click.option('--steps', '-s', default=1, type=int)
@click.option('--motor', '-m', default=0, type=int)
def step(steps, motor):
    linear_guide = LinearGuide()
    linear_guide.move_stepper(steps, motor)


@main.command()
def snapshot():
    """ Trigger an X-ray snapshot. """
    fp = Footpedal()
    fp.trigger_xray()


@main.command()
def servo():
    s = CarmServo()
    s.rotate(90)
    time.sleep(2)
    s.rotate(105)
    time.sleep(2)
    s.rotate(90)
    time.sleep(2)
    s.rotate(75)


@main.command()
def check():
    """ Perform system check.
    """
    tracker = IGTLinkTracker()
    for i in range(1000):
        timestamp, positions = tracker.measure()
        vector = np.append(timestamp, positions)
        print(vector)


@main.command()
def pivot():
    """ Infer pivot point function from pre-recorded data. """
    pivot_path = Path(DATA_PATH) / 'pivot_point'
    df = create_pivot_dataframe(pivot_path)

    learn_pivot_family(df)
    coefficients, target_str, _ = learn_pivot_family(df)
    
    cmap = plt.get_cmap('tab20')
    for i, gantry_height in enumerate(set(df.gantry_height)):
        color_dark = cmap(2 * i)
        color_light = cmap(2 * i + 1)
        df_h = df[df.gantry_height == gantry_height]
        df_h = df_h[~np.isnan(df_h.center_x)]
        print(df_h)
        plt.plot(df_h.gantry_angle, df_h.center_x, linestyle='--', color=color_light)
        plt.scatter(df_h.gantry_angle, df_h.center_x, label=f'height={gantry_height} cm', color=color_dark)
        X = df_h.gantry_angle.values
        Y = df_h.center_x.values
        X_fit = np.linspace(np.min(X) - 20, np.max(X) + 20, 500)
        Y_fit = evaluate_pivot_function(target_str, gantry_height)(X_fit, *coefficients)
        bins = np.arange(0, 720)
        plt.plot(X_fit, np.digitize(Y_fit, bins), label=f'height={gantry_height} cm, curve fit', color=color_dark)

    plt.title('Pivot X determined by calibration phantom')
    plt.xlabel('Gantry Angle [°]')
    plt.ylabel('Pivot X [pixels]')
    plt.legend()
    sns.despine()
    plt.show()


@main.command()
def testdetection():
    pass


if __name__ == '__main__':
    main()