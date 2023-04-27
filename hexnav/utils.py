import subprocess
import time

import click
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN


def say(string, use_tts=True):
    if use_tts:
        subprocess.call(['espeak', string])
    click.secho(string)


def sliding_window(df, window=10, stride=3, reduction=np.mean):
    """
    Reduce a DataFrame with a strided sliding window approach.
    """
    def group_apply(i):
        tmp_df = df.groupby(tmp_index.shift(i)).agg([reduction])
        new_index = df.index[window + i - 1 : : window]
        tmp_df = tmp_df.iloc[:new_index.shape[0], :]
        tmp_df.index = new_index
        return tmp_df
    
    tmp_index = pd.Series(np.arange(df.shape[0]))
    tmp_index = tmp_index // window
    return pd.concat(
        [ group_apply(i) for i in range(0, window, stride) ]
    ).sort_index()


def cluster_df(df, epsilon=4.0, minimum_samples=100):
    """
    Cluster EMT dataframe by x,y,z using DBSCAN clustering
    """
    X = df[['x', 'y', 'z']].values
    dbscan = DBSCAN(eps=epsilon, min_samples=minimum_samples, metric='euclidean')
    model = dbscan.fit(X)
    labels = model.labels_
    outliers_df = df[model.labels_ == -1]
    clusters_df = df[model.labels_ != -1]
    colors_clusters = labels[labels != -1]
    sample_cores = np.zeros_like(labels, dtype=bool)
    sample_cores[dbscan.core_sample_indices_] = True
    return clusters_df, outliers_df, colors_clusters, df.to_numpy(), labels


def rmse(e, axis=None):
    return np.sqrt(np.mean(np.square(e), axis=axis))


import serial
from serial.tools import list_ports


# Legal UUIDs for the devices we use
VID_ARDUINO = 0x2341
vendor_ids = [ VID_ARDUINO ]


def serial_find_and_connect(port='', baud=9600, uuid='', info_command=0x00):
    """
    Find serial device and return a pyserial Serial object.

    If no port is provided, this function will send "info" commands to all serial
    devices attached which have legal vendor IDs (Arduino devices).

    :param port: Serial port (e.g. /dev/ttyUSB0). If no port provided, all open ports will be scanned.
    :param baud: Baud rate, default: 9600
    :param uuid: Device UUID (first line of info header).
    """
    serial_ports = list_ports.comports()
    for portinfo in serial_ports:
        if port:
            break
        if portinfo.vid not in vendor_ids:
            continue
        with serial.Serial(portinfo.device, baud, timeout=2, write_timeout=2) as ser:
            time.sleep(2)
            ser.flushInput()
            ser.flushOutput()
            if uuid:
                ser.write(bytes([info_command]))
                lines = int.from_bytes(ser.read(1), 'big')
                info = []
                for _ in range(lines):
                    info.append(ser.readline().decode('ASCII').strip())
                if len(info) == 0:
                    continue
                if uuid == info[0]:
                    port = portinfo.device
            else:
                port = portinfo.device
    if not port:
        return None
    time.sleep(2)
    return serial.Serial(port, baud)
