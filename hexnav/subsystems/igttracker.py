import threading
import time

import numpy as np
import pandas as pd

# pip install pyigtl
import pyigtl

from . import Subsystem


class IGTLinkTracker(Subsystem):
    """
    Abstraction for tracker connected via IGTLink protocol.
    This can be a custom server or a PlusServer, sending
    "POSITION" type messages.
    """
    def __init__(self, host='127.0.0.1', port=18944, timeout=10, devicename='Tracker'):
        self.client = pyigtl.OpenIGTLinkClient(host, port)
        self.timeout = timeout
        self.devicename = devicename
        self.good = self.client is not None
        # receive and discard one record to test connection
        message = self.client.wait_for_message(self.devicename, timeout=self.timeout)
        if not message:
            self.good = False

    def available(self):
        return self.good

    def measure(self):
        message = self.client.wait_for_message(self.devicename, timeout=self.timeout)
        if not message:
            self.good = False
            raise RuntimeError('Could not connect to IGTLink server.')
        timestamp = message.timestamp
        if timestamp == 0:
            timestamp = time.time() 
        positions = message.positions
        return timestamp, np.array(positions)


class TrackerThread(threading.Thread):
    """
    Thread that continually records tracking data.
    """
    def __init__(self, tracker):
        super().__init__()
        self.data = []
        self.running = True
        self.tracker = tracker

    def run(self):
        while self.running:
            timestamp, positions = self.tracker.measure()
            vector = np.append(timestamp, positions)
            self.data.append(vector)

    def stop_tracking(self):
        self.running = False

    @property
    def df(self):
        """
        Retrieve tracking data as Pandas DataFrame.
        Please make sure to stop the thread (call stop_tracking()) in advance.
        """
        if self.running:
            raise RuntimeError('Error: Tried to export tracking data while tracker is running.')
        d = pd.DataFrame(self.data, columns=['time', 'x', 'y', 'z'])
        return d