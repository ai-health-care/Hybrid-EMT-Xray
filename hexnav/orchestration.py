from datetime import datetime
import json
from pathlib import Path
import time

import cv2

import pandas as pd

from hexnav.subsystems.igttracker import IGTLinkTracker, TrackerThread

from .constants import DEFAULT_CROP
from .subsystems.framegrabber import FrameGrabber, crop_xray
from .subsystems.footpedal import Footpedal, VirtualFootpedal
from .subsystems.carmservo import CarmServo, VirtualCarmServo
from .subsystems.linearguide import LinearGuide
from .paths import MEASUREMENT_PATH
from .subsystems import SubsystemManager
from .utils import say


class NavigationData:
    def __init__(self):
        self.clear()

    def add_xray(self, image, gantry_angle, gantry_height, steps, sensor_z=0.0):
        self.xray_data.append({
            'image': image,
            'angle': gantry_angle,
            'height': gantry_height,
            'steps': steps,
            'time': time.time(),
            'measurement': self.measurement_name,
            'z': sensor_z
        })

    def clear(self):
        timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        self.measurement_name = f'navigation_experiment_{timestamp}'
        self.path = MEASUREMENT_PATH / self.measurement_name
        self.xray_path = self.path / 'xrays'
        Path.mkdir(self.path, exist_ok=True)
        Path.mkdir(self.xray_path, exist_ok=True)
        self.emt_data = []
        self.xray_data = []

    def save(self):
        Path.mkdir(MEASUREMENT_PATH, exist_ok=True)
        self.path = MEASUREMENT_PATH / self.measurement_name
        self.xray_path = self.path / 'xrays'
        Path.mkdir(self.path, exist_ok=True)
        Path.mkdir(self.xray_path, exist_ok=True)

        with open(self.path / 'xrays.json', 'w') as f:
            json.dump(self.xray_data, f)
        df = pd.DataFrame(self.emt_data, columns=['time', 'x', 'y', 'z'])
        df.to_csv(self.path / 'emt.csv')

    @staticmethod
    def merge(A, B):
        C = NavigationData()
        C.xray_data = []
        C.xray_data.extend(A.xray_data)
        C.xray_data.extend(B.xray_data)
        C.measurement_name = A.measurement_name + '___' + B.measurement_name

        C.emt_data = []
        C.emt_data.extend(A.emt_data)
        C.emt_data.extend(B.emt_data)
        return C

    @staticmethod
    def load(path):
        data = NavigationData()
        # TODO check if path matches regex scheme
        folder = Path(path).parts[-1]
        timestamp = datetime.strptime(folder, 'navigation_experiment_%Y_%m_%d__%H_%M_%S')
        timestamp = timestamp.strftime('%Y_%m_%d__%H_%M_%S')
        data.measurement_name = f'navigation_experiment_{timestamp}'
        data.path = MEASUREMENT_PATH / data.measurement_name
        data.xray_path = data.path / 'xrays'
        data.emt_data = pd.read_csv(Path(path) / 'emt.csv').values
        with open(Path(path) / 'xrays.json') as f:
            data.xray_data = json.load(f)
        for xray in data.xray_data:
            if xray.get('measurement'):
                continue
            xray['measurement'] = data.measurement_name
        return data


class Navigation:
    def __init__(self, gantry_height=10, use_pedal=True):
        self.gantry_height = gantry_height
        self.subsystems = SubsystemManager()

        say('Initializing linear guide...')
        self.linearguide = LinearGuide()
        say('Initializing frame grabber...')
        framegrabber = FrameGrabber(default_device=2)
        say('Initializing tracker...')
        tracker = IGTLinkTracker()
        say('Initializing foot pedal...')
        if use_pedal:
            pedal = Footpedal()
            if not pedal.available():
                pedal = VirtualFootpedal()
        else:
            pedal = VirtualFootpedal()

        try:
            servo = CarmServo()
            if not servo.available():
                servo = VirtualCarmServo()
        except:
            servo = VirtualCarmServo()

        self.subsystems.register('linear guide', self.linearguide)
        self.subsystems.register('frame grabber', framegrabber)
        self.subsystems.register('tracker', tracker)
        self.subsystems.register('foot pedal', pedal, mandatory=False)
        self.subsystems.register('c-arm servo', servo, mandatory=False)
        self.subsystems.show()
        self.trackerthread = TrackerThread(self.subsystems.get('tracker'))
        self.navigation_data = NavigationData()
        say('I am ready.')

    def capture_image(self, angle, bounds, sensor_z=0.0):
        self.subsystems.get('c-arm servo').rotate(angle)
        self.subsystems.get('foot pedal').trigger_xray()
        image = self.subsystems.get('frame grabber').shoot_image()
        image, _ = crop_xray(image, bounds)
        timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        image_filename = f'xray_{timestamp}.png'
        Path.mkdir(self.navigation_data.path, exist_ok=True)
        Path.mkdir(self.navigation_data.xray_path, exist_ok=True)
        image_path = str(self.navigation_data.xray_path / image_filename)
        cv2.imwrite(image_path, image)
        self.navigation_data.add_xray(
            image_filename,
            angle,
            self.gantry_height,
            self.subsystems.get('linear guide').steps,
            sensor_z=sensor_z
        )

    def record_line(self, steps, step_size=10, bounds=DEFAULT_CROP, gantry_angles=[75, 90, 105]):
        """
        Move the linear drive in steps, and at each step, take X-rays at
        multiple orientations. In the meantime, EMT data is recorded
        continuously in a separate thread.
        """
        self.trackerthread.start()

        for gantry_angle in gantry_angles:
            self.capture_image(gantry_angle, bounds)

        for i in range(steps - 1):
            print(f'step {i + 1} / {steps}')
            gantry_angles.reverse()
            print(self.linearguide.steps)
            self.linearguide.move_stepper(step_size, motor=0)
            print(self.linearguide.steps)
            for gantry_angle in gantry_angles:
                self.capture_image(gantry_angle, bounds)

        self.subsystems.get('linear guide').move_stepper(-step_size * (steps - 1), 0)

        self.trackerthread.running = False
        self.trackerthread.join()

    def record_compensation(self, steps, bounds=DEFAULT_CROP, sensor_z=0.0):
        """
        """
        self.trackerthread.start()

        for step in range(steps):
            print(f'step {step+1} / {steps}')
            self.capture_image(90, bounds, sensor_z=sensor_z)

        self.trackerthread.running = False
        self.trackerthread.join()

    def save_data(self):
        if self.trackerthread.running:
            raise RuntimeError(
                'Cannot save data when TrackerThread is still running. This is a bug.'
            )
        self.navigation_data.emt_data = self.trackerthread.data
        self.navigation_data.save()


    def clear_data(self):
        """
        """
        if self.trackerthread.running:
            raise RuntimeError(
                'Cannot clear data when TrackerThread is still running. This is a bug.'
            )
        self.trackerthread.data.clear()
        self.navigation_data.clear()