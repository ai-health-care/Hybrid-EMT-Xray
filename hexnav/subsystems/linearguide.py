#!/usr/bin/env python
import time

from . import Subsystem
from ..utils import serial_find_and_connect


LINEAR_GUIDE_UUID = '7a959b2'

MICROSTEP = 1

COMMAND_INFO = 0
COMMAND_RESET = 1
COMMAND_ERROR_CODE = 2
COMMAND_FORWARD = 3
COMMAND_BACKWARD = 4
COMMAND_ON = 5
COMMAND_OFF = 6
COMMAND_HOME = 7

DEFAULT_BAUD = 9600


class LinearGuide(Subsystem):
    """
    Abstraction for linear robot that retracts the sensor.
    """
    def __init__(self, baud=DEFAULT_BAUD):
        self.baud = baud
        self.ser = serial_find_and_connect(baud=baud, uuid=LINEAR_GUIDE_UUID)
        self._steps = 0

    def available(self):
        return self.ser is not None

    def disconnect(self):
        if self.ser:
            self.ser.close()

    def __del__(self):
        self.disconnect()

    def info(self):
        self.ser.write(bytes([COMMAND_INFO]))
        time.sleep(1)
        length = self.ser.read(1)
        length = int.from_bytes(length, 'big')
        return self.ser.readlines(length)

    @property
    def steps(self):
        return self._steps

    def move_stepper(self, steps, motor=0):
        step_cmd = COMMAND_FORWARD
        if steps < 0:
            steps = abs(steps)
            step_cmd = COMMAND_BACKWARD

        self.ser.write(bytes([COMMAND_ON]))
        self.ser.write(bytes([step_cmd | (motor << 4)]))
        self.ser.write(bytes([steps]))
        time.sleep(200 * MICROSTEP * 1e-3 / 1.25)
        self.ser.write(bytes([COMMAND_OFF]))
        self.ser.flush()
        self._steps += steps
