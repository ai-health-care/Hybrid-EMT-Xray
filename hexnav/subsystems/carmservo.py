import time

from . import Subsystem
from ..utils import say, serial_find_and_connect


CARM_SERVO_UUID = '2350a3b3'

COMMAND_INFO = 0
COMMAND_RESET = 1
COMMAND_ERROR_CODE = 2
COMMAND_ON = 3
COMMAND_IDLE = 4


class CarmServo(Subsystem):
    def __init__(self, baud=9600, start_angle=75):
        self.ser = serial_find_and_connect(baud=baud, uuid=CARM_SERVO_UUID)
        time.sleep(2)
        if self.ser:
            self.ser.flushInput()
            self.ser.flushOutput()
        self.angle = start_angle

    def available(self):
        return self.ser is not None

    def rotate(self, angle):
        self.ser.flushInput()
        self.ser.flushOutput()
        time.sleep(1)
        delta = angle - self.angle
        print(f'current angle: {self.angle} angle: {angle} delta: {delta}')
        if delta == 0:
            return
        if delta < 0:
            print(self.ser.write(bytes([COMMAND_ON | (0 << 4)])))
            time.sleep(0.7 + abs(delta / 4))
            print(self.ser.write(bytes([COMMAND_IDLE])))
        if delta > 0:
            print(self.ser.write(bytes([COMMAND_ON | (1 << 4)])))
            time.sleep(1.3 + abs(delta / 4))
            print(self.ser.write(bytes([COMMAND_IDLE])))
        self.angle = angle

    def disconnect(self):
        if hasattr(self, 'ser') and self.ser is not None:
            self.ser.write(bytes([COMMAND_IDLE]))
            self.ser.close()

    def __del__(self):
        self.disconnect()


class VirtualCarmServo(Subsystem):
    def __init__(self, start_angle=75):
        pass

    def rotate(self, angle):
        self.angle = angle
        say(f'Rotate the gantry to {angle} degrees.')
        input('<Press RETURN to continue>')