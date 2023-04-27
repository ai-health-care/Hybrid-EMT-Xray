import time

from . import Subsystem
from ..utils import say, serial_find_and_connect


PEDAL_EMULATOR_UUID = '938f0e00'
COMMAND_BURST = 5


class Footpedal(Subsystem):
    def __init__(self, baud=9600):
        self.ser = serial_find_and_connect(baud=baud, uuid=PEDAL_EMULATOR_UUID)

    def available(self):
        return self.ser is not None

    def trigger_xray(self):
        channel = 0
        status = self.ser.write(bytes([COMMAND_BURST | (channel << 4)])) > 0
        self.ser.flush()
        return status

    def disconnect(self):
        self.ser.close()

    def __del__(self):
        self.disconnect()


class VirtualFootpedal(Subsystem):
    def __init__(self):
        pass

    def available(self):
        return True

    def trigger_xray(self):
        say('Hit the footpedal when ready.')
        return input('<Press RETURN to continue>')