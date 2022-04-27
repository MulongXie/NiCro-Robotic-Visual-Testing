import cv2
from Device import Device
from ppadb.client import Client as AdbClient
client = AdbClient(host="127.0.0.1", port=5037)


class NiCro:
    def __init__(self):
        # Device objects, including their screenshots and GUIs
        self.devices = [Device(i, dev) for i, dev in enumerate(client.devices())]

        self.source_device = self.devices[0]   # the selected source device
        self.target_element = None
        self.action = None

    def get_devices_info(self):
        print('Selected Source Device:')
        self.source_device.get_devices_info()
        print('All Devices:')
        for i, dev in enumerate(self.devices):
            dev.get_devices_info()

    def select_source_device(self, device_id):
        self.source_device = self.devices[device_id]
        self.get_devices_info()

    def detect_gui_info_for_all_devices(self, paddle_ocr, is_load=False, show=True):
        for i, device in enumerate(self.devices):
            print('****** Device [%d / %d] ******' % (i, len(self.devices)))
            device.detect_gui_info(paddle_ocr, is_load, show)
