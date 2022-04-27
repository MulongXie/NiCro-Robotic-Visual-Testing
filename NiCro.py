import cv2
from Device import Device
from ppadb.client import Client as AdbClient
client = AdbClient(host="127.0.0.1", port=5037)


class NiCro:
    def __init__(self):
        # Device objects, including their screenshots and GUIs
        self.devices = [Device(i, dev) for i, dev in enumerate(client.devices())]
        self.source_device = self.devices[0]   # the selected source device

        # the action on the GUI
        # 'type': click, swipe
        # 'coordinate': action target coordinates, click has one coord, swipe has two [start, end]
        self.action = {'type': 'click', 'coordinate': [(-1, -1), (-1, -1)]}
        self.target_element = None

    def load_devices(self):
        self.devices = [Device(i, dev) for i, dev in enumerate(client.devices())]

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
            print('****** Device [%d / %d] ******' % (i + 1, len(self.devices)))
            device.detect_gui_info(paddle_ocr, is_load, show)

    def control_multiple_devices_through_source_device(self, is_replay=False):
        s_dev = self.source_device
        win_resize_ratio = 3
        win_name = s_dev.device.get_serial_no() + ' screen'

        def on_mouse(event, x, y, flags, params):
            x, y = x * win_resize_ratio, y * win_resize_ratio

            if event == cv2.EVENT_LBUTTONDOWN:
                self.action['coordinate'][0] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                x_start, y_start = self.action['coordinate'][0]
                # swipe
                if abs(x_start - x) >= 10 or abs(y_start - y) >= 10:
                    print('Scroll from (%d, %d) to (%d, %d)' % (x_start, y_start, x, y))
                    s_dev.device.input_swipe(x_start, y_start, x, y, 500)
                    # record action
                    self.target_element = None
                    self.action['type'] = 'swipe'
                    self.action['coordinate'][1] = (x, y)
                # click
                else:
                    print('Tap (%d, %d)' % (x_start, y_start))
                    s_dev.device.input_tap(x_start, y_start)
                    # record action
                    self.target_element = s_dev.GUI.get_element_by_coordinate(x_start, y_start)
                    self.action['type'] = 'click'

            img = s_dev.cap_screenshot()
            img = cv2.resize(img, (img.shape[1] // win_resize_ratio, img.shape[0] // win_resize_ratio))
            cv2.imshow(win_name, img)

        screen = cv2.resize(s_dev.screenshot, (s_dev.screenshot.shape[1] // win_resize_ratio, s_dev.screenshot.shape[0] // win_resize_ratio))
        cv2.imshow(win_name, screen)
        cv2.setMouseCallback(win_name, on_mouse)
        cv2.waitKey()
        cv2.destroyWindow(win_name)

    def show_all_device_detection_results(self):
        for device in self.devices:
            device.GUI.show_detection_result()
