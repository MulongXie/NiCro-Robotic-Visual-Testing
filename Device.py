import cv2
from GUI import GUI


class Device:
    def __init__(self, dev_id, device):
        self.id = dev_id
        self.device = device                        # ppadb device
        self.screenshot = self.cap_screenshot()     # cv2.image
        self.GUI = None                             # GUI object

        self.screenshot_path = 'data/screen/' + str(self.id) + '.png'

    def get_devices_info(self):
        print("%d - Number:%s Resolution:%s" % (self.id, self.device.get_serial_no(), self.device.wm_size()))

    def cap_screenshot(self):
        screen = self.device.screencap()
        with open(self.screenshot_path, "wb") as fp:
            fp.write(screen)
        self.screenshot = cv2.imread(self.screenshot_path)
        return self.screenshot

    def detect_gui_info(self, paddle_ocr, is_load=True):
        self.GUI = GUI(self.screenshot_path)
        if is_load:
            self.GUI.load_detection_result()
        else:
            self.GUI.detect_element(True, True, True, paddle_cor=paddle_ocr)

