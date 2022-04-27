import cv2
from GUI import GUI


class Device:
    def __init__(self, dev_id, device):
        self.id = dev_id
        self.screenshot_path = 'data/screen/' + str(self.id) + '.png'

        self.device = device                        # ppadb device
        self.screenshot = self.cap_screenshot()     # cv2.image
        self.GUI = None                             # GUI object

        # the action on the GUI
        # 'type': click, swipe
        # 'coordinate': action target coordinates, click has one coord, swipe has two [start, end]
        self.action = {'type': None, 'coordinate': [(-1, -1), (-1, -1)]}

    def get_devices_info(self):
        print("%d - Number:%s Resolution:%s" % (self.id, self.device.get_serial_no(), self.device.wm_size()))

    def cap_screenshot(self):
        screen = self.device.screencap()
        with open(self.screenshot_path, "wb") as fp:
            fp.write(screen)
        self.screenshot = cv2.imread(self.screenshot_path)
        return self.screenshot

    def detect_gui_info(self, paddle_ocr, is_load=False, show=False):
        self.cap_screenshot()
        self.GUI = GUI(self.screenshot_path)
        if is_load:
            self.GUI.load_detection_result()
        else:
            self.GUI.detect_element(True, True, True, paddle_cor=paddle_ocr)
        if show:
            self.GUI.show_detection_result()

    def get_element_by_clicking_on_image(self):
        self.GUI.get_element_by_clicking()

    def control_app_through_screenshot(self):
        '''
        Control the app through action on the screenshot
        '''

        win_resize_ratio = 3
        win_name = self.device.get_serial_no() + ' screen'

        def on_mouse(event, x, y, flags, params):
            x, y = x * win_resize_ratio, y * win_resize_ratio

            if event == cv2.EVENT_LBUTTONDOWN:
                self.action['coordinate'][0] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                x_start, y_start = self.action['coordinate'][0]
                # swipe
                if abs(x_start - x) >= 10 or abs(y_start - y) >= 10:
                    print('Scroll from (%d, %d) to (%d, %d)' % (x_start, y_start, x, y))
                    self.device.input_swipe(x_start, y_start, x, y, 500)
                # click
                else:
                    print('Tap (%d, %d)' % (x_start, y_start))
                    self.device.input_tap(x_start, y_start)
            img = self.cap_screenshot()
            img = cv2.resize(img, (img.shape[1] // win_resize_ratio, img.shape[0] // win_resize_ratio))
            cv2.imshow(win_name, img)

        screen = cv2.resize(self.screenshot, (self.screenshot.shape[1] // win_resize_ratio, self.screenshot.shape[0] // win_resize_ratio))
        cv2.imshow(win_name, screen)
        cv2.setMouseCallback(win_name, on_mouse)
        cv2.waitKey()
        cv2.destroyWindow(win_name)
