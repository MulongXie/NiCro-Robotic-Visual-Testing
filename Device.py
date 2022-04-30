import cv2
from GUI import GUI


class Device:
    def __init__(self, dev_id, device):
        self.id = dev_id
        self.name = device.get_serial_no()
        self.screenshot_path = 'data/screen/' + str(self.name) + '.png'
        self.device = device                        # ppadb device

        self.screenshot = self.cap_screenshot()     # cv2.image
        self.GUI = GUI(self.screenshot_path)        # GUI object
        self.detect_resize_ratio = self.GUI.detection_resize_height / self.screenshot.shape[0]

        # the action on the GUI
        # 'type': click, swipe
        # 'coordinate': action target coordinates, click has one coord, swipe has two [start, end]
        self.action = {'type': None, 'coordinate': [(-1, -1), (-1, -1)]}

    def get_devices_info(self):
        print("Device ID:%d Name:%s Resolution:%s" % (self.id, self.name, self.device.wm_size()))

    def cap_screenshot(self):
        screen = self.device.screencap()
        with open(self.screenshot_path, "wb") as fp:
            fp.write(screen)
        self.screenshot = cv2.imread(self.screenshot_path)
        return self.screenshot

    def detect_gui_info(self, paddle_ocr, is_load=False, show=False):
        self.GUI = GUI(self.screenshot_path)
        if is_load:
            self.GUI.load_detection_result()
        else:
            self.GUI.detect_element(True, True, True, paddle_cor=paddle_ocr)
        if show:
            self.GUI.show_detection_result()

    def update_screenshot_and_gui(self, paddle_ocr, is_load=False, show=False):
        self.cap_screenshot()
        self.detect_gui_info(paddle_ocr, is_load=is_load, show=show)

    def find_element_by_coordinate(self, x, y, show=False):
        '''
        x, y: in the scale of app screen size
        '''
        x_resize, y_resize = x * self.detect_resize_ratio, y * self.detect_resize_ratio
        ele = self.GUI.get_element_by_coordinate(x_resize, y_resize)
        if ele is None:
            print('No element found at (%d, %d)' % (x_resize, y_resize))
        elif show:
            ele.show_clip()
        return ele

    def match_element(self, target_element, resnet_model, paddle_ocr, scroll_search=True):
        matched_ele = self.GUI.match_elements(target_element.clip, resnet_model, target_element.text_content, min_similarity_img=0.55, show=True)
        # scroll down if current gui is not matched
        if matched_ele is None and scroll_search:
            print('Scroll down and try to match again')
            self.device.input_swipe(50, (self.device.wm_size().height * 0.9), 50, 20, 500)
            self.update_screenshot_and_gui(paddle_ocr)
            self.match_element(target_element, resnet_model, paddle_ocr, scroll_search=False)
        return matched_ele

    def replay_action(self, action, matched_element=None, screen_ratio=None):
        if action['type'] == 'click' and matched_element is not None:
            if matched_element is not None:
                self.execute_action('click', [(int(matched_element.center_x / self.detect_resize_ratio), int(matched_element.center_y / self.detect_resize_ratio))])
        elif action['type'] == 'swipe':
            start_coord = (int(action['coordinate'][0][0] / screen_ratio), action['coordinate'][0][1] / screen_ratio)
            re_dist = ((action['coordinate'][1][0] - action['coordinate'][0][0]) / screen_ratio, (action['coordinate'][1][1] - action['coordinate'][0][1]) / screen_ratio)
            end_coord = (int(start_coord[0] + re_dist[0]), int(start_coord[1] + re_dist[1]))
            self.execute_action('swipe', [start_coord, end_coord])

    def execute_action(self, action_type, coordinates):
        if action_type == 'click':
            self.device.input_tap(coordinates[0][0], coordinates[0][1])
        elif action_type == 'swipe':
            self.device.input_swipe(coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1], 500)
