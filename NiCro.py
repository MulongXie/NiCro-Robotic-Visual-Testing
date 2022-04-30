import cv2
from Device import Device
from element_matching.GUI_pair import GUIPair

from ppadb.client import Client as AdbClient
client = AdbClient(host="127.0.0.1", port=5037)

from paddleocr import PaddleOCR
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

from keras.applications.resnet import ResNet50
resnet_model = ResNet50(include_top=False, input_shape=(32, 32, 3))


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

        self.paddle_ocr = paddle_ocr        # ocr detector for GUI detection
        self.resnet_model = resnet_model    # resnet encoder for image matching

    def load_devices(self):
        self.devices = [Device(i, dev) for i, dev in enumerate(client.devices())]

    def get_devices_info(self):
        print('Selected Source Device:')
        self.source_device.get_devices_info()
        print('\nAll Devices:')
        for i, dev in enumerate(self.devices):
            dev.get_devices_info()

    def select_source_device(self, device_id):
        self.source_device = self.devices[device_id]
        self.get_devices_info()

    def detect_gui_info_for_all_devices(self, is_load=False, show=True):
        for i, device in enumerate(self.devices):
            print('****** Device [%d / %d] ******' % (i + 1, len(self.devices)))
            device.update_screenshot_and_gui(self.paddle_ocr, is_load, show)

    def replay_action_on_device(self, device):
        print('*** Replay Devices Number [%d/%d] ***' % (device.id + 1, len(self.devices)))
        if device.id == self.source_device.id:
            print('Skip the Selected Source Device')
            return
        device.get_devices_info()
        screen_ratio = self.source_device.device.wm_size()[1] / device.device.wm_size()[1]

        matched_element = None
        if self.target_element is not None:
            gui_matcher = GUIPair(self.source_device.GUI, device.GUI, self.resnet_model)
            matched_element = gui_matcher.match_target_element(self.target_element)
        device.replay_action(self.action, matched_element, screen_ratio)

    def replay_action_on_all_devices(self):
        print('Action:', self.action)
        if self.action['type'] == 'click':
            self.target_element = self.source_device.find_element_by_coordinate(self.action['coordinate'][0][0], self.action['coordinate'][0][1], show=True)
        else:
            self.target_element = None
        for dev in self.devices:
            self.replay_action_on_device(dev)

    def control_multiple_devices_through_source_device(self, is_replay=False):
        s_dev = self.source_device
        win_name = s_dev.device.get_serial_no() + ' screen'

        def on_mouse(event, x, y, flags, params):
            '''
            :param params: [board (image), drag (boolean)]
            :param x, y: in the scale of detection image size (height=800)
            '''
            x_app, y_app = int(x / s_dev.detect_resize_ratio), int(y / s_dev.detect_resize_ratio)
            # Press button
            if event == cv2.EVENT_LBUTTONDOWN:
                params[1] = True
                cv2.circle(params[0], (x, y), 10, (255,0,255), 2)
                cv2.imshow(win_name, params[0])
                self.action['coordinate'][0] = (x_app, y_app)
            # Drag
            elif params[1] and event == cv2.EVENT_MOUSEMOVE:
                cv2.circle(params[0], (x, y), 10, (255, 0, 255), 2)
                cv2.imshow(win_name, params[0])
            # Lift button
            elif event == cv2.EVENT_LBUTTONUP:
                params[1] = False
                x_start, y_start = self.action['coordinate'][0]
                # swipe
                if abs(x_start - x_app) >= 10 or abs(y_start - y_app) >= 10:
                    print('\n****** Scroll from (%d, %d) to (%d, %d) ******' % (x_start, y_start, x_app, y_app))
                    s_dev.device.input_swipe(x_start, y_start, x_app, y_app, 500)
                    # record action
                    self.action['type'] = 'swipe'
                    self.action['coordinate'][1] = (x_app, y_app)
                # click
                else:
                    print('\n****** Tap (%d, %d) ******' % (x_start, y_start))
                    s_dev.device.input_tap(x_start, y_start)
                    # record action
                    self.action['type'] = 'click'
                    self.action['coordinate'][1] = (-1, -1)

                if is_replay:
                    self.replay_action_on_all_devices()
                # update the screenshot and GUI of the selected target device
                print("****** Re-detect Selected Device's screenshot and GUI ******")
                s_dev.update_screenshot_and_gui(self.paddle_ocr)
                params[0] = s_dev.GUI.det_result_imgs['merge'].copy()
                cv2.imshow(win_name, params[0])

        board = s_dev.GUI.det_result_imgs['merge'].copy()
        cv2.imshow(win_name, board)
        cv2.setMouseCallback(win_name, on_mouse, [board, False])
        cv2.waitKey()
        cv2.destroyWindow(win_name)

    def show_all_device_detection_results(self):
        for device in self.devices:
            device.GUI.show_detection_result()
