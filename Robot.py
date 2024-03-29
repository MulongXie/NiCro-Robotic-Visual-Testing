import cv2
from robot.robot_control import RobotController
from GUI import GUI


class Robot(RobotController):
    def __init__(self, speed=100000, press_depth=20):
        super().__init__(speed=speed)
        self.press_depth = press_depth
        self.name = 'robot'

        self.camera = None  # height/width = 1000/540
        self.camera_clip_range_height = [80, 900]
        self.camera_clip_range_width = [0, 540]

        self.x_robot2y_cam = round((295-120)/820, 2)    # x_robot_range : cam.height_range
        self.y_robot2x_cam = round(120/540, 2)          # y_robot_range : cam.width_range

        self.GUI = None
        self.photo = None   # image
        self.photo_save_path = 'data/screen/robot_photo.png'
        self.photo_screen_area = None    # image of screen area
        self.detect_resize_ratio = None  # self.GUI.detection_resize_height / self.photo.shape[0]
        self.cap_frame()

    def cap_frame(self):
        if not self.camera or not self.camera.read()[0]:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        ret, frame = self.camera.read()
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        frame = frame[self.camera_clip_range_height[0]: self.camera_clip_range_height[1], self.camera_clip_range_width[0]:self.camera_clip_range_width[1]]
        self.photo = frame
        cv2.imwrite(self.photo_save_path, frame)
        return frame

    def convert_coord_from_camera_to_robot(self, x_cam, y_cam):
        x_robot = int((820 - y_cam) * self.x_robot2y_cam) + 120
        y_robot = int((270 - x_cam) * self.y_robot2x_cam)
        return x_robot, y_robot

    def adjust_camera_clip_range(self):
        def nothing(x):
            pass
        cv2.namedWindow('win')
        cv2.createTrackbar('top', 'win', self.camera_clip_range_height[0], 1000, nothing)
        cv2.createTrackbar('left', 'win', self.camera_clip_range_width[0], 540, nothing)
        cv2.createTrackbar('bottom', 'win', self.camera_clip_range_height[1], 1001, nothing)
        cv2.createTrackbar('right', 'win', self.camera_clip_range_width[1], 541, nothing)
        while 1:
            top = cv2.getTrackbarPos('top', 'win')
            left = cv2.getTrackbarPos('left', 'win')
            bottom = cv2.getTrackbarPos('bottom', 'win')
            right = cv2.getTrackbarPos('right', 'win')
            frame = self.cap_frame()
            frame_clip = frame[top:bottom, left:right]
            cv2.imshow('frame', frame)
            cv2.imshow('clip', frame_clip)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()
        self.camera.release()

    def control_robot_by_clicking_on_cam_video(self):
        def click_event(event, x, y, flags, params):
            x_pre, y_pre = params
            if event == cv2.EVENT_LBUTTONDOWN:
                params[0], params[1] = self.convert_coord_from_camera_to_robot(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                x, y = self.convert_coord_from_camera_to_robot(x, y)
                print(x, y, params)
                # swipe
                if abs(x_pre - x) >= 10 or abs(y_pre - y) >= 10:
                    self.swipe((x_pre, y_pre, self.press_depth), (x, y, self.press_depth))
                # click
                else:
                    self.click((x_pre, y_pre, self.press_depth))
        button_down_coords = [-1, -1]
        while 1:
            frame = self.cap_frame()
            # get the click point on the image
            cv2.imshow('camera', frame)
            cv2.setMouseCallback('camera', click_event, param=button_down_coords)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyWindow('camera')
        self.camera.release()

    def detect_gui_element(self, paddle_ocr, is_load=False, show=False, ocr_opt='paddle', adjust_by_screen_area=True, verbose=True):
        # self.cap_frame()
        self.GUI = GUI(self.photo_save_path)
        self.detect_resize_ratio = self.GUI.detection_resize_height / self.photo.shape[0]
        if is_load:
            self.GUI.load_detection_result()
        else:
            self.GUI.detect_element(True, True, True, paddle_ocr=paddle_ocr, ocr_opt=ocr_opt, verbose=verbose)
        if adjust_by_screen_area:
            self.adjust_elements_by_screen_area(show)
        elif show:
            self.GUI.show_detection_result()

    '''
    **************************************
    *** Adjust Element by Phone Screen ***
    **************************************
    '''
    def recognize_phone_screen(self):
        gui = self.GUI
        for e in gui.elements:
            if e.height / gui.detection_resize_height > 0.5:
                if e.parent is None and e.children is not None:
                    e.is_screen = True
                    gui.screen = e
                    gui.img = e.clip
                    self.photo_screen_area = e.clip
                    return

    def remove_ele_out_screen(self):
        gui = self.GUI
        new_elements = []
        gui.ele_compos = []
        gui.ele_texts = []
        for ele in gui.elements:
            if ele.id in gui.screen.children:
                new_elements.append(ele)
                if ele.category == 'Compo':
                    gui.ele_compos.append(ele)
                elif ele.category == 'Text':
                    gui.ele_texts.append(ele)
        gui.elements = new_elements

    def convert_element_relative_pos_by_screen(self):
        gui = self.GUI
        s_left, s_top = gui.screen.col_min, gui.screen.row_min
        for ele in gui.elements:
            ele.col_min -= s_left
            ele.col_max -= s_left
            ele.row_min -= s_top
            ele.row_max -= s_top

    def resize_screen_and_elements_by_height(self):
        gui = self.GUI
        h_ratio = gui.detection_resize_height / gui.screen.height
        gui.screen.resize_bound(resize_ratio_col=h_ratio, resize_ratio_row=h_ratio)
        gui.img = cv2.resize(gui.img, (int(gui.screen.width * h_ratio), gui.detection_resize_height))
        for ele in gui.elements:
            ele.resize_bound(resize_ratio_col=h_ratio, resize_ratio_row=h_ratio)
            ele.get_clip(gui.img)
        gui.draw_detection_result()

    def adjust_elements_by_screen_area(self, show=False):
        '''
        Recognize the phone screen region if any and adjust the element coordinates according to the screen
        '''
        self.recognize_phone_screen()
        if self.GUI.screen is None:
            return
        self.remove_ele_out_screen()
        self.convert_element_relative_pos_by_screen()
        self.resize_screen_and_elements_by_height()
        if show:
            self.GUI.show_detection_result()

    def convert_element_pos_back(self, element):
        '''
        Convert back the element coordinates from the phone screen-based to the whole image-based (detection_resize_height)
        '''
        gui = self.GUI
        if gui.screen is None:
            return
        h_ratio = gui.screen.height / gui.detection_resize_height
        element.col_min = int((element.col_min + gui.screen.col_min) * h_ratio)
        element.col_max = int((element.col_max + gui.screen.col_min) * h_ratio)
        element.row_min = int((element.row_min + gui.screen.row_min) * h_ratio)
        element.row_max = int((element.row_max + gui.screen.row_min) * h_ratio)
        element.init_bound()
        element.get_clip(gui.img)

    def draw_elements_on_screen(self, show=True):
        gui = self.GUI
        board = gui.screen_img.copy()
        for ele in gui.elements:
            ele.draw_element(board, show=False)
        if show:
            cv2.imshow('Elements on screen', board)
            cv2.waitKey()
            cv2.destroyWindow('Elements on screen')
        return board

    '''
    *********************
    *** Action Replay ***
    *********************
    '''
    def replay_action(self, action, matched_element=None, screen_ratio=None):
        if action['type'] == 'click':
            if matched_element is not None:
                self.click((int(matched_element.center_x / self.detect_resize_ratio), int(matched_element.center_y / self.detect_resize_ratio), self.press_depth))
            else:
                x_rescreen, y_rescreen = int(action['coordinate'][0][0] / screen_ratio), int(action['coordinate'][0][1] / screen_ratio)
                x_robot, y_robot = self.convert_coord_from_camera_to_robot(x_rescreen, y_rescreen)
                print('Screen Coord(%d, %d), Robot Coord(%d, %d)' % (x_rescreen, y_rescreen, x_robot, y_robot))
                self.click((x_robot, y_robot, self.press_depth))
        elif action['type'] == 'swipe':
            x_rescreen, y_rescreen = int(action['coordinate'][0][0] / screen_ratio), int(action['coordinate'][0][1] / screen_ratio)
            x_robot, y_robot = self.convert_coord_from_camera_to_robot(x_rescreen, y_rescreen)
            start_coord = (x_robot, y_robot, self.press_depth)
            re_dist = (int((action['coordinate'][1][1] - action['coordinate'][0][1]) / screen_ratio), int((action['coordinate'][1][0] - action['coordinate'][0][0]) / screen_ratio))
            end_coord = (int(start_coord[0] - re_dist[0]), int(start_coord[1] + re_dist[1]), self.press_depth)
            self.swipe(start_coord, end_coord)


if __name__ == '__main__':
    robot = Robot(speed=1000000)
    robot.control_robot_by_clicking_on_cam_video()
