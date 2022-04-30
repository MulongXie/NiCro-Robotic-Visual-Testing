import cv2
from robot.robot_control import RobotController


class Robot(RobotController):
    def __init__(self, speed=10000):
        super().__init__(speed=speed)

        self.camera = cv2.VideoCapture(0)  # height/width = 1000/540
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.camera_clip_range_height = [80, 900]
        self.camera_clip_range_width = [0, 540]

        self.x_robot2y_cam = round((298-120)/820, 2)    # x_robot_range : cam.height_range
        self.y_robot2x_cam = round(120/540, 2)          # y_robot_range : cam.width_range

    def cap_frame(self):
        ret, frame = self.camera.read()
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        frame = frame[self.camera_clip_range_height[0]: self.camera_clip_range_height[1], self.camera_clip_range_width[0]:self.camera_clip_range_width[1]]
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

    def control_robot_by_clicking_on_cam_video(self):
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                x_r, y_r = self.convert_coord_from_camera_to_robot(x, y)
                print('Clicked Point:(%d,%d) Robot:(%d,%d)' % (x, y, x_r, y_r))
                robot.click((x_r, y_r, 22))
        while 1:
            frame = self.cap_frame()
            # get the click point on the image
            cv2.imshow('camera', frame)
            cv2.setMouseCallback('camera', click_event)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyWindow('camera')


robot = Robot(speed=10000)
robot.control_robot_by_clicking_on_cam_video()
