import cv2


class Operation:
    def __init__(self, ui_img_path, action):
        self.ui_img_path = ui_img_path
        self.ui_img = cv2.imread(ui_img_path)
        self.action = action

        self.target_element = None
