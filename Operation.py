import cv2


class Operation:
    def __init__(self, ui_img_path, action, target_element_bounds):
        self.ui_img_path = ui_img_path
        self.ui_img = cv2.imread(ui_img_path)
        self.ui_img_height, self.ui_img_width = self.ui_img.shape[:2]

        self.action = action
        self.target_element_bounds = target_element_bounds
        self.target_element_img = None
        self.clip_target_element_img()

    def clip_target_element_img(self):
        self.target_element_img = self.ui_img[self.target_element_bounds[0][1]: self.target_element_bounds[1][1], self.target_element_bounds[0][0]: self.target_element_bounds[1][0]]

    def resize(self, width_resize, height_resize):
        width_resize_ratio = width_resize / self.ui_img_width
        height_resize_ratio = height_resize / self.ui_img_height
        self.target_element_bounds[0][0] = int(self.target_element_bounds[0][0] * width_resize_ratio)
        self.target_element_bounds[0][1] = int(self.target_element_bounds[0][1] * height_resize_ratio)
        self.target_element_bounds[1][0] = int(self.target_element_bounds[1][0] * width_resize_ratio)
        self.target_element_bounds[1][1] = int(self.target_element_bounds[1][1] * height_resize_ratio)

        self.ui_img = cv2.resize(self.ui_img, (width_resize, height_resize))
        self.clip_target_element_img()

    def show_target_ele(self):
        board = self.ui_img.copy()
        cv2.rectangle(board, self.target_element_bounds[0], self.target_element_bounds[1], (255,0,0), 2)
        cv2.imshow('target element', board)
        cv2.imshow('clip', self.target_element_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
