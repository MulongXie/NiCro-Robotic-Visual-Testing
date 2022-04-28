import cv2


class Element:
    def __init__(self, element_id, category, position, text_content=None):
        self.id = element_id
        self.category = category        # Compo / Text
        self.text_content = text_content

        self.col_min, self.row_min, self.col_max, self.row_max = int(position['column_min']), int(position['row_min']), int(position['column_max']), int(position['row_max'])
        self.center_x, self.center_y = int((self.col_max + self.col_min) / 2), int((self.row_max + self.row_min) / 2)
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.aspect_ratio = round(self.width / self.height, 3)
        self.area = self.width * self.height
        self.clip = None

        self.children = None    # contained elements within it
        self.parent = None      # parent element

        self.matched_element = None     # the matched Element in another ui
        self.is_popup_modal = False     # if the element is popup modal
        self.is_screen = False          # if the element is phone screen

    def init_bound(self):
        self.center_x, self.center_y = int((self.col_max + self.col_min) / 2), int((self.row_max + self.row_min) / 2)
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.aspect_ratio = round(self.width / self.height)
        self.area = self.width * self.height

    def get_clip(self, org_img):
        left, right, top, bottom = int(self.col_min), int(self.col_max), int(self.row_min), int(self.row_max)
        self.clip = org_img[top: bottom, left: right]

    def resize_bound(self, resize_ratio_col, resize_ratio_row):
        self.col_min = int(self.col_min*resize_ratio_col)
        self.row_min = int(self.row_min*resize_ratio_row)
        self.col_max = int(self.col_max*resize_ratio_col)
        self.row_max = int(self.row_max*resize_ratio_row)

    def get_bound(self):
        return self.col_min, self.row_min, self.col_max, self.row_max

    def draw_element(self, board, color=(225,255,0), line=2, show_id=False, show=False):
        bound = self.get_bound()
        cv2.rectangle(board, (bound[0], bound[1]), (bound[2], bound[3]), color, line)
        if show_id:
            cv2.putText(board, str(self.id), (bound[0] + 3, bound[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if show:
            cv2.imshow('Element' + str(self.id), cv2.resize(board, (int(board.shape[1] * (800 / board.shape[0])), 800)))
            cv2.waitKey()
            cv2.destroyAllWindows()

    def show_clip(self):
        cv2.imshow('clip', self.clip)
        cv2.waitKey()
        cv2.destroyWindow('clip')
