import cv2
import json
import os
import numpy as np
from os.path import join as pjoin
from difflib import SequenceMatcher

from Element import Element
from sklearn.metrics.pairwise import cosine_similarity


class GUI:
    def __init__(self, img_path='data/input', output_dir='data/output', detection_resize_height=800):
        self.img_path = img_path
        self.ui_name = img_path.replace('\\', '/').split('/')[-1].split('.')[0]
        self.output_dir = output_dir
        self.img = cv2.imread(self.img_path)

        self.detection_resize_height = detection_resize_height  # resize the input gui while detecting
        self.detection_resize_width = int(self.img.shape[1] * (self.detection_resize_height / self.img.shape[0]))
        self.det_result_imgs = {'text': None, 'non-text': None, 'merge': None}  # image visualization for different stages
        self.det_result_data = None         # {'compos':[], 'img_shape'}

        self.elements = []                  # list of Element objects for android UI
        self.ele_compos = []
        self.ele_texts = []
        self.elements_mapping = {}          # {'id': Element}
        self.has_popup_modal = False        # if the ui has popup modal
        self.screen = None

    '''
    **********************
    *** GUI Operations ***
    **********************
    '''
    def recognize_phone_screen(self):
        for e in self.elements:
            if e.height / self.detection_resize_height > 0.5:
                if e.parent is None and e.children is not None:
                    e.is_screen = True
                    self.screen = e
                    return

    def recognize_popup_modal(self, height_thresh=0.15, width_thresh=0.5):
        def is_element_modal(element, area_resize):
            gray = cv2.cvtColor(element.clip, cv2.COLOR_BGR2GRAY)
            area_ele = element.clip.shape[0] * element.clip.shape[1]
            # calc the grayscale of the element
            sum_gray_ele = np.sum(gray)
            mean_gray_ele = sum_gray_ele / area_ele
            # calc the grayscale of other region except the element
            sum_gray_other = sum_gray_a - sum_gray_ele
            mean_gray_other = sum_gray_other / (area_resize - area_ele)
            # if the element's brightness is far higher than other regions, it should be a pop-up modal
            if mean_gray_ele > 180 and mean_gray_other < 80:
                return True
            return False

        # calculate the mean pixel value as the brightness
        img_resized = cv2.resize(self.img, (self.detection_resize_width, self.detection_resize_height))
        area_resize = img_resized.shape[0] * img_resized.shape[1]

        sum_gray_a = np.sum(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY))

        if sum_gray_a / (img_resized.shape[0] * img_resized.shape[1]) < 100:
            for ele in self.elements:
                if ele.category == 'Compo' and \
                        ele.height / ele.detection_img_size[0] > height_thresh and ele.width / ele.detection_img_size[1] > width_thresh:
                    ele.get_clip(img_resized)
                    if is_element_modal(ele, area_resize):
                        self.has_popup_modal = True
                        ele.is_popup_modal = True
        if not self.has_popup_modal:
            print("No popup modal")

    '''
    *******************************
    *** Detect or Load Elements ***
    *******************************
    '''
    def detect_element(self, is_text=True, is_nontext=True, is_merge=True, paddle_cor=None):
        if is_text:
            os.makedirs(pjoin(self.output_dir, 'ocr'), exist_ok=True)
            import detect_text.text_detection as text
            if not paddle_cor:
                from paddleocr import PaddleOCR
                paddle_cor = PaddleOCR(use_angle_cls=True, lang="ch")
            self.det_result_imgs['text'], _ = text.text_detection_paddle(self.img_path, pjoin(self.output_dir, 'ocr'), paddle_cor=paddle_cor)
        if is_nontext:
            os.makedirs(pjoin(self.output_dir, 'ip'), exist_ok=True)
            import detect_compo.ip_region_proposal as ip
            key_params = {'min-grad': 6, 'ffl-block': 5, 'min-ele-area': 100, 'merge-contained-ele': False}
            self.det_result_imgs['non-text'] = ip.compo_detection(self.img_path, self.output_dir, key_params, resize_by_height=self.detection_resize_height, adaptive_binarization=False)
        if is_merge:
            os.makedirs(pjoin(self.output_dir, 'merge'), exist_ok=True)
            import detect_merge.merge as merge
            compo_path = pjoin(self.output_dir, 'ip', str(self.ui_name) + '.json')
            ocr_path = pjoin(self.output_dir, 'ocr', str(self.ui_name) + '.json')
            self.det_result_imgs['merge'], self.det_result_data = merge.merge(self.img_path, compo_path, ocr_path, pjoin(self.output_dir, 'merge'), is_remove_bar=True, is_paragraph=False)
            # convert elements as Element objects
            self.cvt_elements()

    def load_detection_result(self, data_path=None):
        if not data_path:
            data_path = pjoin(self.output_dir, 'merge', self.ui_name + '.json')
        self.det_result_data = json.load(open(data_path))
        # convert elements as Element objects
        self.cvt_elements()

    '''
    **************************************
    *** Operations for Element Objects ***
    **************************************
    '''
    def cvt_elements(self):
        '''
        Convert detection result to Element objects
        @ det_result_data: {'elements':[], 'img_shape'}
        '''
        class_map = {'Text': 't', 'Compo': 'c', 'Block': 'b'}
        for i, element in enumerate(self.det_result_data['compos']):
            e = Element(str(i) + class_map[element['class']], element['class'], element['position'], self.det_result_data['img_shape'])
            if element['class'] == 'Text':
                e.text_content = element['text_content']
            if 'children' in element:
                e.children = element['children']
            if 'parent' in element:
                e.parent = element['parent']
            e.get_clip(self.img)
            self.elements.append(e)
            self.elements_mapping[e.id] = e
        self.group_elements()

    def group_elements(self):
        for ele in self.elements:
            if ele.category == 'Compo':
                self.ele_compos.append(ele)
            elif ele.category == 'Text':
                self.ele_texts.append(ele)

    def save_element_clips(self):
        clip_dir = pjoin(self.output_dir, 'clip')
        os.makedirs(clip_dir, exist_ok=True)

        for element in self.elements:
            name = pjoin(clip_dir, element.id + '.jpg')
            cv2.imwrite(name, element.clip)

    def match_elements(self, target_ele_img, resnet_model, target_ele_text=None,
                       matched_shape_thresh=1.5, min_similarity_img=0.8, min_similarity_text=0.85, show=False):
        '''
        :param matched_shape_thresh: the maximum ratio for the shape difference of matched pair
        :param resnet_model: resnet model for encoding image
        :param target_ele_img: img clip of target element
        :param target_ele_text: text content in the target element
        :return: matched Element objects
        '''
        # 1. (optional) match by text content
        matched_ele_text = None
        matched_text_len = None
        if target_ele_text is not None and len(target_ele_text) > 0:
            # find the longest matched text
            target_ele_text = sorted(target_ele_text, key=lambda x: len(x), reverse=True)
            for tar in target_ele_text:
                for text in self.ele_texts:
                    sim = SequenceMatcher(None, text.text_content, tar).ratio()
                    if sim > min_similarity_text:
                        if matched_ele_text is None or len(text.text_content) > matched_text_len:
                            matched_ele_text = text
                            matched_text_len = len(text.text_content)
                if matched_ele_text is not None:
                    print('Match by text')
                    if show:
                        cv2.imshow('target', target_ele_img)
                        matched_ele_text.draw_element(self.img, show=True)
                    return matched_ele_text

        # 2. if no matched text element, match by image similarity
        if not matched_ele_text:
            # encode through resnet
            clips = [cv2.resize(target_ele_img, (32, 32))]
            for ele in self.ele_compos:
                clips.append(cv2.resize(ele.clip, (32, 32)))
            encodings = resnet_model.predict(np.array(clips))
            encodings = encodings.reshape((encodings.shape[0], -1))
            encoding_targe = encodings[0]
            encoding_eles = encodings[1:]

            # match images through encodings
            t_height, t_width = target_ele_img.shape[:2]
            t_aspect_ratio = round(t_width / t_height, 3)
            matched_ele_img = None
            matched_img_sim = None
            for i, ele in enumerate(self.ele_compos):
                # check the shape of the two elements first
                if max(ele.height, t_height) / min(ele.height, t_height) > matched_shape_thresh or\
                        max(ele.width, t_width) / min(ele.width, t_width) > matched_shape_thresh or\
                        max(t_aspect_ratio, ele.aspect_ratio) / min(t_aspect_ratio, ele.aspect_ratio) > matched_shape_thresh:
                    continue
                compo_similarity = cosine_similarity([encoding_targe], [encoding_eles[i]])[0][0]
                if compo_similarity > min_similarity_img:
                    if matched_ele_img is None or compo_similarity > matched_img_sim:
                        matched_ele_img = ele
                        matched_img_sim = compo_similarity
            if matched_ele_img:
                print('Match by image')
                if show:
                    cv2.imshow('target', target_ele_img)
                    matched_ele_img.draw_element(self.img, show=True)
                return matched_ele_img
        print('No matched element found')
        return None

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def show_detection_result(self):
        if self.det_result_imgs['merge'] is not None:
            cv2.imshow('det', cv2.resize(self.det_result_imgs['merge'], (self.detection_resize_width, self.detection_resize_height)))
        elif self.det_result_data is not None:
            self.draw_detection_result()
            cv2.imshow('det', cv2.resize(self.det_result_imgs['merge'], (self.detection_resize_width, self.detection_resize_height)))
        else:
            print('No detection result, run element_detection() or load_detection_result() first')
        cv2.waitKey()
        cv2.destroyAllWindows()

    def draw_detection_result(self, show_id=True):
        '''
        Draw detected elements based on det_result_data
        '''
        color_map = {'Compo': (0,255,0), 'Text': (0,0,255), 'Block':(0,255,255)}

        ratio = self.img.shape[0] / self.det_result_data['img_shape'][0]
        board = self.img.copy()
        for i, element in enumerate(self.elements):
            element.draw_element(board, ratio, color_map[element.category], show_id=show_id)
        self.det_result_imgs['merge'] = board.copy()
        return self.det_result_imgs['merge']

    def draw_popup_modal(self):
        if self.has_popup_modal:
            board = self.img.copy()
            for ele in self.elements:
                if ele.is_popup_modal:
                    ele.draw_element(board, color=(0,0,255), line=5, show_id=False)
            cv2.putText(board, 'popup modal', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
            cv2.imshow('modal', cv2.resize(board, (self.detection_resize_width, self.detection_resize_height)))
            cv2.waitKey()
            cv2.destroyAllWindows()

    def draw_screen(self, show=True):
        board = self.img.copy()
        if self.screen is not None:
            self.screen.draw_element(board, color=(255,0,255), line=5, show_id=False)
        if show:
            cv2.imshow('screen', cv2.resize(board, (self.detection_resize_width, self.detection_resize_height)))
            cv2.waitKey()
            cv2.destroyAllWindows()
        return board
