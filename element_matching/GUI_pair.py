import cv2
import json
import os
import numpy as np
import time
import shutil
from os.path import join as pjoin
from random import randint as rint
from glob import glob
from difflib import SequenceMatcher

import element_matching.matching as matching
from sklearn.metrics.pairwise import cosine_similarity


class GUIPair:
    def __init__(self, gui1, gui2, resnet_model=None):
        self.gui1 = gui1
        self.gui2 = gui2

        self.min_similarity_text = 0.85
        self.min_similarity_img = 0.55

        # the similarity matrix of all elements in gui1 and gui2, shape: (len(gui1.all_elements), len(gui2.all_elements)
        self.image_similarity_matrix = None
        # the preload resnet model for image encoding
        self.resnet_model = resnet_model

    '''
    ******************************
    *** Match Similar Elements ***
    ******************************
    '''
    def calculate_elements_image_similarity_matrix(self):
        # calculate the similarity matrix through resnet for all elements in gui1 and gui2
        clips1 = [ele.clip for ele in self.gui1.elements]
        clips2 = [ele.clip for ele in self.gui2.elements]
        self.image_similarity_matrix = matching.image_similarity_matrix(clips1, clips2, method='resnet', resnet_model=self.resnet_model)

    def match_by_text(self, target_element, compare_elements):
        target_ele_text = target_element.text_content
        if target_ele_text is not list:
            target_ele_text = [target_ele_text]
        else:
            target_ele_text = sorted(target_ele_text, key=lambda x: len(x), reverse=True)
        matched_elements = []
        is_matched = False
        for tar in target_ele_text:
            for text_ele in compare_elements:
                sim = SequenceMatcher(None, text_ele.text_content, tar).ratio()
                if sim > self.min_similarity_text:
                    matched_elements.append(text_ele)
                    is_matched = True
            # only use the longest matched text
            if is_matched:
                break
        return matched_elements

    def match_by_img(self, target_element, compared_elements, hash_check=True):
        if self.image_similarity_matrix is None:
            self.calculate_elements_image_similarity_matrix()
        # similarities between the target element and all elements in gui2
        target_sims = self.image_similarity_matrix[target_element.id]
        # filter by similarity threshold
        matched_elements_id = np.where(target_sims > self.min_similarity_img)[0]
        # select from the input compared_elements
        matched_elements_id = set(matched_elements_id).intersection(set([e.id for e in compared_elements]))
        matched_elements = np.array(self.gui2.elements)[matched_elements_id]

        # double check by dhash
        if hash_check:
            dhash_similarity = matching.image_similarity_matrix([target_element.clip], [e.clip for e in matched_elements])
            matched_elements_id = np.where(dhash_similarity > self.min_similarity_img)[0]
            matched_elements = np.array(self.gui2.elements)[matched_elements_id]
        return matched_elements

    def match_by_shape(self, target_element, compared_elements):
        return []

    def match_by_neighbour(self, target_element, compared_elements):
        return []

    def match_target_element(self, target_element):
        if target_element.category == 'Text' or target_element.text_content is not None:
            matched_elements = self.match_by_text(target_element, self.gui2.ele_texts)
            if len(matched_elements) > 1:
                matched_elements = self.match_by_img(target_element, matched_elements)
            if len(matched_elements) > 1:
                matched_elements = self.match_by_shape(target_element, matched_elements)
            if len(matched_elements) > 1:
                matched_elements = self.match_by_neighbour(target_element, matched_elements)
        else:
            matched_elements = self.match_by_img(target_element, self.gui2.ele_compos)
            if len(matched_elements) > 1:
                matched_elements = self.match_by_shape(target_element, matched_elements)
            if len(matched_elements) > 1:
                matched_elements = self.match_by_neighbour(target_element, matched_elements)
        return matched_elements[0]

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def show_detection_result(self):
        rest1 = self.gui1.draw_detection_result()
        rest2 = self.gui2.draw_detection_result()
        cv2.imshow('detection1', cv2.resize(rest1, (self.gui1.detection_resize_width, self.gui1.detection_resize_height)))
        cv2.imshow('detection2', cv2.resize(rest2, (self.gui2.detection_resize_width, self.gui2.detection_resize_height)))
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visualize_matched_element_pairs(self, line=-1):
        board1 = self.gui1.img.copy()
        board2 = self.gui2.img.copy()
        for pair in self.element_matching_pairs:
            color = (rint(0,255), rint(0,255), rint(0,255))
            pair[0].draw_element(board1, color=color, line=line, show_id=False)
            pair[1].draw_element(board2, color=color, line=line, show_id=False)
        cv2.imshow('android', cv2.resize(board1, (int(board1.shape[1] * (800 / board1.shape[0])), 800)))
        cv2.imshow('ios', cv2.resize(board2, (int(board2.shape[1] * (800 / board2.shape[0])), 800)))
        cv2.waitKey()
        cv2.destroyAllWindows()

    def save_matched_element_pairs_clips(self, category='Compo', start_file_id=None, rm_exit=False, output_dir='data/output/matched_compos'):
        '''
        Save the clips of matched element pairs
        @category: "Compo" or "Text"
        @start_file_id: where the saved clip file name start with
        @rm_exit: if remove all previously saved clips
        @output_dir: the root directory for saving
        '''
        if len(self.element_matching_pairs) == 0:
            print('No similar compos matched, run match_similar_elements first')
            return
        if rm_exit:
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if start_file_id is None:
            files = glob(pjoin(output_dir, '*'))
            file_ids = [int(f.replace('\\', '/').split('/')[-1].split('_')[0]) for f in files]
            start_file_id = max(file_ids) + 1 if len(file_ids) > 0 else 0

        for pair in self.element_matching_pairs:
            if pair[0].category == category:
                cv2.imwrite(pjoin(output_dir, str(start_file_id) + '_a.jpg'), pair[0].clip)
                cv2.imwrite(pjoin(output_dir, str(start_file_id) + '_i.jpg'), pair[1].clip)
                start_file_id += 1
