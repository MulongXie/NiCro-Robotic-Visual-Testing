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
        self.min_similarity_img = 0.8
        self.min_shape_difference = 0.8  # min / max

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
        # similarities between the target element and all elements in gui2
        resnet_sims = matching.image_similarity_matrix([target_element.clip], [e.clip for e in compared_elements], method='resnet', resnet_model=self.resnet_model)[0]
        # filter by similarity threshold
        matched_elements_id = np.where(resnet_sims > self.min_similarity_img)[0]
        # select from the compared_elements
        matched_elements = np.array(compared_elements)[matched_elements_id]
        self.show_target_and_matched_elements(target_element, matched_elements, similarities=resnet_sims[matched_elements_id])

        # double check by dhash
        if hash_check and len(matched_elements) > 0:
            dhash_sims= matching.image_similarity_matrix([target_element.clip], [e.clip for e in matched_elements], method='dhash')[0]
            matched_elements_id = np.where(dhash_sims > self.min_similarity_img)[0]
            matched_elements = matched_elements[matched_elements_id]
            self.show_target_and_matched_elements(target_element, matched_elements, similarities=dhash_sims[matched_elements_id])
        return matched_elements

    def match_by_shape(self, target_element, compared_elements):
        matched_elements = []
        for ele in compared_elements:
            if (min(target_element.aspect_ratio / ele.aspect_ratio) / max(target_element.aspect_ratio, ele.aspect_ratio)) > self.min_shape_difference:
                matched_elements.append(ele)
        return matched_elements

    def match_by_neighbour(self, target_element, compared_elements):
        matched_elements = []
        if self.image_similarity_matrix is None:
            self.calculate_elements_image_similarity_matrix()
        return matched_elements

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
        cv2.imshow('detection1', rest1)
        cv2.imshow('detection2', rest2)
        cv2.waitKey()
        cv2.destroyWindow('detection1')
        cv2.destroyWindow('detection2')

    def show_target_and_matched_elements(self, target, matched_elements, similarities=None):
        board1 = self.gui1.img.copy()
        board2 = self.gui2.img.copy()
        target.draw_element(board1, show=False)
        for i, ele in enumerate(matched_elements):
            text = None
            if similarities is not None:
                text = similarities[i]
            ele.draw_element(board2, put_text=text, show=False)
        cv2.imshow('Target', board1)
        cv2.imshow('Matched Elements', board2)
        cv2.waitKey()
        # cv2.destroyWindow('Target')
        # cv2.destroyWindow('Matched Elements')
