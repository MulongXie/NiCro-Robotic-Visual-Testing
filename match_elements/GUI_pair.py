import cv2
import json
import os
import numpy as np
import time
import shutil
from os.path import join as pjoin
from random import randint as rint
from glob import glob

import match_elements.matching as match
from sklearn.metrics.pairwise import cosine_similarity


class GUIPair:
    def __init__(self, gui1, gui2, output_dir='data/output'):
        self.gui1 = gui1
        self.gui2 = gui2
        self.output_dir = output_dir

        self.elements1 = gui1.elements      # list of Element objects for android UI
        self.elements2 = gui2.elements      # list of Element objects for ios UI
        self.element_matching_pairs = []    # list of matching similar element pairs: [(ele1, ele2)]

    '''
    ******************************
    *** Match Similar Elements ***
    ******************************
    '''
    def match_similar_elements(self, min_similarity_img=0.75, min_similarity_text=0.8, pair_shape_thresh=1.5,
                               img_sim_method='dhash', del_prev=True, resnet_model=None):
        '''
        @min_similarity_img: similarity threshold for Non-text elements
        @min_similarity_text: similarity threshold for Text elements
        @pair_shape_thresh: shape difference threshold for a matched compo pair
        @img_sim_method: the method used to calculate the similarity between two images
            options: 'dhash', 'ssim', 'sift', 'surf'
        @del_prev: if to delete all previously matched compos
        @resnet_model: pre-loaded resnet model
        '''
        if del_prev:
            self.element_matching_pairs = []
            for ele in self.elements1 + self.elements2:
                ele.matched_element = None

        start = time.clock()
        if img_sim_method == 'resnet' and resnet_model is None:
            from keras.applications.resnet import ResNet50
            resnet_model = ResNet50(include_top=False, input_shape=(32, 32, 3))

        mark = np.full(len(self.elements2), False)
        n_compos = 0
        n_texts = 0
        for ele_a in self.elements1:
            for j, ele_b in enumerate(self.elements2):
                # only match elements in the same category
                if ele_b.matched_element is not None or ele_a.category != ele_b.category:
                    continue
                # filter out some impossible pairs
                if mark[j] or \
                        max(ele_a.height, ele_b.height) / min(ele_a.height, ele_b.height) > pair_shape_thresh or max(ele_a.width, ele_b.width) / min(ele_a.width, ele_b.width) > pair_shape_thresh or \
                        max(ele_a.aspect_ratio, ele_b.aspect_ratio) / min(ele_a.aspect_ratio, ele_b.aspect_ratio) > pair_shape_thresh:
                    continue
                # use different method to calc the similarity of of images and texts
                if ele_a.category == 'Compo':
                    # match non-text clip through image similarity
                    compo_similarity = match.image_similarity(ele_a.clip, ele_b.clip, method=img_sim_method, resnet_model=resnet_model)
                    if compo_similarity > min_similarity_img:
                        n_compos += 1
                        self.element_matching_pairs.append((ele_a, ele_b))
                        ele_a.matched_element = ele_b
                        ele_b.matched_element = ele_a
                        mark[j] = True
                        break
                elif ele_a.category == 'Text':
                    # match text by through string similarity
                    text_similarity = match.text_similarity(ele_a.text_content, ele_b.text_content)
                    if text_similarity > min_similarity_text:
                        n_texts += 1
                        self.element_matching_pairs.append((ele_a, ele_b))
                        ele_a.matched_element = ele_b
                        ele_b.matched_element = ele_a
                        mark[j] = True
                        break
        print('[Similar Elements Matching %.3fs] Method:%s Paired Text:%d, Paired Compos:%d' % ((time.clock() - start), img_sim_method, n_texts, n_compos))

    def match_similar_elements_resnet(self, min_similarity_img=0.75, min_similarity_text=0.8, pair_shape_thresh=1.5, del_prev=True, resnet_model=None):
        '''
        @min_similarity_img: similarity threshold for Non-text elements
        @min_similarity_text: similarity threshold for Text elements
        @pair_shape_thresh: shape difference threshold for a matched compo pair
        @del_prev: if to delete all previously matched compos
        @resnet_model: pre-loaded resnet model
        '''
        if del_prev:
            self.element_matching_pairs = []
            for ele in self.elements1 + self.elements2:
                ele.matched_element = None
        start = time.clock()

        # extract compos and texts
        clips = []
        compos1 = []
        texts1 = []
        compos2 = []
        texts2 = []
        for ele in self.elements1:
            if ele.category == 'Compo':
                clips.append(cv2.resize(ele.clip, (32, 32)))
                compos1.append(ele)
            elif ele.category == 'Text':
                texts1.append(ele)
        for ele in self.elements2:
            if ele.category == 'Compo':
                clips.append(cv2.resize(ele.clip, (32, 32)))
                compos2.append(ele)
            elif ele.category == 'Text':
                texts2.append(ele)

        # encode the compos through resnet
        encodings = resnet_model.predict(np.array(clips))
        encodings = encodings.reshape((encodings.shape[0], -1))
        encodings_1 = encodings[:len(compos1)]
        encodings_2 = encodings[len(compos1):]

        # match compos
        mark = np.full(len(compos2), False)
        n_compos = 0
        for i, ele_a in enumerate(compos1):
            candidates = []
            for j, ele_b in enumerate(compos2):
                # filter out some impossible pairs
                if mark[j] or \
                        max(ele_a.height, ele_b.height) / min(ele_a.height, ele_b.height) > pair_shape_thresh or max(ele_a.width, ele_b.width) / min(ele_a.width, ele_b.width) > pair_shape_thresh or \
                        max(ele_a.aspect_ratio, ele_b.aspect_ratio) / min(ele_a.aspect_ratio, ele_b.aspect_ratio) > pair_shape_thresh:
                    continue
                # calculate the similarity between two compos
                compo_similarity = cosine_similarity([encodings_1[i]], [encodings_2[j]])[0][0]
                if compo_similarity > min_similarity_img:
                    candidates.append((j, compo_similarity))
            if len(candidates) > 0:
                n_compos += 1
                max_sim_ele = max(candidates, key=lambda x: x[1])
                ele_b = compos2[max_sim_ele[0]]
                self.element_matching_pairs.append((ele_a, ele_b))
                ele_a.matched_element = ele_b
                ele_b.matched_element = ele_a
                mark[max_sim_ele[0]] = True

        # match texts
        mark = np.full(len(texts2), False)
        n_texts = 0
        for i, ele_a in enumerate(texts1):
            candidates = []
            for j, ele_b in enumerate(texts2):
                # filter out some impossible pairs
                if mark[j] or \
                        max(ele_a.height, ele_b.height) / min(ele_a.height, ele_b.height) > pair_shape_thresh or max(ele_a.width, ele_b.width) / min(ele_a.width, ele_b.width) > pair_shape_thresh or \
                        max(ele_a.aspect_ratio, ele_b.aspect_ratio) / min(ele_a.aspect_ratio, ele_b.aspect_ratio) > pair_shape_thresh:
                    continue
                # match text by through string similarity
                text_similarity = match.text_similarity(ele_a.text_content, ele_b.text_content)
                if text_similarity > min_similarity_text:
                    candidates.append((j, text_similarity))
            if len(candidates) > 0:
                n_texts += 1
                max_sim_ele = max(candidates, key=lambda x: x[1])
                ele_b = texts2[max_sim_ele[0]]
                self.element_matching_pairs.append((ele_a, ele_b))
                ele_a.matched_element = ele_b
                ele_b.matched_element = ele_a
                mark[max_sim_ele[0]] = True

        print('[Similar Elements Matching %.3fs] Method:Resnet Paired Text:%d, Paired Compos:%d' % ((time.clock() - start), n_texts, n_compos))

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
