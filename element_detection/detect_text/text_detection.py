import element_detection.detect_text.ocr as ocr
from element_detection.detect_text.Text import Text
import cv2
import json
import time
from os.path import join as pjoin
import numpy as np
from paddleocr import PaddleOCR
import requests
import base64


def save_detection_json(file_path, texts, img_shape):
    f_out = open(file_path, 'w')
    output = {'img_shape': img_shape, 'texts': []}
    for text in texts:
        c = {'id': text.id, 'content': text.content, 'keyboard': text.keyboard}
        loc = text.location
        c['column_min'], c['row_min'], c['column_max'], c['row_max'] = loc['left'], loc['top'], loc['right'], loc['bottom']
        c['width'] = text.width
        c['height'] = text.height
        output['texts'].append(c)
    json.dump(output, f_out, indent=4)


def visualize_texts(org_img, texts, shown_resize_height=None, show=False, write_path=None):
    img = org_img.copy()
    for text in texts:
        text.visualize_element(img, line=2)

    img_resize = img
    if shown_resize_height is not None:
        img_resize = cv2.resize(img, (int(shown_resize_height * (img.shape[1]/img.shape[0])), shown_resize_height))

    if show:
        cv2.imshow('texts', img_resize)
        cv2.waitKey(0)
        cv2.destroyWindow('texts')
    if write_path is not None:
        cv2.imwrite(write_path, img)
    return img


def text_sentences_recognition(texts):
    '''
    Merge separate words detected by Google ocr into a sentence
    '''
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            # ignore keyboard letters
            if text_a.keyboard:
                temp_set.append(text_a)
                continue
            merged = False
            for text_b in temp_set:
                if text_b.keyboard:
                    continue
                if text_a.is_on_same_line(text_b, 'h', bias_justify=0.2 * min(text_a.height, text_b.height), bias_gap=2 * max(text_a.word_width, text_b.word_width)):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()

    for i, text in enumerate(texts):
        text.id = i
    return texts


def merge_intersected_texts(texts):
    '''
    Merge intersected texts (sentences or words)
    '''
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                inter, iou, ioa, iob = text_a.calc_intersection_area(text_b, (2, 2))
                if inter > 0:
                    # if text_a is contained by others, discard text_a
                    if ioa >= 0.8:
                        merged = True
                        break
                    # if text_a contains text_b, ignore text_b
                    elif iob >= 0.8:
                        continue
                    # if no containment relationship, merge text_a and text_b
                    else:
                        text_b.merge_text(text_a)
                        merged = True
                        changed = True
                        break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return texts


def text_cvt_orc_format(ocr_result, img):
    texts = []
    if ocr_result is not None:
        for i, result in enumerate(ocr_result):
            error = False
            x_coordinates = []
            y_coordinates = []
            text_location = result['boundingPoly']['vertices']
            content = result['description']
            for loc in text_location:
                if 'x' not in loc or 'y' not in loc:
                    error = True
                    break
                x_coordinates.append(loc['x'])
                y_coordinates.append(loc['y'])
            if error: continue
            location = {'left': min(x_coordinates), 'top': min(y_coordinates),
                        'right': max(x_coordinates), 'bottom': max(y_coordinates)}
            text = Text(i, content, location)
            text.get_clip(img)
            texts.append(text)
    return texts


def text_filter_noise(texts):
    valid_texts = []
    for text in texts:
        if len(text.content) <= 1 and text.content.lower() not in ['a', ',', '.', '!', '?', '$', '%', ':', '&', '+']:
            continue
        valid_texts.append(text)
    return valid_texts


def text_split_letters(texts, img):
    new_texts = []
    latest_id = len(texts)
    for text in texts:
        letters = text.split_letters_in_the_word(latest_id)
        if len(letters) > 1:
            latest_id += len(letters) - 1
            for letter in letters:
                letter.get_clip(img)
            new_texts += letters
        else:
            new_texts.append(text)
    return new_texts


def text_recognize_keyboard_letters(texts, img):
    '''
    Recognize keyboard letters from texts.
    If a text and its neighbours are all in the keyboard area, it is recognized as a keyboard letter
    '''
    height, width = img.shape[:2]
    # sort the texts top down
    no_keyboard_letters = 0
    texts = sorted(texts, key=lambda x: x.location['top'])
    for i in range(len(texts) - 1):
        t_i = texts[i]
        # if the ti is in keyboard_area, search for its horizontal neighbours
        if t_i.is_in_keyboard_area(height):
            for j in range(i + 1, len(texts)):
                t_j = texts[j]
                # stop when no horizontal neighbours
                if t_j.location['top'] - t_i.location['bottom'] > t_i.height:
                    break
                # if the neighbour is a keyboard letter or is also in the keyboard area, mark ti as keyboard letter
                if t_j.keyboard:
                    t_i.keyboard = True
                    no_keyboard_letters += 1
                    break
                else:
                    if t_j.is_justified(t_i, direction='h', max_bias_justify=max(t_i.height, t_j.height)) and t_j.is_in_keyboard_area(height):
                        t_i.keyboard = True
                        t_j.keyboard = True
                        no_keyboard_letters += 1
                        break
    # Ignore all keyboard letters if the of letters are too few
    if no_keyboard_letters <= 10:
        for text in texts:
            text.keyboard = False
    return texts


def text_detection_google(input_file='../data/input/30800.jpg', ocr_root='../data/output', show=False):
    start = time.clock()
    name = input_file.split('/')[-1][:-4]
    img = cv2.imread(input_file)

    ocr_result = ocr.ocr_detection_google(input_file)
    texts = text_cvt_orc_format(ocr_result, img)
    texts = merge_intersected_texts(texts)
    texts = text_split_letters(texts, img)
    texts = text_recognize_keyboard_letters(texts, img)
    # texts = text_filter_noise(texts)
    texts = text_sentences_recognition(texts)
    board = visualize_texts(img, texts, shown_resize_height=800, show=show, write_path=pjoin(ocr_root, name+'.png'))
    save_detection_json(pjoin(ocr_root, name+'.json'), texts, img.shape)
    print("[Text Detection Completed in %.3f s] Input: %s Output: %s" % (time.clock() - start, input_file, pjoin(ocr_root, name+'.json')))
    return board, texts


def text_cvt_orc_format_paddle(paddle_result, img):
    texts = []
    for i, line in enumerate(paddle_result):
        points = np.array(line[0])
        location = {'left': int(min(points[:, 0])), 'top': int(min(points[:, 1])), 'right': int(max(points[:, 0])),
                    'bottom': int(max(points[:, 1]))}
        content = line[1][0]
        text = Text(i, content, location)
        text.get_clip(img)
        texts.append(text)
    return texts


def text_detection_paddle(input_file='../data/input/30800.jpg', ocr_root='../data/output', show=False, paddle_ocr=None):
    start = time.time()
    name = input_file.replace('\\', '/').split('/')[-1][:-4]
    img = cv2.imread(input_file)

    # if paddle_ocr is None:
    #     paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    result = paddle_ocr.ocr(input_file, cls=True)
    texts = text_cvt_orc_format_paddle(result, img)

    board = visualize_texts(img, texts, shown_resize_height=800, show=show, write_path=pjoin(ocr_root, name+'.png'))
    save_detection_json(pjoin(ocr_root, name+'.json'), texts, img.shape)
    print("[Text Detection Completed in %.3f s] Input: %s Output: %s" % (time.time() - start, input_file, pjoin(ocr_root, name+'.json')))
    return board, texts
