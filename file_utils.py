# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import cv2
import imgproc
from copy import deepcopy
from PIL import Image
from PerspectiveTransform import four_point_transform
from deeptext.demo import demo
#import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'





def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


def split_image(img_file, image, boxes, root):

    result_folder = f'{root}/output_images/'

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    #bboxes = []
    filename, _ = os.path.splitext(os.path.basename(img_file))
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cv2.polylines(image, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        #warped = four_point_transform(image, poly)
        #bboxes.append(warped)
    # img_ = Image.fromarray(warped, 'RGB')
    cv2.imwrite(f'{root}/output_images/{filename}.jpg', image)



def join_text(text):
    return ' '.join(text) + '\n'


def saveResult(img_file, img, boxes, converter=None, model=None, args=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            text of picture, picture's name

        """
        img = np.array(img)
        img_copy = deepcopy(img)
        # make result file list
        filename, _ = os.path.splitext(os.path.basename(img_file))
        font = cv2.FONT_HERSHEY_SIMPLEX

        bboxes = []

        for idx, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            warped = four_point_transform(img_copy, poly)
            img_ = Image.fromarray(warped, 'RGB')
            bboxes.append(img_)
        text = demo(args, bboxes, model, converter)
        return text, filename
