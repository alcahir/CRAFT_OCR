"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from operator import itemgetter
from PIL import Image
from scipy.spatial import distance
import cv2
from skimage import io
import numpy as np
import pandas as pd
import craft_utils
import imgproc
import file_utils
import json
import zipfile
from craft import CRAFT
from collections import OrderedDict, defaultdict as dd
from deeptext.model import Model
from deeptext.utils import CTCLabelConverter, AttnLabelConverter


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def centroids(boxes):
    return [((box[0][0]+box[2][0])/2, (box[0][1]+box[2][1])/2) for box in boxes]

def manhatten_distances(centers):
    distances = []

    for i, center in enumerate(centers):
        dist = distance.minkowski((0,0), center, 1, w=[0.4, 1])
        distances.append((dist, i))

    return sorted(distances, key=lambda x: x[0])


def sort_boxes(boxes):

    centers = centroids(boxes)
    arrs = sorted(boxes, key = lambda x: x[0][1])
    sorted_boxes = []
    i = 0

    if len(arrs) > 1:
        for idx in range(1, len(arrs)):

            if not(arrs[idx][0][1] - 3 <= centers[idx - 1][1] <= arrs[idx][2][1] + 3) :
                if idx < len(arrs) - 1:
                    sorted_boxes.extend(sorted(arrs[i: idx], key=lambda x: x[0][0]))
                    i = idx
                    continue
                if idx == len(arrs) - 1 and idx - 1 == i:
                    sorted_boxes.extend(sorted(arrs[idx - 1: ], key = lambda x: x[0][0]))
                    break
                else:
                    sorted_boxes.extend(sorted(arrs[i:], key=lambda x: x[0][0]))
                    break

    else:
        sorted_boxes.extend(boxes)

    return sorted_boxes


def order_hash_table(centers, chap_coord, polys):
    chap_coord = np.array([np.array(box).astype(np.int32).reshape((-1)) for box in chap_coord])
    groups = dd(list)
    visited = set()
    for i, chapter in enumerate(chap_coord):
        left_x, left_y, right_x, right_y = chapter[0], chapter[1], chapter[4], chapter[5]

        for j, center in enumerate(centers):

            x, y = center[0], center[1]

            if (left_x - 8 < x < right_x + 8) and (left_y - 8 < y < right_y + 8):
                visited.add(j)
                groups[i].append(polys[j])

        groups[i] = sorted(groups[i], key = lambda x: x[0][0])

    return groups


# parser = argparse.ArgumentParser(description='CRAFT Text Detection')
# parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
# parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
# parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
# parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
# parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
# parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
# parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
# parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
# parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
# parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
# parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
# parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
# parser.add_argument('--image_folder', default='lol', help='path to image_folder which contains text images')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
# parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
# parser.add_argument('--saved_model', help="path to saved_model to evaluation", default='weights/TPS-ResNet-BiLSTM-Attn.pth')
# """ Data processing """
# parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
# parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
# parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
# parser.add_argument('--rgb', action='store_true', help='use rgb input')
# parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
# parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
# parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
# """ Model Architecture """
# parser.add_argument('--Transformation', type=str, help='Transformation stage. None|TPS', default='TPS')
# parser.add_argument('--FeatureExtraction', type=str, help='FeatureExtraction stage. VGG|RCNN|ResNet', default='ResNet')
# parser.add_argument('--SequenceModeling', type=str, help='SequenceModeling stage. None|BiLSTM', default='BiLSTM')
# parser.add_argument('--Prediction', type=str, help='Prediction stage. CTC|Attn', default='Attn')
# parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
# parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
# parser.add_argument('--output_channel', type=int, default=512,
#                     help='the number of output channel of Feature extractor')
# parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
# args = parser.parse_args()


""" For test images in a folder """
#image_list, _, _ = file_utils.get_files(args.test_folder)


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, args, refine_net=None):
    t0 = time.time()


    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



def start_craft(args, ROOT):
    # load net
    net = CRAFT()     # initialize

    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)

    args.num_class = len(converter.character)

    if args.rgb:
        args.input_channel = 3
    model = Model(args)
    print('model input parameters', args.imgH, args.imgW, args.num_fiducial, args.input_channel, args.output_channel,
          args.hidden_size, args.num_class, args.batch_max_length, args.Transformation, args.FeatureExtraction,
          args.SequenceModeling, args.Prediction)
    model = torch.nn.DataParallel(model).to('cpu')

    # load model
    print('loading pretrained model from %s' % args.saved_model)
    model.load_state_dict(torch.load(args.saved_model, map_location='cpu'))

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data


    result_folder = './result/'

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    def save_and_write_coordinates():
        for dirpath, dirnames, filenames in os.walk(args.test_folder):

            if filenames:
                for k, filename in enumerate(filenames):
                    image_path = os.path.join(dirpath, filename)
                    filename, _ = os.path.splitext(os.path.basename(image_path))
                    print("Test image {:d}/{:d}: {:s}".format(k, len(filenames), image_path), end='\r')
                    image = imgproc.loadImage(image_path)
                    bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold,
                                                         args.low_text, args.cuda, args.poly, args, refine_net=refine_net)
                    file_utils.split_image(image_path, image, polys, ROOT)
                    np.save(f'{ROOT}/result/{filename}.npy', polys)



    def ocr_images():
        dataframe = pd.DataFrame(columns=['name', 'characters'])

        for dirpath, dirnames, filenames in os.walk(args.test_folder):
            if filenames:
                folder_name = os.path.split(dirpath)[-1]
                for k, filename in enumerate(filenames):

                    image_path = os.path.join(dirpath, filename)
                    filename, _ = os.path.splitext(os.path.basename(image_path))
                    print("Test image {:d}/{:d}: {:s}".format(k, len(filenames), image_path), end='\r')
                    image = imgproc.loadImage(image_path)
                    bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold,
                                                         args.low_text, args.cuda, args.poly, args, refine_net=refine_net)

                    text, name = file_utils.saveResult(image_path, image[:, :, ::-1], polys, converter=converter, args=args,
                                                      model=model)
                    df = pd.DataFrame(np.array([[folder_name + '_' + str(name), text]]), columns=['name', 'characters'])
                    dataframe = dataframe.append(df, ignore_index=False)

        dataframe.to_csv('out.csv', index=False)

    return  save_and_write_coordinates, ocr_images