import argparse
import os
import sys
import string

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ROOT = os.path.abspath('./')
sys.path.append(ROOT)


from test import start_craft

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Text Reader')

    parser.add_argument('--command',
                        help="'train, detect, evaluate, ocr'")

    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.1, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.3, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.9, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=True, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='images/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
    parser.add_argument('--image_folder', default='lol', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', help="path to saved_model to evaluation", default='weights/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default=string.printable[:-6], help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, help='Transformation stage. None|TPS', default='TPS')
    parser.add_argument('--FeatureExtraction', type=str, help='FeatureExtraction stage. VGG|RCNN|ResNet', default='ResNet')
    parser.add_argument('--SequenceModeling', type=str, help='SequenceModeling stage. None|BiLSTM', default='BiLSTM')
    parser.add_argument('--Prediction', type=str, help='Prediction stage. CTC|Attn', default='Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    args = parser.parse_args()

    args = parser.parse_args()
    save_and_write_coordinates, ocr_images = start_craft(args, ROOT)
    print('Погнали!')

    if args.command == 'ocr':
        ocr_images()

    if args.command == 'get_coord':
        save_and_write_coordinates()


