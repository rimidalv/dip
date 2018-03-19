from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import os.path as osp
import glob
import shutil

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..')

os.chdir(lib_path)

sys.path.append(os.getcwd())

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

import warnings

warnings.filterwarnings('ignore')


def load_dict(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.readlines()
        data = [d[:98].strip() for d in data]
    return data





if __name__ == '__main__':
    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in range(2):
    #     _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'results_fin', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'results_fin', '*.jpg'))

    im_names = filter(os.path.isfile, glob.glob(os.path.join(cfg.DATA_DIR, 'results_fin', '*.jpg')))
    # im_names = filter(os.path.isfile, list(im_names))
    im_names = list(im_names)
    im_names.sort(key=lambda x: os.path.getmtime(x))

    words = load_dict("./print_words.txt")
    new_words = []
    first = True

    index = 0
    i = 0
    gl_index = 0
    while i < len(words) * 2:
        if index == len(words):
            index = gl_index
            continue
        if i % 31 == 0:
            print("i%31==0")
        if i % 31 == 0 and first:
            first = False
        elif i % 31 == 0 and not first:
            first = True
            gl_index = index
            index -= 31



        print(i, index, words[index])
        new_words.append(words[index])
        index += 1
        i += 1

    print(len(new_words),len(words))

    image_index = 0
    text_index = 0
    back_arr=[333, 364]
    for_text=[340, 683]
    with open('data/results_fin/' + 'results.txt', 'w', encoding="utf-8") as f:
        for i in range(len(new_words)):
            if image_index == len(new_words):
                break
            # print(i)
            if i in back_arr:
                image_index += 1
                print("cont",new_words[text_index])
                # continue
            if "перерезываться" ==     new_words[text_index]:
                print("text_index", text_index)
            if i in for_text:
                print("skip text", new_words[text_index])
                text_index += 1

                # continue
            base_name = im_names[image_index].split('/')[-1]
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # print(('Demo for {:s}'.format(im_name)))
            f.write(base_name + "\t" + new_words[text_index]+"\n")

            image_index += 1
            text_index += 1