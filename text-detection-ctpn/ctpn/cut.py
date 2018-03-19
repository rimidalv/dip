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


words_index = 1919
# words = load_dict("./print_words.txt")
# new_words = []


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    image = cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return image, f


def draw_boxes(img, image_name, boxes, index):
    base_name = image_name.split('/')[-1]
    # global new_words
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue

        img = img[int(box[1]):int(box[5]), int(box[0]):int(box[2] + 20)]

    save_name = base_name.split('.')[0] + "_" + str(index) + "." + base_name.split('.')[1]
    print(save_name)
    # new_words.append(save_name + "\t" + word)
    cv2.imwrite(os.path.join("data/results", save_name), img)


def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()
    # global words_index
    image = cv2.imread(image_name)

    slice_count = 6
    off = image.shape[0] // slice_count
    while True:
        img = image[0:off, :]
        print("img.shape", image.shape)
        scale = 1.0
        global words_index
        # img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        scores, boxes = test_ctpn(sess, net, img)

        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])

        # if words_index == 1:
        #     break

        if len(boxes) == 0:
            break

        box = boxes[boxes[:, 1].argsort()][0]
        box = np.expand_dims(box, axis=0)
        print()
        draw_boxes(img, image_name, box, words_index)
        words_index += 1
        print("box[0]", box[0])
        image = image[int(box[0][5] + 20):, :, :]
        print(('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
    timer.toc()


if __name__ == '__main__':
    # if os.path.exists("data/results/"):
    #     shutil.rmtree("data/results/")
    # os.makedirs("data/results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in range(2):
    #     _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    start = False
    for im_name in im_names:
        if im_name.split('/')[-1] == "20180316_124346.jpg":
            start = True

        if start    :
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(('Demo for {:s}'.format(im_name)))
            ctpn(sess, net, im_name)

    # with open('data/results/' + 'results.txt', 'w', encoding="utf-8") as f:
    #     f.write("\n".join(new_words))
