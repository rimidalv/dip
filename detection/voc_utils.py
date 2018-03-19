"""
Convert Pascal VOC 2007+2012 detection dataset to HDF5.

Does not preserve full XML annotations.
Combines all VOC subsets (train, val test) with VOC2012 train for full
training set as done in Faster R-CNN paper.

Code based on:
https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
"""

import argparse
import os
import glob
import xml.etree.ElementTree as ElementTree
import tensorflow as tf
import h5py
import numpy as np

classes = []

parser = argparse.ArgumentParser(
    description='Convert Pascal VOC 2007+2012 detection dataset to HDF5.')
parser.add_argument(
    '-p',
    '--path_to_voc',
    help='path to VOCdevkit directory',
    default='images_output')


def get_boxes_for_id(voc_path, image_id):
    """Get object bounding boxes annotations for given image.

    Parameters
    ----------
    voc_path : str
        Path to VOCdevkit directory.

    image_id : str
        Pascal VOC identifier for given image.

    Returns
    -------
    boxes : array of int
        bounding box annotations of class label, xcenter, ycenter, box_width, box_height, image_width, image_height
        7xN array.
    """
    fname = os.path.join(voc_path, 'obj/{}.txt'.format(image_id))

    boxes = []
    with open(fname) as txt_file:
        for line in txt_file.readlines():
            if(len(line) < 1):
                continue
            try:
                elems = line.split(' ')
                bbox = [
                    float(elems[1]),
                    float(elems[2]),
                    float(elems[3]),
                    float(elems[4]),
                    int(elems[0]),
                ]
                boxes.append(bbox)
            except:
                "problem with get boxes from file: " + str(fname)    
           

    return np.array(boxes)


def get_image_for_id(voc_path, image_id, sess, decoded_jpeg, image_placeholder):
    """Get image data as uint8 array for given image.

    Parameters
    ----------
    voc_path : str
        Path to VOCdevkit directory.

    image_id : str
        Pascal VOC identifier for given image.

    Returns
    -------
    image_data : array of uint8
        Compressed JPEG byte string represented as array of uint8.
    """
    image_path = os.path.join(voc_path, 'obj/{}.JPEG'.format(image_id))

    with open(image_path, 'rb') as f:
        image_data = f.read()

    image = sess.run(decoded_jpeg,
                     feed_dict={image_placeholder: image_data})

    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return np.array(image), width, height

    # print("image name: ",fname)
    # with open(fname, 'rb') as in_file:
    #     data = in_file.read()
    # return np.fromstring(data, dtype='uint8')


def get_ids(voc_path):
    """Get image identifiers for corresponding list of dataset identifies.

    Parameters
    ----------
    voc_path : str
        Path to VOCdevkit directory.
    datasets : list of str tuples
        List of dataset identifiers in the form of (year, dataset) pairs.

    Returns
    -------
    ids : list of str
        List of all image identifiers for given datasets.
    """
    ids = []
    print("voc")

    files_images = glob.iglob(os.path.join(voc_path, "*.JPEG"))
    for x in files_images:
        name = os.path.splitext(os.path.basename(x))[0]
        ids.append(name)
    print("names: ", ids)
    return ids


def get_data(voc_path, ids):
    images = []
    boxes = []
    with tf.Session() as sess:
        image_placeholder = tf.placeholder(dtype=tf.string)
        decoded_jpeg = tf.image.decode_jpeg(image_placeholder, channels=3)
        lenght = len(ids)
        for i, id in enumerate(ids):
            image_data, width, height = get_image_for_id(
                voc_path, id, sess, decoded_jpeg, image_placeholder)
            image_boxes = get_boxes_for_id(voc_path, id)
            images.append(image_data)
            boxes.append(image_boxes)
            # if i % 10 == 0:
            #     print(float(i) / lenght * 100)
    return images, boxes


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_datas(path_to_voc_root="in/data/"):
    voc_path = os.path.expanduser(path_to_voc_root)
    print("root path: ", voc_path)

    classes = get_classes(os.path.join(voc_path, "obj.names"))
    ids = get_ids(os.path.join(voc_path, "obj/"))

    images, boxes = get_data(voc_path, ids)

    # print("classes: {}\nboxes: {}".format(classes,  boxes))
    return classes, images, boxes, ids


def _main(args):
    voc_path = os.path.expanduser(args.path_to_voc)
    get_datas(path_to_voc_root=voc_path)


if __name__ == '__main__':
    _main(parser.parse_args())
