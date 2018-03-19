import numpy as np
import cv2
from .config import cfg
from ..utils.blob import im_list_to_blob


def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    # im_size_min = np.min(im_shape[0:2])
    # im_size_max = np.max(im_shape[0:2])

    processed_ims = [im_orig]
    im_scale_factors = [1.0]

    # for target_size in cfg.TEST.SCALES:
    #     im_scale = float(target_size) / float(im_size_min)
    #     # Prevent the biggest axis from being more than MAX_SIZE
    #     if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
    #         im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    #     im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
    #                     interpolation=cv2.INTER_LINEAR)
    #     im_scale_factors.append(im_scale)
    #     processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def test_ctpn(sess, net, im, boxes=None):
    # print("im.shape", im.shape)
    # print("im", im)
    blobs, im_scales = _get_blobs(im, boxes)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    print("blobs['im_info']", blobs['im_info'])
    # print("blobs", blobs)
    #
    # forward pass
    if cfg.TEST.HAS_RPN:
        feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}

    rois = sess.run([net.get_output('rois')[0]], feed_dict=feed_dict)
    # print("rois 1 === ", np.array(rois).shape)
    rois = rois[0]
    # print("rois 2 === ", np.array(rois))

    # with open("/Users/vladimir/temp/AnacondaProjects/numbers_recognition/text-detection-ctpn/lib/fast_rcnn/1.txt") as f:
    #     a = f.read().split(",")
    #     # rois = np.zeros((len(a)/5, 5), dtype=np.float32)
    #     rois = np.array(a, dtype=np.float32).reshape((-1,5))
    #     # rois = rois[:,[0, 2, 1, 4, 3]]

    # print("rois 3 === ", rois[:10])
    # print("im_scales[0]:", im_scales[0])
    scores = rois[:, 0]

    # print("rois[:]",rois[-10:])
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]
    return scores, boxes


def _get_blobs(im, rois):
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors
