# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Markus Nagel
# --------------------------------------------------------
import tensorflow as tf
from ..fast_rcnn.config import cfg
from ..utils.debug import print_tf

def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, anchors, num_anchors, height, width):
    """
    Tensorflow implementation of the proposal layer. This does exactly the same
    as the original proposal layer.

    Note the results can be slightly different because of:
    1) Tensorflow and numpy sorting sometimes have a different order if the
       scores are the same (`tf.nn.top_k` vs `np.argsort`)
    2) Tensorflow non maximum suppression gives slightly different results
       than the implementation from here. This might be due to a slight
       different implementation or some rounding errors.

    Parameters:
    -----------
    rpn_cls_prob: tf.Tensor
        The classification probability of the RPN layer, aka score of
        objectiveness of the box.
    rpn_bbox_pred: tf.Tensor
        The predicted offset of the bounding box by the RPN layer.
    im_info: tf.Tensor
        The information about the image, [1, width, height].
    cfg_key: string
        The config to use.
    anchors: tf.Tensor
        The anchors of the bounding boxes.
    num_anchors:
        The number of anchors.

    Returns:
    --------
    blob: tf.Tensor
        The final bounding boxes.
    scores: tf.Tensor
        The scores/objectiveness of the bounding boxes.

    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
    min_size = cfg[cfg_key].RPN_MIN_SIZE

    rpn_cls_prob = print_tf(rpn_cls_prob, "rpn_cls_prob")
    print("num_anchors", num_anchors)
    # Get the scores and bounding boxes
    # print("rpn_cls_prob_reshape.shape ", rpn_cls_prob.shape)
    reshape = tf.reshape(rpn_cls_prob, [1, height, width, num_anchors, 2])
    reshape = print_tf(reshape, "reshape")
    # print("reshape.shape ", reshape.shape)
    scores = tf.reshape(reshape[:, :, :, :, 1], [1, height, width, num_anchors])

    # scores = rpn_cls_prob[:, :, :, num_anchors:]
    scores = print_tf(scores, "scores 1")

    bbox_deltas = tf.reshape(rpn_bbox_pred, (-1, 4))
    bbox_deltas = print_tf(bbox_deltas, "bbox_deltas")
    scores = tf.reshape(scores, [-1, 1])
    scores = print_tf(scores, "scores 2")

    proposals_inv = bbox_transform_inv_tf(anchors, bbox_deltas)
    proposals_inv = print_tf(proposals_inv, "proposals_inv")

    proposals_clip = clip_boxes_tf(proposals_inv, im_info[0, :2])
    proposals_clip = print_tf(proposals_clip, "proposals_clip")

    # Pick the top region proposals
    if pre_nms_topN > 0:
        topN = tf.minimum(tf.shape(scores)[0], pre_nms_topN)
        # topN = tf.minimum(tf.constant(1000), pre_nms_topN)
        topN = print_tf(topN, "#topN pre_nms_topN > 0 ")
    else:
        topN = tf.shape(scores)[0]
        topN = print_tf(topN, "#topN pre_nms_topN <= 0 ")
    scores_0 = print_tf(scores[:, 0], "scores[:, 0]")
    scores_topn, order = tf.nn.top_k(scores_0, topN, sorted=True)

    order = print_tf(order, "order")
    scores_topn = print_tf(scores_topn, "scores_topn")

    proposals_topn = tf.gather(proposals_clip, order)
    proposals_topn = print_tf(proposals_topn, "proposals_topn")

    bbox_deltas = tf.gather(bbox_deltas, order)
    bbox_deltas = print_tf(bbox_deltas, "bbox_deltas_topn")
    # Non-maximal suppression
    topN = post_nms_topN if post_nms_topN > 0 else -1
    topN = print_tf(topN, "topN")

    keep = tf.image.non_max_suppression(proposals_topn, scores_topn,
                                        topN, iou_threshold=nms_thresh)
    keep = print_tf(keep, "keep")

    # Select the region proposals after NMS
    proposals_keep = tf.gather(proposals_topn, keep)
    proposals_keep = print_tf(proposals_keep, "proposals_keep")

    # Select the region proposals after NMS
    bbox_deltas = tf.gather(bbox_deltas, keep)
    bbox_deltas = print_tf(bbox_deltas, "bbox_deltas_keep")

    scores_keep = tf.gather(scores_topn, keep)
    scores_keep = print_tf(scores_keep, "scores_keep")

    # Only support single image as input
    # batch_inds = tf.zeros((tf.shape(proposals_keep)[0], 1), dtype=tf.float32)
    # batch_inds = print_tf(batch_inds, "batch_inds")
    blob = tf.concat([scores_keep[:, tf.newaxis], proposals_keep], 1)

    return blob, bbox_deltas#[:, tf.newaxis]





def clip_boxes_tf(boxes, im_shape):
    """
    Clip boxes to image boundaries with tensorflow. Note here we assume
    that boxes is always of shape (n, 4).
    """
    clipped_boxes = tf.concat([
        # x1 >= 0
        tf.maximum(tf.minimum(boxes[:, 0:1], im_shape[1] - 1), 0),
        # y1 >= 0
        tf.maximum(tf.minimum(boxes[:, 1:2], im_shape[0] - 1), 0),
        # x2 < im_shape[1]
        tf.maximum(tf.minimum(boxes[:, 2:3], im_shape[1] - 1), 0),
        # y2 < im_shape[0]
        tf.maximum(tf.minimum(boxes[:, 3:4], im_shape[0] - 1), 0)], 1)
    return clipped_boxes


def bbox_transform_inv_tf(boxes, deltas):
    """
    TF implementation of bbox_transform_inv. Note here we assume
    that boxes and deltas are always of shape (n, 4).
    """

    # boxes = tf.cast(boxes, deltas.dtype)  # TODO maybe remove?
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    print("deltas", dx, dy, dw, dh)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights

    pred_boxes = tf.transpose(tf.stack([
        # x1
        pred_ctr_x - 0.5 * pred_w,
        # y1
        pred_ctr_y - 0.5 * pred_h,
        # x2
        pred_ctr_x + 0.5 * pred_w,
        # y2
        pred_ctr_y + 0.5 * pred_h, ]))

    return pred_boxes
