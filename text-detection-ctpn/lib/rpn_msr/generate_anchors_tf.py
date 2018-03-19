import tensorflow as tf
import numpy as np

from .generate_anchors import generate_anchors
from ..utils.debug import print_tf

def generate_anchors_pre_tf(height, width, feat_stride,
                            anchor_scales=(8, 16, 32),
                            anchor_ratios=(0.5, 1, 2), session=None):
    """
    A wrapper function to generate anchors given different scales and image
    sizes in tensorflow.
    Note, since `anchor_scales` and `anchor_ratios` is in practice always
    the same, the generate 'base anchors' are static and does we only need to
    implement the shifts part in tensorflow which is depending on the image
    size.

    Parameters:
    -----------
    height: tf.Tensor
        The hight of the current image as a tensor.
    width: tf.Tensor
        The width of the current image as a tensor.
    feat_stride: tf.Tensor or scalar
        The stride used for the shifts.
    anchor_scales: list
        The scales to use for the anchors. This is a static parameter and can
        currently not be runtime dependent.
    anchor_ratios: list
        The ratios to use for the anchors. This is a static parameter and can
        currently not be runtime dependent.

    Returns:
    --------
    anchors: tf.Tensor
        2D tensor containing all anchors, it has the shape (n, 4).
    length: tf.Tensor
        A tensor containing the length 'n' of the anchors.

    """
    anchors = generate_anchors(scales=np.array(anchor_scales))#generate_anchors(ratios=np.array(anchor_ratios),
                               # scales=np.array(anchor_scales))
    print("anchors.shape", anchors.shape)
    num_anchors = anchors.shape[0]
    # Calculate all shifts
    shift_x = tf.range(0, width * feat_stride, feat_stride)
    shift_x = print_tf(shift_x, "#shift_x1 ")
    shift_y = tf.range(0, height * feat_stride, feat_stride)
    shift_y = print_tf(shift_y, "#shift_y1 ")
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = print_tf(shift_x, "#shift_x2 ")
    shift_y = print_tf(shift_y, "#shift_y2 ")
    shift_x = tf.reshape(shift_x, [-1, 1])
    shift_y = tf.reshape(shift_y, [-1, 1])
    shift_x = print_tf(shift_x, "#shift_x3 ")
    shift_y = print_tf(shift_y, "#shift_y3 ")
    shifts = tf.concat((shift_x, shift_y, shift_x, shift_y), 1)
    # print("shifts.shape", shifts)

    # a = tf.Variable(shifts)
    # with tf.Session() as sess:
    #     print("height", sess.run(tf.shape(a)))
    # print("width", width.eval(session=session))
    shifts = print_tf(shifts, "#shifts")
    # Combine all base anchors with all shifts
    anchors = anchors[tf.newaxis] + tf.transpose(shifts[tf.newaxis], (1, 0, 2))
    anchors = print_tf(anchors, "#anchors1 ")
    anchors = tf.cast(tf.reshape(anchors, (-1, 4)), tf.float32)
    anchors = print_tf(anchors, "#anchors2 ")
    # length = tf.shape(anchors)[0]

    # print("length", length)
    return anchors, num_anchors
