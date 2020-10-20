#  Copyright 2019 Gabriele Valvano
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf
import numpy as np


def get_shape(tensor):
    """
    It returns the static shape of a tensor when available, otherwise returns its dynamic shape.

    Args:
        tensor (tensor): input tensor

    Returns:
        Static or dynamic shape.

    Examples:
        x.set_shape([32, 128])  # static shape of a is [32, 128]
        x.set_shape([None, 128])  # first dimension of a is determined dynamical

    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [shape[1] if shape[0] is None else shape[0] for shape in zip(static_shape, dynamic_shape)]
    return dims


def reshape_tensor(tensor, dims_list):
    """
    General purpose reshape function to collapse any list of dimensions.

    Args:
        tensor (tensor): input tensor
        dims_list: list of dimension to collapse

    RetuRns:
        reshaped tensor

    Examples:
        We want to convert a Tensor of rank 3 to a tensor of rank 2 by collapsing the second and third dimensions
        into one:
            b = tf.placeholder(tf.float32, [None, 10, 32])
            shape = get_shape(b)
            b = tf.reshape(b, [shape[0], shape[1] * shape[2]])
        With this function, we can easily write:
            b = tf.placeholder(tf.float32, [None, 10, 32])
            b = reshape(b, [0, [1, 2]])  # hence: collapse [1, 2] into the same dimension, leave 0 dimension unchanged

    """
    shape = get_shape(tensor)
    dims_prod = []
    for dims in dims_list:
        if isinstance(dims, int):
            dims_prod.append(shape[dims])
        elif all([isinstance(shape[d], int) for d in dims]):
            dims_prod.append(np.prod([shape[d] for d in dims]))
        else:
            dims_prod.append(tf.reduce_prod([shape[d] for d in dims]))
    tensor = tf.reshape(tensor, dims_prod)
    return tensor


def one_hot(incoming, nb_classes):
    """ Returns one-hot version of incoming tensor. """
    assert get_shape(incoming)[-1] == 1
    return tf.one_hot(indices=incoming, depth=nb_classes)


def from_one_hot_to_rgb(incoming, palette=None, background='black'):
    """ Assign a different color to each class in the input tensor """
    assert background in ['black', 'white']
    bgd_color = {
        'black': (0, 0, 0),
        'white': (255, 255, 255)
    }
    if palette is None:
        palette = np.array(
            [bgd_color[background],
             [31, 120, 180],
             [215, 25, 29],
             [252, 174, 97],
             [172, 218, 233],
             [255, 127, 0],
             [51, 160, 44],
             [177, 89, 40],
             [166, 206, 227],
             [178, 223, 138],
             [251, 154, 153],
             [253, 191, 111],
             [106, 61, 154]], np.uint8)

    with tf.name_scope('from_one_hot_to_rgb'):
        _, W, H, _ = get_shape(incoming)
        palette = tf.constant(palette, dtype=tf.uint8)
        class_indexes = tf.argmax(incoming, axis=-1)

        class_indexes = tf.reshape(class_indexes, [-1])
        color_image = tf.gather(palette, class_indexes)
        color_image = tf.reshape(color_image, [-1, W, H, 3])

        color_image = tf.cast(color_image, dtype=tf.float32)

    return color_image


def add_histogram(writer, tag, values, step, bins=1000):
    """
    Logs the histogram of a list/vector of values.
    From: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Therefore we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
