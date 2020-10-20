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


def dice_coe(output, target, axis=(1, 2, 3), smooth=1e-12):
    """
    Soft Sørensen–Dice coefficient (also known as just DICE coefficient) for evaluating the similarity of two batch of
    data. The coefficient can vary between 0 and 1, where 1 means totally match. It is usually used for binary image
    segmentation (e.g. using the loss function: 1 - dice_coe(...)).

    Args:
        output (Tensor): a distribution with shape: [batch_size, ....], (any dimensions). This is the prediction.
        target (Tensor): the target distribution, format the same with `output`.
        axis (tuple of int): contains all the dimensions to be reduced, default ``[1,2,3]``.
        smooth (float): small value added to the numerator and denominator.

    Returns:
        Dice coefficient.

    References:
        `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """

    assert output.dtype in [tf.float32, tf.float64]
    assert target.dtype in [tf.float32, tf.float64]

    intersection = tf.reduce_sum(output * target, axis=axis)

    a = tf.reduce_sum(output, axis=axis)
    b = tf.reduce_sum(target, axis=axis)

    score = (2. * intersection + smooth) / (a + b + smooth)
    score = tf.reduce_mean(score, name='dice_coe')
    return score


def generalized_dice_coe(output, target, axis=(1, 2, 3), smooth=1e-12):
    """Generalized Soft Sørensen–Dice coefficient for evaluating the similarity of two batch of data. The coefficient
    can vary between 0 and 1, where 1 means totally match. It is usually used for binary image segmentation (e.g. using
    the loss function: 1 - dice_coe(...)).

    Args:
        output (Tensor): a distribution with shape: [batch_size, ....], (any dimensions). This is the prediction.
        target (Tensor): the target distribution, format the same with `output`.
        axis (tuple of int): contains all the dimensions to be reduced, default ``[1,2,3]``.
        smooth (float): small value added to the numerator and denominator.

    Returns:
        Generalized Dice coefficient.

    Examples:
        outputs = softmax(network.outputs)
        dice_loss = 1.0 - generalized_dice_coe(outputs, targets)

    References:
        [1] Sudre, Carole H., et al. "Generalised dice overlap as a deep learning loss function for highly unbalanced
        segmentations." Deep learning in medical image analysis and multimodal learning for clinical decision support.
        Springer, Cham, 2017. 240-248.

    """

    assert output.dtype in [tf.float32, tf.float64]
    assert target.dtype in [tf.float32, tf.float64]

    intersection = tf.reduce_sum(output * target, axis=axis[:-1])
    a = tf.reduce_sum(output, axis=axis[:-1])
    b = tf.reduce_sum(target, axis=axis[:-1])

    # define weights for each class
    weights = 1.0 / ((tf.reduce_sum(target, axis=axis[:-1]) ** 2) + smooth)

    # numerator and denominator:
    numerator = tf.reduce_sum(weights * intersection, axis=1)
    denominator = tf.reduce_sum(weights * (a + b), axis=1)

    score = 2. * (numerator + smooth) / (denominator + smooth)
    score = tf.reduce_mean(score, name='dice_coe')
    return score


def jaccard_coe(output, target, axis=(1, 2, 3), smooth=1e-12, _name='jaccard_coe'):
    """ Soft Jaccard (also known as Intersection over Union) coefficient for evaluating the similarity of two batch of
    data. The coefficient can vary between 0 and 1, where 1 means totally match. It is usually used for binary image
    segmentation (e.g. using the loss function: 1 - dice_coe(...)).

    Args:
        output (Tensor): a distribution with shape: [batch_size, ....], (any dimensions). This is the prediction.
        target (Tensor): the target distribution, format the same with `output`.
        axis (tuple of int): contains all the dimensions to be reduced, default ``[1,2,3]``.
        smooth (float): small value added to the numerator and denominator.

    Returns:
        Jaccard coefficient.

    Examples:
        outputs = softmax(network.outputs)
        jaccard_loss = 1.0 - jaccard_coe(outputs, targets)

    References:
        `Wiki-Jaccard <https://en.wikipedia.org/wiki/Jaccard_index>`__

    """

    assert output.dtype in [tf.float32, tf.float64]
    assert target.dtype in [tf.float32, tf.float64]

    intersection = tf.reduce_sum(output * target, axis=axis)

    a = tf.reduce_sum(output * output, axis=axis)
    b = tf.reduce_sum(target * target, axis=axis)

    union = a + b - intersection
    score = (intersection + smooth) / (union + smooth)
    score = tf.reduce_mean(score, name=_name)
    return score


def iou_coe(output, target, axis=(1, 2, 3), smooth=1e-12):
    """
    Wrapper to Jaccard (also known as Intersection over Union) coefficient

    Args:
        output (Tensor): a distribution with shape: [batch_size, ....], (any dimensions). This is the prediction.
        target (Tensor): the target distribution, format the same with `output`.
        axis (tuple of int): contains all the dimensions to be reduced, default ``[1,2,3]``.
        smooth (float): small value added to the numerator and denominator.

    Returns:
        Jaccard coefficient.

    """
    return jaccard_coe(output, target, axis, smooth, _name='iou_coe')


def shannon_binary_entropy(incoming, axis=(1, 2), unscaled=False, smooth=1e-12):
    """
    Evaluates shannon entropy loss on a binary mask. The last index of the incoming tensor must contain the one-hot
    encoded predictions.

    Args:
        incoming (tensor): incoming tensor (one-hot encoded). On the first dimension there is the number of samples
            (typically the batch size)
        axis (tuple of int): axis containing the input dimension. Assuming 'incoming' to be a 4D tensor, axis has length
            2: width and height; if 'incoming' is a 5D tensor, axis should have length of 3, and so on.
        unscaled (Boolean): The computation does the operations using the natural logarithm log(). To obtain the actual
            entropy alue one must scale this value by log(2) since the entropy should be computed in base 2 (hence
            log2(.)). However, one may desire to use this function in a loss function to train a neural net. Then, the
            log(2) is just a multiplicative constant of the gradient and could be omitted for efficiency reasons.
            Turning this flag to True allows for this behaviour to happen (default is False, then the actual entropy).
        smooth (float): small value added to the numerator and denominator.

    Returns:
        Entropy value of the incoming tensor.

    """

    assert incoming.dtype in [tf.float32, tf.float64]

    # compute probability of label l
    p_l = tf.reduce_sum(incoming, axis=axis)
    p_l = tf.clip_by_value(p_l, clip_value_min=smooth, clip_value_max=1-smooth)
    entropy_l = - p_l * tf.log(p_l) - (1 - p_l) * tf.log(1 - p_l)

    if not unscaled:
        entropy_l = tf.log(2.0) * entropy_l

    entropy = tf.reduce_sum(entropy_l, axis=-1)
    mean_entropy = tf.reduce_mean(entropy, axis=0)

    return mean_entropy
