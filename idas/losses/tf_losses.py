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
from idas.metrics.tf_metrics import dice_coe, generalized_dice_coe, shannon_binary_entropy, jaccard_coe
from idas.models.hyperbolic.hyp_ops import tf_poinc_dist_sq


def dice_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Returns Soft Sørensen–Dice loss """
    return 1.0 - dice_coe(output, target, axis=axis, smooth=smooth)


def generalized_dice_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Returns the Generalized Soft Sørensen–Dice loss """
    return 1.0 - generalized_dice_coe(output, target, axis=axis, smooth=smooth)


def jaccard_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Returns Soft Jaccard loss """
    return 1.0 - jaccard_coe(output, target, axis=axis, smooth=smooth)


def iou_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Wrapper to Soft Jaccard (also known as Intersection over Union) loss """
    return 1.0 - jaccard_coe(output, target, axis=axis, smooth=smooth, _name='iou_coe')


def weighted_softmax_cross_entropy(y_pred, y_true, num_classes, eps=1e-12):
    """
    Define weighted cross-entropy function for classification tasks. Applies softmax on y_pred.
    :param y_pred: tensor [None, width, height, n_classes]
    :param y_true: tensor [None, width, height, n_classes]
    :param eps: (float) small value to avoid division by zero
    :param num_classes: (int) number of classes
    :return:
    """

    n = [tf.reduce_sum(tf.cast(y_true[..., c], tf.float32)) for c in range(num_classes)]
    n_tot = tf.reduce_sum(n)

    weights = [n_tot / (n[c] + eps) for c in range(num_classes)]

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))
    softmax = tf.nn.softmax(y_pred)

    w_cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(softmax + eps), weights), reduction_indices=[1])
    loss = tf.reduce_mean(w_cross_entropy, name='weighted_softmax_cross_entropy')
    return loss


def weighted_cross_entropy(y_pred, y_true, num_classes, weights=None, eps=1e-12):
    """
    Define weighted cross-entropy function for classification tasks. Assuming y_pred already probabilistic.
    :param y_pred: tensor [None, width, height, n_classes]
    :param y_true: tensor [None, width, height, n_classes]
    :param eps: (float) small value to avoid division by zero
    :param weights: (list) if None, compute weights dynamically, to balance across classes
    :param num_classes: (int) number of classes
    :return:
    """

    n = [tf.reduce_sum(tf.cast(y_true[..., c], tf.float32)) for c in range(num_classes)]
    n_tot = tf.reduce_sum(n)

    if weights is None:
        weights = [n_tot / (n[c] + eps) for c in range(num_classes)]

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))

    w_cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred + eps), weights), reduction_indices=[1])
    loss = tf.reduce_mean(w_cross_entropy, name='weighted_cross_entropy')
    return loss


def cross_entropy(y_pred, y_true, num_classes, eps=1e-12):
    """
    Define weighted cross-entropy function for classification tasks. Assuming y_pred already probabilistic.
    :param y_pred: tensor [None, width, height, n_classes]
    :param y_true: tensor [None, width, height, n_classes]
    :param eps: (float) small value to avoid division by zero
    :param num_classes: (int) number of classes
    :return:
    """

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))

    w_cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred + eps), reduction_indices=[1])
    loss = tf.reduce_mean(w_cross_entropy, name='cross_entropy')
    return loss


def shannon_binary_entropy_loss(incoming, axis=(1, 2), unscaled=True, smooth=1e-12):
    """
    Evaluates shannon entropy on a binary mask. The last index contains one-hot encoded predictions.
    :param incoming: incoming tensor (one-hot encoded). On the first dimension there is the number of samples (typically
                the batch size)
    :param axis: axis containing the input dimension. Assuming 'incoming' to be a 4D tensor, axis has length 2: width
                and height; if 'incoming' is a 5D tensor, axis should have length of 3, and so on.
    :param unscaled: The computation does the operations using the natural logarithm log(). To obtain the actual entropy
                value one must scale this value by log(2) since the entropy should be computed in base 2 (hence log2()).
                However, one may desire using this function in a loss function to train a neural net. Then, the log(2)
                is just a multiplicative constant of the gradient and could be omitted for efficiency reasons. Turning
                this flag to False allows for exact actual entropy evaluation; default behaviour is True.
    :param smooth: This small value will be added to the numerator and denominator.
    :return:
    """
    return shannon_binary_entropy(incoming, axis=axis, unscaled=unscaled, smooth=smooth)


def vae_loss(z_mean, z_logstd, y_true, y_pred, n_outputs):
    """
    Computes classical loss for variational auto-encoders, as:
        Loss = Reconstruction_loss + KL_divergence_loss
             = MSE - ELBO

    Args:
        z_mean (tensor): encoded mean in the VAE latent vector.
        z_logstd (tensor): encoded logarithm of the standard deviation in the VAE latent vector.
        y_true (tensor): true input sample to the VAE
        y_pred (tensor): reconstruction of the correspondent sampled sample in the VAE latent code.
        n_outputs (int): number of outputs (i.e. in mnist we have 28x28 = 784)

    Returns:
        VAE loss computed as MSE on the reconstruction - ELBO

    """
    x_true = tf.reshape(y_true, shape=(-1, n_outputs))
    x_reconstructed = tf.reshape(y_pred, shape=(-1, n_outputs))

    # _______
    # Reconstruction loss:
    with tf.variable_scope('Reconstruction_loss'):
        generator_loss = tf.losses.mean_squared_error(x_reconstructed, x_true)

    # _______
    # KL Divergence loss:
    with tf.variable_scope('KL_divergence_loss'):
        kl_div_loss = 1.0 + z_logstd - tf.square(z_mean) - tf.exp(z_logstd)
        kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

    total_loss = tf.reduce_mean(generator_loss + kl_div_loss)

    return total_loss, generator_loss, kl_div_loss


def sobel_gradient_loss(y_pred, y_true, shape):
    """
    Computes gradient loss as the mean square error on the image filtered with Sobel filters. In particular, it computes
    the gradients of both the predicted and the target image and computes the MSE between the two.

    Args:
        y_pred (tensor): predicted image (assumed to have continuous values).
        y_true (tensor): true image (assumed to have continuous values).
        shape (tuple of int): width and height of the image.

    Returns:
        MSE between the gradient of the predicted and true image.

    """
    assert len(shape) == 2

    pred_grad = tf.reduce_sum(tf.image.sobel_edges(y_pred[:, :, :, 0]), -1)
    input_grad = tf.reduce_sum(tf.image.sobel_edges(y_true[:, :, :, 0]), -1)

    x_reconstructed_grad = tf.reshape(pred_grad, shape=(-1, shape[0] * shape[1]))
    x_true_grad = tf.reshape(input_grad, shape=(-1, shape[0] * shape[1]))

    grad_loss = tf.reduce_mean(tf.squared_difference(x_reconstructed_grad, x_true_grad), 1)
    return grad_loss


def contrastive_loss(y_pred, y_true, num_classes, margin=1.0):
    """ Euclidian distance between the two sets of tensors """

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.reshape(y_true, (-1, num_classes))

    # This would be the average distance between classes in Euclidian space
    distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_pred - y_true), axis=1))

    # label here is {0,1} for neg, pos. pairs in the contrastive loss
    loss = tf.reduce_mean(tf.cast(y_true, distances.dtype) * tf.square(distances) +
                          (1. - tf.cast(y_true, distances.dtype)) *
                          tf.square(tf.maximum(margin - distances, 0.)),
                          name='contrastive_loss')
    return loss


def hyperbolic_contrastive_loss(y_pred, y_true, num_classes, margin=1.0, radius=1.0):
    """ Hyperbolic distance between the two sets of tensors """

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.reshape(y_true, (-1, num_classes))

    # This would be the average distance between classes on the Poincaré disk:
    distances = tf_poinc_dist_sq(y_pred, y_true, c=radius)

    # label here is {0,1} for neg, pos. pairs in the contrastive loss
    loss = tf.reduce_mean(tf.cast(y_true, distances.dtype) * tf.square(distances) +
                          (1. - tf.cast(y_true, distances.dtype)) *
                          tf.square(tf.maximum(margin - distances, 0.)),
                          name='hyperbolic_contrastive_loss')
    return loss
