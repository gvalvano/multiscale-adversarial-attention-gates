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


def instance_noise_layer(input_tensor, is_training, mean=0.0, stddev=1.0, truncated=False, scope="InstanceNoiseLayer"):
    """
    Gradient reversal layer. In forward pass, the input tensor gets through unchanged,
    in the backward pass the gradients are propagated with opposite sign.
    :param input_tensor: (tensor) input tensor
    :param is_training: (bool or tf.placeholder)  training mode
    :param mean: (float) mean for the gaussian noise
    :param stddev: (float) standard deviation for the gaussian noise
    :param truncated: (bool) if the noisy output must be thresholded to be positive and smaller then 1 (range [0, 1])
    :param scope: (str) variable scope for the layer
    :return:
    """

    @tf.custom_gradient
    def _truncate(incoming):
        """ Threshold the noisy output to be in the range [0, 1] """
        def grad(g): return g
        with tf.variable_scope("truncate_op"):
            fw_pass = tf.clip_by_value(incoming, 0.0, 1.0)
        return fw_pass, grad

    with tf.variable_scope(scope):
        # add small noise to the input image:
        noise = tf.random_normal(shape=tf.shape(input_tensor), mean=mean, stddev=stddev, dtype=tf.float32)
        forward_pass = tf.cond(is_training, lambda: input_tensor + noise, lambda: input_tensor)

    if truncated:
        forward_pass = _truncate(forward_pass)

    return forward_pass


def label_noise_layer(input_tensor, is_training, prob=0.1, mode='flip_sign', scope="LabelNoiseLayer"):
    """
    Gradient reversal layer. In forward pass, the input tensor gets through unchanged,
    in the backward pass the gradients are propagated with opposite sign.
    :param input_tensor: (tensor) input tensor
    :param is_training: (bool or tf.placeholder) training mode
    :param prob: (float) probability of perturbing prediction
    :param mode: (str) noise modality. Allowed values in ['flip_sign', 'flip_label'] to flip sign or one-hot label in
                    the prediction (i.e. input_tensor)
    :param scope: (str) variable scope for the layer
    :return:
    """

    assert mode in ['flip_sign', 'flip_label']

    @tf.custom_gradient
    def _flip_sign(incoming):
        """ Threshold the noisy output to be in the range [0, 1] """
        def grad(g): return g
        with tf.variable_scope("flip_sign"):
            c = tf.random.uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32)[0]
            fw_pass = tf.cond(tf.greater(c, prob), lambda: incoming, lambda: -1.0 * incoming)
        return fw_pass, grad

    @tf.custom_gradient
    def _flip_label(incoming):
        """Threshold the noisy output to be in the range [0, 1] """
        def grad(g): return g
        with tf.variable_scope("flip_label"):
            c = tf.random.uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32)[0]
            fw_pass = tf.cond(tf.greater(c, prob), lambda: incoming, lambda: tf.abs(incoming - 1.0))
        return fw_pass, grad

    with tf.variable_scope(scope):
        if mode == 'flip_sign':
            forward_pass = _flip_sign(input_tensor)
        elif mode == 'flip_label':
            forward_pass = _flip_label(input_tensor)

    forward_pass = tf.cond(is_training, lambda: forward_pass, lambda: input_tensor)

    return forward_pass
