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
from tensorflow import layers
from .layers.spectral_norm import spectral_norm_conv2d
# from .layers.gradient_reversal_layer import gradient_reversal_layer
from .layers.noise_operations import instance_noise_layer
from .layers.noise_operations import label_noise_layer

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class MultiResDiscriminator(object):

    def __init__(self, is_training, n_filters=64, n_blocks=4, instance_noise=False,
                 out_mode='scalar', name='MultiResDiscriminator'):
        """
        Class for building the mask discriminator
        :param is_training: (tf.placeholder(dtype=tf.bool) or bool) variable to define training or test mode; it is
                        needed for the behaviour of dropout (which is different at train or test time)
        :param n_filters: (int) number of filters at the first convolutional layer
        :param n_blocks: (int) number of down-sample blocks
        :param instance_noise: (bool) whether to apply instance noise or not.
        :param out_mode: (str) output mode: valid entries are ['scalar', 'prob_map']. Defaults to 'scalar'.
                        scalar --> outputs a scalar value as in vanilla GANs
                        prob_map --> outputs a probability map of values (PatchGAN)
        :param name: (str) name for the variable scope

        - - - - - - - - - - - - - - - -
        Notice that:
          - output is linear (this is meant to be used as LeastSquare-GAN)
        - - - - - - - - - - - - - - - -

        Examples:

            '''python
            # 1) instantiate and build the segmentor:
            unet = SAUNet(*unet_args)
            unet = unet.build(input_data_tensor)

            # 2) get high resolution predictions:
            pred_mask_soft = unet.get_prediction(softmax=True)
            pred_mask_oh = unet.get_prediction(one_hot=True)

            # 3) get predicted segmentation at every resolution level of the segmentor
            multiscale_predictions = [sup_pred_mask_soft[..., 1:]] + [el[..., 1:] for el in unet.attention_tensors]

            # 4) feed to the discriminator for adversarial conditioning
            discriminator = MultiResDiscriminator(*disc_args)

            real_input_tensors = [ground_truth_mask[..., 1:]]
            model_real = discriminator.build(real_input_tensors)
            disc_pred_real = model_real.get_prediction()

            model_fake = discriminator.build(multiscale_predictions, reuse=True)
            disc_pred_fake = model_fake.get_prediction()
            '''

        """
        assert out_mode in ['scalar', 'prob_map']
        self.out_mode = out_mode

        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.is_training = is_training
        self.name = name
        self.instance_noise = instance_noise

        self.prediction = None

    def _crate_input_dictionary(self, tensor_list):
        """
        Create dictionary with input tensors for each level.
        :param tensor_list: (list) list of input tensors (for each block)
        """
        in_shape = tensor_list[0].get_shape()
        in_dict = dict()
        for i in range(0, self.n_blocks):

            if len(tensor_list) == self.n_blocks:
                in_dict['lvl_{0}'.format(i)] = tensor_list[i]
            else:
                in_dict['lvl_{0}'.format(i)] = tf.image.resize_bilinear(
                    tensor_list[0], size=[in_shape[1] // (2 ** i), in_shape[2] // (2 ** i)], align_corners=True)
        return in_dict

    def build(self, incoming, reuse=tf.AUTO_REUSE):
        """
        Build the model.
        :param incoming: (list) list of input tensors (for each block)
        :param reuse: (bool) if True, reuse trained weights
        """

        assert isinstance(incoming, list)
        assert len(incoming) == self.n_blocks or len(incoming) == 1

        in_dict = self._crate_input_dictionary(tensor_list=incoming)

        with tf.variable_scope(self.name, reuse=reuse):

            input_0 = in_dict['lvl_0']  # gradient_reversal_layer(in_dict['lvl_0'])
            input_0 = instance_noise_layer(input_0, self.is_training, mean=0.0, stddev=0.2) \
                if self.instance_noise else input_0

            layer = layers.conv2d(input_0, filters=self.n_filters, kernel_size=4, strides=2, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            layer = tf.nn.leaky_relu(layer, alpha=0.2)

            for i in range(1, self.n_blocks):
                squeezed_layer = layers.conv2d(layer, filters=13, kernel_size=1, strides=1, activation=tf.nn.sigmoid)
                input_i = in_dict['lvl_{0}'.format(i)]  # gradient_reversal_layer(in_dict['lvl_{0}'.format(i)])
                layer_and_concat = tf.concat([squeezed_layer, input_i], axis=-1)
                layer = self._conv_and_maybe_downsample_block(layer_and_concat,
                                                              n_filters=self.n_filters * 2 * (2 ** i),
                                                              stride=1 if i == self.n_blocks - 1 else 2,
                                                              scope='sn_conv_{0}'.format(str(i)))
            # output layer with final activation linear
            self.prediction = self._output_layer(layer, mode=self.out_mode)

            if self.instance_noise:
                self.prediction = label_noise_layer(self.prediction, self.is_training, prob=0.1, mode='flip_sign')
        return self

    @staticmethod
    def _conv_and_maybe_downsample_block(incoming, n_filters, stride, scope):
        """
        Applies a spectral-norm convolutional layer using given stride. The output activation is a leaky relu.
        :param incoming: incoming tensor
        :param n_filters: number of filters for the convolutional layer
        :param stride: (int) stride to be used for the convolution. Typical value is stride > 1 (i.e. = 2).
        :param scope: variable scope
        :return: leaky_relu activation of the spectral norm convolutional layer
        """
        with tf.variable_scope(scope):
            n_norm = spectral_norm_conv2d(incoming, filters=n_filters, kernel_size=4, stride=stride, padding='same')
            n_norm_act = tf.nn.tanh(n_norm)
        return n_norm_act

    @staticmethod
    def _output_layer(incoming, mode):
        """
        Output layer for the discriminator
        :param incoming: incoming tensor
        :param mode: (str) output mode: valid entries are ['scalar', 'prob_map']
        :return: prediction with activation linear
        """
        if mode == 'prob_map':
            # output a 2D probability map:
            prediction = spectral_norm_conv2d(incoming, filters=1, kernel_size=4, stride=1,
                                              padding='valid', scope='sn_conv_out')
        elif mode == 'scalar':
            # output a scalar value:
            _, w, h, _ = incoming.get_shape()
            prediction = layers.conv2d(incoming, filters=1, kernel_size=(w, h), strides=1, padding='valid',
                                       kernel_initializer=tf.glorot_uniform_initializer(),
                                       bias_initializer=tf.constant_initializer(0.0))
            prediction = tf.layers.flatten(prediction)
        else:
            raise ValueError

        # final activation: linear
        return prediction

    def get_prediction(self):
        """ Get discriminator prediction. """
        return self.prediction
