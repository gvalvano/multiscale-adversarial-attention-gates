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
from models._internal._sdnet.unet import UNet
from models._internal._sdnet.modality_encoder import ModalityEncoder
from models._internal._sdnet.decoder import Decoder
from models._internal._sdnet.segmentor import Segmentor
from models._internal._sdnet.layers.rounding_layer import rounding_layer

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class SDNet(object):

    def __init__(self, n_anatomical_masks, nz_latent, n_classes, is_training, use_segmentor=True, anatomy=None,
                 name='Model'):
        """
        Implementation of SDNet architecture. For details, refer to:
          "Factorised Representation Learning in Cardiac Image Analysis" (2019), arXiv preprint arXiv:1903.09467
          Chartsias, A., Joyce, T., Papanastasiou, G., Williams, M., Newby, D., Dharmakumar, R., & Tsaftaris, S. A.

        Notice that this implementation does not contain the mask discriminator. The mask discriminator architecture is
        in idas.models._internal._sdnet. You can easily add it following the example at:
            https://github.com/gvalvano/sdnet/blob/b56339534ccbe95261eb2b1642e80aa52d644201/model.py#L199

        :param n_anatomical_masks: (int) number of anatomical masks (s factors)
        :param nz_latent: (int) number of latent dimensions as output of the modality encoder
        :param n_classes: (int) number of classes (4: background, LV, RV, MC)
        :param is_training: (tf.placeholder, or bool) training state, for batch normalization
        :param use_segmentor: (bool) if True, use a separate network to obtain the final mask. If False, the model uses
                        the first (n_classes - 1) anatomical masks as the segmentation channels. Defaults to True (as in
                        the original paper).
        :param anatomy: (tensor) if given, the reconstruction is computed starting from z modality extracted by the
                        input data and the hard anatomy given as argument. Default: compute and use hard anatomy of the
                        input data.
        :param name: variable scope for the model

        - - - - - - - - - - - - - - - -

        Example of usage:

            # build the sdnet:
            sdnet = SDNet(n_anatomical_masks, nz_latent, n_classes, is_training, use_segmentor=False, name='Model')
            sdnet = sdnet.build(input_data)

            # get soft and hard anatomy:
            soft_a = sdnet.get_soft_anatomy()
            hard_a = sdnet.get_hard_anatomy()

            # get z distribution (output of modality encoder)
            z_mean, z_logvar, sampled_z = sdnet.get_z_distribution()

            # get decoder reconstruction:
            rec = sdnet.get_input_reconstruction()

            # get z estimate given the reconstruction
            z_regress = sdnet.get_z_sample_estimate()

        """
        self.is_training = is_training
        self.name = name

        self.n_anatomical_masks = n_anatomical_masks
        self.nz_latent = nz_latent
        self.n_classes = n_classes
        self.anatomy = anatomy

        self.use_segmentor = use_segmentor

        self.soft_anatomy = None
        self.hard_anatomy = None
        self.z_mean = None
        self.z_logvar = None
        self.sampled_z = None
        self.reconstruction = None
        self.z_regress = None
        self.pred_mask = None
        self.input_segmentor = None

    def build(self, input_image, reuse=tf.AUTO_REUSE):
        """
        Build the model.
        """
        with tf.variable_scope(self.name, reuse=reuse):
            # - - - - - - -
            # build Anatomy Encoder
            self.soft_anatomy, self.hard_anatomy, _ = self.build_anatomy_encoder(input_image)

            # - - - - - - -
            # build Modality Encoder
            self.z_mean, self.z_logvar, self.sampled_z, mod_encoder = self.build_modality_encoder(input_image, self.hard_anatomy)

            # - - - - - - -
            # build Decoder to reconstruct the input given sampled z and the hard anatomy
            self.reconstruction, _ = self.build_decoder(self.sampled_z, self.hard_anatomy)

            # - - - - - - -
            # estimate back z_sample from the reconstructed image (only anatomy may be changed, no modality factors)
            self.z_regress, _ = self.build_z_regressor(self.reconstruction, mod_encoder)

            # - - - - - - -
            # build Segmentor
            if self.use_segmentor:
                self.input_segmentor = self.hard_anatomy
                self.pred_mask = self.build_segmentor(self.input_segmentor, n_classes=self.n_classes)
            else:
                non_heart_channels = self.soft_anatomy[..., self.n_classes - 1:]
                non_hart = tf.reduce_sum(non_heart_channels, axis=-1, keepdims=True)
                heart = self.soft_anatomy[..., :self.n_classes - 1]

                pred_mask = tf.concat([non_hart, heart], axis=-1)
                self.pred_mask = pred_mask

        return self

    def build_anatomy_encoder(self, input_image):
        with tf.variable_scope('AnatomyEncoder'):
            unet = UNet(input_image, n_out=self.n_anatomical_masks, is_training=self.is_training, n_filters=64)
            unet_encoder = unet.build_encoder()
            unet_bottleneck = unet.build_bottleneck(unet_encoder)
            unet_decoder = unet.build_decoder(unet_bottleneck)
            coarse_output = unet.build_output(unet_decoder)

            # apply softmax to scale the coarse_output channels in the range [0÷1]. This will avoid the same anatomy to
            # be encoded twice from the model (one anatomy per channel).
            soft_anatomy = tf.nn.softmax(coarse_output)

        with tf.variable_scope('RoundingLayer'):
            hard_anatomy = rounding_layer(soft_anatomy)

        return soft_anatomy, hard_anatomy, unet

    def build_modality_encoder(self, input_image, hard_anatomy):
        with tf.variable_scope('ModalityEncoder'):
            mod_encoder = ModalityEncoder(input_image, hard_anatomy, self.nz_latent, self.is_training).build()
            z_mean, z_logvar = mod_encoder.get_z_stats()
            sampled_z = mod_encoder.get_z_sample()

        return z_mean, z_logvar, sampled_z, mod_encoder

    def build_decoder(self, sampled_z, hard_anatomy):
        with tf.variable_scope('Decoder'):
            decoder = Decoder(sampled_z, hard_anatomy, self.n_anatomical_masks, is_training=self.is_training).build()
            reconstruction = decoder.get_reconstruction()

        return reconstruction, decoder

    @staticmethod
    def build_z_regressor(reconstruction, mod_encoder):
        with tf.variable_scope('ModalityEncoder'):
            z_regress = mod_encoder.estimate_z(reconstruction, reuse=True)
        return z_regress, mod_encoder

    def build_segmentor(self, hard_anatomy, n_classes):
        with tf.variable_scope('Segmentor'):
            segmentor = Segmentor(hard_anatomy, n_classes=n_classes, is_training=self.is_training).build()
            pred_mask = segmentor.get_output_mask()
        return pred_mask

    def get_soft_anatomy(self):
        return self.soft_anatomy

    def get_hard_anatomy(self):
        return self.hard_anatomy

    def get_z_distribution(self):
        return self.z_mean, self.z_logvar, self.sampled_z

    def get_z_sample_estimate(self):
        return self.z_regress

    def get_input_reconstruction(self):
        return self.reconstruction

    def get_input_segmentor(self):
        if not self.use_segmentor:
            raise Exception("No Segmentor registered for this SDNet (see flag 'use_segmentor' in the class __init__()).")
        return self.input_segmentor

    def get_pred_mask(self, one_hot, output=None):
        """
        Get predicted mask.
        Notice that the output is not necessarily one-hot encoded nor in the range [0, 1]. If you want a segmentation
        mask you should use the one-hot flag.
        :param one_hot: (bool) if True, returns one-hot segmentation mask
        :param output: (str) optional, if one-hot is False then output must be either 'linear' or 'softmax'
        :return:
        """
        if not one_hot:
            assert output in ['linear', 'softmax']
            if output == 'linear':
                return self.pred_mask
            else:
                return tf.nn.softmax(self.pred_mask)
        else:
            return tf.one_hot(tf.argmax(self.pred_mask, axis=-1), self.n_classes)
