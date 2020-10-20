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

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class MLP(object):

    def __init__(self, incoming, n_in, n_hidden, n_out, is_training, name='MLP'):
        """
        Class for 3-layered MultiLayer Perceptron (MLP). It uses Batch Normalization by default.

        Args:
            incoming (tensor): incoming tensor
            n_in (int): number of input units
            n_hidden (int): number of hidden units
            n_out (int): number of output units
            is_training (tf.placeholder(dtype=tf.bool) or bool): variable to define training or test mode; it is
                        needed for the behaviour of dropout (which behaves differently at train and test time)
            name (string): variable scope (optional)

        Examples:

            is_training = True
            n_classes = 2

            mlp = MLP(x, 128, 256, n_classes, is_training=training_mode)
            mlp = MLP.build()
            prediction = mlp.get_prediction()

        """

        self.incoming = incoming
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.is_training = is_training
        self.name = name
        self.prediction = None

    def build(self):
        """
        Build the model.
        """
        with tf.variable_scope(self.name):
            incoming = layers.flatten(self.incoming)

            input_layer = layers.dense(incoming, self.n_in, kernel_initializer=he_init, bias_initializer=b_init)
            input_layer = tf.layers.batch_normalization(input_layer, training=self.is_training)
            input_layer = tf.nn.relu(input_layer)

            hidden_layer = layers.dense(input_layer, self.n_hidden, kernel_initializer=he_init, bias_initializer=b_init)
            hidden_layer = tf.layers.batch_normalization(hidden_layer, training=self.is_training)
            hidden_layer = tf.nn.relu(hidden_layer)

            output_layer = layers.dense(hidden_layer, self.n_out, bias_initializer=b_init)
            output_layer = tf.layers.batch_normalization(output_layer, training=self.is_training)
            output_layer = tf.nn.relu(output_layer)

            # final activation: linear
            self.prediction = output_layer

        return self

    def get_prediction(self):
        return self.get_prediction()
