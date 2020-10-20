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


def film_layer(incoming, gamma, beta, name='film'):
    """
    FiLM layer. For details refer to [1].

    Args:
        incoming (tensor): input tensor
        gamma (tensor): gamma factor (typically predicted by another network)
        beta (tensor): beta factor (typically predicted by another network)
        name (string): variable scope (optional)

    Returns:
        A Tensor which is the output of the layer.

    References:
        [1] Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A.C., 2018. FiLM: Visual reasoning with
        a general conditioning layer, in: AAAI, AAAI Press. pp. 3942–3951.

    """
    with tf.name_scope(name):
        # get shape of incoming tensors:
        in_shape = tf.shape(incoming)
        gamma_shape = tf.shape(gamma)
        beta_shape = tf.shape(beta)

        # tile gamma and beta:
        gamma = tf.tile(tf.reshape(gamma, (gamma_shape[0], 1, 1, gamma_shape[-1])),
                        (1, in_shape[1], in_shape[2], 1))
        beta = tf.tile(tf.reshape(beta, (beta_shape[0], 1, 1, beta_shape[-1])),
                       (1, in_shape[1], in_shape[2], 1))

        # compute output:
        output = incoming * gamma + beta

    return output
