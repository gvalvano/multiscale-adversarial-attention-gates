"""
Custom layer that rounds inputs during forward pass and copies the gradients during backward pass
"""
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


@tf.custom_gradient
def rounding_layer(incoming, scope="RoundLayer"):
    """
    Rounding layer. In forward pass, it applies a threshold to round the incoming tensor,
    in the backward pass is simply copies the gradients from the output to the input.
    :param incoming: (tensor) incoming tensor
    :param scope: (str) variable scope for the layer
    :return:
    """
    def grad(g):
        return g

    with tf.variable_scope(scope):
        forward_pass = tf.round(incoming)

    return forward_pass, grad
