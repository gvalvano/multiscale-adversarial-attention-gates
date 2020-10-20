"""
Callback for DSD training [Han et al. 2017].
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
from .callbacks import Callback
import idas.logger.json_logger as jlogger
import os


def run_sparse_step(sess, sparsity=0.30):
    """
    Runs the sparse step for the Dense-Sparse-Dense training [1].
    Here weights smaller than the threshold _lambda are set to 0 accordingly to the desired sparsity (i.e. S=30%).
    The _lambda value is chosen to leave the layer weight matrix with the desired sparsity.

    Args:
        sess (tf.Session): TensorFlow session to run the sparse step
        sparsity (float): sparsity percentage (default = 30%)

    References:
        [1] Han, Song, et al. "DSD: Dense-sparse-dense training for deep neural networks." arXiv preprint
        arXiv:1607.04381 (2016).

    """
    print("Threshold elements below chosen value (Sparsity {0}%)".format(sparsity * 100))
    layers = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('kernel:0')]
    for lyr in layers:
        layer = sess.graph.get_tensor_by_name(lyr)
        flat_layer = tf.reshape(layer, [-1])
        N = tf.to_float(tf.shape(flat_layer)[0])
        k = tf.to_int32(N * (1. - tf.constant(sparsity, dtype=tf.float32)))
        values, indices = tf.nn.top_k(tf.abs(flat_layer), k=k)
        _lambda = tf.reduce_min(values)

        # TODO: define outside tf.assign() operation to avoid adding a new node to the graph every time we call it
        sess.run(tf.assign(layer, tf.multiply(layer, tf.to_float(tf.abs(layer) >= _lambda))))


def check_for_sparse_training(cnn, history_logs):
    """
    Checks if the flag for DSD training (that could have been added by a DSDCallback()) exists and it is True.
    Returns True if it is the case, False otherwise.

    Args:
        cnn (tensor): neural network
        history_logs (str): file with the history

    Returns:
        Returns has_been_run=True if the try succeeds, False otherwise.
    """
    has_been_run = False
    try:
        node = jlogger.read_one_node('SPARSE_TRAINING', file_name=history_logs)
        if node['done_before']:
            sparsity = node['sparsity']
            beta = node['beta']

            if not cnn.perform_sparse_training:
                # otherwise the learning rate has already been reduced by dsd_callback
                print(" | This network was already trained with Sparsity constraint of \033[94m{0}%\033[0m and "
                      "a reduced learning rate by a multiplying factor beta \033[94m{1}%\033[0m."
                      .format(sparsity * 100, beta * 100))
                print(" | The current learning rate ({0}) will be consequently reduced by the same factor: \033[94m"
                      "corrected learning rate = {1}\033[0m".format(cnn.lr, beta * cnn.lr))
                print(' | - - ')
                cnn.lr = beta * cnn.lr

            has_been_run = True

    except (FileNotFoundError, KeyError):
        pass
    return has_been_run


class DSDCallback(Callback):
    """ Callback for DSD training [Han et al., DSD: Dense-sparse-dense training for deep neural networks, 2016]. """

    def __init__(self):
        super().__init__()
        # Define variables here because the callback __init__() is called before the initialization of all variables
        # in the graph.
        self.history_log_file = None

    def on_train_begin(self, training_state, **kwargs):

        cnn = kwargs['cnn']
        if cnn is None:
            raise Exception

        try:
            sparsity = kwargs['sparsity']
        except KeyError:
            sparsity = 0.30
        try:
            beta = kwargs['beta']
        except KeyError:
            beta = 0.10

        self.history_log_file = kwargs['history_log_dir'] + os.sep + 'train_history.json'

        vals = {'done_before': True, 'sparsity': sparsity, 'beta': beta}
        jlogger.add_new_node('SPARSE_TRAINING', vals, file_name=self.history_log_file)

        # set learning rate to 1/10 (beta) of its initial value
        cnn.lr = beta * cnn.lr
        print("\nRunning training with sparse constrain: SPARSITY = \033[94m{0}%\033[0m.".format(sparsity * 100))
        print("Reducing original learning rate to \033[94m{0}%\033[0m of its value.".format(beta * 100))

    def on_epoch_begin(self, training_state, **kwargs):

        sess = kwargs['sess']
        try:
            sparsity = kwargs['sparsity']
        except KeyError:
            sparsity = 0.30

        run_sparse_step(sess, sparsity=sparsity)
