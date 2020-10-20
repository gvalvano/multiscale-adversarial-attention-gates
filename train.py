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
tf.random.set_random_seed(1234)
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
tf.logging.set_verbosity(tf.logging.ERROR)
import time
import config
import importlib
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


config.define_flags()
# noinspection PyUnresolvedReferences
FLAGS = tf.app.flags.FLAGS


def parse_model_type():
    """ Import the correct model for the experiments """
    experiment = FLAGS.experiment
    dataset_name = FLAGS.dataset_name
    model = importlib.import_module('experiments.{0}.{1}'.format(dataset_name, experiment)).Model()
    return model


def main(_):
    # import the correct model for the experiment
    model = parse_model_type()
    model.build()

    start_time = time.time()
    model.train(n_epochs=FLAGS.n_epochs)
    delta_t = time.time() - start_time

    print('\nTook: {0:.3f} hours'.format(delta_t/3600))


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
