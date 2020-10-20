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
#  See the License for the specific langcduage governing permissions and
#  limitations under the License.

import tensorflow as tf


def define_flags():

    tf.flags.DEFINE_string('RUN_ID', None, "Unique identifier for the experiment")
    tf.flags.mark_flag_as_required('RUN_ID')

    # ____________________________________________________ #
    # ====================== MODEL ======================= #

    tf.flags.DEFINE_string('experiment', None, """ Experiment to run. """)
    tf.flags.mark_flag_as_required('experiment')

    # ____________________________________________________ #
    # ========== ARCHITECTURE HYPER-PARAMETERS ========== #

    # Learning rate:
    tf.flags.DEFINE_float('lr', 1e-4, 'Learning rate')

    # Batch size
    tf.flags.DEFINE_integer('b_size', 12, "Batch size")

    # Number of epochs
    tf.flags.DEFINE_integer('n_epochs', None, "Number of training epochs")

    # ____________________________________________________ #
    # =============== TRAINING STRATEGY ================== #

    tf.flags.DEFINE_bool('augment', True, "Perform data augmentation")
    tf.flags.DEFINE_bool('standardize', False, "Perform data standardization (z-score)")  # data already pre-processed
    # (others, such as learning rate decay params...)

    # ____________________________________________________________ #
    # =============== LOGS AND REPORTS SETTINGS ================== #

    # global
    tf.flags.DEFINE_bool('verbose', True, "Verbosity, for print reports.")

    # tensorboard
    tf.flags.DEFINE_bool('tensorboard_on', True, "if True: save tensorboard logs")
    tf.flags.DEFINE_integer('skip_step', 3000, "frequency of printing batch report")
    tf.flags.DEFINE_integer('train_summaries_skip', 100, "number of skips before writing summaries for training steps "
                                                         "(used to reduce its verbosity; put 1 to avoid this)")
    tf.flags.DEFINE_bool('tensorboard_verbose', True, "if True: save also layers weights every N epochs")

    # ____________________________________________________ #
    # ==================== HARDWARE ====================== #

    # internal variables:
    tf.flags.DEFINE_integer('num_threads', 20, "number of threads for loading data")
    tf.flags.DEFINE_integer('CUDA_VISIBLE_DEVICE', 0, "visible gpu")

    # ____________________________________________________ #
    # ===================== DATA SET ====================== #

    # path for the data set:
    tf.flags.DEFINE_string('dataset_name', None, """ Dataset name. """)
    tf.flags.DEFINE_string('data_path', None, """ Path of data files. """)
    tf.flags.mark_flag_as_required('dataset_name')
    tf.flags.mark_flag_as_required('data_path')
    tf.flags.DEFINE_string('results_dir', '.', help="results directory")

    # ids for the data
    tf.flags.DEFINE_string('n_sup_vols', None, """ Number of labelled data to use as training volumes (e.g. 'perc25')""")
    tf.flags.DEFINE_string('split_number', None, """ Split number for cross-validation (e.g. 'split0') """)
    tf.flags.mark_flag_as_required('n_sup_vols')
    tf.flags.mark_flag_as_required('split_number')

    return
