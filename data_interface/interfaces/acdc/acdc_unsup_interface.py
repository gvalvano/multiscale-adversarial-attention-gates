"""
Automatic Cardiac Diagnostic Challenge 2017 database. In total there are images of 100 patients, for which manual
segmentations of the heart cavity, myocardium and right ventricle are provided.
Database at: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
Atlas of the heart in each projection at: http://tuttops.altervista.org/ecocardiografia_base.html
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
import numpy as np
import os
from glob import glob
import cv2
from idas.utils.utils import print_yellow_text, get_available_gpus


class DatasetInterface(object):

    def __init__(self, root_dir, data_ids, input_size, verbose=True):
        """
        Interface to the data set
        :param root_dir: (string) path to directory containing training data
        :param data_ids: (dict) dictionary with train, validation, test volume ids
        :param input_size: (int, int) input size for the neural network. It should be the same across all data sets
        :param verbose: (bool) verbosity level
        """
        self.dataset_name = 'acdc'
        self.input_size = input_size
        self.n_channels_in = 1
        self.verbose = verbose

        path_dict = dict()
        for d_set in ['train_unsup', 'validation']:
            path_list = []

            suffix = '*/' if root_dir.endswith('/') else '/*/'
            subdir_list = [d[:-1] for d in glob(root_dir + suffix)]

            for subdir in subdir_list:
                folder_name = subdir.rsplit('/')[-1]
                if folder_name.startswith('patient'):

                    curr_list = data_ids[d_set] if isinstance(data_ids, dict) else \
                        data_ids if isinstance(data_ids, list) else None

                    if int(folder_name.rsplit('patient')[-1]) in curr_list:
                        prefix = os.path.join(root_dir, folder_name)
                        pt_number = folder_name.split('patient')[1]
                        pt_full_path = os.path.join(prefix, 'patient' + pt_number + '_unsup')
                        path_list.append(pt_full_path)

            path_dict[d_set] = path_list

        self.x_train_paths = path_dict['train_unsup']
        self.x_validation_paths = path_dict['validation']

        assert len(self.x_train_paths) > 0
        assert len(self.x_validation_paths) > 0

    @staticmethod
    def _py_data_augmentation_ops(x_batch):

        n_samples, rows, cols, channels = x_batch.shape
        center = (cols // 2, rows // 2)  # open cv requires swapped rows and cols

        # create and apply transformation matrix for each element
        x_batch_augmented = []
        for i in range(n_samples):
            curr_slice = x_batch[i, ..., 0]

            # sample transformation parameters for the current element of the batch
            minval = - 0.1 * x_batch[i, ..., 0].shape[0]
            maxval = 0.1 * x_batch[i, ..., 0].shape[0]
            tx, ty = np.random.randint(low=minval, high=maxval, size=2)
            scale = np.random.uniform(low=0.98, high=1.02)
            angle = np.random.uniform(low=-90.0, high=90.0)

            # transformation matrices:
            m1 = np.float32([[1, 0, tx], [0, 1, ty]])
            m2 = cv2.getRotationMatrix2D(center, angle, scale)

            # apply transformation
            transform_slice = cv2.warpAffine(curr_slice, m1, (cols, rows))
            transform_slice = cv2.warpAffine(transform_slice, m2, (cols, rows))

            x_batch_augmented.append(np.expand_dims(transform_slice, axis=-1))

        return np.array(x_batch_augmented)

    @staticmethod
    def _data_augmentation_ops(x_train):
        """ Data augmentation pipeline (to be applied on training samples)
        """
        # # image distortions:
        # x_train = tf.image.random_brightness(x_train, max_delta=0.025)
        # x_train = tf.image.random_contrast(x_train, lower=0.95, upper=1.05)

        return tf.cast(x_train, tf.float32)

    def data_parser(self, filename, standardize=False, augment=True):
        """
        Given a subject, returns the sequence of frames for a random z coordinate
        :param filename: (str) path to the patient mri sequence
        :param standardize: (bool) if True, standardize input data
        :param augment: (bool) if True, perform data augmentation operations
        :return: (array) = numpy array with the frames on the first dimension, s.t.: [None, width, height]
        """
        fname = filename.decode('utf-8') + '_img.npy'
        batch = np.load(fname).astype(np.float32)

        if standardize and self.verbose:
            print("Data won't be standardized, as they already have been pre-processed.")
        if augment:
            batch = self._py_data_augmentation_ops(batch).astype(np.float32)
        assert not np.any(np.isnan(batch))

        return batch

    def get_data(self, b_size, augment=False, standardize=False, repeat=False, num_threads=4, seed=None):
        """ Returns iterators on the dataset along with their initializers.
        :param b_size: batch size
        :param augment: if to perform data augmentation
        :param standardize: if to standardize the input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :param num_threads: for parallel computing
        :return: train_init, valid_init, input_data, label
        """
        with tf.name_scope('{0}_data'.format(self.dataset_name)):

            _train_data = tf.constant(self.x_train_paths)
            _valid_data = tf.constant(self.x_validation_paths)
            train_data = tf.data.Dataset.from_tensor_slices(_train_data)
            valid_data = tf.data.Dataset.from_tensor_slices(_valid_data)

            train_data = train_data.shuffle(buffer_size=len(self.x_train_paths), seed=seed)

            train_data = train_data.map(
                lambda filename: tf.py_func(  # Parse the record into tensors
                    self.data_parser,
                    [filename, standardize, augment],
                    [tf.float32]), num_parallel_calls=num_threads)

            valid_data = valid_data.map(
                lambda filename: tf.py_func(  # Parse the record into tensors
                    self.data_parser,
                    [filename, standardize, False],
                    [tf.float32]), num_parallel_calls=num_threads)

            # - - - - - - - - - - - - - - - - - - - -

            if augment:
                train_data = train_data.map(lambda x: self._data_augmentation_ops(x), num_parallel_calls=num_threads)
                valid_data = valid_data.map(lambda v: tf.cast(v, dtype=tf.float32), num_parallel_calls=num_threads)

            seed2 = seed + 1
            train_data = train_data.shuffle(buffer_size=len(self.x_train_paths), seed=seed2)

            if repeat:
                if self.verbose:
                    print_yellow_text(' --> Repeat the input indefinitely  = True', sep=False)
                train_data = train_data.repeat()  # Repeat the input indefinitely

            # un-batch first, then batch the data
            train_data = train_data.apply(tf.data.experimental.unbatch())
            valid_data = valid_data.apply(tf.data.experimental.unbatch())

            train_data = train_data.batch(b_size, drop_remainder=True)
            valid_data = valid_data.batch(b_size, drop_remainder=True)

            # if len(get_available_gpus()) > 0:
            #     # prefetch data to the GPU
            #     # train_data = train_data.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
            #     train_data = train_data.apply(tf.data.experimental.copy_to_device("/gpu:0")).prefetch(1)

            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

            _input_data = iterator.get_next()
            train_init = iterator.make_initializer(train_data)  # initializer for train_data
            valid_init = iterator.make_initializer(valid_data)  # initializer for test_data

            with tf.name_scope('unsup_data'):
                input_data = tf.reshape(_input_data,
                                        shape=[-1, self.input_size[0], self.input_size[1], self.n_channels_in])

            return train_init, valid_init, input_data
