"""
Utilities for hdf5 data
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

import h5py
import os
import logging


def create_hdf5_db(x_train, y_train, x_validation, y_validation, x_test, y_test, db_name='data.h5'):
    """
    Creates hdf5 database containing nodes for x_train, x_validation, x_test, y_train, y_validation, y_test.

    Args:
        x_train: Provide data to initialize the dataset.
        y_train: Provide data to initialize the dataset.
        x_validation: Provide data to initialize the dataset.
        y_validation: Provide data to initialize the dataset.
        x_test: Provide data to initialize the dataset.
        y_test: Provide data to initialize the dataset.
        db_name (str): Name of the dataset.

    """
    print("Building database: " + db_name)

    # Create a hdf5 dataset
    h5f = h5py.File(db_name, 'w')
    h5f.create_dataset('x_train', data=x_train)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('x_validation', data=x_validation)
    h5f.create_dataset('y_validation', data=y_validation)
    h5f.create_dataset('x_test', data=x_test)
    h5f.create_dataset('y_test', data=y_test)
    h5f.close()
    print("Done.")


def get_data(db_name, key):
    """
    Returns what is behind key node on the HDF5 db named db_name.

    Args:
        db_name (str): Name of the dataset.
        key (str): Name of the key in the dataset.

    Returns:
        The data under the given key.
    """
    # Load hdf5 dataset
    hdf5 = h5py.File(db_name, 'r')
    data = hdf5[key]  # i.e. xt = h5f['x_train']
    logging.warning('Remember that the hdf5 dataset is still open.')
    return data


def add_node(db_name, key, shape=None):
    """
    Adds node with name key to hdf5 database.

    Args:
        db_name (str): Name of the dataset.
        key (str): Name of the key in the dataset.
        shape (tuple of int): Dataset shape.  Use "()" for scalar datasets.  Required if "data" isn't provided.

    """
    if not os.path.isfile(db_name):
        h5f = h5py.File(db_name, 'w')
    else:
        h5f = h5py.File(db_name, 'r+')

    h5f.create_dataset(key, shape, maxshape=(None, 1))
    h5f.close()


def update_node(db_name, key):
    """ Change the content of a node. """
    raise NotImplementedError


def add_elements_to_existing_node(db_name, key,):
    """  Add elements below the node. """
    # h5f = h5py.File(db_name, 'r+')
    # h5f[key].resize((curr_num_samples, dimPatches, dimPatches, n_channel))
    # h5f[key][curr_num_samples - 1, :, :, :] = imgMatrix
    # h5f.close()
    raise NotImplementedError
