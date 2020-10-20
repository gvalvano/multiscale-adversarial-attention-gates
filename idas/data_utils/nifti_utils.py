"""
Utilities for nifti data
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

import nibabel as nib
import numpy as np


def get_nifti_matrix(filename, dtype=np.int16):
    """
    Returns the array and the affine matrix contained in a nifti file.

    Args:
        filename (string): filename of the nifti file
        dtype (type): dtype of the output array

    Returns:
        array and affine matrix

    """
    array = nib.load(filename).get_data().astype(dtype)  # array
    affine = nib.load(filename).affine  # affine matrix
    return array, affine


def save_nifti_matrix(array, affine, filename, dtype=np.int16):
    """
    Saves a nifti array with a given affine matrix with the given dtype (default numpy.int16)

    Args:
        array (np.array): array of data to be saved
        affine (np.array): affine matrix
        filename (string): filename of the nifti file
        dtype (type): dtype of the output array

    Returns:

    """
    nimage = nib.Nifti1Image(array.astype(dtype), affine)
    nib.save(nimage, filename=filename)
