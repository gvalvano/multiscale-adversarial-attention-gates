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

import numpy as np


def np_zero_pad(array, size=None, reference=None, offset=None):
    """
    Zero-pads numpy array to the given size.
    
    Args:
        array (np.array): array to be padded
        size (tuple of ints, or list of ints): desired shape
        reference (np.array): if size is None, then the desired size is evaluated as the shape of the reference array
        offset (list, or tuple): list of offsets (number of elements must be equal to the dimension of the array) 

    Returns:
        The padded array.

    """
    # Create an array of zeros with the desired shape
    if size is None:
        output_shape = reference.shape
    else:
        output_shape = size

    result = np.zeros(output_shape)

    # if it is None, fill 'offset' variable with zeros along each dimension:
    if offset is None:
        offset = np.zeros(array.ndim)

    # Create a list of slices from offset to offset + shape in each dimension
    insert_here = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range(array.ndim)]

    # Insert the array in the result at the specified offsets
    result[insert_here] = array

    return result
