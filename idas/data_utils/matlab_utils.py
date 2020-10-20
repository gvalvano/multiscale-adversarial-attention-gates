"""
Utilities for matlab data
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

import scipy.io as sio


def get_matlab_matrix(filename, mdict=None, appendmat=True, **kwargs):
    """
    Gets a Matlab matrix from a given file.

    Args:
        filename (string): path to the numpy file.
        mdict (dict): Dictionary in which to insert matfile variables (optional)
        appendmat (bool): True to append the .mat extension to the end of the given filename, if not already present
            (optional).
        **kwargs:

    Returns:
        mdict, dictionary with variable names as keys, and loaded matrices as values.
    """
    return sio.loadmat(filename, mdict, appendmat, **kwargs)


def save_matlab_matrix(filename, mdict, appendmat=True, format='5', long_field_names=False, do_compression=False,
                       oned_as='row'):
    """
    Saves matlab matrix to given path (filename).

    Args:
        filename (str or file-like object): Name of the .mat file (.mat extension not needed if ``appendmat == True``).
            Can also pass open file_like object.
        mdict (dict): Dictionary from which to save matfile variables.
        appendmat (bool): True (the default) to append the .mat extension to the end of the given filename, if not
            already present.
        format: ({'5', '4'}, string): (optional)
            '5' (the default) for MATLAB 5 and up (to 7.2),
            '4' for MATLAB 4 .mat files.
        long_field_names (bool): (optional) False (the default) - maximum field name length in a structure is 31
            characters which is the documented maximum length. True - maximum field name length in a structure is 63
            characters which works for MATLAB 7.6+.
        do_compression (bool): (optional) Whether or not to compress matrices on write. Default is False.
        oned_as: ({'row', 'column'}): (optional) If 'column', write 1-D numpy arrays as column vectors. If 'row',
            write 1-D numpy arrays as row vectors.

    """
    sio.savemat(filename, mdict, appendmat, format, long_field_names, do_compression, oned_as)

