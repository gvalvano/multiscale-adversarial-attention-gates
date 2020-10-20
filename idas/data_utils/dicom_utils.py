"""
Utilities for dicom data
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

import dicom
import numpy as np


def get_dicom_matrix(filename, dtype=np.float):
    """
    Get pixel array from desired dicom file.

    Args:
        filename (str): name of the dicom file.
        dtype (type): type of the returned array

    Returns:

    """
    mdicom = dicom.read_file(filename)
    return mdicom.pixel_array.astype(dtype)
