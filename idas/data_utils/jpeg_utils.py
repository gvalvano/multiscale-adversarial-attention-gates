"""
Utilities for .jpg and .jpeg data
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

from PIL import Image
import numpy as np


def get_jpg_image(filename):
    """
    Loads JPEG image into 3D Numpy array of shape (width, height, channels).

    Args:
        filename (str): file name
    """
    with Image.open(filename) as image:
        im_arr = np.array(image)
    return im_arr


def save_jpg_image(array, filename):
    """
    Saves JPEG image from array 3D Numpy array of shape (width, height, channels).

    Args:
        array (np.array): array to save
        filename (str): file name

    """
    img = Image.fromarray(array)
    img.save(filename)
