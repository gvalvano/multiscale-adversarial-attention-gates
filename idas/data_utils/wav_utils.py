"""
Utilities for audio .wav data
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

from scipy.io import wavfile


def get_wav_freq_and_data(filename):
    """
    Returns 1D array data and sampling frequency from .wav filename.

    Args:
        filename (string): name of the file .wav

    Returns:
        data, sampling frequency
    """
    freq, data = wavfile.read(filename)
    return data, freq
