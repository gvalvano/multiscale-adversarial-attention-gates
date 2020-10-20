"""
Utilities for .csv data
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

import pandas as pd


def get_csv_data(filename, sep=','):
    """
    Loads CSV into Numpy array.

    Args:
        filename (str): name of the CSV file.
        sep (str): sep
    """
    data_frame = pd.read_csv(filename, sep=sep)
    return data_frame

#
# def save_csv_data(array, filename, sep=','):
#     """ Saves CSV data from Numpy array."""
#     data_frame = pd.DataFrame(data=, index=, columns=)
#     data_frame.to_csv(filename, sep=sep)
