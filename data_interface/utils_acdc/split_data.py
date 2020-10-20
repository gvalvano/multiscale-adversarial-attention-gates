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


def get_splits():
    """""
    Returns an array of splits into validation, test and train indices.
    """

    splits = {

        # ------------------------------------------------------------------------------------------------------------
        # Write here the splits you want to use for training:

        'perc25': {
            #  this is the percentage of annotated data

            'split0': {
                # this is the split for the cross-validation

                'test': [5, 39, 77, 82, 78, 10, 64, 24, 30, 73, 80, 41, 36, 60, 72],
                'validation': [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95],
                'train_unsup': [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90],
                'train_disc': [2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                               23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15],
                'train_sup': [3, 7, 8, 12, 21, 33, 42, 48, 58, 63, 67, 74, 76, 88, 91, 92, 97, 98]
                },
        },

        'perc50': {
            #  this is the percentage of annotated data

            'split0': {
                # this is the split for the cross-validation

                'test': [5, 39, 77, 82, 78, 10, 64, 24, 30, 73, 80, 41, 36, 60, 72],
                'validation': [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95],
                'train_unsup': [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90],
                'train_disc': [2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                               23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15],
                'train_sup': [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                              71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90]
            },
        },

        # -----------------
        # All the data:

        'all_data': {'all': list(np.arange(100) + 1)}
    }
    return splits


if __name__ == '__main__':
    _splits = get_splits()
    for k, v in zip(_splits.keys(), _splits.values()):
        print('\n' + '- '*20)
        print('number of volumes: {0}'.format(k))
        print('values:', v)
