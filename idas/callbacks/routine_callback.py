"""
This is a callback which is always run during the training. 
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

from .callbacks import Callback
from .dsd_callback import check_for_sparse_training
from .lr_annealing_callback import check_for_annealed_lr
import os


class RoutineCallback(Callback):
    """Routine callback: it is always called at the beginning of the training. """

    def __init__(self):
        super().__init__()
        self.history_log_file = None

    def on_train_begin(self, training_state, **kwargs):

        print("\nRunning RoutineCallback...")
        if kwargs['cnn'] is None:
            raise Exception

        self.history_log_file = kwargs['history_log_dir'] + os.sep + 'train_history.json'

        # Here perform any check operation on log files, etc.:
        runs = list()
        runs.append(check_for_sparse_training(kwargs['cnn'], self.history_log_file))
        runs.append(check_for_annealed_lr(kwargs['cnn'], kwargs['sess'], self.history_log_file))

        # check if the callback has performed any operation:
        has_been_run = runs.count(True) > 0
        if has_been_run:
            print(" >> More then zero actions performed.")
        print("Done.")
