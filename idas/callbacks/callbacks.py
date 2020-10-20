"""
File with the definition of the callbacks.
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


class Callback(object):
    """ Callback base class. """
    def __init__(self):
        pass

    def on_train_begin(self, training_state, **kwargs):
        pass

    def on_epoch_begin(self, training_state, **kwargs):
        pass

    def on_batch_begin(self, training_state, **kwargs):
        pass

    def on_batch_end(self, training_state, **kwargs):
        pass

    def on_epoch_end(self, training_state, **kwargs):
        pass

    def on_train_end(self, training_state, **kwargs):
        pass


class ChainCallback(Callback):
    """
    Series of callbacks
    """
    def __init__(self, callbacks=None):
        super().__init__()
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    def on_train_begin(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(training_state, **kwargs)

    def on_epoch_begin(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(training_state, **kwargs)

    def on_batch_begin(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(training_state, **kwargs)

    def on_batch_end(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(training_state, **kwargs)

    def on_epoch_end(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(training_state, **kwargs)

    def on_train_end(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(training_state, **kwargs)

    def add(self, callback):
        if not isinstance(callback, Callback):
            raise Exception(str(callback) + " is an invalid Callback object")

        self.callbacks.append(callback)
