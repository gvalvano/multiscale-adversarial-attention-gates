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

import time
import sys


class BColors:
    """
    Colors for formatted text.
    Examples:
        print(bcolors.WARNING + "Warning: This is a warning." + bcolors.ENDC)
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ProgressBar(object):
    def __init__(self, update_delay):
        """
        Progress bar.

        Args:
            update_delay (int): time to wait before updating the progress bar (in seconds).

        Examples:

        Example 1:

            bar = ProgressBar(0.2)
            bar.attach()
            for i in range(100):
                bar.monitor_progress()
                time.sleep2(0.1)
            bar.detach()

        Example 2: monitor also ETA (Estimated Time of Arrival)
                    notice that the progress bar is detached first and than the lta is updated

            bar = ProgressBar(0.5)
            for iter_step in range(3):
                bar.attach()
                t0 = time.time()
                for i in range(20):
                    bar.monitor_progress()
                    time.sleep(0.2)
                bar.detach() # detach first
                bar.update_lta(time.time() - t0) # then update lta

        """
        self.update_delay = update_delay
        self.lta = None  # last time of arrival (for ETA monitoring)

        # internal variables:
        self._old_time = None
        self._n_steps = 0  # number of progress bar steps
        self._last_suffix_len = 0

    def attach(self):
        # setup progress bar
        prefix = "  \033[31mProgress\033[0m:    "
        suffix = "" if self.lta is None else " ETA {0:.3f} s ".format(self.lta)
        self._last_suffix_len = len(suffix)
        sys.stdout.write(prefix + "[=> " + suffix)
        sys.stdout.flush()
        self._old_time = time.time()

    def monitor_progress(self):
        if self._old_time is None:
            self.WrongInitializationError.report()
            raise self.WrongInitializationError

        if time.time() - self._old_time > self.update_delay:
            # update the progress bar every 'self.update_delay' seconds
            if self.lta is None:
                n_back = 2
                suffix = ""
            else:
                self._n_steps += 1
                eta = self.lta - (self.update_delay * self._n_steps)
                n_back = 2 + self._last_suffix_len
                suffix = " ETA {0:.3f} s ".format(eta)
                self._last_suffix_len = len(suffix)

            sys.stdout.write("\b" * n_back + "=> " + suffix)
            sys.stdout.flush()
            self._old_time = time.time()

    def update_lta(self, lta):
        """ Update LTA with the value of the last time of arrival. """
        self.lta = lta

    def detach(self):
        """ This ends the progress bar. """

        if self._old_time is None:
            self.WrongInitializationError.report()
            raise self.WrongInitializationError

        if self.lta is None:
            n_back = 2
            sys.stdout.write("\b" * n_back + "=]{0}\n".format(" " * 50))  # this ends the progress bar
        else:
            n_back = 2 + self._last_suffix_len
            sys.stdout.write("\b" * n_back + "=]{0}\n".format(" " * self._last_suffix_len))

            # reset internal variables:
        self._n_steps = 0
        self._last_suffix_len = 0

    class WrongInitializationError(Exception):
        """ Raised if the progress bar is not correctly initialized """

        @staticmethod
        def report():
            instr_0 = 'bar = ProgressBar()'
            instr_1 = 'bar.attach()'
            instr_2 = 'bar.monitor_progress()'
            instr_3 = 'bar.detach()'
            print("\033[91m\nRemember that the correct procedure to use a ProgressBar is the following:"
                  "\n{4}{0}\n{4}{1}\n{4}{2}\n{4}{3}\n\033[0m".format(instr_0, instr_1, instr_2, instr_3, ' ' * 3))
            time.sleep(0.2)
