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

from tensorflow.python.client import device_lib
import os


def get_available_gpus():
    """
    Prints the available GPUs on the machine.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    """
    Prints the available CPUs on the machine.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


def safe_mkdir(path):
    """
    Create a directory if there isn't one already.

    Args:
        path (string): path to the directory to create.
    """
    try:
        os.makedirs(path)
    except OSError:
        pass


def create_gif(gif_name, path, ext='.png', delay=30, loop=0):
    """
    Create gif from the list of images under the given path. On Mac OS X, it requires "brew install ImageMagick".

    Args:
        gif_name (string): name of the gif file.
        path (string): path to the images. This is also the folder where the output gif will be saved
        ext (str): extension of the images
        delay (int): delay time
        loop (int): loop parameter (defaults to 0, i.e. infinite loop)

    """
    if path[-1] == '/':
        path = path[:-1]
    if gif_name[-4:] == '.gif':
        gif_name = gif_name[:-4]

    cmd = 'convert -delay {0} -loop {1} {2}/*{3} {2}/{4}.gif'.format(delay, loop, path, ext, gif_name)
    os.system(cmd)
    print('gif successfully saved as {0}.gif'.format(os.path.join(path, gif_name)))


def print_yellow_text(text, sep=True):
    """
    Useful for debug. Prints yellow colored text.

    Args:
        text (string): text to print
        sep (Boolean): if True, a separation line is printed before the text.

    """
    if sep:
        print('_' * 40)  # line separator
    print('\033[1;33m{0}\033[0m'.format(text))


def processing_time(test_function, *args):
    """
    Evaluates the time needed to execute the given function

    Args:
        test_function (function): function to execute
        *args:

    Returns:
        Processing time and outputs of the function

    """
    start_time = time.time()
    outputs = test_function(*args)
    delta_t = time.time() - start_time
    return delta_t, outputs


class Colors:
    """ Colors for formatted text.
    Example: print(Colors.WARNING + "Warning: This is a warning." + Colors.ENDC)
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
