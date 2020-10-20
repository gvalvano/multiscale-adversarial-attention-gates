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

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import config
import importlib
from idas.utils.utils import Colors, safe_mkdir
from data_interface.utils_acdc.prepare_dataset import *

# ----------------------------------------------------------------------------------- #
# test our model on ACDC test data
EXPERIMENT = 'model_ours_full_acdc'
DATASET_NAME = 'acdc'
TEST_ROOT_DIR = '../DATA/ACDC_testing'
OUT_DIR = './acdc_test_results'
# ----------------------------------------------------------------------------------- #

safe_mkdir(OUT_DIR)
config.define_flags()
# noinspection PyUnresolvedReferences
FLAGS = tf.app.flags.FLAGS

# ----------------------------------------------------------------------------------- #


def parse_info_cfg(filename):
    """
    Extracts information contained in the Info.cfg file given as input.
    :param filename: path/to/patient/folder/Info.cfg
    :return: values for: ed, es
    """
    ed, es = None, None
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ED: '):
                ed = int(line.split('ED: ')[1])

            elif line.startswith('ES: '):
                es = int(line.split('ES: ')[1])

    assert all(v is not None for v in [ed, es])
    return ed, es


# noinspection PyBroadException
def get_volume_specs(filename):
    """ Pre-processing pipeline.
     With respect to mask_pre_processing_pipeline():
            point 7 uses bi-cubic interpolation and point 9 is performed
    """
    # load nifti file
    img = nib.load(filename)

    # get image resolution on the slice axis x and y
    header = img.header
    try:
        dx, dy, dz, dt = header.get_zooms()
    except:
        dx, dy, dz = header.get_zooms()

    specs = {'resolution': (dx, dy),
             'shape': img.shape,
             'header': header,
             'affine': img.affine}

    return specs


def slice_pre_processing_pipeline(filename):
    """ Pre-processing pipeline.
    """
    # load nifti file
    img = nib.load(filename)

    # evaluate output shape after rescaling
    x_max_scaled = 224
    y_max_scaled = 224

    # get array
    img_array = img.get_fdata()

    # put all the slices on the first axis
    img_array = get_independent_slices(img_array)

    # interpolate to obtain desired output shapes
    size = (x_max_scaled, y_max_scaled)
    img_array = resize_2d_slices(img_array, new_size=size, interpolation=cv2.INTER_CUBIC)

    # standardize and clip values outside of 5-95 percentile range
    img_array = standardize_and_clip(img_array)

    return img_array


def get_processed_volumes(fname):
    # get preprocessed image:
    img_array = slice_pre_processing_pipeline(fname)
    img_array = np.expand_dims(img_array, -1)

    # get specs from nifti files:
    specs = get_volume_specs(fname)
    return img_array, specs


def resize_predicted_batch(batch, new_size, interpolation):
    """
    Resize the frames
    :param batch: [np.array] input batch of images, with shape [n_batches, width, height]
    :param new_size: [int, int] output size, with shape (N, M)
    :param interpolation: interpolation type
    :return: resized batch, with shape (n_batches, N, M)
    """
    n_batches, x, y, n_classes = batch.shape
    output = []
    for k in range(n_batches):
        img = []
        for c in range(n_classes):
            pred = batch[k, ..., c].astype(np.float32)
            pred = cv2.resize(pred, (new_size[1], new_size[0]), interpolation=interpolation)
            img.append(pred)
        img = np.stack(img, axis=-1)
        output.append(img)
    return np.array(output)


def post_process_segmentation(soft_mask, specs):
    shape = [specs['shape'][0], specs['shape'][1]]
    soft_mask = resize_predicted_batch(soft_mask, new_size=shape, interpolation=cv2.INTER_CUBIC)

    # get argmax and then go back to original axis
    soft_mask = np.argmax(soft_mask, axis=-1)
    soft_mask = np.transpose(soft_mask, axes=[1, 2, 0])
    return soft_mask


def save_nifti_files(name, mask, specs):
    header = specs['header']
    affine = specs['affine']
    new_image = nib.Nifti1Image(mask, affine, header)
    nib.save(new_image, name)


def parse_model_type():
    """ Import the correct model for the experiments """
    experiment = FLAGS.experiment
    dataset_name = FLAGS.dataset_name
    model = importlib.import_module('experiments.{0}.{1}'.format(dataset_name, experiment)).Model()
    return model


def test(sess, model):
    """ Test the model on ACDC test data """
    # Test
    # sess.run(model.test_init)  # initialize data set iterator on test set:
    y_pred = model.prediction_soft  # model prediction (the output of a softmax)

    # iterate over the testing volumes
    for idx in range(101, 151):
        pt_number = str(idx).zfill(3)
        print('Processing test volume: {0}'.format(pt_number))

        folder_name = 'patient{0}'.format(pt_number)
        prefix = os.path.join(TEST_ROOT_DIR, folder_name)

        # get ED and ES infos and then the patient path
        ed, es = parse_info_cfg(os.path.join(prefix, 'Info.cfg'))

        # -------------------------------------------------------------------
        # get ED data and test
        pt_full_path = os.path.join(prefix, 'patient' + pt_number + '_frame{0}.nii.gz'.format(str(ed).zfill(2)))
        img_array, specs = get_processed_volumes(fname=pt_full_path)
        prediction = sess.run(y_pred, feed_dict={model.acdc_sup_input_data: img_array, model.is_training: False})
        prediction = post_process_segmentation(prediction, specs)

        # save
        out_name = os.path.join(OUT_DIR, 'patient' + pt_number + '_ED.nii.gz')
        save_nifti_files(out_name, prediction, specs)

        # -------------------------------------------------------------------
        # get ES data and test
        pt_full_path = os.path.join(prefix, 'patient' + pt_number + '_frame{0}.nii.gz'.format(str(es).zfill(2)))
        img_array, specs = get_processed_volumes(fname=pt_full_path)
        prediction = sess.run(y_pred, feed_dict={model.acdc_sup_input_data: img_array, model.is_training: False})
        prediction = post_process_segmentation(prediction, specs)

        # save
        out_name = os.path.join(OUT_DIR, 'patient' + pt_number + '_ES.nii.gz')
        save_nifti_files(out_name, prediction, specs)


def main(_):
    print('\n\n' + '__' * 20 + '\n')
    # import the correct model for the experiment
    model = parse_model_type()
    print('\033[31m  RUN_ID\033[0m: \033[1;33m{0}\033[0m'.format(model.run_id))
    print('\033[31m  Checkpoint\033[0m: {0}'.format(model.checkpoint_dir + '/checkpoint'))

    # model.batch_size = 1 for a test slice by slice
    model.get_data()
    model.define_model()
    # model.define_eval_metrics()

    # config for the session: allow growth for GPU to avoid OOM when other processes are running
    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True

    with tf.Session(config=configs) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state(os.path.dirname(model.last_checkpoint_dir + '/best_model/checkpoint'))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model.checkpoint_dir + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError("{0}\n  Checkpoint not found: {2}{1}\n".format(Colors.FAIL, Colors.ENDC,
                                                                                   model.checkpoint_dir))

        epoch = sess.run(model.g_epoch)
        print('\033[31m  Epoch\033[0m: = {0}'.format(epoch))

        # do a test:
        test(sess, model)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
