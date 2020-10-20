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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import random
import tensorflow as tf
from data_interface.interfaces.dataset_wrapper import DatasetInterfaceWrapper
from architectures.multi_res_segmentor import SAUNet
from architectures.multi_res_discriminator import MultiResDiscriminator
from idas.callbacks import callbacks as tf_callbacks
from idas.callbacks.routine_callback import RoutineCallback
from idas.utils import utils
from idas.callbacks.early_stopping_callback import EarlyStoppingCallback, EarlyStoppingException, NeedForTestException
from idas.metrics.tf_metrics import dice_coe
from idas.losses.tf_losses import weighted_cross_entropy
from idas.utils.progress_bar import ProgressBar
from idas.optimization.learning_rate import cyclic_learning_rate
from tensorflow.core.framework import summary_pb2
from data_interface.utils_acdc.split_data import get_splits


class Model(DatasetInterfaceWrapper):
    def __init__(self, run_id=None, config=None):
        """
        :param run_id: (str) used when we want to load a specific pre-trained model. Default run_id is taken from
                config_file.py
        :param config: either a tf.app.flags.FLAGS or an object with necessary attributes
        """

        # noinspection PyUnresolvedReferences
        FLAGS = tf.app.flags.FLAGS if (config is None) else config

        self.run_id = FLAGS.RUN_ID if (run_id is None) else run_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.CUDA_VISIBLE_DEVICE)
        self.verbose = FLAGS.verbose

        if self.verbose:
            print('CUDA_VISIBLE_DEVICE: \033[94m{0}\033[0m\n'.format(str(FLAGS.CUDA_VISIBLE_DEVICE)))
            print('RUN_ID = \033[94m{0}\033[0m'.format(self.run_id))

        self.num_threads = FLAGS.num_threads

        # -----------------------------
        # Model hyper-parameters:
        self.lr = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False, name='learning_rate')
        self.batch_size = FLAGS.b_size

        # -----------------------------
        # Data
        # data specifics
        self.input_size = (224, 224)  # FLAGS.input_size
        self.n_classes = 4  # FLAGS.n_classes

        # Get the list of paths for the training and validation files for ACDC
        self.acdc_data_path = FLAGS.data_path
        if self.verbose:
            print('Dataset dir: \033[94m{0}\033[0m\n'.format(self.acdc_data_path))

        # get volume ids for train, validation and test sets:
        self.acdc_data_ids = get_splits()[FLAGS.n_sup_vols][FLAGS.split_number]

        # data pre-processing
        self.augment = FLAGS.augment  # perform data augmentation
        self.standardize = FLAGS.standardize  # perform data standardization

        # -----------------------------
        # Report
        # path to save checkpoints and graph
        prefix = os.path.join(FLAGS.results_dir, 'results/{0}/'.format(FLAGS.dataset_name))
        self.last_checkpoint_dir = prefix + 'checkpoints/' + FLAGS.RUN_ID
        self.checkpoint_dir = prefix + 'checkpoints/' + FLAGS.RUN_ID
        self.graph_dir = prefix + 'graphs/' + FLAGS.RUN_ID + '/convnet'
        self.history_log_dir = prefix + 'history_logs/' + FLAGS.RUN_ID
        # verbosity
        self.skip_step = FLAGS.skip_step  # frequency of batch report
        self.train_summaries_skip = FLAGS.train_summaries_skip  # number of skips before writing train summaries
        self.tensorboard_verbose = FLAGS.tensorboard_verbose  # (bool) save also layers weights at the end of epoch
        # epoch offset for before early stopping:
        self.ep_offset = None

        # -----------------------------
        # Callbacks
        # init the list of callbacks to be called and relative arguments
        self.callbacks = []
        self.callbacks_kwargs = {'history_log_dir': self.history_log_dir}
        self.callbacks.append(RoutineCallback())  # routine callback always runs
        # Early stopping callback:
        self.callbacks_kwargs['es_loss'] = None
        self.last_val_loss = tf.Variable(1e10, dtype=tf.float32, trainable=False, name='last_val_loss')
        self.update_last_val_loss = self.last_val_loss.assign(
            tf.placeholder(tf.float32, None, name='best_val_loss_value'), name='update_last_val_loss')
        self.callbacks_kwargs['test_on_minimum'] = True
        self.callbacks.append(EarlyStoppingCallback(min_delta=1e-5, patience=2000))

        # -----------------------------
        # Other settings

        # Define global step for training e validation and counter for global epoch:
        self.g_train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_train_step')
        self.g_valid_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_validation_step')
        self.g_test_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_test_step')
        self.g_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_epoch')

        # define their update operations
        up_value = tf.placeholder(tf.int32, None, name='update_value')
        self.update_g_train_step = self.g_train_step.assign(up_value, name='update_g_train_step')
        self.update_g_valid_step = self.g_valid_step.assign(up_value, name='update_g_valid_step')
        self.update_g_test_step = self.g_test_step.assign(up_value, name='update_g_test_step')
        self.increase_g_epoch = self.g_epoch.assign_add(1, name='increase_g_epoch')

        # training or test mode (needed for the behaviour of dropout, BN, ecc.)
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # lr decay
        # self.decay_lr = self.lr.assign(tf.multiply(self.lr, 1.0), name='decay_lr')
        self.update_lr = self.lr.assign(
            cyclic_learning_rate(self.g_epoch, step_size=20,
                                 learning_rate=FLAGS.lr // 10, max_lr=FLAGS.lr,
                                 mode='triangular', gamma=.997), name='update_lr')

        # -----------------------------
        # initialize wrapper to the data set
        super().__init__(augment=self.augment,
                         standardize=self.standardize,
                         batch_size=self.batch_size,
                         input_size=self.input_size,
                         num_threads=self.num_threads,
                         verbose=FLAGS.verbose)

        # -----------------------------
        # initialize placeholders for the class
        # data pipeline placeholders:
        self.global_seed = None
        self.acdc_sup_train_init = None
        self.acdc_sup_valid_init = None
        self.acdc_sup_test_init = None
        self.acdc_sup_input_data = None
        self.acdc_sup_output_scrib = None
        self.acdc_sup_output_mask = None
        self.acdc_disc_train_init = None
        self.acdc_disc_valid_init = None
        self.acdc_disc_output_mask = None
        self.acdc_unsup_train_init = None
        self.acdc_unsup_valid_init = None
        self.acdc_unsup_input_data = None
        # tensors of the model:
        self.sup_pred_mask_soft = None
        self.sup_pred_mask_oh = None
        self.disc_pred_mask_soft = None
        self.disc_pred_mask_oh = None
        self.sup_multiscale_segmentor_pred = None
        self.disc_multiscale_segmentor_pred = None
        self.attention_tensors = None
        self.sup_disc_pred_real = None
        self.sup_disc_pred_fake = None
        self.disc_pred_real = None
        self.disc_pred_fake = None
        # metrics:
        self.masked_wxentropy_loss = None
        self.adv_disc_loss = None
        self.adv_gen_loss = None
        self.sup_disc_loss = None
        self.sup_loss = None
        self.generator_loss = None
        self.discriminator_loss = None
        self.dice_sup = None
        self.train_op_sup = None
        self.global_train_op = None
        # summaries:
        self.sup_train_scalar_summary_op = None
        self.sup_valid_scalar_summary_op = None
        self.sup_valid_images_summary_op = None
        self.sup_test_images_summary_op = None
        self.all_train_scalar_summary_op = None
        self.weights_summary = None

        # -----------------------------
        # output for test interface:
        self.test_init = None
        self.input = None
        self.prediction = None
        self.ground_truth = None
        self.multi_scale_prediction = None

        # -----------------------------
        # progress bar
        self.progress_bar = ProgressBar(update_delay=20)

    def build(self):
        """ Build the computation graph """
        if self.verbose:
            print('Building the computation graph...')
        self.get_data()
        self.define_model()
        self.define_losses()
        self.define_optimizers()
        self.define_eval_metrics()
        self.define_summaries()

    def get_data(self):
        """ Define the dataset iterators for each task (supervised, unsupervised, mask discriminator, future prediction)
        They will be used in define_model().
        """

        self.global_seed = tf.placeholder(tf.int64, shape=())

        # Repeat indefinitely all the iterators, exception made for the one iterating over the annotated dataset. This
        # ensures that every data is used during training.

        self.acdc_sup_train_init, self.acdc_sup_valid_init, self.acdc_sup_test_init, \
            self.acdc_sup_input_data, self.acdc_sup_output_scrib, self.acdc_sup_output_mask = \
            super(Model, self).get_acdc_sup_scribble_data(data_path=self.acdc_data_path, data_ids=self.acdc_data_ids,
                                                          repeat=False, seed=self.global_seed)

        self.acdc_unsup_train_init, self.acdc_unsup_valid_init, self.acdc_unsup_input_data, _ = \
            super(Model, self).get_acdc_unsup_data(data_path=self.acdc_data_path, data_ids=self.acdc_data_ids,
                                                   repeat=True, seed=self.global_seed)

        self.acdc_disc_train_init, self.acdc_disc_valid_init, _, self.acdc_disc_output_mask = \
            super(Model, self).get_acdc_disc_data(data_path=self.acdc_data_path, data_ids=self.acdc_data_ids,
                                                  repeat=True, seed=self.global_seed)

        self.test_init = self.acdc_sup_test_init

    def define_model(self):
        """ Define the network architecture. """

        # - - - - - - -
        # build Mask Generator - UNet (Least Square GAN)
        with tf.variable_scope('Generator'):
            unet = SAUNet(n_out=self.n_classes, n_out_att=self.n_classes, n_filters=32,
                          is_training=self.is_training, name='UNet')

            # -----------------
            # SUPERVISED BRANCH
            unet_sup = unet.build(self.acdc_sup_input_data)
            self.sup_pred_mask_soft = unet_sup.get_prediction(softmax=True)
            self.sup_pred_mask_oh = unet_sup.get_prediction(one_hot=True)

            self.sup_multiscale_segmentor_pred = [self.sup_pred_mask_soft[..., 1:]]
            self.sup_multiscale_segmentor_pred.extend([el[..., 1:] for el in unet_sup.attention_tensors])
            self.attention_tensors = unet_sup.attention_tensors

            # ------------------
            # ADVERSARIAL BRANCH
            unet_disc = unet.build(self.acdc_unsup_input_data, reuse=True)
            self.disc_pred_mask_soft = unet_disc.get_prediction(softmax=True)
            self.disc_pred_mask_oh = unet_disc.get_prediction(one_hot=True)

            self.disc_multiscale_segmentor_pred = [self.disc_pred_mask_soft[..., 1:]]
            self.disc_multiscale_segmentor_pred.extend([el[..., 1:] for el in unet_disc.attention_tensors])

        # - - - - - - -
        # build Mask Discriminator (Least Square GAN)
        with tf.variable_scope('Discriminator'):
            discriminator = MultiResDiscriminator(self.is_training, n_filters=32, out_mode='scalar',
                                                  instance_noise=True)
            # -----------------
            # Supervised track
            real_input_tensors = [self.acdc_disc_output_mask[..., 1:]]
            model_real = discriminator.build(real_input_tensors)
            self.sup_disc_pred_real = model_real.get_prediction()

            model_fake = discriminator.build(self.sup_multiscale_segmentor_pred, reuse=True)
            self.sup_disc_pred_fake = model_fake.get_prediction()

            # -----------------
            # Unsupervised track
            real_input_tensors = [self.acdc_disc_output_mask[..., 1:]]
            model_real = discriminator.build(real_input_tensors, reuse=True)
            self.disc_pred_real = model_real.get_prediction()

            model_fake = discriminator.build(self.disc_multiscale_segmentor_pred, reuse=True)
            self.disc_pred_fake = model_fake.get_prediction()

        self.input = self.acdc_sup_input_data
        self.prediction = self.sup_pred_mask_oh
        self.ground_truth = self.acdc_sup_output_mask

        # attention maps
        self.multi_scale_prediction = [self.sup_pred_mask_soft]
        self.multi_scale_prediction.extend(self.attention_tensors)

    def define_losses(self):
        """
        Define loss function for each task.
        """
        # _______
        # Weighted Cross Entropy loss:
        with tf.variable_scope('WXEntropy_loss'):
            masked_pred = self.sup_pred_mask_soft * self.acdc_sup_output_scrib
            self.masked_wxentropy_loss = weighted_cross_entropy(y_pred=masked_pred,
                                                                y_true=self.acdc_sup_output_scrib,
                                                                num_classes=self.n_classes)

        # _______
        # Mask Discriminator loss:
        # this is a LeastSquare GAN: use MSE as loss
        with tf.variable_scope('Discriminator_loss'):
            self.adv_disc_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.disc_pred_real, 1.0)) + \
                                 0.5 * tf.reduce_mean(tf.squared_difference(self.disc_pred_fake, -1.0))

            self.adv_gen_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.disc_pred_fake, 1.0))

            self.sup_disc_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.sup_disc_pred_real, 1.0)) + \
                                 0.5 * tf.reduce_mean(tf.squared_difference(self.sup_disc_pred_fake, -1.0))

        # define weights for the cost contributes:
        w_adv = 0.2

        # find dynamic weight to balance current loss:
        w_dynamic = tf.abs(tf.stop_gradient(self.sup_disc_loss / self.masked_wxentropy_loss))

        # define losses for supervised, unsupervised and frame prediction steps:
        self.sup_loss = w_dynamic * self.masked_wxentropy_loss + 0.1 * self.sup_disc_loss

        self.generator_loss = w_adv * self.adv_gen_loss
        self.discriminator_loss = w_adv * self.adv_disc_loss

    def define_optimizers(self):
        """
        Define training op
        using Adam Gradient Descent to minimize cost
        """

        # define lr decay
        decay = 1e-4
        self.lr = self.lr / (1. + tf.multiply(decay, tf.cast(self.g_epoch, tf.float32)))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            # -----------------
            # Adversarial loss contributes:
            with tf.name_scope("discriminator_train"):
                disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("Discriminator")]
                disc_optimizer = tf.train.AdamOptimizer(self.lr)
                disc_grads_and_vars = disc_optimizer.compute_gradients(self.discriminator_loss, var_list=disc_vars)
                train_op_discriminator = disc_optimizer.apply_gradients(disc_grads_and_vars,
                                                                        global_step=self.g_train_step)

            # In classical GAN architectures, you update discriminator and generator sequentially not in parallel.
            # Since I'm going to use tf.group() I will control dependencies before the updates
            with tf.name_scope("generator_train"):
                with tf.control_dependencies([train_op_discriminator]):  # , self.train_op_sup
                    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("Generator")]
                    gen_optimizer = tf.train.AdamOptimizer(self.lr)
                    gen_grads_and_vars = gen_optimizer.compute_gradients(self.generator_loss, var_list=gen_vars)
                    train_op_generator = gen_optimizer.apply_gradients(gen_grads_and_vars,
                                                                       global_step=self.g_train_step)

            # -----------------
            # Supervised segmentation loss
            with tf.name_scope("supervised_train"):
                self.train_op_sup = gen_optimizer.minimize(self.sup_loss)

        # train op generator has dependencies on the rest
        # self.global_train_op = train_op_generator
        self.global_train_op = tf.group(self.train_op_sup, train_op_generator)

    def define_eval_metrics(self):
        """
        Evaluate the model on the current batch
        """
        with tf.variable_scope('Dice_sup'):
            self.dice_sup = dice_coe(output=self.sup_pred_mask_oh[..., 1:],
                                     target=self.acdc_sup_output_mask[..., 1:])

    def define_summaries(self):
        """
        Create summaries to write on TensorBoard
        """
        # Scalar summaries:
        with tf.name_scope('Adversarial'):
            tr_adv_disc = tf.summary.scalar('train/adv_disc', self.adv_disc_loss)
            tr_adv_gen = tf.summary.scalar('train/adv_gen', self.adv_gen_loss)
            # tr_adv_pred_corr = tf.summary.scalar('test/tr_adv_pred_corr', self.adv_prediction_cost)
            # tr_adv_pred_grads = tf.summary.scalar('test/tr_adv_pred_grads', tf.reduce_mean(self.gradients))
            val_adv_disc = tf.summary.scalar('validation/adv_disc', self.adv_disc_loss)
            val_adv_gen = tf.summary.scalar('validation/adv_gen', self.adv_gen_loss)

        with tf.name_scope('Dice_loss'):
            tr_dice_loss = tf.summary.scalar('train/dice_sup', 1.0 - self.dice_sup)
            val_dice_loss = tf.summary.scalar('validation/dice_sup', 1.0 - self.dice_sup)

        with tf.name_scope('Supervised_loss'):
            tr_wxentropy_loss = tf.summary.scalar('train/masked_wxentropy', self.masked_wxentropy_loss)
            tr_disc_loss = tf.summary.scalar('train/discriminator', self.sup_disc_loss)
            val_wxentropy_loss = tf.summary.scalar('validation/masked_wxentropy', self.masked_wxentropy_loss)
            val_disc_loss = tf.summary.scalar('validation/discriminator', self.sup_disc_loss)

        # Image summaries:
        with tf.name_scope('0_Input'):
            img_inp_s = tf.summary.image('0_input_sup', self.acdc_sup_input_data, max_outputs=1)
        with tf.name_scope('1_Segmentation'):
            img_pred_mask = tf.summary.image('0_pred_mask', self.disc_pred_mask_oh[..., 1:], max_outputs=3)
        with tf.name_scope('2_Segmentation'):
            img_mask = tf.summary.image('0_gt_mask', self.acdc_disc_output_mask[..., 1:], max_outputs=3)
        with tf.name_scope('3_AttentionMaps'):
            img_att_maps = [tf.summary.image('0_attention_map_{0}'.format(i),
                                             self.sup_multiscale_segmentor_pred[i], max_outputs=3)
                            for i in range(len(self.sup_multiscale_segmentor_pred))]
        with tf.name_scope('4_Scribbles'):
            img_scrib_0 = tf.summary.image('0_input_sup', self.acdc_sup_input_data, max_outputs=2)
            img_scrib_1 = tf.summary.image('1_pred_segm', self.sup_pred_mask_oh[..., 1:], max_outputs=2)
            img_scrib_2 = tf.summary.image('2_gt_segm', self.acdc_sup_output_mask[..., 1:], max_outputs=2)
            img_scrib_3 = tf.summary.image('3_gt_scribble', self.acdc_sup_output_scrib[..., 1:], max_outputs=2)
            img_scrib_4 = tf.summary.image('4_all_gt_scribble', tf.reduce_sum(self.acdc_sup_output_scrib, axis=-1,
                                                                              keepdims=True), max_outputs=2)
            scrib_summaries = [img_scrib_0, img_scrib_1, img_scrib_2, img_scrib_3, img_scrib_4]
        with tf.name_scope('5_TEST_RESULTS'):
            img_test_results = list()
            img_test_results.append(tf.summary.image('0_input_sup', self.acdc_sup_input_data, max_outputs=10))
            img_test_results.append(tf.summary.image('1_pred_segm', self.sup_pred_mask_oh[..., 1:], max_outputs=10))

        # merging all scalar summaries:
        sup_train_scalar_summaries = [tr_dice_loss, tr_adv_disc, tr_adv_gen, tr_wxentropy_loss, tr_disc_loss]
        sup_valid_scalar_summaries = [val_dice_loss, val_adv_disc, val_adv_gen, val_wxentropy_loss, val_disc_loss]
        sup_test_images_summaries = img_test_results

        self.sup_train_scalar_summary_op = tf.summary.merge(sup_train_scalar_summaries)
        self.sup_valid_scalar_summary_op = tf.summary.merge(sup_valid_scalar_summaries)

        all_train_summaries = []
        all_train_summaries.extend(sup_train_scalar_summaries)
        self.all_train_scalar_summary_op = tf.summary.merge(all_train_summaries)

        # _______________________________
        # merging all images summaries:
        sup_valid_images_summaries = [img_inp_s, img_mask, img_pred_mask]
        sup_valid_images_summaries.extend(img_att_maps)
        sup_valid_images_summaries.extend(scrib_summaries)

        self.sup_valid_images_summary_op = tf.summary.merge(sup_valid_images_summaries)
        self.sup_test_images_summary_op = tf.summary.merge(sup_test_images_summaries)

        # ---- #
        if self.tensorboard_verbose:
            _vars = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'kernel' in v.name]
            weights_summary = [tf.summary.histogram(v, tf.get_default_graph().get_tensor_by_name(v)) for v in _vars]
            self.weights_summary = tf.summary.merge(weights_summary)

    def _train_all_op(self, sess, writer, step):
        _, sl, scalar_summaries = sess.run([self.global_train_op,
                                            self.sup_loss,
                                            self.all_train_scalar_summary_op],
                                           feed_dict={self.is_training: True})

        if random.randint(0, self.train_summaries_skip) == 0:
            writer.add_summary(scalar_summaries, global_step=step)

        return sl, 0.0

    def train_one_epoch(self, sess, iterator_init_list, writer, step, caller, seed):
        """ train the model for one epoch. """
        start_time = time.time()

        # setup progress bar
        self.progress_bar.attach()
        self.progress_bar.monitor_progress()

        # initialize data set iterators:
        for init in iterator_init_list:
            sess.run(init, feed_dict={self.global_seed: seed})

        total_sup_loss = 0
        total_unsup_loss = 0
        n_batches = 0

        try:
            while True:
                self.progress_bar.monitor_progress()

                caller.on_batch_begin(training_state=True, **self.callbacks_kwargs)

                sup_loss, unsup_loss = self._train_all_op(sess, writer, step)
                total_sup_loss += sup_loss
                total_unsup_loss += unsup_loss
                step += 1

                n_batches += 1
                if (n_batches % self.skip_step) == 0 and self.verbose:
                    print('\r  ...training over batch {1}: {0} batch_sup_loss = {2:.4f}\tbatch_unsup_loss = {3:.4f} {0}'
                          .format(' ' * 3, n_batches, total_sup_loss, total_unsup_loss), end='\n')

                caller.on_batch_end(training_state=True, **self.callbacks_kwargs)

        except tf.errors.OutOfRangeError:
            # End of the epoch. Compute statistics here:
            total_loss = total_sup_loss + total_unsup_loss
            avg_loss = total_loss / n_batches
            delta_t = time.time() - start_time

        # update global epoch counter:
        sess.run(self.increase_g_epoch)
        sess.run(self.update_g_train_step, feed_dict={'update_value:0': step})

        # detach progress bar and update last time of arrival:
        self.progress_bar.detach()
        self.progress_bar.update_lta(delta_t)

        print('\033[31m  TRAIN\033[0m:{0}{0} average loss = {1:.4f} {0} Took: {2:.3f} seconds'
              .format(' ' * 3, avg_loss, delta_t))
        return step

    def _eval_all_op(self, sess, writer, step):
        sl, d3chl, sup_sc_summ, sup_im_summ, = \
            sess.run([self.sup_loss, self.dice_sup,
                      self.sup_valid_scalar_summary_op,
                      self.sup_valid_images_summary_op],
                     feed_dict={self.is_training: False})
        writer.add_summary(sup_sc_summ, global_step=step)
        writer.add_summary(sup_im_summ, global_step=step)
        return sl, d3chl, 0.0

    def eval_once(self, sess, iterator_init_list, writer, step, caller):
        """ Eval the model once """
        start_time = time.time()

        # initialize data set iterators:
        for init in iterator_init_list:
            sess.run(init)

        total_sup_loss = 0
        total_dice_score = 0
        total_unsup_loss = 0
        n_batches = 0
        try:
            while True:
                caller.on_batch_begin(training_state=False, **self.callbacks_kwargs)

                sup_loss, dice_score, unsup_loss = self._eval_all_op(sess, writer, step)
                total_dice_score += dice_score
                total_sup_loss += sup_loss
                total_unsup_loss += unsup_loss
                step += 1

                n_batches += 1
                caller.on_batch_end(training_state=False, **self.callbacks_kwargs)

        except tf.errors.OutOfRangeError:
            # End of the validation set. Compute statistics here:
            total_loss = total_sup_loss + total_unsup_loss
            avg_loss = total_loss / n_batches
            avg_dice = total_dice_score / n_batches
            dice_loss = 1.0 - avg_dice
            delta_t = time.time() - start_time

            value = summary_pb2.Summary.Value(tag="Dice_1/validation/dice_3channels_avg", simple_value=avg_dice)
            summary = summary_pb2.Summary(value=[value])
            writer.add_summary(summary, global_step=step)

            pass

        # update global epoch counter:
        sess.run(self.update_g_valid_step, feed_dict={'update_value:0': step})

        print('\033[31m  VALIDATION\033[0m:  average loss = {1:.4f} {0} Took: {2:.3f} seconds'
              .format(' ' * 3, avg_loss, delta_t))
        return step, dice_loss

    def test_once(self, sess, sup_test_init, writer, step, caller):
        """ Test the model once """
        start_time = time.time()

        # initialize data set iterators:
        sess.run(sup_test_init)

        total_dice_score = 0
        n_batches = 0
        try:
            while True:
                caller.on_batch_begin(training_state=False, **self.callbacks_kwargs)

                dice_sup, images_summaries = sess.run([self.dice_sup, self.sup_test_images_summary_op],
                                                      feed_dict={self.is_training: False})
                total_dice_score += dice_sup
                writer.add_summary(images_summaries, global_step=step)

                n_batches += 1

        except tf.errors.OutOfRangeError:
            # End of the test set. Compute statistics here:
            avg_dice = total_dice_score / n_batches
            delta_t = time.time() - start_time

            step += 1
            value = summary_pb2.Summary.Value(tag="y_TEST/test/dice_3channels_avg", simple_value=avg_dice)
            summary = summary_pb2.Summary(value=[value])
            writer.add_summary(summary, global_step=step)
            pass

        # update global epoch counter:
        sess.run(self.update_g_test_step, feed_dict={'update_value:0': step})

        print('\033[31m  TEST\033[0m:{0}{0} \033[1;33m average dice = {1:.4f}\033[0m on \033[1;33m{2}\033[0m batches '
              '{0} Took: {3:.3f} seconds'.format(' ' * 3, avg_dice, n_batches, delta_t))
        return step

    def train(self, n_epochs):
        """ The train function alternates between training one epoch and evaluating """
        if self.verbose:
            print("\nStarting network training... Number of epochs to train: \033[94m{0}\033[0m".format(n_epochs))
            print("Tensorboard verbose mode: \033[94m{0}\033[0m".format(self.tensorboard_verbose))
            print("Tensorboard dir: \033[94m{0}\033[0m".format(self.graph_dir))
            print("Data augmentation: \033[94m{0}\033[0m, Data standardization: \033[94m{1}\033[0m."
                  .format(self.augment, self.standardize))

        utils.safe_mkdir(self.checkpoint_dir)
        utils.safe_mkdir(self.history_log_dir)
        writer = tf.summary.FileWriter(self.graph_dir, tf.get_default_graph())

        # config for the session: allow growth for GPU to avoid OOM when other processes are running
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver(max_to_keep=2)  # keep_checkpoint_every_n_hours=2
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.last_checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            trained_epochs = self.g_epoch.eval()
            if self.verbose:
                print("Model already trained for \033[94m{0}\033[0m epochs.".format(trained_epochs))
            t_step = self.g_train_step.eval()  # global step for train
            v_step = self.g_valid_step.eval()  # global step for validation
            test_step = self.g_test_step.eval()  # global step for test

            # Define a caller to call the callbacks
            self.callbacks_kwargs.update({'sess': sess, 'cnn': self})
            caller = tf_callbacks.ChainCallback(callbacks=self.callbacks)
            caller.on_train_begin(training_state=True, **self.callbacks_kwargs)

            # trick to find performance bugs: this will raise an exception if any new node is inadvertently added to the
            # graph. This will ensure that I don't add many times the same node to the graph (which could be expensive):
            tf.get_default_graph().finalize()

            # saving callback:
            self.callbacks_kwargs['es_loss'] = 100  # some random initialization

            if self.ep_offset is None:
                print('\nNo offset specified for the early stopping criterion')
                self.ep_offset = n_epochs // 2
                print(' >> Proceeding with default early stopping offset: \033[94m{0} epochs\033[0m'.format(self.ep_offset))

            for epoch in range(n_epochs):
                ep_str = str(epoch + 1) if (trained_epochs == 0) else '({0}+) '.format(trained_epochs) + str(epoch + 1)
                print('_' * 40 + '\n\033[1;33mEPOCH {0}\033[0m - {1} : '.format(ep_str, self.run_id))
                caller.on_epoch_begin(training_state=True, **self.callbacks_kwargs)

                global_ep = sess.run(self.g_epoch)
                sess.run(self.update_lr)

                seed = global_ep

                # TRAIN MODE ------------------------------------------
                iterator_init_list = [self.acdc_sup_train_init,
                                      self.acdc_unsup_train_init,
                                      self.acdc_disc_train_init]
                t_step = self.train_one_epoch(sess, iterator_init_list, writer, t_step, caller, seed)

                # VALIDATION MODE ------------------------------------------
                if global_ep >= self.ep_offset or not ((global_ep + 1) % 15):  # when to evaluate the model
                    iterator_init_list = [self.acdc_sup_valid_init,
                                          self.acdc_disc_valid_init,
                                          self.acdc_unsup_valid_init]
                    v_step, val_loss = self.eval_once(sess, iterator_init_list, writer, v_step, caller)

                    if global_ep >= self.ep_offset:
                        self.callbacks_kwargs['es_loss'] = val_loss
                    sess.run(self.update_last_val_loss, feed_dict={'best_val_loss_value:0': val_loss})

                if self.tensorboard_verbose and (global_ep % 10 == 0):
                    # writing summary for the weights:
                    summary = sess.run(self.weights_summary)
                    writer.add_summary(summary, global_step=t_step)

                try:
                    caller.on_epoch_end(training_state=True, **self.callbacks_kwargs)
                except EarlyStoppingException:
                    utils.print_yellow_text('\nEarly stopping...\n')
                    break
                except NeedForTestException:
                    if global_ep >= self.ep_offset:  # minimum epochs to wait before applying early stopping
                        # save the updated variables and weights --> this is the best model found (early stopping)
                        saver.save(sess, self.checkpoint_dir + '/checkpoint', t_step)

            caller.on_train_end(training_state=True, **self.callbacks_kwargs)

            # end of the training: save the current weights in a new sub-directory
            utils.safe_mkdir(self.checkpoint_dir + '/last_model')
            saver.save(sess, self.checkpoint_dir + '/last_model/checkpoint', t_step)

            # load best model and do a test:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            _ = self.test_once(sess, self.acdc_sup_test_init, writer, test_step, caller)

        writer.close()


if __name__ == '__main__':
    print('\n' + '-' * 3)
    model = Model()
    model.build()
    model.train(n_epochs=2)
