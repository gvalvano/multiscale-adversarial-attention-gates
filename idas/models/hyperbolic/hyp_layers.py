import tensorflow as tf
from idas.models.hyperbolic import hyp_ops
import numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


# def hyp_pixel_embedding(incoming, radius=1.0):
#
#     def _embed_pixel_by_pixel(x_in, chs_in):
#
#         if len(x_in.get_shape()) > 1:
#             # in tf.map_fn the parameter parallel_iterations defaults to 10
#             return tf.map_fn(lambda x: _embed_pixel_by_pixel(x, chs_in), x_in, back_prop=True)# TODO ?, parallel_iterations=1)
#         else:
#             x_in = tf.expand_dims(x_in, axis=0)  # add one dimension for compatibility with tf_exp_map_zero()
#             x_out = hyp_ops.tf_exp_map_zero(v=x_in, c=radius)
#             x_out = tf.squeeze(x_out, axis=0)  # remove the extra dimension and return
#             return x_out
#
#     # the output matrix will have shape [None, W, H, C]
#     channels_in = incoming.get_shape()[-1]
#     out_matrix = _embed_pixel_by_pixel(incoming, channels_in)
#
#     return out_matrix

from idas.utils.utils import print_yellow_text


def hyp_pixel_embedding(incoming, radius=1.0):

    # the output matrix will have shape [None, W, H, C]
    _, W, H, C = incoming.get_shape().as_list()
    incoming_flat = tf.reshape(incoming, shape=[-1, C])  # [batch_size * W * H, C])

    out_matrix = hyp_ops.tf_exp_map_zero(v=incoming_flat, c=radius)

    out_shape = tf.convert_to_tensor([-1, W, H, C])  # [batch_size, W, H, num_classes])
    out_matrix = tf.reshape(out_matrix, shape=out_shape)

    return out_matrix


def hyp_pixel_classification(incoming, num_classes, radius=1.0):

    # the output matrix will have shape [None, W, H, C]
    _, W, H, C = incoming.get_shape().as_list()
    incoming_flat = tf.reshape(incoming, shape=[-1, C])  # [batch_size * W * H, C])

    out_matrix = hyp_mlr(incoming_flat, before_mlr_dim=C, num_classes=num_classes, radius=radius, mlr_geom='hyp')

    out_matrix = tf.reshape(out_matrix, shape=[-1, W, H, num_classes])  # [batch_size, W, H, num_classes])

    return out_matrix


# def hyp_pixel_classification(incoming, num_classes, radius=1.0):
#
#     # number of input channels
#     channels_in = incoming.get_shape()[-1]
#
#     # batch on last dimension
#     incoming = tf.transpose(incoming, [1, 2, 3, 0])
#
#     def _classify_pixel_by_pixel(x_in):
#
#         if len(x_in.get_shape()) > 2:
#             # in tf.map_fn the parameter parallel_iterations defaults to 10
#             return tf.map_fn(lambda x: _classify_pixel_by_pixel(x), x_in, back_prop=True)# TODO ?, parallel_iterations=1)
#         else:
#
#             # put back batch axis on first dimension:
#             x_in = tf.transpose(x_in, [1, 0])  # now shape is: [None, K]
#
#             x_out = hyp_mlr(x_in, before_mlr_dim=channels_in, num_classes=num_classes, radius=radius)
#
#             # put back batch axis on last dimension:
#             x_out = tf.transpose(x_out, [1, 0])  # now shape is: [None, K]
#
#             return x_out
#
#     # the output matrix will have shape [None, W, H, C]
#     out_matrix = _classify_pixel_by_pixel(incoming)
#
#     # batch on first dimension again
#     out_matrix = tf.transpose(out_matrix, [3, 0, 1, 2])
#
#     return out_matrix


def hyp_conv2d(incoming):
    raise NotImplementedError


def hyp_dense(incoming, shape, activation='id', radius=1.0, bias_geom='hyp', mlr_geom='hyp', dropout=1.0):
    """ Fully connected layer in hyperbolic space
    :param incoming: incoming tensor
    :param shape: [hidden_dim, before_mlr_dim]
    :param activation: activation function. Supported values are in ['id', 'relu', 'tanh', 'sigmoid']
    :param radius: radius of the Poincaré ball
    :param bias_geom: geometry for the bias
    :param mlr_geom: TODO
    :param dropout: dropout keep probability (defaults to 1.0, no dropout). Values greater than 1.0 are treated as 1.0
    :return:
    """

    with tf.variable_scope('HyperbolicDenseLayer'):

        hidden_dim, before_mlr_dim = shape

        # Define variables for the feed-forward layer: W * x + b
        W = tf.get_variable('W_hyp', dtype=tf.float32, shape=[hidden_dim, before_mlr_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b_hyp', dtype=tf.float32, shape=[1, before_mlr_dim],
                            initializer=tf.constant_initializer(0.0))

        # matrix multiplication:
        mat_mul = hyp_ops.tf_mob_mat_mul(W, incoming, radius)

        # add bias:
        if bias_geom == 'eucl':
            b = hyp_ops.tf_exp_map_zero(b, radius)
        output = hyp_ops.tf_mob_add(mat_mul, b, radius)

        # apply activation function:
        output = hyp_ops.tf_hyp_non_lin(output, non_lin=activation,
                                        hyp_output=(mlr_geom == 'hyp' and dropout == 1.0),
                                        c=radius)

        # Mobius dropout
        if dropout < 1.0:
            # If we are here, then output should be Euclidean.
            output = tf.nn.dropout(output, keep_prob=dropout)
            if mlr_geom == 'hyp':  # TODO check MLR (Multi-class logistic regression) or move to hyp_mlr(.)
                output = hyp_ops.tf_exp_map_zero(output, radius)

    return output


def hyp_mlr(incoming, before_mlr_dim, num_classes, radius=1.0, reuse=tf.AUTO_REUSE,
            scope='HyperbolicMLR', mlr_geom='hyp'):
    """
    Multi-logistic regression in hyperbolic space.

    :param incoming: incoming tensor with shape [batch_size x before_mlr_dim]
    :param before_mlr_dim: last dimension of the incoming tensor
    :param num_classes: number of output classes
    :param radius: radius of the Poincaré ball
    :param scope: scope for the operation
    :param mlr_geom: TODO
    :return:
    """

    with tf.variable_scope(scope, reuse=reuse):
        A_mlr = []
        P_mlr = []
        logits_list = []
        for cl in range(num_classes):
            A_mlr.append(tf.get_variable('A_mlr' + str(cl),
                                         dtype=tf.float32,
                                         shape=[1, before_mlr_dim],
                                         initializer=tf.contrib.layers.xavier_initializer()))

            P_mlr.append(tf.get_variable('P_mlr' + str(cl),
                                         dtype=tf.float32,
                                         shape=[1, before_mlr_dim],
                                         initializer=tf.constant_initializer(0.0)))

            if mlr_geom == 'eucl':
                logits_list.append(tf.reshape(hyp_ops.tf_dot(-P_mlr[cl] + incoming, A_mlr[cl]), [-1]))

            elif mlr_geom == 'hyp':
                minus_p_plus_x = hyp_ops.tf_mob_add(-P_mlr[cl], incoming, radius)
                norm_a = hyp_ops.tf_norm(A_mlr[cl])
                lambda_px = hyp_ops.tf_lambda_x(minus_p_plus_x, radius)
                px_dot_a = hyp_ops.tf_dot(minus_p_plus_x, tf.nn.l2_normalize(A_mlr[cl]))

                logit = 2. / np.sqrt(radius) * norm_a * tf.asinh(np.sqrt(radius) * px_dot_a * lambda_px)
                logits_list.append(tf.reshape(logit, [-1]))

        logits = tf.stack(logits_list, axis=1)

    return logits
