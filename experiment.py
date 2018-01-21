import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np


M = 32


BASEDIR = "/home/brand/Research/hypercomplex test/"
LEARNING_RATE = 10.0
LOSS_COLLECTION = "loss_collection"


HXC_LIBDIR = "/home/brand/tensorflow/bazel-bin/tensorflow/core/user_ops/hypercomplex.so"
HCX = tf.load_op_library(HXC_LIBDIR)


@ops.RegisterGradient("HypercomplexConjugate")
def _hypercomplex_conjugate_grad(op, grad):
    return [HCX.hypercomplex_conjugate(grad)]


@ops.RegisterGradient("HypercomplexMultiply")
def _hypercomplex_multiply_grad(op, grad):
    return [
    HCX.hypercomplex_multiply(
        grad,
        HCX.hypercomplex_conjugate(op.inputs[1])),
    HCX.hypercomplex_multiply(
        HCX.hypercomplex_conjugate(op.inputs[0]),
        grad)]


def _hcx_dot(a, b):
    """ 
    Compute the dot product across the last dim of a and first dim of b, using the hypercomplex product

    EX:
    a  - shape(1, 4)
    b  - shape(4, 16)

    RETURNS:
    p  - shape(1, 16)
    """
    if a.shape[1] != b.shape[0]:
        print("Invalid Argument, received dims", a.shape[1], "and", b.shape[0], "do not match")
        return None
    a, b = tf.tile(
        tf.expand_dims(a, 1),
        [1, int(b.shape[1]), 1]), tf.transpose(
            tf.tile(
                tf.expand_dims(b, 2),
                [1, 1, int(a.shape[0])]))
    return tf.reshape(
        tf.reduce_sum(HCX.hypercomplex_multiply(a, b), axis=2),
        [int(a.shape[0]), int(b.shape[1])])


HCX.dot = _hcx_dot


def initialize_weights_cpu(
        name,
        shape,
        standard_deviation=0.01,
        decay_factor=None,
        collection=None):
    with tf.device("/cpu:0"):
        weights = tf.get_variable(
            name,
            shape,
            initializer=tf.truncated_normal_initializer(
                stddev=standard_deviation,
                dtype=tf.float32),
            dtype=tf.float32)
    if decay_factor is not None and collection is not None:
        weight_decay = tf.multiply(
            tf.nn.l2_loss(weights),
            decay_factor)
        tf.add_to_collection(collection, weight_decay)
    return weights


def initialize_biases_cpu(
        name,
        shape):
    with tf.device("/cpu:0"):
        biases = tf.get_variable(
            name,
            shape,
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32)
    return biases


with tf.Graph().as_default():


    weights_one = initialize_weights_cpu("weights_one", [M, M])
    weights_two = initialize_weights_cpu("weights_two", [M, M])
    weights_three = initialize_weights_cpu("weights_three", [M, M])
    biases_one = initialize_biases_cpu("biases_one", [1, M])
    biases_two = initialize_biases_cpu("biases_two", [1, M])
    biases_three = initialize_biases_cpu("biases_three", [1, M])
    input_tensor = tf.placeholder(tf.float32, shape=[1, M], name="input_tensor")
    label_tensor = tf.placeholder(tf.float32, shape=[1, M], name="label_tensor")


    layer_one = tf.nn.sigmoid(
        HCX.dot(
            input_tensor,
            weights_one) + biases_one)
    layer_two = tf.nn.sigmoid(
        HCX.dot(
            layer_one,
            weights_two) + biases_two)
    layer_three = HCX.dot(
        layer_two,
        weights_three) + biases_three
    output_tensor = tf.nn.softmax(
        layer_three,
        dim=1,
        name="output_tensor")


    loss_tensor = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=layer_three,
            labels=label_tensor,
            dim=1),
        name="loss_tensor")
    backprop_op = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE).minimize(
            loss_tensor,
            var_list=[
                weights_one,
                weights_two,
                weights_three,
                biases_one,
                biases_two,
                biases_three
            ], name="backprop")
    init_op = tf.global_variables_initializer()


    data_in = np.random.normal(0, 1, (1, M))
    data_out = np.random.normal(0, 1, (1, M))


    with tf.Session() as session:
        session.run(init_op)
        for i in range(1000):
            loss, _ = session.run(
                [loss_tensor, backprop_op],
                feed_dict={
                    input_tensor: data_in,
                    label_tensor: data_out
                })
            print(loss)