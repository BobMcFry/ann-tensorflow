import sys; sys.path.insert(0, '..')

import tensorflow as tf
import numpy as np

import argparse
from util import fully_connected

from mnist_gan_helper import MNIST_GAN
from mnist_gan_layers import *


def deconv3d(prev_layer, output_shape, strides, padding=0, name = "filter", activation_function=tf.nn.relu):
    # Deconv layer
    w = tf.Variable(tf.random_normal([64, 4, 4, 1, 1]), name='w_'+name)
    b = tf.Variable(tf.zeros([1]), name=('b_'+name) )

    if padding == 0:
        deconv = tf.nn.conv3d_transpose(prev_layer, w, output_shape=output_shape, strides=strides, padding="SAME")
    else:
        deconv = tf.nn.conv3d_transpose(prev_layer, w, output_shape=output_shape, strides=strides, padding="VALID")
    deconv = tf.nn.bias_add(deconv, b)
    deconv = activation_function(deconv)
    return deconv

class Generator():
    def __init__(self, learning_rate = 0.01, debug=True):


        ## 1 network definition ##
        self.z_vec = tf.placeholder(tf.float32, shape=[1, 50], name='zvector')
        hidden_layer_1 = feed_forward_layer(self.z_vec,
            target_size = 1024,
            normalize = False,
            activation_function = tf.nn.relu)

        if debug:
            print("Forward-Layer: " + str(self.z_vec.shape) + " -> " + str(hidden_layer_1.shape))

        hidden_layer_1 = tf.reshape(hidden_layer_1, [64,4,4,1,1])

        if debug:
            print("Reshape: " + str(hidden_layer_1.shape) + " -> " + str(hidden_layer_1.shape) )

        hidden_layer_2 = deconv3d(
                prev_layer = hidden_layer_1,
                output_shape = [32,7,7,1,1],
                strides = [1,1,1,1,1],
                name = "conv1",
                activation_function = tf.nn.relu
            )

        if debug:
            print("conv3d transpose 1: " + str(hidden_layer_1.shape) + " -> " + str(hidden_layer_2.shape) )

        hidden_layer_3 = deconv3d(
                prev_layer = hidden_layer_2,
                output_shape = [16,14,14,1,1],
                strides = [1,1,1,1,1],
                name = "conv2",
                activation_function = tf.nn.relu
            )

        if debug:
            print("conv3d transpose 2: " + str(hidden_layer_2.shape) + " -> " + str(hidden_layer_3.shape) )

        hidden_layer_4 = deconv3d(
                prev_layer = hidden_layer_3,
                output_shape = [1,28,28,1,1],
                strides = [1,1,1,1,1],
                name = "conv3",
                activation_function = tf.nn.tanh
            )

        if debug:
            print("conv3d transpose 3: " + str(hidden_layer_3.shape) + " -> " + str(hidden_layer_4.shape) )


        ## 1 network definition ##


def main():
    print("Test")
    gen = Generator(learning_rate = 0.01)



if __name__ == '__main__':
    main()

