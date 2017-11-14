import numpy as np
from matplotlib import pyplot as plt
from cifar_helper import CIFAR

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input')
kernel_shape = (5, 5, 3, 16)
bias_shape = (5, 5, 3, 16)
kernels1 = tf.Variable(tf.truncated_normal(kernel_shape), kernel_shape)
biases1 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=(16,)))
conv1 = tf.nn.conv2d(x, kernels1, (1, 1, 1, 1), 'SAME')
activation1 = tf.nn.tanh(conv1 + biases1)
