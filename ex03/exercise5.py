import numpy as np
from matplotlib import pyplot as plt
from cifar_helper import CIFAR

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input')
l = tf.placeholder(dtype=tf.uint8, shape=(None, 1), name='labels')
l_one_hot = tf.squeeze(tf.one_hot(l, 10), axis=1)

kernel_shape1 = (5, 5, 3, 16)
bias_shape1 = (16,)
kernels1 = tf.Variable(tf.truncated_normal(kernel_shape1, stddev=0.1, seed=10), kernel_shape1)
biases1 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=bias_shape1))
conv1 = tf.nn.conv2d(x, kernels1, (1, 1, 1, 1), 'SAME', name='conv1')
activation1 = tf.nn.tanh(conv1 + biases1, name='activation1')

pool1 = tf.nn.max_pool(activation1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
        padding='SAME')

kernel_shape2 = (3, 3, 16, 32)
bias_shape2 = (32,)
kernels2 = tf.Variable(tf.truncated_normal(kernel_shape2, stddev=0.1, seed=10), kernel_shape2)
biases2 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=bias_shape2))
conv2 = tf.nn.conv2d(pool1, kernels2, (1, 1, 1, 1), 'SAME', name='conv2')
activation2 = tf.nn.tanh(conv2 + biases2, name='activation2')

pool2 = tf.nn.max_pool(activation2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
        padding='SAME')

pool2_reshaped = tf.reshape(pool2, (-1, 2048), name='reshaped1')
fc_weights1 = tf.get_variable(
        'fully-connected-1-weights',
        initializer=tf.truncated_normal_initializer(),
        shape=(8 * 8 * 32, 512),
        dtype=tf.float32
      )
fc_bias1 = tf.get_variable('fully-connected-1-bias',
        initializer=tf.zeros_initializer(), shape=(512,))
fc1 = tf.nn.tanh(tf.matmul(pool2_reshaped, fc_weights1) + fc_bias1)

fc_weights2 = tf.get_variable(
        'fully-connected-2-weights',
        initializer=tf.truncated_normal_initializer(),
        shape=(512, 10),
        dtype=tf.float32
      )
fc_bias2 = tf.get_variable('fully-connected-2-bias',
        initializer=tf.zeros_initializer(), shape=(10,))
fc2_logit = tf.matmul(fc1, fc_weights2) + fc_bias2

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2_logit,
        labels=l_one_hot)
mean_cross_entropy = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(mean_cross_entropy)

# check if neuron firing strongest coincides with max value position in real
# labels
correct_prediction = tf.equal(tf.argmax(fc2_logit, 1), tf.argmax(l_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 3
    cifar = CIFAR()
    for epoch in range(epochs):
        print('Starting epoch %d' % epoch)
        for index, (data, labels) in enumerate(cifar.get_training_batch(100)):
            sess.run([train_step],
                    feed_dict={x: data, l: labels[:, np.newaxis]})
            if index % 20 == 0:
                test_acc = sess.run([accuracy], feed_dict={x:
                    cifar._training_data[:1000], l: cifar._training_labels[:,
                        np.newaxis][:1000]})
                print('Current train accuracy %f' % test_acc[0])
