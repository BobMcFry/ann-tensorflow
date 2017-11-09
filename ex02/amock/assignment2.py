import tensorflow as tf
import numpy as np
from ext.mnist_helper import MNIST
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mini_batch_size = 100
num_mini_batches = 500
num_epochs = 3

FLAGS = dict()
FLAGS['summaries_dir'] = './tmp/mnist'

if __name__ == "__main__":
    # Tensorflow mnist dataset handler
    mnist = input_data.read_data_sets('res', one_hot=True)

    # TENSORFLOW DEFINING NETWORK START
    x = tf.placeholder(tf.float32, shape=[None, 784])
   
    y_ = tf.placeholder(tf.float32, shape=[None, 10])


    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.zeros([784, 10]))
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([10]))
        with tf.name_scope('Wx_plus_b'):
            y = tf.matmul(x,W) + b
    
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)

    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()


    # TENSORFLOW DEFINING NETWORK END

    # create mini batches
    mini_batches = []
    for i in range(num_mini_batches):
        mini_batches.append(mnist.train.next_batch(mini_batch_size))

    # training with mini batches, num_epochs times
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        for e in range(num_epochs):
            for i in range(num_mini_batches):
                batch = mini_batches[i]
                summary, acc = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1]})
                
                train_writer.add_summary(summary, e*i)

    # show graphs with 'tensorboard --logdir=.' 

