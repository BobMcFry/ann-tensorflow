import tensorflow as tf
import numpy as np
from mnist import MNISTLoader

# load the data
loader = MNISTLoader()
d_train, l_train, d_test, l_test = (loader.training_data, loader.training_labels,
        loader.test_data, loader.test_labels)

d_test = np.reshape(d_test, (-1, 28 * 28))
d_train = np.reshape(d_train, (-1, 28 * 28))
l_test = l_test[:, np.newaxis]
l_train = l_train[:, np.newaxis]

batch_size = 1000

# Weight matrix
W = tf.get_variable('weights', initializer=tf.random_normal_initializer(0.0,
    0.000002, seed=1), shape=[28 * 28, 10])

# bias vector
b = tf.get_variable('bias', initializer=tf.zeros_initializer(), shape=[10])

# data vector
x = tf.placeholder(tf.float32, [None, 28 * 28], name='input')

# desired output (ie real labels)
d = tf.placeholder(tf.int32, [None, 1], name='labels')
d_1_hot = tf.squeeze(tf.one_hot(d, 10), axis=1)

# computed output of the network without activation
y = tf.matmul(x, W) + b

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=d_1_hot)
optimizer     = tf.train.GradientDescentOptimizer(learning_rate=0.5)
training_step = optimizer.minimize(cross_entropy)

# check if neuron firing strongest coincides with max value position in real
# labels
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d_1_hot, 1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

training_step_accuracy = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Training')
    n_epochs = 3

    for _ in range(n_epochs):
        for mb, labels in loader.batches(d_train, l_train, batch_size=batch_size):
            sess.run(training_step, feed_dict={x: mb, d: labels})
        current_accuracy = sess.run(accuracy, feed_dict={x: d_train, d: l_train})
        print('Current accuracy: %f' % current_accuracy)
        training_step_accuracy.append(current_accuracy)

    print("Final test accuracy: %f" % sess.run(accuracy, feed_dict={x:
        d_test, d: l_test}))
    print("min/max training accuracy: %f/%f" %
            (np.min(training_step_accuracy), np.max(training_step_accuracy)))

