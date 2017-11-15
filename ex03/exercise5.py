import numpy as np
from matplotlib import pyplot as plt
from cifar_helper import CIFAR

import tensorflow as tf
SEED = 5
np.random.seed(SEED)

conv_n = 0
fc_n = 0
def conv_layer(input, kshape, strides=(1, 1, 1, 1)):
    global conv_n
    conv_n += 1
    with tf.variable_scope('conv%d' % conv_n):
        kernels = tf.Variable(tf.truncated_normal(kshape, stddev=0.1, seed=SEED),
                kshape)
        bias_shape = (kshape[-1],)
        biases = tf.Variable(tf.truncated_normal(bias_shape, stddev=0.1, seed=SEED))
        conv = tf.nn.conv2d(input, kernels, strides, padding='SAME', name='conv')
        activation = tf.nn.tanh(conv + biases, name='activation')
        return activation

def fully_connected(input, n_out, with_activation=False):
    global fc_n
    fc_n += 1
    with tf.variable_scope('fully%d' % fc_n):
        init = tf.truncated_normal_initializer(stddev=0.1, seed=SEED)
        W = tf.get_variable(
                'weights',
                initializer=init,
                shape=(input.shape[-1], n_out),
                dtype=tf.float32
                )
        bias = tf.get_variable('bias', initializer=init, shape=(n_out,))
        drive = tf.matmul(input, W) + bias
        if with_activation:
            return tf.nn.tanh(drive)
        else:
            return drive

def train(batch_size=500, learning_rate=1e-4, epochs=10, record_step=20):

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input')
    l = tf.placeholder(dtype=tf.uint8, shape=(None, 1), name='labels')
    l_one_hot = tf.squeeze(tf.one_hot(l, 10), axis=1)

    kernel_shape1 = (5, 5, 3, 16)
    activation1 = conv_layer(x, kernel_shape1)

    pool1 = tf.nn.max_pool(activation1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

    kernel_shape2 = (3, 3, 16, 32)
    activation2 = conv_layer(pool1, kernel_shape2)

    pool2 = tf.nn.max_pool(activation2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

    pool2_reshaped = tf.reshape(pool2, (-1, 2048), name='reshaped1')
    fc1 = fully_connected(pool2_reshaped, 512, with_activation=True)

    fc2_logit = fully_connected(fc1, 10)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2_logit,
            labels=l_one_hot)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_cross_entropy)

    # check if neuron firing strongest coincides with max value position in real
    # labels
    correct_prediction = tf.equal(tf.argmax(fc2_logit, 1), tf.argmax(l_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cifar = CIFAR()
    N, _, _= cifar.get_sizes()
    n_propagations = (N // batch_size)
    if N % batch_size != 0:
        n_propagations += 1
    n_entropies = n_propagations * epochs
    entropies = np.zeros(n_entropies, dtype=np.float32)
    n_accuracies = n_entropies // record_step
    if n_entropies % record_step != 0:
        n_accuracies += 1
    accuracies = np.zeros(n_accuracies, dtype=np.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        propagation = 0
        accu_counter = 0
        for epoch in range(epochs):
            print('Starting epoch %d' % epoch)
            for data, labels in cifar.get_training_batch(batch_size):
                entropy, _ = sess.run([mean_cross_entropy, train_step],
                        feed_dict={x: data, l: labels[:, np.newaxis]})
                entropies[propagation] = entropy
                if propagation % record_step == 0:
                    test_acc = sess.run([accuracy], feed_dict={x:
                        cifar._test_data, l: cifar._test_labels[:,
                            np.newaxis]})
                    accuracies[accu_counter] = test_acc[0]
                    accu_counter += 1
                    print('Current test accuracy %f' % test_acc[0])
                propagation += 1
    return entropies, accuracies

def main():
    entropies, accuracies = train(epochs=5)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_title('Mean entropy & test accuracy')
    ax.set_xlabel('Training step')
    ax.plot(np.linspace(0, len(entropies), num=len(accuracies)), accuracies,
            label='accuracies')
    ax.plot(entropies, label='entropies')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
