from matplotlib import pyplot as plt
import tensorflow as tf
# truncated normal distribution discards values more than 2 stds off-mean
from tensorflow import truncated_normal_initializer
import numpy as np
from mnist import MNISTLoader


def train(n_epochs=10, batch_size=50):
    '''Run training.

    Parameters
    ----------
    n_epochs    :   int
                    Number of times to visit the entire dataset
    batch_size  :   int
                    Size of the minibatches

    Returns
    -------
    tuple
            tuple of 5 np.ndarrays.

                - List of accuracies over entire training set, collected every
                10th batch
                - List of accuracies over entire test set, collected every 10th
                batch
                - List of average cross entropies over entire training set,
                collected every 10th batch
                - List of average cross entropie over entire test set, collected
                every 10th batch
                - List of weights collected every 10th batch

    '''
    # load the data
    loader = MNISTLoader()
    d_train, l_train, d_test, l_test = (loader.training_data,
                                        loader.training_labels,
                                        loader.test_data, loader.test_labels)

    ##########################################################################
    #               The data comes in image format, which we flatten         #
    ##########################################################################
    # keep the first dim as it is
    d_test = np.reshape(d_test, (-1, 28 * 28))
    d_train = np.reshape(d_train, (-1, 28 * 28))
    ##########################################################################
    #        The labels only have 1 dimension, we need to blow it up to 2    #
    ##########################################################################
    l_test = l_test[:, np.newaxis]
    l_train = l_train[:, np.newaxis]

    # Weight matrix
    mean = 0.0
    std = 0.000002
    W = tf.get_variable(
        'weights',
        initializer=truncated_normal_initializer(mean, std, seed=1),
        shape=[28 * 28, 10]
    )

    # bias vector
    b = tf.get_variable('bias', initializer=tf.zeros_initializer(), shape=[10])

    # data vector
    x = tf.placeholder(tf.float32, [None, 28 * 28], name='input')

    # desired output (ie real labels)
    d = tf.placeholder(tf.int32, [None, 1], name='labels')
    # one-hot encoding produuces a vecor of shape (batch, 1, 10) instead of
    # (batch, 10)
    d_1_hot = tf.squeeze(tf.one_hot(d, 10), axis=1)

    # computed output of the network without activation
    y = tf.matmul(x, W) + b

    # loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=y, labels=d_1_hot)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    training_step = optimizer.minimize(cross_entropy)

    # check if neuron firing strongest coincides with max value position in real
    # labels
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d_1_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # record accuracy
    training_step_accuracy = []
    test_step_accuracy = []

    # record cross-entropy
    training_step_entropy = []
    test_step_entropy = []

    # record weights
    weights = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 0
        for epoch in range(n_epochs):
            print('Epoch %d' % epoch)
            for mb, labels in loader.batches(d_train, l_train, batch_size):
                # pass a dict to later retrieve tensors by name. We need the
                # weigweights only for the record
                values = sess.run(
                    {'weights': W, 'step': training_step},
                    feed_dict={x: mb, d: labels})
                if i % 10 == 0:
                    # run the ops that will give us accuracies and entropies
                    # (not needed otherwise)
                    current_train_accuracy = sess.run(
                        accuracy, feed_dict={x: d_train, d: l_train})
                    current_test_accuracy = sess.run(
                        accuracy, feed_dict={x: d_test, d: l_test})

                    training_step_accuracy.append(current_train_accuracy)
                    test_step_accuracy.append(current_test_accuracy)

                    current_train_entropy = sess.run(
                        mean_cross_entropy, feed_dict={
                            x: d_train, d: l_train})
                    current_test_entropy = sess.run(
                        mean_cross_entropy, feed_dict={
                            x: d_test, d: l_test})

                    training_step_entropy.append(current_train_entropy)
                    test_step_entropy.append(current_test_entropy)

                    weights.append(np.reshape(values['weights'], (28, 28, 10)))

                # increment batch counter
                i += 1

    return (training_step_accuracy, test_step_accuracy, training_step_entropy,
            test_step_entropy, weights)


if __name__ == "__main__":
    training_accuracy, test_accuracy, training_entropy, test_entropy, weights = train(
        3, 500)
    # Problem: We append the accuray every 10th step, so we may miss the last
    # one
    print('(Almost) final test accuracy: %f' % test_accuracy[-1])
    ############################################################################
    #                         Plot entropy and accuray                         #
    ############################################################################
    f = plt.figure()
    ax_acc = f.add_subplot(121)
    ax_acc.set_title('Accuracy over training and test sets')
    ax_acc.set_xlabel('(n*10)th batch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.plot(test_accuracy, linestyle=':', label='Test set')
    ax_acc.plot(training_accuracy, linestyle=':', label='Training set')
    ax_acc.legend()

    ax_entropy = f.add_subplot(122)
    ax_entropy.set_title('Cross Entropy over training and test sets')
    ax_entropy.set_xlabel('(n*10)th batch')
    ax_entropy.set_ylabel('Cross Entropy')
    ax_entropy.plot(test_entropy, linestyle=':', label='Test set')
    ax_entropy.plot(training_entropy, linestyle=':', label='Training set')
    ax_entropy.legend()
    plt.show()

    ############################################################################
    #                        Plot weights interactively                        #
    ############################################################################
    rows, cols = (2, 5)
    f2, axarr = plt.subplots(rows, cols)
    plt.ion()
    for row in range(2):
        for col in range(5):
            ax = axarr[row][col]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    for i in range(len(weights)):
        current_weights = weights[i]
        for row in range(2):
            for col in range(5):
                f2.suptitle('Step %d of %d' % (i, len(weights)))
                ax = axarr[row][col]
                ax.cla()
                index = row * rows + col
                ax.set_title('Neuron %d' % index)
                # there's many diverging cmaps
                # (https://matplotlib.org/examples/color/colormaps_reference.html)
                ax.imshow(current_weights[..., index], cmap='Spectral')

        # pause so that it always takes 5 seconds
        # Note: The animation seems to slow down linearly, unless we clear the
        # axes (see above)
        plt.pause(3 / len(weights))
