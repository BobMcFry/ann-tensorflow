import numpy as np
from matplotlib import pyplot as plt
if __name__ == "__main__":
    # relative import does not work when running as a script (w/o hacks)
    from cifar_helper import CIFAR
else:
    # if used as module, do this
    from .cifar_helper import CIFAR

import tensorflow as tf
SEED = 5
np.random.seed(SEED)

# counter for autmatically creating conv layer variable names
conv_n = 0

def conv_layer(input, kshape, strides=(1, 1, 1, 1)):
    '''Create a convolutional layer with fixed activation function and variable
    initialisation. The activation function is ``tf.nn.tanh`` and variables are
    initialised from a truncated normal distribution with an stddev of 0.1

    Parameters
    ----------
    input   :   tf.Variable
                Input to the layer
    kshape  :   tuple or list
                Shape of the kernel tensor
    strides :   tuple or list
                Strides

    Returns
    -------
    tf.Variable
            The variable representing the layer activation (tanh(conv + bias))

    '''
    global conv_n
    conv_n += 1
    # this adds a prefix to all variable names
    with tf.variable_scope('conv%d' % conv_n):
        kernels = tf.Variable(tf.truncated_normal(kshape, stddev=0.1, seed=SEED),
                kshape)
        bias_shape = (kshape[-1],)
        biases = tf.Variable(tf.truncated_normal(bias_shape, stddev=0.1, seed=SEED))
        conv = tf.nn.conv2d(input, kernels, strides, padding='SAME', name='conv')
        activation = tf.nn.tanh(conv + biases, name='activation')
        return activation

# counter for autmatically creating fully-connected layer variable names
fc_n = 0
def fully_connected(input, n_out, with_activation=False):
    '''Create a fully connected layer with fixed activation function and variable
    initialisation. The activation function is ``tf.nn.tanh`` and variables are
    initialised from a truncated normal distribution with an stddev of 0.1

    Parameters
    ----------
    input   :   tf.Variable
                Input to the layer
    n_out   :   int
                Number of neurons in the layer
    with_activation :   bool
                        Return activation or drive (useful when planning to use
                        ``softmax_cross_entropy_with_logits`` which requires
                        unscaled logits)


    Returns
    -------
    tf.Variable
            The variable representing the layer activation (tanh(input * Weights
            + bias))
    '''
    global fc_n
    fc_n += 1
    with tf.variable_scope('fully%d' % fc_n):
        init = tf.truncated_normal_initializer(stddev=0.1, seed=SEED)
        W = tf.get_variable(
                'weights',
                initializer=init,
                shape=(input.shape[-1], n_out),     # the last dim of the input
               dtype=tf.float32                     # is the first dim of the weights2
            )
        bias = tf.get_variable('bias', initializer=init, shape=(n_out,))
        drive = tf.matmul(input, W) + bias
        if with_activation:
            return tf.nn.tanh(drive)
        else:
            return drive

def train(batch_size=500, learning_rate=1e-4, epochs=10, record_step=20,
        return_records=False, optimizer='GradientDescent'):
    '''Train the fixed graph on CIFAR-10.

    Parameters
    ----------
    batch_size  :   int
                    Size of training batch
    learning_rate   :   float
                        Learning rate for the ADAM optimizer
    epochs          :   int
                        Number of times to visit the entire training set
    record_step     :   int
                        Accuracy on test set will be recorded every
                        ``record_step`` training steps
    return_records  :   bool
                        If False, return only final test set accuracy, otherwise
                        return vectors of entropies and accuracies (according to
                        record_step)
    optimizer   :   str
                    Name of the optimizer to use

    Returns
    -------
    float or tuple
            Final accuracies or array of cross entropies and array of test
            accuracies in a tuple
    '''

    assert batch_size > 0, 'Batch size must be positive'
    assert learning_rate > 0, 'Learning rate must be positive'
    assert epochs > 0, 'Number of epochs must be positive'
    assert record_step > 0, 'Recording step must be positive'
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
    optimizer_obj = getattr(tf.train, optimizer + 'Optimizer')
    print('Using %s optimizer' % optimizer)
    train_step = optimizer_obj(learning_rate=learning_rate).minimize(mean_cross_entropy)

    # check if neuron firing strongest coincides with max value position in real
    # labels
    correct_prediction = tf.equal(tf.argmax(fc2_logit, 1), tf.argmax(l_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cifar = CIFAR()
    entropies = []
    accuracies = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        propagation = 0
        for epoch in range(epochs):
            print('Starting epoch %d' % epoch)
            for data, labels in cifar.get_training_batch(batch_size):
                entropy, _ = sess.run([mean_cross_entropy, train_step],
                        feed_dict={x: data, l: labels[:, np.newaxis]})
                if return_records:
                    entropies.append(entropy)
                    if propagation % record_step == 0:
                        test_acc = sess.run([accuracy], feed_dict={x:
                            cifar._test_data, l: cifar._test_labels[:,
                                np.newaxis]})
                        accuracies.append(test_acc[0])
                        print('Current test accuracy %f' % test_acc[0])
                    propagation += 1

        final_accuracy = sess.run([accuracy], feed_dict={x:
                        cifar._test_data, l: cifar._test_labels[:,
                            np.newaxis]})
    if return_records:
        if propagation % record_step == 1:  #   we just recorded
            pass
        else:
            accuracies.append(final_accuracy[0])
        return entropies, accuracies
    else:
        return final_accuracy[0]

def main():
    entropies, accuracies = train(batch_size=1000, epochs=3, return_records=True)
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
