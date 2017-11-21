import tensorflow as tf
from util import conv_layer, fully_connected, weighted_pool_layer, inception2d
from .svhn_helper import SVHN
import numpy as np

def train(batch_size=500, learning_rate=1e-4, epochs=10, record_step=20,
          return_records=False, optimizer='GradientDescent',
          activation=tf.nn.relu):
    svhn = SVHN()

    ############################################################################
    #                             Define the graph                             #
    ############################################################################
    # It turns out that this network from ex03 is already capable of memorizing
    # the entire training or validation set, so we need to tweak generalization,
    # not capacity
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name='input')
    y_ = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')

    kernel_shape1 = (5, 5, 1, 16)
    activation1 = conv_layer(x, kernel_shape1, activation=activation)

    pool1 = weighted_pool_layer(
        activation1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)
    )

    kernel_shape2 = (3, 3, 16, 32)
    activation2 = conv_layer(pool1, kernel_shape2, activation=activation)

    pool2 = weighted_pool_layer(
        activation2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)
    )

    pool2_reshaped = tf.reshape(pool2, (-1, 2048), name='reshaped1')
    fc1 = fully_connected(pool2_reshaped, 512, with_activation=True,
            activation=activation)

    fc2_logit = fully_connected(fc1, 10, activation=activation)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2_logit,
                                                            labels=y_)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    optimizer_obj = getattr(tf.train, optimizer + 'Optimizer')
    print('Using %s optimizer' % optimizer)
    train_step = optimizer_obj(
        learning_rate=learning_rate).minimize(mean_cross_entropy)

    # check if neuron firing strongest coincides with max value position in real
    # labels
    correct_prediction = tf.equal(tf.argmax(fc2_logit, 1, output_type=tf.int32), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    entropies = []
    accuracies = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        propagation = 0
        for epoch in range(epochs):
            print('Starting epoch %d' % epoch)
            for data, labels in svhn.get_training_batch(batch_size):
                entropy, _ = sess.run(
                    [mean_cross_entropy, train_step],
                    feed_dict={x: data, y_: labels})
                if return_records:
                    entropies.append(entropy)
                if propagation % record_step == 0:
                    test_acc = sess.run(
                        [accuracy],
                        feed_dict={x: svhn._validation_data,
                                   y_: svhn._validation_labels})
                    if return_records:
                        accuracies.append(test_acc[0])
                    print('Current training accuracy %f' % test_acc[0])
                propagation += 1

        final_accuracy = sess.run(
            [accuracy],
            feed_dict={x: svhn._validation_data, y_: svhn._validation_labels})
    if return_records:
        if propagation % record_step == 1:  # we just recorded
            pass
        else:
            accuracies.append(final_accuracy[0])
        return entropies, accuracies
    else:
        return final_accuracy[0]
