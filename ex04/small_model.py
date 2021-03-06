from util import (conv_layer, fully_connected, weighted_pool_layer,
        batch_norm_layer)
from model import BaseModel
import tensorflow as tf
import numpy as np

class Model(BaseModel):
    '''Smaller model so we clock in at < 4mb'''

    def predict(self, session, data):
        return session.run([self.prediction], feed_dict={self.x: data,
            self.y_: np.zeros(data.shape[0])})[0]

    def run_training_step(self, session, data, labels):
        entropy, _ = session.run(
            [self.mean_cross_entropy, self.train_step],
            feed_dict={self.x: data, self.y_: labels})
        return entropy

    def get_accuracy(self, session, data, labels):
        return session.run([self.accuracy], feed_dict={self.x: data, self.y_:
            labels})[0]


    def __init__(self, optimizer, activation):
        super().__init__(optimizer, activation)

        ############################################################################
        #                             Define the graph                             #
        ############################################################################
        # It turns out that this network from ex03 is already capable of memorizing
        # the entire training or validation set, so we need to tweak generalization,
        # not capacity
        # In order to speed up convergence, we added batch normalization.
        # Our best effort was Adam optimizer with bs=32, lr=0.001 (only possible
        # because of norm)
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name='input')
        y_ = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')

        self.x = x
        self.y_ = y_

        kernel_shape1 = (5, 5, 1, 8)
        activation1 = conv_layer(x, kernel_shape1, activation=activation)

        normalize1 = batch_norm_layer(activation1)

        pool1 = weighted_pool_layer(
            normalize1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)
        )

        kernel_shape2 = (3, 3, 8, 10)
        activation2 = conv_layer(pool1, kernel_shape2, activation=activation)

        normalize2 = batch_norm_layer(activation2)

        pool2 = weighted_pool_layer(
            normalize2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)
        )

        pool2_reshaped = tf.reshape(pool2, (-1, 8*8*10), name='reshaped1')
        fc1 = fully_connected(pool2_reshaped, 512, with_activation=True,
                activation=activation)

        fc2_logit = fully_connected(fc1, 10, activation=activation)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2_logit,
                                                                labels=y_)
        mean_cross_entropy = tf.reduce_mean(cross_entropy)
        self.mean_cross_entropy = mean_cross_entropy
        train_step = optimizer.minimize(mean_cross_entropy)
        self.train_step = train_step
        self.prediction = tf.cast(tf.argmax(fc2_logit, 1), tf.int32)

        # check if neuron firing strongest coincides with max value position in real
        # labels
        correct_prediction = tf.equal(self.prediction, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = accuracy

