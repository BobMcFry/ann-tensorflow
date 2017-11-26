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

        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name='input')
        y_ = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')

        self.x = x
        self.y_ = y_
        # logits = (LinearWrap(image)
        #               .Conv2D('conv1', 24, 5, padding='VALID')
        #               .MaxPooling('pool1', 2, padding='SAME')
        #               .Conv2D('conv2', 32, 3, padding='VALID')
        #               .Conv2D('conv3', 32, 3, padding='VALID')
        #               .MaxPooling('pool2', 2, padding='SAME')
        #               .Conv2D('conv4', 64, 3, padding='VALID')
        #               .Dropout('drop', 0.5)
        #               .FullyConnected('fc0', 512,
        #                               b_init=tf.constant_initializer(0.1), nl=tf.nn.relu)
        #               .FullyConnected('linear', out_dim=10, nl=tf.identity)())
        # tf.nn.softmax(logits, name='output')


        l1 = conv_layer(x, (5, 5, 1, 24), activation=None,
                padding='VALID', use_bias=False)
        l2 = tf.nn.relu(batch_norm_layer(l1))

        l3 = tf.nn.max_pool(l2, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

        l4 = conv_layer(l3, (3, 3, 24, 32), padding='VALID',
                activation=None, use_bias=False)
        l5 = tf.nn.relu(batch_norm_layer(l4))

        l6 = conv_layer(l5, (3, 3, 32, 32), padding='VALID',
                activation=None, use_bias=False)
        l7 = tf.nn.relu(batch_norm_layer(l6))

        l8 = tf.nn.max_pool(l7, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

        l8_ = conv_layer(l8, (3, 3, 32, 64), padding='VALID',
                activation=None, use_bias=False)
        l9 = tf.nn.relu(batch_norm_layer(l8_))

        l10 = tf.nn.dropout(l9, 0.5)

        l11 = tf.reshape(l10, (-1, 3*3*64), name='reshaped1')

        l12 = fully_connected(l11, 512, with_activation=True,
                activation=tf.nn.relu)

        l13 = fully_connected(l12, 10, with_activation=False, use_bias=False)


        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l13, labels=y_)
        mean_cross_entropy = tf.reduce_mean(cross_entropy)
        self.mean_cross_entropy = mean_cross_entropy
        train_step = optimizer.minimize(mean_cross_entropy)
        self.train_step = train_step
        self.prediction = tf.cast(tf.argmax(l13, 1), tf.int32)

        # check if neuron firing strongest coincides with max value position in real
        # labels
        correct_prediction = tf.equal(self.prediction, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = accuracy


