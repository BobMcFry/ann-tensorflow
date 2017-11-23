from util import conv_layer, fully_connected, weighted_pool_layer, inception2d
from model import BaseModel
import tensorflow as tf

class Model(BaseModel):
    '''This is the model from exercise 3 but with miced pooling layers. This
    improves accuracy by ~3%.'''

    def run_training_step(self, session, data, labels):
        entropy, _ = session.run(
            [self.mean_cross_entropy, self.train_step],
            feed_dict={self.x: data, self.y_: labels})
        return entropy

    def get_accuracy(self, session, data, labels):
        return session.run([self.accuracy], feed_dict={self.x: data, self.y_:
            labels})[0]


    def __init__(self, learning_rate, optimizer, activation):
        super().__init__(learning_rate, optimizer, activation)

        ############################################################################
        #                             Define the graph                             #
        ############################################################################
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name='input')
        y_ = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')

        self.x = x
        self.y_ = y_

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
        self.mean_cross_entropy = mean_cross_entropy
        optimizer_obj = getattr(tf.train, optimizer + 'Optimizer')
        train_step = optimizer_obj(
            learning_rate=learning_rate).minimize(mean_cross_entropy)
        self.train_step = train_step

        # check if neuron firing strongest coincides with max value position in real
        # labels
        correct_prediction = tf.equal(tf.argmax(fc2_logit, 1, output_type=tf.int32), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = accuracy


