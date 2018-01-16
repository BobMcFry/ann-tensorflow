import sys; sys.path.insert(0, '..')
import tensorflow as tf
from util import batch_norm_layer
from mnist_gan_layers import feed_forward_layer, transposed_conv_layer, conv_layer


class GAN:
    '''Generator network class.'''

    def __init__(self, batch_size=32):
        ############################################################################################
        #                                        GENERATOR                                         #
        ############################################################################################
        with tf.variable_scope('generator'):
            self.input       = tf.placeholder(tf.float32, shape=(batch_size, 50))
            self.is_training = tf.placeholder(tf.bool, shape=[])
            expanded_input   = feed_forward_layer(self.input, 64 * 4 * 4, self.is_training,
                                                normalize=True, activation_function=tf.nn.relu)
            with tf.variable_scope('layer1'):
                layer_1   = tf.reshape(expanded_input, (-1, 4, 4, 64))

            with tf.variable_scope('layer2'):
                layer_2 = transposed_conv_layer(layer_1, (5, 5, 32, 64), (batch_size, 7, 7, 32),
                                                (1, 2, 2, 1), self.is_training, normalize=True,
                                                activation_function=tf.nn.relu)
            with tf.variable_scope('layer3'):
                layer_3 = transposed_conv_layer(layer_2, (5, 5, 16, 32), (batch_size, 14, 14, 16),
                                                (1, 2, 2, 1), self.is_training, normalize=True,
                                                activation_function=tf.nn.relu)

            with tf.variable_scope('layer4'):
                layer_4 = transposed_conv_layer(layer_3, (5, 5, 1, 16), (batch_size, 28, 28, 1),
                                                (1, 2, 2, 1), self.is_training, normalize=False,
                                                activation_function=tf.nn.tanh)
            self.gen_output = layer_4

        with tf.variable_scope('discriminator'):
            self.input_reals = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
            inputs = tf.concat([self.gen_output, self.input_reals], 0)
            self.labels = tf.concat([tf.ones((batch_size, 1), tf.float32),
                                    tf.zeros((batch_size, 1), tf.float32)], 0)
            act_fn = tf.nn.leaky_relu
            with tf.variable_scope('layer1'):
                conv1 = conv_layer(inputs, 8, 5, 2, activation_function=act_fn, normalize=False)
            with tf.variable_scope('layer2'):
                conv2 = conv_layer(conv1, 16, 5, 2, activation_function=act_fn, normalize=False)
            with tf.variable_scope('layer3'):
                conv3 = conv_layer(conv2, 32, 5, 2, activation_function=act_fn, normalize=False)
            with tf.variable_scope('layer4'):
                self.dis_output = feed_forward_layer(tf.reshape(conv3, shape=(batch_size * 2, 4 * 4 * 32)),
                                                1,
                                                self.is_training, normalize=False,
                                                activation_function=tf.nn.sigmoid)

            entropy_dis = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.dis_output)
            self.loss_dis = tf.reduce_mean(entropy_dis)

            entropy_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones((batch_size, 1), tf.float32), logits=self.dis_output[batch_size:]

            )
            self.loss_gen = tf.reduce_mean(entropy_gen)

    def feed(self, session, vector, real_images, is_training):
        return session.run(self.dis_output, feed_dict={self.input: vector, self.input_reals:
                                                       real_images, self.is_training: is_training})



def main():
    from mnist_gan_helper import MNIST_GAN
    import numpy as np

    mnist_helper = MNIST_GAN('data')

    batch_size   = 32
    gan = GAN(batch_size)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for batch in mnist_helper.get_batch(batch_size):
            print('Feeding')
            z = np.random.uniform(-1, 1, size=(batch_size, 50))
            gan.feed(session, z, batch, True)


if __name__ == "__main__":
    main()
