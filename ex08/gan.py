import sys; sys.path.insert(0, '..')
import tensorflow as tf
from util import batch_norm_layer
from mnist_gan_layers import feed_forward_layer, transposed_conv_layer, conv_layer
from mnist_gan_helper import MNIST_GAN
import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt



class GAN:
    '''.. :py:class::``GAN``

    This class encapsulates both generator and discriminator.

    Attributes
    ----------
    input   :   tf.placeholder
                Placeholder of shape (batch_size, 50) of uniform random values from [-1,1]
    is_training :   tf.placeholder
                    Bool placeholder for correct batch norm during inference
    gen_output  :   tf.Tensor
                    Output of the generator portion
    dis_output  :   tf.Tensor
                    Output of the discriminator portion
    loss_dis    :   Cross-entropy loss of the discriminator
    loss_gen    :   Cross-entropy loss of the generator
    input_reals :   tf.placeholder
                    Placeholder for feeding batch_size real input images
    '''

    def __init__(self, batch_size=32, learning_rate=0.0004):
        ############################################################################################
        #                                        GENERATOR                                         #
        ############################################################################################
        # note that we forego naming the ops since they are usually easily identified by their index
        # and variable scope prefix
        with tf.variable_scope('generator') as scope_gen:
            self.input       = tf.placeholder(tf.float32, shape=(batch_size, 50))
            self.is_training = tf.placeholder(tf.bool, shape=[])

            # blow up to enough neurons
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

            # collect all vars from this scope for the generator
            variables_gen = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_gen.name)

        with tf.variable_scope('discriminator') as scope_dis:
            self.input_reals = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
            inputs           = tf.concat([self.gen_output, self.input_reals], 0)
            labels           = tf.concat([tf.zeros((batch_size, 1), tf.float32),
                                          tf.ones((batch_size, 1), tf.float32)], 0)
            act_fn = tf.nn.leaky_relu
            with tf.variable_scope('layer1'):
                conv1 = conv_layer(inputs, 8, 5, 2, activation_function=act_fn, normalize=False)
            with tf.variable_scope('layer2'):
                conv2 = conv_layer(conv1, 16, 5, 2, activation_function=act_fn, normalize=False)
            with tf.variable_scope('layer3'):
                conv3 = conv_layer(conv2, 32, 5, 2, activation_function=act_fn, normalize=False)
            with tf.variable_scope('layer4'):
                conv3_reshaped  = tf.reshape(conv3, shape=(batch_size * 2, 4 * 4 * 32))
                self.dis_output = feed_forward_layer(conv3_reshaped,
                                                     1,
                                                     self.is_training,
                                                     normalize=False,
                                                     activation_function=tf.nn.sigmoid)

            # collect all vars from this scope for the discriminator
            variables_dis = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_dis.name)

            entropy_dis   = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.dis_output)
            self.loss_dis = tf.reduce_mean(entropy_dis)

            entropy_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones((batch_size, 1), tf.float32), logits=self.dis_output[:batch_size]

            )
            self.loss_gen       = tf.reduce_mean(entropy_gen)
            self.train_step_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(self.loss_gen, var_list=variables_gen)
            self.train_step_dis = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(self.loss_dis, var_list=variables_dis)

    def train_step(self, session, vector, real_images, is_training):
        '''Run one step of training.'''

        fetches = [self.loss_dis, self.loss_gen, self.train_step_dis, self.train_step_gen]
        feeds = {self.input: vector,
                 self.input_reals: real_images,
                 self.is_training: is_training}
        loss_dis, loss_gen, _, _ = session.run(fetches, feed_dict=feeds)
        return loss_dis, loss_gen

    def generate_images(self, session, vectors):
        fetches = self.gen_output
        feeds = {self.input: vectors, self.is_training: False}
        return session.run(fetches, feed_dict=feeds)


def plot_images(imgs):
    n, h, w, c = imgs.shape
    cols = int(np.sqrt(n))
    rows = int(n / cols) + 1
    fig, axarr = plt.subplots(rows, cols)
    for index in range(n):
        row = index // cols
        col = index % cols
        ax = axarr[row][col]
        ax.imshow(imgs[index, :, :, 0], cmap='gray')

    # delete empty plots
    for index in range(n, rows*cols):
        row = index // cols
        col = index % cols
        fig.delaxes(axarr[row][col])

    plt.show()

def main():
    mnist_helper = MNIST_GAN('data')
    epochs       = 5
    batch_size   = 64
    gan          = GAN(batch_size)

    losses_dis = []
    losses_gen = []
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            print(f'Starting epoch {epoch+1}/{epochs}')
            for batch in mnist_helper.get_batch(batch_size):
                z = np.random.uniform(-1, 1, size=(batch_size, 50))
                loss_dis, loss_gen = gan.train_step(session, z, batch, True)
                losses_dis.append(loss_dis)
                losses_gen.append(loss_gen)
        imgs = gan.generate_images(session, np.random.uniform(-1, 1, size=(batch_size, 50)))
        plot_images(imgs)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_title('Entropy of GAN')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Sigmoid cross-entropy')
    ax.plot(losses_dis, label='Discriminator')
    ax.plot(losses_gen, label='Generator')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
