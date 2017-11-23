import sys
import argparse
import importlib
import tensorflow as tf
import numpy as np

SEED = 5
np.random.seed(SEED)
tf.set_random_seed(SEED)

pool_n = 0


def max_pool_layer(input, ksize, strides):
    global pool_n
    pool_n += 1
    with tf.variable_scope('pool%d' % pool_n):
        return tf.nn.max_pool(input,
                ksize=ksize, strides=strides, padding='SAME')

conv_n = 0


def conv_layer(input, kshape, strides=(1, 1, 1, 1), activation=tf.nn.tanh):
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
        kernels = tf.Variable(
            tf.truncated_normal(
                kshape,
                stddev=0.1),
            kshape, name='kernels')
        bias_shape = (kshape[-1],)
        biases = tf.Variable(
            tf.truncated_normal(
                bias_shape,
                stddev=0.1), name='bias')
        conv = tf.nn.conv2d(
            input,
            kernels,
            strides,
            padding='SAME',
            name='conv')
        return activation(tf.nn.tanh(conv + biases, name='activation'))


# counter for autmatically creating fully-connected layer variable names
fc_n = 0


def fully_connected(input, n_out, with_activation=False, activation=tf.nn.tanh):
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
        init = tf.truncated_normal_initializer(stddev=0.1)
        W = tf.get_variable(
                'weights',
                initializer=init,
                shape=(input.shape[-1], n_out), # the last dim of the input
               dtype=tf.float32                 # is the 1st dim of the weights
            )
        bias = tf.get_variable('bias', initializer=init, shape=(n_out,))
        drive = tf.matmul(input, W) + bias
        if with_activation:
            return activation(drive)
        else:
            return drive

# counter for autmatically creating fully-connected layer variable names
bn_n = 0


def batch_normalization_layer(input_layer, dimension):
    '''Helper function to do batch normalziation

    Parameters
    ----------
    input_layer :   tf.Tensor
                    4D tensor
    dimension   :   int
                    input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    Returns
    -------
    tf.Tensor
           Tthe 4D tensor after being normalized
    '''
    global bn_n
    bn_n += 1
    with tf.variable_scope('batch_norm%d' % bn_n):
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        # normalise by mean and variance, and use no offset or scaling (0, 1)
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, None,
                None, 0.001)

        return bn_layer

# counter for autmatically creating fully-connected layer variable names
weighted_pool_n = 0


def weighted_pool_layer(input_layer, ksize, strides=(1, 1, 1, 1)):
    '''Helper function to do mixed max/avg pooling

    Parameters
    ----------
    input_layer :   tf.Tensor
                    4D tensor
    Returns
    -------
    tf.Tensor
           Tthe 4D tensor after being pooled
    '''
    global weighted_pool_n
    weighted_pool_n += 1
    with tf.variable_scope('weight_pool%d' % weighted_pool_n):
        a = tf.get_variable('a',
                initializer=tf.truncated_normal_initializer(),
                shape=(1,),
                dtype=tf.float32, trainable=True)
        max_pool = tf.nn.max_pool(input_layer, ksize, strides, padding='SAME')
        avg_pool = tf.nn.avg_pool(input_layer, ksize, strides, padding='SAME')
        pool = (a * max_pool + (1 - a) * avg_pool)
        return pool

inc_n = 0

def inception2d(x, in_channels, filter_count):
    '''Helper function to create inception module

    Parameters
    ----------
    in_channels :   int
                    number of input channels
    filter_count    :   int
                        number of filters to use for soemthing ?

    Returns
    -------
    tf.Tensor
           Tensor with filter_count*3 +1 output channels
    '''
    global inc_n
    inc_n += 1
    with tf.variable_scope('inception%d' % inc_n):
        # bias dimension = 3*filter_count and then the extra in_channels for the avg
        # pooling
        bias = tf.Variable(tf.truncated_normal([3*filter_count + in_channels]))

        # 1x1
        one_filter = tf.Variable(tf.truncated_normal([1, 1, in_channels,
            filter_count]))
        one_by_one = tf.nn.conv2d(x, one_filter,
                    strides=[1, 1, 1, 1], padding='SAME')

        # 3x3
        three_filter = tf.Variable(tf.truncated_normal([3, 3, in_channels,
            filter_count]))
        three_by_three = tf.nn.conv2d(x,
                    three_filter, strides=[1, 1, 1, 1], padding='SAME')

        # 5x5
        five_filter = tf.Variable(tf.truncated_normal([5, 5, in_channels,
            filter_count]))
        five_by_five = tf.nn.conv2d(x, five_filter,
                    strides=[1, 1, 1, 1], padding='SAME')

        # avg pooling
        pooling = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                padding='SAME')

        x = tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)
        # Concat in the 4th dim to stack
        x = tf.nn.bias_add(x, bias)
        return tf.nn.relu(x)


class ParameterTest(object):
    '''Test one set of parameters to the train() function.'''
    def __init__(self, model, batch_size, epochs,
            train_function):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.accuracy = None
        self.train_function=train_function

    def run(self):
        '''Run the training process with the specified settings.'''

        save_fname = '{name}_{batch}_{lr}_{epochs}_{opti}_{act}.ckpt'.format(
                name=self.model.__class__.__name__,
                batch=self.batch_size,
                lr=self.model.opt._learning_rate,
                epochs=self.epochs,
                opti=self.model.opt.get_name(),
                act=self.model.act_fn.__name__
        )
        self.accuracy = self.train_function(self.model, self.batch_size,
                self.epochs, save_fname, return_records=False,
                record_step=30)

    def __str__(self):
        return ('{opti:30}, learning rate={lr:5.4f}, batch size={bs:<5d}, '
                'epochs={epochs:<5d}, accuracy={acc:4.3f}'.format(
                    lr=self.model.opt._learning_rate,
                    opti=self.model.opt.get_name(),
                    bs=self.batch_size,
                    epochs=self.epochs,
                    acc=self.accuracy
                )
        )


def main():
    tf_optimizers = {class_name[:-len('Optimizer')] for class_name in dir(tf.train) if 'Optimizer'
            in class_name and class_name != 'Optimizer'}
    parser = argparse.ArgumentParser(description='Test the net on one parameter set')
    parser.add_argument('-o', '--optimizer', required=True, type=str,
            choices=tf_optimizers, help='Optimization algorithm')
    parser.add_argument('-l', '--learning-rate', required=True, type=float,
            help='Learning rate for the optimizer')
    parser.add_argument('-b', '--batch-size', required=True, type=int,
            help='Batch size')
    parser.add_argument('-e', '--epochs', required=True, type=int,
            help='Number of epochs')
    parser.add_argument('-f', '--file', required=True, type=str,
            help='File to write result to')
    parser.add_argument('-m', '--model', required=True, type=str,
            help='Package path where Model class is located')
    parser.add_argument('-t', '--train', required=True, type=str,
            help='Module to search for train_model() function.')

    args = parser.parse_args()
    model_cls = __import__(args.model, globals(), locals(), ['Model']).Model
    train_fn = __import__(args.train, globals(), locals(),
            ['train_model']).train_model

    optimizer_cls = getattr(tf.train, args.optimizer + 'Optimizer')
    optimizer = optimizer_cls(args.learning_rate)
    model = model_cls(optimizer, tf.nn.relu)

    pt = ParameterTest(model, args.batch_size, args.epochs, train_fn)
    pt.run()
    print(pt)
    # the OS ensures sequential writes with concurrent processes
    with open(args.file, 'a') as f:
        f.write(str(pt) + '\n')
        f.flush()

if __name__ == '__main__':
    main()
