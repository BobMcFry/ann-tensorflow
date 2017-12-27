from argparse import ArgumentParser
import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, DropoutWrapper
from tensorflow import (placeholder, cond, reduce_mean, reduce_sum, where, not_equal, ones_like,
                        zeros_like, reshape, equal, constant, cast, concat, argmax)
from tensorflow import nn

import sys; sys.path.insert(0, '..')
from util import get_weights_and_bias, get_optimizer, fully_connected
from imdb_helper import IMDB


class OptimizerSpec(dict):
    '''Encapsulate all the info needed for creating any kind of optimizer.'''

    def parse_learning_rate(self, rate):
        '''Create a tf.piecewise_constant learning rate scheduling from a string.'''
        try:
            self.learning_rate = float(rate)
            self.step_counter   = None
            print('Using fixed learning rate...')
        except ValueError:
            print('Creating schedule...')
            boundary_str, lr_str = rate.split(':')
            boundaries           = [int(n) for n in boundary_str.split(',')]
            lrs                  = [float(n) for n in lr_str.split(',')]
            step_counter          = tf.Variable(0, trainable=False, dtype=tf.int32,
                                               name='step_counter')
            self.learning_rate   = tf.train.piecewise_constant(step_counter, boundaries, lrs,
                                                               name='rate_schedule')
            self.step_counter     = step_counter

    def __str__(self):
        key_val_str = ', '.join(str(k) + '=' + str(v) for k, v in self.items())
        return f'<Optimizer: {key_val_str}>'

    def __init__(self, **kwargs):
        if not 'kind' in kwargs:
            raise ValueError('No optimizer name given')
        self.parse_learning_rate(kwargs['learning_rate'])
        self.update(kwargs)

    def create(self):
        kind          = self['kind']
        learning_rate = self.learning_rate
        name          = self.get('name', 'optimizer')
        optimizer_cls = get_optimizer(kind)
        if kind in ['Momentum', 'RMSProp']:
            try:
                momentum = self['momentum']
            except KeyError:
                raise ValueError('Momentum parameter is necessary for MomentumOptimizer')
            if kind == 'Momentum':
                if 'use_nesterov' in self:
                    use_nesterov = self['use_nesterov']
                else:
                    use_nesterov = False
                return optimizer_cls(learning_rate, momentum, use_nesterov, name=name)
            else:
                return optimizer_cls(learning_rate, momentum, name=name)
        else:
            return optimizer_cls(learning_rate, name=name)



def length(sequences, padding_value=1):
    '''Find the actual sequence length for each sequence in a tensor. Sequences could be padded with
    1s if they were shorter than the cutoff length chosen.

    Parameters
    ----------
    sequences   :   tf.Tensor
                    Tensor of shape [batch_size x sequence_length]

    Returns
    -------
    tf.Tensor
        Tensor of shape [batch_size,], each value being the true length of its associated sequence
    '''
    _1         = tf.fill(tf.shape(sequences), padding_value)
    _0         = zeros_like(sequences)
    # set values != 1 to 1 and the rest to 0, so the sum is the number
    # of nonzeros
    is_padding = where(not_equal(sequences, _1), _1, _0)
    return reduce_sum(is_padding, axis=1)


class IMDBModel(object):
    '''Model for IMBD movie review classification.'''

    def __init__(self, **kwargs):
        '''The following arguments are accepted:

        Parameters
        ----------
        vocab_size  :   int
                        Size of the vocabulary for creating embeddings
        embedding_matrix    :   int
                                Dimensionality of the embedding space
        memory_size :   int
                        LSTM memory size
        keep_prob   :   Inverse of dropout percentage for embedding and LSTM
        subsequence_length  :   Length of the subsequences (all embeddings are padded to this length)
        optimizer   :   str or tf.train.Optimizer
                        Class name of the optimizer to use, or an optimizer object

        '''
        ############################################################################################
        #                                 Get all hyperparameters                                  #
        ############################################################################################
        vocab_size         = kwargs['vocab_size']
        embedding_size     = kwargs['embedding_size']
        memory_size        = kwargs['memory_size']
        keep_prob          = kwargs['keep_prob']
        subsequence_length = kwargs['subsequence_length']
        optimizer_spec     = kwargs['optimizer']
        optimizer          = optimizer_spec.create()
        self.learning_rate = optimizer_spec.learning_rate
        self.step_counter   = optimizer_spec.step_counter

        ############################################################################################
        #                                        Net embeddings                                    #
        ############################################################################################
        self.batch_size   = placeholder(tf.int32,   shape=[],                  name='batch_size')
        self.is_training  = placeholder(tf.bool,    shape=[],                  name='is_training')
        self.word_ids     = placeholder(tf.int32,   shape=(None, subsequence_length),
                                                                               name='word_ids')
        self.labels       = placeholder(tf.int32,   shape=(None,),             name='labels')
        self.hidden_state = placeholder(tf.float32, shape=(None, memory_size), name='hidden_state')
        self.cell_state   = placeholder(tf.float32, shape=(None, memory_size), name='cell_state')

        self.lengths = length(self.word_ids)

        ############################################################################################
        #                                        Embedding                                         #
        ############################################################################################
        self.embedding_matrix, _bias = get_weights_and_bias((vocab_size, embedding_size))
        embeddings = cond(self.is_training,
                         lambda: nn.dropout(
                             nn.embedding_lookup(self.embedding_matrix, self.word_ids),
                             keep_prob=keep_prob),
                         lambda: nn.embedding_lookup(self.embedding_matrix, self.word_ids)
                         )

        ############################################################################################
        #                                        LSTM layer                                        #
        ############################################################################################
        cell = BasicLSTMCell(memory_size)

        # during inference, use entire ensemble
        keep_prob = cond(self.is_training, lambda: constant(keep_prob), lambda: constant(1.0))
        cell      = DropoutWrapper(cell, output_keep_prob=keep_prob)

        # what's the difference to just creating a zero-filled tensor tuple?
        self.zero_state = cell.zero_state(self.batch_size, tf.float32)
        state           = LSTMStateTuple(c=self.cell_state, h=self.hidden_state)

        # A dynamic rnn creates the graph on the fly, so it can deal with embeddings of different
        # lengths. We do not need to unstack the embedding tensor to get rows, instead we compute
        # the actual sequence lengths and pass that
        outputs, self.state = nn.dynamic_rnn(cell, embeddings, sequence_length=self.lengths,
                                             initial_state=state)
        # Recreate tensor from list
        outputs      = reshape(concat(outputs, 1), [-1, subsequence_length * memory_size])
        self.outputs = reduce_mean(outputs)

        ############################################################################################
        #                        Fully connected layer, loss, and training                         #
        ############################################################################################
        ff1             = fully_connected(outputs, 2, with_activation=False, use_bias=True)
        loss            = reduce_mean(
                                nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                            logits=ff1))
        if self.step_counter:
            self.train_step = optimizer.minimize(loss, global_step=self.step_counter)
        else:
            self.train_step = optimizer.minimize(loss)
        self.predictions   = nn.softmax(ff1)
        correct_prediction = equal(cast(argmax(self.predictions, 1), tf.int32), self.labels)
        self.accuracy      = reduce_mean(cast(correct_prediction, tf.float32))


    def get_zero_state(self, session, batch_size):
        '''Retrieve the LSTM zero state.

        Parameters
        ----------
        session :   tf.Session
                    Open session to run the op in
        batch_size  :   int
                        Batch size (required for the tensor shapes, since the state cannot have
                        variable dimensions)

        Returns
        -------
        LSTMStateTuple
            Tuple of zero tensors of shape [batch_size x memory_size]
        '''
        return session.run(self.zero_state, feed_dict = {self.batch_size: batch_size})

    def run_training_step(self, session, subsequence_batch, labels, state):
        '''Run one training step.

        Parameters
        ----------
        session :   tf.Session
                    Open session to run ops in
        subsequence_batch   :   np.ndarray
                                Array of subsequences
        labels  :   np.ndarray
                    Array of labels for each batch
        state   :   LSTMStateTuple
                    LSTM memory state from the last step

        Returns
        -------
        LSTMStateTuple
            New memory state

        '''
        # Get state of last step
        _state, _, = session.run([self.state, self.train_step],
            feed_dict = {
                self.word_ids:     subsequence_batch,
                self.labels:       labels,
                self.cell_state:   state.c,
                self.hidden_state: state.h,
                self.batch_size:   subsequence_batch.shape[0],
                self.is_training:  True
            })
        return _state

    def run_test_step(self, session, subsequence_batch, labels):
        '''Run one test step.

        Parameters
        ----------
        session :   tf.Session
                    Open session to run ops in
        subsequence_batch   :   np.ndarray
                                Array of subsequences
        labels  :   np.ndarray
                    Array of labels for each batch

        Returns
        -------
        float
            Accuracy

        '''
        batch_size = subsequence_batch.shape[0]
        zero_state = self.get_zero_state(session, batch_size)
        predictions, accuracy = session.run([self.predictions, self.accuracy],
            feed_dict = {
                self.word_ids:     subsequence_batch,
                self.labels:       labels,
                self.cell_state:   zero_state.c,
                self.hidden_state: zero_state.h,
                self.batch_size:   batch_size,
                self.is_training:  False
            })
        return accuracy

def main():
    ################################################################################################
    #                                        Load the data                                         #
    ################################################################################################
    print('Loading IMDB data')
    helper = IMDB('data')

    ################################################################################################
    #                                       Parse arguments                                        #
    ################################################################################################
    parser = ArgumentParser()
    parser.add_argument('-v', '--vocabulary_size', type=int, default=20000)
    parser.add_argument('-s', '--sequence_length', type=int,
                        default=100, help='Length for subsequence training.')
    parser.add_argument('-c', '--cutoff', type=int, default=300,
                        help='Cutoff length for reviews.')
    parser.add_argument('-b', '--batch_size', type=int, default=250, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.03,
                        help='Initial learning rate (scheduling is used)')
    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding dimensionality')
    parser.add_argument('--memory_size', type=int, default=64, help='Memory size')
    parser.add_argument('-k', '--keep_probability', type=float, default=0.85,
                        help='Percentage of neurons to keep during dropout')
    parser.add_argument('-m', '--momentum', type=float, default=0.5,
                        help='Momentum (only used for Momentum optimizer)')
    parser.add_argument('-o', '--optimizer', type=str, default='Adam',
                        help='Optimizer class')
    parser.add_argument('--schedule', type=str, default=None,
                        help='Learning rate schedule. Format: "boundary1,...,boundaryN-1:lr1,...,lrN"')

    args = parser.parse_args()
    helper.create_dictionaries(args.vocabulary_size, args.cutoff)

    if args.schedule:
        learning_rate = args.schedule
    else:
        learning_rate = args.learning_rate
    opti_spec = OptimizerSpec(learning_rate=learning_rate, schedule=args.schedule,
                              kind=args.optimizer, momentum=args.momentum)
    print(f'Using optimizer {opti_spec}')
    batch_size = args.batch_size
    epochs = args.epochs

    ################################################################################################
    #                                     Initialise the model                                     #
    ################################################################################################
    print('Creating model')
    model = IMDBModel(vocab_size=args.vocabulary_size, subsequence_length=args.sequence_length,
                    optimizer=opti_spec, embedding_size=args.embedding_size,
                    memory_size=args.memory_size, keep_prob=args.keep_probability)

    train_data = helper._training_data
    max_len = np.max([len(sample) for sample in train_data])
    steps = int(np.ceil(max_len / args.subsequence_length))
    print(f'Estimated number of steps: {steps}')


    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            print(f'Starting epoch {epoch}')

            for batch_idx, (batch, labels) in enumerate(helper.get_training_batch(batch_size)):
                # reset state for each batch
                state = model.get_zero_state(session, batch_size)

                for subsequence_batch in helper.slice_batch(batch, args.sequence_length):
                    # push one subsequenc of each batch member
                    state = model.run_training_step(session, subsequence_batch, labels, state)

                if batch_idx % 10 == 0:
                    ##############################
                    #  Test with all test data.  #
                    ##############################
                    test_data, test_labels = helper._test_data, helper._test_labels
                    test_data = next(helper.slice_batch(test_data, args.sequence_length))
                    accuracy = model.run_test_step(session, test_data, test_labels)
                    print(f'Accuracy = {accuracy:3.3f}')

if __name__ == "__main__":
    main()
