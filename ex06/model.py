from imdb_helper import IMDB
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, DropoutWrapper
from tensorflow import (placeholder, cond, reduce_mean, reduce_sum, where, not_equal, ones_like,
                        zeros_like, reshape, equal, constant, cast, concat, argmax)
from tensorflow import nn
import sys; sys.path.insert(0, '..')
from util import get_weights_and_bias, get_optimizer, fully_connected

def length(sequences):
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
    _1         = ones_like(sequences)
    _0         = zeros_like(sequences)
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
        optimizer          = get_optimizer(kwargs['optimizer'])

        ############################################################################################
        #                                        Net embeddings                                        #
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

        # what's the difference to just creating a zero-filled variable?
        self.zero_state = cell.zero_state(self.batch_size, tf.float32)
        state           = LSTMStateTuple(c=self.cell_state, h=self.hidden_state)

        # A dynamic rnn creates the graph on the fly, so it can deal with embeddings of different
        # lengths. We do not need to unstack the embedding tensor to get rows, instead we compute
        # thethe actual sequence lengths and pass that
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
        self.train_step    = optimizer.minimize(loss)
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
    args = parser.parse_args()
    batch_size = args.batch_size
    helper.create_dictionaries(args.vocabulary_size, args.cutoff)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    epochs = args.epochs

    ################################################################################################
    #                                     Initialise the model                                     #
    ################################################################################################
    print('Creating model')
    model = IMDBModel(vocab_size=args.vocabulary_size, subsequence_length=args.sequence_length,
                      optimizer=optimizer, embedding_size=args.embedding_size,
                      memory_size=args.memory_size, keep_prob=args.keep_probability)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            print(f'Starting epoch {epoch}')

            for batch, labels in helper.get_training_batch(batch_size):
                # reset state for each batch
                state = model.get_zero_state(session, batch_size)

                for subsequence_batch in helper.slice_batch(batch, args.sequence_length):
                    # push one subsequenc of each batch member
                    state = model.run_training_step(session, subsequence_batch, labels, state)

            ##############################
            #  Test with all test data.  #
            ##############################
            test_data, test_labels = helper._test_data, helper._test_labels
            test_data = next(helper.slice_batch(test_data, args.sequence_length))
            accuracy = model.run_test_step(session, test_data, test_labels)
            print(f'Accuracy = {accuracy:3.3f}')

if __name__ == "__main__":
    main()
