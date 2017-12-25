import numpy as np
import tensorflow as tf
import sys; sys.path.insert(0, '..')
from util import get_weights_and_bias, get_optimizer, fully_connected

class IMDBModel(object):
    '''Model for imbd movie review classification.'''

    def __init__(self, **kwargs):
        ############################################################################################
        #                                 Get all hyperparameters                                  #
        ############################################################################################
        vocab_size         = kwargs.get('vocab_size', 20000)
        embedding_size     = kwargs.get('embedding_size', 64)
        memory_size        = kwargs.get('memory_size', 64)
        keep_prob          = kwargs.get('keep_prob', 0.85)
        subsequence_length = kwargs.get('subsequence_length', 100)
        batch_size         = kwargs.get('batch_size', 1)
        optimizer          = get_optimizer(kwargs.get('optimizer', 'Adam'))()

        ############################################################################################
        #                                        Net inputs                                        #
        ############################################################################################
        self.input_ids    = tf.placeholder(tf.int32, shape=(batch_size, subsequence_length))
        self.labels        = tf.placeholder(tf.int32, shape=(batch_size,))
        self.hidden_state = tf.placeholder(tf.float32, shape=(batch_size, memory_size))
        self.cell_state   = tf.placeholder(tf.float32, shape=(batch_size, memory_size))

        ############################################################################################
        #                                        Embedding                                         #
        ############################################################################################
        self.embedding_matrix, _ = get_weights_and_bias((vocab_size, embedding_size))
        # inputs                   = tf.nn.dropout(
        #                                tf.nn.embedding_lookup(self.embedding_matrix, self.input_ids),
        #                                keep_prob=keep_prob
        #                            )
        inputs                   = tf.nn.embedding_lookup(self.embedding_matrix, self.input_ids)

        ############################################################################################
        #                            LSTM stuff pasted from slides ???                             #
        ############################################################################################
        cell = tf.contrib.rnn.BasicLSTMCell(memory_size, state_is_tuple=True)
        # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        # what's the difference to just creating a zero-filled variable?
        self.zero_state = cell.zero_state(batch_size, tf.float32)
        state           = tf.contrib.rnn.LSTMStateTuple(c=self.cell_state, h=self.hidden_state)
        # tf.nn.static_rnn expects list of time step vectors
        # Unroll the model, returns list of outputs and final cell state
        sequences           = tf.unstack(inputs, num=subsequence_length, axis=1)
        outputs, self.state = tf.contrib.rnn.static_rnn(cell, sequences, initial_state=state)
        # Recreate tensor from list
        outputs      = tf.reshape(tf.concat(outputs, 1),
                                  [batch_size, subsequence_length * memory_size])
        self.outputs = tf.reduce_mean(outputs)

        ############################################################################################
        #                        Fully connected layer, loss, and training                         #
        ############################################################################################
        ff1             = fully_connected(outputs, 2, with_activation=False, use_bias=True)
        loss            = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=ff1)
        self.train_step = optimizer.minimize(loss)

        self.predictions = tf.nn.softmax(ff1)
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.predictions, 1), tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def get_zero_state(self, session):
        return session.run(self.zero_state)

    def run_training_step(self, session, subsequence_batch, label, state):
        # Get state of last step
        _state, _ = session.run([self.state, self.train_step],
            feed_dict = {
                self.input_ids: subsequence_batch,
                self.labels: label,
                self.cell_state: state.c,
                self.hidden_state: state.h
            })
        return _state

    def run_test_step(self, session, subsequence_batch, labels):
        zero_state = self.get_zero_state(session)
        predictions, accuracy = session.run([self.predictions, self.accuracy],
            feed_dict = {
                self.input_ids: subsequence_batch,
                self.labels: labels,
                self.cell_state: zero_state.c,
                self.hidden_state: zero_state.h
            })
        return accuracy

    def get_embeddings(self):
        return self.embedding_matrix.eval()

def main():
    from imdb_helper import IMDB
    print('Loading IMDB data')
    helper = IMDB('data')

    vocab_size = 20000
    cutoff = 300
    batch_size = 250
    subseq_len = 100
    helper.create_dictionaries(vocab_size, cutoff)

    epochs = 5

    print('Creating model')
    model = IMDBModel(vocab_size=vocab_size, batch_size=batch_size, subsequence_length=subseq_len)
    n_batches = helper.get_sizes()[0] // batch_size

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for batch_idx, (batch, labels) in enumerate(helper.get_training_batch(batch_size)):
                state = model.get_zero_state(session)
                print(f'Training ... {batch_idx} of {epochs * n_batches}')
                for subsequence_batch in helper.slize_batch(batch, subseq_len):
                    model.run_training_step(session, subsequence_batch, labels, state)

            ########################################################################################
            #                               Test with one test batch                               #
            ########################################################################################
            test_data, test_labels = next(helper.get_test_batch(batch_size))
            test_data = next(helper.slize_batch(test_data, subseq_len))
            accuracy = model.run_test_step(session, test_data, labels)
            print(f'Accuracy = {accuracy:3.3f}')

if __name__ == "__main__":
    main()
