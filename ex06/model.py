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
        cell_memory_size   = kwargs.get('cell_memory_size', 64)
        hidden_size        = kwargs.get('hidden_size', cell_memory_size)
        keep_prob          = kwargs.get('keep_prob', 0.85)
        subsequence_length = kwargs.get('subsequence_length', 100)
        batch_size         = kwargs.get('batch_size', 250)
        optimizer          = get_optimizer(kwargs.get('optimizer', 'Adam'))

        ############################################################################################
        #                                        Net inputs                                        #
        ############################################################################################
        self.input_ids    = tf.placeholder(tf.int32, shape=(None, subsequence_length))
        self.label        = tf.placeholder(tf.int32, shape=(None, 1))
        self.hidden_state = tf.placeholder(tf.float32, shape(hidden_size))
        self.cell_state   = tf.placeholder(tf.float32, shape(cell_memory_size))

        ############################################################################################
        #                                        Embedding                                         #
        ############################################################################################
        self.embedding_matrix, _ = get_weights_and_bias((vocab_size, embedding_size))
        inputs                   = tf.nn.dropout(
                                       tf.nn.embedding_lookup(self.embedding_matrix, input_ids),
                                       keep_prob=keep_prob
                                   )

        ############################################################################################
        #                            LSTM stuff pasted from slides ???                             #
        ############################################################################################
        cell = tf.nn.rnn_cell.BasicLSTMCell(cell_memory_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        # what's the difference to just creating a zero-filled variable?
        self.zero_state = cell.zero_state(batch_size, tf.float32)
        state           = tf.nn.rnn_cell.LSTMStateTuple(c=cell_state, h=hidden_state)
        # tf.nn.static_rnn expects list of time step vectors
        # Unroll the model, returns list of outputs and final cell state
        sequences           = tf.unstack(inputs, num=subsequence_length, axis=1)
        outputs, self.state = tf.nn.static_rnn(cell, sequences, initial_state=state)
        # Recreate tensor from list
        outputs      = tf.reshape(tf.concat(outputs, 1),
                                  [batch_size, subsequence_length, cell_memory_size])
        self.outputs = tf.reduce_mean(outputs)

        ############################################################################################
        #                        Fully connected layer, loss, and training                         #
        ############################################################################################
        ff1             = fully_connected(outputs, n_out, with_activation=False, use_bias=True)
        loss            = tf.nn.sparse_softmax_cross_entropy_with_logits(label, ff1)
        self.train_step = optimizer.minimize(loss)

    def get_zero_state(self, session):
        return session.run(self.zero_state)

    def run_training_step(self, session, sequences, labels, state):
        for subsequence in subsequences(data):
            # Get state of last step
            _state, _ = session.run([self.state, self.train_step],
                feed_dict = {
                    self.input_ids: subsequence,
                    self.label: label,
                    self.cell_state: _state.c,
                    self.hidden_state: _state.h
                })

    def get_embeddings(self):
        return self.embedding_matrix.eval()

def main():
    from imdb_helper import IMDB
    helper = IMDB('data')

if __name__ == "__main__":
    main()
