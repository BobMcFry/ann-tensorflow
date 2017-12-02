import tensorflow as tf
from book_helper import Book
import numpy as np
import sys; sys.path.insert(0, '..')
from util import get_weights_and_bias, get_optimizer



class SkipGramModel(object):
    '''A minimal customizable SkipGramModel for predicting one context word from
    1 target word.

    Attributes
    ----------
    target_word_id  :   tf.placeholder of shape (batch,)
                        Batch of input word ids to predict for
    target_context_id   :   tf.placeholder of shape (batch, 1)
                            Batch of target context word ids
    loss    :   tf.op
                operation for the NCE loss
    merged  :   tf.summary.scalar
                Summary var for the loss
    train_step  :   tf.Variable
                    optimizer training step
    '''

    def __init__(self, vocabulary_size, embedding_size, validation_set, **kwargs):
        '''
        Parameters
        ----------
        vocabulary_size :   int
                            Number of classes to predict from (the size of the
                            considerer vocabulary)
        embedding_size  :   int
                            Dimensionality of the space to project the words
                            onto. Length of the vectors representing a word
        validation_set  :   Set of word ids to check the similarity for

        learning_rate   :   float
                            Optimizer learning rate
        optimizer       :   str
                            Name of the tf optimizer to use (e.g.  "GradientDescent")
        noise_samples   :   int
                            Number of noise samples for NCE sampling
        '''
        ###########################
        #  Extract relevant args  #
        ###########################
        optimizer_cls = get_optimizer(kwargs.get('optimizer', 'GradientDescent'))
        learning_rate = kwargs.get('learning_rate', 1)
        noise_samples = kwargs.get('noise_samples', 64)

        ############################################
        #  Input word id + target context word id  #
        ############################################
        self.target_word_id = tf.placeholder(tf.int32, shape=(None,),
                name='target')
        self.target_context_id = tf.placeholder(tf.int32, shape=(None, 1),
                name='target')

        ##################
        #  Hidden layer  #
        ##################
        W_context, _ = get_weights_and_bias((vocabulary_size, embedding_size))
        a_h = tf.nn.embedding_lookup(W_context, self.target_word_id)

        ##################
        #  Output layer  #
        ##################
        # notice that - strangely - the weights matrix must be transposed from
        # what you would use for multiplying a_h * W. This seems to be a quirk
        # of tf nce_loss
        initializer = tf.random_normal_initializer(stddev=1 /
                np.sqrt(embedding_size))
        W_target, b_target = get_weights_and_bias((vocabulary_size, embedding_size),
                shape_b=(vocabulary_size, ), initializer_w=initializer)

        ##########
        #  Loss  #
        ##########
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=W_target,
                        biases=b_target,
                        labels=self.target_context_id,
                        inputs=a_h,
                        num_sampled=noise_samples,
                        num_classes=vocabulary_size
                        )
                    )

        #########################
        #  TensorBoard logging  #
        #########################
        with tf.variable_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()

        self.train_step = optimizer_cls(learning_rate).minimize(self.loss)


        ########################################
        #  Accuracy for a fixed set of words.  #
        ########################################
        # Compute the cosine similarity between minibatch examples and all
        # embeddings. Copypasta from stackoverflow
        norm = tf.sqrt(tf.reduce_sum(tf.square(W_context), 1, keep_dims=True))
        normalized_embeddings = W_context / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                validation_set)
        self.similarity = tf.matmul(valid_embeddings, normalized_embeddings,
                transpose_b=True)
        # get the 8 closeset, because the first closest is always the word
        # itself.
        self.closest_words = tf.nn.top_k(self.similarity, 8).indices[:, 1:]


    def run_training_step(self, session, inputs, labels):
        '''Run one training step.

        Returns
        -------
        tuple
                Tuple with loss, array of closest ids for the validation set,
                and the summary variable for saving
        '''

        _, loss, closest, summary = session.run([self.train_step, self.loss,
            self.closest_words, self.merged], feed_dict={
                self.target_word_id: inputs,
                self.target_context_id: labels,
                })

        return loss, closest, summary

def main():
    book = Book()
    vocabulary_size = 10000
    embedding_size = 64
    skip_window = 2
    optimizer = 'GradientDescent'
    learning_rate = 1
    epochs = 10
    batch_size = 128

    summary_dir = './summary/train/'

    book.create_dictionaries(vocabulary_size)

    # 4: Test the mapping
    words, contexts = next(book.get_training_batch(1, 10))
    words = ['the', 'god', 'jesus']
    ids = book.words2ids(words)
    assert words == book.ids2words(ids), 'Shits not working, yo.'

    # Rest of the sheet
    # the validation set is fixed
    validation_set = book.words2ids(['5', 'make', 'god', 'jesus', 'year', 'sin',
        'israel'])

    # we instantiate the model
    model = SkipGramModel(vocabulary_size, embedding_size, validation_set,
            optimizer=optimizer, learning_rate=learning_rate)

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(summary_dir, session.graph)

        for epoch in range(epochs):

            for i, (inputs, context) in enumerate(book.get_training_batch(batch_size, skip_window)):
                loss, closest, summary = model.run_training_step(session,
                        inputs, context)
                # every so often, write the loss summary tensor
                if i % 20 == 0:
                    train_writer.add_summary(summary, epoch*i)

            ##########################
            #  Print the validation  #
            ##########################
            print(f'loss={loss:3.4f}')
            for idx, word in enumerate(validation_set):
                closest_words = ' '.join(book.ids2words(closest[idx, :]))
                word_id = book.ids2words(word)
                print(f'Closest for {word_id}: {closest_words}')


if __name__ == '__main__':
    main()
