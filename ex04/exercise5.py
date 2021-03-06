import tensorflow as tf
from .svhn_helper import SVHN
import os
import numpy as np

def train_model(model, batch_size, epochs, save_fname, return_records=False,
        record_step=20, ignore_saved=False):
    '''Train a model on the SVHN dataset.

    Parameters
    ----------
    model           :   Model (defined above)
                        The training model.
    batch_size      :   int
                        Size of training batch.
    epochs          :   int
                        Number of times to visit the entire training set.
    save_fname      :   string
                        The filename of the file carrying all the learned
                        variables.
    return_records  :   bool
                        Determines whether only the final accuracy (False) or a
                        history of all entropies and accuracies is returned.
    record_step     :   int
                        Accuracy on test set will be recorded every
                        ``record_step`` training steps.
    ignore_saved    :   bool
                        Do not load saved weights, if found

    Returns
    -------
    float OR tuple
            If ``return_records`` is set, all entropies and accuracies are
            returned. Else the best accuracy is returned.

    '''

    svhn = SVHN()

    # keeep records of performance
    entropies = []
    accuracies = []
    best_accuracy = 0

    with tf.Session() as sess:

        ########################################################################
        #                             Load weights                             #
        ########################################################################
        saver = tf.train.Saver()
        if not ignore_saved and os.path.exists(save_fname + '.meta'):
            print('Using saved weights.')
            saver.restore(sess, save_fname)
            final_accuracy = model.get_accuracy(sess, svhn._validation_data,
                                svhn._validation_labels)
            return final_accuracy
        else:
            ####################################################################
            #                             Training                             #
            ####################################################################
            sess.run(tf.global_variables_initializer())

            # number of training steps
            training_step = 0

            for epoch in range(epochs):
                print('Starting epoch %d' % epoch)

                # run one batch
                for data, labels in svhn.get_training_batch(batch_size):
                    entropy = model.run_training_step(sess, data, labels)
                    entropies.append(entropy)

                    # compute validation accuracy every record_step steps
                    if training_step % record_step == 0:
                        val_accuracy = model.get_accuracy(sess, svhn._validation_data,
                                svhn._validation_labels)
                        accuracies.append(val_accuracy)
                        # in case we need it later
                        final_accuracy = val_accuracy
                        print('Current validation accuracy %f' % val_accuracy)

                        # save if better
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            print('Saving model with accuracy %f.' % val_accuracy)
                            saver.save(sess, save_fname)

                    training_step += 1

                    # stop early if convergence too slow
                    if epoch == 1:
                        if val_accuracy < 0.2:
                            raise RuntimeError('This isn\'t going anywhere.')

            ####################################################################
            #               Make final recordings, if necessary                #
            ####################################################################
            if training_step % record_step == 1:
                # we just recorded, final_accuracy already correct
                pass
            else:
                # we need to recompute the final accuracy
                final_accuracy = model.get_accuracy(sess, svhn._validation_data,
                                    svhn._validation_labels)
                accuracies.append(final_accuracy)

            ####################################################################
            #                     Print misclassifications                     #
            ####################################################################
            if return_records:
                return entropies, accuracies
            else:
                return best_accuracy

