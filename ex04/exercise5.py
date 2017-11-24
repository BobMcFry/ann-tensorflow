import tensorflow as tf
from .svhn_helper import SVHN
import os

def train_model(model, batch_size, epochs, save_fname, return_records=False,
        record_step=20):
    '''Train a model on the SVHN dataset.

    Returns
    -------
    float
        the best validation accuracy found
    '''

    svhn = SVHN()

    # keeep records of performance
    entropies = []
    accuracies = []
    best_accuracy = 0

    with tf.Session() as sess:

        saver = tf.train.Saver()
        if os.path.exists(save_fname + '.meta'):
            print('Using saved weights.')
            saver.restore(sess, save_fname)
            final_accuracy = model.get_accuracy(sess, svhn._validation_data,
                                svhn._validation_labels)
            return final_accuracy
        else:
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

            if training_step % record_step == 1:
                # we just recorded, final_accuracy already correct
                pass
            else:
                # we need to recompute the final accuracy
                final_accuracy = model.get_accuracy(sess, svhn._validation_data,
                                    svhn._validation_labels)
                accuracies.append(final_accuracy)

            if return_records:
                return entropies, accuracies
            else:
                return best_accuracy

