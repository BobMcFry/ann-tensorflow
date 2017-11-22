import tensorflow as tf
from .svhn_helper import SVHN

def train_model(model, batch_size, epochs, save_fname, return_records=False,
        record_step=20):
    '''Train a model on the SVHN dataset.'''

    entropies = []
    accuracies = []
    svhn = SVHN()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        propagation = 0
        for epoch in range(epochs):
            print('Starting epoch %d' % epoch)
            for data, labels in svhn.get_training_batch(batch_size):
                entropy = model.run_training_step(sess, data, labels)
                if return_records:
                    entropies.append(entropy)
                if propagation % record_step == 0:
                    test_acc = model.get_accuracy(sess, svhn._validation_data,
                            svhn._validation_labels)
                    if return_records:
                        accuracies.append(test_acc[0])
                    print('Current training accuracy %f' % test_acc[0])
                propagation += 1
        saver.save(sess, save_fname, global_step=epoch)

        final_accuracy = model.get_accuracy(sess, svhn._validation_data,
                            svhn._validation_labels)
    if return_records:
        if propagation % record_step == 1:  # we just recorded
            pass
        else:
            accuracies.append(final_accuracy[0])
        return entropies, accuracies
    else:
        return final_accuracy[0]

