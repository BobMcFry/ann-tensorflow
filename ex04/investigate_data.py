import numpy as np
from matplotlib import pyplot as plt

def print_statistics():
    ############################################################################
    #                         Print class distribution                         #
    ########### #################################################################
    print('Percentage of labels in train and validation set')
    train_labels = svhn._training_labels
    validation_labels = svhn._validation_labels
    train_dist = np.histogram(train_labels)[0] / len(train_labels)
    validation_dist = np.histogram(validation_labels)[0]  / len(validation_labels)
    print('%15s | %10s' % ('train', 'validation'))
    print(' ' * 5 + '-' * 23)
    for index, (t, v) in enumerate(zip(train_dist, validation_dist)):
        print('%5d%10f | %10f' % (index + 1, t, v))

def plot_mispredictions(model, filename, data, labels):
    import tensorflow as tf
    with tf.Session().as_default() as session:
        saver = tf.train.Saver()
        saver.restore(session, filename)
        validation_predictions = model.predict(session, data)
        actual_labels = labels
        mispredictions = np.argwhere(actual_labels != validation_predictions)
        print(f'Number of misclassifications: {len(mispredictions):d}')
        print(f'Percent: {len(mispredictions)/len(actual_labels)*100:f}')
        N = 10**2
        mislabeled_data = data[mispredictions, ...]
        mislabeled_labels = validation_predictions[mispredictions][:, 0]
        plot(list(zip(mislabeled_data[:N], mislabeled_labels[:N])))

def plot(data):

    N = len(data)
    categories = ['0','1','2','3','4','5','6','7','8','9','0']
    h, w = (int(np.floor(np.sqrt(N))), int(np.ceil(np.sqrt(N))))
    f, axarr = plt.subplots(h, w)
    for i in range(h):
        for j in range(w):
            index = 4*i+j

            ax = axarr[i][j]

            img = data[index][0].reshape((32,32))

            label = data[index][1]

            ax.set_title(categories[label])
            ax.imshow(img, cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

def main():
    N = 100
    N_train, N_val = svhn.get_sizes()[:2]
    random_indices_train = np.random.choice(N_train, N, replace=False)
    random_indices_val = np.random.choice(N_val, N, replace=False)

    train_batch = list(zip(svhn._training_data[random_indices_train],
        svhn._training_labels[random_indices_train]))

    val_batch = list(zip(svhn._validation_data[random_indices_val],
        svhn._validation_labels[random_indices_val]))

    print_statistics()
    plot(train_batch)
    plot(val_batch)

if __name__ == "__main__":
    from svhn_helper import SVHN
    svhn = SVHN()
    main()
else:
    from .svhn_helper import SVHN
    svhn = SVHN()


