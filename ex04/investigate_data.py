import numpy as np
from svhn_helper import SVHN
from matplotlib import pyplot as plt

svhn = SVHN()

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

def plot(N=100):

    N_train, N_val = svhn.get_sizes()[:2]
    random_indices_train = np.random.choice(N_train, N, replace=False)
    random_indices_val = np.random.choice(N_val, N, replace=False)

    train_batch = list(zip(svhn._training_data[random_indices_train],
        svhn._training_labels[random_indices_train]))

    val_batch = list(zip(svhn._validation_data[random_indices_val],
        svhn._validation_labels[random_indices_val]))

    categories = ['0','1','2','3','4','5','6','7','8','9','0']
    h, w = (int(np.floor(np.sqrt(N))), int(np.ceil(np.sqrt(N))))
    f_train, axarr_train = plt.subplots(h, w)
    f_val, axarr_val = plt.subplots(h, w)
    for i in range(h):
        for j in range(w):
            index = 4*i+j

            ax_train = axarr_train[i][j]
            ax_val = axarr_val[i][j]

            img_train = train_batch[index][0].reshape((32,32))
            img_val = val_batch[index][0].reshape((32,32))

            label_train = train_batch[index][1]
            label_val = val_batch[index][1]

            ax_train.set_title(categories[label_train])
            ax_train.imshow(img_train, cmap='gray')
            ax_train.get_xaxis().set_visible(False)
            ax_train.get_yaxis().set_visible(False)

            ax_val.set_title(categories[label_val])
            ax_val.imshow(img_val, cmap='gray')
            ax_val.get_xaxis().set_visible(False)
            ax_val.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    print_statistics()
    plot()
