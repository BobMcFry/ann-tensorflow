from matplotlib import pyplot as plt
from math import sqrt
from mnist import MNISTLoader


def plot_mnist_digits(*digits_labels):
    '''Plot an aribtrary number of mnis digits on an automatic grid.

    Parameters
    ----------
    digits_labels   :   list
                        List of tuples (np.ndarray, int) giving an image of
                        shape (28, 28) and a label
    Returns
    -------
    pyplot.Figure
            Figure with digits drawn into the only axes
    '''
    num = len(digits_labels)
    # try a square shape, but round to the nearest integer (down for rows, up
    # for cols)
    rows = int(sqrt(num))
    cols = int(num / rows + 0.5)
    f, axarr = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            ax = axarr[row][col]
            # linear index from two indeces
            index = row * rows + col
            ax.imshow(digits_labels[index][0], cmap='gray')
            # place class label to the left of plot
            ax.set_title(digits_labels[index][1], x=-0.1, y=0.5)
            # remove the ticks (pointless for images)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    return f


if __name__ == "__main__":
    loader = MNISTLoader()
    train_imgs = loader.training_data
    train_labels = loader.training_labels
    f = plot_mnist_digits(*[(train_imgs[i, ...], train_labels[i]) for i in
        range(20)])
    plt.show()
