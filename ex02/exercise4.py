from matplotlib import pyplot as plt
from math import sqrt
from mnist import MNISTLoader

def plot_mnist_digits(*digits_labels):
    num = len(digits_labels)
    rows = int(sqrt(num))
    cols = int(num / rows + 0.5)
    f, axarr = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            ax = axarr[row][col]
            index = row * rows + col
            ax.imshow(digits_labels[index][0],
                    cmap='gray')
            ax.set_title(digits_labels[index][1], x=-0.1,y=0.5)
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
