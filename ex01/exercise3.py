# Ex 3
# Copyright 2017 Rasmus

# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Rasmus wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer or coffee in return

import numpy as np
from numpy.random import normal
from scipy.special import expit as logistic
from matplotlib import pyplot as plt
np.random.seed(1)

sample_size = 60


# Note to the reader: Exercise sheet specifies to use a variance of V in the text, but then uses an
# std of V in the code
def dog_distribution(shape):
    return normal(45, 15, shape)


def cat_distribution(shape):
    return normal(25, 5, shape)


W = np.array([[-2.5, 2.5]])


def logistic_derivative(batch):
    e = np.exp(batch)
    return e / np.square(1 + e)


def forward_pass(batch):
    assert batch.shape[1] == 2
    return logistic(np.dot(W, batch))


def loss_function(predictions, targets):
    return 0.5 * np.sum(np.square(predictions - targets))


def gradient_loss(batch, targets):
    assert batch.shape[1] == 2
    e = np.exp(-np.dot(W, batch))
    term_1 = targets - forward_pass(batch)
    term_2 = - logistic(batch)
    term_3 = batch
    grad = term_1 * term_2 * term_3
    print('gradient shape: %s' % grad.shape)
    return grad


def batches(data, targets, size):
    assert size >= 1
    assert data.shape[1] == 2
    data_len = data.shape[0]
    data_targets = np.concatenate(data, targets, axis=1)
    np.random.shuffle(data_targets)
    data = data_targets[:, :2]
    targets = data_targets[:, 2]
    for i in range(0, data_len // size):
        start = i * size
        end = min(start + size, data_len)
        yield data[start:end, :], targets[start:end, :]


def train(data, targets):
    epochs = 10
    learning_rate = 0.01
    for epoch in range(epochs):
        for batch, batch_targets in batches(data, targets, 10):
            gradient = gradient_loss(batch, batch_targets)
            global W
            W = W - learning_rate * gradient
        error = loss_function(forward_pass(data), targets)
        print(error)


def main():
    ################################################################################################
    #                                        Generate data                                         #
    ################################################################################################
    x_cat = cat_distribution((sample_size//2, 2))
    x_dog = dog_distribution((sample_size//2, 2))

    ################################################################################################
    #                                        Normalize data                                        #
    ################################################################################################
    # Shouldn't this be done per dimension?
    mean = np.mean([x_cat, x_dog])
    std = np.std([x_cat, x_dog])

    # not equivalent ???????
    # x_cat -=  mean
    # x_dog -= mean
    # x_cat /= std
    # x_dog /= std
    x_cat = (x_cat - mean) / std
    x_dog = (x_dog - mean) / std

    ################################################################################################
    #                                          Plot data                                           #
    ################################################################################################
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.set_title('Data distribution')
    ax.set_xlabel('Length')
    ax.set_ylabel('Height')
    ax.scatter(x_cat[:, 0], x_cat[:, 1], c='blue')
    ax.scatter(x_dog[:, 0], x_dog[:, 1], c='orange')

    plt.show()

    ################################################################################################
    #                                       Generate targets                                       #
    ################################################################################################
    data = np.concatenate((x_cat, x_dog), axis=0)
    # 1 = Cat, 0 = Dog
    targets_cats = np.ones((sample_size//2, 1))
    targets_dogs = np.zeros((sample_size//2, 1))
    targets = np.concatenate((targets_cats, targets_dogs), axis=0)

    train(data, targets)

if __name__ == "__main__":
    main()
