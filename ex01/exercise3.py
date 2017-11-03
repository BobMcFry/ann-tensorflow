# Ex 3
# Copyright 2017 Rasmus

# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Rasmus wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer or coffee in return

import numpy as np
from numpy.random import normal
from matplotlib import pyplot as plt
np.random.seed(1)

sample_size = 60

# Note to the reader: Exercise sheet specifies to use a variance of V in the text, but then uses an
# std of V in the code

def dog_distribution(shape):
    return normal(45, 15, shape)

def cat_distribution(shape):
    return normal(25, 5, shape)

x_cat = cat_distribution((2, sample_size//2))
x_dog = dog_distribution((2, sample_size//2))

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

f = plt.figure()
ax = f.add_subplot(1,1,1)
ax.set_title('Data distribution')
ax.set_xlabel('Length')
ax.set_ylabel('Height')
ax.scatter(x_cat[0,:], x_cat[1, :], c='blue')
ax.scatter(x_dog[0,:], x_dog[1, :], c='orange')

plt.show()
