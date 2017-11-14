import numpy as np
from matplotlib import pyplot as plt
from cifar_helper import CIFAR

cifar = CIFAR()
test_batch = list(zip(cifar._training_data[:16], cifar._training_labels[:16]))
categories = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse',
        'Ship', 'Truck']
f, axarr = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        index = 4*i+j
        ax = axarr[i][j]
        img = test_batch[index][0]
        label = test_batch[index][1]
        ax.set_title(categories[label])
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()

