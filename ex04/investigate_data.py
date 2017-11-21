import numpy as np
import tensorflow as tf
from svhn_helper import SVHN
from matplotlib import pyplot as plt

if __name__ == "__main__":
    print("Starting SVHN Training")

    svhn_helper = SVHN()
    
    num_investigations = 16

    test_batch = list(zip(svhn_helper._training_data[:16], svhn_helper._training_labels[:16]))

    categories = ['0','1','2','3','4','5','6','7','8','9','0']
    f, axarr = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            index = 4*i+j
            ax = axarr[i][j]
            img = test_batch[index][0].reshape((32,32))
            label = test_batch[index][1]
            ax.set_title(categories[label])
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()




