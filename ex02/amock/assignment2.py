import tensorflow as tf
import numpy as np
from ext.mnist_helper import MNIST
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == "__main__":
    mnist_loader = MNIST("res")

    training_batch = mnist_loader.get_training_batch(1)

    num_trained = 0

    images = []
    labels = []

    num_train = 1

    for (image, label) in training_batch:
        images.append(np.squeeze(image, axis=0) )
        labels.append(np.squeeze(label,axis=0) )
        
        num_trained = num_trained + 1
        if(num_trained >= num_train):
            break

    for i,image in enumerate(images):
        plt.imshow(image)
        plt.title(str(labels[i]))
        
    plt.show()
