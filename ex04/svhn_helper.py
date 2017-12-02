import os
import numpy as np
import scipy.io as scio

class SVHN():
    def __init__(self, directory = "./"):
        self._directory = directory

        self._training_data = np.array([])
        self._training_labels = np.array([])
        self._test_data = np.array([])
        self._test_labels = np.array([])

        self._load_traing_data()
        #self._load_test_data()

        np.random.seed(0)
        samples_n = self._training_labels.shape[0]
        random_indices = np.random.choice(samples_n, samples_n // 10, replace = False)
        np.random.seed()

        self._validation_data = self._training_data[random_indices]
        self._validation_labels = self._training_labels[random_indices]
        self._training_data = np.delete(self._training_data, random_indices, axis = 0)
        self._training_labels = np.delete(self._training_labels, random_indices)
        ########################################################################
        #                            Stuff we tried                            #
        ########################################################################
        # 1.
        # simple mean-variance normalization doesn't add anything in face of
        # batch norm layers
        # training_data = (training_data - np.mean(training_data)) / np.var(training_data)
        # 2. Add data with random brightness and constrast
        # n = training_data.shape[0]
        # import tensorflow as tf
        # with tf.Session().as_default():
        #     data_random_brightness = tf.image.random_brightness(training_data[:n//2,...],
        #             0.5).eval()
        #     data_random_contrast = tf.image.random_contrast(training_data[n//2:,...], 0,
        #             0.5).eval()
        # training_data = np.concatenate((training_data, data_random_contrast,
        #     data_random_brightness))
        # training_labels = np.concatenate((training_labels,
        #     training_labels[:n//2],
        #     training_labels[n//2:]))

        # 3. Invert the training data so we don't get issues with white on
        # black.
        # inverted_train_data = np.zeros((10000, 32, 32, 1))
        # maxima = np.max(self._training_data[:10000], axis=(1,2,3))
        # inverted_train_data = np.subtract(maxima[:, np.newaxis, np.newaxis, np.newaxis], self._training_data[:10000],
        #     inverted_train_data)
        # self._training_data = np.concatenate((self._training_data,
        #     inverted_train_data))
        # self._training_labels = np.concatenate((self._training_labels,
        #     self._training_labels[:10000]))


    def _load_traing_data(self):
        training_data, training_labels = self._load_data("train_32x32.mat")

        self._training_data = training_data
        self._training_labels = training_labels

    def _load_test_data(self):
        self._test_data, self._test_labels = self._load_data("test_32x32.mat")

    def _rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def _load_data(self, file):
        path = os.path.join(self._directory, file)

        mat = scio.loadmat(path)
        data = np.moveaxis(mat["X"], 3, 0)
        data = self._rgb2gray(data)
        data = data.reshape(data.shape + (1,))

        labels = mat["y"].reshape(mat["y"].shape[0])
        labels[labels == 10] = 0

        return data, labels

    def get_training_batch(self, batch_size):
        return self._get_batch(self._training_data, self._training_labels, batch_size)

    def get_validation_batch(self, batch_size):
        return self._get_batch(self._validation_data, self._validation_labels, batch_size)

    #def get_test_batch(self, batch_size):
    #    return self._get_batch(self._test_data, self._test_labels, batch_size)

    def _get_batch(self, data, labels, batch_size):
        samples_n = labels.shape[0]

        if batch_size <= 0:
            batch_size = samples_n

        random_indices = np.random.choice(samples_n, samples_n, replace = False)
        data = data[random_indices]
        labels = labels[random_indices]
        for i in range(samples_n // batch_size):
            on = i * batch_size
            off = on + batch_size
            yield data[on:off], labels[on:off]


    def get_sizes(self):
        training_samples_n = self._training_labels.shape[0]
        validation_samples_n = self._validation_labels.shape[0]
        test_samples_n = self._test_labels.shape[0]
        return training_samples_n, validation_samples_n, test_samples_n


