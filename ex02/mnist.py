import struct
import os
from urllib.request import urlopen
from urllib.parse import urlparse, urljoin
import numpy as np
import gzip
from gzip import GzipFile
# make all of ths reproducible
np.random.seed(1)


class MNISTLoader():
    '''``MNISTLoader`` class can load MNIST dataset from the web or disk.

    Attributes
    ----------
    training_data_name  :   str
                            Name of the train data file
    test_data_name  :   str
                        Name of the test data file
    training_labels_name    :   str
                                Name of the train labels file
    test_labels_name    :   str
                            Name of the test labels file
    data_folder :   str
                    Name of the folder to save data in
    training_data   :   np.ndarray
                        Numpy array with the training data
    test_data   :   np.ndarray
                    Numpy array with the test data
    training_labels :   np.ndarray
                        Training labels
    test_labels :   np.ndarray
                    Test labels
    '''
    def __init__(self, directory='data', base_link='http://yann.lecun.com/exdb/mnist/'):
        '''Initialise loader.

        Parameters
        ----------
        directory   :   str
                        Directory to save downloaded data to
        base_link   :   str
                        Uri forming the base for all data parts. File names will
                        be appended to this base
        '''
        self.training_data_name = 'train-images-idx3-ubyte.gz'
        self.training_labels_name = 'train-labels-idx1-ubyte.gz'
        self.test_data_name = 't10k-images-idx3-ubyte.gz'
        self.test_labels_name = 't10k-labels-idx1-ubyte.gz'

        self.data_folder = directory

        # if folder doesn't exist yet, create it and download
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
            # helper to shorten url creation
            url = lambda name: urljoin(base_link, name)
            self.test_data = self._load(url(self.test_data_name), False, True)
            self.test_labels = self._load(url(self.test_labels_name),
                                          True, True)
            self.training_data = self._load(url(self.training_data_name),
                                            False, True)
            self.training_labels = self._load(url(self.training_labels_name),
                                              True, True)
        # else simply load from disk
        else:
            # helper to shorten path creation
            path = lambda name: os.path.join(directory, name)
            self.test_data = self._load(path(self.test_data_name))
            self.test_labels = self._load(path(self.test_labels_name), True)
            self.training_data = self._load(path(self.training_data_name))
            self.training_labels = self._load(path(self.training_labels_name), True)

    def _load(self, path_or_url, labels=False, save=False):
        '''Load the MNIST data set, either from disk or from the web.

        Parameters
        ----------
        path_or_url :   str
                        Url of the file, or filesystem path
        labels  :   bool
                    Whether or not the file contains labels as opposed to data
        save    :   bool
                    Serialize the data to disk (useless if fetching locally)

        Returns
        -------
        np.ndarray
                Data (in shape (N, 28, 28)) or labels (in shape (N))
        '''

        # we unify loading from web and file system by creating file:// uris for
        # local files and just using urllib on them as well
        parse_result = urlparse(path_or_url)
        if not parse_result.scheme:
            # looks like a file
            if not os.path.exists(path_or_url):
                raise RuntimeError('Found data directory, but not %s in it. '
                                   'Remove it and restart' % path_or_url)
            path_or_url = 'file://' + os.path.abspath(path_or_url)
        else:
            print('Downloading from web...')
        print(path_or_url)
        # stream over tcp/file
        with urlopen(path_or_url) as request_stream:
            zip_file = GzipFile(fileobj=request_stream, mode='rb')
            zip_name = os.path.join(self.data_folder, os.path.basename(path_or_url))
            if save:
                # first save the file
                with gzip.open(zip_name, mode='wb') as f:
                    f.write(zip_file.read())
            # then read it back in and fill data
            # note we cannot simply seek(0) above since this isn't a real file
            # but a web resource
            with gzip.open(zip_name, mode='rb') as fd:
                # first unpack magic numer and number of elements (4 bytes each)
                magic, numberOfItems = struct.unpack('>ii', fd.read(2 * 4))
                if (not labels and magic != 2051) or (labels and magic != 2049):
                    raise LookupError('Not a MNIST file')
                if not labels:
                    # then unpack format
                    rows, cols = struct.unpack('>II', fd.read(8))
                    # to use np.frombuffer, we need a bytearray, doesn't work
                    # directly from file
                    b = bytearray(fd.read())
                    images = np.frombuffer(b, dtype='uint8')
                    images = images.reshape((numberOfItems, rows, cols))
                    return images
                else:
                    b = bytearray(fd.read())
                    labels = np.frombuffer(b, dtype='uint8')
                    return labels

    def batches(self, data, labels, batch_size):
        '''Generate a set of random minibatches from the given data.

        Parameters
        ----------
        data    :   np.ndarray
                    Data array
        lablels :   np.ndarray
                    Labels vectory
        batch_size  :   int
                        Size of the minibatches (must be > 0)

        Yields
        ------
        tuple
            Tuples of (data_batch, labels_batch), the union of which exactly
            equals the data set
        '''
        samples_n = labels.shape[0]
        if batch_size <= 0:
            batch_size = samples_n

        random_indices = np.random.choice(samples_n, samples_n, replace=False)
        data = data[random_indices]
        labels = labels[random_indices]
        for i in range(samples_n // batch_size):
            on = i * batch_size
            off = on + batch_size
            yield data[on:off], labels[on:off]


if __name__ == '__main__':
    loader = MNISTLoader()
