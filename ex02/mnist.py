import struct
import os
from urllib.request import urlopen
from urllib.parse import urlparse, urljoin
import numpy as np
import gzip
from gzip import GzipFile
from io import BytesIO


class MNISTLoader():
    def __init__(self, directory=None, base_link='http://yann.lecun.com/exdb/mnist/'):
        training_data_name = 'train-images-idx3-ubyte.gz'
        training_labels_name = 'train-labels-idx1-ubyte.gz'
        test_data_name = 't10k-images-idx3-ubyte.gz'
        test_labels_name = 't10k-labels-idx1-ubyte.gz'
        if not directory:
            self.test_data = self._load(
                urljoin(base_link, test_data_name)
            )
            self.test_labels = self._load(urljoin(base_link, test_labels_name), True)
            self.training_data = self._load(urljoin(base_link, training_data_name))
            self.training_labels = self._load(urljoin(base_link, training_labels_name), True)
        else:
            self.test_data = self._load(os.path.join(directory, test_data_name))
            self.test_labels = self._load(os.path.join(directory,
                                                       test_labels_name), True)
            self.training_data = self._load(os.path.join(directory, training_data_name))
            self.training_labels = self._load(os.path.join(directory,
                                                           training_labels_name), True)

    def _load(self, path_or_url, labels=False):

        parse_result = urlparse(path_or_url)
        if not parse_result.scheme:
            # looks like a file
            path_or_url = 'file://' + os.path.abspath(path_or_url)
        else:
            print('Downloading from web...')
        print(path_or_url)
        with urlopen(path_or_url) as request_stream:
            zip_file = GzipFile(fileobj=request_stream, mode='rb')
            with zip_file as fd:
                magic, numberOfItems = struct.unpack('>ii', fd.read(8))
                if (not labels and magic != 2051) or (labels and magic != 2049):
                    raise LookupError('Not a MNIST file')

                if not labels:
                    rows, cols = struct.unpack('>II', fd.read(8))
                    b = bytearray(fd.read())
                    images = np.frombuffer(b, dtype='uint8')
                    images = images.reshape((numberOfItems, rows, cols))
                    return images
                else:
                    b = bytearray(fd.read())
                    images = np.frombuffer(b, dtype='uint8')
                    return labels


if __name__ == '__main__':
    loader = MNISTLoader()
