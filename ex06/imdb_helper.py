import os
import pickle
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import numpy as np

PICKLE_NAME = 'imdb_helper.pckl'

class IMDB:

    def load(self):
        '''Deserialise self from pickeld file.'''
        f = open(PICKLE_NAME, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self):
        '''Serialise self to pickeld file.'''
        f = open(PICKLE_NAME, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def __init__(self, directory):
        if os.path.exists(PICKLE_NAME):
            self.load()
        else:
            self._directory = directory

            self._training_data, self._training_labels = self._load_data('train')
            self._test_data, self._test_labels = self._load_data('test')

            np.random.seed(0)
            samples_n = self._training_labels.shape[0]
            random_indices = np.random.choice(samples_n, samples_n // 7, replace = False)
            np.random.seed()

            self._validation_data = self._training_data[random_indices]
            self._validation_labels = self._training_labels[random_indices]
            self._training_data = np.delete(self._training_data, random_indices, axis = 0)
            self._training_labels = np.delete(self._training_labels, random_indices)

            joined_written_ratings = [word for text in self._training_data for word in text]
            print('Unique words: ' + str(len(Counter(joined_written_ratings))))
            print('Mean length: ' + str(np.mean([len(text) for text in self._training_data])))
            self.save()


    def _load_data(self, data_set_type):
        data = []
        labels = []
        # Iterate over conditions
        for condition in ['neg', 'pos']:
            directory_str = os.path.join(self._directory, 'aclImdb', data_set_type, condition)
            directory = os.fsencode(directory_str)

            for file in os.listdir(directory):
                filename = os.fsdecode(file)

                label = 0 if condition == 'neg' else 1
                labels.append(label)

                # Read written rating from file
                with open(os.path.join(directory_str, filename)) as fd:
                    written_rating = fd.read()
                    written_rating = written_rating.lower()
                    tokenizer = RegexpTokenizer(r'\w+')
                    written_rating = tokenizer.tokenize(written_rating)
                    data.append(written_rating)

        return np.array(data), np.array(labels)

    def create_dictionaries(self, vocabulary_size, cutoff_length):
        if not hasattr(self, '_word2id'):
            joined_written_ratings = [word for text in self._training_data for word in text]
            words_and_count = Counter(joined_written_ratings).most_common(vocabulary_size - 2)

            word2id = {word: word_id for word_id, (word, _) in enumerate(words_and_count, 2)}
            word2id['_UNKNOWN_'] = 0
            word2id['_NOT_A_WORD_'] = 1

            id2word = dict(zip(word2id.values(), word2id.keys()))

            self._word2id = word2id
            self._id2word = id2word

            self._training_data = np.array([self.words2ids(text[:cutoff_length]) for text in self._training_data])
            self._validation_data = np.array([self.words2ids(text[:cutoff_length]) for text in self._validation_data])
            self._test_data = np.array([self.words2ids(text[:cutoff_length]) for text in self._test_data])
            self.save()


    def words2ids(self, words):
        if type(words) == list or type(words) == range or type(words) == np.ndarray:
            return [self._word2id.get(word, 0) for word in words]
        else:
            return self._word2id.get(words, 0)

    def ids2words(self, ids):
        if type(ids) == list or type(ids) == range or type(ids) == np.ndarray:
            return [self._id2word.get(wordid, '_UNKNOWN_') for wordid in ids]
        else:
            return self._id2word.get(ids, '_UNKNOWN_')


    def get_training_batch(self, batch_size):
        return self._get_batch(self._training_data, self._training_labels, batch_size)

    def get_validation_batch(self, batch_size):
        return self._get_batch(self._validation_data, self._validation_labels, batch_size)

    def get_test_batch(self, batch_size):
        return self._get_batch(self._test_data, self._test_labels, batch_size)

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


    def slice_batch(self, batch, slice_size):
        max_len = np.max([len(sample) for sample in batch])
        steps = int(np.ceil(max_len / slice_size))
        max_len = slice_size * steps

        # Resize all samples in batch to same size
        batch_size = len(batch)
        # fill buffer with _NOT_A_WORD_
        buffer = np.ones((batch_size, max_len), dtype = np.int32)
        for i, sample in enumerate(batch):
            buffer[i, 0:len(sample)] = sample

        for i in range(steps):
            on = i * slice_size
            off = on + slice_size
            yield buffer[:, on:off]


    def get_sizes(self):
        training_samples_n   = self._training_labels.shape[0]
        validation_samples_n = self._validation_labels.shape[0]
        test_samples_n       = self._test_labels.shape[0]
        return training_samples_n, validation_samples_n, test_samples_n

