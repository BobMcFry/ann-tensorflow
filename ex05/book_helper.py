import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import os, pickle

class Book:
    def __init__(self, file_name='10.txt.utf-8'):
        pickle_name = 'book_text.pckl'
        if os.path.exists(pickle_name):
            with open(pickle_name, 'rb') as f:
                self._book_text = pickle.load(f)
        else:
            with open(file_name) as f:
                book = f.read()
                book = book.lower()
                tokenizer = RegexpTokenizer(r'\w+')
                book = [w.lower() for w in tokenizer.tokenize(book)]
                self._book_text = book
            with open(pickle_name, 'wb') as f:
                pickle.dump(book, f)

        print("Unique words: " + str(len(Counter(self._book_text))))


    def create_dictionaries(self, vocabulary_size):
        '''Create a dict mapping words to word ids. It works by finding the
        `vocabulary_size` most common words and assigning ids based on the
        frequency. The most common word will have id 1, the second most common
        one id 2, and so forth. All other words (with the key "UNKNOWN") will
        have id 0.

        Parameters
        ----------
        vocabulary_size :   int
                            Number of most common words to create ids for
        '''
        words_and_count = Counter(self._book_text).most_common(vocabulary_size - 1)

        word2id = {word.lower(): word_id for word_id, (word, _) in enumerate(words_and_count, 1)}
        word2id["UNKNOWN"] = 0

        id2word = dict(zip(word2id.values(), word2id.keys()))

        # Map words to ids
        self._book = [word2id.get(word, 0) for word in self._book_text]

        self._word2id = word2id
        self._id2word = id2word


    def words2ids(self, words):
        if type(words) == list or type(words) == range or type(words) == np.ndarray:
            return [self._word2id.get(word.lower(), 0) for word in words]
        else:
            return self._word2id.get(words, 0)

    def ids2words(self, ids):
        if type(ids) == list or type(ids) == range or type(ids) == np.ndarray:
            return [self._id2word.get(wordid, "UNKNOWN") for wordid in ids]
        else:
            return self._id2word.get(ids, 0)


    def get_training_batch(self, batch_size, skip_window):
        valid_indices = range(skip_window, len(self._book) - (skip_window + 1))
        context_range = [x for x in range(-skip_window, skip_window + 1) if x != 0]
        wordid_contextid_pairs = [(word_id, word_id + shift) for word_id in
                valid_indices for shift in context_range]

        np.random.shuffle(wordid_contextid_pairs)

        counter = 0
        words = np.zeros((batch_size), dtype = np.int32)
        contexts = np.zeros((batch_size, 1), dtype = np.int32)

        for word_index, context_index in wordid_contextid_pairs:
            words[counter] = self._book[word_index]
            contexts[counter, 0] = self._book[context_index]
            counter += 1

            if counter == batch_size:
                yield words, contexts
                counter = 0

