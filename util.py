import sys
import argparse
import importlib
import tensorflow as tf

class ParameterTest(object):
    '''Test one set of parameters to the train() function.'''
    def __init__(self, optimizer, learning_rate, batch_size, epochs,
            train_function):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.accuracy = None
        self.train_function = train_function

    def run(self):
        '''Run the training process with the specified settings.'''
        self.accuracy = self.train_function(optimizer=self.optimizer,
                learning_rate=self.learning_rate, batch_size=self.batch_size,
                epochs=self.epochs, return_records=False)

    def __str__(self):
        return ('{opti:30}, learning rate={lr:5.4f}, batch size={bs:<5d}, '
                'epochs={epochs:<5d}, accuracy={acc:4.3f}'.format(
                    opti=self.optimizer,
                    lr=self.learning_rate,
                    bs=self.batch_size,
                    epochs=self.epochs,
                    acc=self.accuracy
                )
        )


def main():
    tf_optimizers = {class_name[:-len('Optimizer')] for class_name in dir(tf.train) if 'Optimizer'
            in class_name and class_name != 'Optimizer'}
    parser = argparse.ArgumentParser(description='Test the net on one parameter set')
    parser.add_argument('-o', '--optimizer', required=True, type=str,
            choices=tf_optimizers, help='Optimization algorithm')
    parser.add_argument('-l', '--learning-rate', required=True, type=float,
            help='Learning rate for the optimizer')
    parser.add_argument('-b', '--batch-size', required=True, type=int,
            help='Batch size')
    parser.add_argument('-e', '--epochs', required=True, type=int,
            help='Number of epochs')
    parser.add_argument('-f', '--file', required=True, type=str,
            help='File to write result to')
    parser.add_argument('-m', '--module', required=True, type=str,
            help='Module to search for train() function.')

    args = parser.parse_args()
    train_fn = __import__(args.module, globals(), locals(), ['train']).train
    pt = ParameterTest(args.optimizer, args.learning_rate, args.batch_size,
        args.epochs, train_fn)
    pt.run()
    print(pt)
    # the OS ensures sequential writes
    with open(args.file, 'a') as f:
        f.write(str(pt) + '\n')
        f.flush()

if __name__ == '__main__':
    main()
