class BaseModel(object):
    '''Base model class.

    Attributes
    ----------
    lr  :   float
            Learning rate for the optimizer
    optimizer   :   str
                    TF optimizer to use
    activation  :   function
                    tf.nn function for neuron activation
    '''

    def run_training_step(self, session, data, labels):
        '''Run forward pass through net and apply gradients once.

        Parameters
        ----------
        session :   tf.Session
                    Session to use for executing everything
        data    :   np.ndarray
                    Input data
        labels  :   np.ndarray
                    Input labels
        '''
        pass

    def get_accuracy(self, session, data, labels):
        '''Run forward pass through net and compute accuracy.

        Parameters
        ----------
        session :   tf.Session
                    Session to use for executing everything
        data    :   np.ndarray
                    Input data
        labels  :   np.ndarray
                    Input labels

        Returns
        -------
        float
        '''
        pass

    def __init__(self, learning_rate, optimizer, activation):
        '''Init new model

        Parameters
        ----------
        lr  :   float
                Learning rate for the optimizer
        optimizer   :   str
                        TF optimizer to use
        activation  :   function
                        tf.nn function for neuron activation
        '''
        self.lr = learning_rate
        self.opt = optimizer
        self.act_fn = activation
