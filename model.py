class BaseModel(object):
    '''Base model class.

    Attributes
    ----------
    opt   :   str
                    TF optimizer to use
    act_fn  :   function
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

    def predict(self, session, data):
        pass

    def __init__(self, optimizer, activation):
        '''Init new model

        Parameters
        ----------
        lr  :   float
                Learning rate for the optimizer
        optimizer   :   tf.train.Optimizer
                        TF optimizer to use
        activation  :   function
                        tf.nn function for neuron activation
        '''
        self.opt = optimizer
        self.act_fn = activation
