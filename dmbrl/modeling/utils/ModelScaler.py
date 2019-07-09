import tensorflow as tf
import numpy as np

from .TensorStandardScaler import TensorStandardScaler, TensorStandardScaler1D

class ModelScaler:
    """Normalise the inputs and outputs to the NN model.
    """
    def __init__(self, xdim=None):
        """ xdim for consistency, not use """
        self.stateScaler = TensorStandardScaler1D(name=0)
        self.actionScaler = TensorStandardScaler(2, name=1)
        self.targetScaler = TensorStandardScaler1D(name=2)

    def fit(self, inputs, targets):
        """Fits the scaler to the model inputs and targets.

        Arguments:
            inputs (np.array or tf.Tensor): shape N x 16 + 2
            targets (np.array or tf.Tensor): shape N x 16
        """
        self.stateScaler.fit(inputs[:,:-2])
        self.actionScaler.fit(inputs[:,-2:])
        self.targetScaler.fit(targets)

    def transformInput(self, input):
        """Normalises the inputs to the NN model.

        Arguments:
            input (np.array or tf.Tensor): shape N x nS + nU or M x N x nS + nU

        Returns:
            np.array or tf.Tensor: shape N x nS + nU or M x N x nS + nU
        """
        if len(input.shape) == 2:
            state = self.stateScaler.transform(input[:,:-2])
            action = self.actionScaler.transform(input[:,-2:])
        else:
            state = self.stateScaler.transform(input[:,:,:-2])
            action = self.actionScaler.transform(input[:,:,-2:])
        return tf.concat([state, action], -1)

    def transformTarget(self, targets):
        """Normalises the targets to the NN model.

        Arguments:
            targets (np.array or tf.Tensor): shape N x nS

        Returns:
            np.array or tf.Tensor: shape N x nS
        """
        return self.targetScaler.transform(targets)

    def inverse_transformInput(self, input):
        """Returns the inverse transform of the inputs

        Arguments:
            input (np.array or tf.Tensor): shape N x nS + nU or M x N x nS + nU

        Returns:
            np.array or tf.Tensor: shape N x nS + nU or M x N x nS + nU
        """
        if len(input.shape) == 2:
            state = self.stateScaler.inverse_transform(input[:,:-2])
            action = self.actionScaler.inverse_transform(input[:,-2:])
        else:
            state = self.stateScaler.inverse_transform(input[:,:,:-2])
            action = self.actionScaler.inverse_transform(input[:,:,-2:])
        return tf.concat([state, action], -1)

    def inverse_transformOutput(self, mean, variance):
        """Normalises the inverse transform of the targets to the NN model.

        Arguments:
            mean (np.array or tf.Tensor): shape N x nS
            variance (np.array or tf.Tensor): shape N x nS

        Returns:
            np.array or tf.Tensor: shape N x nS
        """
        variance *= tf.square(self.targetScaler.sigma)
        return self.targetScaler.inverse_transform(mean), variance

    def get_vars(self):
        """Returns the tf.variables of the scaler objects used"""
        return self.stateScaler.get_vars() + self.actionScaler.get_vars() + self.targetScaler.get_vars()
