import tensorflow as tf
import numpy as np

from .TensorStandardScaler import TensorStandardScaler, TensorStandardScaler1D

class ModelScaler:
    def __init__(self, xdim=None):
        """ xdim for consistency, not use """
        self.stateScaler = TensorStandardScaler1D(name=0)
        self.actionScaler = TensorStandardScaler(2, name=1)
        self.targetScaler = TensorStandardScaler1D(name=2)

    def fit(self, inputs, targets):
        """
        Inputs : np.array, shape N x 16 + 2
        Targets : np.array, shape N x 16
        """
        self.stateScaler.fit(inputs[:,:-2])
        self.actionScaler.fit(inputs[:,-2:])
        self.targetScaler.fit(targets)

    def transformInput(self, input):
        """ input: tf.tensor, shape N x 16 + 2 or M x N x 16 + 2"""
        if len(input.shape) == 2:
            state = self.stateScaler.transform(input[:,:-2])
            action = self.actionScaler.transform(input[:,-2:])
        else:
            state = self.stateScaler.transform(input[:,:,:-2])
            action = self.actionScaler.transform(input[:,:,-2:])
        return tf.concat([state, action], -1)

    def transformTarget(self, targets):
        """ Target: tf.tensor, shape N x 16 """
        return self.targetScaler.transform(targets)

    def inverse_transformInput(self, input):
        """ input: tf.tensor, shape N x 16 + 2 """
        if len(input.shape) == 2:
            state = self.stateScaler.inverse_transform(input[:,:-2])
            action = self.actionScaler.inverse_transform(input[:,-2:])
        else:
            state = self.stateScaler.inverse_transform(input[:,:,:-2])
            action = self.actionScaler.inverse_transform(input[:,:,-2:])
        return tf.concat([state, action], -1)

    def inverse_transformOutput(self, mean, variance):
        """ Target: tf.tensor, shape N x 16 """
        variance *= tf.square(self.targetScaler.sigma)
        return self.targetScaler.inverse_transform(mean), variance

    def get_vars(self):
        return self.stateScaler.get_vars() + self.actionScaler.get_vars() + self.targetScaler.get_vars()

    def cache(self):
        raise NotImplementedError

    def load_cache(self):
        raise NotImplementedError


