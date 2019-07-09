from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class Optimizer:
    """Framework for Optimizer subclasses"""
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function, tf_compatible):
        """Function called upon initialisation of the MPC class."""
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """Function iteratively called at MPC.act()"""
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        """Compute optimisation problem solution"""
        raise NotImplementedError("Must be implemented in subclass.")
