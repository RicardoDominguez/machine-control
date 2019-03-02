"""
Cluster side software
"""

import os
import numpy as np

class Cluster:
    def __init__(self, shared_cfg):
        self.s_cfg = shared_cfg

    # --------------------------------------------------------------------------
    # COMMS FUNCTIONS
    # --------------------------------------------------------------------------

    def getStates(self):
        """Read current system state outputted by machine"""
        print('Waiting for states...')
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.state
        rdy = dir + cfg.rdy_name

        # Wait until RDY signal is provided
        while(not os.path.isdir(rdy)): pass
        os.rmdir(rdy) # Delete RDY

        # Read data to array
        states = np.load(dir+cfg.f_name)
        print('States received')
        return states

    def sendAction(self, actions):
        """Send computed action to machine side"""
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.action

        # Write actions into npy file
        np.save(dir+cfg.f_name, actions)
        os.mkdir(dir+cfg.rdy_name) # RDY signal
        print('Actions saved')

    def computeAction(self, states):
        """Return control action given the current machine state"""
        return states

    def loop(self):
        while(True):
            states = self.getStates()
            actions = self.computeAction(states)
            self.sendAction(actions)