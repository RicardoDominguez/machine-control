"""
Cluster side software
"""

import os
import numpy as np

class Cluster:
    def __init__(self, shared_cfg, contol_cfg):
        self.s_cfg = shared_cfg
        self.c_cfg = control_cfg

        self.state_traj = []
        self.action_traj = []
        self.n_pieces = 8
        self.a_l = np.array([0.57, 75])
        self.a_u = np.array([1.8, 140])

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
        self.state_traj.append(states)
        action = np.random.rand(self.n_pieces, 2)*(self.a_u-self.a_l)+self.a_l
        print("Action selected", action)
        return action

    def loop(self):
        while(True):
            states = self.getStates()
            actions = self.computeAction(states)
            self.sendAction(actions)