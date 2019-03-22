"""
Cluster side software
"""

import os
import numpy as np

class Cluster:
    def __init__(self, shared_cfg, control_cfg):
        self.s_cfg = shared_cfg
        self.c_cfg = control_cfg

        self.state_traj = []
        self.action_traj = []
        self.n_pieces = 8
        self.a_l = np.array([0.57, 75])
        self.a_u = np.array([1.8, 140])

        self.n_parts = 4 # GET THIS DIRECTLY FROM CONFIG

        self.action_state = 0 # slow, fast
        self.n_states = 4

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
        action = np.zeros((states.shape[0], 2))

        use_state = self.action_state
        for i in range(self.n_parts):
            if use_state == 0:
                action[i,0] = 0.57
                action[i,1] = 75
            elif use_state == 1:
                action[i,0] = 1.8
                action[i,1] = 75
            elif use_state == 2:
                action[i,0] = 0.57
                action[i,1] = 140
            elif use_state == 3:
                action[i,0] = 1.8
                action[i,1] = 140
            use_state = (use_state+1) % self.n_states
        self.action_state = (self.action_state+1)%self.n_states

        print("Action selected", action)
        return action
        
    def initAction(self):
        action = np.zeros((self.n_parts, 2))

        use_state = self.action_state
        for i in range(self.n_parts):
            if use_state == 0:
                action[i,0] = 0.57
                action[i,1] = 75
            elif use_state == 1:
                action[i,0] = 1.8
                action[i,1] = 75
            elif use_state == 2:
                action[i,0] = 0.57
                action[i,1] = 140
            elif use_state == 3:
                action[i,0] = 1.8
                action[i,1] = 140
            use_state = (use_state+1) % self.n_states
        self.action_state = (self.action_state+1)%self.n_states

        print("Action selected", action)
        return action
        

    def loop(self):
        self.sendAction(self.initAction())
        while(True):
            states = self.getStates()
            actions = self.computeAction(states)
            self.sendAction(actions)
