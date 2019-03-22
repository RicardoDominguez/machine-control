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
        self.ac_ub = control_cfg.ctrl_cfg.ac_ub
        self.ac_lb = control_cfg.ctrl_cfg.ac_lb

        self.n_parts = control_cfg.n_parts

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
        action = np.random.rand(states.shape[0], 2)*(self.ac_ub-self.ac_lb)+self.ac_lb
        self.action_traj.append(action)
        print("Action selected", action)
        return action

    def initAction(self):
        action = np.random.rand(self.n_parts, 2)*(self.ac_ub-self.ac_lb)+self.ac_lb
        print("Action selected", action)
        return action

    def log(self):
        N, M = len(self.state_traj), self.state_traj[0].shape[0]
        state_log, action_log = np.zeros((N, M, 16)), np.zeros((N, M, 2))
        for i in range(N):
            state_log[i, :, :] = self.state_traj[i]
            action_log[i, :, :] = self.action_traj[i]
        np.save("state_log.npy", state_log)
        np.save("action_log.npy", action_log)

    def loop(self):
        self.sendAction(self.initAction())
        while(True):
            states = self.getStates()
            actions = self.computeAction(states)
            self.sendAction(actions)
            self.log()

if __name__ == '__main__':
    from config_windows import returnSharedCfg
    from config_cluster import returnClusterPretrainedCfg

    s_cfg = returnSharedCfg()
    c_cfg = returnClusterPretrainedCfg()

    cluster = Cluster(s_cfg, c_cfg)
    cluster.loop()
