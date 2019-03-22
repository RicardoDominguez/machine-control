"""
Cluster side software
"""

import os
import numpy as np
from dmbrl.controllers.MPC import MPC

class Cluster:
    def __init__(self, shared_cfg, control_cfg):
        self.s_cfg = shared_cfg
        self.c_cfg = control_cfg

        self.policy = MPC(control_cfg.ctrl_cfg)

        self.t = 0
        self.H = shared_cfg.env.horizon
        self.train_freq = control_cfg.train_freq

        self.state_traj = np.zeros((shared_cfg.env.n_parts, self.H, shared_cfg.env.nS))
        self.action_traj = np.zeros((shared_cfg.env.n_parts, self.H, 2))
        self.pred_cost = np.zeros((self.H, shared_cfg.env.n_parts))

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

    # --------------------------------------------------------------------------
    # SAMPLE ACTIONS
    # --------------------------------------------------------------------------
    def computeAction(self, states):
        """Return control action given the current machine state"""
        cfg = c_cfg

        self.state_traj[:, self.t, :] = states
        self.get_pred_cost = not ((not cfg.pretrained) and (self.t < self.train_freq))

        # Pretrained models do not learn. Cannot train first step.
        if not cfg.pretrained and self.t!=0 and (self.t)%self.train_freq==0:
            print("Training model...")
            obs_in = self.state_traj[:, self.t-self.train_freq:self.t, :].reshape(-1, self.state_traj.shape[-1])
            obs_out = self.state_traj[:, self.t-self.train_freq+1:self.t+1, :].reshape(-1, self.state_traj.shape[-1])
            acs = self.action_traj[:, self.t-self.train_freq:self.t, :].reshape(-1, self.action_traj.shape[-1])
            self.policy.train(obs_in, obs_out, acs)

        print("Sampling actions")
        action = np.zeros((self.s_cfg.env.n_parts, 2))
        for part in range(self.s_cfg.env.n_parts):
            if self.get_pred_cost:
                action[part, :], self.pred_cost[self.t,part] = self.policy.act(
                                                                states[part, :], self.t,
                                                                get_pred_cost=True)
            else:
                action[part, :] = self.policy.act(states[part, :], self.t, get_pred_cost=False)
                self.pred_cost[self.t,part] = 0

        self.action_traj[:, self.t, :] = action
        self.t+=1

        return action

    def initAction(self):
        # Init with random actions
        states = np.zeros((self.s_cfg.env.n_parts, 16))
        action = np.zeros((self.s_cfg.env.n_parts, 2))
        for part in range(self.s_cfg.env.n_parts):
            action[part, :] = self.policy.act(states[part, :], self.t, get_pred_cost=False)
        return action

    def log(self):
        np.save("state_traj.npy", self.state_traj)
        np.save("action_traj.npy", self.action_traj)
        np.save("pred_cost.npy", self.pred_cost)

    def loop(self):
        self.sendAction(self.initAction())
        while self.t < self.H:
            states = self.getStates()
            actions = self.computeAction(states)
            self.sendAction(actions)
            self.log()

if __name__ == '__main__':
    from config_windows import returnSharedCfg
    from config_cluster import returnClusterPretrainedCfg, returnClusterUnfamiliarCfg

    s_cfg = returnSharedCfg()
    c_cfg = returnClusterPretrainedCfg()

    cluster = Cluster(s_cfg, c_cfg)
    cluster.loop()
