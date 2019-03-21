"""
Cluster side software
"""

import os
import numpy as np

class Cluster:
    def __init__(self, shared_cfg, contol_cfg):
        self.s_cfg = shared_cfg
        self.c_cfg = control_cfg

        self.policy = MPC(cfg.ctrl_cfg)

        self.t = 0
        self.H = shared_cfg.horizon

        self.state_traj = np.zeros((shared_cfg.env.n_parts, shared_cfg.horizon, shared_cfg.env.nS))
        self.action_traj = np.zeros((shared_cfg.env.n_parts, shared_cfg.horizon, 2))
        self.pred_cost = np.zeros((shared_cfg.horizon))

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

        # Pretrained models do not learn. Cannot train first step.
        if not cfg.pretrained and self.t!=0 and (self.t)%self.train_freq==0:
            print("Training model...")
            obs = self.state_traj[:, self.t-self.train_freq:self.t+1, :].reshape(-1, self.state_traj.shape[-1])
            acs = self.action_traj[:, self.t-self.train_freq:self.t, :].reshape(-1, self.action_traj.shape[-1])
            self.policy.train(obs[:-1, :], obs[1:, :], acs)

        print("Sampling actions")
        action = np.zeros((self.s_cfg.n_parts, 2))
        for part in range(self.s_cfg.n_parts):
            action[part, :], pred_cost[part] = self.policy.act(states[part, :], self.t,
                                              get_pred_cost=True)

        self.action_traj[:, self.t, :] = action
        self.t+=1

        return action

    def saveStateVars(self):
        np.save("state_traj.npy", self.state_traj)
        np.save("action_traj.npy", self.action_traj)
        np.save("pred_cost.npy", self.pred_cost)

    def loop(self):
        while self.t < self.H:
            states = self.getStates()
            actions = self.computeAction(states)
            self.sendAction(actions)
            self.saveStateVars()
