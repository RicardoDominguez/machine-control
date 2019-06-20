"""
Cluster side software
"""

import os
import numpy as np
from dmbrl.controllers.MPC import MPC

class Cluster:
    def __init__(self, shared_cfg, pretrained_cfg, learned_cfg):
        self.s_cfg = shared_cfg
        self.c_pre_cfg = pretrained_cfg
        self.c_ler_cfg = learned_cfg

        self.policyPret = MPC(pretrained_cfg.ctrl_cfg)
        self.policyLear = MPC(learned_cfg.ctrl_cfg)

        self.t = 0
        self.H = shared_cfg.env.horizon
        self.train_freq = learned_cfg.train_freq

        self.n_parts = shared_cfg.env.n_parts
        self.n_parts_pretrained = pretrained_cfg.n_parts
        self.n_parts_learned = learned_cfg.n_parts
        assert self.n_parts_pretrained+self.n_parts_learned == self.n_parts

        self.pret_state_traj = np.zeros((self.n_parts_pretrained, self.H, shared_cfg.env.nS))
        self.pret_action_traj = np.zeros((self.n_parts_pretrained, self.H, 2))
        self.lear_state_traj = np.zeros((self.n_parts_learned, self.H, shared_cfg.env.nS))
        self.lear_action_traj = np.zeros((self.n_parts_learned, self.H, 2))
        self.pred_cost_pret = np.zeros((self.H, self.n_parts_pretrained))
        self.pred_cost_lear = np.zeros((self.H, self.n_parts_learned))

        self.save_dirs = [shared_cfg.save_dir1, shared_cfg.save_dir2]

        self.clearComms()

    # --------------------------------------------------------------------------
    # COMMS FUNCTIONS
    # --------------------------------------------------------------------------
    def clearComms(self):
        cfg = self.s_cfg.comms
        dir_action = cfg.dir+cfg.action.rdy_name
        dir_state = cfg.dir+cfg.state.rdy_name
        if os.path.isdir(dir_action): os.rmdir(dir_action)
        if os.path.isdir(dir_state): os.rmdir(dir_state)

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
        self.pret_state_traj[:, self.t, :] = states[:self.n_parts_pretrained, :]
        self.lear_state_traj[:, self.t, :] = states[self.n_parts_pretrained:, :]

        # At least one part learned, not trained first step
        if self.n_parts_learned > 0 and self.t!=0 and (self.t)%self.train_freq==0:
            print("Training model...")
            obs_in = self.lear_state_traj[:, self.t-self.train_freq:self.t, :].reshape(-1, self.lear_state_traj.shape[-1])
            obs_out = self.lear_state_traj[:, self.t-self.train_freq+1:self.t+1, :].reshape(-1, self.lear_state_traj.shape[-1])
            acs = self.lear_action_traj[:, self.t-self.train_freq:self.t, :].reshape(-1, self.lear_action_traj.shape[-1])
            self.policyLear.train(obs_in, obs_out, acs)

        action = np.zeros((self.s_cfg.env.n_parts, 2))
        for part in range(self.s_cfg.env.n_parts):
            print("Sampling actions %d/%d" % (part, self.s_cfg.env.n_parts))
            if part < self.n_parts_pretrained: # Pretrained policy
                action[part, :], self.pred_cost_pret[self.t,part] = self.policyPret.act(states[part, :], self.t, get_pred_cost=True)
            else: # Learned policy
                if self.t < self.train_freq: # Do not predict cost
                    action[part, :] = self.policyLear.act(states[part, :], self.t, get_pred_cost=False)
                    self.pred_cost_lear[self.t,part-self.n_parts_pretrained] = 0
                else:
                    action[part, :], self.pred_cost_lear[self.t,part-self.n_parts_pretrained] = \
                        self.policyLear.act(states[part, :], self.t, get_pred_cost=True)

        self.pret_action_traj[:, self.t, :] = action[:self.n_parts_pretrained, :]
        self.lear_action_traj[:, self.t, :] = action[self.n_parts_pretrained:, :]

        self.t+=1

        return action

    def initAction(self):
        return np.ones((self.s_cfg.env.n_parts, 2)) * self.s_cfg.env.init_params

    def log(self):
        for i in range(len(self.save_dirs)):
            np.save(self.save_dirs[i]+"pret_state_traj.npy", self.pret_state_traj)
            np.save(self.save_dirs[i]+"pret_action_traj.npy", self.pret_action_traj)
            np.save(self.save_dirs[i]+"pret_pred_cost.npy", self.pred_cost_pret)
            np.save(self.save_dirs[i]+"lear_state_traj.npy", self.lear_state_traj)
            np.save(self.save_dirs[i]+"lear_action_traj.npy", self.lear_action_traj)
            np.save(self.save_dirs[i]+"lear_pred_cost.npy", self.pred_cost_lear)

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
    cp_cfg = returnClusterPretrainedCfg() # Pretrained
    cl_cfg = returnClusterUnfamiliarCfg() # Learned

    cluster = Cluster(s_cfg, cp_cfg, cl_cfg)
    cluster.loop()
