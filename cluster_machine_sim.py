import os
import numpy as np
from dmbrl.controllers.MPC import MPC
from dotmap import DotMap
from sim_model.machine_model import MachineModelEnv

def obs_cost_fn(obs):
    target = 980
    k = 1000
    if isinstance(obs, np.ndarray):
        return -np.exp(-np.sum(np.square((obs-target)), axis=-1)/k)
    else:
        return -tf.exp(-tf.reduce_sum(tf.square((obs-target)), axis=-1)/k)

def ac_cost(x): return 0

class Machine:
    def __init__(self, shared_cfg, machine_cfg):
        self.s_cfg = shared_cfg
        self.m_cfg = machine_cfg

        # Simulated environment
        self.env = MachineModelEnv('sim_model/', 'env', ac_cost, obs_cost_fn, stochastic=True, randInit=True)
        self.n_parts = machine_cfg.aconity.n_parts
        self.horizon = machine_cfg.aconity.layers[1]-machine_cfg.aconity.layers[0]+1
        self.t = 0
        self.states = np.zeros((self.horizon+1, self.n_parts, 16))
        self.actions = np.zeros((self.horizon, self.n_parts, 2))

        self.logdir = 'sim_model/'

    def nextStep(self, action):
        """ action (n_parts, 2) """
        print("Applying", action)
        self.actions[self.t, :, :] = action

        # Initialise if appropiate
        if self.t == 0:
            for part in range(self.n_parts):
                self.states[0, part, :] = self.env.reset()

        # Transition x(k) -> x(k+1)
        for part in range(self.n_parts):
            self.states[self.t+1, part, :] = self.env.model_transition_state(
                self.states[self.t, part, :], action[part, :])

        self.t += 1
        print("Mean temps", self.states[self.t, :, :].mean(-1))
        return self.states[self.t, :, :]

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
        # Init with 1.125, 110
        print("Initial action is 1.125, 110")
        return np.ones((self.s_cfg.env.n_parts, 2)) * [1.125, 110]

    def loop(self):
        self.sendAction(self.initAction())
        while self.t < self.H:
            states = self.getStates()
            actions = self.computeAction(states)
            self.sendAction(actions)
            self.log()

    def log(self):
        np.save("tttpret_state_traj.npy", self.pret_state_traj)
        np.save("tttpret_action_traj.npy", self.pret_action_traj)
        np.save("tttpret_pred_cost.npy", self.pred_cost_pret)
        np.save("tttlear_state_traj.npy", self.lear_state_traj)
        np.save("tttlear_action_traj.npy", self.lear_action_traj)
        np.save("tttlear_pred_cost.npy", self.pred_cost_lear)

if __name__ == '__main__':
    from config_windows import returnSharedCfg
    from config_cluster import returnClusterPretrainedCfg, returnClusterUnfamiliarCfg
    from config_windows import returnSharedCfg, returnMachineCfg

    s_cfg = returnSharedCfg()
    m_cfg = returnMachineCfg()
    cp_cfg = returnClusterPretrainedCfg() # Pretrained
    cl_cfg = returnClusterUnfamiliarCfg() # Learned

    machine = Machine(s_cfg, m_cfg)
    cluster = Cluster(s_cfg, cp_cfg, cl_cfg)

    action = cluster.initAction()
    t = 0
    T = 160
    while t < T:
        print("t %d T %d" % (t+1, T))
        states = machine.nextStep(action)
        action = cluster.computeAction(states)
        cluster.log()
        t+=1
