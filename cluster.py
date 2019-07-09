import os
import numpy as np
from dmbrl.controllers.MPC import MPC

class Cluster:
    """
    Computes optimal process parameters, at each layer, given feedback obtained
    from the machine sensors.

    Arguments:
        shared_cfg (dotmap):
            - **env.n_parts** (*int*): Total number of parts built under feedback control.
            - **env.horizon** (*int*): Markov Decision Process horizon (here number of layers).
            - **env.nS** (*int*): Dimension of the state vector.
            - **comms** (*dotmap*): Parameters for communication with other classes.
        pretrained_cfg (dotmap):
            - **n_parts** (*dotmap*): Number of parts built under this control scheme.
            - **ctrl_cfg** (*dotmap*): Configuration parameters passed to the MPC class.
        learned_cfg (dotmap):
            - **n_parts** (*dotmap*): Number of parts built under this control scheme.
            - **ctrl_cfg** (*dotmap*): Configuration parameters passed to the MPC class.
    """
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
        assert self.n_parts_pretrained+self.n_parts_learned == self.n_parts, "Number of parts does not match"

        self.pret_state_traj = np.zeros((self.n_parts_pretrained, self.H, shared_cfg.env.nS))
        self.pret_action_traj = np.zeros((self.n_parts_pretrained, self.H, 2))
        self.lear_state_traj = np.zeros((self.n_parts_learned, self.H, shared_cfg.env.nS))
        self.lear_action_traj = np.zeros((self.n_parts_learned, self.H, 2))
        self.pred_cost_pret = np.zeros((self.H, self.n_parts_pretrained))
        self.pred_cost_lear = np.zeros((self.H, self.n_parts_learned))

        self.save_dirs = [shared_cfg.save_dir1, shared_cfg.save_dir2]

    # --------------------------------------------------------------------------
    # COMMS FUNCTIONS
    # --------------------------------------------------------------------------

    def getStates(self):
        """Load state vectors uploaded to the server by the `Machine` class.

        This function waits for the `comms.dir/comms.state.rdy_name` folder to be
        created by the `Machine` class, before reading the file where the states
        are located, `comms.dir/comms.state.f_name`

        Returns:
            np.array: State vector with shape (`n_parts`, `nS`)
        """
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
        """Saves the computed actions.

        Signals the `Machine` class that actions are ready to be downloaded by
        locally creating the `comms.dir/comms.action.rdy_name` folder

        Arguments:
            actions (np.array): Action vector with shape (`n_parts`, `nU`)
        """
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
        """Computes the control actions given the observed system states.

        Arguments:
            states (np.array): Observed states, shape (`n_parts`, `nS`)

        Returns:
            np.array: Computed actions, with shape (`n_parts`, `nU`)
        """
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
        """ Returns the initial action vector.

        This function is required because an initial layer must be built before
        any feedback is available.

        Returns:
            np.array: Initial action vector with shape (`n_parts`, `nU`)
        """
        print("Initial action is 1.125, 110")
        return np.ones((self.s_cfg.env.n_parts, 2)) * [1.125, 110]

    def log(self):
        """ Logs the state and action trajectories, as well as the predicted cost,
        which may be of interest to tune some algorithmic parameters.
        """
        for i in range(len(self.save_dirs)):
            np.save(self.save_dirs[i]+"pret_state_traj.npy", self.pret_state_traj)
            np.save(self.save_dirs[i]+"pret_action_traj.npy", self.pret_action_traj)
            np.save(self.save_dirs[i]+"pret_pred_cost.npy", self.pred_cost_pret)
            np.save(self.save_dirs[i]+"lear_state_traj.npy", self.lear_state_traj)
            np.save(self.save_dirs[i]+"lear_action_traj.npy", self.lear_action_traj)
            np.save(self.save_dirs[i]+"lear_pred_cost.npy", self.pred_cost_lear)

    def loop(self):
        """ While within the time horizon, read the states provided by the `Machine`
        class, and compute and save the corresponding actions.

        Allows the class functionality to be conveniently used as follows::

            cluster = Cluster(s_cfg, cp_cfg, cl_cfg)
            cluster.loop()
        """
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
