"""
Machine side software
"""

import os
import numpy as np
from dotmap import DotMap
import pysftp

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
        self.horizon = machine_cfg.aconity.layers[1]-machine_cfg.aconity.layers[0]
        self.t = 0
        self.states = np.zeros((self.horizon+1, self.n_parts, 16))
        self.actions = np.zeros((self.horizon, self.n_parts, 2))

        self.logdir = 'sim_model/'

        # Comms setup
        self._initSFTP()

    # --------------------------------------------------------------------------
    # COMMS FUNCTIONS
    # --------------------------------------------------------------------------

    def _initSFTP(self):
        """Initialise SFTP connection"""
        print("Initialising SFTP...")
        cfg = self.m_cfg.comms.sftp
        comms = self.s_cfg.comms

        # Set up connection
        self.sftp = pysftp.Connection(host=cfg.host, username=cfg.user,
            password=cfg.pwd)

        # Create comms dir and cd to it
        if not os.path.isdir(comms.dir): os.mkdir(comms.dir)
        if not self.sftp.isdir(comms.dir): self.sftp.mkdir(comms.dir)

        print("SFTP initialised")

    def getActions(self):
        """Read action file outputted by cluster"""
        print('Waiting for actions...')
        dir_c = self.m_cfg.comms.cluster_dir + self.s_cfg.comms.dir
        dir_m = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.action
        rdy_c = dir_c + cfg.rdy_name
        rdy_m = dir_m + cfg.rdy_name

        # Wait until RDY signal is provided
        while(not self.sftp.isdir(rdy_c)): pass
        self.sftp.rmdir(rdy_c) # Delete RDY

        # Copy file and signal acconity
        print('Copying file...')
        self.sftp.get(dir_c+cfg.f_name, localpath=dir_m+cfg.f_name)

        # Read data to array
        actions = np.load(dir_m+cfg.f_name)
        print('Control signal received')
        return actions

    def sendStates(self, states):
        """Send state information to cluster side"""
        dir_c = self.m_cfg.comms.cluster_dir + self.s_cfg.comms.dir
        dir_m = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.state

        # Write states into npy file
        print('Saving states...')
        np.save(dir_m+cfg.f_name, states)

        # Upload to server
        print('Uploading states...')
        self.sftp.put(dir_m+cfg.f_name, remotepath=dir_c+cfg.f_name)
        self.sftp.mkdir(dir_c+cfg.rdy_name) # RDY signal
        print('States sent')

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

    def log(self):
        np.save(self.logdir+'states.npy', self.states)
        np.save(self.logdir+'actions.npy', self.actions)

    def loop(self):
        while self.t < self.horizon:
            print("Timestep %d/%d" % (self.t+1, self.horizon))
            action = self.getActions()
            states = self.nextStep(action)
            self.sendStates(states)
            self.log()


if __name__ == '__main__':
    from config_windows import returnSharedCfg, returnMachineCfg
    s_cfg = returnSharedCfg()
    m_cfg = returnMachineCfg()
    machine = Machine(s_cfg, m_cfg)
    machine.loop()