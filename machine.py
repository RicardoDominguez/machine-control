"""
Machine side software
"""

import os
import numpy as np
from dotmap import DotMap
import pysftp


class Machine:
    def __init__(self, shared_cfg, machine_cfg):
        self.s_cfg = shared_cfg
        self.m_cfg = machine_cfg

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
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.action
        rdy = dir + cfg.rdy_name

        # Wait until RDY signal is provided
        while(not self.sftp.isdir(rdy)): pass
        self.sftp.rmdir(rdy) # Delete RDY

        # Copy file
        print('Copying file...')
        self.sftp.get(dir+cfg.f_name, localpath=dir+cfg.f_name)

        # Read data to array
        actions = np.load(dir+cfg.f_name)
        print('Control signal received')
        return actions

    def sendStates(self, states):
        """Send state information to cluster side"""
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.state

        # Write states into npy file
        print('Saving states...')
        np.save(dir+cfg.f_name, states)

        # Upload to server
        print('Uploading states...')
        self.sftp.put(dir+cfg.f_name, remotepath=dir+cfg.f_name)
        self.sftp.mkdir(dir+cfg.rdy_name) # RDY signal
        print('States sent')

    def performLayer(self):
        """Start building next layer with the specified parameters"""
        pass

    def getStates(self):
        """Read raw data from the pyrometer and processes it into states"""
        pass
