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

        self._initSFTP()

        self.processing_uninitialised = True
        self.rectangle_limits_computed = False
        self.curr_layer = machine_cfg.aconity.layers[0]
        self.square_limits = []

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

        # Copy file and signal acconity
        print('Copying file...')
        self.sftp.get(dir+cfg.f_name, localpath=dir+cfg.f_name)
        os.mkdir(rdy) # RDY signal
        print('Actions saved')

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

    # --------------------------------------------------------------------------
    # PROCESS FUNCTIONS
    # --------------------------------------------------------------------------

    def initProcessing(self):
        cfg = self.m_cfg.aconity.process
        def getLatestConfigJobFolder(folder_name, init_str):
            latest_n, name = -1, None
            for filename in os.listdir(folder_name):
                match = re.search(r''+init_str+'_(\d+)_(\w+)', filename)
                if match and match.group(2) is not 'none' and int(match.group(1)) > latest_n:
                        latest_n, name = int(match.group(1)), filename
            if name is None: raise ValueError('No suitable '+init_str+' folders found')
            return name
        config_folder = getLatestConfigJobFolder(cfg.sess_dir, 'config')
        job_folder = getLatestConfigJobFolder(cfg.sess_dir+config_folder+'/', 'job')
        self.data_folder = cfg.sess_dir+config_folder+'/'+job_folder+'/sensors/2Pyrometer/pyrometer2/'
        self.processing_uninitialised = False
        print("Data folder found is " + self.data_folder)

    def getFileName(self, layer, piece):
        return self.data_folder+str(pieceNumber(piece))+'/'+str(np.round(layer*0.03, 2))+'.pcd'

    def getStates(self):
        """Read raw data from the pyrometer and processes it into states"""
        if processing_uninitialised:
            rdy = self.s_cfg.comms.dir + self.s_cfg.comms.state.rdy_name
            while(not os.path.isdir(rdy)): pass
            os.rmdir(rdy)
            self.initProcessing()

        cfg = self.m_cfg.aconity
        nS = self.s_cfg.env.nS
        n_div = int(sqrt(nS))
        layer = self.curr_layer
        states = np.zeros((cfg.n_parts, nS))
        for part in range(cfg.n_parts):
            filename = self.getFileName(layer, part)
            print("Expected file "+filename)
            while(not os.path.isfile(filename)): pass
            time.sleep(cfg.process.sleep_t) # Prevent reading too soon

            # Read and process data
            data = loadData(filename)
            if not self.rectangle_limits_computed:
                self.square_limits.append(divide4squares(data))
                print("Square limits found", self.square_limits[-1])

            data, cutoff = removeColdLines(data, returnMode=2)
            print("Cut-off value is %d" % (cutoff))
            data_divisions, ratio, error = divideDataRectangleLimits(data,
                    square_limits[part], returnMode=4)
            print("Delete ratio %.4f, error %d" % (ratio, error))
            for sample in range(4):
                states[part, :] = pieceStateMeans(data_divisions[sample],
                                                  square_limits[part, sample],
                                                  n_div)
        self.curr_layer += 1
        return states
