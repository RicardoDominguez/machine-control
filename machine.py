"""
Machine side software
"""

import os
import numpy as np
from dotmap import DotMap
import pysftp
import re

from process.process_main import *

def pieceNumber(piece_indx, n_ignore):
    """ 0->4, 1->7, 2->10, etc... """
    return int((piece_indx+n_ignore)+1)

class Machine:
    def __init__(self, shared_cfg, machine_cfg):
        self.s_cfg = shared_cfg
        self.m_cfg = machine_cfg

        self._initSFTP()

        # Aconity variables
        self.processing_uninitialised = True
        self.curr_layer = machine_cfg.aconity.layers[0]
        self.rectangle_limits_computed = np.zeros((self.m_cfg.aconity.n_parts,), dtype=bool)
        self.square_limits = []

        self.state_log = None

        self.n_ignore = shared_cfg.n_ignore_buffer + shared_cfg.n_rand + machine_cfg.aconity.open_loop.shape[0]

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
        os.mkdir(rdy_m) # RDY signal
        print('Actions saved')

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

    # --------------------------------------------------------------------------
    # PROCESS FUNCTIONS
    # --------------------------------------------------------------------------
    def initProcessing(self):
        cfg = self.m_cfg.aconity.process
        def getLatestSession(folder_name):
            best, name = None, None
            for filename in os.listdir(folder_name):
                match = re.search(r'session_(\d+)_(\d+)_(\d+)_(\d+)-(\d+)-(\d+).(\d+)', filename)
                if match:
                    date = [int(match.group(i+1)) for i in range(7)]
                    for i in range(7):
                        if best is None or date[i] > best[i]:
                            best, name = date, filename
                            break
            return name
        def getLatestConfigJobFolder(folder_name, init_str):
            latest_n, name = -1, None
            for filename in os.listdir(folder_name):
                match = re.search(r''+init_str+'_(\d+)_(\w+)', filename)
                if match and match.group(2) is not 'none' and int(match.group(1)) > latest_n:
                        latest_n, name = int(match.group(1)), filename
            if name is None: raise ValueError('No suitable '+init_str+' folders found')
            return name
        sess_folder = cfg.sess_dir+getLatestSession(cfg.sess_dir)+'/'
        config_folder = sess_folder+getLatestConfigJobFolder(sess_folder, 'config')+'/'
        job_folder = config_folder+getLatestConfigJobFolder(config_folder, 'job')+'/'
        self.data_folder = job_folder+'sensors/2Pyrometer/pyrometer2/'
        self.processing_uninitialised = False
        print("Data folder found is " + self.data_folder)

    def getFileName(self, layer, piece):
        return self.data_folder+str(pieceNumber(piece, self.n_ignore))+'/'+str(np.round(layer*0.03, 2))+'.pcd'

    def getStates(self):
        """Read raw data from the pyrometer and processes it into states"""
        if self.processing_uninitialised:
            rdy = self.s_cfg.comms.dir + self.s_cfg.comms.state.rdy_name
            while(not os.path.isdir(rdy)): pass
            time.sleep(5) # TO MAKE SURE I READ THE CORRECT JOB
            os.rmdir(rdy)
            self.initProcessing()

        cfg = self.m_cfg.aconity
        nS = self.s_cfg.env.nS
        n_div = int(np.sqrt(nS))
        layer = self.curr_layer
        states = np.zeros((cfg.n_parts, nS))
        for part in range(cfg.n_parts):
            filename = self.getFileName(layer, part)
            print("Expected file "+filename)
            while(not os.path.isfile(filename)): time.sleep(0.05)
            time.sleep(cfg.process.sleep_t) # Prevent reading too soon

            # Read and process data
            data = loadData(filename)
            if not self.rectangle_limits_computed[part]:
                self.square_limits.append(divideSingleSquare(data))
                print("Square limits found", self.square_limits[-1])
                self.rectangle_limits_computed[part] = True
            data, cutoff = removeColdLines(data, returnMode=2)
            print("Cut-off value is %d" % (cutoff))
            data, ratio, error = divideDataRectangleLimits(data,
                    self.square_limits[part], returnMode=4, plot=True, saveName=filename)
            print("Delete ratio %.4f, error %d" % (ratio, error))
            states[part, :] = pieceStateMeans(data, self.square_limits[part], n_div)
            plotTemperaturesState(states[part, :], saveFig=filename+'_temp.png')
        self.curr_layer += 1
        return states

    def sendAction(self, action):
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.action
        rdy = dir + cfg.rdy_name
        np.save(dir+cfg.f_name, action)
        os.mkdir(rdy) # RDY signal
        print('Actions saved')

    def test_loop(self):
        initialised=False
        while(True):
            states = self.getStates()
            if not initialised:
                state_log = np.empty((0, states.shape[0], states.shape[1]))
                initialised = True
            state_log = np.concatenate((state_log, states[None]), axis=0)
            print("Saving states...")
            np.save("states.npy", state_log)

    def log(self, states):
        if self.state_log is None:
            state_log = np.empty((0, states.shape[0], states.shape[1]))
        self.state_log = np.concatenate((self.state_log, states[None]), axis=0)
        np.save("saves/machinestate_log.npy", self.state_log)


    def loop(self):
        while(True):
            self.getActions()
            states = self.getStates()
            self.sendStates(states)
            self.log(states)
