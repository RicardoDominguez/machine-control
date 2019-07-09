import os
import numpy as np
from dotmap import DotMap
import pysftp
import re

from process.process_main import *

<<<<<<< HEAD
def pieceNumber(piece_indx, n_ignore):
    """Returns the index given by AconityStudio to each individual part.

    For instance, if the first part should be ignored, and part numbers increase
    three by three, then `return int((piece_indx+1)*3+1)` should be used, thus
    0 -> 4, 1 -> 7, 2 -> 10, etc.

    On the other hand, if the first three parts should be ignored, and part numbers
    increase one by one, then `return int((piece_indx+3)+1)` should be used, thus
    0 -> 4, 1 -> 5, 2 -> 6, etc.

    Arguments:
        piece_indx (int): Input index, starting from 0.
        n_ignore (int): Number of initial parts that should be ignored.

    Returns:
        int: Output index as used by AconityStudio.
    """
    return int((piece_indx+n_ignore)+1)

=======
>>>>>>> 85c52db3870692cc2b998f39ae6e609c5d8c6190
class Machine:
    """Reads the raw sensory data outputted by the aconity machine, processes it
    into a low-dimensional state vector and uploads it a remote server for
    parameter optimisation.

    Arguments:
        shared_cfg (dotmap):
            - **n_ignore** (*int*): Number of additional parts to be built on top of `env.n_parts` (pyrometer may not record data for the first few parts).
            - **env.nS** (*int*): Dimensionality of the state vector.
            - **comms** (*dotmap*): Configuration parameters for server communication.
        machine_cfg (dotmap):
            - **aconity.layers** (*array of int*): Layer range to be built, as [layer_min, layer_max].
            - **aconity.open_loop** (*np.array*): Parameters used to build the parts built using fixed parameters, np.array with shape (`n_fixed_parts`, 2).
            - **aconity.n_parts** (*int*): Number of parts to be built, excluding ignored parts.
            - **process.sess_dir** (*str*): Folder where pyrometer data is stored by the Aconity machine.
            - **process.sleep_t** (*float*): Time between a sensor data file being first detected and attempting to read it. Prevents errors emerging from opening the file while it is still being written.
    """
    def __init__(self, shared_cfg, machine_cfg):
        self.s_cfg = shared_cfg
        self.m_cfg = machine_cfg

        self._initSFTP()

        # Aconity variables
        self.processing_uninitialised = True
        self.curr_layer = machine_cfg.aconity.layers[0]
        self.rectangle_limits_computed = np.zeros((self.s_cfg.env.n_parts,), dtype=bool)
        self.square_limits = []

        self.state_log = None

<<<<<<< HEAD
        self.n_ignore = shared_cfg.n_ignore + shared_cfg.n_rand + machine_cfg.aconity.open_loop.shape[0]
=======
        self.n_ignore = shared_cfg.parts_ignored
>>>>>>> 85c52db3870692cc2b998f39ae6e609c5d8c6190

    # --------------------------------------------------------------------------
    # COMMS FUNCTIONS
    # --------------------------------------------------------------------------

    def _initSFTP(self):
        """Initialises a SFTP connection with a remote server."""
        print("Initialising SFTP...")
        cfg = self.m_cfg.comms.sftp
        comms = self.s_cfg.comms

        # Set up connection
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        self.sftp = pysftp.Connection(host=cfg.host, username=cfg.user,
            password=cfg.pwd, cnopts=cnopts)

        # Create comms dir and cd to it
        if not os.path.isdir(comms.dir): os.mkdir(comms.dir)
        if not self.sftp.isdir(comms.dir): self.sftp.mkdir(comms.dir)

        print("SFTP initialised")

    def getActions(self):
        """Download locally the action file outputted by the remote server."""
        print('Waiting for actions...')
        dir_c = self.m_cfg.comms.cluster_dir + self.s_cfg.comms.dir
        dir_m = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.action
        rdy_c = dir_c + cfg.rdy_name
        rdy_m = dir_m + cfg.rdy_name

        # Wait until RDY signal is provided
        print('Listening to remote '+rdy_c)
        while(not self.sftp.isdir(rdy_c)): pass
        self.sftp.rmdir(rdy_c) # Delete RDY

        # Copy file and signal acconity
        print('Copying file...')
        self.sftp.get(dir_c+cfg.f_name, localpath=dir_m+cfg.f_name)
        os.mkdir(rdy_m) # RDY signal
        print('Actions saved')

    def sendStates(self, states):
        """Uploads to the the remote server the input state vector.

        Arguments:
            states (np.array): Processed state vector.
        """
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
    def pieceNumber(self, piece_indx, buffer):
        """ 0->4, 1->7, 2->10, etc... """
        """ 0->2, 1->3, 2->4, etc..."""
        return int((piece_indx+buffer)*self.m_cfg.aconity.part_delta+1)

    def initProcessing(self):
        """Obtains the folder in which data will be written by the pyrometer sensor.

        This function automatically detects the latest session and job folders.
        """
        cfg = self.m_cfg.aconity.process
        def getLatestSession(folder_name):
            """Returns the name of the most recent session folder within `folder_name`.

            Arguments:
                folder_name (str): Directory in which to look for session folders.

            Returns:
                str: Name of most recent session folder.
            """
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
            """Returns the name of the latest config/job folder within `folder_name`.

            Arguments:
                folder_name (str): Directory in which to look for job/config folders.
                init_str (str): Folder name must begin with `init_str`.

            Returns:
                str: Name of most recent config/job folder.
            """
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
<<<<<<< HEAD
        """Returns the pyrometer data file path for a given layer and part number.

        This function accounts for the parts being ignored. The layer thickness
        is 0.03 mm.

        Arguments:
            layer (int): Layer number.
            piece (int): Part number.

        Returns:
            str: File path.
        """
        return self.data_folder+str(pieceNumber(piece, self.n_ignore))+'/'+str(np.round(layer*0.03, 2))+'.pcd'
=======
        return self.data_folder+str(self.pieceNumber(piece, self.n_ignore))+'/'+str(np.round(layer*0.03, 2))+'.pcd'
>>>>>>> 85c52db3870692cc2b998f39ae6e609c5d8c6190

    def getStates(self):
        """Read the raw data outputted from the pyrometer and processes it into
        low-dimensional state vectors.

        For every part that must be observed, the raw data is red from the file
        outputted by the Aconity machine, cold lines are removed, the data pertaining
        to the object manufactured is kept, and it is discretised into a number
        of discrete regions in which the mean sensor value is computed.

        Returns:
            np.array: State vectors with shape (`n_parts`,  `nS`)
        """
        if self.processing_uninitialised:
            rdy = self.s_cfg.comms.dir + self.s_cfg.comms.state.rdy_name
            while(not os.path.isdir(rdy)): pass
            time.sleep(5) # TO MAKE SURE I READ THE CORRECT JOB
            os.rmdir(rdy)
            self.initProcessing() # Get folder where to look for

        cfg = self.m_cfg.aconity
        nS = self.s_cfg.env.nS
        n_div = int(np.sqrt(nS))
        layer = self.curr_layer
        n_parts = self.s_cfg.env.n_parts
        states = np.zeros((n_parts, nS))
        n_cumul = 0
        max_cumul = 7
        for part in range(n_parts):
            filename = self.getFileName(layer, part)
            print("Expected file "+filename)
            while(not os.path.isfile(filename)): time.sleep(0.05)
            if n_cumul < max_cumul:
                time.sleep(cfg.process.sleep_t) # Prevent reading too soon
            n_cumul+=1

            # Read and process data

            # Try to prevent wierd exceptions
            no_error = 1
            while(no_error):
                try:
                    data = loadData(filename,timeit=True)
                    no_error = 0
                except:
                    print("Something went wrong reading the data...")
                    pass
                    
            if not self.rectangle_limits_computed[part]:
                self.square_limits.append(divideSingleSquare(data))
                print("Square limits found", self.square_limits[-1])
                self.rectangle_limits_computed[part] = True
            #`data = purgeData(0.25, data)
            data, cutoff = removeColdLines(data, returnMode=2)
            print("Cut-off value is %d" % (cutoff))
            data, ratio, error = divideDataRectangleLimits(data,
                    self.square_limits[part], returnMode=4, plot=True, saveName=filename)
            print("Delete ratio %.4f, error %d" % (ratio, error))
            states[part, :] = pieceStateMeans(data, self.square_limits[part], n_div)
            plotTemperaturesState(states[part, :], saveFig=filename+'_temp.png')
        self.curr_layer += 1
        return states

    def log(self, states):
        """ Locally saves state information.

        Arguments:
            states (np.array): State vectors with shape (`n_parts`, `nS`)
        """
        if self.state_log is None:
            state_log = np.empty((0, states.shape[0], states.shape[1]))
        self.state_log = np.concatenate((self.state_log, states[None]), axis=0)
        np.save("saves/machinestate_log.npy", self.state_log)


    def log(self, states):
        if self.state_log is None:
            self.state_log = np.empty((0, states.shape[0], states.shape[1]))
        self.state_log = np.concatenate((self.state_log, states[None]), axis=0)
        np.save("saves/machinestate_log.npy", self.state_log)


    def loop(self):
        """ Iteratively obtain next layer's parameters from the remote server,
        read and process raw pyrometer data, and upload the low-dimensional
        states to the remote server to compute the next set of optimal parameters.

        Allows the class functionality to be conveniently used as follows::

            machine = Machine(s_cfg, m_cfg)
            machine.loop()
        """
        while(True):
            self.getActions()
            states = self.getStates()
            self.sendStates(states)
            self.log(states)
<<<<<<< HEAD
=======


if __name__ == '__main__':
    from config_windows import returnSharedCfg, returnMachineCfg

    s_cfg = returnSharedCfg()
    m_cfg = returnMachineCfg()
    machine = Machine(s_cfg, m_cfg)
    machine.loop()
>>>>>>> 85c52db3870692cc2b998f39ae6e609c5d8c6190
