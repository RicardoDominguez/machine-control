import os
import re
import numpy as np
from dotmap import DotMap
import pysftp

# Aconity3D API
import asyncio
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from AconitySTUDIO_client import AconitySTUDIOPythonClient
from AconitySTUDIO_client import utils

from process.process_main import *

def getLoginData():
    """ Returns the login credentials for the Aconity machine.

    Returns:
        dict: Dictionary with keys `rest_url`, `ws_url`, `email`, and `password`
    """
    login_data = { 'rest_url' : f'http://localhost:9000',
                    'ws_url' : f'ws://localhost:9000',
                    'email' : 'admin@aconity3d.com', 'password' : 'passwd'}
    return login_data

def getExecutionScript():
    """Returns the Aconity execution script, as generated by the AconitySTUDIO web application.

    For instance, the execution script for unheated monitoring is::

        layer = function(){
        for(p:$p){
        $m.record_sensor($s[2Pyrometer],p)
        $m.expose(p[next;$h],$c[scanner_1])
        $m.stop_record_sensor($s[2Pyrometer])
        }
        $m.add_layer($g)
        }
        repeat(layer)

    Returns:
        str: Execution script as generated by the AconitySTUDIO web application.
    """
    execution_script = \
    '''layer = function(){
    for(p:$p){
    $m.record_sensor($s[2Pyrometer],p)
    $m.expose(p[next;$h],$c[scanner_1])
    $m.stop_record_sensor($s[2Pyrometer])
    }
    $m.add_layer($g)
    }
    repeat(layer)'''
    return execution_script

class Aconity:
    """Makes use of the API provided by Aconity to automatically start, pause and
    resume a build, and to change individual part parameters in real-time.

    Arguments:
        shared_cfg (dotmap):
            - **env.n_parts** (*int*): Number of parts to be built, not including the first `n_ignore` parts, which are often no recorded by the pyrometer.
            - **n_ignore** (*int*): Number of parts built in addition to `n_parts`. Typically, the first 3 parts are not fully recorded by the pyrometer.
            - **ctrl_cfg.ac_lb** (*array of floats*): Lower bounds of the build parameter in the form [speed (m/s), power (W)].
            - **ctrl_cfg.ac_ub** (*array of floats*): Upper bounds of the build parameters in the form [speed (m/s), power (W)].
            - **comms** (*dotmap*): Parameters for communication with other classes.
        aconity_cfg (dotmap):
            - **config_name** (*str*): Configuration name, i.e. `Unheated 3D Monitoring`.
            - **job_name** (*str*): Job name as displayed in the AconitySTUDIO web application.
            - **layers** (*array of int*): Start and end layers for the build in the form [start_layer, end_layer].
            - **fixed_parameters** (*np.array, shape N x 2*): Parameters used for parts built with fixed parameters.
    """
    def __init__(self, shared_cfg, aconity_cfg):
        self.s_cfg = shared_cfg
        self.a_cfg = aconity_cfg

        # Aconity variables
        self.job_started = False
        self.curr_layer = aconity_cfg.layers[0]
        self.job_paused = False

        # Parts distributed as...
        #   - [1, parts_ignored] - Parts built in case pyrometer does not record
        #   - [parts_ignored + 1, control_buffer] - Closed loop parts
        #   - [control_buffer + 1, fixed_buffer] - Parts with fixed parameters
        self.parts_ignored = shared_cfg.parts_ignored
        self.control_buffer = self.parts_ignored
        self.fixed_buffer = self.control_buffer + shared_cfg.env.n_parts

        self.uploadConfigFiles()

    # --------------------------------------------------------------------------
    # COMMS FUNCTIONS
    # --------------------------------------------------------------------------
    async def getActions(self):
        """Gets the parameters generated using closed-loop control, copied locally
        from the remote server by the `Machine` class.

        Returns:
            np.array: Parameters generated using closed-loop control, with shape (`parts under closed-loop control`, 2)
        """
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.action
        rdy = dir + cfg.rdy_name

        # Wait until RDY signal is provided
        print("Waiting for actions...")
        no_error = 1
        while(no_error):
            while(not os.path.isdir(rdy)): await asyncio.sleep(0.05)
            try:
                os.rmdir(rdy) # Delete RDY
                no_error = 0
            except:
                pass

        # Read data to array
        actions = np.load(dir+cfg.f_name)
        print('Control signal received')
        return actions

    def signalJobStarted(self):
        """Signals the `Machine` class that the job has been stated.

        This is done by creating a folder locally in `comms.dir/comms.state.rdy_name`.

        This prompts the `Machine` class to read and analyse the pyrometer data.
        """
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.state
        os.mkdir(dir+cfg.rdy_name) # RDY signal

    def uploadConfigFiles(self):
        comms = self.m_cfg.comms
        cfg = self.m_cfg.comms.sftp

        # Set up connection
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        sftp = pysftp.Connection(host=cfg.host, username=cfg.user,
            password=cfg.pwd, cnopts=cnopts)

        # Transfer config files
        print("Uploading configuration files to %s..." % (comms.cluster_dir))
        config_files = ['config_cluster.py', 'config_dmbrl.py', 'config_windows.py']
        for file in config_files: sftp.put(file, comms.cluster_dir+file)

        # Clear local comms
        cfg = self.s_cfg.comms
        dir_action = cfg.dir+cfg.action.rdy_name
        dir_state = cfg.dir+cfg.state.rdy_name
        if os.path.isdir(dir_action): os.rmdir(dir_action)
        if os.path.isdir(dir_state): os.rmdir(dir_state)

        sftp.close()


    # --------------------------------------------------------------------------
    # ACONITY FUNCTIONS
    # --------------------------------------------------------------------------
    def pieceNumber(self, piece_indx, buffer):
        """ 0->4, 1->7, 2->10, etc... """
        """ 0->2, 1->3, 2->4, etc..."""
        return int((piece_indx+buffer)*self.m_cfg.aconity.part_delta+1)

    async def initAconity(self):
        """Creates a new connection to the AconityMINI using the job name provided.

        The latest session created by the AconityStudio web application is used.
        """
        self.client = await AconitySTUDIOPythonClient.create(getLoginData())
        await self.client.get_job_id(self.a_cfg.job_name)
        await self.client.get_machine_id('AconityMini')
        await self.client.get_config_id(self.a_cfg.config_name)
        await self.client.get_session_id(n=-1) # Latest
        print("The Aconity machine has been initialised")

    async def initialParameterSettings(self):
        """Sets the build parameters for those parts which use fixed parameters throughout the build.

        This parameters are defined in the configuration files rather than being
        passed an input to this function.
        """
        # Slowly scan the ones ignored
        for i in range(self.parts_ignored):
            await self._changeMarkSpeed(i, self.m_cfg.aconity.ignored_parts_speed*1000, 0)
            await self._changeLaserPower(i, self.m_cfg.aconity.ignored_parts_power, 0)

        # Parameters that will remain unchanged
        fixed_params = self.m_cfg.aconity.fixed_params
        for i in range(fixed_params.shape[0]):
            await self._changeMarkSpeed(i, fixed_params[i,0]*1000, self.fixed_buffer)
            await self._changeLaserPower(i, fixed_params[i,1], self.fixed_buffer)

    async def _changeMarkSpeed(self, part, value, buffer):
        """Changes the mark speed of an individual part.

        Arguments:
            part (int): Index of part (from 0 to n_parts).
            value (float): Value to be assigned as the mark speed.
            buffer (int): Global part number is buffer + part.
        """
        print("Writing to %d " % (self.pieceNumber(part, buffer)))
        await self.client.change_part_parameter(self.pieceNumber(part, buffer), 'mark_speed', value)

    async def _changeLaserPower(self, part, value, buffer):
        """Changes the laser power of an individual part.

        Arguments:
            part (int): Index of part (from 0 to n_parts).
            value (float): Value to be assigned as the laser power.
            buffer (int): Global part number is buffer + part.
        """
        if self.m_cfg.aconity.laser_on:
            await self.client.change_part_parameter(self.pieceNumber(part, buffer), 'laser_power', value)
        else:
            await self.client.change_part_parameter(self.pieceNumber(part, buffer), 'laser_power', 0)

    async def _pauseUponLayerCompletion(self, sleep_time=0.05):
        """Awaits the current layer to be complete and then pauses the build.

        Layer completion is checked by pooling `client.job_info['AddLayerCommands']`,
        which increases by one upon the completion of a layer.

        Arguments:
            sleep_time (float, optional): Period with which the status of the build is checked, in seconds.
        """
        print("Awaiting for layer completion...")
        original = self.client.job_info['AddLayerCommands']
        while(original==self.client.job_info['AddLayerCommands']):
            await asyncio.sleep(sleep_time)
        print("Job being paused...")
        await self.client.pause_job()
        self.job_paused = True

    async def performLayer(self, actions):
        """Builds a single layer using the specified input parameters.

        Before the layer is started, the relevant parameters are changed according
        to the input array `actions`. Upon the completion of the layer, the build is paused.

        Arguments:
            actions (np.array): Input parameters to be used for the new layer, with shape (`n_parts`, 2)
        """
        print("Actions chosen", actions)
        layers = self.m_cfg.aconity.layers
        n_parts = self.s_cfg.env.n_parts
        assert n_parts == actions.shape[0], \
            "Mismatch %d != %d" % (n_parts, actions.shape[0])

        # Random parameters
        rand_param = np.random.rand(self.n_rand, 2)*(self.ac_ub-self.ac_lb)+self.ac_lb
        for i in range(self.n_rand):
            print("Rand", i, "Piece", pieceNumber(i, self.rnd_buffer), "Setting parameters...", rand_param[i, :])
            await self.client.change_part_parameter(pieceNumber(i, self.rnd_buffer), 'mark_speed', rand_param[i,0]*1000)
            await self.client.change_part_parameter(pieceNumber(i, self.rnd_buffer), 'laser_power', rand_param[i,1])

        # Change parameters
        for part in range(n_parts):
            print("Part", part, "Setting parameters...", actions[part, :])
            await self._changeMarkSpeed(part, actions[part, 0]*1000, self.control_buffer)
            await self._changeLaserPower(part, actions[part, 1], self.control_buffer)

        # Resume / start job as appropiate
        if self.job_started:
            print("Wait for job to be paused")
            while(not self.job_paused): pass
            await self.client.resume_job(layers=layers)
            self.job_paused = False
        else:
            execution_script = getUnheatedMonitoringExecutionScript()
            await self.client.start_job(layers, execution_script)
            await self.initialParameterSettings()
            print("Job started...")
            self.job_started = True
            self.signalJobStarted()

        # Log actions taken
        np.save("saves/rand_param_l_%d.npy" % (self.curr_layer), rand_param)
        np.save("saves/param_l_%d.npy" % (self.curr_layer), actions)

        # Log actions taken
        np.save("saves/param_l_%d.npy" % (self.curr_layer), actions)

        await self._pauseUponLayerCompletion()
        self.curr_layer += 1

    async def loop(self):
        """ While the build is unfinished, iteratively builds layers using the
        provided build parameters.

        The `initAconity()` function must always be called before this function.
        Iterates between reading the build parameters outputted in real-time by
        the cluster, and using these build parameters to build individual layers.

        Allows the class functionality to be conveniently used as follows::

            aconity = Aconity(s_cfg, a_cfg)
            aconity.initAconity()
            aconity.loop()
        """
        max_layer = self.a_cfg.aconity.layers[1]
        while self.curr_layer <= max_layer:
            print("Layer %d/%d" % (self.curr_layer, max_layer))
            actions = await self.getActions()
            await self.performLayer(actions)

if __name__ == '__main__':
    import sys
    from config_windows import returnSharedCfg, returnMachineCfg

    async def main():
        s_cfg = returnSharedCfg()
        m_cfg = returnMachineCfg()
        aconity = Aconity(s_cfg, m_cfg)

        utils.log_setup(sys.argv[0], directory_path='')
        await aconity.initAconity()
        await aconity.loop()

    asyncio.run(main(), debug=True)
