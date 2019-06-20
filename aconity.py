"""
Machine side software
"""

import os
import numpy as np
from dotmap import DotMap
import pysftp
import re

import asyncio
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from AconitySTUDIO_client import AconitySTUDIOPythonClient
from AconitySTUDIO_client import utils

from process.process_main import *

def getLoginData():
    login_data = { 'rest_url' : f'http://localhost:9000',
                    'ws_url' : f'ws://localhost:9000',
                    'email' : 'admin@aconity3d.com', 'password' : 'passwd'}
    return login_data

def getUnheatedMonitoringExecutionScript():
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
    def __init__(self, shared_cfg, machine_cfg):
        self.s_cfg = shared_cfg
        self.m_cfg = machine_cfg

        # Aconity variables
        self.job_started = False
        self.curr_layer = machine_cfg.aconity.layers[0]
        self.job_paused = False

        # Parts distributed as...
        #   - [1, parts_ignored] - Parts built in case pyrometer does not record
        #   - [parts_ignored + 1, control_buffer] - Closed loop parts
        #   - [control_buffer + 1, fixed_buffer] - Parts with fixed parameters
        self.parts_ignored = shared_cfg.parts_ignored
        self.control_buffer = self.parts_ignored
        self.fixed_buffer = self.control_buffer + shared_cfg.env.n_parts

    # --------------------------------------------------------------------------
    # COMMS FUNCTIONS
    # --------------------------------------------------------------------------
    async def getActions(self):
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

    def sendStates(self):
        """Send state information to cluster side"""
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.state
        os.mkdir(dir+cfg.rdy_name) # RDY signal

    # --------------------------------------------------------------------------
    # ACONITY FUNCTIONS
    # --------------------------------------------------------------------------
    def pieceNumber(self, piece_indx, buffer):
        """ 0->4, 1->7, 2->10, etc... """
        """ 0->2, 1->3, 2->4, etc..."""
        return int((piece_indx+buffer)*self.m_cfg.aconity.part_delta+1)

    async def initAconity(self):
        cfg = self.m_cfg.aconity
        self.client = await AconitySTUDIOPythonClient.create(getLoginData())
        await self.client.get_job_id(cfg.info.job_name)
        await self.client.get_machine_id('AconityMini')
        await self.client.get_config_id(cfg.info.config_name)
        await self.client.get_session_id(n=-1)
        print("Done init")

    async def _changeMarkSpeed(self, part, value, buffer):
        print("Writing to %d " % (self.pieceNumber(part, buffer)))
        await self.client.change_part_parameter(self.pieceNumber(part, buffer), 'mark_speed', value)

    async def _changeLaserPower(self, part, value, buffer):
        if self.m_cfg.aconity.laser_on:
            await self.client.change_part_parameter(self.pieceNumber(part, buffer), 'laser_power', value)

    async def initialParameterSettings(self):
        # Slowly scan the ones ignored
        for i in range(self.parts_ignored):
            await self._changeMarkSpeed(i, self.m_cfg.aconity.ignored_parts_speed, 0)

        # Parameters that will remain unchanged
        fixed_params = self.m_cfg.aconity.fixed_params
        for i in range(fixed_params.shape[0]):
            await self._changeMarkSpeed(i, fixed_params[i,0]*1000, self.fixed_buffer)
            await self._changeLaserPower(i, fixed_params[i,1], self.fixed_buffer)

    async def _pauseUponLayerCompletion(self, sleep_time=0.05):
        """ sleep time in seconds """
        print("Awaiting for layer completion...")
        original = self.client.job_info['AddLayerCommands']
        while(original==self.client.job_info['AddLayerCommands']):
            await asyncio.sleep(sleep_time)
        print("Job being paused...")
        await self.client.pause_job()
        self.job_paused = True

    async def performLayer(self, actions):
        """Start building next layer with the specified parameters"""
        print("Actions chosen", actions)
        layers = self.m_cfg.aconity.layers
        n_parts = self.s_cfg.env.n_parts
        assert n_parts == actions.shape[0], \
            "Mismatch %d != %d" % (n_parts, actions.shape[0])

        # Change parameters
        for part in range(n_parts):
            print("Part", part, "Setting parameters...", actions[part, :])
            await self._changeMarkSpeed(part, actions[part, 0]*1000, self.control_buffer)
            await self._changeLaserPower(part, actions[part, 1], self.control_buffer)

        # Resume / start job
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
            self.sendStates()

        # Log actions taken
        np.save("saves/param_l_%d.npy" % (self.curr_layer), actions)

        await self._pauseUponLayerCompletion()
        self.curr_layer += 1

    async def loop(self):
        max_layer = self.m_cfg.aconity.layers[1]
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
