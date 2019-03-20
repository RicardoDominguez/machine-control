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

from process.mainFile import *

def pieceNumber(piece_indx):
    """ 0->4, 1->7, 2->10, etc... """
    return int((piece_indx+1)*3+1)

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

    # --------------------------------------------------------------------------
    # COMMS FUNCTIONS
    # --------------------------------------------------------------------------
    async def getActions(self):
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.action
        rdy = dir + cfg.rdy_name

        # Wait until RDY signal is provided
        print("Waiting for actions...")
        while(not os.path.isdir(rdy)): await asyncio.sleep(0.05)
        os.rmdir(rdy) # Delete RDY

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

    async def initAconity(self):
        cfg = self.m_cfg.aconity
        self.client = await AconitySTUDIOPythonClient.create(getLoginData())
        await self.client.get_job_id(cfg.info.job_name)
        await self.client.get_machine_id('AconityMini')
        await self.client.get_config_id(cfg.info.config_name)
        await self.client.get_session_id(n=-1)
        print("Done init")

    async def _changeMarkSpeed(self, part, value):
        await self.client.change_part_parameter(pieceNumber(part), 'mark_speed', value)

    async def _changeLaserPower(self, part, value):
        await self.client.change_part_parameter(pieceNumber(part), 'laser_power', value)

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
        cfg = self.m_cfg.aconity
        assert cfg.n_parts == actions.shape[0], \
            "Mismatch %d != %d" % (cfg.n_parts, actions.shape[0])

        # Change parameters
        for part in range(cfg.n_parts):
            await self._changeMarkSpeed(part, actions[part, 0])
            #await self._changeLaserPower(part, actions[part, 1])

        # Resume / start job
        if self.job_started:
            print("Wait for job to be paused")
            while(not self.job_paused): pass
            await self.client.resume_job(layers=cfg.layers)
            self.job_paused = False
        else:
            execution_script = getUnheatedMonitoringExecutionScript()
            await self.client.start_job(cfg.layers, execution_script)
            self.job_started = True
            self.sendStates()

        await self._pauseUponLayerCompletion()
        self.curr_layer += 1

    async def loop(self):
        max_layer = self.m_cfg.aconity.layers[1]
        while self.curr_layer <= max_layer:
            print("Layer %d/%d" % (self.curr_layer, max_layer))
            actions = await self.getActions()
            await self.performLayer(actions)
