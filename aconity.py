"""
Machine side software
"""

import os
import numpy as np
from dotmap import DotMap
import pysftp

import asyncio
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from AconitySTUDIO_client import AconitySTUDIOPythonClient
from AconitySTUDIO_client import utils

from process.mainFile import *

def pieceNumber(self, piece_indx):
    """ 0->4, 1->7, 2->10, etc... """
    return int((piece_indx+1)*3+1)

def getLoginData():
    login_data = { 'rest_url' : f'http://localhost:9000',
                    'ws_url' : f'ws://localhost:9000',
                    'email' : 'admin@aconity3d.com', 'password' : 'passwd'}

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

def makeDirIfDoesNotExist(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

class Aconity:
    def __init__(self, shared_cfg, machine_cfg):
        self.s_cfg = shared_cfg
        self.m_cfg = machine_cfg

        self._initAconity()

        # Aconity variables
        self.job_started = False
        self.processing_uninitialised = True
        self.curr_layer = machine_cfg.aconity.layers[0]
        self.rectangle_limits_computed = False

    # --------------------------------------------------------------------------
    # COMMS FUNCTIONS
    # --------------------------------------------------------------------------
    async def getActions(self):
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.action
        rdy = dir + cfg.rdy_name

        # Wait until RDY signal is provided
        while(not os.path.isdir(rdy)): await asyncio.sleep(0.05)
        os.rmdir(rdy) # Delete RDY

        # Read data to array
        actions = np.load(dir+cfg.f_name)
        print('Control signal received')
        return actions

    def sendStates(self, states):
        """Send state information to cluster side"""
        dir = self.s_cfg.comms.dir
        cfg = self.s_cfg.comms.state
        os.mkdir(dir+cfg.rdy_name) # RDY signal

    # --------------------------------------------------------------------------
    # ACONITY FUNCTIONS
    # --------------------------------------------------------------------------

    async def _initAconity(self):
        cfg = self.machine_cfg.aconity
        self.client = await AconitySTUDIOPythonClient.create(getLoginData())
        await self.client.get_job_id(cfg.info.job_name)
        await self.client.get_machine_id('AconityMini')
        await self.client.get_config_id(cfg.info.config_name)
        await self.client.get_session_id(n=-1)

    async def _changeMarkSpeed(self, part, value):
        await self.client.change_part_parameter(pieceNumber(part), 'mark_speed', value)

    async def _changeLaserPower(self, part, value):
        await self.client.change_part_parameter(pieceNumber(part), 'laser_power', value)

    async def _pauseUponLayerCompletion(self, sleep_time=0.05):
        """ sleep time in seconds """
        original = self.client.job_info['AddLayerCommands']
        while(original==self.client.job_info['AddLayerCommands']):
            await asyncio.sleep(sleep_time)
        await client.pause_job()

    async def performLayer(self, actions):
        """Start building next layer with the specified parameters"""
        cfg = self.machine_cfg.aconity
        assert cfg.n_parts == actions.shape[0], \
            "Mismatch %d != %d" % (cfg.n_parts, actions.shape[0])

        # Change parameters
        for part in range(cfg.n_parts):
            await self._changeMarkSpeed(part, actions[part, 0])
            #await self._changeLaserPower(part, actions[part, 1])

        # Resume / start job
        if self.job_started:
            await self.client.resume_job(layers=cfg.layers)
        else:
            execution_script = getUnheatedMonitoringExecutionScript()
            await self.client.start_job(cfg.layers, execution_script)
            self.job_started = True
            self.sendStates()

        await self._pauseUponLayerCompletion()
        self.curr_layer += 1

    async def loop(self):
        max_layer = self.machine_cfg.aconity.layers[1]
        while self.curr_layer <= max_layer:
            print("Layer %d/%d" % (self.curr_layer, max_layer))
            actions = await self.getActions()
            await self.performLayer(actions)