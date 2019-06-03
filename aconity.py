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

def pieceNumber(piece_indx, n_ignore):
    """ 0->4, 1->7, 2->10, etc... """
    """ 0->2, 1->3, 2->4, etc..."""
    #return int((piece_indx+n_ignore)*3+1)
    return int((piece_indx+n_ignore)+1)

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

        # self.n_ignore_buffer = shared_cfg.n_ignore_buffer
        # self.n_ignore = self.n_ignore_buffer + shared_cfg.n_rand + machine_cfg.aconity.open_loop.shape[0]
        #
        self.n_rand = shared_cfg.n_rand
        # self.n_ignore_rand = self.n_ignore_buffer + machine_cfg.aconity.open_loop.shape[0]

        self.n_ignore_buffer = shared_cfg.n_ignore_buffer
        self.cl_buffer = shared_cfg.n_ignore_buffer
        self.ol_buffer = self.cl_buffer + shared_cfg.env.n_parts
        self.rnd_buffer = self.ol_buffer + machine_cfg.aconity.open_loop.shape[0]

        #
        # self.n_ignore = self.n_ignore_buffer + shared_cfg.n_rand + machine_cfg.aconity.open_loop.shape[0]
        #
        # self.n_ignore_rand = self.n_ignore_buffer + machine_cfg.aconity.open_loop.shape[0]

        self.ac_lb = shared_cfg.ctrl_cfg.ac_lb
        self.ac_ub = shared_cfg.ctrl_cfg.ac_ub

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

    async def initAconity(self):
        cfg = self.m_cfg.aconity
        self.client = await AconitySTUDIOPythonClient.create(getLoginData())
        await self.client.get_job_id(cfg.info.job_name)
        await self.client.get_machine_id('AconityMini')
        await self.client.get_config_id(cfg.info.config_name)
        await self.client.get_session_id(n=-1)
        print("Done init")

    async def initialParameterSettings(self):
        # Slowly scan the ones ignored
        for i in range(self.n_ignore_buffer):
            await self.client.change_part_parameter(i+1, 'mark_speed', 3000)

        # Parameters that will remain unchanged
        open_loop = self.m_cfg.aconity.open_loop
        for i in range(open_loop.shape[0]):
            await self.client.change_part_parameter(pieceNumber(i, self.ol_buffer), 'mark_speed', open_loop[i,0]*1000)
            await self.client.change_part_parameter(pieceNumber(i, self.ol_buffer), 'laser_power', open_loop[i,1])

    async def _changeMarkSpeed(self, part, value):
        print("Writing speed to %d " % (pieceNumber(part, self.cl_buffer)))
        await self.client.change_part_parameter(pieceNumber(part, self.cl_buffer), 'mark_speed', value)

    async def _changeLaserPower(self, part, value):
        await self.client.change_part_parameter(pieceNumber(part, self.cl_buffer), 'laser_power', value)

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
        cfg = self.m_cfg.aconity
        assert cfg.n_parts == actions.shape[0], \
            "Mismatch %d != %d" % (cfg.n_parts, actions.shape[0])

        # Random parameters
        rand_param = np.random.rand(self.n_rand, 2)*(self.ac_ub-self.ac_lb)+self.ac_lb
        for i in range(self.n_rand):
            print("Rand", i, "Piece", pieceNumber(i, self.rnd_buffer), "Setting parameters...", rand_param[i, :])
            await self.client.change_part_parameter(pieceNumber(i, self.rnd_buffer), 'mark_speed', rand_param[i,0]*1000)
            await self.client.change_part_parameter(pieceNumber(i, self.rnd_buffer), 'laser_power', rand_param[i,1])

        # Change parameters
        for part in range(cfg.n_parts):
            print("Part", part, "Setting parameters...", actions[part, :])
            await self._changeMarkSpeed(part, actions[part, 0]*1000)
            await self._changeLaserPower(part, actions[part, 1])

        # Resume / start job
        if self.job_started:
            print("Wait for job to be paused")
            while(not self.job_paused): pass
            await self.client.resume_job(layers=cfg.layers)
            self.job_paused = False
        else:
            execution_script = getUnheatedMonitoringExecutionScript()
            await self.client.start_job(cfg.layers, execution_script)
            await self.initialParameterSettings()
            print("Job started...")
            self.job_started = True
            self.sendStates()

        # Log actions taken
        np.save("saves/rand_param_l_%d.npy" % (self.curr_layer), rand_param)
        np.save("saves/param_l_%d.npy" % (self.curr_layer), actions)

        await self._pauseUponLayerCompletion()
        self.curr_layer += 1

    async def test_loop(self, actions):
        max_layer = self.m_cfg.aconity.layers[1]
        while self.curr_layer <= max_layer:
            await self.performLayer(actions)

    async def loop(self):
        max_layer = self.m_cfg.aconity.layers[1]
        while self.curr_layer <= max_layer:
            print("Layer %d/%d" % (self.curr_layer, max_layer))
            actions = await self.getActions()
            await self.performLayer(actions)
