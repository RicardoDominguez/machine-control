import asyncio
import aiohttp
import json, itertools
import sys
import time
from time import strftime
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
import logging

from AconitySTUDIO_client import AconitySTUDIOPythonClient
from AconitySTUDIO_client import utils

def current_time():
    return strftime('%Y-%m-%d %H:%M:%S')
def console_output(msg):
    print(f'{current_time()} -- executed {msg}\n')

# ---------------------------------
# CHANGE THIS TO IGNORE FIRST PIECE
# ---------------------------------
def pieceNumber(self, piece_indx):
    """ 0->4, 1->7, 2->10, etc... """
    return int((piece_indx+1)*3+1)

# ---------------------------------
# PREDETERMINED STRINGS
# ---------------------------------




# ---------------------------------
# CHANGE PARAMETERS
# ---------------------------------
async def changeMarkSpeed(client, part, value):
    await client.change_part_parameter(pieceNumber(part), 'mark_speed', value)

async def changeLaserPower(client, part, value):
    await client.change_part_parameter(pieceNumber(part), 'laser_power', value)

# ---------------------------------
# JOB MANAGING
# ---------------------------------
async def initJob(client, layers):
    execution_script = getUnheatedMonitoringExecutionScript()
    await client.start_job(layers, execution_script)
async def pauseJob(client): await client.pause_job()
async def resumeJob(client, layers): await client.resume_job(layers=layers)

self.client = await AconitySTUDIOPythonClient.create(getLoginData())
await self.client.get_job_id(info['job_name'])
await self.client.get_machine_id(info['machine_name'])
await self.client.get_config_id(info['config_name'])
await self.client.get_session_id(n=-1)



    def pauseUponLayerCompletion(self, sleep_time=0.05):
        """ sleep time in seconds """
        original = self.client.job_info['AddLayerCommands']
        while(original==self.client.job_info['AddLayerCommands']):
            await asyncio.sleep(sleep_time)
        await self.pauseJob()

    def setActions(self, actions):
        """ Set scan speed and laser power
        Assume job has been previously paused, then resume
        Inputs: actions - np.array (N x 2)
        """
        assert self.n_parts == actions.shape[0], \
            "Mismatch %d != %d" % (self.n_parts, actions.shape[0])
        for part in range(self.n_parts):
            self._changeMarkSpeed(part, actions[part, 0])
            if not self.onlySpeed: self._changeLaserPower(part, actions[part, 1])
        await self.resumeJob()
