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

async def waitNextLayer(client, sleep_time=0.05):
    original = client.job_info['AddLayerCommands']
    print("Awaiting for next layer (not %d)" % (original))
    while(original==client.job_info['AddLayerCommands']):
        await asyncio.sleep(sleep_time)
    return

async def main(login_data, info):
    client = await AconitySTUDIOPythonClient.create(login_data)

    await client.get_job_id(info['job_name'])
    await client.get_machine_id(info['machine_name'])
    await client.get_config_id(info['config_name'])
    await client.get_session_id(n=-1) # this is the last session in time

    # start job
    layers = [1,166]
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
    await client.start_job(layers, execution_script)

    print('\n ...starting to create tasks and executing commands...\n')

    speedSlow = True
    sleep_time = 60
    start_layer = layers[0]
    while(True):
        print("Pausing job...")
        start_time = time.time()
        await client.pause_job()
        speed = 400 if speedSlow else 3000
        speedSlow = False if speedSlow else True
        await client.change_part_parameter(1, 'mark_speed', speed)
        await client.change_part_parameter(4, 'mark_speed', speed)
        await client.change_part_parameter(7, 'mark_speed', speed)
        await client.change_part_parameter(10, 'mark_speed', speed)
        built_layers = client.job_info['AddLayerCommands']
        print("Resuming job...")
        await client.resume_job(layers=layers)
        print("Took %.4f to change parameters" % (time.time()-start_time))
        start_time = time.time()
        print("Speed changed to %d" % (speed))
        await waitNextLayer(client)
        print("New layer %d, time %.4f" % (start_layer+built_layers, time.time()-start_time))

if __name__ == '__main__':
    '''
    Example on how to use the python client for executing scripts.
    Please change login_data and info to your needs.
    '''

    #Create a logfile for this session in the example folder.
    #The logfile contains the name of this script with a timestamp.
    #Note: This is just one possible way to log.
    #Logging configuration should be configured by the user, if any logging is to be used at all.
    utils.log_setup(sys.argv[0], directory_path='')

    #change login_data to your needs
    login_data = {
        'rest_url' : f'http://143.167.193.57:9000',
        'ws_url' : f'ws://143.167.193.57:9000',
        'email' : 'admin@aconity3d.com',
        'password' : 'passwd'
    }

    #change info to your needs
    info = {
        'machine_name' : 'AconityMini',
        'config_name' : 'Unheated 3D Monitoring',
        'job_name' : 'SpeedTest'
    }

    result = asyncio.run(main(login_data, info), debug=True)
