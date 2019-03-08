"""
Showcases an example on how to use the python client and AconitySCRIPT to
control the light in the process chamber, Slider and Supplier.
"""

import asyncio
import aiohttp
import json, itertools
import sys
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


async def turn_lights_off_and_on(client, time=1):
    """ Use the execute command to turn ligh on and off """
    light_on = '$m.on($c[light])'
    light_off = '$m.off($c[light])'

    for i in itertools.count(): # Counts up
        await client.execute(channel='manual', script=light_on)
        console_output(light_on)
        await asyncio.sleep(time)

        await client.execute(channel='manual', script=light_off)
        console_output(light_off)
        await asyncio.sleep(time)

async def move_slider(client, time):
    while True:
        move_up = '$m.move_rel($c[slider], -20)'
        move_down = '$m.move_rel($c[slider], +20)'

        await client.execute(channel='manual_move', script=move_up)
        console_output(move_up)
        await asyncio.sleep(time)

        await client.execute(channel='manual_move', script=move_down)
        console_output(move_down)
        await asyncio.sleep(time)

async def move_supplier(client, time):
    while True:
        move_up = '$m.move_rel($c[supplier], -3)'
        move_down = '$m.move_rel($c[supplier], +3)'

        await client.execute(channel='manual_move', script=move_up)
        console_output(move_up)
        await asyncio.sleep(time)

        await client.execute(channel='manual_move', script=move_down)
        console_output(move_down)
        await asyncio.sleep(time)


async def move_optical_axis(client, time):
    while True:
        move_up = '$m.move_rel($c[optical_axis], -15)'
        move_down = '$m.move_rel($c[optical_axis], +15)'

        await client.execute(channel='manual_move', script=move_up)
        console_output(move_up)
        await asyncio.sleep(time)

        await client.execute(channel='manual_move', script=move_down)
        console_output(move_down)
        await asyncio.sleep(time)


async def main(login_data, info):
    #create client with factory method
    client = await AconitySTUDIOPythonClient.create(login_data)

    #gather information about machine id, config id, job id
    machine_name = info['machine_name']
    await client.get_machine_id(machine_name)

    config_name = info['config_name']
    await client.get_config_id(config_name)

    print('\n ...starting to create tasks and executing commands...\n')

    # Move commands
    slider = asyncio.create_task(move_slider(client,2))
    #supplier = asyncio.create_task(move_supplier(client,10))
    #optic = asyncio.create_task(move_optical_axis(client,10))

    # Flash LED
    lighter = asyncio.create_task(turn_lights_off_and_on(client, 0.2))

    await slider # Will not finish


if __name__ == '__main__':
    #Create a logfile for this session in the example folder.
    #The logfile contains the name of this script with a timestamp.
    #Note: This is just one possible way to log.
    #Logging configuration should be configured by the user, if any logging is to be used at all.
    utils.log_setup(sys.argv[0], directory_path='')

    #change login_data to your needs
    login_data = {
        'rest_url' : f'http://localhost:9000',
        'ws_url' : f'localhost:9000',
        'email' : 'admin@aconity3d.com',
        'password' : 'passwd'
    }

    #change info to your needs
    info = {
        'machine_name' : 'AconityMini',
        'config_name' : 'Unheated 3D Monitoring'
    }

    result = asyncio.run(main(login_data, info), debug=True)
