"""
Subscribes to various Topics and Reports in order to get machine data.
Setup a mongodb database in which to continually save the machine data.
"""

import asyncio
import aiohttp
import json

import logging
from pytz import timezone, utc
from datetime import datetime

import sys
import signal, json
signal.signal(signal.SIGINT, signal.SIG_DFL)

from AconitySTUDIO_client import AconitySTUDIOPythonClient
from AconitySTUDIO_client import utils

def check_topic_state(client, msg, component):
    '''checks the state of a component'''
    if 'topic' in msg and msg['topic'] == 'State':
        for data in msg['data']:
            if data['cid'] == component:
                information = data['data']
                for info in information:
                    #print(json.dumps(info, indent=3))
                    #print(json.dumps(msg, indent=3))
                    value = info["value"]
                    #color = Fore.RED if value == 'False' else Fore.GREEN
                    print(f'{component:30}\t{info["name"]:10}{info["value"]}')
                break

def print_run_report(client, msg):
    '''looks out if a channel gets paused, resumed or stopped'''
    if 'topic' in msg and msg['topic'] == 'run':
        time_string = utils.get_time_string(msg['data'][0]['ts'])
        messages = ''
        for run_msg in msg['data']:
            raw_time_str = run_msg['ts']
            time_str = '' #get_time_string(run_msg['ts'], format = '%H:%M:%S' )
            messages += f"\t{run_msg['channel']}: {run_msg['msg']}\t({raw_time_str})\n"
        if 'finished' in messages or 'paused' in messages or 'stopped' in messages:
            messages += '-----------------------------------\n'
        print(f'\nrun report{time_string}:\n{messages}')
        #print(f'\nrun report{time_string}:\n{json.dumps(msg, indent=3)}\n')

def layer_number_cmd(client, msg):
    '''looks out for how many layers where added during a running job'''
    if 'topic' in msg and msg['topic'] == 'cmds' and 'data' in msg:
        for data in msg['data']:
            if 'name' in data and 'value' in data and data['name'] == 'report':
                value = json.loads(data['value'])
                print(value)

def pan_processor(client, msg):
    '''simply prints every ws message'''
    print('GENERIC PROCESSOR---------------\n')
    print(json.dumps(msg, indent=3))
    print('------------END GENERIC PROCESSOR')

async def main(login_data, info):
    '''
    1) Setup python client.
    2) (optional) Get information about available topics.
    3) (optional) Activate the pymongo database (uncomment the line with
        ...enable_pymongo_database).
    4) Subscribe to various topics and reports.
    5) Process the data received from the websocket connection using the
        processor functions defined above (check_topic_state, print_run_report,
        layer_number_cmd, pan_processor). A processor function takes the Python
        client as its first argument and a websocket message (dict/json) as its
        second argument.
    '''

    # 1)
    client = await AconitySTUDIOPythonClient.create(login_data)

    # 2)
    if 'config_name' in info:
        await client.get_config_id(info['config_name'])
        topics_info_route = f'configurations/{client.config_id}/topics'
        available_topics = await client.get(topics_info_route)

        # uncomment next line to print out available_topics
        print(json.dumps(available_topics, indent=3), 'End of Topics/Reports\n\n')


    # 3)
    # enable the process_data() function to write to pymongo database
    # If the name 'dashboard_database' gets changed and the dashboard should be used, please change dashboard.py accordingly
    # keep only the last 60 seconds in the database
    # client.enable_pymongo_database(name='dashboard_database', keep_last = 60) #enable this if you have MongoDB Community Edition installed


    # 4)
    # Information about the commands running on the Server.
    await client.subscribe_report('cmds')
    # Relevant for script execution.
    await client.subscribe_report('run')
    # Enables us to see the status of the light, the cover lock and many other
    await client.subscribe_topic('State')

    # 5)
    # Indiscriminatory processor (print out all websocket messages received)
    #client.processors.append(pan_processor)

    # Print number of layers
    #client.processors.append(layer_number_cmd) # While running a job.

     # Run report prints out information about script execution.
    client.processors.append(print_run_report)

    # Enable and disable light and see what happens here
    client.processors.append(lambda client, msg: check_topic_state(client, msg, 'process_chamber::illumination'))

    # Cover lock status
    #client.processors.append(lambda client, msg: check_topic_state(client, msg, 'cover_lock::lock'))

    # Runs until the ws connection created in client._receive_websocket_data is closed.
    # This coroutine put on the event loopd during step 1) in the creation of the client.
    await client.ws_processing_task

if __name__ == '__main__':
    """
    Example on how to use the python client to continually get machine data.
    """
    utils.log_setup(sys.argv[0], directory_path='')

    #change login_data to your needs
    login_data = {
        'rest_url' : f'http://192.168.1.62:9000',
        'ws_url' : f'ws://192.168.1.62:9000',
        'email' : 'admin@aconity3d.com',
        'password' : 'passwd'
    }

    #change info to your needs
    info = {
        'machine_name' : 'AconityMini',
        'config_name' : 'Unheated 3D Monitoring' #'SingleLaser'
    }

    result = asyncio.run(main(login_data, info), debug=True)

