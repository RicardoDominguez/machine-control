"""
Example on how to continually download the pyrometer and profiler data.

This script will periodically check if new data is available on the server and
download it to a local folder

To specify where the files shall be downloaded, control the quality (in percent,
downsampling parameter) and configure if the Pyrometer or Profiler (or both)
data should be downloaded please open the script in a text editor and customize
the `info` dictionary at the bottom of the script.

For each started session, started config and started job there exists individual
data. This means, `session_id`, `config_id` and `job_id` influence which files
will be downloaded. Note that inside `analyse_workunit_metadata` if statements
can be disabled so that all/more data is downloaded, for example all data from
one session.
"""

import asyncio
import aiohttp
import json
import sys
import os
import time
import logging
from datetime import datetime
from pytz import timezone, utc

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from AconitySTUDIO_client import AconitySTUDIOPythonClient
from AconitySTUDIO_client import utils


def analyse_workunit_metadata(client, workunit_metadata, topics):
    '''
    This function analyses what is received by the route
    data/{client.config_id}/workUnit/{client.job_id}.

    The main purpose is to create the download url.
    '''

    info = set()
    for data in workunit_metadata:
        sid = data['sessionId']
        cid = data['configId']
        jid = data['workId']
        hub = data['topic']
        creation = data['workCreation']
        #print(sid, cid, jid, hub)
        if (sid == client.session_id and \
            cid.split('_')[2] == client.config_id and \
            jid.split('_')[2] == client.job_id and \
            hub in topics):
            #data we are interested in! If interested in all data, simply remove ore change the statement.
            for attr in data['attributes']:
                sensorId = attr['id']
                for group in attr['groups']:
                    subpIdx = group['id']
                    for height in group['data']:
                        url = f'sessions/{sid}/configId/{cid}/jobIdx/{jid}/hub/{hub}/sensorId/{sensorId}/subpIdx/{subpIdx}/z/{height}'
                        info.add((url, sid, cid, jid, hub, sensorId, subpIdx, height, creation))
    return info

async def download_batch_data(client, batch_infos, base_path_pyrometer_data, quality=14):
    tasks = []
    print(client.rest_url)
    for info in batch_infos:
        url = info[0]
        info_tuple = info[1:]
        print(client.rest_url)
        info = {
                    #'session_id': info_tuple[0],
                    'session_id': '2019_02_13_10-16-09.845',
                    'config_id': info_tuple[1],
                    'job_id': info_tuple[2],
                    'topic': info_tuple[3],
                    'sensor_id': info_tuple[4],
                    'subpart_id': info_tuple[5],
                    'height': info_tuple[6],
                    'creation_time': str(int(info_tuple[7]) / 1000)
                }

        url += f'/quality/{quality}'

        base_path = f'{base_path_pyrometer_data}/{info["session_id"]}/{info["config_id"]}/{info["job_id"]}/sensors/{info["topic"]}/{info["sensor_id"]}/{info["subpart_id"]}'
        os.makedirs(base_path, exist_ok = True)

        save_to = base_path + f'/quality_{quality}%_layer_' + info["height"]
        if not os.path.isfile(save_to): #dont redownload files we already have
            tasks.append(asyncio.create_task(client.download_chunkwise(url, save_to, chunk_size=1024)))

    results = await asyncio.gather(*tasks)
    return results

async def main(login_data, info):
    '''
    1 Setup python client
    2 Get Batch data
        - every 5 seconds (sleep_time) we make a request to the server checking for new data
        - if new data arrived, we download it and save it locally

    Note: the folder structure from the server where the data is located is replicated for local use.
    '''

    # 1 Setup python client
    client = await AconitySTUDIOPythonClient.create(login_data)

    await client.get_job_id(info['job_name'])
    await client.get_machine_id(info['machine_name'])
    await client.get_config_id(info['config_name'])
    await client.get_session_id(n=-1) # this is the last session in time

    # 2 Get Batch data

    topics = info['topics']
    sleep_time = info['sleep_time']
    quality = info['quality']

    # this route gives us information about the batch data we can get
    workunit_metadata_url = f'data/{client.config_id}/workUnit/{client.job_id}'

    batch_infos = set([])
    while True:
        # extract informaton
        workunit_metadata_new = await client.get(workunit_metadata_url, log_level='debug')
        batch_infos_new = analyse_workunit_metadata(client, workunit_metadata_new, topics)

        #remove the old information from data already downloaded.
        batch_infos_changed = batch_infos_new - batch_infos

        logging.info(f'found {len(batch_infos_changed)} (possibly) new files to download')
        #print(f'NEW STUFF ({len(batch_infos_changed)}):', batch_infos_changed)

        await download_batch_data(client, batch_infos_changed, info['base_path_pyrometer_data'], quality)
        await asyncio.sleep(sleep_time)
        batch_infos = batch_infos_new




if __name__ == '__main__':
    '''
    Example on how to use the python client to continually get pyrometer data data.
    '''
    utils.log_setup(sys.argv[0], directory_path='')

    #change login_data to your needs
    login_data = {
        'rest_url' : 'http://localhost:9000',
        'ws_url' : 'ws://localhost:9000',
        'email' : 'admin@aconity3d.com',
        'password' : 'passwd'
    }

    #change info to your needs
    info = {
        'machine_name' : 'AconityMini',
        'config_name' : 'Unheated 3D Monitoring',
        'job_name' : 'SpeedTest',
        'base_path_pyrometer_data' : 'C:/AconitySTUDIO/log', #where to save the data locally
        'topics': ['2Pyrometer'], #['2Pyrometer', 'profiler'], #type of data
        'sleep_time': 5, #time between different checks of new data
        'quality' : 15 # the original data will be downsampled to {quality} percent.
    }

    result = asyncio.run(main(login_data, info), debug=True)
