"""
shared config parameters
------------------------
    |- comms
        |- dir          - directory where files will be written
        |- action
            |- rdy_name - name of file created to signal data has been written
            |- f_name   - name of file containing .npy data
        |- state
            |- dir      - directory where files will be written before being sent
            |- rdy_name - name of file uploaded to signal end of data upload
            |- f_name   - name of file containing .npy data
    |- env
        |- nS           - number of states
        |- n_parts
        |- horizon

machine config parameters
------------------------
    |- comms
        |- cluster_dir   - directory in which machine-control/ is located on the cluster
        |- sftp          - sftp config
            |- host
            |- user
            |- pwd
    |- aconity
        |- info
            |- config_name
            |- job_name
        |- layers       - [start, end]
        |- n_parts
        |- process
            |- sess_dir - folder where data is logged
            |- sleep_t  - after detecting file and before reading it
            |- debug_dir- save information useful for debuging
        |- fixed_params    - settings for open loop parts
        |- ignored_parts_speed
"""
from dotmap import DotMap
import numpy as np

TEMPERATURE_TARGET = 980
LAYERS = [1, 168]

def returnSharedCfg():
    cfg = DotMap()

    cfg.comms.dir = 'io/'
    cfg.comms.action.rdy_name = 'action_rdy'
    cfg.comms.action.f_name = 'actions.npy'
    cfg.comms.state.rdy_name = 'state_rdy'
    cfg.comms.state.f_name = 'states.npy'
    cfg.env.nS = 16
    cfg.env.n_parts = 5
    cfg.env.horizon = LAYERS[1]-LAYERS[0]+1

    cfg.save_dir1 = 'saves/'
    cfg.save_dir2 = ''#'/home/ricardo/'

    cfg.parts_ignored = 3

    cfg.env.init_params = [4, 0]

    return cfg

def returnMachineCfg():
    cfg = DotMap()

    cfg.comms.cluster_dir = 'CEM-OPT/'#'machine-control/'
    cfg.comms.sftp.host = 'scentrohpc.shef.ac.uk'
    cfg.comms.sftp.user = 'ricardo'
    cfg.comms.sftp.pwd = 'aw%qzv'
    cfg.aconity.info.config_name = 'Unheated 3D Monitoring Recalibrated'
    cfg.aconity.info.job_name = 'H282Control2'
    cfg.aconity.layers = LAYERS
    cfg.aconity.process.sess_dir = 'C:/AconitySTUDIO/log/'
    cfg.aconity.process.sleep_t = 3.5
    cfg.aconity.ignored_parts_speed = 3000
    cfg.aconity.part_delta = 1
    cfg.aconity.fixed_params = np.ones((25,2))*[4, 0]
    cfg.aconity.laser_on = False
    return cfg
