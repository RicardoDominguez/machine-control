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
        |- open_loop    - settings for open loop parts
"""
from dotmap import DotMap
import numpy as np

def get_n_parts(): return 20
def get_layers(): return [1, 153]

def returnSharedCfg():
    cfg = DotMap()

    cfg.comms.dir = 'io/'
    cfg.comms.action.rdy_name = 'action_rdy'
    cfg.comms.action.f_name = 'actions.npy'
    cfg.comms.state.rdy_name = 'state_rdy'
    cfg.comms.state.f_name = 'states.npy'
    cfg.env.nS = 16
    cfg.env.n_parts = get_n_parts()
    layers = get_layers()
    cfg.env.horizon = layers[1]-layers[0]+1

    cfg.save_dir1 = 'saves/'
    cfg.save_dir2 = ''#'/home/ricardo/'

    cfg.n_ignore = 3
    cfg.n_rand = 0

    cfg.ctrl_cfg.ac_ub = np.array([1.8, 140])
    cfg.ctrl_cfg.ac_lb = np.array([0.57, 75])

    return cfg

def returnMachineCfg():
    cfg = DotMap()

    cfg.comms.cluster_dir = 'machine-control/'
    cfg.comms.sftp.host = 'scentrohpc.shef.ac.uk'
    cfg.comms.sftp.user = 'ricardo'
    cfg.comms.sftp.pwd = 'aw%qzv'
    # cfg.comms.cluster_dir = '/data/coa16r/machine-control/'
    # cfg.comms.sftp.host = 'sharc.shef.ac.uk'
    # cfg.comms.sftp.user = 'coa16r'
    # cfg.comms.sftp.pwd = 'Pazz2314'
    cfg.aconity.info.config_name = 'Unheated 3D Monitoring Recalibrated'
    cfg.aconity.info.job_name = 'H282Control2'
    cfg.aconity.layers = get_layers()
    cfg.aconity.n_parts = get_n_parts()
    cfg.aconity.process.sess_dir = 'C:/AconitySTUDIO/log/'
    cfg.aconity.process.sleep_t = 2.5

    cfg.aconity.open_loop = np.ones((10,2))*[1.125, 110]
    return cfg
