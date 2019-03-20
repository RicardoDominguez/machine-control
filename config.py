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

machine config parameters
------------------------
    |- comms
        |- sftp         - sftp config
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
"""
from dotmap import DotMap

def returnSharedCfg():
    cfg = DotMap()

    cfg.comms.dir = 'io/'
    cfg.comms.action.rdy_name = 'action_rdy'
    cfg.comms.action.f_name = 'actions.npy'
    cfg.comms.state.rdy_name = 'state_rdy'
    cfg.comms.state.f_name = 'states.npy'
    cfg.env.nS = 16

    return cfg

def returnMachineCfg():
    cfg = DotMap()

    cfg.comms.sftp.host = 'scentrohpc.shef.ac.uk'
    cfg.comms.sftp.user = 'ricardo'
    cfg.comms.sftp.pwd = 'aw%qzv'
    cfg.aconity.info.config_name = 'Unheated 3D Monitoring'
    cfg.aconity.info.job_name = 'SpeedTest'
    cfg.aconity.layers = [1, 166]
    cfg.aconity.n_parts = 3
    cfg.aconity.process.sess_dir = 'C:/AconitySTUDIO/log/'
    cfg.aconity.process.sleep_t = 2
    return cfg
