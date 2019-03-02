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

machine config parameters
------------------------
    |-comms
        |- sftp         - sftp config
            |- host
            |- user
            |- pwd
"""
from dotmap import DotMap

def returnSharedCfg():
    cfg = DotMap()

    cfg.comms.dir = 'io/'
    cfg.comms.action.rdy_name = 'action_rdy'
    cfg.comms.action.f_name = 'actions.npy'
    cfg.comms.state.rdy_name = 'state_rdy'
    cfg.comms.state.f_name = 'states.npy'

    return cfg

def returnMachineCfg():
    cfg = DotMap()

    cfg.comms.sftp.host = 'scentrohpc.shef.ac.uk'
    cfg.comms.sftp.user = 'ricardo'
    cfg.comms.sftp.pwd = 'aw%qzv'

    return cfg