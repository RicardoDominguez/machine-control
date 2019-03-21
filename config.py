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

control config parameters
    |- pretrained       - learns or model is given
    |- train_freq       - model is learned after this number of layers
    |- ctrl_cfg
        |- per          - reoptimization frequency
        |- ac_ub        - action upper bounds
        |- ac_lb        - action lower bounds
        |- dO           - number of observations
        |- dU           - number of actions
        |- prop_cfg
            |- model_pretrained
            |- model_init_cfg
                |- ensemble_size
                |- load_model
                |- model_dir
                |- model_name
            |- model_train_cfg
                |- batch_size
                |- epochs
                |- hide_progress
            |- npart
        |- opt_cfg
            |- mode
            |- plan_hor
            |- cfg
                |- popsize
                |- max_iters
                |- num_elites
            |- prop_cfg
                |- mode - ie "TSinf"
"""
from dotmap import DotMap
from dmbrl_config import create_dmbrl_config

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

def returnClusterPretrainedCfg():
    cfg = create_dmbrl_config()

    cfg.pretrained = True
    cfg.train_freq = None
    # --------------------------------------------------------------------------
    # CONTROL CONFIGURATION
    # --------------------------------------------------------------------------
    cfg.ctrl_cfg.per = 1
    cfg.ctrl_cfg.prop_cfg.model_pretrained = True
    cfg.ctrl_cfg.ac_ub = np.array([1.8, 140])
    cfg.ctrl_cfg.ac_lb = np.array([0.57, 75])

    cfg.ctrl_cfg.prop_cfg.model_init_cfg.num_nets = 1
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.load_model = True
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_dir = 'dmbrl/trained_models/'
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_name = 'pretrained_model'

    cfg.ctrl_cfg.prop_cfg.model_train_cfg["batch_size"] = 32
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["epochs"] = 5
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["hide_progress"] = False

    cfg.ctrl_cfg.prop_cfg.npart = 20

    cfg.ctrl_cfg.opt_cfg.mode = "CEM"
    cfg.ctrl_cfg.opt_cfg.plan_hor = 3

    cfg.ctrl_cfg.opt_cfg.cfg["popsize"] = 2000
    cfg.ctrl_cfg.opt_cfg.cfg["max_iters"] = 10
    cfg.ctrl_cfg.opt_cfg.cfg["num_elites"] = int(2000*0.05)

    cfg.ctrl_cfg.prop_cfg.mode = "TSinf"

    return cfg


def returnClusterUnfamiliarCfg():
    cfg = create_dmbrl_config()

    cfg.pretrained = False
    cfg.train_freq = 10
    # --------------------------------------------------------------------------
    # CONTROL CONFIGURATION
    # --------------------------------------------------------------------------
    cfg.ctrl_cfg.per = 1
    cfg.ctrl_cfg.prop_cfg.model_pretrained = False
    cfg.ctrl_cfg.ac_ub = np.array([1.8, 140])
    cfg.ctrl_cfg.ac_lb = np.array([0.57, 75])

    cfg.ctrl_cfg.prop_cfg.model_init_cfg.num_nets = 5
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.load_model = False
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_dir = ''

    cfg.ctrl_cfg.prop_cfg.model_train_cfg["batch_size"] = 32
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["epochs"] = 5
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["hide_progress"] = False

    cfg.ctrl_cfg.prop_cfg.npart = 20

    cfg.ctrl_cfg.opt_cfg.mode = "CEM"
    cfg.ctrl_cfg.opt_cfg.plan_hor = 3

    cfg.ctrl_cfg.opt_cfg.cfg["popsize"] = 2000
    cfg.ctrl_cfg.opt_cfg.cfg["max_iters"] = 10
    cfg.ctrl_cfg.opt_cfg.cfg["num_elites"] = int(2000*0.05)

    cfg.ctrl_cfg.prop_cfg.mode = "TSinf"

    return cfg
