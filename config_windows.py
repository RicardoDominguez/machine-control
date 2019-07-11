""" Defines the configuration parameters used by the `Aconity`, `Machine` and `Cluster` classes.

    - LASER_ON (bool): Laser is enabled when True.
    - JOB_NAME (str): Job name as displayed in the AconitySTUDIO web application.
    - LAYERS (array of int): Layer range to be built, as [layer_min, layer_max].
    - N_PARTS (int): Number of parts to be built (not regarding ignored parts).
    - N_STATES (int): Number of low-dimensional states used for the processing of the raw pyrometer data.
    - TEMPERATURE_TARGET (float): Temperature target in mV.
    - N_PARTS_IGNORED (int): Number of additional parts to be built on top of `N_PARTS` (pyrometer may not record data for the first few parts).
    - IGNORED_PARTS_SPEED (float): Scan speed used for parts being "ignored".
    - IGNORED_PARTS_POWER (float): Laser power used for parts being "ignored".
    - N_PARTS_FIXED_PARAMS (int): Number of parts built using fixed build parameters.
    - FIXED_PARAMS (array): Parameters to be used for those parts being built with fixed build parameters, as [speed (m/s), power (W)]
    - SLEEP_TIME_READING_FILES (float): Time between a sensor data file being first detected and attempting to read it. Prevents errors emerging from opening the file while it is still being written.
    - PART_DELTA (int): Parts of interest may increase 1 by 1, or 3 by 3 (refer to the AconitySTUDIO web application).
"""
from dotmap import DotMap
import numpy as np

LASER_ON = True
JOB_NAME = 'Constrained'
LAYERS = [1, 165]
N_PARTS = 33

N_STATES = 16
TEMPERATURE_TARGET = 980
INITIAL_PARAMETERS = [1.125, 110]

N_PARTS_IGNORED = 3
IGNORED_PARTS_SPEED = 1.125
IGNORED_PARTS_POWER = 110

N_PARTS_FIXED_PARAMS = 0
FIXED_PARAMS = [1.125, 110]

SLEEP_TIME_READING_FILES = 5
PART_DELTA = 1
N_PARTS_FIXED_PARAMS


def returnSharedCfg():
    """Return shared configuration parameters.

        - comms: Configuration parameters for server communication.
            - dir: Directory where files and folders will be written.
            - action
                - rdy_name: Name of folder created to signal action data has been written.
                - f_name: Name of .npy file containing the actions computed.
            - state
                - rdy_name: Name of folder created to signal state data has been uploaded.
                - f_name: Name of .npy file containing the state vectors processed.
        - env)]
            - nS: Dimensionality of the state vector.
            - n_parts: Number of parts to be built (not regarding ignored parts).
            - horizon: MDP event horizon (number of timesteps = number of layers).
        - n_ignore: Number of additional parts to be built on top of `env.n_parts` (pyrometer may not record data for the first few parts).
        - ctrl_cfg
            - ac_ub: Upper bounds of the build parameter in the form [speed (m/s), power (W)].
            - ac_lb: Lower bounds of the build parameter in the form [speed (m/s), power (W)].
    """
    cfg = DotMap()
    cfg.comms.dir = 'io/'
    cfg.comms.action.rdy_name = 'action_rdy' # Parts of interest may increase 1 by 1, or 3 by 3 (refer to the AconitySTUDIO web application).
    cfg.comms.action.f_name = 'actions.npy'
    cfg.comms.state.rdy_name = 'state_rdy'
    cfg.comms.state.f_name = 'states.npy'
    cfg.env.nS = N_STATES
    cfg.env.n_parts = N_PARTS
    cfg.env.horizon = LAYERS[1]-LAYERS[0]+1

    cfg.save_dir1 = 'saves/'
    cfg.save_dir2 = ''#'/home/ricardo/'

    cfg.parts_ignored = 3

    cfg.env.init_params = INITIAL_PARAMETERS

    cfg.save_dir1 = 'saves/'
    cfg.save_dir2 = ''

    cfg.n_ignore = N_PARN_PARTS_FIXED_PARAMSTS_IGNORED

    cfg.ctrl_cfg.ac_ub = np.array([1.8, 140])
    cfg.ctrl_cfg.ac_ub = np.array([0.57, 75])

    return cfg

def returnMachineCfg():
    """ Return configuration parameters for the `Machine` class.

        - comms: Configuration parameters for server communication.
            - cluster_dir: Location of the code within the remote server.
            - sftp: Configuration parameters for SFTP communication.
                - host: Address of the remote server, i.e. scentrohpc.shef.ac.uk
                - user: User name (login credentials).
                - pwd: Password (login credentials).
        - aconity: Machine parameters for the use of the Aconity API.
            - info
                - config_name: Configuration name, i.e. `Unheated 3D Monitoring`.
                - job_name: Job name as displayed in the AconitySTUDIO web application.
            - layers: Layer range to be bu- open_loop: Parameters used to build the parts built using fixed parameters, np.array with shape (`n_fixed_parts`, 2) as [layer_min, layer_max]
            - n_parts: Number of parts to be built, excluding ignored parts.
            - process: ConfiguLASER_ONration parameters for processing sensory data.
                - sess_dir: Folder where pyrometer data is stored by the Aconity machine.
                - sleep_t: Time between a sensor data file being first detected and attempting to read it. Prevents errors emerging from opening the file while it is still being written.
                - debug_dir: Folder where to save information useful for debugging.
            - ignored_parts_power: Scan speed used for parts being "ignored".
            - ignored_parts_speed: Laser power used for parts being "ignored".
            - part_delta: Parts of interest may increase 1 by 1, or 3 by 3 (refer to the AconitySTUDIO web application).
            - fixed_params: Parameters used for those parts being built with fixed parameters, with shape (`n_fixed_parts`, 2)
    """
    cfg = DotMap()

    cfg.comms.cluster_dir = 'CONSTRAINS/'#'machine-control/'
    cfg.comms.sftp.host = 'scentrohpc.shef.ac.uk'
    cfg.comms.sftp.user = 'ricardo'
    cfg.comms.sftp.pwd = 'aw%qzv'
    cfg.aconity.info.config_name = 'Unheated 3D Monitoring Recalibrated'
    cfg.aconity.info.job_name = JOB_NAME
    cfg.aconity.layers = LAYERS
    cfg.aconity.process.sess_dir = 'C:/AconitySTUDIO/log/'
    cfg.aconity.process.sleep_t = SLEEP_TIME_READING_FILES
    cfg.aconity.ignored_parts_speed = IGNORED_PARTS_SPEED
    cfg.aconity.ignored_parts_power = IGNORED_PARTS_POWER
    cfg.aconity.part_delta = 1
    cfg.aconity.fixed_params = np.ones((N_PARTS_FIXED_PARAMS,2))*FIXED_PARAMS
    cfg.aconity.laser_on = LASER_ON
    return cfg
