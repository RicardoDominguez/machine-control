""" Defines the configuration parameters used by the `Aconity`, `Machine` and `Cluster` classes. """
from dotmap import DotMap
import numpy as np

def get_n_parts():
    """Returns the number of parts to be built.

    This excludes the parts being ignored (cfg.n_ignore)

    Returns:
        int: Number of parts to be built.
    """
    return 20

def get_layers():
    """Returns the layer range to be built, as [layer_min, layer_max]

    Returns:
        array of ints: Layer range to build.
    """
    return [1, 153]

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
        - env
            - nS: Dimensionality of the state vector.
            - n_parts: Number of parts to be built (not regarding ignored parts).
            - horizon: MDP event horizon (number of timesteps = number of layers).
        - n_ignore: Number of additional parts to be built on top of `env.n_parts` (pyrometer may not record data for the first few parts).
        - n_rand: Number of parts to be built with random parameters.
        - ctrl_cfg
            - ac_ub: Upper bounds of the build parameter in the form [speed (m/s), power (W)].
            - ac_lb: Lower bounds of the build parameter in the form [speed (m/s), power (W)].
    """
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
    cfg.save_dir2 = ''

    cfg.n_ignore = 3
    cfg.n_rand = 0

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
            - layers: Layer range to be built, as [layer_min, layer_max]
            - n_parts: Number of parts to be built, excluding ignored parts.
            - process: Configuration parameters for processing sensory data.
                - sess_dir: Folder where pyrometer data is stored by the Aconity machine.
                - sleep_t: Time between a sensor data file being first detected and attempting to read it. Prevents errors emerging from opening the file while it is still being written.
                - debug_dir: Folder where to save information useful for debugging.
            - open_loop: Parameters used to build the parts built using fixed parameters, np.array with shape (`n_fixed_parts`, 2).
    """
    cfg = DotMap()

    cfg.comms.cluster_dir = 'machine-control/'
    cfg.comms.sftp.host = 'scentrohpc.shef.ac.uk'
    cfg.comms.sftp.user = 'ricardo'
    cfg.comms.sftp.pwd = 'aw%qzv'
    cfg.aconity.info.config_name = 'Unheated 3D Monitoring Recalibrated'
    cfg.aconity.info.job_name = 'H282Control2'
    cfg.aconity.layers = get_layers()
    cfg.aconity.n_parts = get_n_parts()
    cfg.aconity.process.sess_dir = 'C:/AconitySTUDIO/log/'
    cfg.aconity.process.sleep_t = 2.5

    cfg.aconity.open_loop = np.ones((10,2))*[1.125, 110]
    return cfg
