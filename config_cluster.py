""" Defines the control configuration parameters.
    - pretrained: `True` if model used for MPC has been trained on previous data, `False` if it is learned in real-time.
    - train_freq: model is trained every `train_freq` layers.
    - n_parts: number of parts built using this control strategy.
    - ctrl_cfg: Configuration parameters for the control algorithm.
        - per: How often the action sequence will be optimized, i.e, for per=1 it is reoptimized at every call to `MPC.act()`.
        - ac_ub: Upper bounds of the build parameter in the form [speed (m/s), power (W)].
        - ac_lb: Lower bounds of the build parameter in the form [speed (m/s), power (W)].
        - prop_cfg: Configuration parameters for modeling and uncertainty propagation.
            - model_pretrained: `True` if model used for MPC has been trained on previous data, `False` otherwise.
            - model_init_cfg: Configuration parameters for model initialisation.
                - ensemble_size: Number of models within the ensemble.
                - load_model: `True` for a pretrained model to be loaded upon initialisation.
                - model_dir: Directory in which the model files (.mat, .nns) are located.
                - model_name: Name of the model files (model_dir/model_name.mat or model_dir/model_name.nns)
            - model_train_cfg: Configuration parameters for model training optimisation
                - batch_size: Batch size.
                - epochs: Number of training epochs.
                - hide_progress: If 'True', additional information regarding model training is printed.
            - npart: Number of particles used for uncertainty propagation.
        - opt_cfg: Configuration parameters for optimisation.
            - mode: Uncertainty propagation method.
            - plan_hor: Planning horizon for the model predictive control algorithm.
            - cfg
                - popsize: Number of cost evaluations per iteration.
                - max_iters: Maximum number of optimisation iterations.
                - num_elites: Number of elites.
            - prop_cfg
                - mode: Uncertainty propagation method, ie "TSinf"
"""
from dotmap import DotMap
from dmbrl_config import create_dmbrl_config
from config_windows import get_n_parts
import numpy as np

def returnClusterPretrainedCfg():
    """A pretrained agent makes use solely of a model trained using previously
    collected data.

    Returns:
        dotmap: Configuration parameters for control with a method trained on previous data.
    """
    cfg = create_dmbrl_config()

    cfg.pretrained = True
    cfg.train_freq = None
    cfg.n_parts = 10
    # --------------------------------------------------------------------------
    # CONTROL CONFIGURATION
    # --------------------------------------------------------------------------
    cfg.ctrl_cfg.per = 1
    cfg.ctrl_cfg.prop_cfg.model_pretrained = True
    cfg.ctrl_cfg.ac_ub = np.array([1.8, 140])
    cfg.ctrl_cfg.ac_lb = np.array([0.57, 75])

    cfg.ctrl_cfg.prop_cfg.model_init_cfg.load_model = True
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_dir = 'dmbrl/trained_models/'
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_name = 'pretrained_model'

    cfg.ctrl_cfg.prop_cfg.model_init_cfg.num_nets = 1
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["batch_size"] = 32
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["epochs"] = 5
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["hide_progress"] = False

    cfg.ctrl_cfg.prop_cfg.npart = 20

    cfg.ctrl_cfg.opt_cfg.mode = "CEM"
    cfg.ctrl_cfg.opt_cfg.plan_hor = 1

    cfg.ctrl_cfg.opt_cfg.cfg["popsize"] = 500
    cfg.ctrl_cfg.opt_cfg.cfg["max_iters"] = 10
    cfg.ctrl_cfg.opt_cfg.cfg["num_elites"] =int(500*0.05)

    cfg.ctrl_cfg.prop_cfg.mode = "TS1"

    return cfg


def returnClusterUnfamiliarCfg():
    """ An unfamiliar agent pretrained agent makes use solely of a model trained
    using data collected in real-time.

    Returns:
        dotmap: Configuration parameters for control with a method trained in real-time.
    """
    cfg = create_dmbrl_config()

    cfg.pretrained = False
    cfg.train_freq = 1
    cfg.n_parts = 10
    # --------------------------------------------------------------------------
    # CONTROL CONFIGURATION
    # --------------------------------------------------------------------------
    cfg.ctrl_cfg.per = 1
    cfg.ctrl_cfg.prop_cfg.model_pretrained = False
    cfg.ctrl_cfg.ac_ub = np.array([1.8, 140])
    cfg.ctrl_cfg.ac_lb = np.array([0.57, 75])

    cfg.ctrl_cfg.prop_cfg.model_init_cfg.load_model = False
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_dir = ''
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_name = 's75_2'


    cfg.ctrl_cfg.prop_cfg.model_init_cfg.num_nets = 5
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["batch_size"] = 32
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["epochs"] = 5
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["hide_progress"] = False

    cfg.ctrl_cfg.prop_cfg.npart = 20

    cfg.ctrl_cfg.opt_cfg.mode = "CEM"
    cfg.ctrl_cfg.opt_cfg.plan_hor = 1

    cfg.ctrl_cfg.opt_cfg.cfg["popsize"] = 500
    cfg.ctrl_cfg.opt_cfg.cfg["max_iters"] = 10
    cfg.ctrl_cfg.opt_cfg.cfg["num_elites"] = int(500*0.05)

    cfg.ctrl_cfg.prop_cfg.mode = "TS1"

    return cfg
