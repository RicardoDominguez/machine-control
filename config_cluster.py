"""
control config parameters
-------------------------
    |- pretrained       - learns or model is given
    |- train_freq       - model is learned after this number of layers
    |- n_parts          - number of physical parts being built
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
from config_dmbrl import create_dmbrl_config
import numpy as np

def returnClusterPretrainedCfg():
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
