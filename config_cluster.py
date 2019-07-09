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
    cfg.n_parts = 0
    # --------------------------------------------------------------------------
    # CONTROL CONFIGURATION
    # --------------------------------------------------------------------------
    cfg.ctrl_cfg.per = 1
    cfg.ctrl_cfg.prop_cfg.model_pretrained = True
    upper_bounds = [1.8, 140]
    lower_bounds = [0.57, 75]
    cfg.ctrl_cfg.opt_cfg.constrains = [[np.array(lower_bounds), np.array(upper_bounds)], [60, 180], [70, 160]]

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
    cfg.n_parts = 33
    # --------------------------------------------------------------------------
    # CONTROL CONFIGURATION
    # --------------------------------------------------------------------------
    cfg.ctrl_cfg.per = 1
    cfg.ctrl_cfg.prop_cfg.model_pretrained = False
    upper_bounds = [1.8, 140]
    lower_bounds = [0.57, 75]
    cfg.ctrl_cfg.opt_cfg.constrains = [[np.array(lower_bounds), np.array(upper_bounds)], [65, 1000], [0, 104]]

    cfg.ctrl_cfg.change_target = True
    cfg.ctrl_cfg.n_parts_targets = [5, 5, 5, 18]
    cfg.ctrl_cfg.targets = [980, 1010, 1040, 1030]
    if cfg.ctrl_cfg.change_target:
        assert sum(cfg.ctrl_cfg.n_parts_targets) == cfg.n_parts, "Part missmatch change target"
        for i in range(len(cfg.ctrl_cfg.targets)-1):
            cfg.ctrl_cfg.n_parts_targets[i+1] += cfg.ctrl_cfg.n_parts_targets[i]

    cfg.ctrl_cfg.force.on = True
    cfg.ctrl_cfg.force.start_part = 16
    cfg.ctrl_cfg.force.n_parts = 3 # for each lower and upper
    cfg.ctrl_cfg.force.n_repeats = [1, 2, 3] # repeated for each lower and upper
    if cfg.ctrl_cfg.force.on:
        assert np.all(np.diff(cfg.ctrl_cfg.force.n_repeats)>0), "Must be in ascending order"
        assert cfg.ctrl_cfg.force.start_part-1+cfg.ctrl_cfg.force.n_parts*2*len(cfg.ctrl_cfg.force.n_repeats) == cfg.n_parts, "Part missmatch force"
    cfg.ctrl_cfg.force.init_buffer = 20
    cfg.ctrl_cfg.force.delta = 20
    cfg.ctrl_cfg.force.upper_init = 105
    cfg.ctrl_cfg.force.upper_delta = 5
    cfg.ctrl_cfg.force.lower_init = 60
    cfg.ctrl_cfg.force.lower_delta = -5
    cfg.ctrl_cfg.force.fixed_speed = 1.125

    cfg.ctrl_cfg.prop_cfg.model_init_cfg.load_model = False
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_dir = ''
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_name = 'learned_model'


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
