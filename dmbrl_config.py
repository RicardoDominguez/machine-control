from dotmap import DotMap
import numpy as np
import tensorflow as tf
from dmbrl.modeling.models import BNN
from dmbrl.modeling.layers import FC
from dmbrl.misc.DotmapUtils import get_required_argument
from custom_env.machine_model import MachineModelEnv

# Controller parameters
# ------------------------------------------------------------------------------
CEM_ALPHA = 0.1
CEM_EPS = 0.001

# Policy search parameters
# ------------------------------------------------------------------------------
NTRAIN_ITERS = 5

# Model parameters
# ------------------------------------------------------------------------------
MODEL_IN, MODEL_OUT = 18, 16
HOLDOUT_RATIO = 0.0

# Model constructor
# ------------------------------------------------------------------------------
def bnn_constructor(model_init_cfg):
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    SESS = tf.Session(config=cfg)
    model = BNN(DotMap(
        name=get_required_argument(model_init_cfg, "model_name",
            "Must provide model name size"),
        num_networks=get_required_argument(model_init_cfg, "num_nets",
            "Must provide ensemble size"),
        sess=SESS,
        load_model=model_init_cfg.get("load_model", False),
        model_dir=model_init_cfg.get("model_dir", None)
    ))
    if not model_init_cfg.get("load_model", False):
        model.add(FC(model_init_cfg.n_neurons, input_dim=model_init_cfg.model_in,
            activation="swish", weight_decay=model_init_cfg.wd_in))
        for i in range(model_init_cfg.n_layers):
             model.add(FC(model_init_cfg.n_neurons, activation="swish",
                weight_decay=model_init_cfg.wd_hid))
        model.add(FC(model_init_cfg.model_out, weight_decay=model_init_cfg.wd_out))
    model.finalize(tf.train.AdamOptimizer, {"learning_rate": model_init_cfg.learning_rate})
    return model

# Preprocessing and postprocessing functions
# ------------------------------------------------------------------------------
def obs_preproc(obs):
    return obs

def obs_postproc(obs, pred):
    return obs + pred

def targ_proc(obs, next_obs):
    return next_obs - obs

# Cost functions
# ------------------------------------------------------------------------------
def obs_cost_fn(obs):
    target = 980
    k = 1000
    if isinstance(obs, np.ndarray):
        return -np.exp(-np.sum(np.square((obs-target)), axis=-1)/k)
    else:
        return -tf.exp(-tf.reduce_sum(tf.square((obs-target)), axis=-1)/k)

def ac_cost_fn(acs):
    return 0 # No constrains on actuators

# ------------------------------------------------------------------------------
# FUNCTION TO BE IMPORTED
# ------------------------------------------------------------------------------
def create_dmbrl_config():
    cfg = DotMap()

    cfg.ctrl_cfg.nO = 16
    cfg.ctrl_cfg.nU = 2

    cfg.ctrl_cfg.prop_cfg.model_init_cfg = DotMap(
        model_class = BNN,
        model_constructor = bnn_constructor
    )

    cfg.ctrl_cfg.prop_cfg.model_train_cfg = {"holdout_ratio" : HOLDOUT_RATIO}

    cfg.ctrl_cfg.opt_cfg.cfg = {"epsilon" : CEM_EPS,
                                "alpha" : CEM_ALPHA}

    cfg.ctrl_cfg.prop_cfg.obs_preproc = obs_preproc
    cfg.ctrl_cfg.prop_cfg.obs_postproc = obs_postproc
    cfg.ctrl_cfg.prop_cfg.targ_proc = targ_proc

    cfg.ctrl_cfg.opt_cfg.obs_cost_fn = obs_cost_fn
    cfg.ctrl_cfg.opt_cfg.ac_cost_fn = ac_cost_fn

    # Controller logging info
    cfg.ctrl_cfg.log_cfg.save_all_models = False
    cfg.ctrl_cfg.log_cfg.log_traj_preds = False
    cfg.ctrl_cfg.log_cfg.log_particles = False

    return cfg
