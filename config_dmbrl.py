""" Low level configuration for modeling and optimisation.

Global variables:
    - MODEL_IN (int): Number of inputs to the model.
    - MODEL_OUT (int): Number of outputs of the model.
    - HOLDOUT_RATIO (float): Percentage of training data used for checking model performance.
    - CEM_ALPHA (float): Alpha parameter of the CEM optimisation algorithm.
    - CEM_EPS (float): Epsilon parameter of the CEM optimisation algorithm.
"""
from dotmap import DotMap
import numpy as np
import tensorflow as tf
import config_windows as cfg_global
from dmbrl.modeling.models import BNN
from dmbrl.modeling.layers import FC
from dmbrl.misc.DotmapUtils import get_required_argument

# Model parameters
# ------------------------------------------------------------------------------
MODEL_IN, MODEL_OUT = 18, 16
HOLDOUT_RATIO = 0.0


# Controller parameters
# ------------------------------------------------------------------------------
CEM_ALPHA = 0.1
CEM_EPS = 0.001


# Model constructor
# ------------------------------------------------------------------------------
def bnn_constructor(model_init_cfg):
    """ Constructs the Bayesian Neural Network model.

    Moodel_init_cfg is a dotmap object containing:
        - model_in (int): Number of inputs to the model.
        - model_out (int): Number of outputs to the model.
        - n_layers (int): Number of hidden layers.
        - n_neurons (int): Number of neurons per hidden layer.
        - learning_rate (float): Learning rate.
        - wd_in (float): Weight decay for the input layer neurons.
        - wd_hid (float): Weight decay for the hidden layer neurons.
        - wd_out (float): Weight decay for the output layer neurons.

    Returns:
        BNN class object
    """
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    SESS = tf.Session(config=cfg) # Tensorflow session
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
    """Modifies observations (in a 2D matrix) before they are passed into the model.

    Arguments:
        obs (np.array or tf.Tensor): Array of observations with shape (`n`, `dO`).

    Returns:
        np.array or tf.Tensor with shape (`n`, `model_in`)
    """
    return obs # Not modified

def obs_postproc(obs, pred):
    """Modifies observations and model predictions before being passed to cost function.

    Arguments:
        obs (np.array or tf.Tensor): Array of observations with shape (`n`, `dO`).
        pred (np.array or tf.Tensor): Array of predictions (model outputs) with shape (`n`, `model_output`).

    Returns:
        np.array or tf.Tensor with shape (`n`, `dO`)
    """
    return obs + pred

def targ_proc(obs, next_obs):
    """Takes current observations and next observations and returns the array of
    targets (so that the model learns the mapping obs -> targ_proc(obs, next_obs))

    Arguments:
        obs (np.array or tf.Tensor): Array of observations at time `t` with shape (`n`, `dO`).
        next_obs (np.array or tf.Tensor): Array of observations at time `t+1` with shape (`n`, `dO`).

    Returns:
        np.array or tf.Tensor with shape (`n`, `model_out`)
    """
    return next_obs - obs

# Cost functions
# ------------------------------------------------------------------------------
<<<<<<< HEAD:dmbrl_config.py
def obs_cost_fn(obs, target):
    """ Cost function (state-dependent) used in the optimisation problem.

    Should process both np.arrays and tf.Tensor inputs.

    Arguments:
        obs (np.array or tf.Tensor): Array of observations with shape (`n`, `dO`).
        target (float)

    Returns:
        float
    """
    target = 980
=======
def obs_cost_fn(obs, target):
>>>>>>> 85c52db3870692cc2b998f39ae6e609c5d8c6190:config_dmbrl.py
    k = 1000
    if isinstance(obs, np.ndarray):
        return -np.exp(-np.sum(np.square((obs-target)), axis=-1)/k)
    else:
        return -tf.exp(-tf.reduce_sum(tf.square((obs-target)), axis=-1)/k)

def ac_cost_fn(acs):
    """ Cost function (action-dependent) used in the optimisation problem.

    Should process both np.arrays and tf.Tensor inputs.

    Arguments:
        acs (np.array or tf.Tensor): Array of actions with shape (`n`, `dU`).

    Returns:
        float
    """
    return 0 # No constrains on actuators

# ------------------------------------------------------------------------------
# FUNCTION TO BE IMPORTED
# ------------------------------------------------------------------------------
def create_dmbrl_config():
    """Returns the low-level modeling and optimisation configuration parameters."""

    cfg = DotMap()

    cfg.ctrl_cfg.dO = 16
    cfg.ctrl_cfg.dU = 2

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
    cfg.ctrl_cfg.opt_cfg.target = cfg_global.TEMPERATURE_TARGET

    # Controller logging info
    cfg.ctrl_cfg.log_cfg.save_all_models = False
    cfg.ctrl_cfg.log_cfg.log_traj_preds = False
    cfg.ctrl_cfg.log_cfg.log_particles = False

    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_in = MODEL_IN
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_out = MODEL_OUT
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.n_layers = 3
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.n_neurons = 750
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.wd_in = 8.213e-05
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.wd_hid = 1.188e-05
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.wd_out = 1.004e-05
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.learning_rate = 3.61e-4
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_constructor = bnn_constructor

    return cfg
