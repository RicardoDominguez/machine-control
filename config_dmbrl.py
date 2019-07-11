""" Low level configuration for modeling and optimisation.

    - ctrl_cfg: Configuration parameters for the control algorithm.

        -dO: dimensionality of observations
        -dU: dimensionality of control inputs
        - per: How often the action sequence will be optimized, i.e, for per=1 it is reoptimized at every call to `MPC.act()`.
        - constrains: [[np.array([min v, min q]), np.array([max v, max q])], [min q/v, max q/v], [min q/sqrt(v), max q/sqrt(v)]]
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
            - model_in: Number of inputs to the model.
            - model_out: Number of outputs to the model.
            - n_layers: Number of hidden layers.
            - n_neurons: Number of neurons per hidden layer.
            - learning_rate: Learning rate.
            - wd_in: Weight decay for the input layer neurons.
            - wd_hid: Weight decay for the hidden layer neurons.
            - wd_out: Weight decay for the output layer neurons.

        - opt_cfg: Configuration parameters for optimisation.

            - mode: Uncertainty propagation method.
            - plan_hor: Planning horizon for the model predictive control algorithm.
            - cfg

                - popsize: Number of cost evaluations per iteration.
                - max_iters: Maximum number of optimisation iterations.
                - num_elites: Number of elites.
                - alpha: Alpha parametero of the CEM optimisation algorithm.
                - eps: Epsilon parameter of the CEM optimisation algorithm.

            - prop_cfg

                - mode: Uncertainty propagation method, ie "TSinf"

        - change_target: True if multiple setpoints used, i.e. 980 and 1010
        - n_parts_targets: Number of parts to be built for each target
        - targets: Different temperature setpoints to be used (must be of same length as `n_parts_targets`)
        - force: Configuration parameters to periodically overwrite ("force") predefined build parameters

            - on: Force functionality enabled if True
            - start_part: First part where functionality is enabled (disregarding the first few ignored parts)
            - n_parts: Number of parts for which the functionality is enabled
            - n_repeats: Number of consecutive layers for which inputs are forced. For [1,2], n_parts will be forced only once (periodically), while a further n_parts will be forced two times consecutively (periodically)
            - init_buffer: Initial number of layers for which parameters are not forced
            - upper_init: Upper bound is initialised to this.
            - upper_delta: Upper bound increases by this. For instance, for upper_init=105 and upper_delta=5, the upper bound sequence will be 105, 110, 115...
            - lower_init: Lower bound is initialised to this.
            - lower_delta: Lower bound is increased by this. For instance, for lower_init=65 and lower_delta=-5, the lower bound sequence will be 60, 55, 50...
            - fixed_speed: For the forced parameters, power will be adjusted but mark speed will be kept fixed to this value.

"""

from dotmap import DotMap
import numpy as np
import tensorflow as tf
import config_windows as cfg_global
from dmbrl.modeling.models import BNN
from dmbrl.modeling.layers import FC
from dmbrl.misc.DotmapUtils import get_required_argument

# Model parameters
# --------------------------------------------------------------------------
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
    """Modifies observations and model p
    redictions before being passed to cost function.

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
def obs_cost_fn(obs, target):
    """ Cost function (state-dependent) used in the optimisation problem.

    Should process both np.arrays and tf.Tensor inputs.

    Arguments:
        obs (np.array or tf.Tensor): Array of observations with shape (`n`, `dO`).
        target (float)

    Returns:
        float
    """
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

    cfg.ctrl_cfg.per = 1
    cfg.ctrl_cfg.prop_cfg.model_pretrained = False
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.load_model = False
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_dir = ''
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_name = 'learned_model'

    cfg.ctrl_cfg.prop_cfg.model_init_cfg.num_nets = 5
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["batch_size"] = 32
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["epochs"] = 5
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["hide_progress"] = False

    cfg.ctrl_cfg.change_target = False
    cfg.ctrl_cfg.force.on = False

    cfg.ctrl_cfg.prop_cfg.npart = 20

    cfg.ctrl_cfg.opt_cfg.mode = "CEM"
    cfg.ctrl_cfg.opt_cfg.plan_hor = 1

    cfg.ctrl_cfg.opt_cfg.cfg["popsize"] = 500
    cfg.ctrl_cfg.opt_cfg.cfg["max_iters"] = 10
    cfg.ctrl_cfg.opt_cfg.cfg["num_elites"] =int(500*0.05)

    cfg.ctrl_cfg.prop_cfg.mode = "TS1"

    return cfg
