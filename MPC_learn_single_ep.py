from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import pprint

import time
from dotmap import DotMap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.modeling.models import BNN
from dmbrl.controllers.MPC import MPC
from custom_env.machine_config import create_config
from custom_env.machine_model import MachineModelEnv
from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC

def obs_cost_fn(obs):
    target = 980
    k = 1000
    if isinstance(obs, np.ndarray):
        return -np.exp(-np.sum(np.square((obs-target)), axis=-1)/k)
    else:
        return -tf.exp(-tf.reduce_sum(tf.square((obs-target)), axis=-1)/k)
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

def setModelinit_param(cfg, model_in, model_out, n_layers, n_neurons, wd_in, wd_hid,
        wd_out, learning_rate, num_nets):
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_in = model_in
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_out = model_out
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.n_layers = n_layers
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.n_neurons = n_neurons
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.wd_in = wd_in
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.wd_hid = wd_hid
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.wd_out = wd_out
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.learning_rate = learning_rate
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_constructor = bnn_constructor
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.num_nets = num_nets
    return cfg

def setModeltrain_param(cfg, batch_size, epochs, hide_progress):
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["batch_size"] = batch_size
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["epochs"] = epochs
    cfg.ctrl_cfg.prop_cfg.model_train_cfg["hide_progress"] = hide_progress
    return cfg

def setMPC_param(cfg, plan_hor=None, n_particles=None, cem_popsize=None,
        cem_elites_pct=None, cem_max_iters=None, prop_mode=None, ign_var=None):
    if plan_hor is not None: cfg.ctrl_cfg.opt_cfg.plan_hor = plan_hor
    if n_particles is not None: cfg.ctrl_cfg.prop_cfg.npart = n_particles
    if cem_popsize is not None:
        cfg.ctrl_cfg.opt_cfg.cfg["popsize"] = cem_popsize
        if cem_elites_pct is not None:
            cfg.ctrl_cfg.opt_cfg.cfg["num_elites"] = int(cem_popsize*cem_elites_pct)
        else:
            cfg.ctrl_cfg.opt_cfg.cfg["num_elites"] = int(cem_popsize*0.05)
    if cem_max_iters is not None: cfg.ctrl_cfg.opt_cfg.cfg["max_iters"] = cem_max_iters
    if prop_mode is not None: cfg.ctrl_cfg.prop_cfg.mode = prop_mode
    if ign_var is not None: cfg.ctrl_cfg.prop_cfg.ign_var = ign_var
    return cfg

def setENV(cfg, model_dir, model_name, T=None, stochastic=True, randInit=True,
        x0=None, sig0=None):
    def ac_cost(x): return 0
    ENV = MachineModelEnv(model_dir, model_name, ac_cost, obs_cost_fn, stochastic, randInit)
    if x0 is not None: ENV.x0mu = x0
    if sig0 is not None: ENV.x0sig = sig0
    if T is not None: cfg.exp_cfg.sim_cfg.task_hor = T
    cfg.exp_cfg.sim_cfg.env = ENV
    cfg.ctrl_cfg.env = ENV
    return cfg

def runSingleEpExperiment(env, n_parts, horizon, train_freq, policy, run_name):
    print("Experiment", run_name, "started")

    # For logging purposes
    state_log = np.zeros((horizon+1, n_parts, 16))
    action_log = np.zeros((horizon, n_parts, 2))
    rewards_log = np.zeros((horizon, n_parts))
    predcost_log = np.zeros((horizon, n_parts))

    action = np.zeros((n_parts, 2))
    states = np.zeros((n_parts, 16))

    # Initialise state
    for part in range(n_parts): states[part, :] = env.reset()
    state_log[0, :, :]  = states

    for t in range(horizon):
        print("timestep %d/%d" % (t, horizon))
        # Retrain if appropiate
        if t != 0 and (t % train_freq) == 0:
            print("Training model...")
            obs_in = state_log[t-train_freq:t, :, :].reshape(-1, state_log.shape[-1])
            obs_out = state_log[t-train_freq+1:t+1, :, :].reshape(-1, state_log.shape[-1])
            acs = action_log[t-train_freq:t, :, :].reshape(-1, action_log.shape[-1])
            policy.trainTargs(obs_in, obs_out, acs)

        # Sample action
        for part in range(n_parts):
            if t >= train_freq: # Get predicted cost
                action[part, :], predcost_log[t,part] = policy.act(
                                                        states[part, :], t,
                                                        get_pred_cost=True)
            else:
                action[part, :] = policy.act(states[part, :], t, get_pred_cost=False)
                predcost_log[t,part] = 0
        action_log[t, :, :] = action

        # Transition to next state
        for part in range(n_parts):
            states[part, :] = env.model_transition_state(states[part, :], action[part, :])
        state_log[t+1, :, :] = states

        # Compute reward
        for part in range(n_parts): rewards_log[t, part] = -obs_cost_fn(states[part,:])

    # Save logged info
    np.save(run_name+"_states", state_log)
    np.save(run_name+"_actions", action_log)
    np.save(run_name+"_rewards", rewards_log)
    np.save(run_name+"_predcost", predcost_log)

    return rewards_log.sum(0)


def unfamiliarMPC(cfg, n_parts, horizon, train_freq, run_name):
    print("Executing run "+run_name)
    cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    exp = MBExperiment(cfg.exp_cfg)
    env = cfg.exp_cfg.sim_cfg.env
    policy = exp.policy
    runSingleEpExperiment(env, n_parts, horizon, train_freq, policy, run_name)

def checkParam(n_repeats, n_parts, run_name, plan_hor_in=None, cem_popsize_in=None,
        cem_elites_pct_in=None, n_particles_in=None, epochs_in=None, num_nets_in=None,
        prop_mode_in=None, deterministic_in=None, train_freq_in=None):
    """ n_repeats experiments with a n_episodes episodes """
    model_in, model_out = 18, 16
    num_nets, n_layers, n_neurons = 5, 3, 750
    learning_rate = 3.61e-4
    wd_in, wd_hid, wd_out = 8.213e-05, 1.188e-05, 1.004e-05

    batch_size, epochs, hide_progress = 32, 5, True

    plan_hor, cem_popsize, cem_max_iters, cem_elites_pct, n_particles = 3, 500, 10, 0.05, 20

    prop_mode = "TSinf"
    deterministic=False

    env_dir, env_name = 'custom_env/', 'run2_1'
    T = 167
    stochastic, randInit = True, True

    # train frequency
    train_freq = 5

    if plan_hor_in is not None: plan_hor = plan_hor_in
    if cem_popsize_in is not None: cem_popsize = cem_popsize_in
    if cem_elites_pct_in is not None: cem_elites_pct = cem_elites_pct_in
    if n_particles_in is not None: n_particles = n_particles_in
    if epochs_in is not None: epochs = epochs_in
    if num_nets_in is not None: num_nets = num_nets_in
    if prop_mode_in is not None: prop_mode = prop_mode_in
    if deterministic_in is not None: deterministic=deterministic_in
    if train_freq_in is not None: train_freq = train_freq_in


    rew = np.zeros((n_repeats, n_parts))

    for run in range(n_repeats):
        print("Experiment %d/%d" % (run+1, n_repeats))
        tf.reset_default_graph()
        cfg = create_config()
        cfg = setModelinit_param(cfg, model_in, model_out, n_layers, n_neurons, wd_in,
                wd_hid, wd_out, learning_rate, num_nets)
        cfg = setModeltrain_param(cfg, batch_size, epochs, hide_progress)
        cfg = setMPC_param(cfg, plan_hor, n_particles, cem_popsize, cem_elites_pct,
                cem_max_iters, prop_mode, ign_var=deterministic)
        cfg = setENV(cfg, env_dir, env_name, T=T, stochastic=stochastic, randInit=randInit)

        cfg.pprint()

        rew[run, :] = unfamiliarMPC(cfg, n_parts, T, train_freq, run_name+"_repeat"+str(run))

    np.save(run_name+"_rew.npy", rew)

if __name__ == '__main__':
    # checkParam(5, 10, 'ep2000t3')
    checkParam(5, 10, 'ep2000t1', plan_hor_in=1)
    # checkParam(5, 50, '2000t3TS1', cem_popsize_in=2000, prop_mode_in="TS1")
    # checkParam(5, 50, '2000t3DET', cem_popsize_in=2000, deterministic_in=True)



