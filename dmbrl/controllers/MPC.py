from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import tensorflow as tf
import numpy as np
from scipy.io import savemat

from .Controller import Controller
from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.optimizers import RandomOptimizer, CEMOptimizer


class MPC(Controller):
    """
    Arguments:
        params (dotmap): Configuration parameters.
            .env (gym.env):
                Environment for which this controller will be used.
            .update_fns (list<func>):
                A list of functions that will be invoked (possibly with a
                tensorflow session) every time this controller is reset.
            .ac_ub (np.ndarray): (optional)
                An array of action upper bounds. Defaults to environment action
                upper bounds.
            .ac_lb (np.ndarray): (optional)
                An array of action lower bounds. Defaults to environment action
                lower bounds.
            .per (int): (optional)
                Determines how often the action sequence will be optimized.
                Defaults to 1 (reoptimizes at every call to act()).
            .prop_cfg
                .model_init_cfg (DotMap):
                    A DotMap of initialization parameters for the model.
                    .model_constructor (func):
                        A function which constructs an instance of this model, given model_init_cfg.
                .model_train_cfg (dict): (optional)
                    A DotMap of training parameters that will be passed into the
                    model every time is is trained. Defaults to an empty dict.
                .model_pretrained (bool): (optional)
                    If True, assumes that the model has been trained upon construction.
                .mode (str):
                    Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                    See https://arxiv.org/abs/1805.12114 for details.
                .npart (int):
                    Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                .ign_var (bool): (optional)
                    Determines whether or not variance output of the model will be ignored.
                    Defaults to False unless deterministic propagation is being used.
                .obs_preproc (func): (optional)
                    A function which modifies observations (in a 2D matrix) before
                    they are passed into the model. Defaults to lambda obs: obs.
                    Note: Must be able to process both NumPy and Tensorflow arrays.
                .obs_postproc (func): (optional)
                    A function which returns vectors calculated from the previous
                    observations and model predictions, which will then be passed
                    into the provided cost function on observations. Defaults to
                    lambda obs, model_out: model_out. Note: Must be able to process
                    both NumPy and Tensorflow arrays.
                .obs_postproc2 (func): (optional)
                    A function which takes the vectors returned by obs_postproc and (possibly)
                    modifies it into the predicted observations for the next time step.
                    Defaults to lambda obs: obs. Note: Must be able to process both NumPy and Tensorflow arrays.
                .targ_proc (func): (optional)
                    A function which takes current observations and next observations
                    and returns the array of targets (so that the model learns the mapping
                    obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                    Note: Only needs to process NumPy arrays.

            .opt_cfg
                .mode (str):
                    Internal optimizer that will be used. Choose between [CEM, Random].
                .cfg (DotMap):
                    A map of optimizer initializer parameters.
                .plan_hor (int):
                    The planning horizon that will be used in optimization.
                .obs_cost_fn (func):
                    A function which computes the cost of every observation in a 2D matrix.
                    Note: Must be able to process both NumPy and Tensorflow arrays.
                .ac_cost_fn (func):
                    A function which computes the cost of every action in a 2D matrix.
                .constrains (np.array):
                    An array with the optimisation constrains = [[lb, ub], [lc1, uc1], [lc2, uc2]]
                    so that if u = [v, q], lb <= u <= ub, lc1 <= q/v <= uc2, lc2 <= q/sqrt(v) <= uc2.
                    Overwrites ac_lb and ac_ub is constrains[0] is not None.

            .log_cfg
                .save_all_models (bool): (optional)
                    If True, saves models at every iteration.
                    Defaults to False (only most recent model is saved).
                    Warning: Can be very memory-intensive.
                .log_traj_preds (bool): (optional)
                    If True, saves the mean and variance of predicted particle trajectories.
                    Defaults to False.
                .log_particles (bool) (optional)
                    If True, saves all predicted particles trajectories.
                    Defaults to False. Note: Takes precedence over log_traj_preds.
                    Warning: Can be very memory-intensive
    """
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, params):
        super().__init__(params)
        self.dO, self.dU = params.dO, params.dU
        constrains = get_required_argument(params.opt_cfg, "constrains", "Must provide the optimisation constrains.")
        self.ac_lb = constrains[0][0]
        self.ac_ub = constrains[0][1]
        print("lb", self.ac_lb)
        print("ub", self.ac_ub)
        self.update_fns = params.get("update_fns", [])
        self.per = params.get("per", 1)

        self.model = get_required_argument(
            params.prop_cfg.model_init_cfg, "model_constructor", "Must provide a model constructor."
        )(params.prop_cfg.model_init_cfg)
        self.model_train_cfg = params.prop_cfg.get("model_train_cfg", {})
        self.prop_mode = get_required_argument(params.prop_cfg, "mode", "Must provide propagation method.")
        self.npart = get_required_argument(params.prop_cfg, "npart", "Must provide number of particles.")
        self.ign_var = params.prop_cfg.get("ign_var", False) or self.prop_mode == "E"

        self.obs_preproc = params.prop_cfg.get("obs_preproc", lambda obs: obs) # Defaults to do nothing
        self.obs_postproc = params.prop_cfg.get("obs_postproc", lambda obs, model_out: model_out) # Defaults to do nothing
        self.obs_postproc2 = params.prop_cfg.get("obs_postproc2", lambda next_obs: next_obs) # Defaults to do nothing
        self.targ_proc = params.prop_cfg.get("targ_proc", lambda obs, next_obs: next_obs) # Defaults to do nothing

        self.opt_mode = get_required_argument(params.opt_cfg, "mode", "Must provide optimization method.")
        self.plan_hor = get_required_argument(params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        self.og_plan_hor = self.plan_hor
        self.obs_cost_fn = get_required_argument(params.opt_cfg, "obs_cost_fn", "Must provide cost on observations.")
        self.target = get_required_argument(params.opt_cfg, "target", "Must provide cost on observations.")
        self.ac_cost_fn = get_required_argument(params.opt_cfg, "ac_cost_fn", "Must provide cost on actions.")

        self.save_all_models = params.log_cfg.get("save_all_models", False)
        self.log_traj_preds = params.log_cfg.get("log_traj_preds", False)
        self.log_particles = params.log_cfg.get("log_particles", False)

        self.useConstantAction = params.get("useConstantAction", False)
        if self.useConstantAction:
            self.constantActionToUse = get_required_argument(params, "constantActionToUse", "Must provide constant action")

        # Perform argument checks
        if self.prop_mode not in ["E", "DS", "MM", "TS1", "TSinf"]:
            raise ValueError("Invalid propagation method.")
        if self.prop_mode in ["TS1", "TSinf"] and self.npart % self.model.num_nets != 0:
            raise ValueError("Number of particles must be a multiple of the ensemble size.")
        if self.prop_mode == "E" and self.npart != 1:
            raise ValueError("Deterministic propagation methods only need one particle.")

        # Create action sequence optimizer
        opt_cfg = params.opt_cfg.get("cfg", {})
        self.optimizer = MPC.optimizers[params.opt_cfg.mode](
            sol_dim=self.plan_hor*self.dU,
            constrains=constrains,
            max_resamples=params.opt_cfg.get("max_resamples", 10),
            tf_session=None if not self.model.is_tf_model else self.model.sess,
            **opt_cfg
        )

        self.has_been_trained = params.prop_cfg.get("model_pretrained", False)

        # Init parameters for the optimizer
        # --------------------------------------------------------------------
        # Buffer of actions if not planning at each step
        self.ac_buf = np.array([]).reshape(0, self.dU) # (sol_dim,)
        # Previous solution is mean between min and max action
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor]) # (sol_dim,)
        # sigma ~= 4 * (max - min) for Gaussian, var = sigma^2
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor]) # (sol_dim,)
        # print("Init var", self.init_var)

        # Init parameters for model training
        # ---------------------------------------------------------------------
        # shape going in is (~, n actions + n input observations)
        self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        # shaope is (~, n targets)
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        )
        if self.model.is_tf_model:
            self.sy_cur_obs = tf.Variable(np.zeros(self.dO), dtype=tf.float32) # (dO,)
            self.ac_seq = tf.placeholder(shape=[1, self.plan_hor*self.dU], dtype=tf.float32) # (1, H*dU)
            self.pred_cost, self.pred_traj = self._compile_cost(self.ac_seq, get_pred_trajs=True)
            self.optimizer.setup(self._compile_cost, True)
            self.model.sess.run(tf.variables_initializer([self.sy_cur_obs]))
        else:
            raise NotImplementedError()

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
              ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            print("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            print("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            print("Trajectory prediction logging is disabled.")

    def changePlanHor(self, T, freq=None, change_over=False):
        """Dynamically changes the planning horizon of the MPC algorithm.

        Arguments:
            T (int): New planning horizon.
        """
        if not change_over and T >= self.plan_hor: return

        dim_diff = self.prev_sol.shape[0] - T*self.dU
        if freq is not None and dim_diff < freq*self.dU: return

        print("Plan horizon changed to %d" % (T))

        self.plan_hor = T
        self.optimizer.changeSolDim(self.plan_hor*self.dU)
        self.optimizer.lb = np.tile(self.ac_lb, [self.plan_hor])
        self.optimizer.ub = np.tile(self.ac_ub, [self.plan_hor])
        self.ac_seq = tf.placeholder(shape=[1, self.plan_hor*self.dU], dtype=tf.float32) # (1, H*dU)
        self.pred_cost, self.pred_traj = self._compile_cost(self.ac_seq, get_pred_trajs=True)
        self.optimizer.setup(self._compile_cost, True)

        if dim_diff > 0:
            self.prev_sol = self.prev_sol[:-dim_diff]
            self.init_var = self.init_var[:-dim_diff]

    def changeTargetCost(self, target):
        print("Target changed to", target)
        self.target = target
        self.pred_cost, self.pred_traj = self._compile_cost(self.ac_seq, get_pred_trajs=True)
        self.optimizer.setup(self._compile_cost, True)

    def train(self, obs_trajs, obs_prime_trajs, acs_trajs): # Just trains model!!
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: (N, nS)
            obs_prime_trajs: (N, nS) observations at next time step
            acs_trajs: (N, nU)

        Returns: None.
        """
        # Construct new training points and add to training set
        new_train_in = np.concatenate((self.obs_preproc(obs_trajs), acs_trajs), axis=-1)
        new_train_targs = self.targ_proc(obs_trajs, obs_prime_trajs)

        # Append with previous data
        self.train_in = np.concatenate((self.train_in, new_train_in), axis=0)
        self.train_targs = np.concatenate((self.train_targs, new_train_targs), axis=0)

        # Train the model
        self.model.train(self.train_in, self.train_targs, **self.model_train_cfg)
        self.has_been_trained = True

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        if self.plan_hor != self.og_plan_hor:
            self.changePlanHor(self.og_plan_hor, change_over=True)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor]) # (sol_dim,)
        self.optimizer.reset()
        if self.model.is_tf_model:
            for update_fn in self.update_fns:
                update_fn(self.model.sess)

    def act(self, obs, t, get_pred_cost=False):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        if self.useConstantAction:
            return self.constantActionToUse

        # Use random action if no model has been trained!
        if not self.has_been_trained:
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)

        # If not replanning at each step take next planned action!
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        if self.model.is_tf_model:
            self.sy_cur_obs.load(obs, self.model.sess)

        # Obtain a solution (actions which lead to highest result using CEM
        # optimizer, where cost is given by MPC._compile_cost)
        # Solution is a sequence of actions within planning horizon
        soln = self.optimizer.obtain_solution(self.prev_sol, self.init_var) # (H*nU,)
        # print("Soln", soln)
        # print("Prev sol", self.prev_sol)
        # Eliminate first 'per' actions, fill with zeros behind (H*nU,)
        # self.prev_sol = np.concatenate([np.copy(soln)[self.per*self.dU:], np.zeros(self.per*self.dU )])
        self.prev_sol = np.concatenate([np.copy(soln)[self.per*self.dU:], (self.ac_lb + self.ac_ub) / 2])

        # Load to buffer first 'per' actions (self.per, nU)
        self.ac_buf = soln[:self.per*self.dU].reshape(-1, self.dU) # Load to buffer first
        # print("Ac buf", self.ac_buf)
        if get_pred_cost and not (self.log_traj_preds or self.log_particles):
            if self.model.is_tf_model:
                pred_cost = self.model.sess.run(
                    self.pred_cost,
                    feed_dict={self.ac_seq: soln[None]}
                )[0] # Compute the predicted cost of the computed action sequence
            else:
                raise NotImplementedError()
            return self.act(obs, t), pred_cost

        # Loging (not interested) ---------------------------------------------
        elif self.log_traj_preds or self.log_particles:
            pred_cost, pred_traj = self.model.sess.run(
                [self.pred_cost, self.pred_traj],
                feed_dict={self.ac_seq: soln[None]}
            )
            # print("Solution:", soln[None])
            # print("Pred cost:", pred_cost)
            pred_cost, pred_traj = pred_cost[0], pred_traj[:, 0]
            if self.log_particles:
                self.pred_particles.append(pred_traj)
            else:
                self.pred_means.append(np.mean(pred_traj, axis=1))
                # print(pred_traj.shape)
                # print(self.pred_means[-1].shape)
                # print((pred_traj - self.pred_means[-1]).shape)
                # print(np.mean(pred_traj - self.pred_means[-1]))
                self.pred_vars.append(np.mean(np.square(pred_traj - self.pred_means[-1]), axis=1))
            if get_pred_cost:
                return self.act(obs, t), pred_cost
        # ---------------------------------------------------------------------

        return self.act(obs, t) # Calls back itself, but now action buffer has
                                # additional elements

    def dump_logs(self, primary_logdir, iter_logdir):
        """Saves logs to either a primary log directory or another iteration-specific directory.
        See __init__ documentation to see what is being logged.

        Arguments:
            primary_logdir (str): A directory path. This controller assumes that this directory
                does not change every iteration.
            iter_logdir (str): A directory path. This controller assumes that this directory
                changes every time dump_logs is called.

        Returns: None
        """
        self.model.save(iter_logdir if self.save_all_models else primary_logdir)
        if self.log_particles:
            savemat(os.path.join(iter_logdir, "predictions.mat"), {"predictions": self.pred_particles})
            self.pred_particles = []
        elif self.log_traj_preds:
            savemat(
                os.path.join(iter_logdir, "predictions.mat"),
                {"means": self.pred_means, "vars": self.pred_vars}
            )
            self.pred_means, self.pred_vars = [], []

    def _compile_cost(self, ac_seqs, get_pred_trajs=False):
        """
        Arguments:
            ac_seqs: sequence of actions with shape (N, H*dU)
            get_pred_trajs (bool): If True, returns the predicted trajectories.

        Trajectories are only predicted for logging purposes.
        """
        # time step
        t = tf.constant(0)

        # number of candidate solutions to be evaluated (for MPC optimization)
        nopt = tf.shape(ac_seqs)[0]

        # Cost initialized to 0, shape (N, particles)
        init_costs = tf.zeros([nopt, self.npart])

        # Sequence of actions, shape (H, N * particles, dU)
        ac_seqs = tf.reshape(ac_seqs, [-1, self.plan_hor, self.dU]) # (N, H, dU)
        ac_seqs = tf.reshape(tf.tile(
            tf.transpose(ac_seqs, [1, 0, 2])[:, :, None], # (H, N, dU) -> (H, N, 1, dU)
            [1, 1, self.npart, 1] # (H, N, particles, dU)
        ), [self.plan_hor, -1, self.dU]) # (H, N * particles, dU)

        # (N*particles, n observations)
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])

        def continue_prediction(t, *args): # Just check if t < horizon
            """Checks if `t` is within the planning horizon"""
            return tf.less(t, self.plan_hor)

        if get_pred_trajs:
            pred_trajs = init_obs[None] # (1, N*particles, observations)

            def iteration(t, total_cost, cur_obs, pred_trajs):
                """
                Arguments:
                    t (int)
                    total_cost (N, particles)
                    cur_obs (N * particles, dO)
                    pred_trajs (~, N * particles, dO): Dynamically changes size up to (H+1, N*part, dO)
                """
                # Select action, get next predicted state
                cur_acs = ac_seqs[t] # actions to take at timestep, (N * particles, dU)
                curr_obs_mean = tf.reduce_mean(cur_obs, axis=-1)
                next_obs = self._predict_next_obs(cur_obs, cur_acs) # predict next observation (N*particles, dO)
                # next_obs = tf.Print(next_obs,[curr_obs_mean], "cur_obs: ", summarize=10)
                # next_obs = tf.Print(next_obs,[cur_acs], "cur_acs: ", summarize=10)
                next_obs_mean = tf.reduce_mean(next_obs, axis=-1)
                # next_obs = tf.Print(next_obs,[next_obs_mean], "next_obs: ", summarize=10)
                # Compute cost of states (both observation and action)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs, self.target) + self.ac_cost_fn(cur_acs), [-1, self.npart] # (N, particles)
                )

                # Postporcess again
                next_obs = self.obs_postproc2(next_obs) #  (N*particles, dO)

                # Contatenate to predicted trajectories
                pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)

                return t + 1, total_cost + delta_cost, next_obs, pred_trajs

            _, costs, _, pred_trajs = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs, pred_trajs],
                shape_invariants=[
                    t.get_shape(), init_costs.get_shape(), init_obs.get_shape(), tf.TensorShape([None, None, self.dO])
                ]
            )

            # Average out particles, leading to (N,)
            # Replaces NaN costs with very high cost
            costs = tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)

            # (H+1, N, particles, dO)
            pred_trajs = tf.reshape(pred_trajs, [self.plan_hor + 1, -1, self.npart, self.dO])

            return costs, pred_trajs
        else:
            def iteration(t, total_cost, cur_obs):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs, self.target) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                return t + 1, total_cost + delta_cost, self.obs_postproc2(next_obs)

            _, costs, _ = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs]
            )

            # Replace nan costs with very high cost
            return tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)

    def _predict_next_obs(self, obs, acs):
        """
        Arguments:
            obs (N*particles, nU)
            acs (N*particles, nU)
        """
        # Preprocess observation
        proc_obs = self.obs_preproc(obs) # (N*particles, nOP)

        if self.model.is_tf_model:
            # TS Optimization: Expand so that particles are only passed through one of the networks.
            if self.prop_mode == "TS1":
                proc_obs = tf.reshape(proc_obs, [-1, self.npart, proc_obs.get_shape()[-1]]) # (N, particles, nOP)
                # Random indexes with shape (N, particles)
                sort_idxs = tf.nn.top_k(
                    tf.random_uniform([tf.shape(proc_obs)[0], self.npart]),
                    k=self.npart
                ).indices
                # Count up from 0 to N-1 with shape (N, particles, 1)
                tmp = tf.tile(tf.range(tf.shape(proc_obs)[0])[:, None], # (N, 1)
                              [1, self.npart])[:, :, None] # (N, particles, 1)
                # [count_up; random_idx] (N, particles, 2)
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)

                # I imagine this keeps N unsorted, while randomizes the particles
                proc_obs = tf.gather_nd(proc_obs, idxs) # (N, particles, nOP)
                proc_obs = tf.reshape(proc_obs, [-1, proc_obs.get_shape()[-1]]) # (N*particles, nOP)
            if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
                # Returns:  proc_obs (nets, N * particles / nets, nOP)
                #           acs (nets, N * particles / nets, nU)
                proc_obs, acs = self._expand_to_ts_format(proc_obs), self._expand_to_ts_format(acs)

            # Model inputs by concatenating observations and actions...
            inputs = tf.concat([proc_obs, acs], axis=-1) # (nets, N * particles / nets, nOP+nU)

            # Output Gaussian distribution
            mean, var = self.model.create_prediction_tensors(inputs) # (nets, N * particles / nets, nOprediction)

            # Prediction from distribution...
            if self.model.is_probabilistic and not self.ign_var: # Variance
                # Sample with variance
                # stddev=1) * sqrt(var) used because tf.random_normal only accepts float stddev
                # (nets, N * particles / nets, nOprediction)
                predictions = mean + tf.random_normal(shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var)

                # Disregard moment matching...
                # --------------------------------------
                if self.prop_mode == "MM":
                    model_out_dim = predictions.get_shape()[-1].value

                    predictions = tf.reshape(predictions, [-1, self.npart, model_out_dim])
                    prediction_mean = tf.reduce_mean(predictions, axis=1, keep_dims=True)
                    prediction_var = tf.reduce_mean(tf.square(predictions - prediction_mean), axis=1, keep_dims=True)
                    z = tf.random_normal(shape=tf.shape(predictions), mean=0, stddev=1)
                    samples = prediction_mean + z * tf.sqrt(prediction_var)
                    predictions = tf.reshape(samples, [-1, model_out_dim])
                # --------------------------------------
            else: # Ignores variance, no propagation
                predictions = mean

            # TS Optimization: Remove additional dimension
            if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
                predictions = self._flatten_to_matrix(predictions) # (N * particles, nOprediction)
            if self.prop_mode == "TS1":
                # (N, particles, nOprediction)
                predictions = tf.reshape(predictions, [-1, self.npart, predictions.get_shape()[-1]])
                # Sort back
                sort_idxs = tf.nn.top_k(
                    -sort_idxs,
                    k=self.npart
                ).indices
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                predictions = tf.gather_nd(predictions, idxs)
                # (N * particles, nOprediction)
                predictions = tf.reshape(predictions, [-1, predictions.get_shape()[-1]])

            return self.obs_postproc(obs, predictions) # Post process
        else:
            raise NotImplementedError()

    def _expand_to_ts_format(self, mat): # Expand to trajectory sampling format
        """
        Arguments:
            mat (N*parts, M)

        M can be either the dimension of preprocessed observation or the dimension
        of the action space.
        """
        dim = mat.get_shape()[-1] # M
        return tf.reshape(
            tf.transpose( # (nets, N, particles / num nets, M)
                # (N, nets, particles / num nets, M)
                tf.reshape(mat, [-1, self.model.num_nets, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [self.model.num_nets, -1, dim] # (nets, N * particles / num nets, M)
        )

    def _flatten_to_matrix(self, ts_fmt_arr):
        """
        Arguments:
            mat (nets, N * particles / nets, nOprediction)

        Returns:
            (N * particles, nOprediction)
        """
        dim = ts_fmt_arr.get_shape()[-1] # nOprediction
        return tf.reshape(
            tf.transpose( # (N, nets, particles / nets, nOprediction)
                # (nets, N, particles / nets, nOprediction)
                tf.reshape(ts_fmt_arr, [self.model.num_nets, -1, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [-1, dim] # (N * particles, nOprediction)
        )
