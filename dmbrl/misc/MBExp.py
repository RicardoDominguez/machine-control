from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from time import time, localtime, strftime

import numpy as np
from scipy.io import savemat
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.Agent import Agent


class MBExperiment:
    def __init__(self, params):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """
        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        if params.sim_cfg.get("stochastic", False):
            self.agent = Agent(DotMap(
                env=self.env, noisy_actions=True,
                noise_stddev=get_required_argument(
                    params.sim_cfg,
                    "noise_std",
                    "Must provide noise standard deviation in the case of a stochastic environment."
                )
            ))
        else:
            readjustT = params.exp_cfg.get("readjustT", False)
            readjustFreq = params.exp_cfg.get("readjustFreq", None)
            self.agent = Agent(DotMap(env=self.env, noisy_actions=False,
                readjustT=readjustT, readjustFreq=readjustFreq))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.ninit_rollouts = params.exp_cfg.get("ninit_rollouts", 1)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")

        self.logdir = os.path.join(
            get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory."),
            strftime("%Y-%m-%d--%H:%M:%S", localtime())
        )
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)


    def run_experiment(self, run_name=None):
        """Perform experiment.
        """
        os.makedirs(self.logdir, exist_ok=True)

        traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []

        # Perform initial rollouts, then train policy
        samples = []
        rwards = np.zeros(self.ninit_rollouts + self.ntrain_iters)
        for i in range(self.ninit_rollouts):
            out_roll = self.agent.sample(self.task_hor, self.policy)
            samples.append(out_roll)
            rwards[i] = np.sum(out_roll["rewards"])
            print("R: %f"  % (rwards[i]))
            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])

        # Batch learning placed here
        if self.ninit_rollouts > 0:
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples]
            )

        # Training loop
        for i in range(self.ntrain_iters):
            print("####################################################################")
            print("Starting training iteration %d out of %d." % ((i + 1), self.ntrain_iters))

            iter_dir = os.path.join(self.logdir, "train_iter%d" % (i + 1))
            os.makedirs(iter_dir, exist_ok=True)

            samples = []
            for j in range(self.nrecord): # OUT
                print("Sample rollout %d out of %d." % (j, self.nrecord))
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy,
                        os.path.join(iter_dir, "rollout%d.mp4" % j)
                    )
                )
            if self.nrecord > 0:
                for item in filter(lambda f: f.endswith(".json"), os.listdir(iter_dir)):
                    os.remove(os.path.join(iter_dir, item))
            for j in range(max(self.neval, self.nrollouts_per_iter) - self.nrecord):
                print("Training rollout %d out of %d." % (j, max(self.neval, self.nrollouts_per_iter) - self.nrecord))
                out_roll = self.agent.sample(self.task_hor, self.policy)
                samples.append(out_roll)
                rwards[i+self.ninit_rollouts] = np.sum(out_roll["rewards"])
                print("R: %f" % (rwards[i+self.ninit_rollouts]))

            print("Rewards obtained:", [sample["reward_sum"] for sample in samples[:self.neval]])
            traj_obs.extend([sample["obs"] for sample in samples[:self.nrollouts_per_iter]])
            traj_acs.extend([sample["ac"] for sample in samples[:self.nrollouts_per_iter]])
            traj_rets.extend([sample["reward_sum"] for sample in samples[:self.neval]])
            traj_rews.extend([sample["rewards"] for sample in samples[:self.nrollouts_per_iter]])
            samples = samples[:self.nrollouts_per_iter]

            # self.policy.dump_logs(self.logdir, iter_dir)
            # savemat(
            #     os.path.join(self.logdir, "logs.mat"),
            #     {
            #         "observations": traj_obs,
            #         "actions": traj_acs,
            #         "returns": traj_rets,
            #         "rewards": traj_rews
            #     }
            # )
            # Delete iteration directory if not used
            if len(os.listdir(iter_dir)) == 0:
                os.rmdir(iter_dir)

            if i < self.ntrain_iters - 1:
                self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
                )
        import datetime
        now = datetime.datetime.now()
        suffix = "_%d_%d_%d_%d_%d.npy" % (now.month, now.day, now.hour, now.minute, now.second)
        np.save("p_rewards"+suffix, rwards)
        if run_name is None:
            prefix = ''
        else:
            prefix = run_name + '_'

        def saveArray(array, name):
            l = len(array)
            new_array = np.zeros([l]+list(array[0].shape))
            for i in range(l): new_array[i] = array[i]
            np.save(name, new_array)
            return new_array

        exp_obs = saveArray(traj_obs, prefix+'observations'+suffix)
        exp_acs = saveArray(traj_acs, prefix+'actions'+suffix)
        exp_rew = saveArray(traj_rews, prefix+'rewards'+suffix)

        return exp_obs, exp_acs, exp_rew # (N+1, H+1, nS/nA/1)

    def simple_rollouts(self, n_runs, save_name=None):
        """
        Just use agent.sample, no model learning
        """
        os.makedirs(self.logdir, exist_ok=True)

        rwards = np.zeros(n_runs)
        observations = np.zeros((n_runs, self.task_hor+1, self.agent.env.nS))
        actions = np.zeros((n_runs, self.task_hor, self.agent.env.nU))
        for i in range(n_runs):
            print("Run %d/%d" %(i+1, n_runs))
            out_roll = self.agent.sample(self.task_hor, self.policy)
            rwards[i] = np.sum(out_roll["rewards"])
            observations[i, :, :] = out_roll["obs"]
            actions[i, :, :] = out_roll["ac"]


        print("Mean reward: %f, stdev: %f" % (np.mean(rwards), np.std(rwards)))
        if save_name is not None:
            np.save(save_name+'_rew.npy', rwards)
            np.save(save_name+'_obs.npy', observations)
            np.save(save_name+'_ac.npy', actions)
        return rwards, observations, actions

    def batch_experiment(self, ninit_rollouts=10, n_steps=50, epochs=50,
                         hld_ratio=0.3, batch_size=32, savePlot=True, saveData=True):
        # Perform initial rollouts
        samples = [] # length ninit_rollouts
        for i in range(ninit_rollouts):
            print("Rollout %d/%d..." % (i+1, ninit_rollouts))
            samples.append(self.agent.sample(n_steps, self.policy))
            # Add observations, actions and rewards
            #   obs is array of ninit_rollouts with (T+1, nS)
            #   ac is arrat of ninit_rollouts with (T, nA)
            #   rewards is array of ninit_rollouts with (T, )

        # Configure model training
        self.policy.model_train_cfg["epochs"] = epochs
        self.policy.model_train_cfg["holdout_ratio"] = hld_ratio
        self.policy.model_train_cfg["batch_size"] = batch_size

        import time
        start = time.time()
        self.policy.train(
            [sample["obs"] for sample in samples],
            [sample["ac"] for sample in samples],
            [sample["rewards"] for sample in samples]
        )
        end = time.time()

        # Save model data
        if saveData:
            train_losses, val_losses = self.policy.model.return_train_losses()

            import datetime
            now = datetime.datetime.now()
            np.save("train_loss_%d_%d_%d_%d_%d" % (now.month, now.day, now.hour, now.minute, now.second), train_losses)
            np.save("val_loss_%d_%d_%d_%d_%d" % (now.month, now.day, now.hour, now.minute, now.second), val_losses)

            if savePlot:
                avg_train_loss = np.mean(train_losses, axis=1)
                avg_val_loss = np.mean(val_losses, axis=1)

                data_size = int(ninit_rollouts * n_steps * (1 - self.policy.model_train_cfg["holdout_ratio"]))
                batch_size = self.policy.model_train_cfg["batch_size"]
                time_elapsed = int(end - start)

                import matplotlib
                matplotlib.use('agg') # This allows to use it in cluster
                import matplotlib.pyplot as plt
                plt.plot(avg_train_loss, label='Training loss')
                plt.plot(avg_val_loss, label='Validation loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title("Data N: %d, batch size: %d, time: %d (s)" % (data_size, batch_size, time_elapsed))
                plt.legend()
                plt.savefig("d%d_b%d_%d_%d_%d_%d_%d.png" % (data_size, batch_size,
                                                        now.month, now.day, now.hour, now.minute, now.second))
