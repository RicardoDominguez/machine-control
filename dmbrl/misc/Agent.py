from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from gym.monitoring import VideoRecorder
from dotmap import DotMap

import time


class Agent:
    """An general class for RL agents.
    """
    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        self.env = params.env
        self.readjustT = params.readjustT
        self.readjustFreq = params.readjustFreq

    def sample(self, horizon, policy, record_fname=None):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        
        times, rewards = [], []
        # 1. Reset env, get initial observation
        O, A, reward_sum, done = [self.env.reset()], [], 0, False

        # 2. Reset policy
        policy.reset()
        for t in range(horizon):
            print("Timestep %d of %d" % (t, horizon))
            if video_record:
                recorder.capture_frame()
            start = time.time()

            # Change planning horizon according to timestep to not overplan
            if self.readjustT:
                policy.changePlanHor(horizon-t, self.readjustFreq)

        # 3. Sample action from current observation, t
            A.append(policy.act(O[t], t))
            times.append(time.time() - start)

            if self.noise_stddev is None:
        # 4. Env step, given the action get new observation
                obs, reward, done, info = self.env.step(A[t])
            else:
                action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU])
                action = np.minimum(np.maximum(action, self.env.action_space.low), self.env.action_space.high)
                obs, reward, done, info = self.env.step(action)
        # 5. Record reward (and observation)
            O.append(obs)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

        if video_record:
            recorder.capture_frame()
            recorder.close()

        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }
