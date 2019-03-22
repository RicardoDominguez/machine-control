from dotmap import DotMap
import numpy as np
import tensorflow as tf
from gym import spaces
from dmbrl.modeling.models import BNN

class MachineModelEnv:
    def __init__(self, model_dir, model_name, ac_cost, state_cost, stochastic=True, randInit=True):
        """
        Inputs:
            model_dir (str), location of files from which to save model
            model_name (str), name of model (same as that of the saved files)
                              cannot be 'model' (errors raised in MPC)
            ac_cost (func), returns cost associated with a particular action
            state_cost (func), returns cost associated with a particular state
            stochastic (bool), if False only mean of model output is used
        """
        print("Loaded "+model_dir+model_name)
        # Data specific to environment
        self.nS, self.nU = 16, 2
        self.maxU, self.minU = np.array([1.8, 140.]), np.array([0.57, 75.])
        self.x0mu, self.x0sig = 870, 12

        # Load model from file
        model_cfg = DotMap(model_dir = model_dir, name=model_name, load_model=True)
        self.model = BNN(model_cfg)
        self.model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0})

        # For env.gym
        self.action_space = spaces.Box(low=self.minU, high=self.maxU)
        self.observation_space = spaces.Box(low=0, high=1, shape=((self.nS,)))

        # Cost functions
        self.cost_func_state = state_cost
        self.cost_func_action = ac_cost

        # Use model uncertainty
        self.stochastic = stochastic
        self.randInit = randInit

    def reset(self):
        """
        Resets state to the initial state.

        Returns:
        state, shape(nS,)
        """
        if self.randInit:
            self.state = np.random.normal(self.x0mu, self.x0sig, self.nS)
        else:
            self.state = np.ones(self.nS) * self.x0mu
        # print("Init", self.state)
        return self.state

    def step(self, action):
        """
        Transitions to the next timestep.

        Returns:
            state_, shape (nS)
            reward (float)
            False, no terminal states
            {}, to comply with gym.env standard
        """
        action_clip = np.clip(action, self.minU, self.maxU)
        # print("Action provided", action_clip)
        state_ = self.model_transition_state(self.state, action_clip)
        reward = -(self.cost_func_state(state_.reshape(1,-1))  \
                  + self.cost_func_action(action_clip))
        # print("Current state", self.state.mean(-1))
        # print("Next state", state_.mean(-1))
        self.state = state_
        return state_, reward, False, {}

    def model_transition_state(self, s, a):
        """
        Implements mapping s(k+1) <- f(s(k), a(k))

        Inputs
            s, shape(nS,)
            a, shape(nU,)

        Returns
            s_, shape(nS,)
        """
        input = np.concatenate((s.reshape(1,-1), a.reshape(1,-1)), axis=1)
        output, var = self.model.predict(input, factored=False)

        if self.stochastic:
            output += np.random.normal(0, 1, size=(1,var.shape[0])) * np.sqrt(var)

        return s + output.reshape(-1)
