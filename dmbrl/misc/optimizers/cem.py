from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.stats as stats

from .optimizer import Optimizer


class CEMOptimizer(Optimizer):
    """A Tensorflow-compatible CEM optimizer.

    Arguments:
        sol_dim (int): The dimensionality of the problem space
        max_iters (int): The maximum number of iterations to perform during optimization
        popsize (int): The number of candidate solutions to be sampled at every iteration
        num_elites (int): The number of top solutions that will be used to obtain the distribution at the next iteration.
        constrains (array): [[np.array([min v, min q]), np.array([max v, max q])], [min q/v, max q/v], [min q/sqrt(v), max q/sqrt(v)]]
        tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None, in which case any functions passed in cannot be tf.Tensor-valued.
        epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is stopped.
        alpha (float): Controls how much of the previous mean and variance is used for the next iteration. next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
    """
    def __init__(self, sol_dim, max_iters, popsize, num_elites, constrains,
                tf_session=None, epsilon=0.001, alpha=0.25, max_resamples=10):
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites
        self.constrains = constrains
        self.lb, self.ub = constrains[0][0], constrains[0][1]
        self.epsilon, self.alpha = epsilon, alpha
        self.tf_sess = tf_session

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("CEMSolver") as scope:
                    self.init_mean = tf.placeholder(dtype=tf.float32, shape=[sol_dim], name="Init_mean")
                    self.init_var = tf.placeholder(dtype=tf.float32, shape=[sol_dim], name="Init_var")

        self.num_opt_iters, self.mean, self.var = None, None, None
        self.tf_compatible, self.cost_function = None, None
        self.max_resamples = max_resamples

    def changeSolDim(self, sol_dim):
        """Change the dimension of the CEM optimisation solution.

        Arguments:
            sol_dim (int): New dimension of the CEM optimisation solution.
        """
        self.sol_dim = sol_dim
        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("CEMSolver") as scope:
                    self.init_mean = tf.placeholder(dtype=tf.float32, shape=[sol_dim], name="Init_mean")
                    self.init_var = tf.placeholder(dtype=tf.float32, shape=[sol_dim], name="Init_var")

    def setup(self, cost_function, tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        if tf_compatible and self.tf_sess is None:
            raise RuntimeError("Cannot pass in a tf.Tensor-valued cost function without passing in a TensorFlow "
                               "session into the constructor")

        self.tf_compatible = tf_compatible

        if not tf_compatible:
            self.cost_function = cost_function
        else:
            def continue_optimization(t, mean, var, best_val, best_sol):
                # Loop until variance is low enough (converged) or number of timesteps achieved
                return tf.logical_and(tf.less(t, self.max_iters), tf.reduce_max(var) > self.epsilon)

            def iteration(t, mean, var, best_val, best_sol):
                """
                Arguments:
                    mean (H*nU,)
                    var (H*nU,)
                    best_val (1,)
                    best_sol (H*nU,)
                """
                # Adjust variance according to upper and lower bound of actions
                # mean - lb ~= 2 * sigma, var = sigma^2
                # Pick min among input and the two computed variances, otherwise
                # we sample significantly outside of upper or lower bound
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
                # constrained_var = tf.Print(constrained_var,[constrained_var], "CS: ", summarize=10)

                # Sample from normal distribution, with shape (popsize, H*nU)
                # Original line: samples = tf.truncated_normal([self.popsize, self.sol_dim], mean, tf.sqrt(constrained_var))
                def getConstrainViolation(candidates):
                    # Return true for candidtes which violate the constrains
                    under_bounds = np.any(candidates <= self.lb, 1)
                    over_bounds = np.any(candidates >= self.ub, 1)
                    bounds = np.logical_or(under_bounds, over_bounds)

                    v = candidates[:,0]
                    q = candidates[:,1]

                    if self.constrains[1] is not None:
                        q_over_v = q / v
                        c1 = np.logical_or(q_over_v <= self.constrains[1][0], q_over_v >= self.constrains[1][1])
                        bounds = np.logical_or(bounds, c1)

                    if self.constrains[2] is not None:
                        q_over_sqrt_v = q / np.sqrt(v)
                        c2 = np.logical_or(q_over_sqrt_v <= self.constrains[2][0], q_over_sqrt_v >= self.constrains[2][1])
                        bounds = np.logical_or(bounds, c2)

                    return bounds

                def sampleCandidates(mu, var):
                    sigma = np.sqrt(var)
                    samples = np.random.normal(mu, sigma, [self.popsize, self.sol_dim])
                    for n in range(self.max_resamples-1):
                        indx_violation = getConstrainViolation(samples)
                        n_violations = np.sum(indx_violation)
                        if n_violations == 0: return samples.astype(np.float32)
                        samples[indx_violation] = np.random.normal(mu, sigma, [n_violations, self.sol_dim])
                    indx_violation = getConstrainViolation(samples)
                    samples[indx_violation] = samples[np.random.choice(np.argwhere(np.logical_not(indx_violation)).reshape(-1),
                                                        indx_violation.sum())]
                    return samples.astype(np.float32)

                samples = tf.numpy_function(sampleCandidates, [mean, constrained_var], tf.float32)
                samples.set_shape(tf.TensorShape([self.popsize, self.sol_dim]))

                # samples = tf.Print(samples,[samples], "Samples: ", summarize=10)
                # samples = tf.Print(samples,[best_val], "best_val: ", summarize=10)
                # samples = tf.Print(samples,[best_sol], "best_sol: ", summarize=10)

                # Evaluate cost function at sampled points
                costs = cost_function(samples)

                # Pick elites (M sequences with lowest cost)
                values, indices = tf.nn.top_k(-costs, k=self.num_elites, sorted=True)
                # indices = tf.Print(indices,[values], "values: ", summarize=10)
                elites = tf.gather(samples, indices)
                # elites = tf.Print(elites,[elites], "elites: ", summarize=10)
                # elites = tf.Print(elites,[-values[0]], "found val: ", summarize=10)
                # elites = tf.Print(elites,[samples[indices[0]]], "found sol: ", summarize=10)



                # Compute mean and variance of elites
                new_mean = tf.reduce_mean(elites, axis=0)
                # new_mean = tf.Print(new_mean,[new_mean], "New_mean: ", summarize=10)

                new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)
                new_sq = 4*tf.sqrt(new_var)/new_mean
                # new_var = tf.Print(new_var,[new_sq], "new_var: ", summarize=10)

                # Update mean and variance (alpha controls how much of the
                # previous mean and variance is used for the next iteration)
                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                # Update best action sequence so far and corresponding value = -cost
                best_val, best_sol = tf.cond(
                    tf.less(-values[0], best_val),
                    lambda: (-values[0], samples[indices[0]]),
                    lambda: (best_val, best_sol)
                )
                return t + 1, mean, var, best_val, best_sol

            with self.tf_sess.graph.as_default():
                self.num_opt_iters, self.mean, self.var, self.best_val, self.best_sol = tf.while_loop(
                    cond=continue_optimization, body=iteration,
                    loop_vars=[0, self.init_mean, self.init_var, float("inf"), self.init_mean]
                )

    def reset(self):
        """Blank function for compatibility with optimisation class framework."""
        pass

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if self.tf_compatible:
            sol, solvar = self.tf_sess.run(
                [self.mean, self.var],
                feed_dict={self.init_mean: init_mean, self.init_var: init_var}
            )
            # print("Solvar", solvar)
        else:
            mean, var, t = init_mean, init_var, 0
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

            while (t < self.max_iters) and np.max(var) > self.epsilon:
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

                samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
                costs = self.cost_function(samples)
                elites = samples[np.argsort(costs)][:self.num_elites]

                new_mean = np.mean(elites, axis=0)
                new_var = np.var(elites, axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                t += 1
            sol, solvar = mean, var
        return sol
