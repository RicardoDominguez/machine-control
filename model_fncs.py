import matplotlib.pyplot as plt
from dotmap import DotMap
import numpy as np
import tensorflow as tf
from dmbrl.modeling.models import BNN
from dmbrl.modeling.layers import FC
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

def loadPickle(name):
    file = open(name, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def savePickle(name, data):
    file = open(name, 'wb')
    pickle.dump(data, file)
    file.close()

class WrapScaler:
    def __init__(self, scaler):
        self.tempScaler = scaler()
        self.actionScaler = scaler()
        self.targetScaler = scaler()

    def fit(self, model_in, targets):
        self.tempScaler.fit(model_in[:,:-2].reshape(-1,1))
        self.actionScaler.fit(model_in[:,-2:])
        self.targetScaler.fit(targets.reshape(-1,1))

    def transformModelIn(self, model_in):
        temp_in = self.tempScaler.transform(
            model_in[:,:-2].reshape(-1,1)).reshape(-1, model_in.shape[1]-2)
        action_in = self.actionScaler.transform(model_in[:,-2:])
        return np.concatenate((temp_in, action_in), axis=1)

    def transformTarget(self, targets):
        return self.targetScaler.transform(targets.reshape(-1,1)).reshape(targets.shape)

    def inverse_transformModelIn(self, scaled_model_in):
        temp_in = self.tempScaler.inverse_transform(
            scaled_model_in[:,:-2].reshape(-1,1)).reshape(-1, scaled_model_in.shape[1]-2)
        action_in = self.actionScaler.inverse_transform(scaled_model_in[:,-2:])
        return np.concatenate((temp_in, action_in), axis=1)

    def inverse_transformTarget(self, targets):
        return self.targetScaler.inverse_transform(targets.reshape(-1,1)).reshape(targets.shape)

class ModelWrapper:
    def __init__(self, scaler):
        self.scaler = scaler

    def build_model(self, model_in, model_out, n_layers, n_neurons, l_rate, wd_in, wd_hid, wd_out, num_networks):
        sess = tf.Session()
        params = DotMap(name="model1", num_networks=num_networks, sess=sess)
        self.model = BNN(params)
        self.model.add(FC(n_neurons, input_dim=model_in, activation="swish", weight_decay=wd_in))
        for i in range(n_layers): self.model.add(FC(n_neurons, activation="swish", weight_decay=wd_hid))
        self.model.add(FC(model_out, weight_decay=wd_out))
        self.model.finalize(tf.train.AdamOptimizer, {"learning_rate": l_rate})

    def train(self, train_x, train_y, batch_size, epochs, rescale=True, hide_progress=False):
        if rescale: self.scaler.fit(train_x, train_y)
        x = self.scaler.transformModelIn(train_x)
        y = self.scaler.transformTarget(train_y)
        self.model.train(x, y, batch_size=batch_size, epochs=epochs, hide_progress=hide_progress)

    def trainCheck(self, train_x, train_y, test_x, test_y, batch_size, epochs, rescale=True, hide_progress=False):
        if rescale: self.scaler.fit(train_x, train_y)
        x = self.scaler.transformModelIn(train_x)
        y = self.scaler.transformTarget(train_y)
        x_t = self.scaler.transformModelIn(test_x)
        y_t = self.scaler.transformTarget(test_y)
        return self.model.trainCheck(x, y, x_t, y_t, batch_size=batch_size, epochs=epochs, hide_progress=hide_progress)

    def loss(self, test_x, test_y):
        x = self.scaler.transformModelIn(test_x)
        y = self.scaler.transformTarget(test_y)
        return self.model.get_mse_loss(x, y).mean()

    def predict(self, input, get_std=False, factored=False):
        input_scaled = self.scaler.transformModelIn(input)
        mean, var = self.model.predict(input_scaled, factored=factored)
        out = self.scaler.inverse_transformTarget(mean.reshape(-1, mean.shape[-1])).reshape(mean.shape)
        if get_std:
            upper = self.scaler.inverse_transformTarget(mean+(np.sqrt(var)).reshape(-1, var.shape[-1])).reshape(var.shape)
            return out + input[:, :-2], upper - out
        else:
            return out + input[:, :-2]

    def metrics(self, test_x, test_y):
        x = self.scaler.transformModelIn(test_x)
        y = self.scaler.transformTarget(test_y)
        return self.model.get_all_loss(x, y)

    def getLogPred(self, test_x, test_y):
        x = self.scaler.transformModelIn(test_x)
        y = self.scaler.transformTarget(test_y)
        return self.model.get_log_loss(x, y)

    def mse(self, test_x, test_y):
        x = self.scaler.transformModelIn(test_x)
        y = self.scaler.transformTarget(test_y)
        return self.model.get_mse_loss(x, y)

    def save(self, name, dir):
        self.model.name = name
        self.learning_rate = self.model.optimizer._lr
        self.model.save(savedir=dir)
        md = self.model
        self.model = 0
        self.name = name
        self.dir = dir
        savePickle(name, self)
        self.model = md

    def load(self):
        sess = tf.Session()
        params = DotMap(name=self.name, load_model=True, sess=sess, model_dir=self.dir)
        self.model = BNN(params)
        self.model.finalize(tf.train.AdamOptimizer, {"learning_rate": self.learning_rate})

def plotR2(model, test_x, test_y, state, probs=False, n_points=None, save_name=None):
    if probs:
        mean_scaled_, std_ = model.predict(test_x, get_std=True)
        mean_scaled, std = mean_scaled_[:, state], std_[:, state]
    else:
        mean_scaled = model.predict(test_x)[:, state]
    out_scaled = test_y[:, state]

    if n_points is not None and n_points < mean_scaled.shape[0]:
        np.random.seed(0)
        indx_ints = np.random.choice(np.arange(mean_scaled.shape[0]), n_points, replace=False)
        mean_scaled, out_scaled = mean_scaled[indx_ints], out_scaled[indx_ints]
        if probs: std = std[indx_ints]

    minx = min(out_scaled.min(), mean_scaled.min())
    maxx = max(out_scaled.max(), mean_scaled.max())

    plt.figure(figsize=(10,6))
    plt.plot(out_scaled, mean_scaled, '.', markersize=2)
    plt.plot([minx, maxx], [minx, maxx])

    if probs:
        # markers, caps, bars = plt.errorbar(out_scaled, mean_scaled, yerr=2*std,
        #     fmt='none', ecolor='lightskyblue', elinewidth=2, capsize=0) # 2stdev
        # [bar.set_alpha(0.5) for bar in bars]
        markers, caps, bars = plt.errorbar(out_scaled, mean_scaled, yerr=std,
            fmt='none', ecolor='lightskyblue', elinewidth=2, capsize=0); # 1stdev
    plt.xlabel("Actual T")
    plt.ylabel("Predicted T")
    if save_name is not None: plt.savefig(save_name, bbox_inches='tight', dpi=100)

def plotIndiv(model, test_x, test_y, state, n_points=None, lower=None, upper=None, probs=False, save_name=None):
    if probs:
        out_scaled_, std_ = model.predict(test_x, get_std=True)
        out_scaled, std = out_scaled_[:, state], std_[:, state]
    else:
        out_scaled = model.predict(test_x)[:, state]

    # Index within specified bounds
    if lower is None:
        indx_over = np.ones((out_scaled.shape[0],), dtype=bool)
    else:
        indx_over = out_scaled >= lower
    if upper is None:
        indx_under = np.ones((out_scaled.shape[0],), dtype=bool)
    else:
        indx_under = out_scaled <= upper
    indx_bounds = np.logical_and(indx_over, indx_under)
    indx_ints = np.argwhere(indx_bounds).reshape(-1)

    # Sample n_points if appropiate
    if n_points is not None and n_points < indx_ints.shape[0]:
        np.random.seed(0)
        indx_ints = np.random.choice(indx_ints, n_points, replace=False)

    # Sort ascending value of ground truth
    zs = np.argsort(test_y[indx_ints, state])
    sorted_indx = indx_ints[zs.reshape(-1)]

    plt.figure(figsize=(20,6))
    plt.plot(out_scaled[sorted_indx], '.', markersize=3)
    plt.plot(test_y[sorted_indx, state], zorder=10)
    if probs:
        markers, caps, bars = plt.errorbar(np.arange(out_scaled[sorted_indx].shape[0]),
            out_scaled[sorted_indx], yerr=2*std[sorted_indx], fmt='none',
            ecolor='lightskyblue', elinewidth=2, capsize=0) # 2stdev
        [bar.set_alpha(0.5) for bar in bars]
        markers, caps, bars = plt.errorbar(np.arange(out_scaled[sorted_indx].shape[0]),
            out_scaled[sorted_indx], yerr=std[sorted_indx], fmt='none',
            ecolor='lightskyblue', elinewidth=2, capsize=0); # 1stdev
    plt.grid()
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight', dpi=100)

def predMetrics(model, test_x, test_y, indiv=False):
    out, std = model.predict(test_x, get_std=True)
    mse = np.zeros((test_y.shape[-1],))
    r2 = np.zeros((test_y.shape[-1],))
    for s in range(test_y.shape[-1]):
        r2[s] = r2_score(test_y[:,s], out[:,s])
        mse[s] = mean_squared_error(test_y[:,s], out[:,s])
        if indiv: print(("S: %d, R2: %.4f, RMSE: %.4f std: %.4f")%(
        s, r2[s], mse[s], std[:,s].mean()))
    print(("Overall, R2: %.4f (%.4f), MSE: %.4f (%.4f), std: %.4f (%.4f)")%(r2.mean(), r2.std(),
        mse.mean(), mse.std(), std.mean(), std.std()))
    return r2, mse, std.mean(0)