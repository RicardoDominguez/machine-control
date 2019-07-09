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

from model_fncs import *

save_name = 's1'
save_dir = ''

# Load data from files
X = np.load("Xexp2.npy")
Y = np.load("Yexp2.npy")
U = np.load("Aexp2.npy") # (pieces, nU)

# Targets
XU = np.concatenate((X, U), axis=1)
Yd = Y - X

# Divide into training and testing data
test_ratio = 0
num_test = int(X.shape[0] * test_ratio)
np.random.seed(0)
permutation = np.random.permutation(X.shape[0])

# Split train, test
train_x, test_x = XU[permutation[num_test:]], XU[permutation[:num_test]]
train_y, test_y = Yd[permutation[num_test:]], Yd[permutation[:num_test]]

scaler = WrapScaler(StandardScaler)

# Build model
# ------------------------------------------------------------------------------
model_in, model_out = train_x.shape[-1], train_y.shape[-1]
num_networks, n_layers, n_neurons = 1, 3, 750
l_rate = 0.00036104642915607132 #0.00075 [0.000127172772944074, 1.2264572049853182e-05, 8.390402425698618e-05, 1.1416633101206519e-05]
wd_in, wd_hid, wd_out = 8.2132302104955207e-05, 1.1882799523428946e-05, 1.0043651379126443e-05
# wd_in, wd_hid, wd_out = 0.00025, 0.0005, 0.00075

model = ModelWrapper(scaler)
model.build_model(model_in, model_out, n_layers, n_neurons, l_rate, wd_in, wd_hid, wd_out, num_networks)
model.train(train_x, train_y, 32, 45, rescale=True)
model.save('s45', save_dir)
model.train(train_x, train_y, 32, 30, rescale=True)
model.save('s75', save_dir)
model.train(train_x, train_y, 32, 50, rescale=True)
model.save('s125', save_dir)
