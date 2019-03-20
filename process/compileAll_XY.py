import os
import numpy as np
import matplotlib.pyplot as plt
from mainFile import *

state_data_dir = '/media/ricardo/DATA/data_files/run2_2/saves/'
action_data_txt_file = 'control_inputs.txt'
save_dir= '/media/ricardo/DATA/data_files/run2_2/'

min_part = 0
max_part = 120
nS = 16
nU = 2
part_one = False
verbose = False

# Pick relevant files, and sort
part_numbers = []
for filename in os.listdir(state_data_dir):
    if filename.endswith('_X.npy'):
        name = filename[:-6]
        if name.isdigit():
            numb = int(name)
            if min_part <= numb <= max_part:
                part_numbers.append(numb)
part_numbers = sorted(part_numbers)
if verbose: print("Identified part numbers:", part_numbers)

# Read all relevant files
actions = loadActionFile(action_data_txt_file, part_one)
X, Y, U = np.empty((0, nS)), np.empty((0, nS)), np.empty((0, nU))
emisitivy = []
mean_z = []
for n in part_numbers:
    str_name = str(n)
    x = np.load(state_data_dir + str_name + '_X.npy')
    y = np.load(state_data_dir + str_name + '_Y.npy')
    if verbose: print("Part "+str_name+':', x.shape[0], y.shape[0])
    u = np.tile(actions[str_name].reshape(1, -1), [x.shape[0], 1])
    X = np.concatenate((X, x))
    Y = np.concatenate((Y, y))
    U = np.concatenate((U, u))
    emisitivy.append(actions[str_name][1]/actions[str_name][0])
    mean_z.append(x.mean())

def plot_emisvity_vs_temp(emisitivy, mean_z):
    plt.plot(emisitivy, mean_z, '.')
    plt.xlabel('Power/speed')
    plt.ylabel('Avg temperature')
    plt.savefig(save_dir+'emisivity_vs_temp.png')

np.save(save_dir+'X.npy', X)
np.save(save_dir+'Y.npy', Y)
np.save(save_dir+'U.npy', U)