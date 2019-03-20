import numpy as np
import os
from mainFile import *

state_data_dir = 'full1/individual_state_data/'
action_data_txt_file = 'control_inputs.txt'
save_dir= 'full1/'

min_part = 7
max_part = 52
nS = 16
nU = 2
part_one = False

# Pick relevant files, and sort
part_numbers = []
for filename in os.listdir(state_data_dir):
    if filename.endswith('.npy'):
        name = filename[:-4]
        if name.isdigit():
            numb = int(name)
            if min_part <= numb <= max_part:
                part_numbers.append(numb)
part_numbers = sorted(part_numbers)

# Read all relevant files
actions = loadActionFile(action_data_txt_file, part_one)
X, Y, A = np.empty((0, nS)), np.empty((0, nS)), np.empty((0, nU))
for n in part_numbers:
    str_name = str(n)
    file_name = state_data_dir + str_name + '.npy'
    s = np.load(file_name)
    x = s[:, 0:-1, :].reshape(-1, nS)
    print(X.shape[0])
    y = s[:, 1:, :].reshape(-1, nS)
    a = np.tile(actions[str_name].reshape(1, -1), [x.shape[0], 1])
    X = np.concatenate((X, x))
    Y = np.concatenate((Y, y))
    A = np.concatenate((A, a))

np.save(save_dir+'X.npy', X)
np.save(save_dir+'Y.npy', Y)
np.save(save_dir+'A.npy', A)