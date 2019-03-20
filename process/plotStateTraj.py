# Plot state trajectories

import os
import numpy as np
import matplotlib.pyplot as plt
from mainFile import *

state_data_dir = '/media/ricardo/DATA/data_files/run2_1/saves/'
save_dir= '/media/ricardo/DATA/data_files/run2_1/images/trajectories/'

min_part = 91
max_part = 92
nS = 16

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
# for n in range(len(part_numbers)):
#     print("Part %d/%d" % (n+1, len(part_numbers)))
#     str_name = str(part_numbers[n])
#     states = np.load(state_data_dir + str_name + '.npy') # (4, N, nS)
#     valid = isValidState(states[0,:,:])
#     x = np.arange(states.shape[1])[valid]
#     mus = states.mean(0)[valid, :]
#     stds = states.std(0)[valid, :]
#     fig, axs = plt.subplots(nS,1,figsize=(20,3*nS))
#     for s in range(nS):
#         mu = mus[:, s]
#         std = stds[:, s]
#         axs[s].plot(x, mu, '-', color='gray')
#         axs[s].fill_between(x, mu - std, mu + std,
#                          color='gray', alpha=0.4)
#         axs[s].fill_between(x, mu - 2*std, mu + 2*std,
#                       color='gray', alpha=0.2)
#         axs[s].grid()
#     plt.savefig(save_dir+str_name+'_traj.png',
#         bbox_inches='tight', dpi=150)
#     plt.close()

s = 0
for n in range(len(part_numbers)):
    print("Part %d/%d" % (n+1, len(part_numbers)))
    str_name = str(part_numbers[n])
    states = np.load(state_data_dir + str_name + '.npy') # (4, N, nS)
    valid = isValidState(states[0,:,:])
    x = np.arange(states.shape[1])[valid]
    mus = states.mean(0)[valid, :]
    stds = states.std(0)[valid, :]
    plt.figure(figsize=(20,6))
    mu = mus[:, s]
    std = stds[:, s]
    plt.plot(x, mu, '-', color='r')
    plt.fill_between(x, mu - std, mu + std,
                     color='gray', alpha=0.75)
    plt.fill_between(x, mu - 2*std, mu + 2*std,
                  color='gray', alpha=0.4)
    plt.grid()
    plt.savefig(save_dir+str_name+'_traj.png',
        bbox_inches='tight', dpi=150)
    plt.close()