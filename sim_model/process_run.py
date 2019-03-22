import numpy as np
import matplotlib.pyplot as plt

s = np.load("sim_model/states.npy") # (N+1, H, 16)
a = np.load("sim_model/actions.npy") # (N, H, 2)

plt.plot(s[0, :, :].mean(-1))