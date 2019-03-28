import numpy as np
import matplotlib.pyplot as plt

actions = np.array([[1, 75],
                   [1, 140],
                   [0.57, 110],
                   [1.8, 110]]) # (n, 2)

states = np.load("states.npy") # (T, N, 16)

emisivity = actions[:,1] / actions[:,0]
state_mean_acs = states.mean(0).mean(-1)

for i in range(state_mean_acs.shape[0]):
    print("S: %.3f, P: %d, T: %.2f" % (actions[i,0], actions[i,1], state_mean_acs[i]))

plt.plot(emisivity, state_mean_acs)
plt.show()

plt.plot(state_mean_acs)
plt.show()
