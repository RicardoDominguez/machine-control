import numpy as np
import matplotlib.pyplot as plt

s = np.load("sim_model/states.npy") # (H+1, N, 16)
a = np.load("sim_model/actions.npy") # (H+1, N, 2)

from sklearn.metrics import mean_squared_error

mean_squared_error(s[1:,0,:], np.ones((s.shape[0]-1, s.shape[-1]))*980)
mean_squared_error(s[1:,1,:], np.ones((s.shape[0]-1, s.shape[-1]))*980)
mean_squared_error(s[1:,2,:], np.ones((s.shape[0]-1, s.shape[-1]))*980)
mean_squared_error(s[1:,3,:], np.ones((s.shape[0]-1, s.shape[-1]))*980)

# # Below has to do with emisivity
# # -----------------------------------------------------------------------------
# s_ = s[1:, :, :].reshape(-1, 16).mean(-1)
# a = a.reshape(-1, 2)
# e = a[:,1] / a[:,0]
# plt.plot(e, s_, '*')
#
# plt.xlabel("emisivity")
# plt.ylabel("avg temperature layer")
# plt.savefig("emisivity.png")
#
# plt.plot(s[20:, 0, :].mean(-1))
#
# a.shape
# plt.plot(a[:, 0, 0])
# plt.plot(a[:, 0, 1])
#
# a[0,0,:]
#
# plt.plot(s[:,1,0])