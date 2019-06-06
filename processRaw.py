import numpy as np
from process.process_main import *

parts = [4,10]
layers = [0.03, 0.9]
path = '/media/ricardo/Ricardo/job_4_5ca61a9c1f00001d0021d515/sensors/2Pyrometer/pyrometer2/'


pats = np.arange(parts[0], parts[1]+0.1)
lays = np.arange(layers[0], layers[1]+0.01, 0.03)
states = np.zeros((len(pats), len(lays), 16))
for part in range(len(pats)):
    for lay in range(len(lays)):
        filename = path + str(int(pats[part])) + "/" + str(round(lays[lay],2)) + ".pcd"
        print(filename)
        data = loadData(filename)
        if lay == 0:
            square_limits = divideSingleSquare(data)
        data, cutoff = removeColdLines(data, returnMode=2)
        data, ratio, error = divideDataRectangleLimits(data, square_limits,
            returnMode=4, plot=True, saveName='img_%d_%f.png')
        states[part, lay, :] = pieceStateMeans(data, square_limits, 4)
