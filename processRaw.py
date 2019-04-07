import numpy as np
from process.process_main import *

def saveLayers(parts, layers, path, save_name, img_dir):
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
            data, ratio, error = divideDataRectangleLimits(data, square_limits, returnMode=4)
            states[part, lay, :] = pieceStateMeans(data, square_limits, 4)
    print('Done')
    np.save(save_name, states)

if __name__ == '__main__':
    layers = [0.03, 4.95]
    path = '/media/ricardo/Ricardo/realexp1/job_3_5ca74a5a3d00001c001ad653/sensors/2Pyrometer/pyrometer2/'
    img_dir = '/media/ricardo/Ricardo/realexp1/square_saves/'
    save_dir = '/media/ricardo/Ricardo/realexp1/states/'

    # Pre trained cl - [4, 13]
    # Learned cl - [14, 23]
    # Open loop Alistair - [24, 33]
    # parts = [24,33]
    # name = 'alis_ol'
    # saveLayers(parts, layers, path, save_dir+name+'.npy', img_dir+name+'/')
    # Open loop Mine - [34, 43]
    parts = [34,43]
    name = 'mine_ol'
    saveLayers(parts, layers, path, save_dir+name+'.npy', img_dir+name+'/')
    # Random - [44, 53]
    parts = [44,53]
    name = 'rand'
    saveLayers(parts, layers, path, save_dir+name+'.npy', img_dir+name+'/')
