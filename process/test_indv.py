raw_data_folder_name = '/media/ricardo/DATA/run2_1/'
run_name = '/media/ricardo/Ricardo/full1_exp2/'

from mainFile import *

data = loadData(raw_data_folder_name+'4/0.09.pcd')
square_limits = divide4squares(data)
divideDataRectangleLimits(data, square_limits, returnMode=0, plot=True)

leave_out = 0.95
nout = int(data.shape[0]*leave_out)
maxy = data[-nout:, 1].max()
plt.plot(data[-nout:, 0], data[-nout:, 1])
