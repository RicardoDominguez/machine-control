"""
FUNCTIONS IMPLEMENTED:
    - loadData
    - purgeData
    - removeColdLines
    - divide4squares - provides square_limits when parts are composed of 4 squares
    - divideDataRectangleLimits - divides data according to square_limits
    - pieceStateMeans - coarse data into several discrete features
    - divideDataXY - provides X, Y given sequence of states and valid info
    - plotTemperaturesState - single square
    - plotAllStatesFile - all plots for a part within same figure
    - getInvalidState
    - isValidState
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
import time

def loadData(file_name, timeit=False, verbose=False):
    """ Load file, with structure [X, Y, mV, mV] """
    if timeit:
        print("loadData() function.")
        s = time.time()

    if verbose: print('Reading file, this may take a bit...')
    data_raw = np.loadtxt(file_name)
    if verbose: print('Data loaded')

    # Remove redundant mV column
    data = np.delete(data_raw, 3, 1)

    if timeit: print("Time elapsed: % d" % (time.time()-s))
    return data

def loadActionFile(file_name, run_one):
    "returns dictionary where actions['name'] = np.arr[speed, power]"
    def processFunctionRun1(input):
        """
        in = [label_in, speed, power] (floats)
        out = label_out (str), np.arr[speed, power]
        """
        label_in = input[0]
        if label_in % 2 == 0:
            return False, 0, 0
        n = 3*(label_in+1)/2 - 2
        label_out = str(int(n))
        data_out = np.array(input[1:])
        return True, label_out, data_out

    def processFunctionRun2(input):
        """
        in = [label_in, speed, power] (floats)
        out = label_out (str), np.arr[speed, power]
        """
        label_in = input[0]
        if label_in % 2 != 0:
            return False, 0, 0
        n = 3*(label_in)/2 - 2
        label_out = str(int(n))
        data_out = np.array(input[1:])
        return True, label_out, data_out

    a = np.loadtxt(file_name)

    actions = {}
    for i in range(a.shape[0]):
        if run_one:
            valid, label, data = processFunctionRun1(a[i,:])
        else:
            valid, label, data = processFunctionRun2(a[i,:])
        if valid: actions[label] = data

    return actions

def purgeData(percent, data):
    """ Purge "1 - percent" (0 < % < 1) of the data """
    assert(percent <= 1)
    assert(percent > 0)
    n_del = int((1-percent) * data.shape[0])
    random_index = np.random.choice(data.shape[0], n_del, replace=False)
    return np.delete(data, random_index, 0)

def removeColdLines(data, returnMode = 0, timeit=False, verbose=False,
        plot=False, saveName=None, saveDir=None):
    """
    The aim of this function is to find the peak in the data around 800mV and
    remove it. This is done by deleting any data below a value, which we have
    set as the peak + its width at half height

    Return modes:
        - 0: returns data above cutoff value
        - 1: returns cutoff value
        - 2: returns both data and cutoff value
    """
    if timeit:
        print("removeColdLines() function.")
        s = time.time()

    Z = data[:,2]

    #Create a histogram(eque) array representing the data
    # [b,xi]=ksdensity(Z);
    xi = np.linspace(Z.min(), Z.max(), 100) # MATLAB uses 100 points by default
    dxi = xi[1]-xi[0]
    kde = stats.gaussian_kde(Z)
    b = kde(xi)

    # Find the peaks in this data, sort the prominence of these peaks
    # [pks,loc,width,prom]=findpeaks(b,xi,'Annotate','extents','WidthReference','halfheight');
    peaks = signal.find_peaks(b, prominence=1e-4, width=0, rel_height=0.5)
    if verbose: print(peaks)

    if peaks[0].size > 0:
        mean = xi[peaks[0][0]]
        width = dxi*peaks[1]["widths"][0]
        cutoff = mean + width
        if verbose: print("Removed all below %d" % (cutoff))

    else:
        if verbose: print("No peaks were found")
        mean, width, cutoff = 0, 0, 850

    if timeit: print("Time elapsed: % d" % (time.time()-s))

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(xi, b)
        if mean != 0: plt.plot([cutoff, cutoff], [0, max(b)])
        plt.title(("%s, mean: %.2f, width: %.2f, cutoff: %.2f") % (saveName, mean, width, cutoff))
        if saveName is not None:
            if saveDir is None: saveDir = ''
            plt.savefig(saveDir+saveName+".png", bbox_inches='tight', dpi=100)
            plt.close()

    # Return...
    if returnMode == 0:
        return data[Z >= cutoff]
    if returnMode == 1:
        return cutoff
    if returnMode == 2:
        return data[Z >= cutoff], cutoff

def divide4squares(data):
    """ Divides the input data into 4 squares with known dimensions.

    Inputs:
        data - shape (N, 3)

    Outputs:
        square_limits - list of length 4, where
            square_limits[i] = [xmin, xmax, ymin, ymax]

    1. Cut of first/last X% of data, to remove sensor displacement line
    2. Find min/max of x and y
    3. Estimate squares using preknown proportions

        dx
    ---------    ---------
    --- 0 ---    --- 1 ---  dy
    ---------    ---------

    ---------    ---------
    --- 2 ---    --- 3 ---
    ---------    ---------

    Top of '0' and '1' not perfectly aligned, '0' higher by top_y_diff units
    Sides of '0' and '2' not perfectly aligned, '2' further to the left by left_x_diff
    """
    def bigFlatRegion(arr):
        largePeak = 10 # Flat regions can only begin with a big peak
        flatsize = [] # Size of flat
        flatstart = [] # Start of flat

        if np.abs(arr[0]) > 0:
            peak = True
            bigPeak = True
        else:
            peak = False
            flatArea = True
            starti = 0

        for i in range(1, arr.shape[0]):
            val = np.abs(arr[i])

            if peak:
                if val > largePeak:
                    bigPeak = True
                elif val == 0:
                    peak = False
                    flatArea = bigPeak
                    starti = i
            else:
                if val > 0:
                    peak = True
                    bigPeak = val > largePeak
                    if flatArea:
                        flatstart.append(starti)
                        flatsize.append(i-starti-1)
        if flatArea:
            flatstart.append(starti)
            flatsize.append(arr.shape[0]-starti-1)

        return flatstart, flatsize

    # Known proportions (measured with MATLAB, vary slightly among parts)
    dx = 384 # piece length
    dy = 414 # piece breadth
    top_y_diff = 5
    left_x_diff = 0

    # Remove first and last X% of data
    leave_out = 0.95
    nout = int(data.shape[0]*leave_out)
    maxy = data[-nout:, 1].max()

    # Find minx, maxx
    leave_out = 0.04
    nout = int(data.shape[0]*leave_out)
    data_ = data[nout:-nout, 0]
    minx, maxx = data_.min(), data_.max()

    # Find miny
    # --------------------------------------------------------------------------
    explore_max, N = 0.1, 100
    nout = int(data.shape[0]*explore_max)
    dn = int(nout/N)
    min_window = []
    inner_min = data[nout:, 1].min()
    for i in range(N): min_window.append(min(inner_min, data[i*dn:nout, 1].min()))
    starti, ni = bigFlatRegion(np.diff(min_window))
    miny = min_window[starti[-1]+1]

    # Apply proportions
    square_limits = np.array([[minx+left_x_diff, minx+dx+left_x_diff, maxy-dy, maxy],
                            [maxx-dx, maxx, maxy-dy-top_y_diff, maxy-top_y_diff],
                            [minx, minx+dx, miny-5, miny+dy],
                            [maxx-dx, maxx, miny, miny+dy]])

    return square_limits

def divideSingleSquare(data):
    """ Divides the input data into a single square with known dimension.

    Inputs: data - shape (N, 3)

    Outputs: square_limits - shape(4,) as [xmin, xmax, ymin, ymax]
    """
    # Remove first and last X% of data
    leave_out = 0.15
    nout = int(data.shape[0]*leave_out)
    xmin = data[nout:-nout, 0].min()
    xmax = data[nout:-nout, 0].max()
    ymin = data[nout:-nout, 1].min()
    ymax = data[nout:-nout, 1].max()
    return np.array([xmin, xmax, ymin, ymax])

def divideDataRectangleLimits(data, square_limits, returnMode=1,  plot=False,
        saveName=None, saveDir=None):
    """ Returns a list with the data for each of the pieces

    data : nparray (N, 3)
    square_limits: nparray (4,) [min X, max X, min Y, max Y]
    plot: bool
    -------
    return: nparray (M, 3)

    Output mode:
        0 - no return
        1 - only data within limits
        2 - data within limits, data outside of limits
        3 - data within limits, ratio of data outside limits / total data
        4 - data, ratio, error
    """
    # print('Dividing data according to min/max...')
    data_pieces = [] # List of nparrays with data for each piece
    data_trim = data[:, :]
    square_limits = square_limits.reshape(1,-1)
    n_pieces = square_limits.shape[0]

    valid = True
    for piece in range(n_pieces):
        # Above X min
        ok_arg = np.argwhere(data_trim[:, 0] >= square_limits[piece, 0])
        # Under X max
        indx = data_trim[ok_arg, 0] <= square_limits[piece, 1]
        ok_arg = ok_arg[indx]
        # Above Y min
        indx = data_trim[ok_arg, 1] >= square_limits[piece, 2]
        ok_arg = ok_arg[indx]
        # Under Y min
        indx = data_trim[ok_arg, 1] <= square_limits[piece, 3]
        ok_arg = ok_arg[indx]
        # Append and delete
        if not ok_arg.any(): valid = False
        data_pieces.append(data_trim[ok_arg, :])
        data_trim = np.delete(data_trim, ok_arg, axis=0)

    delete_ratio = data_trim.shape[0] / (data_trim.shape[0] + data.shape[0])

    if plot:
        def plotSquare(a, i):
            """ a is (4,), i is odd for rightmost"""
            # x = [a[0], a[0], a[1], a[1], a[0]]
            # y = [a[2], a[3], a[3], a[2], a[2]]
            x = [a[i%2], a[i%2]]
            y = [a[2], a[3]]
            plt.plot(x, y, 'g')
        plt.figure(figsize=(10, 6))
        for piece in range(n_pieces): # Data kept
            plt.scatter(data_pieces[piece][:, 0], data_pieces[piece][:, 1], 1, 'lightskyblue')
        plt.scatter(data_trim[:,0], data_trim[:, 1], 1, 'r') # Removed data
        for piece in range(n_pieces):
            plotSquare(square_limits[piece, :], piece)
        minx, maxx = square_limits[:,0].min(), square_limits[:,1].max()
        miny, maxy = square_limits[:,2].min(), square_limits[:,3].max()
        dx, dy = maxx - minx, maxy - miny
        plt.xlim(minx-0.2*dx, maxx+0.2*dx)
        plt.ylim(miny-0.2*dy, maxy+0.2*dy)
        plt.title(("%s, delete ratio: %.4f") % (saveName, delete_ratio))
        if saveName is not None:
            if saveDir is None: saveDir = ''
            plt.savefig(saveDir+saveName+".png", bbox_inches='tight', dpi=100)
            plt.close()
        else: plt.show()

    if returnMode == 1:
        return data_pieces[0]
    if returnMode == 2:
        return data_pieces[0], data_trim
    if returnMode == 3:
        return data_pieces[0], delete_ratio
    if returnMode == 4:
        return data_pieces[0], delete_ratio, not valid

def pieceStateMeans(piece_data, piece_limits=None, n_splits=None):
    """
    Convert the raw data of a single piece into its state by dividing the piece
    into squares and taking the mean of the temperature within that square.

    The number of states will be n_splits * n_splits

    If a particular square has no data points, then average contigous states.

    Inputs
    ------
        piece_data: nparray, shape(N, 3)
        piece_limits: nparray, shape(4,) [xmin, xmax, ymin, ymax]
        n_splits: int

    Output
    ------
        states: nparray, shape(n_splits*n_splits,)
    """
    X, Y, Z = piece_data[:,0], piece_data[:,1], piece_data[:,2]

    if piece_limits is None:
        piece_limits = np.zeros((4,))
        piece_limits[0], piece_limits[1] = np.min(X), np.max(X)
        piece_limits[2], piece_limits[3] = np.min(Y), np.max(Y)

    # Ranges of X and Y boxes
    X_splits = np.linspace(piece_limits[0], piece_limits[1], n_splits+1)
    Y_splits = np.linspace(piece_limits[2], piece_limits[3], n_splits+1)

    # Indices of data for every one of the x bands
    X_indxs = [0] * n_splits
    overX = np.ones((X.shape[0],), dtype=bool)
    for x_s in range(n_splits):
        underX = X <= X_splits[x_s+1]
        X_indxs[x_s] = np.logical_and(overX, underX)
        overX = np.logical_not(underX)

    # Indices of data for every one of the y bands
    Y_indxs = [0] * n_splits
    overY = np.ones((Y.shape[0],), dtype=bool)
    for y_s in range(n_splits):
        underY = Y <= Y_splits[y_s+1]
        Y_indxs[y_s] = np.logical_and(overY, underY)
        overY = np.logical_not(underY)

    # Indices of data within the square (x, y) obtained with AND
    XY_limits = np.zeros((n_splits * n_splits, X.shape[0]), dtype=bool)
    for x in range(n_splits):
        for y in range(n_splits):
            XY_limits[x + n_splits*y, :] = np.logical_and(X_indxs[x], Y_indxs[y])

    shuffle = [(n_splits-1-i)*n_splits+j for i in range(n_splits) for j in range(n_splits)]
    XY_limits = XY_limits[shuffle, :]

    # Mean Z value for each square
    Z_means = np.zeros((n_splits*n_splits,))
    for x in range(n_splits*n_splits):
        indx = XY_limits[x, :]
        if indx.any():
            Z_means[x] = np.mean(Z[indx])
        else:
            # Check if there are contigous blocks left/right/up/down
            goleft = (x%n_splits != 0) and x > 0
            goright = ((x+1)%n_splits != 0) and (x < (n_splits*n_splits)-1)
            goup = x >= n_splits
            godown = x < (n_splits*(n_splits-1))

            # Average over all contigous blocks
            if goup: indx = np.logical_or(indx, XY_limits[x-n_splits, :])
            if godown: indx = np.logical_or(indx, XY_limits[x+n_splits, :])
            if goleft: indx = np.logical_or(indx, XY_limits[x-1, :])
            if goright: indx = np.logical_or(indx, XY_limits[x+1, :])
            if goup and goleft: indx = np.logical_or(indx, XY_limits[x-n_splits-1, :])
            if goup and goright: indx = np.logical_or(indx, XY_limits[x-n_splits+1, :])
            if godown and goleft: indx = np.logical_or(indx, XY_limits[x+n_splits-1, :])
            if godown and goright: indx = np.logical_or(indx, XY_limits[x+n_splits+1, :])

            if indx.any():
                Z_means[x] = np.mean(Z[indx])
            else: # If no info available still, take mean over entire dataset
                Z_means[x] = np.mean(Z)

    return Z_means

def divideDataXY(states, error):
    """
    Inputs
        state - shape (4, N, nS)
        error - shape (N, ) if nonzero, error in that measurement
    Outputs
        X - shape (M*4, nS)
        Y - shape (M*4, nS)
    """
    def XYarray(states, X, Y):
        """ Assuming no error measurements
        Input -> state shape(4, N, nS)
        Output -> X & Y, shape()
        """
        X = np.concatenate((X, states[:, :-1, :]), axis=1)
        Y = np.concatenate((Y, states[:, 1:, :]), axis=1)
        return X, Y
    nS = states.shape[-1]
    X, Y = np.zeros((4, 0, nS)), np.zeros((4, 0, nS))
    # Index where fault occurred
    error_indx = np.argwhere(error).reshape(-1)
    if error_indx.size == 0: # No faults
        X, Y = XYarray(states, X, Y)
    else:
        # Find M
        X, Y = XYarray(states[:, :error_indx[0], :], X, Y)
        if error_indx.size > 1:
            for i in range(error_indx.size-1):
                X, Y = XYarray(states[:, error_indx[i]+1:error_indx[i+1], :], X, Y)
        X, Y = XYarray(states[:, error_indx[-1]+1:, :], X, Y)
    return X.reshape(-1, nS), Y.reshape(-1, nS)


def plotTemperaturesState(states, vlimits=None, ax=None, saveFig=None, fig=None, title=None):
    """
    Plot the states in a 2D temperature graph

    Inputs
    ------
        states: nparray, shape(nS,)
        vlimits: None or  nparray, shape(2,) [vmin, vmax]
        ax: for plotting within a subplot
        saveFig: filename of figure save
    """
    n_splits = int(np.sqrt(states.size))
    assert(n_splits == np.sqrt(states.size)) # Make sure sqrt is an int

    # Indexes so that plot is in the correct orientation
    nS = int(np.sqrt(states.size))
    idx = [(nS-1-i)*nS+j for i in range(nS) for j in range(nS)]

    states = states[idx].reshape((n_splits, n_splits))

    X_splits = np.linspace(0, n_splits, n_splits+1)
    Y_splits = np.linspace(0, n_splits, n_splits+1)

    if vlimits is None:
        vmin = abs(states).min()
        vmax = abs(states).max()
    else:
        vmin = vlimits[0]
        vmax = vlimits[1]

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
        p = ax.pcolor(X_splits, Y_splits, states, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        cb = fig.colorbar(p)
        if title is not None: fig.title(title)
    else:
        p = ax.pcolor(X_splits, Y_splits, states, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        if fig is not None: cb = fig.colorbar(p)
        if title is not None: plt.title(title)

    if saveFig is not None:
        plt.savefig(saveFig)
        plt.close()

def plotAllStatesFile(states, file_name, layer_nums):
    """Plot all states into a single large image"""
    N = states.shape[1]
    pltrow, pltcol = int(np.sqrt(N)), int(np.ceil(np.sqrt(N)))
    vlimits = [states.min(), states.max()]
    for sample in range(4):
        fig, ax = plt.subplots(pltrow, pltcol, figsize=(pltcol*4, pltrow*4))
        fig.subplots_adjust(hspace=.1, wspace=.1)
        for row in range(pltrow):
            for col in range(pltcol):
                i = row*pltcol + col
                fig_in = None if i else fig # only show color bar in last picture
                ax[row,col].xaxis.set_visible(False)
                ax[row,col].yaxis.set_visible(False)
                if i < states.shape[1]:
                    plotTemperaturesState(states[sample, i, :],
                        vlimits=vlimits, ax=ax[row,col], fig=fig_in)
                    ax[row,col].set_title(("Sample %d, l = %.2f")%(sample, layer_nums[i]))
        plt.savefig(file_name+(("s%d.png")%(sample)),
            bbox_inches='tight', dpi=50)
        plt.close()


def getInvalidState(nS):
    """
    When incorrect readings are encountered, nan values will prevent the entire
    piece from being properly plotted. In this case, assign state values that are
    characteristic and can be easily recognised.

    i.e. where O is a low value, and - and average one
    O - - O
    - - - -
    - - - -
    O - - O
    """
    lowval, medval = 900, 1050
    state = np.ones((nS,)) * medval
    nrow = int(np.sqrt(nS))
    state[0], state[nrow-1], state[-nrow], state[-1] = lowval, lowval, lowval, lowval
    return state

def isValidState(states):
    """
    Returns an array of bools indicating if the corresponding states are valid
    or not.

    Inputs: states - shape(N, nS)
    Outputs: valid - shape(N, nS)
    """
    nS = states.shape[-1]
    invalid_state = getInvalidState(nS)
    return (states == invalid_state).sum(-1) < nS