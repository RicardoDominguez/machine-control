"""Implements functions for the processing of raw pyrometer data."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
import time

def loadData(file_name, timeit=False, verbose=False):
    """Load data from a pyrometer file into a np.array.

    The pyrometer .pcb file has structure [X, Y, mV, mV].

    Arguments:
        file_name (str): Path to file.
        timeit (bool, optional): If True, returns the time taken the read the file.
        verbose (bool, optional): If True, prints to screen additional information.

    Returns:
        np.array: Data loaded from the .pcb file, with shape (`n`, 3)
    """
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

def removeColdLines(data, returnMode=0, timeit=False, verbose=False,
        plot=False, saveName=None, saveDir=None):
    """Removes cold points by deleting all data before the first peak in the
    pyrometer data frequency distribution (around 800mV).

    This is done by deleting any data bellow the peak + its width at half height.

    Arguments:
        data (np.array): Raw pyrometer data with shape (`n`, 3).
        returnMode (int, optional): Determines what parameters will be returned.
        timeit (bool, optional): If True, returns the time taken to execute the function.
        verbose (bool, optional): If True, prints to screen additional information.
        plot (bool, optional): If True, produces a plot of the data distribution and cutoff value.
        saveName (str, optional): Name of file where the image is to be saved.
        saveDir (str, optional): Name of directory where the image is to be saved.

    Return modes:
        - 0: returns data above cutoff value (np.array).
        - 1: returns cutoff value (float).
        - 2: returns both data and cutoff value (np.array, float).
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

def divideSingleSquare(data):
    """ Divides the input data into a single square.

    Arguments:
        data (np.array): Raw pyrometer data with shape (`n`, 3)

    Returns:
        np.array: Square limits as [xmin, xmax, ymin, ymax] with shape (4,)
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
    """Returns a list with the data for a square given its limits.

    Arguments:
        data (np.array): Raw pyrometer data with shape (`n`, 3)
        square_limits (np.array): Limits of the square with shape (4,)
        returnMode (int, optional): Determines what parameters will be returned.
        plot (bool, optional): If True, produces a plot of the data distribution and cutoff value.
        saveName (str, optional): Name of file where the image is to be saved.
        saveDir (str, optional): Name of directory where the image is to be saved.

    Output modes:
        - 1: only data within limits
        - 2: data within limits, data outside of limits
        - 3: data within limits, ratio of data outside limits / total data
        - 4: data, ratio of data deleted, error flag
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
    """Convert the raw data of a single piece into a low-dimensional state vector
    by dividing the piece into regions (squares) and taking the mean of the
    temperature within that square.

    If a particular square has no data points, then average contiguous states.

    Arguments:
        piece_data (np.array): Part data with shape (`n`, 3)
        piece_limits (np.array, optional): Limits of the rectangular part as [xmin, xmax, ymin, ymax]
        n_splits (int, optional): Divided into n_splits*n_splits regions.

    Returns:
        np.array: State vector with shape (`n_splits*n_splits`,)
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

def plotTemperaturesState(states, vlimits=None, ax=None, saveFig=None, fig=None, title=None):
    """Plots the temperature in each of the regions comprising the low-dimensional states

    Arguments:
        states (np.array): State vector with shape (`nS`,)
        vlimits (np.array, optional): Temperature limits as [vmin, vmax]
        ax (matplotlib.Axes, optional): Produce plot within a subplot environment.
        saveFig (str, optional): Name of file to save the image.
        fig (matplotlib.figure.Figure, optional): Figure to which add the plot.
        title (str, optional): Plot title.
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
