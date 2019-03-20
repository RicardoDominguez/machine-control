
states = np.empty((4, n_state_divisions**2))

# only done once
data = loadData(folder_dir+sorted_layer_names[0])
square_limits = divide4squares(data)
divideDataRectangleLimits(data, square_limits, returnMode=0, plot=True,
        saveName=part_id, saveDir=square_division_dir)

# all other layers
data = loadData(folder_dir+sorted_layer_names[curr_layer])
data, cutoff_values[curr_layer] = removeColdLines(data, returnMode=2,
                                  plot=individualGraphs, saveName=layer_name,
                                  saveDir=cold_line_imgs_part_dir)
data_divisions, delete_ratio[curr_layer], error_division[curr_layer] = \
    divideDataRectangleLimits(data, square_limits, returnMode=4)

for sample in range(4):
    if error_division[curr_layer]:
        states[sample, curr_layer, :] = getInvalidState(int(n_state_divisions**2))
    else:
        states[sample, curr_layer, :] = pieceStateMeans(data_divisions[sample],
                                        square_limits[sample], n_state_divisions)
X, Y = divideDataXY(states, error_division)