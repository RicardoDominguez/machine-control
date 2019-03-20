"""
Folder structure
  |- raw_data_folder_name/
  |- processed_data_folder_name/
    |- name_state_data.npy (parameters x 4 x layers x states)
    |- images/
      |- square_images/
      |- cold_line_imgs/ (with red line and value)
        |- part
      |- state_evolve_imgs/
        |- part
      |- ratio_removed/ (removed:hold)
      |- cutoff/
    |- individual_state_data/
        |- name_individual_state_data.npy (4 x layers x states)
        |- square_limits.npy (parameters x 4 x 4)
"""
from mainFile import *
import os
import time

raw_data_folder_name = '/media/ricardo/DATA/run2_1/'
run_name = '/media/ricardo/DATA/data_files/run2_1/'
min_part = 4
max_part = 115
min_layer = 0
max_layer = 0.87
n_state_divisions = 4
updateHighLevelData = True # Otherwise small changes overwrite a lot of data
individualGraphs = True

# Save directories
imgs_dir = run_name+'/images/'
square_division_dir = imgs_dir+'square_images/'
cold_line_imgs_dir = imgs_dir+'cold_line/'
state_evolve_dir = imgs_dir+'state_evolve/'
delete_ratio_dir = imgs_dir+'delete_ratio/'
cutoff_dir = imgs_dir+'cutoff_dir/'
individual_state_data = run_name+'/saves/'

def makeDirIfDoesNotExist(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def initialise_layer_structure(dir, min_layer, max_layer):
    """ Returns a sorted list of all valid '.pcd' files contained in 'dir'"""
    valid_names = []
    valid_nums = []
    for filename in os.listdir(dir):
        if filename.endswith(".pcd"):
            layer_num = float(filename[:-4])
            if min_layer <= layer_num <= max_layer:
                valid_names.append(filename)
                valid_nums.append(layer_num)
    indx = sorted(range(len(valid_nums)), key=valid_nums.__getitem__) # argsort
    return [valid_names[i] for i in indx]

def plotFigMeanStd(x, y, name):
    plt.figure(figsize=(10,6))
    plt.plot(x, y)
    plt.title(("Mean: %.4f, std: %.4f, max: %.4f, min: %.4f") % (y.mean(), y.std(), y.max(), y.min()))
    plt.savefig(name+'.png', bbox_inches='tight', dpi=100)
    np.savetxt(name+'.txt', np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), 1), '%.4f')
    plt.close()

makeDirIfDoesNotExist(imgs_dir)
makeDirIfDoesNotExist(square_division_dir)
makeDirIfDoesNotExist(cold_line_imgs_dir)
makeDirIfDoesNotExist(state_evolve_dir)
makeDirIfDoesNotExist(delete_ratio_dir)
makeDirIfDoesNotExist(cutoff_dir)
makeDirIfDoesNotExist(individual_state_data)

# Find all directories within data folder
d = raw_data_folder_name
part_folders = [os.path.join(d, o) for o in os.listdir(d)
                if (os.path.isdir(os.path.join(d,o)) and o.isdigit() and
                (min_part <= int(o) <= max_part))]
len_folders = len(part_folders)
# Sort them
folder_n = [int(os.path.basename(part_folders[i])) for i in range(len_folders)]
indx = sorted(range(len(folder_n)), key=folder_n.__getitem__)
part_folders = [part_folders[i] for i in indx]

# Iterate through each part folder
total_start_t = time.time()
for curr_folder in range(len_folders):
    start_t = time.time()
    layer_structure_init = False
    folder_dir = part_folders[curr_folder]
    part_id = os.path.basename(folder_dir) # name of folder
    folder_dir += '/'
    if not layer_structure_init:
        save_name = folder_dir
        makeDirIfDoesNotExist(save_name)
        sorted_layer_names = initialise_layer_structure(save_name,
                                                        min_layer, max_layer)
        n_layers = len(sorted_layer_names)
        layer_structure_init = True

    # Explore layers
    # ----------------------------------------------------------------------
    print("Exploring "+folder_dir+((" (%d/%d)")%(curr_folder+1, len_folders)))
    cold_line_imgs_part_dir = cold_line_imgs_dir + part_id + '/'
    makeDirIfDoesNotExist(cold_line_imgs_part_dir)
    cutoff_values = np.empty((n_layers,))
    delete_ratio = np.empty((n_layers,))
    error_division = np.zeros((n_layers,))
    states = np.empty((4, n_layers, n_state_divisions**2))
    layer_nums = np.empty((n_layers,))
    # Find square limits
    data = loadData(folder_dir+sorted_layer_names[0])
    square_limits = divide4squares(data)
    divideDataRectangleLimits(data, square_limits, returnMode=0, plot=True,
            saveName=part_id, saveDir=square_division_dir)
    # Iterate layers
    for curr_layer in range(n_layers):
        print(("F:%d/%d, L:%d/%d")%(curr_folder+1, len_folders, curr_layer+1, n_layers))
        layer_name = sorted_layer_names[curr_layer][:-4]
        layer_nums[curr_layer] = float(layer_name)
        layer_name = "%.2f"%layer_nums[curr_layer]
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

    # Saves and plots
    if updateHighLevelData:
        np.save(individual_state_data+part_id+'.npy', states)
        np.save(individual_state_data+'sq'+part_id+'.npy', square_limits)
        np.save(individual_state_data+part_id+'_X.npy', X)
        np.save(individual_state_data+part_id+'_Y.npy', Y)
        np.save(individual_state_data+part_id+'_errors.npy', error_division)
        plotFigMeanStd(layer_nums, cutoff_values, cutoff_dir+part_id)
        plotFigMeanStd(layer_nums, delete_ratio, delete_ratio_dir+part_id)
        save_dir = state_evolve_dir+part_id+'/'
        makeDirIfDoesNotExist(save_dir)
        plotAllStatesFile(states, save_dir, layer_nums)
    print(("Run time for the part: %.2f")%(time.time()-start_t))
print("Finished.")
print(("Total run time: %.2f")%(time.time()-total_start_t))
