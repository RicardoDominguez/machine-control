import os
import numpy as np
import re
import time

# CHECK WHEN EACH FOLDER IS MADE

# Full path must be provided
session_folder = 'C:/AconitySTUDIO/log/session_2019_03_08_13-20-43.547/'
layers = [1, 167]
n_pieces = 3
copy_waitTime = 2

def getLatestConfigJobFolder(folder_name, init_str):
    latest_n = -1
    name = None
    for filename in os.listdir(folder_name):
        match = re.search(r''+init_str+'_(\d+)_(\w+)', filename)
        if match and match.group(2) is not 'none' and int(match.group(1)) > latest_n:
                latest_n = int(match.group(1))
                name = filename
    if name is None: raise ValueError('No suitable '+init_str+' folders found')
    return name

def pieceNumber(piece_indx):
    return (piece_indx+1)*3+1

def getNextLayerPiece(layer, piece_indx, n_pieces, retPieces=True):
    next_layer = layer
    next_piece_indx = (piece_indx + 1) % n_pieces
    if next_piece_indx == 0: next_layer += 1
    if retPieces:
        current_piece = pieceNumber(piece_indx)
        next_piece = pieceNumber(next_piece_indx)
        return current_piece, next_layer, next_piece_indx, next_piece
    else:
        return next_layer, next_piece_indx

# Extract latest config
config_folder = getLatestConfigJobFolder(session_folder, 'config')
job_folder = getLatestConfigJobFolder(session_folder+config_folder+'/', 'job')
data_folder = session_folder+config_folder+'/'+job_folder+'/sensors/2Pyrometer/pyrometer2/'
print("Data folder found is "+data_folder)

curr_layer = layers[0]
curr_piece_indx = 0
last_curr_piece_indx = curr_piece_indx
while(True):

    current_piece, next_layer, next_piece_indx, next_piece = \
        getNextLayerPiece(curr_layer, curr_piece_indx, n_pieces)

    # Check if files appeared
    no_new_file = True

    expected_file = data_folder+str(current_piece)+'/'+str(np.round(curr_layer*0.03, 2))+'.pcd'
    next_file = data_folder+str(next_piece)+'/'+str(np.round(next_layer*0.03, 2))+'.pcd'
    print("Expected file "+expected_file)
    #print("Next file "+next_file)
    while(no_new_file):
        if os.path.isfile(expected_file):
            no_new_file, skipPiece = False, False
        elif os.path.isfile(next_file):
            no_new_file, skipPiece = False, True

    if skipPiece:
        print("Piece skipped")
        expected_file = next_file
        curr_layer, curr_piece_indx = next_layer, next_piece_indx

    #if curr_piece_indx <= last_curr_piece_indx:
    print("Sleeping...")
    time.sleep(copy_waitTime)
    
    data = np.loadtxt(expected_file)
    print("Data shape is", data.shape)
    
    print("Red layer %d, piece %d" % (curr_layer, curr_piece_indx))
    print("File name "+expected_file)
    curr_layer, curr_piece_indx = getNextLayerPiece(curr_layer, curr_piece_indx, n_pieces, retPieces=False)
