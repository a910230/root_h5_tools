import os
import sys
import h5py
import numpy as np
from utils import h5py_dataset_iterator

def merge(input_folder, output_file):
    if not os.path.isdir(input_folder):
        raise SyntaxError("Arg 1 is not a folder")
    input_files = np.sort([f for f in os.listdir(input_folder) if os.path.splitext(f)[1] == ".h5"])
    if input_files.size == 0:
        raise SyntaxError("No h5 file in the input folder")
    
    # Assume all h5 files in the folder have the same structure
    metadata = {}
    f_in = h5py.File(os.path.join(input_folder, input_files[0]), "r")

    for (path, dset) in h5py_dataset_iterator(f_in):
        metadata[path] = [[0, *(dset.shape[1:])], dset.dtype, 0]
    f_in.close()

    for f in input_files:
        f_in = h5py.File(os.path.join(input_folder, f), "r")
        for key in metadata:
            metadata[key][0][0] += f_in[key].shape[0]
        f_in.close()

    f_out = h5py.File(output_file, "w").close()
    f_out = h5py.File(output_file, "w")
    for key in metadata:
        f_out.create_dataset(key, metadata[key][0], maxshape=(None, *(metadata[key][0][1:])), dtype=metadata[key][1])
    for f in input_files:
        with h5py.File(os.path.join(input_folder, f), "r") as f_in:
            for key in metadata:
                cur_ind = metadata[key][2]
                cur_len = f_in[key].shape[0]
                f_out[key][cur_ind:cur_ind + cur_len] = f_in[key][:]
                metadata[key][2] += cur_len
    f_out.close()

if __name__ == '__main__':

    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    merge(input_folder, output_file)