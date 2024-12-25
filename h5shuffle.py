import os
import sys
import h5py
import numpy as np
from utils import h5py_dataset_iterator

def shuffle(input_file):
    output_file = os.path.splitext(input_file)[0] + "_shuffled.h5"
    h5py.File(output_file, "w").close()
    f_out = h5py.File(output_file, "w")
    with h5py.File(input_file, "r") as f_in:
        shuffled_ind = np.random.permutation(f_in["INPUTS/Source/MASK"][:].shape[0])
        for (path, dset) in h5py_dataset_iterator(f_in):
            shuffled = dset[:][shuffled_ind]
            f_out.create_dataset(path, shuffled.shape, maxshape=(None, *(shuffled.shape[1:])), dtype=shuffled.dtype)
            f_out[path][:] = shuffled
        
    f_out.close()

if __name__ == '__main__':

    input_file = sys.argv[1]
    shuffle(input_file)    