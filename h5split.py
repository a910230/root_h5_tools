import os
import sys
import h5py
import numpy as np
from utils import h5py_dataset_iterator

def split(input_file, n):
    file_name = os.path.splitext(input_file)
    output_file_1 = file_name[0] + f"_{n}" + file_name[1]
    output_file_2 = file_name[0] + "_res" + file_name[1]

    f_out_1 = h5py.File(output_file_1, "w-")
    f_out_2 = h5py.File(output_file_2, "w-")
    with h5py.File(input_file, "r") as f_in:
        for (path, dset) in h5py_dataset_iterator(f_in):
            print(path)
            print(dset.shape)
            if (dset.shape[0] < n):
                print("Asked partition is too large.")
                break
            f_out_1.create_dataset(path, (n, *(dset.shape[1:])), maxshape=(None, *(dset.shape[1:])), dtype=dset.dtype)
            f_out_1[path][:] = dset[:n]
            f_out_2.create_dataset(path, (dset.shape[0] - n, *(dset.shape[1:])), maxshape=(None, *(dset.shape[1:])), dtype=dset.dtype)
            f_out_2[path][:] = dset[n:]
        
    f_out_1.close()
    f_out_2.close()

if __name__ == '__main__':

    input_file = sys.argv[1]
    n = int(sys.argv[2])
    split(input_file, n)

    