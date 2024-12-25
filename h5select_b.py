import os
import sys
import h5py
import numpy as np
from utils import h5py_dataset_iterator

def select(input_file, nbj):
    file_name = os.path.splitext(input_file)
    output_file_1 = file_name[0] + f"_{nbj}b" + file_name[1]
    output_file_2 = file_name[0] + "_res" + file_name[1]

    f_out_1 = h5py.File(output_file_1, "w-")
    f_out_2 = h5py.File(output_file_2, "w-")
    with h5py.File(input_file, "r") as f_in:
        btag = f_in["INPUTS/Source/btag"][:]
        mask = (btag.sum(axis=1) >= nbj)

        for (path, dset) in h5py_dataset_iterator(f_in):
            selected = dset[:][mask, ...]
            unselected = dset[:][~mask, ...]
            print(path)
            print(selected.shape)
            f_out_1.create_dataset(path, selected.shape, maxshape=(None, *(selected.shape[1:])), dtype=selected.dtype)
            f_out_1[path][:] = selected
            f_out_2.create_dataset(path, unselected.shape, maxshape=(None, *(unselected.shape[1:])), dtype=unselected.dtype)
            f_out_2[path][:] = unselected
        
    f_out_1.close()
    f_out_2.close()

if __name__ == '__main__':

    input_file = sys.argv[1]
    nbj = int(sys.argv[2])
    select(input_file, nbj)    