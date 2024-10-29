import os
import sys
import h5py
import numpy as np
from utils import h5py_dataset_iterator

def show(input_file):
    with h5py.File(input_file, "r") as f_in:
        for (path, dset) in h5py_dataset_iterator(f_in):
            print(path, dset.shape, dset.dtype)


if __name__ == '__main__':

    input_file = sys.argv[1]
    show(input_file)