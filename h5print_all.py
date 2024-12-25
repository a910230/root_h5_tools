import os
import sys
import h5py
import numpy as np
from utils import h5py_dataset_iterator

def print_all(input_file, n=5):
    with h5py.File(input_file, "r") as f_in:
        for (path, dset) in h5py_dataset_iterator(f_in):
            print(os.path.split(path)[1])
            print(dset[:n])

if __name__ == '__main__':

    input_file = sys.argv[1]
    if len(sys.argv) == 3:
        n = int(sys.argv[2])
        print_all(input_file, n)
    else:
        print_all(input_file)