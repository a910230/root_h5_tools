import os
import sys
import h5py
import numpy as np
from utils import h5py_dataset_iterator

def print_ans(input_file):
    with h5py.File(input_file, "r") as f_in:
        b11 = f_in["TARGETS/h1/b1"][:]
        b12 = f_in["TARGETS/h1/b2"][:]
        b21 = f_in["TARGETS/h2/b1"][:]
        b22 = f_in["TARGETS/h2/b2"][:]
        b31 = f_in["TARGETS/h3/b1"][:]
        b32 = f_in["TARGETS/h3/b2"][:]
    ans = np.array([b11, b12, b21, b22, b31, b32]).transpose()
    print(ans)



if __name__ == '__main__':

    input_file = sys.argv[1]
    print_ans(input_file)