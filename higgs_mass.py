import sys
import os
import uproot
import h5py
import numpy as np
from event_selection import get_event_and_jet_mask
from utils import get_jet_data, get_momentum, get_mass
from b_jet_match import get_b_jet_index


def get_higgs_mass(b_jets, jet_PT, jet_Eta, jet_Phi, jet_Mass): 
    pt = np.take_along_axis(jet_PT, b_jets, axis=1)
    eta = np.take_along_axis(jet_Eta, b_jets, axis=1)
    phi = np.take_along_axis(jet_Phi, b_jets, axis=1)
    m = np.take_along_axis(jet_Mass, b_jets, axis=1)

    p = get_momentum(pt, eta, phi, m)
    q = p.reshape((p.shape[0], 3, 2, 4)).sum(axis=2)
    qt = np.sqrt(q[..., 1] ** 2 + q[..., 2] ** 2)
    qm = get_mass(q)
    qtinds = qt.argsort(axis=1)[:,::-1]
    qm = np.take_along_axis(qm, qtinds, axis=1)
    
    return qm

def get_batch_higgs_mass(root_file, event_mask, jet_mask, n_start, n_end):
    
    jet_PT, jet_Eta, jet_Phi, jet_Mass, jet_BTag = get_jet_data(root_file, event_mask, MAX_JETS, n_start, n_end)
    b_jets = get_b_jet_index(root_file, event_mask, jet_mask, MAX_JETS, n_start, n_end)

    matched = np.take_along_axis(jet_BTag, b_jets, axis=1).all(axis=1) & (b_jets != -1).all(axis=1)

    return get_higgs_mass(b_jets[matched], jet_PT[matched], jet_Eta[matched], jet_Phi[matched], jet_Mass[matched])

def export_higgs_mass(root_path, h5_path, mode, n_batch):
    root_file = uproot.open(root_path)["Delphes;1"]
    n_file = len(root_file["Event.Weight"].array())
    event_mask, jet_mask = get_event_and_jet_mask(root_file, MAX_JETS, mode)

    higgs_mass = []
    for n in range(0, n_file // n_batch):
        print("  Processing batch", n)
        n_start = n * n_batch
        n_end = (n + 1) * n_batch
        higgs_mass.append(get_batch_higgs_mass(root_file, event_mask[n_start:n_end], jet_mask[n_start:n_end], n_start, n_end))
    if n_file % n_batch != 0:
        print("  Processing batch", n_file // n_batch)
        n_start = n_file // n_batch * n_batch
        n_end = n_file
        higgs_mass.append(get_batch_higgs_mass(root_file, event_mask[n_start:n_end], jet_mask[n_start:n_end], n_start, n_end))
    if (len(higgs_mass) == 1):
        higgs_mass = higgs_mass[0]
    else:
        higgs_mass = np.concatenate(higgs_mass, axis=0)

    with h5py.File(h5_path, "a") as f_out:
        f_out.create_dataset("mass", (higgs_mass.shape[0], 3), dtype="<f4")
        f_out["mass"][:] = higgs_mass
    root_file.close()

if __name__ == '__main__':
    MAX_JETS = 15
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    preselect_mode = int(sys.argv[3])
    n_batch = int(sys.argv[4])
    
    if not os.path.isdir(input_folder):
        raise SyntaxError("Arg 1 is not a folder")    
    if not os.path.isdir(output_folder):
        raise SyntaxError("Arg 2 is not a folder")
    
    input_files = np.sort([f for f in os.listdir(input_folder) if os.path.splitext(f)[1] == ".root"])
    for input_file in input_files:
        print("Processing file", input_file, "...")
        root_path = os.path.join(input_folder, input_file)
        h5_path = os.path.join(output_folder, os.path.splitext(input_file)[0]) + "_mass.h5"      
        # clean existing h5 file
        h5py.File(h5_path, "w").close()
        export_higgs_mass(root_path, h5_path, preselect_mode, n_batch)