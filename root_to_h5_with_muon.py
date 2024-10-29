import sys
import os
import numpy as np
import h5py
import uproot
from event_selection import get_event_and_jet_mask
from b_jet_match import get_b_jet_index
from muon_correction import muon_correction
from utils import get_jet_data, get_momentum


def create_triHiggs_dataset(f, n_selected):
    f.create_dataset("INPUTS/Source/MASK", (n_selected, MAX_JETS), dtype="|b1")
    f.create_dataset("INPUTS/Source/pt", (n_selected, MAX_JETS), dtype="<f4")
    f.create_dataset("INPUTS/Source/eta", (n_selected, MAX_JETS), dtype="<f4")
    f.create_dataset("INPUTS/Source/phi", (n_selected, MAX_JETS), dtype="<f4")
    f.create_dataset("INPUTS/Source/mass", (n_selected, MAX_JETS), dtype="<f4")
    f.create_dataset("INPUTS/Source/btag", (n_selected, MAX_JETS), dtype="|b1")
    f.create_dataset("TARGETS/h1/b1", (n_selected,), dtype="<i8")
    f.create_dataset("TARGETS/h1/b2", (n_selected,), dtype="<i8")
    f.create_dataset("TARGETS/h2/b1", (n_selected,), dtype="<i8")
    f.create_dataset("TARGETS/h2/b2", (n_selected,), dtype="<i8")
    f.create_dataset("TARGETS/h3/b1", (n_selected,), dtype="<i8")
    f.create_dataset("TARGETS/h3/b2", (n_selected,), dtype="<i8")
    f.create_dataset("CLASSIFICATIONS/EVENT/signal", (n_selected,), dtype="|b1")

def higgs_pt_sort(b_jet_index, jet_pt, jet_eta, jet_phi, jet_mass):
    sortable = (b_jet_index != -1).all(axis=1)
    pt = np.take_along_axis(jet_pt[sortable], b_jet_index[sortable], axis=1)
    eta = np.take_along_axis(jet_eta[sortable], b_jet_index[sortable], axis=1)
    phi = np.take_along_axis(jet_phi[sortable], b_jet_index[sortable], axis=1)
    mass = np.take_along_axis(jet_mass[sortable], b_jet_index[sortable], axis=1)
    p = get_momentum(pt, eta, phi, mass)
    q = p.reshape((p.shape[0], 3, 2, 4)).sum(axis=2)
    qt = np.sqrt(q[..., 1] ** 2 + q[..., 2] ** 2)
    qtinds = qt.argsort(axis=1)[:,::-1]
    sortable_jets = b_jet_index[sortable]
    sortable_jets = sortable_jets.reshape((sortable_jets.shape[0], 3, 2))
    sortable_jets = np.take_along_axis(sortable_jets, qtinds[:,:,None], axis=1)
    sortable_jets.sort(axis=2)
    sortable_jets = sortable_jets.reshape((sortable_jets.shape[0], 6))
    b_jet_index[sortable] = sortable_jets

def write_data(f, root_file, event_mask, jet_mask, n_start, n_end, i_selected):
    n_selected = event_mask.sum()
    pt, eta, phi, mass, btag = get_jet_data(root_file, event_mask, MAX_JETS, n_start, n_end)
    f["INPUTS/Source/MASK"][i_selected:i_selected + n_selected] = jet_mask[event_mask]
    f["INPUTS/Source/pt"][i_selected:i_selected + n_selected] = pt
    f["INPUTS/Source/eta"][i_selected:i_selected + n_selected] = eta
    f["INPUTS/Source/phi"][i_selected:i_selected + n_selected] = phi
    f["INPUTS/Source/mass"][i_selected:i_selected + n_selected] = mass
    f["INPUTS/Source/btag"][i_selected:i_selected + n_selected] = btag.filled(False)

    b_jet_index = get_b_jet_index(root_file, event_mask, jet_mask, MAX_JETS, n_start, n_end)
    muon_correction(root_file, event_mask, b_jet_index, n_start, n_end, pt, eta, phi, mass)
    return
    higgs_pt_sort(b_jet_index, pt, eta, phi, mass)
    f["TARGETS/h1/b1"][i_selected:i_selected + n_selected] = b_jet_index[:, 0]
    f["TARGETS/h1/b2"][i_selected:i_selected + n_selected] = b_jet_index[:, 1]
    f["TARGETS/h2/b1"][i_selected:i_selected + n_selected] = b_jet_index[:, 2]
    f["TARGETS/h2/b2"][i_selected:i_selected + n_selected] = b_jet_index[:, 3]
    f["TARGETS/h3/b1"][i_selected:i_selected + n_selected] = b_jet_index[:, 4]
    f["TARGETS/h3/b2"][i_selected:i_selected + n_selected] = b_jet_index[:, 5]

    f["CLASSIFICATIONS/EVENT/signal"][i_selected:i_selected + n_selected] = 1

    return i_selected + n_selected

def root_to_h5(root_path, h5_path, mode, n_batch):
    # preselected_mode:
    #   1. More than 6 jets have pT > 25 GeV and |η| < 2.5
    #   2. More than 6 jets have pT > 25 GeV, |η| < 2.5, and are b-tagged
    #   3. More than 6 jets have pT > 20 GeV and |η| < 2.5. 4 of them have pT > 40. 4 of them are b-tagged
    assert(preselect_mode in (1, 2, 3))
    root_file = uproot.open(root_path)["Delphes;1"]
    n_file = len(root_file["Event.Weight"].array())
    event_mask, jet_mask = get_event_and_jet_mask(root_file, MAX_JETS, mode)
    n_selected = event_mask.sum()

    h5_file = h5py.File(h5_path, "a")
    create_triHiggs_dataset(h5_file, n_selected)
    i_selected = 0
    for n in range(0, n_file // n_batch):
        print("  Processing batch", n)
        n_start = n * n_batch
        n_end = (n + 1) * n_batch
        i_selected = write_data(h5_file, root_file, event_mask[n_start:n_end], jet_mask[n_start:n_end], n_start, n_end, i_selected)
    if n_file % n_batch != 0:
        print("  Processing batch", n_file // n_batch)
        n_start = n_file // n_batch * n_batch
        n_end = n_file
        i_selected = write_data(h5_file, root_file, event_mask[n_start:n_end], jet_mask[n_start:n_end], n_start, n_end, i_selected)
    h5_file.close()


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
        h5_path = os.path.join(output_folder, os.path.splitext(input_file)[0]) + ".h5"      
        # clean existing h5 file
        h5py.File(h5_path, "w").close()
        root_to_h5(root_path, h5_path, preselect_mode, n_batch)