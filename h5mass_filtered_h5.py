import sys
import h5py
import numpy as np
from utils import get_momentum


# def get_higgs_mass(b_jets, jet_PT, jet_Eta, jet_Phi, jet_Mass): 
#     pt = np.take_along_axis(jet_PT, b_jets, axis=1)
#     eta = np.take_along_axis(jet_Eta, b_jets, axis=1)
#     phi = np.take_along_axis(jet_Phi, b_jets, axis=1)
#     m = np.take_along_axis(jet_Mass, b_jets, axis=1)

#     p = get_momentum(pt, eta, phi, m)
#     q = p.reshape((p.shape[0], 3, 2, 4)).sum(axis=2)
#     qt = np.sqrt(q[..., 1] ** 2 + q[..., 2] ** 2)
#     qm = get_mass(q)
#     qtinds = qt.argsort(axis=1)[:,::-1]
#     qm = np.take_along_axis(qm, qtinds, axis=1)
    
#     return qm

# def get_batch_higgs_mass(root_file, event_mask, jet_mask, n_start, n_end):
    
#     jet_PT, jet_Eta, jet_Phi, jet_Mass, jet_BTag = get_jet_data(root_file, event_mask, MAX_JETS, n_start, n_end)
#     b_jets = get_b_jet_index(root_file, event_mask, jet_mask, MAX_JETS, n_start, n_end)

#     matched = np.take_along_axis(jet_BTag, b_jets, axis=1).all(axis=1) & (b_jets != -1).all(axis=1)

#     return get_higgs_mass(b_jets[matched], jet_PT[matched], jet_Eta[matched], jet_Phi[matched], jet_Mass[matched])

def tmp(f, n_selected):
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

def get_match_answer(f):
    h11 = f["TARGETS/h1/b1"][:]
    h12 = f["TARGETS/h1/b2"][:]
    h21 = f["TARGETS/h2/b1"][:]
    h22 = f["TARGETS/h2/b2"][:]
    h31 = f["TARGETS/h3/b1"][:]
    h32 = f["TARGETS/h3/b2"][:]

    ans = np.array([h11, h12, h21, h22, h31, h32]).transpose()
    return ans

def get_jet_data(f):
    pt = f["INPUTS/Source/pt"][:]
    eta = f["INPUTS/Source/eta"][:]
    phi = f["INPUTS/Source/phi"][:]
    mass = f["INPUTS/Source/mass"][:]
    return pt, eta, phi, mass

def get_pt_mass(p):
    pt = np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)
    mass = np.sqrt(p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2)
    return pt, mass

def get_higgs_mass(match_answer, all_pt, all_eta, all_phi, all_mass):
    jets_pt = np.take_along_axis(all_pt, match_answer, axis=1)
    jets_eta = np.take_along_axis(all_eta, match_answer, axis=1)
    jets_phi = np.take_along_axis(all_phi, match_answer, axis=1)
    jets_mass = np.take_along_axis(all_mass, match_answer, axis=1)

    jets_momentum = get_momentum(jets_pt, jets_eta, jets_phi, jets_mass)
    higgs_momentum = jets_momentum.reshape((jets_momentum.shape[0], 3, 2, 4)).sum(axis=2)

    higgs_pt, higgs_mass = get_pt_mass(higgs_momentum)
    ptinds = higgs_pt.argsort(axis=1)[:,::-1]
    higgs_mass = np.take_along_axis(higgs_mass, ptinds, axis=1)

    return higgs_mass


def export_higgs_mass(input_path, output_path):
    with h5py.File(input_path, "r") as input_file:
        # n_file = input_file["CLASSIFICATIONS/EVENT/signal"].shape[0]
        match_answer = get_match_answer(input_file)
        jet_pt, jet_eta, jet_phi, jet_mass = get_jet_data(input_file)
        higgs_mass = get_higgs_mass(match_answer, jet_pt, jet_eta, jet_phi, jet_mass)

    with h5py.File(output_path, "a") as f_out:
        f_out.create_dataset("mass", (higgs_mass.shape[0], 3), dtype="<f4")
        f_out["mass"][:] = higgs_mass

if __name__ == '__main__':
    MAX_JETS = 15
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("Processing file", input_path, "...")    
    # clean existing h5 file
    h5py.File(output_path, "w").close()
    export_higgs_mass(input_path, output_path)