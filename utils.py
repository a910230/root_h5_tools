import numpy as np
import awkward as ak
import h5py
from itertools import combinations

def ak_to_np(ak_arr, max_jets=0):
    if max_jets == 0:
        max_jets = np.max(ak.num(ak_arr))
    return ak.to_numpy(ak.pad_none(ak_arr, max_jets)[...,:max_jets])

def get_deltaR(eta1, phi1, eta2, phi2):
    dEta = eta1[:,:,np.newaxis] - eta2[:,np.newaxis,:]
    dPhi = np.abs(phi1[:,:,np.newaxis] - phi2[:,np.newaxis,:])
    dPhi = np.where(dPhi > np.pi, 2 * np.pi - dPhi, dPhi)
    dR = (dPhi ** 2 + dEta ** 2) ** 0.5

    return dR

def get_momentum(pt, eta, phi, m):
    p = pt * np.cosh(eta)
    p0 = np.sqrt(p ** 2 + m ** 2)
    p1 = pt * np.cos(phi)
    p2 = pt * np.sin(phi)
    p3 = pt * np.sinh(eta)

    return np.moveaxis(np.array([p0, p1, p2, p3]), 0, -1)

def get_mass(p):
    ps = np.maximum(0, p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2)
    if (ps == 0).any():
        print(p[ps == 0])
    return np.sqrt(ps)

def sort_b_jet_index(b_jet_index):
    for row in b_jet_index:
        row[row==-1] = 16
        arr = row.reshape(3, 2)
        arr.sort(axis=1)
        arr.view("i8, i8").sort(order=["f1"], axis=0)
        arg = np.where(arr[:,1] == 16)[0]
        arg = arg[0] if arg.size else 3
        arr[:arg].view("i8, i8").sort(order=["f0"], axis=0)
        arr[arg:].view("i8, i8").sort(order=["f0"], axis=0)
        row[row==16] = -1
    return

def choose_from_six():
    # Example list of 6 numbers
    numbers = [0, 1, 2, 3, 4, 5]

    # Step 1: Generate all combinations of pairs
    pairs = list(combinations(numbers, 2))

    # Step 2: Generate all possible groupings of 3 pairs
    result = []

    # Generate combinations of pairs while ensuring unique selection of numbers
    for group in combinations(pairs, 3):
        # Flatten the group to check for unique numbers
        flat = [num for pair in group for num in pair]
        # Check if we have exactly 6 unique numbers
        if len(set(flat)) == 6:
            result.append(group)

    # Convert the result to a more readable format
    formatted_result = np.array([(a, b, c, d, e, f) for ((a, b), (c, d), (e, f)) in result])
    return formatted_result

def get_jet_data(root_file, event_mask, max_jets, n_start, n_end):
    # load jet data
    jet_PT = ak_to_np(root_file["Jet.PT"].array(entry_start=n_start, entry_stop=n_end), max_jets)[event_mask]
    jet_Eta = ak_to_np(root_file["Jet.Eta"].array(entry_start=n_start, entry_stop=n_end), max_jets)[event_mask]
    jet_Phi = ak_to_np(root_file["Jet.Phi"].array(entry_start=n_start, entry_stop=n_end), max_jets)[event_mask]
    jet_Mass = ak_to_np(root_file["Jet.Mass"].array(entry_start=n_start, entry_stop=n_end), max_jets)[event_mask]
    jet_BTag = ak_to_np(root_file["Jet.BTag"].array(entry_start=n_start, entry_stop=n_end), max_jets)[event_mask]

    return jet_PT, jet_Eta, jet_Phi, jet_Mass, jet_BTag

def h5py_dataset_iterator(g, prefix=''):
    for key, item in g.items():
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_dataset_iterator(item, path)