import numpy as np
from utils import ak_to_np

def get_event_and_jet_mask(root_file, max_jets, mode):
    pt = ak_to_np(root_file["Jet.PT"].array(), max_jets)
    eta = ak_to_np(root_file["Jet.Eta"].array(), max_jets)
    btag = ak_to_np(root_file["Jet.BTag"].array(), max_jets)

    if mode == 1:
        selected_jet = (pt > 25) & (np.abs(eta) < 2.5)
        selected_event = (selected_jet.sum(axis=1) >= 6)
    elif mode == 2:
        selected_jet = (pt > 25) & (np.abs(eta) < 2.5) & btag
        selected_event = (selected_jet.sum(axis=1) >= 6)
    elif mode == 3:
        selected_jet = (pt > 20) & (np.abs(eta) < 2.5)
        selected_event_1 = (selected_jet.sum(axis=1) >= 6)
        selected_jet_2 = selected_jet & (pt > 40)
        selected_event_2 = (selected_jet_2.sum(axis=1) >= 4)
        selected_jet_3 = selected_jet & btag
        selected_event_3 = (selected_jet_3.sum(axis=1) >= 4)
        selected_event = selected_event_1 & selected_event_2 & selected_event_3
        print(selected_event_1.shape)
    else:
        raise ValueError("Preselect mode can only be 1, 2 or 3")

    return selected_event, selected_jet