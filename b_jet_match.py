import numpy as np
from utils import get_deltaR, get_jet_data

def get_particle_data(root_file, event_mask, n_start, n_end):
    # load particle data
    pid = root_file["Particle.PID"].array(entry_start=n_start, entry_stop=n_end)[event_mask]
    d1 = root_file["Particle.D1"].array(entry_start=n_start, entry_stop=n_end)[event_mask]
    d2 = root_file["Particle.D2"].array(entry_start=n_start, entry_stop=n_end)[event_mask]
    eta = root_file["Particle.Eta"].array(entry_start=n_start, entry_stop=n_end)[event_mask]
    phi = root_file["Particle.Phi"].array(entry_start=n_start, entry_stop=n_end)[event_mask]

    return pid, d1, d2, eta, phi

def get_final_h_index(pid, d1):
    final_h_index = set()
    for j in range(len(pid)):
        if pid[j] == 25:
            h = j
            while d1[h] > h and pid[d1[h]] == 25:
                h = d1[h]
            if pid[d1[h]] != 25:
                final_h_index.add(h)

    return list(final_h_index)

def get_final_b_index(pid, d1, d2):
    final_h_index = get_final_h_index(pid, d1)
    final_b_index = []
    for h in final_h_index:
        # b quark
        b = d1[h]
        while pid[d1[b]] == 5:
            b = d1[b]
        final_b_index.append(b)
        # b~ quark
        b = d2[h]
        while pid[d1[b]] == -5:
            b = d1[b]
        final_b_index.append(b)

    return final_b_index

def get_b_jet_match(dR, jet_mask):
    match_mask = ((dR < 0.4) & np.expand_dims(jet_mask, axis=-1)) # n * 15 * 6

    all_b_jet_match = []
    for event in match_mask:
        event_sum = event.sum(axis=0)
        b_jet_match = np.full((6,), -1)

        while 1 in event_sum:
            i = np.where(event_sum == 1)[0][0]
            b_jet_match[i] = np.argmax(event[:,i])
            event[b_jet_match[i],:] = False
            event_sum = event.sum(axis=0)
        all_b_jet_match.append(b_jet_match)

    return np.array(all_b_jet_match)


def get_b_jet_index(root_file, event_mask, jet_mask, max_jets, n_start, n_end):
    pid, d1, d2, eta, phi = get_particle_data(root_file, event_mask, n_start, n_end) # masked
    _, jet_Eta, jet_Phi, _, _ = get_jet_data(root_file, event_mask, max_jets, n_start, n_end) # masked
    
    n_selected = event_mask.sum()
    b_jet_index = np.full((n_selected, 6), -1)

    quark_Eta = []
    quark_Phi = []
    for i in range(n_selected):
        final_b_index = get_final_b_index(pid[i], d1[i], d2[i])
        quark_Eta.append(np.take(eta[i], final_b_index))
        quark_Phi.append(np.take(phi[i], final_b_index))
    quark_Eta = np.array(quark_Eta)
    quark_Phi = np.array(quark_Phi)
    dR = get_deltaR(jet_Eta, jet_Phi, quark_Eta, quark_Phi) # n * 15 * 6
    b_jet_index = get_b_jet_match(dR, jet_mask[event_mask])
    
    return b_jet_index