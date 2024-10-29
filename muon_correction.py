import numpy as np

def get_final_mu_index(pid, d1):
    final_mu_index = set()
    for j in range(len(pid)):
        if abs(pid[j]) == 13:
            mu = j
            while d1[mu] > mu and abs(pid[d1[mu]]) == 13:
                mu = d1[mu]
            if abs(pid[d1[mu]]) != 13:
                final_mu_index.add(mu)
    return np.array(list(final_mu_index))

def get_deltaR(eta1, phi1, eta2, phi2): # different from what in utils.py
    dEta = eta1[:,np.newaxis] - eta2[np.newaxis,:]
    dPhi = np.abs(phi1[:,np.newaxis] - phi2[np.newaxis,:])
    dPhi = np.where(dPhi > np.pi, 2 * np.pi - dPhi, dPhi)
    dR = (dPhi ** 2 + dEta ** 2) ** 0.5

    return dR

def get_particle_data(root_file, event_mask, n_start, n_end): # different from what in b_jet_match.py
    # load particle data
    pid = root_file["Particle.PID"].array(entry_start=n_start, entry_stop=n_end)[event_mask]
    d1 = root_file["Particle.D1"].array(entry_start=n_start, entry_stop=n_end)[event_mask]
    d2 = root_file["Particle.D2"].array(entry_start=n_start, entry_stop=n_end)[event_mask]
    eta = root_file["Particle.Eta"].array(entry_start=n_start, entry_stop=n_end)[event_mask]
    phi = root_file["Particle.Phi"].array(entry_start=n_start, entry_stop=n_end)[event_mask]
    pt = root_file["Particle.PT"].array(entry_start=n_start, entry_stop=n_end)[event_mask]
    mass = root_file["Particle.Mass"].array(entry_start=n_start, entry_stop=n_end)[event_mask]

    return pid, d1, d2, eta, phi, pt, mass

def get_momentum(pt, eta, phi, m): # different
    p = pt * np.cosh(eta)
    p0 = np.sqrt(p ** 2 + m ** 2)
    p1 = pt * np.cos(phi)
    p2 = pt * np.sin(phi)
    p3 = pt * np.sinh(eta)

    return np.array([p0, p1, p2, p3]).transpose()

def get_pt_eta_phi_m(p):
    return
    p = pt * np.cosh(eta)
    p0 = np.sqrt(p ** 2 + m ** 2)
    p1 = pt * np.cos(phi)
    p2 = pt * np.sin(phi)
    p3 = pt * np.sinh(eta)

    return np.moveaxis(np.array([p0, p1, p2, p3]), 0, -1)

def muon_correction(root_file, event_mask, b_jet_index, n_start, n_end, jet_pt, jet_Eta, jet_Phi, jet_mass):
    pid, d1, d2, eta, phi, pt, mass = get_particle_data(root_file, event_mask, n_start, n_end) # masked
    
    n_selected = event_mask.sum()

    for i in range(n_selected):
        final_mu_index = get_final_mu_index(pid[i], d1[i])
        if (final_mu_index.size == 0): continue
        muon_Eta = eta[i][final_mu_index]
        muon_Phi = phi[i][final_mu_index]
        avai_jet_Eta = jet_Eta[i][b_jet_index[i]]
        avai_jet_Phi = jet_Phi[i][b_jet_index[i]]
        dR = get_deltaR(avai_jet_Eta, avai_jet_Phi, muon_Eta, muon_Phi) # 6 * n_muon
        muon_pt = pt[i][final_mu_index]
        muon_mass = mass[i][final_mu_index]
        muon_dR_min = np.minimum(0.4, 0.04 + 10 / muon_pt)
        avai_muon = (dR < muon_dR_min).filled(False)
        double_assigned_muon = np.where(avai_muon.sum(axis=0) > 1)[0]
        for mu in double_assigned_muon:
            choice = dR[:, mu].argmin()
            avai_muon[:, mu] = False
            avai_muon[choice, mu] = True
        for i_jet, i_mu in np.transpose(np.where(avai_muon)):
            jet_p = get_momentum(jet_pt[i][i_jet], jet_Eta[i][i_jet], jet_Phi[i][i_jet], jet_mass[i][i_jet])
            print(jet_p.shape)
            jet_p += get_momentum(muon_pt[i_mu], muon_Eta[i_mu], muon_Phi[i_mu], muon_mass[i_mu])
            get_pt_eta_phi_m(jet_p)


    # return jet_pt, jet_eta, jet_phi, jet_mass