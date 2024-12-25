import sys
import h5py
import numpy as np
from utils import get_momentum, get_mass, choose_from_six
# from itertools import combinations
# from atpbar import atpbar



def get_jet_data(file):
    f = h5py.File(file, "r")
    pt = f["INPUTS/Source/pt"][...]
    eta = f["INPUTS/Source/eta"][...]
    phi = f["INPUTS/Source/phi"][...]
    m = f["INPUTS/Source/mass"][...]
    jet_mask = f["INPUTS/Source/MASK"][...]
    b_tag = f["INPUTS/Source/btag"][...]    
    f.close()
    return pt, eta, phi, m, jet_mask, b_tag

def find_min_index(arr, try_first=6):
    s = arr.shape
    arr2 = arr.copy().reshape((s[0], s[1] * s[2]))
    
    # Get the indices of the smallest `try_first` elements along each row
    indices = np.argpartition(arr2, try_first, axis=1)[:, :try_first]
    
    # Create an array to store the sorted indices
    sorted_indices = np.empty_like(indices)
    
    for i in range(s[0]):
        # Extract the actual values for the smallest `try_first` elements
        values = arr2[i, indices[i]]
        
        # Get the sorted order of indices based on the values
        sorted_order = np.argsort(values)
        
        # Reorder the indices based on the sorted order
        sorted_indices[i] = indices[i][sorted_order]
    
    # Compute row and column indices from the sorted 1D indices
    col_idx = sorted_indices % s[1]
    depth_idx = sorted_indices // s[1]
    
    # Stack the indices to get the final output
    ind = np.stack((depth_idx, col_idx), axis=-1)
    
    # Create a boolean mask where arr[..., 0] < arr[..., 1]
    mask = ind[..., 0] < ind[..., 1]

    # Initialize a list to store filtered elements for each slice
    filtered_slices = []

    # Iterate through the slices of the array
    for i in range(ind.shape[0]):
        # Apply the mask to the current slice
        filtered_slice = ind[i][mask[i]]
        # Append the filtered slice to the list
        filtered_slices.append(filtered_slice)

    return filtered_slices

def find_pair(min1, min2, min3, min_ind1, min_ind2, min_ind3, event):
    avail_pairs = []
    for i in min_ind1[event]:
        for j in min_ind2[event]:
            for k in min_ind3[event]:
                ind = [*i, *j, *k]
                if len(set(ind)) == 6:
                    avail_pairs.append(ind)
    min_res = 1E+10
    min_pair = [-1, -1, -1, -1, -1, -1]
    for p in avail_pairs:
        res = min1[event, p[0], p[1]] + min2[event, p[2], p[3]] + min3[event, p[4], p[5]]
        if res < min_res:
            min_res = res
            min_pair = p

    return np.array(min_pair)

def get_match_answer(file): # no sort
    f = h5py.File(file, "r")
    b0 = f["TARGETS/h1/b1"][...]
    b1 = f["TARGETS/h1/b2"][...]
    b2 = f["TARGETS/h2/b1"][...]
    b3 = f["TARGETS/h2/b2"][...]
    b4 = f["TARGETS/h3/b1"][...]
    b5 = f["TARGETS/h3/b2"][...]
    f.close()
    ans = np.array((b0, b1, b2, b3, b4, b5)).transpose()
    ans[:,0:2].sort(axis=1)
    ans[:,2:4].sort(axis=1)
    ans[:,4:6].sort(axis=1)
    return ans

# def sort_b_jet_index(row):
#     arr = row.reshape(3, 2)
#     arr.sort(axis=1)
#     # arr.view("i8, i8").sort(order=["f0"], axis=0)
#     return 

def compare_ans(guess, ans):
    ans_set = {frozenset(row) for row in ans.reshape(3, 2)}
    guess_set = {frozenset(row) for row in guess.reshape(3, 2)}

    compare = sum([1 for p in guess_set if p in ans_set])
    return compare

def compare_jet_list_triHiggs_optimized(pair1, pair2):
    # 將pair1和pair2分別轉換為三個Higgs的集合
    h_true_sets = [{pair1[i], pair1[i+1]} for i in range(0, 6, 2)]
    h_test_sets = [{pair2[i], pair2[i+1]} for i in range(0, 6, 2)]
    
    # 計算匹配的Higgs數量
    nh = sum(1 for h_true in h_true_sets if h_true in h_test_sets)

    return nh

def chi(file, start=0, batch=0):
    pt, eta, phi, m, jet_mask, b_tag = get_jet_data(file)
    if batch != 0:
        pt = pt[start:start+batch]
        eta = eta[start:start+batch]
        phi = phi[start:start+batch]
        m = m[start:start+batch]
        jet_mask = jet_mask[start:start+batch]
        b_tag = b_tag[start:start+batch]
    
    match_ans = get_match_answer(file)
    if batch != 0:
        match_ans = match_ans[start:start+batch]

    # p_all = get_momentum(pt, eta, phi, m)
    # all_combinations = np.array(list(combinations(range(15), 6)))
    # six_combinations = choose_from_six()
    # comb = []
    # for all_c in all_combinations:
    #     for six_c in six_combinations:
    #         jets = np.take(all_c, six_c)
    #         comb.append(jets)
    # comb = np.array(comb)

    # TODO: fix match algorithm (only choose 6)

    matchable = (match_ans != -1).all(axis=1)
    match_ans = match_ans[matchable]
    pt = pt[matchable]
    np.nan_to_num(pt, copy=False)
    eta = eta[matchable]
    phi = phi[matchable]
    m = m[matchable]
    jet_mask = jet_mask[matchable]
    b_tag = b_tag[matchable] & jet_mask

    nbj = b_tag.sum(axis=1)
    nj = jet_mask.sum(axis=1)

    total = matchable.sum()
    total_4b = (nbj == 4).sum()
    total_5b = (nbj == 5).sum()
    total_6b = (nbj >= 6).sum()
    total_6j = (nj == 6).sum()
    total_7j = (nj == 7).sum()
    total_8j = (nj >= 8).sum()
    matched = 0
    matched_456b = np.zeros((3, 4), dtype="<i8")
    matched_678j = np.zeros((3, 4), dtype="<i8")

    
    six_combinations = choose_from_six()

    for i in range(total):
        guess = []
        if nbj[i] >= 6:
            for j in range(MAX_JETS):
                if b_tag[i, j]:
                    guess.append(j)
                    if len(guess) == 6:
                        break
        else:
            budget = 6 - nbj[i]
            for j in range(MAX_JETS):
                if b_tag[i, j]:
                    guess.append(j)
                    if len(guess) == 6:
                        break
                elif budget > 0:
                    budget -= 1
                    guess.append(j)
                    if len(guess) == 6:
                        break

        diff_min = 1E+8
        ind_min = np.ndarray((6,), dtype="<i8")
        for c in six_combinations:
            jets = np.take(guess, c) # = guess[c]
            p = get_momentum(np.take(pt[i], jets), np.take(eta[i], jets), np.take(phi[i], jets), np.take(m[i], jets))
            q = p.reshape(3, 2, 4).sum(axis=1)
            diff = np.abs(get_mass(q) - M).sum() 
            # diff = ((get_mass(q) - M) ** 2).sum()
            if diff < diff_min:
                diff_min = diff
                ind_min = jets
                    
        score = compare_ans(ind_min, match_ans[i])
        score2 = compare_jet_list_triHiggs_optimized(ind_min, match_ans[i])
        if score != score2:
            print("score", score)
            print("score2", score2)
            print("guess", ind_min)
            print("answer", match_ans[i])
            return
        if score > 0:
            matched += 1
        matched_456b[min(nbj[i] - 4, 2), score] += 1
        matched_678j[min(nj[i] - 6, 2), score] += 1


    np.set_printoptions(precision=3)
    print("File length", matchable.shape[0])
    print("Total:", total)
    print("Matched:", matched)

    print("Total 4b:", total_4b)
    print("Total 5b:", total_5b)
    print("Total >=6b:", total_6b)
    print("Matched 456b:")
    print(matched_456b.sum(axis=0)[:,np.newaxis].shape)
    matched_456b = np.concatenate((matched_456b, matched_456b.sum(axis=0)[np.newaxis,:]))
    print(matched_456b)
    print("Matched 456b effeciency:")
    matched_456b_effciency = matched_456b[:,::-1] / np.array([total_4b, total_5b, total_6b, total])[:,np.newaxis]
    print(matched_456b_effciency)
    print("Higgs effeciency:")
    print((matched_456b_effciency * np.array([3, 2, 1, 0])[np.newaxis,:]).sum(axis=1) / 3)
    

    print("Total 6j:", total_6j)
    print("Total 7j:", total_7j)
    print("Total >=8j:", total_8j)
    print("Matched 678j:")
    print(matched_678j.sum(axis=0)[:,np.newaxis].shape)
    matched_678j = np.concatenate((matched_678j, matched_678j.sum(axis=0)[np.newaxis,:]))
    print(matched_678j)
    print("Matched 678j effeciency:")
    matched_678j_effciency = matched_678j[:,::-1] / np.array([total_6j, total_7j, total_8j, total])[:,np.newaxis]
    print(matched_678j_effciency)
    print("Higgs effeciency:")
    print((matched_678j_effciency * np.array([3, 2, 1, 0])[np.newaxis,:]).sum(axis=1) / 3)
    

if __name__ == '__main__':
    # M = (119, 115, 111) or (120, 115, 110)
    MAX_JETS = 15
    M = np.array((118.37, 114.86, 110.86))
    h5_file = sys.argv[1]
    if len(sys.argv) == 2:
        chi(h5_file)
    elif len(sys.argv) == 4:
        start = int(sys.argv[2])
        batch = int(sys.argv[3])
        chi(h5_file, start, batch)