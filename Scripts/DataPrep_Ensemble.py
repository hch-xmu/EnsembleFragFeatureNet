"""
Prepare the Data from Environment Input into Fragment Output files
For EnsembleModels
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"


import numpy as np
import pickle
import pandas
import matplotlib.pyplot as plt
import time
from sklearn import decomposition as dcmp
import torch.utils.data as utils
import torch

from argparse import ArgumentParser

parser = ArgumentParser(description="Build Files")
parser.add_argument("bound_frags", type=str, help="input - boundfrags_zeros.txt - change")
parser.add_argument("fragments_CID", type=str, help="input - GrandCID.dict - same")
parser.add_argument("fragments_pca", type=str, help="input - simpca_t.matr - same")
parser.add_argument("output_folder", type=str, help="output folder for labels")
parser.add_argument("--pca_dim", type=int, default = 20, help="PCA dimensions describind > 85% variance")
parser.add_argument("--mod_nums", type=int, default = 20, help="Number of Ensemble Models to construct")

args = parser.parse_args()

# ===================
# Load data into memory
# ===================

# Dataframe, (59732, 3),  CID || CanonicalSMILES | IsomericSMILES | Mol
# rowID gives the position in simpca_all
GrandCID = pickle.load(open(args.fragments_CID,"rb"))

# fragment descriptors after pca, (59732, 40)
simpca_all = pickle.load(open(args.fragments_pca,"rb"))

# boundfragments (N, var), var...CIDs of bound fragments per environment (zero as padding)
BoundFrags = np.loadtxt(args.bound_frags, delimiter=',')

# select num of pca features that describe >85%
pca_dim = args.pca_dim

# select num of ensemble models
mod_nums = args.mod_nums

# ===================
# Prepares the fragment CIDs for each model in a N x mod_nums matrix
# ===================

#prepares CID collection for the different models
frags = np.empty((BoundFrags.shape[0], mod_nums),dtype=int)

for j, i in enumerate(BoundFrags):
    fragIDs = i[np.nonzero(i)]
    fragN = fragIDs.shape[0]
    #more than mod_nums bound fragments
    if fragN > mod_nums:
        #incides of selected Fragments - select mod_nums out of fragN
        indx = np.random.choice(range(fragN), mod_nums, replace=False)
    #less than mod_nums bound fragments
    else:
        mult = mod_nums//fragN
        rem = mod_nums%fragN
        indx = np.empty(mod_nums, dtype=int)
        # shuffles the frags i.e. for fragN=3 -> 123321213231
        for i in range(mult):
            if fragN*(i+1) == mod_nums:
                indx[fragN*i:] = np.random.choice(range(fragN), fragN, replace=False)
            else:
                indx[fragN*i:fragN*(i+1)] = np.random.choice(range(fragN), fragN, replace=False)
        if rem > 0:
            indx[-rem:] = np.random.choice(range(fragN), rem, replace=False)
    frags[j,:] = fragIDs[indx]


# ===================
# Matches the CIDs to their corresponding pca_dim values and outputs labels
# ===================

for k in range(mod_nums):
    # (N,) CIDs of bound fragments to environments 
    frags_CID = frags[:,k]
    # (N,) rowIDs in simpca
    frags_rows = np.empty(frags_CID.shape[0])

    #input CID of fragment, output row in simpca_all
    #GrandCID.index is the CID_labeled index of the dataframe
    for i,j in enumerate(frags_CID):
        frags_rows[i] = GrandCID.index.get_loc(j)

    # (N, pca_dim) dominant PCA features of respective compounds
    labels = np.empty((frags_CID.shape[0], pca_dim))

    # select dominant PCA features
    for i,j in enumerate(frags_rows):
        labels[i] = simpca_all[int(j),:pca_dim]
        
    if k <10:
        pickle.dump(labels, open("%s/labels_model0%s.matr" %(args.output_folder, k), "wb"))
    else:
        pickle.dump(labels, open("%s/labels_model%s.matr" %(args.output_folder, k), "wb"))
pickle.dump(frags, open("%s/labels_indices.matr" %(args.output_folder), "wb"))

print('Files created!')