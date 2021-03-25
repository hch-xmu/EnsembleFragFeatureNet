"""
Create true labels file (true_labels - (test_N, max_frag))
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

from argparse import ArgumentParser
import numpy as np
import pickle

parser = ArgumentParser(description="Build Files")
parser.add_argument("bound_frags", type=str, help="input - boundfrags_zeros.txt - change")
parser.add_argument("fragments_CID", type=str, help="input - GrandCID.dict - same")
parser.add_argument("outputfolder", type=str, help="outputfolder location")
parser.add_argument("CDK2_indx", type=str, help="outputfile of Step 1")
#parser.add_argument("--test_percent", type=int, default = 15, help="Percentage to be used as test")

args = parser.parse_args()

# ===================
# Load data into memory
# ===================

# Dataframe, (59732, 3),  CID || CanonicalSMILES | IsomericSMILES | Mol
# rowID gives the position in simpca_all
GrandCID = pickle.load(open(args.fragments_CID,"rb"))
# boundfragments (N, var), var...CIDs of bound fragments per environment (zero as padding)
BoundFrags = np.loadtxt(args.bound_frags, delimiter=',')

CDK2_indx = pickle.load(open(args.CDK2_indx,"rb")) #

print('data loaded!')

# ===================
# Replicates the indices used for training
# ===================

#np.random.seed(0)
#ss = np.random.choice(range(Features_all.shape[0]), int(args.test_percent/100*Features_all.shape[0]), replace=False)
ss = np.array(CDK2_indx)

true_la = np.full((ss.shape[0], BoundFrags.shape[1]), -1, dtype=int)

label_CIDs = BoundFrags[ss]

# ===================
# Replaces the CIDs by their row_ID inside the PCA matrix
# ===================

for j, i in enumerate(label_CIDs):
	for l, k in enumerate(i):
		if k == 0:
			break
		else:
			true_la[j,l] = GrandCID.index.get_loc(k)

print(true_la.shape)
pickle.dump(true_la, open('%strue_labels.mtr' %(args.outputfolder), "wb"))


# ===================
# Replicates the indices used for training in Validation set
# ===================

BoundFrags = np.delete(BoundFrags, ss, 0)

np.random.seed(0)
ss = np.random.choice(range(BoundFrags.shape[0]), int(0.05*BoundFrags.shape[0]), replace=False)

true_la = np.full((ss.shape[0], BoundFrags.shape[1]), -1, dtype=int)

label_CIDs = BoundFrags[ss]

# ===================
# Replaces the CIDs by their row_ID inside the PCA matrix
# ===================

for j, i in enumerate(label_CIDs):
	for l, k in enumerate(i):
		if k == 0:
			break
		else:
			true_la[j,l] = GrandCID.index.get_loc(k)

print(true_la.shape)
pickle.dump(true_la, open('%svalidation_labels.mtr' %(args.outputfolder), "wb"))

