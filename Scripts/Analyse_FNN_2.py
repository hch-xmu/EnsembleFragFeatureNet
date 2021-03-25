"""
Analyse the trained FNN Models for the clustered DataSet
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"


from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from FNN.models.model011 import FeatureResNeXt

parser = ArgumentParser(description="Build Files")
parser.add_argument("state_dict", type=str, help="trained weights file of model - .pytorch")
parser.add_argument("ensmodel_name", type=str, help="ensmodel_name")
parser.add_argument("environments", type=str, help="input - property.pvar")
parser.add_argument("labels", type=str, help="input - labels.matr")
parser.add_argument("fragments_pca", type=str, help="input - simpca_t.matr - same")
parser.add_argument("true_labels", type=str, help="input - true_labels.matr - changes depending on test percentage")
parser.add_argument("cdk2_indx", type=str, help="input - CDK2indxlistCDK2.mtr")
parser.add_argument("--kdTree", type=bool, default=False, help="use kdTree")
parser.add_argument("--cardinality", type=int, default=8, help="Cardinality")
parser.add_argument("--depth", type=int, default=65, help="Depth of the network")
parser.add_argument("--nlabels", type=int, default=20, help="Number of pca dimensions")
parser.add_argument("--batch_size", "-bs", type=int, default=128, help="Batchsize used originally")

args = parser.parse_args()

# ===================
# Load weights into network
# ===================

#initialise net with parameters
net = FeatureResNeXt(args.cardinality, args.depth, args.nlabels, 8) #cardinality, depth, nlabels, base_width
loaded_state_dict = torch.load(args.state_dict)
#update model with weights from trained model
temp = {}
for key, val in list(loaded_state_dict.items()):
    # parsing keys for ignoring 'module.' in keys
    temp[key[7:]] = val
loaded_state_dict = temp
net.load_state_dict(loaded_state_dict)
net = net.eval()
# ===================
# Load data into memory
# ===================

Features_all = pickle.load(open(args.environments, "rb")) # N x 480
labels_all = pickle.load(open(args.labels, "rb")) # N x 20
true_la = pickle.load(open(args.true_labels, "rb")) # testN x max_frag

# ===================
# Normalises data
# ===================

#normalises the feature vectors
mean = [Features_all[:,i].mean() for i in range(480)]
std = [Features_all[:,i].std() for i in range(480)]
for i in range (480):
    if std[i] != 0:
        Features_all[:,i] = (Features_all[:,i] - mean[i])/std[i]
    else:
        Features_all[:,i] = Features_all[:,i]
Features_all = np.resize(Features_all, (Features_all.shape[0], 1, 6, 80))

# ===================
# Select List of Indices for Validation - Testing 
# ===================

ss = np.array(pickle.load(open(args.cdk2_indx,"rb")))

# ===================
# Split training and test set 
# ===================

#select test tensors
test_tensor_x = torch.stack([torch.Tensor(i) for i in Features_all[ss,:,:,:]]) # transform to torch tensors
test_tensor_y = torch.stack([torch.Tensor(i) for i in true_la]) #pca_row indices of possible fragments (test_N, max_frag)
test_data = utils.TensorDataset(test_tensor_x,test_tensor_y) # create your datset


#CHANGE HERE DEPENDING ON GPU USAGE
test_loader = utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)


# ===================
# Load FragmentSpace and construct K-d Tree
# ===================

#load PCA matrix
simpca_t = pickle.load(open(args.fragments_pca,"rb"))
simpca_t = simpca_t[:,:args.nlabels] # cmpds x 20 dimensions (59732, 20)
#for i in range(args.nlabels):
#    simpca_t[:,i] = (simpca_t[:,i] - mean_o[i])/ std_o[i]
    
#convert to tensor
dim_t = torch.stack([torch.Tensor(i) for i in simpca_t])

sin_values =  pickle.load(open("/home/masv/Documents/EnsFragFeatureMScripts/PCA/singular_values_.matr","rb"))
t_weight = torch.Tensor(sin_values[:20]/sin_values[:20].sum())

def weighted_mse_loss(t_input, t_target):
    return torch.sum(t_weight * (t_input - t_target) ** 2)

# ===================
# Run Testdata through the network and collect 10 nearest neighbours 
# ===================

summary = np.empty((test_tensor_x.shape[0],2,10)) # 3635 x [indx1, indx2, ...], [dist1, dist2,...]
minhits = np.empty(test_tensor_x.shape[0]) # 3635 x [indx1, indx2, ...], [dist1, dist2,...]

if args.kdTree:
    print(args.kdTree, 'kdTree')
    tree = KDTree(simpca_t, leaf_size=20)
    
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = net(data)
        dist, indx = tree.query(pred.detach(), k=10)
        if data.shape[0] < test_loader.batch_size:
            summary[batch_idx*test_loader.batch_size:,0,:] = indx
            summary[batch_idx*test_loader.batch_size:,1,:] = dist
        else:
            summary[batch_idx*data.shape[0]:batch_idx*data.shape[0]+data.shape[0],0,:] = indx
            summary[batch_idx*data.shape[0]:batch_idx*data.shape[0]+data.shape[0],1,:] = dist

        for j, row in enumerate(target):
            row = row[row!=-1]
            temp = np.empty(row.shape[0])
            for l, pca_indx in enumerate(row):
                tens = torch.Tensor(simpca_t[int(pca_indx), :20])
                temp[l] = F.mse_loss(pred[j], tens)
            minhits[batch_idx*test_loader.batch_size+j] = temp.min()    
            
else:
    print('LF')
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = net(data)
        #for i in range(20):
        #    pred[:,i] = pred[:,i] * std_o[i] + mean_o[i]

        for j,i in enumerate(data):
            
            temp = np.where(np.abs(simpca_t[:,0]-float(pred[j][0]))<0.1)[0]
            if temp.shape[0] < 10: #exception for when a prediction item is in the noise
                temp = np.where(np.abs(simpca_t[:,0]-float(pred[j][0]))<0.5)[0]
            if temp.shape[0] < 10: #exception for when a prediction item is in the noise
                temp = np.array(range(simpca_t.shape[0]))
            tmp_mse = np.empty(temp.shape[0])

            for k,l in enumerate(temp):
                tmp_mse[k]=weighted_mse_loss(pred[j], dim_t[l]) 
            for m in range(10):
                indx = tmp_mse.argmin()
                summary[batch_idx*test_loader.batch_size+j,0,m] = temp[indx]
                summary[batch_idx*test_loader.batch_size+j,1,m] = tmp_mse.min()
                tmp_mse[tmp_mse.argmin()]=100

        for j, row in enumerate(target):
            row = row[row!=-1]
            temp = np.empty(row.shape[0])
            for l, pca_indx in enumerate(row):
                tens = torch.Tensor(simpca_t[int(pca_indx), :20])
                #tens = (tens - torch.Tensor(mean_o)) / torch.Tensor(std_o)
                #loss function
                temp[l] = weighted_mse_loss(pred[j], tens)
            minhits[batch_idx*test_loader.batch_size+j] = temp.min()


pickle.dump(summary, open('CLU.STER_FinalResults/CLU.STER.kdTsummary_%s.mtr' %(args.ensmodel_name), "wb"))
pickle.dump(minhits, open('CLU.STER_FinalResults/CLU.STER.kdTminhits_%s.mtr' %(args.ensmodel_name), "wb"))

# ===================
# Match the predictions with true labels
# summary - (test_N, 2, 10)
# true_labels - (test_N, max_frag)
# output -> count (test_N, hit_position)
# ===================

def evaluate_hits(summary_mat, true_labels):
    count = np.full((summary_mat.shape[0], 2, 10), -1, dtype=int)
    for j, i in enumerate(summary_mat[:,0]):
        xy, i_indx,_ = np.intersect1d(i, true_labels[j], return_indices=True)
        if len(xy) >= 1:
            for l, k in enumerate(xy):
                count[j,0,l] = int(xy[l])
                count[j,1,l] = i_indx[l]
        elif len(xy) == 0:
            continue
        else: 
            print('Too many matches, add more count space')
    return count

hitcount = evaluate_hits(summary, true_la)

pickle.dump(hitcount, open('CLU.STER_FinalResults/CLU.STER.kdThitcount_%s.mtr' %(args.ensmodel_name), "wb"))

sums = 0
for i in range(10):
    print('Hit for proximity %s: ' %(i), hitcount[:,1][hitcount[:,1]==i].shape[0])
    sums+=hitcount[:,1][hitcount[:,1]==i].shape[0]
indiv = 0
for i in hitcount[:,0,0]:
    if i != -1:
        indiv +=1
print('Environments matched:', indiv, 'All hits:', sums, 'All Environments:', summary.shape[0])