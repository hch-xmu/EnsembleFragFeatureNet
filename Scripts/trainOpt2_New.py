# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on FEATURE Vectors

"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

import argparse
import os
import json
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.model011 import FeatureResNeXt
import torch.utils.data as utils
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on FEATURE Vectors', formatter_class= argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('features_data_path', type=str, help='Root for Features Dict.')
    parser.add_argument('label_data_path', type=str, help='Root for Labels.')
    parser.add_argument("true_labels", type=str, help="input - true_labels.mtr")
    parser.add_argument("validation_labels", type=str, help="input - validation_labels.mtr")
    parser.add_argument("fragments_pca", type=str, help="input - simpca_t.matr - same")
    parser.add_argument("CDK2_indx", type=str, help="outputfile of Step 1")

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The Learning Rate.') #def 0.1
    parser.add_argument('--momentum', '-m', type=float, default=0.3, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[120, 135], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./outputmodels', help='Folder to save checkpoints.')
    # Architecture
    parser.add_argument('--depth', type=int, default=92, help='Model depth - Multiple of 3*no_stages (5, 10)')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group) in the DxD convolutionary layer.')
    parser.add_argument('--base_width', type=int, default=8, help='Number of channels in each group. Output of the first convolution layer. Modify stages in model.py')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor between every block')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=8, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--name', type=str, default='model', help='Name your model')
    parser.add_argument('--checkpoint', type=bool, default=False, help='Load from checkpoint?')
    args = parser.parse_args()

    # Init logger
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    log = open(os.path.join(args.save, args.name + '_log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')

    # Calculate number of epochs wrt batch size
    args.epochs = args.epochs * args.batch_size // args.batch_size
    args.schedule = [x * args.batch_size // args.batch_size for x in args.schedule]
    
    #initialise Dataset
    Features_all = pickle.load(open(args.features_data_path, "rb"))
    labels_all = pickle.load(open(args.label_data_path, "rb"))
    nlabels = labels_all.shape[1]
    simpca_t = pickle.load(open(args.fragments_pca, "rb"))
    true_labels = pickle.load(open(args.true_labels, "rb"))
    val_labels = pickle.load(open(args.validation_labels, "rb"))

    CDK2_indx = pickle.load(open(args.CDK2_indx,"rb")) 

    
    #normalise FEATURE Vectors
    mean = [Features_all[:,i].mean() for i in range(480)]
    std = [Features_all[:,i].std() for i in range(480)]
    for i in range (480):
        if std[i] != 0:
            Features_all[:,i] = (Features_all[:,i] - mean[i])/std[i]
        else:
            Features_all[:,i] = Features_all[:,i]
    Features_all = np.resize(Features_all, (Features_all.shape[0], 1, 6, 80))
    
    
#     Features_all = np.resize(Features_all, (Features_all.shape[0], 1, 6, 80))
#     mean = [Features_all[:,:,:,i].mean() for i in range(80)]
#     std = [Features_all[:,:,:,i].std() for i in range(80)]
#     for i in range (80):
#         if std[i] != 0:
#             Features_all[:,:,:,i] = (Features_all[:,:,:,i] - mean[i])/std[i]
#         else:
#             Features_all[:,:,:,i] = Features_all[:,:,:,i]

    ss = np.array(CDK2_indx)    
    
    #select test tensors
    test_tensor_x = torch.stack([torch.Tensor(i) for i in Features_all[ss,:,:,:]]) # transform to torch tensors
    test_tensor_y = torch.stack([torch.Tensor(i) for i in true_labels]) #pca_row indices of possible fragments (test_N, max_frag)
    
    Features_all = np.delete(Features_all, ss, 0)
    labels_all = np.delete(labels_all, ss, 0)
    
    #for validation
    np.random.seed(0)
    ss = np.random.choice(range(Features_all.shape[0]), int(0.05*Features_all.shape[0]), replace=False)
    
    validation_tensor_x = torch.stack([torch.Tensor(i) for i in Features_all[ss,:,:,:]]) 
    validation_tensor_y = torch.stack([torch.Tensor(i) for i in val_labels]) 
    
    #select train tensors                             
    train_tensor_x = torch.stack([torch.Tensor(i) for i in np.delete(Features_all, ss, 0)]) # transform to torch tensors
    train_tensor_y = torch.stack([torch.Tensor(i) for i in np.delete(labels_all, ss, 0)])                               
    
    train_data = utils.TensorDataset(train_tensor_x,train_tensor_y) # create your datset
    test_data = utils.TensorDataset(test_tensor_x,test_tensor_y) # create your datset
    val_data = utils.TensorDataset(validation_tensor_x,validation_tensor_y) # create your datset

    #test code                                  
    #train_data,_ = torch.utils.data.random_split(train_data, [1024,15450-1024])
    #test_data,_ = torch.utils.data.random_split(test_data, [256,2726-256])   

    train_loader = utils.DataLoader(train_data,batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True) # create your dataloader
    test_loader = utils.DataLoader(test_data,batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)
    val_loader = utils.DataLoader(val_data,batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)        

    # Init model, criterion, and optimizer
    net = FeatureResNeXt(args.cardinality, args.depth, nlabels, args.base_width, args.widen_factor)
    print(net)
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    # train function (forward, backward, update)
    def train():
        net.train()
        loss_avg = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            # forward
            output = net(data)

            # backward
            optimizer.zero_grad()
#LOSS FUNCTION
            loss = F.mse_loss(output, target)
            print('train - loss',loss)
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
            print('train - loss avg', loss_avg)
        state['train_loss'] = loss_avg

    # val function (forward only)
    def val():
        net.eval()
        loss_avg = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
            # forward
            output = net(data)
            # find best MSE to best label
            temp_loss = 0
            for j, row in enumerate(target):
                row = row[row!=-1]
                temp = np.empty(row.shape[0])
                for l, pca_indx in enumerate(row):
                    tens = torch.Tensor(simpca_t[int(pca_indx), :20])
                    tens = tens.cuda()
                    #loss function
                    temp[l] = F.mse_loss(output[j], tens)
                temp_loss += temp.min()
            loss = temp_loss/target.shape[0]
            
            print('val - loss', loss)

            # val loss average
            loss_avg += float(loss)

        state['val_loss'] = loss_avg / len(val_loader)
        print('val_loss_avg', loss_avg / len(val_loader))        
        
        

    # test function (forward only)
    def test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
            # forward
            output = net(data)
            # find best MSE to best label
            temp_loss = 0
            for j, row in enumerate(target):
                row = row[row!=-1]
                temp = np.empty(row.shape[0])
                for l, pca_indx in enumerate(row):
                    tens = torch.Tensor(simpca_t[int(pca_indx), :20])
                    tens = tens.cuda()
                    #loss function
                    temp[l] = F.mse_loss(output[j], tens)
                temp_loss += temp.min()
            loss = temp_loss/target.shape[0]
            
            print('test - loss', loss)

            # test loss average
            loss_avg += float(loss)

        state['test_loss'] = loss_avg / len(test_loader)
        print('test_loss_avg', loss_avg / len(test_loader))


    # Main loop
    best_accuracy = 100.0
    
    if args.checkpoint == True:
        loaded_state_dict = torch.load('Data/KIN.ALL/ModelOutput/model00_cp.pytorch')        
#         temp = {}
#         for key, val in list(loaded_state_dict.items()):
#             temp[key[7:]] = val
#         loaded_state_dict = temp
        net.load_state_dict(loaded_state_dict)
        
    for epoch in range(args.epochs):
        #updates learning rate 
        if epoch in args.schedule:
            state['learning_rate'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']
                
        state['epoch'] = epoch
        train()
        val()
        test()
        
        #decide whether to save the model
        if state['test_loss'] < best_accuracy:
            best_accuracy = state['test_loss']
            torch.save(net.state_dict(), os.path.join(args.save, args.name + '.pytorch'))
        if epoch%20 == 0 and epoch > 0:
            torch.save(net.state_dict(), os.path.join(args.save, args.name + '_cp.pytorch'))
            #write in log file
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print("Best accuracy: %f" % best_accuracy)
        
    torch.save(net.state_dict(), os.path.join(args.save, args.name + '_final.pytorch'))

    log.close()
