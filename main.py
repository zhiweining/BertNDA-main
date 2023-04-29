import os
import argparse
import random
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, auc
import torch.utils.data as Data
from model import BertNDA
from data_train_test import train, test, normalized_laplacian
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import heapq
from matplotlib.colors import LinearSegmentedColormap
import time
from torch.utils.tensorboard import SummaryWriter
import shutil
from datapreprocess import data_preprocess

parser=argparse.ArgumentParser()

def get_args_parser():
    parser=argparse.ArgumentParser('Set Model args')
    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--epochs',default=20,type=int)
    parser.add_argument('--dropout',default=0.05,type=float)
    parser.add_argument('--subgraph_size',default=6,type=int)
    parser.add_argument('--seed',default=0,type=int)
    parser.add_argument('--device',default="cuda:0",help="device for training /testing")
    parser.add_argument('--dataset_sort',default="dataset1",help="dataset selected")
    parser.add_argument('--wl_max_iter', default=2, type=int)
    return parser.parse_args()

def main(args):
    shutil.rmtree('log')
    os.mkdir('log')
    writer=SummaryWriter('log')
    # Using cuda
    gpu = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using Cuda") if torch.cuda.is_available() == 1 else print("Using cpu")

    dataset_sort = args.dataset_sort
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(
        os.path.dirname(current_path) + os.path.sep + ".")

    # Cofig
    config = {
        'embed_sort': 'laplacian',
        'fold': 0,
        'batch_size': 32,
        'dimension': 4,
        'd_model': 16,
        'n_heads': 1,
        'e_layers': 8,
        'd_ff': 2,
        'nodes_in_sub': 6
    }
    random.seed(args.seed)
    # data_preprocess(args)

    # ---------------------- associated data obtain-------------------#
    adj_matrix=pd.read_csv(father_path+'/data/'+dataset_sort +
                            '/adjmatrix(m_l_d).csv'.format(None, sep=',')).values.astype(np.int)
    sim_matrix = pd.read_csv(father_path+'/data/'+dataset_sort +
                            '/sim_matrix.csv'.format(None, sep=',')).values.astype(np.float64)
    positive_read_list = np.load(
        father_path+'/data/'+dataset_sort+'/positive_ij.npy')
    negative_read_list = np.load(
        father_path+'/data/'+dataset_sort+'/negative_ij.npy')
    # index of neighbor nodes in Subgraph
    subgraph_index = np.load(father_path+'/data/' +
                            dataset_sort+'/subgraph_index.npy')
    WL_embed = np.load(father_path+'/data/'+dataset_sort +
                            '/WL.npy', allow_pickle=True).item()  # WL_Embedding
    ngraph=adj_matrix.shape[0]
    sample_number=int(adj_matrix.sum()/2)
    WL_embed_matrix=np.zeros((ngraph,ngraph))
    for i in range(ngraph):
        WL_embed_matrix[i]=WL_embed[i]
    print("Data is loaded!!")
    
    lap_matrix = normalized_laplacian(adj_matrix)  # Laplacian Matrix
    for i in range(ngraph):
        lap_matrix[i, i] = 0
    print('Laplacian matrix is calculated!!')

    # **************** acquire the positive sample and negative sample origin features
    X_list = []
    for i, temp_list in enumerate(positive_read_list):
        tmp1 = adj_matrix[subgraph_index[temp_list[0]], :].reshape(1, -1)
        tmp2 = adj_matrix[subgraph_index[temp_list[1]], :].reshape(1, -1)
        temp_list3 = np.concatenate((tmp1, tmp2, lap_matrix[temp_list[0], :].reshape(1, -1), lap_matrix[temp_list[1], :].reshape(1, -1),
                                    WL_embed_matrix[temp_list[0], :].reshape(1, -1), WL_embed_matrix[temp_list[1], :].reshape(1, -1)), axis=1)
        X_list.append(torch.tensor(temp_list3))

    np.random.shuffle(negative_read_list)
    for i, temp_list in enumerate(negative_read_list):
        tmp1 = adj_matrix[subgraph_index[temp_list[0]], :].reshape(1, -1)
        tmp2 = adj_matrix[subgraph_index[temp_list[1]], :].reshape(1, -1)
        temp_list3 = np.concatenate((tmp1, tmp2, lap_matrix[temp_list[0], :].reshape(1, -1), lap_matrix[temp_list[1], :].reshape(1, -1),
                                    WL_embed_matrix[temp_list[0], :].reshape(1, -1), WL_embed_matrix[temp_list[1], :].reshape(1, -1)), axis=1)
        X_list.append(torch.tensor(temp_list3))
        if i==len(positive_read_list)-1:
            break

    # reshape the tensor shape
    X = torch.cat((X_list), dim=0)
    X = X.view(2*sample_number, -1)

    # reshape the shape of Y
    Y = torch.cat([torch.from_numpy(np.ones((sample_number, 1))),
                torch.from_numpy(np.zeros((sample_number, 1)))], dim=0)

    # the squence shuffle
    index = [i for i in range(len(X))]
    random.shuffle(index)

    # shuffle the X and the Y
    shuffle_X,shuffle_Y = X[index],Y[index]

    del X, X_list, Y

    # AUC_list,recall_list,precision_list,AUPR_list,ACC_list=[],[],[],[],[]
    # labels_list,preds_list=[],[]
    # fold5_cross_valid
    for mod in range(5):
        train_X_list,train_Y_list,test_X_list,test_Y_list = [],[],[],[]
        for i in range(2 * sample_number):
            train_X_list.append(
                shuffle_X[i]) if i % 5 != mod else test_X_list.append(shuffle_X[i])
            train_Y_list.append(
                shuffle_Y[i]) if i % 5 != mod else test_Y_list.append(shuffle_Y[i])

        # creat the train and test tensor
        train_X = torch.cat(train_X_list, dim=0).view(-1, ngraph * 16)
        train_Y = torch.cat(train_Y_list, dim=0).view(-1, 1)
        test_X = torch.cat(test_X_list, dim=0).view(-1, ngraph * 16)
        test_Y = torch.cat(test_Y_list, dim=0).view(-1, 1)

        # creat the train and test dataset
        train_dataset = Data.TensorDataset(train_X, train_Y)
        test_dataset = Data.TensorDataset(test_X, test_Y)

        tr_set = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True)

        tt_set = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=True)

        model = BertNDA(
            subgraph=config['nodes_in_sub'], seq_len=ngraph, args=args, d_model=config['d_model'],n_heads=config['n_heads'], 
            e_layers=config['e_layers'], d_ff=config['d_ff']).to(gpu)

        print("="*10+"fold_"+str(mod+1)+" train start "+"="*10)
        model_loss_record = train(tr_set, model, args, gpu, writer)
        print("="*10+"fold_"+str(mod+1)+" train finished "+"="*10)

        print("="*10+"fold_"+str(mod+1)+" test start "+"="*10)
        labels, preds = test(tt_set, model, gpu)
        print("="*10+"fold_"+str(mod+1)+" test end "+"="*10)

        AUC = roc_auc_score(labels, preds)
        precision, recall, _ = precision_recall_curve(labels, preds)
        AUPR = auc(recall, precision)
        pred = np.array([1 if p > 0.5 else 0 for p in preds])
        ACC = accuracy_score(labels, pred)
        F1= f1_score(labels,pred)
        print("AUC:{}, AUPR:{}, ACC{}, F1{}".format(AUC, AUPR, ACC,F1))

        np.save(father_path+'/result/Train_result/BertNDA_'+dataset_sort+'_fold' +
                str(mod+1)+'_labels.npy', labels)
        np.save(father_path+'/result/Train_result/BertNDA_'+dataset_sort+'_fold' +
                str(mod+1)+'_preds.npy', preds)

    print("BertNDA Train and Test Finished !! ")

if __name__=="__main__":
    args=get_args_parser()
    main(args)
