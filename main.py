# BertNDA: a Model Based on Graph-Bert and Multi-scale Information Fusion for ncRNA-disease Association Prediction
# @Institution: Department of Electronic Information, Xian Jiaotong University, China
# @Author: Zhiwei Ning 
# @Contact: 2193612777@stu.xjtu.edu.cn


import os
import argparse
import random
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, f1_score, auc
import torch.utils.data as Data
from model import BertNDA
from data_train_test import train, test, normalized_laplacian
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from case_study import data_write_csv, topk_index_ouput
from datapreprocess import data_preprocess
from torch.utils.tensorboard import SummaryWriter
import shutil

parser=argparse.ArgumentParser()
def get_args_parser():
    parser=argparse.ArgumentParser('Set Model args')
    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--epochs',default=1,type=int)
    parser.add_argument('--dropout',default=0.05,type=float)
    parser.add_argument('--subgraph_size',default=6,type=int)
    parser.add_argument('--nodes_in_sub',default=6,type=int)
    parser.add_argument('--seed',default=0,type=int)
    parser.add_argument('--device',default="cuda:0",help="device for training /testing")
    parser.add_argument('--dataset_sort',default="dataset1",help="dataset selected")
    parser.add_argument('--wl_max_iter', default=2, type=int)
    parser.add_argument('--batch_size',default=32,type=int,help="train/test batch size")
    parser.add_argument('--loss_alpha',default=1,type=float,help='the alpha in two loss function')
    parser.add_argument('--e_layers',default=8,type=int,help="encoder layers in self-attention pattern")
    parser.add_argument('--n_heads',default=1,type=int,help="multi_heads number in self-attention pattern")
    parser.add_argument('--d_model',default=16,type=int,help="linear connnection layer dimension output")
    parser.add_argument('--d_ff',default=2,type=int)
    parser.add_argument('--topk',default=20,help="the name of Data melos predicted by model")
    return parser.parse_args()


def main(FLAGS):

    # Using cuda
    gpu = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")
    print("Using Cuda") if torch.cuda.is_available() == 1 else print("Using cpu")

    dataset_sort = FLAGS.dataset_sort
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(
        os.path.dirname(current_path) + os.path.sep + ".")

    log_path=father_path+'/log'
    shutil.rmtree(log_path)
    os.mkdir(log_path)
    writer=SummaryWriter(log_path)
    #random seed init
    random.seed(FLAGS.seed)

    ''' before running the next content, make sure the preprocess data/file is already existed (by run datapreprocess.py or uncomment the fllowing code)
    '''
    # data_preprocess(FLAGS)

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
    print("Data is loaded")
    
    lap_matrix = normalized_laplacian(adj_matrix)  # Laplacian Matrix
    for i in range(ngraph):
        lap_matrix[i, i] = 0

    # **************** acquire the positive sample and negative sample origin features
    X_list = []
    for i, temp_list in enumerate(positive_read_list):
        tmp1 = sim_matrix[subgraph_index[temp_list[0]], :].reshape(1, -1)
        tmp2 = sim_matrix[subgraph_index[temp_list[1]], :].reshape(1, -1)
        temp_list3 = np.concatenate((tmp1, tmp2, lap_matrix[temp_list[0], :].reshape(1, -1), lap_matrix[temp_list[1], :].reshape(1, -1),
                                    WL_embed_matrix[temp_list[0], :].reshape(1, -1), WL_embed_matrix[temp_list[1], :].reshape(1, -1)), axis=1)
        X_list.append(torch.tensor(temp_list3))

    np.random.shuffle(negative_read_list)
    for i, temp_list in enumerate(negative_read_list):
        tmp1 = sim_matrix[subgraph_index[temp_list[0]], :].reshape(1, -1)
        tmp2 = sim_matrix[subgraph_index[temp_list[1]], :].reshape(1, -1)
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

    ''' uncomment the fllowing code for case study, TOP40 potiental associted for ['Ovarian cancer','Colon cancer','hsa-mir-373','MALAT1']
    '''
    ##--------------------the case study start------------------------##
    # melo_name_list,k=['Ovarian cancer','Colon cancer','hsa-mir-373','MALAT1'],FLAGS.topk
    ##----make sure the model saved in correct path---##
    # model=torch.load(father_path+'/models/model_fold_0.pth')
    # topk_pred_value_list , topk_pred_name_list = topk_index_ouput(melo_name_list,k,model,ngraph,adj_matrix,sim_matrix,subgraph_index,lap_matrix,WL_embed_matrix,father_path,FLAGS)
    # data_write_csv(father_path+'/models/epoch_'+str(FLAGS.epochs)+'_top_'+str(k)+'_value.csv',topk_pred_value_list)
    # data_write_csv(father_path+'/models/epoch_'+str(FLAGS.epochs)+'_top_'+str(k)+'_name.csv',topk_pred_name_list)
    ##--------------------the case study end------------------------##

    # fold5_cross_valid
    for mod in range(5):
        train_X_list,train_Y_list,test_X_list,test_Y_list = [],[],[],[]
        for i in range(2 * sample_number):
            train_X_list.append(shuffle_X[i]) if i % 5 != mod else test_X_list.append(shuffle_X[i])
            train_Y_list.append(shuffle_Y[i]) if i % 5 != mod else test_Y_list.append(shuffle_Y[i])

        # creat the train and test tensor
        train_X = torch.cat(train_X_list, dim=0).view(-1, ngraph * (2*FLAGS.subgraph_size+4))
        train_Y = torch.cat(train_Y_list, dim=0).view(-1, 1)
        test_X = torch.cat(test_X_list, dim=0).view(-1, ngraph * (2*FLAGS.subgraph_size+4))
        test_Y = torch.cat(test_Y_list, dim=0).view(-1, 1)

        # creat the train and test dataset
        train_dataset = Data.TensorDataset(train_X, train_Y)
        test_dataset = Data.TensorDataset(test_X, test_Y)

        tr_set = Data.DataLoader(
            train_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True)

        tt_set = Data.DataLoader(
            test_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True)

        model = BertNDA(ngraph, FLAGS).to(gpu)

        print("="*10+"fold_"+str(mod+1)+" train start "+"="*10)
        train(tr_set, model, FLAGS, gpu, writer)
        print("="*10+"fold_"+str(mod+1)+" train finished "+"="*10)
        torch.save(model,father_path+'/models/model_fold_'+str(mod)+'.pth')

        print("="*10+"fold_"+str(mod+1)+" test start "+"="*10)
        labels, preds = test(tt_set, model, gpu, FLAGS)
        print("="*10+"fold_"+str(mod+1)+" test end "+"="*10)

        AUC = roc_auc_score(labels, preds)
        precision, recall, _ = precision_recall_curve(labels, preds)
        AUPR = auc(recall, precision)
        pred = np.array([1 if p > 0.5 else 0 for p in preds])
        ACC = accuracy_score(labels, pred)
        F1= f1_score(labels,pred)
        print("fold_"+str(mod)+" test result "+"AUC:{}, AUPR:{}, ACC{}, F1{}".format(AUC, AUPR, ACC,F1))

        np.save(father_path+'/result/BertNDA_'+dataset_sort+'_fold_' +
                str(mod+1)+'_labels.npy', labels)
        np.save(father_path+'/result/BertNDA_'+dataset_sort+'_fold_' +
                str(mod+1)+'_preds.npy', preds)

    print("BertNDA Train and Test Finished")

if __name__=="__main__":
    FLAGS=get_args_parser()
    main(FLAGS)