# BertNDA: a Model Based on Graph-Bert and Multi-scale Information Fusion for ncRNA-disease Association Prediction
# @Institution: Department of Electronic Information, Xian Jiaotong University, China
# @Author: Zhiwei Ning 
# @Contact: 2193612777@stu.xjtu.edu.cn


import os
from random import *
import numpy as np
import pandas as pd
from pylab import *
from WLNodeEmbedding import MethodWLNodeColoring
import argparse

current_path = os.path.abspath(__file__)
father_path = os.path.abspath(
    os.path.dirname(current_path) + os.path.sep + ".")

parser=argparse.ArgumentParser()
def get_args_parser():
    parser=argparse.ArgumentParser('Set Model args')
    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--epochs',default=20,type=int)
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

def data_preprocess(FLAGS):
    dataset_sort = FLAGS.dataset_sort
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(
        os.path.dirname(current_path) + os.path.sep + ".")
    MLD_matrix=pd.read_csv(father_path+'/data/'+dataset_sort +
                            '/adjmatrix(m_l_d).csv'.format(None, sep=',')).values.astype(np.int)
    mir_num=len(open(father_path+'/data/' + dataset_sort + '/mir.txt',encoding='gbk').read().strip('\n').split('\n'))
    lnc_num = len(open(father_path+'/data/' + dataset_sort + '/lnc.txt',encoding='gbk').read().strip('\n').split('\n'))
    dis_num = len(open(father_path+'/data/' + dataset_sort + '/dis.txt',encoding='gbk').read().strip('\n').split('\n'))
    ngraph=MLD_matrix.shape[0]
    #------ Create the disease Similarity -------#
    disease_similarity(MLD_matrix, mir_num, lnc_num, dis_num, dataset_sort)

    #------- Create the ncRNA Similarity -------#
    miRNA_func_similarity(MLD_matrix,mir_num,lnc_num,dis_num, dataset_sort)
    lncRNA_func_similarity(MLD_matrix,mir_num,lnc_num,dis_num, dataset_sort)

    #------- Create the MLD Similarity Matrix -----#
    final_sim_matrix(MLD_matrix,mir_num,lnc_num,dis_num, ngraph, dataset_sort)
    #------- Create the Intimacy Matrix S -----#
    Diag=np.sum(MLD_matrix,axis=1)
    D=np.diag(Diag)
    for i in range(D.shape[0]):
        if D[i][i]==0:
            D[i][i]=1#avoid division by zero operator

    tmp_a=0.15
    score_matrix=np.zeros((D.shape[0],D.shape[1]))
    S=tmp_a*np.linalg.inv(np.eye(mir_num+lnc_num+dis_num)-(1-tmp_a)*(np.dot(MLD_matrix,np.linalg.inv(D))))
    print('intimacr_matrix is create')
    one_position=np.where(MLD_matrix==1)
    pair_number=len(one_position[0])
    for i in range(pair_number):
        score_matrix[one_position[0][i]][one_position[1][i]]=S[one_position[0][i]][one_position[1][i]]
    df=pd.DataFrame(score_matrix)
    df.to_csv(father_path+'/data/'+dataset_sort+'/sim_score.csv',index= False, header= True)

    #------------- Get the index of positive/negative Sample -------------#
    tmp_positive_position=np.where(MLD_matrix==1)
    tmp_negative_position=np.where(MLD_matrix==0)
    len_tmp_pos=len(tmp_positive_position[0])
    len_tmp_neg=len(tmp_negative_position[1])
    positive_position,negative_position=[],[]
    for i in range(len_tmp_pos):
        if tmp_positive_position[0][i]<tmp_positive_position[1][i]:
            positive_position.append([tmp_positive_position[0][i],tmp_positive_position[1][i]])
    for i in range(len_tmp_neg):
        if tmp_negative_position[0][i]<tmp_negative_position[1][i]:
            if (tmp_negative_position[0][i]<mir_num and mir_num<=tmp_negative_position[1][i]) or (tmp_negative_position[0][i]<mir_num+lnc_num and mir_num+lnc_num<=tmp_negative_position[1][i]):
                negative_position.append([tmp_negative_position[0][i],tmp_negative_position[1][i]])
    #transfer the list into array
    positive_ij=np.array(positive_position,dtype=int)
    tmp_negative=np.array(negative_position,dtype=int)
    np.random.shuffle(tmp_negative)
    negative_ij=tmp_negative[:len(positive_position),:]
    #save as npy
    np.save(father_path+'/data/'+dataset_sort+'/negative_ij.npy',negative_ij)
    np.save(father_path+'/data/'+dataset_sort+'/positive_ij.npy',positive_ij)

    #------------- Greate the index of nodes in Subgraph -------------#
    Diag=np.sum(MLD_matrix,axis=1)
    D=np.diag(Diag)
    for i in range(D.shape[0]):
        if D[i][i]==0:
            D[i][i]=1
    tmp_a=0.15
    S=tmp_a*np.linalg.inv(np.eye(mir_num+lnc_num+dis_num)-(1-tmp_a)*(np.dot(MLD_matrix,np.linalg.inv(D))))
    k_index_list = []
    for i in range(S.shape[0]):
        k_index_list.append(np.array(pd.Series(S[i]).sort_values(ascending = False).index[:FLAGS.subgraph_size]))#subgraph in this model

    subgraph_index=np.array(k_index_list)
    np.save(father_path+'/data/'+dataset_sort+'/subgraph_index.npy',subgraph_index)

    # ------------- Greate WL role embedding -------------#
    tmp_wl = MethodWLNodeColoring(FLAGS,ngraph).run()
    np.save(father_path+'/data/'+dataset_sort +'/WL.npy', tmp_wl)

def disease_similarity(MLD_matrix, mir_num, lnc_num, dis_num, dataset_sort):
    D_IP_SUM = MLD_matrix[mir_num + lnc_num:mir_num + lnc_num + dis_num, 0:mir_num + lnc_num].sum()
    D_GAMMA = mir_num / D_IP_SUM
    D_GIP_SIM_MAT = np.zeros((dis_num, dis_num))
    D_ML_matrix = MLD_matrix[mir_num + lnc_num:mir_num + lnc_num + dis_num, 0:mir_num + lnc_num]
    for i in range(dis_num):
        for j in range(dis_num):
            DIS = np.sum(np.square(D_ML_matrix[i] - D_ML_matrix[j]))
            D_GIP_SIM_MAT[i][j] = np.exp(-1 * D_GAMMA * DIS)
        print(i)

    DIS_SIM_MAT = D_GIP_SIM_MAT
    df = pd.DataFrame(DIS_SIM_MAT)
    df.to_csv(father_path + '/data/' + dataset_sort + '/dis_sim_matrix.csv', index=False, header=True)

def miRNA_func_similarity(MLD_matrix, mir_num, lnc_num, dis_num, dataset_sort):
    DIS_SIM_MAT = pd.read_csv(father_path+'/data/'+dataset_sort+'/dis_sim_matrix.csv', sep=',').values
    MIR_FUN_SIM_MAT = np.zeros((mir_num, mir_num))
    for i in range(mir_num):
        for j in range(i, mir_num):
            D_ASS_1 = np.where(MLD_matrix[i, mir_num + lnc_num:mir_num + lnc_num + dis_num] == 1)
            D_ASS_2 = np.where(MLD_matrix[j, mir_num + lnc_num:mir_num + lnc_num + dis_num] == 1)
            max_list = []
            len_1 = len(D_ASS_1[0])
            len_2 = len(D_ASS_2[0])
            if len_2 != 0:
                for k in range(len_1):
                    temp_array = DIS_SIM_MAT[D_ASS_1[0][k], D_ASS_2[0]]
                    temp_max = np.max(temp_array)
                    max_list.append(temp_max)
            if len_1 != 0:
                for k in range(len_2):
                    temp_array = DIS_SIM_MAT[D_ASS_2[0][k], D_ASS_1[0]]
                    temp_max = np.max(temp_array)
                    max_list.append(temp_max)
            if len_1 + len_2 == 0:
                MIR_FUN_SIM_MAT[i][j] = 0
            else:
                MIR_FUN_SIM_MAT[i][j] = sum(max_list) / (len(D_ASS_1[0]) + len(D_ASS_2[0]))

    MIR_FUN_SIM_MAT = MIR_FUN_SIM_MAT + MIR_FUN_SIM_MAT.T
    for i in range(mir_num):
        MIR_FUN_SIM_MAT[i][i] = 1
    df = pd.DataFrame(MIR_FUN_SIM_MAT)
    df.to_csv(father_path + '/data/' + dataset_sort + '/mir_sim_matrix.csv', index=False, header=True)

def lncRNA_func_similarity(MLD_matrix, mir_num, lnc_num, dis_num, dataset_sort):
    DIS_SIM_MAT = pd.read_csv(father_path+'/data/'+dataset_sort+'/dis_sim_matrix.csv', sep=',').values
    LNC_FUN_SIM_MAT = np.zeros((lnc_num, lnc_num))
    print('lnc_start')
    for i in range(lnc_num):
        for j in range(i, lnc_num):
            D_ASS_1 = np.where(MLD_matrix[mir_num + i, mir_num + lnc_num:mir_num + lnc_num + dis_num] == 1)
            D_ASS_2 = np.where(MLD_matrix[mir_num + j, mir_num + lnc_num:mir_num + lnc_num + dis_num] == 1)
            max_list = []
            len_1 = len(D_ASS_1[0])
            len_2 = len(D_ASS_2[0])
            if len_2 != 0:
                for k in range(len_1):
                    temp_array = DIS_SIM_MAT[D_ASS_1[0][k], D_ASS_2[0]]
                    temp_max = np.max(temp_array)
                    max_list.append(temp_max)
            if len_1 != 0:
                for k in range(len_2):
                    temp_array = DIS_SIM_MAT[D_ASS_2[0][k], D_ASS_1[0]]
                    temp_max = np.max(temp_array)
                    max_list.append(temp_max)
            if len_1 + len_2 == 0:
                LNC_FUN_SIM_MAT[i][j] = 0
            else:
                LNC_FUN_SIM_MAT[i][j] = sum(max_list) / (len(D_ASS_1[0]) + len(D_ASS_2[0]))
        print(i)

    LNC_FUN_SIM_MAT = LNC_FUN_SIM_MAT + LNC_FUN_SIM_MAT.T
    for i in range(lnc_num):
        LNC_FUN_SIM_MAT[i][i] = 1
    df = pd.DataFrame(LNC_FUN_SIM_MAT)
    df.to_csv(father_path + '/data/' + dataset_sort + '/lnc_sim_matrix.csv', index=False, header=True)

def final_sim_matrix(MLD_matrix, mir_num, lnc_num, dis_num, ngraph, dataset_sort):
    DIS_SIM_MAT = pd.read_csv(father_path+ '/data/'+dataset_sort+'/dis_sim_matrix.csv', sep=',').values
    MIR_FUN_SIM_MAT = pd.read_csv(father_path+ '/data/'+dataset_sort+'/mir_sim_matrix.csv', sep=',').values
    LNC_FUN_SIM_MAT = pd.read_csv(father_path+ '/data/'+dataset_sort+'/lnc_sim_matrix.csv', sep=',').values
    sim_matrix=MLD_matrix.astype(np.float)

    sim_matrix[0:mir_num, 0:mir_num] = MIR_FUN_SIM_MAT
    sim_matrix[mir_num:mir_num + lnc_num, mir_num:mir_num + lnc_num] = LNC_FUN_SIM_MAT
    sim_matrix[mir_num + lnc_num:mir_num + lnc_num + dis_num,
    mir_num + lnc_num:mir_num + lnc_num + dis_num] = DIS_SIM_MAT

    for i in range(ngraph):
        sim_matrix[i][i] = 1
    df = pd.DataFrame(sim_matrix)
    df.to_csv(father_path + '/data/' + dataset_sort + '/sim_matrix.csv', index=False, header=True)

if __name__=="__main__":
    FLAGS=get_args_parser()
    data_preprocess(FLAGS)

