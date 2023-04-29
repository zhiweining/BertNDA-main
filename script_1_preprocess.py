#预处理部分作用为从原有的矩阵信息中构建出link与node结构，并获取raw_embedding,Batch_embedding以及Hop_embedding信息
import random
import numpy as np
import torch
import pandas as pd

from bertcode.DatasetLoader import DatasetLoader
from bertcode.MethodWLNodeColoring import MethodWLNodeColoring
from bertcode.MethodGraphBatching import MethodGraphBatching
from bertcode.MethodHopDistance import MethodHopDistance
from bertcode.ResultSaving import ResultSaving
from bertcode.Settings import Settings

node_and_link_create=1
raw_embedding_create=1
raw_wl_create=1
raw_hop_create=1

if node_and_link_create:
    dataset_sort='mini'
    MLD_data_path='./data/mydataset/'+dataset_sort#获取主目录
    MLD_matrix_path=MLD_data_path+'/adjmatrix(m_l_d).csv'

    if dataset_sort=='mini':
        mir_num=464
        lnc_num=56
        dis_num=121
    if dataset_sort=='max':
        mir_num=1596
        lnc_num=2189
        dis_num=1296

    MLD_matrix = pd.read_csv(MLD_matrix_path,sep=',').values
    MLD_matrix=MLD_matrix+np.eye(641)#添加对角线的"1"

    MLD_list=[[int(i) for i in MLD_matrix[j]] for j in range(MLD_matrix.shape[0])]
    for i in range(mir_num+lnc_num+dis_num):
        if i<mir_num:
            MLD_list[i].append('mir')
        elif i>=mir_num and i<mir_num+lnc_num: MLD_list[i].append('lnc')
        else :MLD_list[i].append('dis')
        MLD_list[i].insert(0, i)
        MLD_list[i].append('\n')
    random.shuffle(MLD_list)#在matrix中添加类别与index,index从0开始

    MLD_str=''#将list转变为可存储的类型
    for i in range(mir_num+lnc_num+dis_num):
        for j in range(mir_num+lnc_num+dis_num+2):
            MLD_str=MLD_str+str(MLD_list[i][j])+','
        print("finish",str(i))
        MLD_str.strip(',')
        MLD_str=MLD_str+'\n'
    MLD_str.strip('\n')

    MLD_link_str=''#储存link的字符列表
    link_num=0
    for i in range(lnc_num+mir_num+dis_num):
        for j in range(i+1,lnc_num+mir_num+dis_num):
            if MLD_matrix[i][j]==1:
                MLD_link_str=MLD_link_str+str(i)+','+str(j)+'\n'
                link_num+=1
    MLD_link_str.strip('\n')

    with open('./data/mydataset/node','w',encoding='utf-8') as f:#生成node并保存
        f.write(MLD_str)

    with open('./data/mydataset/link','w',encoding='utf-8') as f:#生成link并保存
        f.write(MLD_link_str)

if raw_embedding_create:

    dataset_name = 'mydataset'
    datasort='mini'
    np.random.seed(1)
    torch.manual_seed(1)
    nclass = 3
    if datasort=='mini':
        nfeature = 641
        ngraph = 641

    #---- Step 1: WL based graph coloring ----
    if raw_wl_create:
        print('************ Start ************')
        print('WL, dataset: ' + dataset_name)
        # ---- objection initialization setction ---------------
        data_obj = DatasetLoader()
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.dataset_name = dataset_name

        method_obj = MethodWLNodeColoring()

        result_obj = ResultSaving()
        result_obj.result_destination_folder_path = './result/WL/'
        result_obj.result_destination_file_name = dataset_name

        setting_obj = Settings()

        evaluate_obj = None
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.load_run_save_evaluate()
        # ------------------------------------------------------

        print('************ Finish ************')
    #------------------------------------

    #---- Step 2: intimacy calculation and subgraph batching ----
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:#, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        print('************ Start ************')
        print('Subgraph Batching, dataset: ' + dataset_name + ', k: ' + str(k))
        # ---- objection initialization setction ---------------
        data_obj = DatasetLoader()
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.dataset_name = dataset_name
        data_obj.compute_s = True

        method_obj = MethodGraphBatching()
        method_obj.k = k

        result_obj = ResultSaving()
        result_obj.result_destination_folder_path = './result/Batch/'
        result_obj.result_destination_file_name = dataset_name + '_' + str(k)

        setting_obj = Settings()

        evaluate_obj = None
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.load_run_save_evaluate()
        # ------------------------------------------------------

        print('************ Finish ************')
    #------------------------------------

    #---- Step 3: Shortest path: hop distance among nodes ----
    if raw_hop_create:
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print('************ Start ************')
            print('HopDistance, dataset: ' + dataset_name + ', k: ' + str(k))
            # ---- objection initialization setction ---------------
            data_obj = DatasetLoader()
            data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
            data_obj.dataset_name = dataset_name

            method_obj = MethodHopDistance()
            method_obj.k = k
            method_obj.dataset_name = dataset_name

            result_obj = ResultSaving()
            result_obj.result_destination_folder_path = './result/Hop/'
            result_obj.result_destination_file_name = 'hop_' + dataset_name + '_' + str(k)

            setting_obj = Settings()

            evaluate_obj = None
            # ------------------------------------------------------

            # ---- running section ---------------------------------
            setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
            setting_obj.load_run_save_evaluate()
            # ------------------------------------------------------

            print('************ Finish ************')
    #------------------------------------