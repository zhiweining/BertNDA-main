# BertNDA: a Model Based on Graph-Bert and Multi-scale Information Fusion for ncRNA-disease Association Prediction
# @Institution: Department of Electronic Information, Xian Jiaotong University, China
# @Author: Zhiwei Ning 
# @Contact: 2193612777@stu.xjtu.edu.cn


import codecs
import csv
import torch
from data_train_test import test
import heapq
import numpy as np
import torch.utils.data as Data

def data_write_csv(file_name, datas):#save as csv
        file_csv = codecs.open(file_name,'w+','utf-8')
        writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for data in datas:
            writer.writerow(data)
        print("save to csv file finished! ")

def topk_index_ouput(melo_name_list,k,model,ngraph,origin_matrix,sim_matrix,subgraph_index,lap_matrix,WL_embed_matrix,base_path,args):
    categories,melo_list=['mir','lnc','dis'],[]
    for category in categories:
        with open(base_path+'/data/'+args.dataset_sort+'/'+category+'.txt','r') as file:
            for context in file:
                melo_list.append(context.strip('\n'))
    multi_value_list=[]
    multi_melo_list=[]
    for melo_name in melo_name_list:
        melo_name=melo_name.replace(' ','').replace('-','').replace('_','').replace("'","").replace(".","").lower()
        CASE_STUDY_X_list=[]
        melo_index=melo_list.index(melo_name)
        mele_pos_index=[index for index,value in enumerate(origin_matrix[melo_index].tolist()) if value==1]
        case_study_test_index=[index for index in range(ngraph) if index not in mele_pos_index]
        for temp_index in case_study_test_index:
            tmp1 = sim_matrix[subgraph_index[melo_index], :].reshape(1, -1)
            tmp2 = sim_matrix[subgraph_index[temp_index], :].reshape(1, -1)
            temp_list3 = np.concatenate((tmp1, tmp2, lap_matrix[melo_index, :].reshape(1, -1), lap_matrix[temp_index, :].reshape(1, -1),
                                        WL_embed_matrix[melo_index, :].reshape(1, -1), WL_embed_matrix[temp_index, :].reshape(1, -1)), axis=1)
            CASE_STUDY_X_list.append(torch.tensor(temp_list3))
        X = torch.cat((CASE_STUDY_X_list), dim=0)
        CASE_STUDY_X = X.view(len(case_study_test_index), -1)
        CASE_STUDY_Y = torch.zeros([CASE_STUDY_X.shape[0],1],dtype=torch.float)
        CASE_STUDY_DATA=Data.TensorDataset(CASE_STUDY_X, CASE_STUDY_Y)
        case_study_tt_set = torch.utils.data.DataLoader(
            CASE_STUDY_DATA,
            batch_size=args.batch_size,
            shuffle=True)
        gpu = torch.device(args.device if torch.cuda.is_available() else "cpu")
        _ , preds = test(case_study_tt_set, model, gpu)
        topk_index_list = heapq.nlargest(k, range(len(preds)), preds.__getitem__)
        print('topk_index_list of '+melo_name+' is :',topk_index_list)
        print('topk_value_list of '+melo_name+' is :',preds[topk_index_list])
        multi_value_list.append(preds[topk_index_list])
        pred_name_list=[melo_list[index] for index in topk_index_list]
        multi_melo_list.append(pred_name_list)
    return multi_value_list,multi_melo_list
