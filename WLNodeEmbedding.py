# BertNDA: a Model Based on Graph-Bert and Multi-scale Information Fusion for ncRNA-disease Association Prediction
# @Institution: Department of Electronic Information, Xian Jiaotong University, China
# @Author: Zhiwei Ning 
# @Contact: 2193612777@stu.xjtu.edu.cn


import hashlib
import numpy as np
import math
import os

current_path = os.path.abspath(__file__)
father_path = os.path.abspath(
    os.path.dirname(current_path) + os.path.sep + ".")

class MethodWLNodeColoring:
    def __init__(self,FLAGS,dim):
        self.datasort=FLAGS.dataset_sort
        self.dim=dim
        self.data = None
        self.max_iter = FLAGS.wl_max_iter
        self.node_color_dict = {}
        self.node_neighbor_dict = {}

    def setting_init(self, node_list, link_list):
        for node in node_list:
            self.node_color_dict[node] = 1
            self.node_neighbor_dict[node] = {}

        for pair in link_list:
            u1, u2 = pair
            if u1 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u1] = {}
            if u2 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u2] = {}
            self.node_neighbor_dict[u1][u2] = 1
            self.node_neighbor_dict[u2][u1] = 1

    def WL_recursion(self, node_list):
        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in node_list:
                neighbors = self.node_neighbor_dict[node]
                neighbor_color_list = [self.node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(self.node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if self.node_color_dict == new_color_dict or iteration_count == self.max_iter:
                for i in range(self.dim):
                    tmp_list=[]
                    for l in range(int(self.dim / 2)):
                        tmp_list.append(math.sin(self.node_color_dict[i]/math.pow(10000,float(2*l)/self.dim)))
                        tmp_list.append(math.cos(self.node_color_dict[i]/math.pow(10000,float(2*l+1)/self.dim)))
                    if self.dim%2==1:
                        tmp_list.append(math.sin(self.node_color_dict[i]/float(10000)))
                    self.node_color_dict[i]=np.array(tmp_list).reshape(1,-1)
                return
            else:
                self.node_color_dict = new_color_dict
            iteration_count += 1

    def run(self):
        link_list=np.load(father_path+ '/data/'+self.datasort+'/positive_ij.npy').tolist()
        node_list=[i for i in range(self.dim)]
        self.setting_init(node_list, link_list)
        self.WL_recursion(node_list)
        return self.node_color_dict