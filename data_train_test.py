# BertNDA: a Model Based on Graph-Bert and Multi-scale Information Fusion for ncRNA-disease Association Prediction
# @Institution: Department of Electronic Information, Xian Jiaotong University, China
# @Author: Zhiwei Ning 
# @Contact: 2193612777@stu.xjtu.edu.cn


import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, auc
from torch.utils.tensorboard import SummaryWriter

class MultiLoss(nn.Module):
    def __init__(self,alpha):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target):
        loss_1 = F.binary_cross_entropy(pred[0], target)
        loss_2 = F.binary_cross_entropy(pred[1], target)
        loss = self.alpha * loss_1 + (1-self.alpha)* loss_2
        return loss

# Data Loader
def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1) + 1
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.zeros((R_sqrt.shape[0], R_sqrt.shape[0]))
    for i in range(R_sqrt.shape[0]):
        D_sqrt[i][i] = R_sqrt[i]
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

def get_data(Ai, ij):
    data = []
    for item in ij:
        feature = np.array([Ai[0][item[0]], Ai[0][item[1]]])
        for dim in range(1, Ai.shape[0]):
            feature = np.concatenate((feature, np.array([Ai[dim][item[0]], Ai[dim][item[1]]])))
        data.append(feature)
    return np.array(data)


def train(tr_set, model, FLAGS, gpu, writer):
    fn_loss=MultiLoss(FLAGS.loss_alpha)
    n_epochs = FLAGS.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    epoch = 0
    while epoch < n_epochs:
        model.train()
        print('epoch=%d'%epoch)
        for i,(x, y) in enumerate(tr_set):
            optimizer.zero_grad()
            x, y = x.to(gpu).to(torch.float32), y.to(gpu).to(torch.float32)
            pred = model(x)
            loss = fn_loss(pred, y)
            loss.backward()
            optimizer.step()
            if i==0:
                writer.add_scalar('train_loss_cpu', loss.detach().cpu().numpy(), epoch)
            if i%10==0:
                print('loss=%.6f'%loss.detach().cpu().numpy())
        epoch += 1
    print('Finished training after {} epochs'.format(epoch))

# Test
def test(tt_set, model, device, FLAGS):
    model.eval()
    preds = []
    labels = []
    for x, y in tt_set:
        x,y= x.to(device).to(torch.float32),y.to(device).to(torch.float32)
        with torch.no_grad():
            pred = model(x)
            pred = FLAGS.loss_alpha*pred[0]+ (1-FLAGS.loss_alpha)*pred[1]
            preds.append(pred.detach().cpu())
        labels.append(y.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return labels, preds

