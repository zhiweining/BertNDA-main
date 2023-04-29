# BertNDA
BertNDA: Predicting associations of miRNAs,lncRNAs and diseases based on Graph-Bert and Multi-scale feature extraction
 
![BertNDA](imgs/Method.svg)

## Introduction:


## Installation:

### Requirements:
All the codes are tested in the following environment:

```Linux (tested on Ubuntu 18.04)```, 
```Python 3.8.16```
```PyTorch 1.10.1```
```numpy 1.23.5```
```Pillow 8.3.2```

a. Install the dependent python libraries.
```shell
pip install -r requirements.txt 
```

### Dataset Preparation
The dataset deafult used in our code is "dataset1", if you want to train in the dataset2, please prepare the data in the following steps:
```shell
run main.py --dataset_sort=dataset2
```
some files will be generated in the dataset2 fold, follows as:
```
dataset2
├── sim_matrix.csv
├── negative_ij.npy
├── positive_ij.npy
├── subgraph_index.npy
├── WL.npy
```
If you want to transfer our model to your dataset, please prepaer the dataset as follows:
```
your_dataset
├── adjmatrix(m_l_d).csv
├── mir.txt
├── lnc.txt
├── dis.txt
```
then run our code to train the model in your dataset.

## online-platform:
we also a website to show the predict result in user-friendly, click [online-platform](39.106.16.168:8017) for details.

## Others:
### The Methods compared in our work
CNNMDA:a model based on CNN for predict the associations between miRNA and dieases. [paper_url](https://pubmed.ncbi.nlm.nih.gov/30977780/)





If you have any questions, welcome to contact we at 2193612777@stu.xjtu.edu.cn!


