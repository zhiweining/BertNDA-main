## BertNDA
BertNDA: Predicting ncRNA and disease trinary associations based on Multi-scale feature Fusion and Graph-Bert
 
![BertNDA](/imgs/Method.svg)

## Introduction
In this work, we propose a predictive framework, called BertNDA, which aims to predict association between disease miRNA and lncRNA. The framework employs Laplace transform of graph structure and WL(Weisfeiler-Lehman) absolute role coding to extract global information. Construct a connectionless subgraph to aggregate neighbor feature to identify local information. Moreover, an EMLP structure is adopted to obtain the multi-scale feature representation of node. Furtherly, nodes are encoded using Transformer-encoder structure, and prediction-layer outputs the final correlation between miRNA-lncRNA and diseases. A 5-fold cross-validation further demonstrate that BertNDA outperforms the state-of-the-art method in predicting assignment. Furthermore, an online prediction platform that embeds our prediction model is designed for users to experience. Overall, our model provides an efficient, accurate, and comprehensive tool for predicting ncRNA-disease associations.

## Installation
### Requirements
All the codes are tested in the following environment:
`Linux(Ubuntu 18.04)` `Python 3.8.16` `PyTorch 1.10.1` `CUDA 11.X` `numpy 1.23.5` `Pillow 8.3.2`
Install the dependent python libraries by

```
pip install -r requirements.txt
```
### Dataset Preparation
The dataset deafult used in our code is `dataset1`, if you want to train in the dataset2, please download the dataset according to [README.MD](data/dataset2/README.MD) and preprare the preprocess data by the following step:

```
python main.py --dataset_sort=dataset2
```
Some files will be generated in the dataset2 folder, follows as:

```
dataset2
|—— sim_matrix.csv
|—— negative_ij.npy
|—— positive_ij.npy
|—— subgraph_index.npy
|—— WL.npy
```
If you want to utilize our model to yourself dataset, please prepare the dataset folder as follows:

```
your dataset folder
|—— adj_matrix(m_l_d).csv
|—— mir.txt
|—— lnc.txt
|—— dis.txt
```
then running main.py to train and eval model in your dataset.

## Online Platform
We also design a website to show the predict result in user-friendly page, click [HERE](39.106.16.168:8017) for experience.
 
## Others
### The Method compared in our work
- [CNNMDA](https://pubmed.ncbi.nlm.nih.gov/30977780):A novel based on CNN-based framework for predicting the associations between miRNA and dieases.
- [DEFLMDA](https://pubmed.ncbi.nlm.nih.gov/35325038/): A model employs a novel computational approach of random forest ensemble learning to predictmiRNA-disease associations, integrating autoencoders for low-dimensional feature representation.
- [HGCNMDA](https://pubmed.ncbi.nlm.nih.gov/35776822/): A model incorporates a gene layer in constructing a heterogeneous network.The model refines thefeature of nodes into initial features and induction features and then learns miRNA and disease embeddings viaa multi-graph convolutional network model.
- [LDAformer](https://pubmed.ncbi.nIm.nih.gov/36094081/):A model leverages topological feature extraction and Transformer encoder.Specifically, a pivotalprocess is designed for extracting potential multi-hop path feature from adjacent matrices during the topologicalinformation extraction stage.
- [PADLMHOOI](https://pubmed.ncbi.nlm.nih.gov/31191710/):A model implements high-order orthogonal iterations to predict potential associations andevaluates predictive performance through global and local leave-one-out cross-validation.

### Contact
If you have any questions, welcome to contact me at 2193612777@stu.xjtu.edu.cn!
