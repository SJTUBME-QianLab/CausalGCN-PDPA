# CausalGCN-PDPA

This repository holds the code for the paper

**A Causal-Driven Graph Convolutional Networks for Postural Abnormality Identification in Parkinsonians**

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserve the copyright and all legal rights of these codes.

# Author List

Xinlu Tang, Xiaohua Qian\*

# required

Our code is based on **Python3.9** There are a few dependencies to run the code. The major libraries we depend are

- PyTorch1.10.0 (http://pytorch.org/)
- numpy

# set up

```
conda install --yes --file requirements.txt
```

Attention: Please run this project on linux. In different pytorch environment, the model may obtain different results.

# quickly start

run the `main.py` by this command:

```shell
python main.py --config ./train_causal_pre.yaml
```

The results will be saved in `./result` folder.

# description of input data structure

Data loading: *FeederGraph* in `/tools/feeder.py`

Shape of input data: `[B,P,d]`.  B: batch size, P: number of patches within a sample or number of nodes within a graph, d: length of node feature vector. 

Input data acts as the node feature matrix, and patch coordinate information is needed additionally. 

Adjacency matrix calculation: *ConstructGraph* in `/tools/construct_ graph_simple.py`

Shape of output matrix:`[B,P,P]`, formed by stacking the adjacency matrices of all graphs along the first dimension.

For the data from users, the adjacency matrix stack can be directly loaded in or be calculated through a customized function. Finally, the node feature matrix and the adjacency matrix stack are fed into the model.

# Contact

For any question, feel free to contact

```
Xinlu Tang : tangxl20@sjtu.edu.cn
```