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

# Contact

For any question, feel free to contact

```
Xinlu Tang : tangxl20@sjtu.edu.cn
```