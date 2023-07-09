import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.spatial.distance import cdist
import scipy.spatial.distance as ssp_dist


class ConstructGraph:
    def __init__(self, node_type=None, edge_type='corr', dist_type='gau', adj_norm='DAD'):
        self.node_type = node_type
        self.edge_type = edge_type
        self.dist_type = dist_type
        self.adj_norm = adj_norm

    def construct(self, pxs, embedding, coord):
        B, P = pxs.shape[:2]  # 8,200
        pxs = pxs.reshape(B, P, -1)  # [B,P,w**3]
        node_components = self.node_type.split('_')
        components = {'px': pxs, 'emb': embedding, 'coord': coord}
        node = np.concatenate([components[kk] for kk in node_components if kk != 'edge'], axis=-1)

        edge_list = []
        for i in range(B):
            node_i = node[i, :, :]
            coord_i = coord[i, :, :]
            # feature similarity
            edge_square = cal_similarity_matrix(node_i, type1=self.edge_type)
            # Gaussian distance
            if self.dist_type is not None:
                edge_square *= cal_distance(coord_i, dist_type=self.dist_type)
            # laplacian normalization
            if self.adj_norm is not None:
                edge_square[np.diag_indices_from(edge_square)] = 0
                edge_square = normalize_digraph(edge_square, norm_type=self.adj_norm)
            # finish
            edge_list.append(edge_square)

        edges = np.stack(edge_list, axis=0)
        return node, edges  # [N,P,d+], [N,P,P]


def normalize_digraph(A, norm_type='DAD'):
    flag = False
    if isinstance(A, torch.Tensor) and A.layout == torch.sparse_coo:
        A = A.to_dense()
        flag = True

    if isinstance(A, torch.Tensor):
        num_node = A.shape[0]
        A = A + torch.eye(num_node).to(A.device)  # \hat{A}
        rowsum = A.sum(1)
        if norm_type == 'DA':  # DA, D^-1 @ A
            r_inv = rowsum.pow(-1).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_inv[torch.isnan(r_inv)] = 0.
            D = torch.diag(r_inv)
            return Tensor2Sparse(D @ A) if flag else D @ A
        elif norm_type == 'DAD':
            r_inv = rowsum.pow(-1 / 2).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_inv[torch.isnan(r_inv)] = 0.
            D = torch.diag(r_inv)
            return Tensor2Sparse(D @ A @ D) if flag else D @ A @ D

    elif isinstance(A, np.ndarray):
        num_node = A.shape[0]
        A = A + np.eye(num_node)  # \hat{A}
        rowsum = A.sum(1)
        if norm_type == 'DA':  # DA, D^-1 @ A
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_inv[np.isnan(r_inv)] = 0.
            D = np.diag(r_inv)
            return D @ A
        elif norm_type == 'DAD':
            r_inv = np.power(rowsum, -0.5).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_inv[np.isnan(r_inv)] = 0.
            D = np.diag(r_inv)
            return D @ A @ D
    else:
        raise ValueError('Unknown type of XA')


def Tensor2Sparse(Ad):
    assert isinstance(Ad, torch.Tensor) and not Ad.layout == torch.sparse_coo
    idx = torch.nonzero(Ad).T
    data = Ad[idx[0], idx[1]]
    An = torch.sparse.FloatTensor(idx, data, Ad.shape)
    return An


def cal_distance(coord, dist_type):
    fore, after = [kk.lower() for kk in dist_type.split('_')]
    if fore is not None:
        A = cal_similarity_matrix(coord, type1=fore)
    else:
        raise ValueError('distance type?')

    if after in ['gau', 'gaussian']:
        sigma = np.std(A[np.triu_indices_from(A)])
        A = np.exp(- A ** 2 / (2 * sigma ** 2))
    elif after in ['1+gau']:
        sigma = np.std(A[np.triu_indices_from(A)])
        A = np.exp(- A ** 2 / (2 * sigma ** 2))
        A += np.ones_like(A)
    else:
        raise ValueError('distance type?')

    return A


def cal_similarity_matrix(XA, type1):
    if isinstance(XA, torch.Tensor):
        P = XA.shape[0]
        edge_square = torch.zeros((P, P)).to(XA.device)
        for j in range(P):
            for k in range(j, P):
                edge_square[j, k] = cal_similarity_vector(XA[j, :], XA[k, :], type1=type1)
                edge_square[k, j] = edge_square[j, k]
        if abs(edge_square[0, 0] - 1) < 1e-5:
            edge_square = torch.where(torch.isnan(edge_square), torch.full_like(edge_square, 0), edge_square)
        assert sum(torch.isnan(edge_square)[0]) == 0

    elif isinstance(XA, np.ndarray):
        if type1 in ['euc', 'euclidean', 'l2']:
            edge_square = cdist(XA, XA, metric='euclidean')
        elif type1 in ['dot']:
            edge_square = XA @ XA.T  # np.dot(XA, XA.T)
        elif type1 in ['pearson', 'corr']:
            edge_square = 1 - cdist(XA, XA, metric='correlation')
        elif type1 in ['abspearson', 'abscorr']:
            edge_square = abs(1 - cdist(XA, XA, metric='correlation'))
        else:
            raise ValueError('similarity type?')
        if abs(edge_square[0, 0] - 1) < 1e-5:
            edge_square[np.isnan(edge_square)] = 0.0
        assert np.isnan(edge_square).sum() == 0

    else:
        raise ValueError('Unknown type of XA')
    return edge_square


def cal_similarity_vector(tensor1, tensor2, type1):
    if isinstance(tensor1, torch.Tensor):
        if type1 in ['euc', 'euclidean', 'l2']:
            pdist = nn.PairwiseDistance(p=2)
            return pdist(tensor1, tensor2)  # torch.norm(te1-te2, p=2)
        elif type1 in ['dot']:
            return tensor1.T @ tensor2  # torch.sum(tensor1*tensor2)
        elif type1 in ['pearson', 'corr']:
            vx = tensor1 - torch.mean(tensor1)
            vy = tensor2 - torch.mean(tensor2)
            cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            return cost  # np.corrcoef(x[i, j, :].data.cpu().numpy(), x[i, k, :].data.cpu().numpy())[0, 1]
        elif type1 in ['abspearson', 'abscorr']:
            vx = tensor1 - torch.mean(tensor1)
            vy = tensor2 - torch.mean(tensor2)
            cost = torch.abs(torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
            return cost  # np.corrcoef(x[i, j, :].data.cpu().numpy(), x[i, k, :].data.cpu().numpy())[0, 1]
        else:
            raise ValueError('similarity type?')
    elif isinstance(tensor1, np.ndarray):
        if type1 in ['euc', 'euclidean', 'l2']:
            return ssp_dist.euclidean(tensor1, tensor2)  # np.linalg.norm(t1 - t2, ord=2)
        elif type1 in ['dot']:
            return np.sum(tensor1 * tensor2)
        elif type1 in ['pearson', 'corr']:
            return np.corrcoef(tensor1, tensor2)[0, 1]
        elif type1 in ['abspearson', 'abscorr']:
            return abs(np.corrcoef(tensor1, tensor2)[0, 1])
        else:
            raise ValueError('similarity type?')
    else:
        raise ValueError('Unknown type of tensor1')
