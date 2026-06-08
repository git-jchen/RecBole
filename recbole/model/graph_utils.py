# -*- coding: utf-8 -*-
"""
recbole.model.graph_utils
##########################

Shared utility functions for graph-based recommender models.
Extracted from duplicated code in LightGCN, NGCF, NCL, GCMC, and SpectralCF.
"""

import numpy as np
import scipy.sparse as sp
import torch


def sp_mat_to_tensor(sp_mat):
    r"""Convert a scipy sparse matrix to a torch sparse tensor.

    Args:
        sp_mat (scipy.sparse): A scipy sparse matrix.

    Returns:
        torch.sparse.FloatTensor: The corresponding sparse tensor.
    """
    sp_mat = sp_mat.tocoo()
    indices = torch.LongTensor(np.array([sp_mat.row, sp_mat.col]))
    values = torch.FloatTensor(sp_mat.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(sp_mat.shape))


def build_norm_adj_matrix(interaction_matrix, n_users, n_items):
    r"""Build the symmetric normalized adjacency matrix from the interaction matrix.

    Constructs the user-item bipartite graph adjacency matrix and normalizes it
    using symmetric normalization.

    .. math::
        \hat{A} = D^{-0.5} \times A \times D^{-0.5}

    Args:
        interaction_matrix (scipy.sparse): The user-item interaction matrix in COO format.
        n_users (int): Number of users.
        n_items (int): Number of items.

    Returns:
        torch.sparse.FloatTensor: The normalized adjacency matrix.
    """
    A = sp.dok_matrix(
        (n_users + n_items, n_users + n_items), dtype=np.float32
    )
    inter_M = interaction_matrix
    inter_M_t = interaction_matrix.transpose()
    data_dict = dict(
        zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz)
    )
    data_dict.update(
        dict(
            zip(
                zip(inter_M_t.row + n_users, inter_M_t.col),
                [1] * inter_M_t.nnz,
            )
        )
    )
    A._update(data_dict)
    # norm adj matrix
    sumArr = (A > 0).sum(axis=1)
    # add epsilon to avoid divide by zero Warning
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D * A * D
    return sp_mat_to_tensor(L)


def build_sparse_eye_matrix(num):
    r"""Construct a sparse identity matrix.

    Args:
        num (int): Size of the square identity matrix.

    Returns:
        torch.sparse.FloatTensor: Sparse identity matrix of shape (num, num).
    """
    indices = torch.LongTensor([range(0, num), range(0, num)])
    values = torch.FloatTensor([1] * num)
    return torch.sparse_coo_tensor(indices, values, torch.Size([num, num]))


def get_ego_embeddings(user_embedding, item_embedding):
    r"""Concatenate user and item embedding weights into one embedding matrix.

    Args:
        user_embedding (torch.nn.Embedding): User embedding layer.
        item_embedding (torch.nn.Embedding): Item embedding layer.

    Returns:
        torch.Tensor: Concatenated embedding matrix of shape
            (n_users + n_items, embedding_dim).
    """
    return torch.cat([user_embedding.weight, item_embedding.weight], dim=0)
