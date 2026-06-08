# -*- coding: utf-8 -*-
# @Time   : 2020/9/15 10:57
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
# @File   : fnn.py

r"""
FNN
################################################
Reference:
    Weinan Zhang1 et al. "Deep Learning over Multi-field Categorical Data" in ECIR 2016
"""

import torch.nn as nn

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers


class FNN(ContextRecommender):
    """FNN which also called DNN is a basic version of CTR model that use mlp from field features to predict score.

    Note:
        Based on the experiments in the paper above, This implementation incorporate
        Dropout instead of L2 normalization to relieve over-fitting.
        Our implementation of FNN is a basic version without pretrain support.
        If you want to pretrain the feature embedding as the original paper,
        we suggest you to construct a advanced FNN model and train it in two-stage
        process with our FM model.
    """

    def __init__(self, config, dataset):
        super(FNN, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        size_list = [
            self.embedding_size * self.num_feature_field
        ] + self.mlp_hidden_size

        # define layers and loss
        self.mlp_layers = MLPLayers(
            size_list, self.dropout_prob, activation="tanh", bn=False
        )  # use tanh as activation
        self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1, bias=True)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, interaction):
        fnn_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        batch_size = fnn_all_embeddings.shape[0]

        output = self.predict_layer(
            self.mlp_layers(fnn_all_embeddings.view(batch_size, -1))
        )
        return output.squeeze(-1)


