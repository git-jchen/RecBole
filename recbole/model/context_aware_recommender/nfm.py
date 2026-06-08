# -*- coding: utf-8 -*-
# @Time   : 2020/7/14
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
# @File   : nfm.py

r"""
NFM
################################################
Reference:
    He X, Chua T S. "Neural factorization machines for sparse predictive analytics" in SIGIR 2017
"""

import torch.nn as nn

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import BaseFactorizationMachine, MLPLayers


class NFM(ContextRecommender):
    """NFM replace the fm part as a mlp to model the feature interaction."""

    def __init__(self, config, dataset):
        super(NFM, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        size_list = [self.embedding_size] + self.mlp_hidden_size
        self.fm = BaseFactorizationMachine(reduce_sum=False)
        self.bn = nn.BatchNorm1d(num_features=self.embedding_size)
        self.mlp_layers = MLPLayers(
            size_list, self.dropout_prob, activation="sigmoid", bn=True
        )
        self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1, bias=False)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, interaction):
        nfm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        bn_nfm_all_embeddings = self.bn(self.fm(nfm_all_embeddings))

        output = self.predict_layer(
            self.mlp_layers(bn_nfm_all_embeddings)
        ) + self.first_order_linear(interaction)
        return output.squeeze(-1)


