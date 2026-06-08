# -*- coding: utf-8 -*-
# @Time   : 2020/7/8
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : deepfm.py

# UPDATE:
# @Time   : 2020/8/14
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

r"""
DeepFM
################################################
Reference:
    Huifeng Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." in IJCAI 2017.
"""

import torch.nn as nn

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import BaseFactorizationMachine, MLPLayers


class DeepFM(ContextRecommender):
    """DeepFM is a DNN enhanced FM which both use a DNN and a FM to calculate feature interaction.
    Also DeepFM can be seen as a combination of FNN and FM.

    """

    def __init__(self, config, dataset):
        super(DeepFM, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [
            self.embedding_size * self.num_feature_field
        ] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(
            self.mlp_hidden_size[-1], 1
        )  # Linear product to the final score

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, interaction):
        deepfm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        batch_size = deepfm_all_embeddings.shape[0]
        y_fm = self.first_order_linear(interaction) + self.fm(deepfm_all_embeddings)

        y_deep = self.deep_predict_layer(
            self.mlp_layers(deepfm_all_embeddings.view(batch_size, -1))
        )
        y = y_fm + y_deep
        return y.squeeze(-1)


