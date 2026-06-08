# -*- coding: utf-8 -*-
# @Time   : 2020/08/30
# @Author : Xinyan Fan
# @Email  : xinyan.fan@ruc.edu.cn
# @File   : lr.py

r"""
LR
#####################################################
Reference:
    Matthew Richardson et al. "Predicting Clicks Estimating the Click-Through Rate for New Ads." in WWW 2007.
"""

import torch.nn as nn

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.init import xavier_normal_initialization


class LR(ContextRecommender):
    r"""LR is a context-based recommendation model.
    It aims to predict the CTR given a set of features by using logistic regression,
    which is ideally suited for probabilities as it always predicts a value between 0 and 1:

    .. math::
        CTR = \frac{1}{1+e^{-Z}}

        Z = \sum_{i} {w_i}{x_i}
    """

    def __init__(self, config, dataset):
        super(LR, self).__init__(config, dataset)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, interaction):
        output = self.first_order_linear(interaction)
        return output.squeeze(-1)


