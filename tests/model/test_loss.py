# -*- coding: utf-8 -*-
# Tests for recbole/model/loss.py

import torch
import pytest

from recbole.model.loss import BPRLoss, RegLoss, EmbLoss, EmbMarginLoss


class TestBPRLoss:
    def test_basic(self):
        loss_fn = BPRLoss()
        pos_score = torch.tensor([1.0, 2.0, 3.0])
        neg_score = torch.tensor([0.5, 1.0, 1.5])
        loss = loss_fn(pos_score, neg_score)
        assert loss.item() > 0

    def test_equal_scores(self):
        loss_fn = BPRLoss()
        scores = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(scores, scores)
        # When pos == neg, sigmoid(0) = 0.5, -log(0.5 + gamma) ~ 0.693
        assert loss.item() > 0

    def test_pos_greater_than_neg(self):
        loss_fn = BPRLoss()
        pos_score = torch.tensor([5.0, 5.0, 5.0])
        neg_score = torch.tensor([0.0, 0.0, 0.0])
        loss = loss_fn(pos_score, neg_score)
        # Large positive difference → small loss
        assert loss.item() < 0.1

    def test_neg_greater_than_pos(self):
        loss_fn = BPRLoss()
        pos_score = torch.tensor([0.0, 0.0, 0.0])
        neg_score = torch.tensor([5.0, 5.0, 5.0])
        loss = loss_fn(pos_score, neg_score)
        # Large negative difference → large loss
        assert loss.item() > 1.0

    def test_gradient_flows(self):
        loss_fn = BPRLoss()
        pos_score = torch.tensor([1.0, 2.0], requires_grad=True)
        neg_score = torch.tensor([0.5, 1.0], requires_grad=True)
        loss = loss_fn(pos_score, neg_score)
        loss.backward()
        assert pos_score.grad is not None
        assert neg_score.grad is not None

    def test_custom_gamma(self):
        loss_fn = BPRLoss(gamma=1e-5)
        pos_score = torch.tensor([1.0, 2.0, 3.0])
        neg_score = torch.tensor([0.5, 1.0, 1.5])
        loss = loss_fn(pos_score, neg_score)
        assert loss.item() > 0


class TestRegLoss:
    def test_basic(self):
        loss_fn = RegLoss()
        params = [torch.randn(3, 4), torch.randn(5, 6)]
        loss = loss_fn(params)
        assert loss.item() > 0

    def test_zero_params(self):
        loss_fn = RegLoss()
        params = [torch.zeros(3, 4)]
        loss = loss_fn(params)
        assert loss.item() == 0.0

    def test_single_param(self):
        loss_fn = RegLoss()
        param = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = loss_fn([param])
        expected = param.norm(2)
        assert torch.isclose(loss, expected)

    def test_multiple_params(self):
        loss_fn = RegLoss()
        p1 = torch.tensor([[1.0, 0.0]])
        p2 = torch.tensor([[0.0, 1.0]])
        loss = loss_fn([p1, p2])
        expected = p1.norm(2) + p2.norm(2)
        assert torch.isclose(loss, expected)


class TestEmbLoss:
    def test_basic(self):
        loss_fn = EmbLoss(norm=2)
        emb = torch.randn(10, 64)
        loss = loss_fn(emb)
        assert loss.item() > 0

    def test_zero_embeddings(self):
        loss_fn = EmbLoss(norm=2)
        emb = torch.zeros(10, 64)
        loss = loss_fn(emb)
        assert loss.item() == 0.0

    def test_multiple_embeddings(self):
        loss_fn = EmbLoss(norm=2)
        emb1 = torch.randn(10, 64)
        emb2 = torch.randn(10, 32)
        loss = loss_fn(emb1, emb2)
        assert loss.item() > 0

    def test_require_pow(self):
        loss_fn = EmbLoss(norm=2)
        emb = torch.randn(10, 64)
        loss_pow = loss_fn(emb, require_pow=True)
        loss_no_pow = loss_fn(emb, require_pow=False)
        # Both should be positive but different
        assert loss_pow.item() > 0
        assert loss_no_pow.item() > 0

    def test_different_norms(self):
        emb = torch.randn(10, 64)
        loss_l1 = EmbLoss(norm=1)(emb)
        loss_l2 = EmbLoss(norm=2)(emb)
        # L1 and L2 norms should give different values
        assert not torch.isclose(loss_l1, loss_l2)


class TestEmbMarginLoss:
    def test_basic(self):
        loss_fn = EmbMarginLoss()
        emb = torch.randn(10, 64) * 2  # Scale up so some exceed margin
        loss = loss_fn(emb)
        assert loss.item() >= 0

    def test_small_embeddings(self):
        loss_fn = EmbMarginLoss()
        # Embeddings with norm < 1 should give 0 loss
        emb = torch.ones(10, 64) * 0.01
        loss = loss_fn(emb)
        assert loss.item() == 0.0

    def test_multiple_embeddings(self):
        loss_fn = EmbMarginLoss()
        emb1 = torch.randn(10, 64) * 2
        emb2 = torch.randn(10, 32) * 2
        loss = loss_fn(emb1, emb2)
        assert loss.item() >= 0

    def test_custom_power(self):
        emb = torch.randn(10, 64) * 2
        loss_p2 = EmbMarginLoss(power=2)(emb)
        loss_p3 = EmbMarginLoss(power=3)(emb)
        # Different powers should give different results
        assert loss_p2.item() != loss_p3.item()
