# -*- coding: utf-8 -*-
# Tests for recbole/evaluator/utils.py

import numpy as np
import torch
import pytest

from recbole.evaluator.utils import pad_sequence, trunc, cutoff, _binary_clf_curve


class TestPadSequence:
    def test_equal_length_sequences(self):
        seq1 = torch.tensor([1.0, 2.0, 3.0])
        seq2 = torch.tensor([4.0, 5.0, 6.0])
        sequences = [seq1, seq2]
        len_list = [3, 3]
        result = pad_sequence(sequences, len_list)
        assert result.shape == (2, 3)
        assert torch.equal(result[0], seq1)
        assert torch.equal(result[1], seq2)

    def test_variable_length_sequences(self):
        seq1 = torch.tensor([1.0, 2.0, 3.0])
        seq2 = torch.tensor([4.0, 5.0])
        sequences = [seq1, seq2]
        len_list = [3, 2]
        result = pad_sequence(sequences, len_list)
        assert result.shape == (2, 3)
        assert torch.equal(result[0], seq1)
        # Second row: [4.0, 5.0, -inf]
        assert result[1][0].item() == 4.0
        assert result[1][1].item() == 5.0
        assert result[1][2].item() == float("-inf")

    def test_pad_to_specific_length(self):
        seq1 = torch.tensor([1.0, 2.0])
        sequences = [seq1]
        len_list = [2]
        result = pad_sequence(sequences, len_list, pad_to=5)
        assert result.shape == (1, 5)

    def test_single_sequence(self):
        seq = torch.tensor([1.0, 2.0, 3.0])
        result = pad_sequence([seq], [3])
        assert result.shape == (1, 3)


class TestTrunc:
    def test_ceil(self):
        scores = np.array([1.2, 2.7, 3.1])
        result = trunc(scores, "ceil")
        np.testing.assert_array_equal(result, np.array([2.0, 3.0, 4.0]))

    def test_floor(self):
        scores = np.array([1.2, 2.7, 3.9])
        result = trunc(scores, "floor")
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_around(self):
        scores = np.array([1.2, 2.7, 3.5])
        result = trunc(scores, "around")
        np.testing.assert_array_equal(result, np.array([1.0, 3.0, 4.0]))

    def test_negative_values(self):
        scores = np.array([-1.2, -2.7, -3.1])
        result = trunc(scores, "ceil")
        np.testing.assert_array_equal(result, np.array([-1.0, -2.0, -3.0]))


class TestCutoff:
    def test_basic(self):
        scores = np.array([0.1, 0.5, 0.8, 0.3, 0.9])
        result = cutoff(scores, threshold=0.5)
        np.testing.assert_array_equal(result, np.array([0, 0, 1, 0, 1]))

    def test_all_above_threshold(self):
        scores = np.array([0.6, 0.7, 0.8, 0.9])
        result = cutoff(scores, threshold=0.5)
        np.testing.assert_array_equal(result, np.array([1, 1, 1, 1]))

    def test_all_below_threshold(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        result = cutoff(scores, threshold=0.5)
        np.testing.assert_array_equal(result, np.array([0, 0, 0, 0]))

    def test_exact_threshold(self):
        scores = np.array([0.5])
        result = cutoff(scores, threshold=0.5)
        np.testing.assert_array_equal(result, np.array([0]))

    def test_zero_threshold(self):
        scores = np.array([0.0, 0.1, -0.1])
        result = cutoff(scores, threshold=0.0)
        np.testing.assert_array_equal(result, np.array([0, 1, 0]))


class TestBinaryClf:
    def test_basic(self):
        trues = np.array([1, 0, 1, 0, 1])
        preds = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        fps, tps = _binary_clf_curve(trues, preds)
        # At highest threshold (0.9): 1 true positive, 0 false positive
        assert tps[0] == 1
        assert fps[0] == 0

    def test_perfect_predictions(self):
        trues = np.array([1, 1, 0, 0])
        preds = np.array([0.9, 0.8, 0.2, 0.1])
        fps, tps = _binary_clf_curve(trues, preds)
        # At max threshold position, should have 2 TP and 0 FP at some point
        assert tps[-1] == 2
        assert fps[-1] == 2

    def test_all_positive(self):
        trues = np.array([1, 1, 1])
        preds = np.array([0.9, 0.8, 0.7])
        fps, tps = _binary_clf_curve(trues, preds)
        assert tps[-1] == 3
        assert fps[-1] == 0

    def test_all_negative(self):
        trues = np.array([0, 0, 0])
        preds = np.array([0.9, 0.8, 0.7])
        fps, tps = _binary_clf_curve(trues, preds)
        assert tps[-1] == 0
        assert fps[-1] == 3
