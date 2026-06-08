# -*- coding: utf-8 -*-
# Tests for recbole/utils/utils.py

import os
import tempfile
import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import pytest

from recbole.utils.utils import (
    get_local_time,
    ensure_dir,
    get_model,
    get_trainer,
    early_stopping,
    calculate_valid_score,
    dict2str,
    init_seed,
    list_to_latex,
)
from recbole.utils.enum_type import ModelType


class TestGetLocalTime:
    def test_returns_string(self):
        result = get_local_time()
        assert isinstance(result, str)

    def test_format(self):
        result = get_local_time()
        # Format: Mon-DD-YYYY_HH-MM-SS
        parts = result.split("_")
        assert len(parts) == 2
        date_part, time_part = parts
        assert len(date_part.split("-")) == 3
        assert len(time_part.split("-")) == 3


class TestEnsureDir:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "test_dir")
            assert not os.path.exists(new_dir)
            ensure_dir(new_dir)
            assert os.path.exists(new_dir)
            assert os.path.isdir(new_dir)

    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise even if directory already exists
            ensure_dir(tmpdir)
            assert os.path.exists(tmpdir)

    def test_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c")
            ensure_dir(nested)
            assert os.path.exists(nested)


class TestGetModel:
    def test_valid_model(self):
        model_class = get_model("BPR")
        assert model_class.__name__ == "BPR"

    def test_valid_sequential_model(self):
        model_class = get_model("GRU4Rec")
        assert model_class.__name__ == "GRU4Rec"

    def test_invalid_model(self):
        with pytest.raises((ValueError, ModuleNotFoundError)):
            get_model("NonExistentModel12345")


class TestGetTrainer:
    def test_general_model_type(self):
        trainer_class = get_trainer(ModelType.GENERAL, "SomeModel")
        assert trainer_class.__name__ == "Trainer"

    def test_knowledge_model_type(self):
        trainer_class = get_trainer(ModelType.KNOWLEDGE, "SomeModel")
        assert trainer_class.__name__ == "KGTrainer"

    def test_traditional_model_type(self):
        trainer_class = get_trainer(ModelType.TRADITIONAL, "SomeModel")
        assert trainer_class.__name__ == "TraditionalTrainer"


class TestEarlyStopping:
    def test_improvement_bigger_is_better(self):
        best, cur_step, stop, update = early_stopping(
            value=0.9, best=0.8, cur_step=3, max_step=5, bigger=True
        )
        assert best == 0.9
        assert cur_step == 0
        assert stop is False
        assert update is True

    def test_no_improvement_bigger_is_better(self):
        best, cur_step, stop, update = early_stopping(
            value=0.7, best=0.8, cur_step=3, max_step=5, bigger=True
        )
        assert best == 0.8
        assert cur_step == 4
        assert stop is False
        assert update is False

    def test_stop_triggered_bigger_is_better(self):
        best, cur_step, stop, update = early_stopping(
            value=0.7, best=0.8, cur_step=5, max_step=5, bigger=True
        )
        assert best == 0.8
        assert cur_step == 6
        assert stop is True
        assert update is False

    def test_improvement_smaller_is_better(self):
        best, cur_step, stop, update = early_stopping(
            value=0.1, best=0.2, cur_step=3, max_step=5, bigger=False
        )
        assert best == 0.1
        assert cur_step == 0
        assert stop is False
        assert update is True

    def test_no_improvement_smaller_is_better(self):
        best, cur_step, stop, update = early_stopping(
            value=0.3, best=0.2, cur_step=3, max_step=5, bigger=False
        )
        assert best == 0.2
        assert cur_step == 4
        assert stop is False
        assert update is False

    def test_stop_triggered_smaller_is_better(self):
        best, cur_step, stop, update = early_stopping(
            value=0.3, best=0.2, cur_step=5, max_step=5, bigger=False
        )
        assert best == 0.2
        assert cur_step == 6
        assert stop is True
        assert update is False

    def test_equal_value_bigger_is_better(self):
        best, cur_step, stop, update = early_stopping(
            value=0.8, best=0.8, cur_step=2, max_step=5, bigger=True
        )
        assert best == 0.8
        assert cur_step == 0
        assert update is True

    def test_equal_value_smaller_is_better(self):
        best, cur_step, stop, update = early_stopping(
            value=0.2, best=0.2, cur_step=2, max_step=5, bigger=False
        )
        assert best == 0.2
        assert cur_step == 0
        assert update is True


class TestCalculateValidScore:
    def test_with_specific_metric(self):
        result = {"Recall@10": 0.5, "NDCG@10": 0.3}
        score = calculate_valid_score(result, valid_metric="NDCG@10")
        assert score == 0.3

    def test_default_metric(self):
        result = {"Recall@10": 0.5, "NDCG@10": 0.3}
        score = calculate_valid_score(result)
        assert score == 0.5

    def test_none_metric(self):
        result = {"Recall@10": 0.5, "NDCG@10": 0.3}
        score = calculate_valid_score(result, valid_metric=None)
        assert score == 0.5


class TestDict2Str:
    def test_basic(self):
        result = dict2str({"Recall@10": 0.5, "NDCG@10": 0.3})
        assert "Recall@10" in result
        assert "0.5" in result
        assert "NDCG@10" in result
        assert "0.3" in result

    def test_empty_dict(self):
        result = dict2str({})
        assert result == ""

    def test_single_item(self):
        result = dict2str({"metric": 1.0})
        assert "metric : 1.0" in result


class TestInitSeed:
    def test_reproducibility_true(self):
        init_seed(42, reproducibility=True)
        a = torch.rand(5)
        init_seed(42, reproducibility=True)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_reproducibility_false(self):
        init_seed(42, reproducibility=False)
        a = torch.rand(5)
        init_seed(42, reproducibility=False)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds(self):
        init_seed(42, reproducibility=True)
        a = torch.rand(5)
        init_seed(123, reproducibility=True)
        b = torch.rand(5)
        assert not torch.equal(a, b)


class TestListToLatex:
    def test_basic_conversion(self):
        data = [{"model": "BPR", "Recall@10": 0.5}, {"model": "Pop", "Recall@10": 0.3}]
        df, tex = list_to_latex(data)
        assert len(df) == 2
        assert "BPR" in tex or "BPR" in df.values.tolist()[0]

    def test_empty_subset_columns(self):
        data = [{"model": "BPR", "Recall@10": 0.5}]
        df, tex = list_to_latex(data, subset_columns=[])
        assert df is not None
        assert tex is not None
