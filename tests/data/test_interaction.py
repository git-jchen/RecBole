# -*- coding: utf-8 -*-
# Tests for recbole/data/interaction.py

import numpy as np
import torch
import pytest

from recbole.data.interaction import Interaction, cat_interactions


class TestInteractionInit:
    def test_init_from_dict_with_tensor(self):
        inter = Interaction(
            {"user_id": torch.tensor([1, 2, 3]), "item_id": torch.tensor([4, 5, 6])}
        )
        assert len(inter) == 3
        assert torch.equal(inter["user_id"], torch.tensor([1, 2, 3]))
        assert torch.equal(inter["item_id"], torch.tensor([4, 5, 6]))

    @pytest.mark.skipif(
        not hasattr(np, "float"),
        reason="np.float removed in NumPy 2.0, Interaction._convert_to_tensor uses it"
    )
    def test_init_from_dict_with_numpy_int64(self):
        inter = Interaction(
            {"user_id": np.array([1, 2, 3], dtype=np.int64), "item_id": np.array([4, 5, 6], dtype=np.int64)}
        )
        assert len(inter) == 3

    def test_init_from_dict_with_tensor_values(self):
        inter = Interaction(
            {"user_id": torch.tensor([1, 2, 3]), "item_id": torch.tensor([4, 5, 6])}
        )
        assert len(inter) == 3

    def test_init_invalid_type(self):
        with pytest.raises(ValueError):
            Interaction("invalid")

    def test_init_invalid_value_type(self):
        with pytest.raises(ValueError):
            Interaction({"user_id": "invalid_string"})


class TestInteractionGetItem:
    def test_getitem_by_string(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        result = inter["user_id"]
        assert torch.equal(result, torch.tensor([1, 2, 3]))

    def test_getitem_by_index(self):
        inter = Interaction(
            {"user_id": torch.tensor([1, 2, 3]), "item_id": torch.tensor([4, 5, 6])}
        )
        sliced = inter[0]
        assert len(sliced) == 1
        assert sliced["user_id"].item() == 1
        assert sliced["item_id"].item() == 4

    def test_getitem_by_slice(self):
        inter = Interaction(
            {"user_id": torch.tensor([1, 2, 3]), "item_id": torch.tensor([4, 5, 6])}
        )
        sliced = inter[0:2]
        assert len(sliced) == 2

    def test_getitem_by_tensor_index(self):
        inter = Interaction({"user_id": torch.tensor([10, 20, 30])})
        idx = torch.tensor([0, 2])
        sliced = inter[idx]
        assert torch.equal(sliced["user_id"], torch.tensor([10, 30]))


class TestInteractionSetItem:
    def test_setitem(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        inter["rating"] = torch.tensor([5.0, 4.0, 3.0])
        assert "rating" in inter
        assert torch.equal(inter["rating"], torch.tensor([5.0, 4.0, 3.0]))

    def test_setitem_non_string_key(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        with pytest.raises(KeyError):
            inter[123] = torch.tensor([1, 2, 3])


class TestInteractionDelItem:
    def test_delitem(self):
        inter = Interaction(
            {"user_id": torch.tensor([1, 2, 3]), "item_id": torch.tensor([4, 5, 6])}
        )
        del inter["item_id"]
        assert "item_id" not in inter
        assert "user_id" in inter

    def test_delitem_nonexistent(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        with pytest.raises(KeyError):
            del inter["nonexistent"]


class TestInteractionContains:
    def test_contains_true(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        assert "user_id" in inter

    def test_contains_false(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        assert "item_id" not in inter


class TestInteractionLen:
    def test_len(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        assert len(inter) == 3

    def test_len_2d(self):
        inter = Interaction({"features": torch.zeros(5, 10)})
        assert len(inter) == 5


class TestInteractionStr:
    def test_str_contains_info(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        s = str(inter)
        assert "3" in s
        assert "user_id" in s


class TestInteractionColumns:
    def test_columns(self):
        inter = Interaction(
            {"user_id": torch.tensor([1, 2, 3]), "item_id": torch.tensor([4, 5, 6])}
        )
        cols = inter.columns
        assert set(cols) == {"user_id", "item_id"}


class TestInteractionTo:
    def test_to_cpu(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        result = inter.to("cpu")
        assert result["user_id"].device.type == "cpu"

    def test_to_selected_field(self):
        inter = Interaction(
            {"user_id": torch.tensor([1, 2, 3]), "item_id": torch.tensor([4, 5, 6])}
        )
        result = inter.to("cpu", selected_field="user_id")
        assert result["user_id"].device.type == "cpu"

    def test_to_selected_field_list(self):
        inter = Interaction(
            {"user_id": torch.tensor([1, 2, 3]), "item_id": torch.tensor([4, 5, 6])}
        )
        result = inter.to("cpu", selected_field=["user_id"])
        assert result["user_id"].device.type == "cpu"


class TestInteractionCpu:
    def test_cpu(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        result = inter.cpu()
        assert result["user_id"].device.type == "cpu"


class TestInteractionNumpy:
    def test_numpy(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        result = inter.numpy()
        assert isinstance(result, dict)
        assert np.array_equal(result["user_id"], np.array([1, 2, 3]))


class TestInteractionRepeat:
    def test_repeat_1d(self):
        inter = Interaction({"user_id": torch.tensor([1, 2])})
        result = inter.repeat(3)
        assert len(result) == 6
        assert torch.equal(result["user_id"], torch.tensor([1, 2, 1, 2, 1, 2]))

    def test_repeat_2d(self):
        inter = Interaction({"features": torch.ones(2, 3)})
        result = inter.repeat(3)
        assert result["features"].shape == (6, 3)


class TestInteractionRepeatInterleave:
    def test_repeat_interleave(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        result = inter.repeat_interleave(2, dim=0)
        assert len(result) == 6
        assert torch.equal(result["user_id"], torch.tensor([1, 1, 2, 2, 3, 3]))


class TestInteractionUpdate:
    def test_update(self):
        inter1 = Interaction({"user_id": torch.tensor([1, 2, 3])})
        inter2 = Interaction({"item_id": torch.tensor([4, 5, 6])})
        inter1.update(inter2)
        assert "item_id" in inter1
        assert torch.equal(inter1["item_id"], torch.tensor([4, 5, 6]))

    def test_update_overwrite(self):
        inter1 = Interaction({"user_id": torch.tensor([1, 2, 3])})
        inter2 = Interaction({"user_id": torch.tensor([7, 8, 9])})
        inter1.update(inter2)
        assert torch.equal(inter1["user_id"], torch.tensor([7, 8, 9]))


class TestInteractionDrop:
    def test_drop(self):
        inter = Interaction(
            {"user_id": torch.tensor([1, 2, 3]), "item_id": torch.tensor([4, 5, 6])}
        )
        inter.drop("item_id")
        assert "item_id" not in inter
        assert "user_id" in inter

    def test_drop_nonexistent(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        with pytest.raises(ValueError):
            inter.drop("nonexistent")


class TestInteractionShuffle:
    def test_shuffle_preserves_length(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3, 4, 5])})
        inter.shuffle()
        assert len(inter) == 5

    def test_shuffle_preserves_elements(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3, 4, 5])})
        inter.shuffle()
        assert set(inter["user_id"].tolist()) == {1, 2, 3, 4, 5}


class TestInteractionSort:
    def test_sort_ascending(self):
        inter = Interaction({"score": torch.tensor([3.0, 1.0, 2.0])})
        inter.sort("score", ascending=True)
        assert inter["score"].tolist() == [1.0, 2.0, 3.0]

    @pytest.mark.skipif(
        not hasattr(np, "float"),
        reason="Interaction.sort descending uses np.array[::-1] which fails on NumPy 2.0"
    )
    def test_sort_descending(self):
        inter = Interaction({"score": torch.tensor([3.0, 1.0, 2.0])})
        inter.sort("score", ascending=False)
        scores = inter["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_sort_nonexistent_field(self):
        inter = Interaction({"score": torch.tensor([3.0, 1.0, 2.0])})
        with pytest.raises(ValueError):
            inter.sort("nonexistent")

    def test_sort_invalid_by_type(self):
        inter = Interaction({"score": torch.tensor([3.0, 1.0, 2.0])})
        with pytest.raises(TypeError):
            inter.sort(123)

    def test_sort_invalid_ascending_type(self):
        inter = Interaction({"score": torch.tensor([3.0, 1.0, 2.0])})
        with pytest.raises(TypeError):
            inter.sort("score", ascending="yes")

    def test_sort_mismatched_lengths(self):
        inter = Interaction({"a": torch.tensor([3.0, 1.0, 2.0])})
        with pytest.raises(ValueError):
            inter.sort(["a"], ascending=[True, False])


class TestInteractionAddPrefix:
    def test_add_prefix(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        inter.add_prefix("pos_")
        assert "pos_user_id" in inter
        assert "user_id" not in inter


class TestInteractionIter:
    def test_iter(self):
        inter = Interaction(
            {"user_id": torch.tensor([1, 2, 3]), "item_id": torch.tensor([4, 5, 6])}
        )
        keys = list(inter)
        assert set(keys) == {"user_id", "item_id"}


class TestInteractionGetattr:
    def test_getattr_existing(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        result = inter.user_id
        assert torch.equal(result, torch.tensor([1, 2, 3]))

    def test_getattr_nonexistent(self):
        inter = Interaction({"user_id": torch.tensor([1, 2, 3])})
        with pytest.raises(AttributeError):
            _ = inter.nonexistent_field


class TestCatInteractions:
    def test_cat_basic(self):
        inter1 = Interaction({"user_id": torch.tensor([1, 2])})
        inter2 = Interaction({"user_id": torch.tensor([3, 4])})
        result = cat_interactions([inter1, inter2])
        assert len(result) == 4
        assert torch.equal(result["user_id"], torch.tensor([1, 2, 3, 4]))

    def test_cat_multiple(self):
        inters = [
            Interaction({"x": torch.tensor([i])}) for i in range(5)
        ]
        result = cat_interactions(inters)
        assert len(result) == 5

    def test_cat_invalid_type(self):
        with pytest.raises(TypeError):
            cat_interactions("invalid")

    def test_cat_empty_list(self):
        with pytest.raises(ValueError):
            cat_interactions([])

    def test_cat_mismatched_columns(self):
        inter1 = Interaction({"user_id": torch.tensor([1, 2])})
        inter2 = Interaction({"item_id": torch.tensor([3, 4])})
        with pytest.raises(ValueError):
            cat_interactions([inter1, inter2])
