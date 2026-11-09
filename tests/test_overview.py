"""Fitting any number of episodes into the overview's columns."""
from qgate.overview import columns


def test_a_small_dataset_gets_one_column_per_episode():
    assert columns([3, 1, 2], max_cols=10) == [[1], [2], [3]]


def test_a_large_dataset_is_split_into_even_runs_of_consecutive_episodes():
    cols = columns(list(range(1000)), max_cols=300)
    assert len(cols) == 300
    assert [i for c in cols for i in c] == list(range(1000))             # every episode once, in order
    assert {len(c) for c in cols} <= {3, 4}
