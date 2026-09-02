"""Regression tests for per-run RNG snapshots."""

import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BeingH.train.rng_state import (  # noqa: E402
    capture_rng_state,
    load_rng_state,
    save_rng_state,
)


def _draw_values():
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(4),
    )


def test_round_trip_restores_all_rng_streams(tmp_path):
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    state = capture_rng_state()
    expected = _draw_values()

    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    save_rng_state(tmp_path, state=state)
    assert load_rng_state(tmp_path) is not None

    actual = _draw_values()
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_rank_snapshots_are_isolated_per_checkpoint(tmp_path):
    first = tmp_path / "0000001"
    second = tmp_path / "0000002"

    torch.manual_seed(101)
    first_state = capture_rng_state()
    save_rng_state(first, state=first_state)

    torch.manual_seed(202)
    second_state = capture_rng_state()
    save_rng_state(second, state=second_state)

    assert (first / "rng_state.pkl").is_file()
    assert (second / "rng_state.pkl").is_file()

    load_rng_state(first)
    first_values = torch.rand(3)
    torch.manual_seed(101)
    torch.testing.assert_close(first_values, torch.rand(3))

    load_rng_state(second)
    second_values = torch.rand(3)
    torch.manual_seed(202)
    torch.testing.assert_close(second_values, torch.rand(3))


def test_nonzero_rank_does_not_fall_back_to_rank_zero(tmp_path):
    torch.manual_seed(7)
    save_rng_state(tmp_path, rank=0)
    assert load_rng_state(tmp_path, rank=1) is None
