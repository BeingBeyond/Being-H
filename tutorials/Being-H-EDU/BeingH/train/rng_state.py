"""Utilities for checkpointing the random state of a training process.

Random state belongs to a particular training run (and, for distributed
training, to a particular rank).  Keeping the snapshot with the checkpoint
prevents a later run in the same checkout from silently inheriting state from
an unrelated experiment.
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


RNG_STATE_FILENAME = "rng_state.pkl"
_RANK_STATE_TEMPLATE = "rng_state.rank{rank}.pkl"


def capture_rng_state() -> Dict[str, Any]:
    """Capture all RNG streams used by the training input and model code."""

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    # NumPy is an optional dependency for the small utility itself, although
    # it is present in the training environment.  Keeping this optional makes
    # the helper usable by lightweight tests and installations.
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state["numpy"] = np.random.get_state()

    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    if not isinstance(state, dict) or state.get("torch_cpu") is None:
        raise ValueError("RNG checkpoint is missing the torch CPU state")

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            np.random.set_state(numpy_state)

    torch.set_rng_state(state["torch_cpu"])

    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def _atomic_dump(state: Dict[str, Any], path: Path) -> None:
    """Write a pickle without exposing a partially written checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as stream:
            pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _rank_state_path(directory: Path, rank: int) -> Path:
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    return directory / _RANK_STATE_TEMPLATE.format(rank=rank)


def save_rng_state(
    directory: Union[str, os.PathLike[str]],
    rank: int = 0,
    state: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a rank-local RNG state in ``directory``."""

    directory_path = Path(directory)
    state = capture_rng_state() if state is None else state
    rank_path = _rank_state_path(directory_path, rank)
    _atomic_dump(state, rank_path)
    if rank == 0:
        _atomic_dump(state, directory_path / RNG_STATE_FILENAME)
    return rank_path


def load_rng_state(
    directory: Union[str, os.PathLike[str]], rank: int = 0
) -> Optional[Path]:
    """Restore a rank-local state from a checkpoint directory."""

    directory_path = Path(directory)
    rank_path = _rank_state_path(directory_path, rank)
    candidates = [rank_path]
    if rank == 0:
        candidates.append(directory_path / RNG_STATE_FILENAME)

    state_path = next((path for path in candidates if path.is_file()), None)
    if state_path is None:
        return None

    try:
        with state_path.open("rb") as stream:
            state = pickle.load(stream)
        restore_rng_state(state)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to restore RNG state from {state_path}"
        ) from exc
    return state_path
