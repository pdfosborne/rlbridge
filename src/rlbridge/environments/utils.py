"""
Utility helpers for serialising NumPy / Gymnasium types to JSON-compatible
Python objects and reconstructing them for environment calls.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

from ..protocol.messages import (
    BoxSpace,
    DiscreteSpace,
    DictSpace,
    MultiBinarySpace,
    MultiDiscreteSpace,
    SpaceDescription,
    TupleSpace,
    TextSpace,
)


# ── Numpy ────────────────────────────────────────────────────────────────────

def numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(v) for v in obj]
    return obj


# ── Space serialisation ──────────────────────────────────────────────────────

def space_to_description(space: "spaces.Space") -> SpaceDescription:  # type: ignore[name-defined]
    """Convert a Gymnasium space to an rlbridge SpaceDescription."""
    if not _GYM_AVAILABLE:
        raise ImportError("gymnasium is required for space_to_description")

    if isinstance(space, spaces.Discrete):
        return DiscreteSpace(n=int(space.n), start=int(space.start))

    if isinstance(space, spaces.Box):
        return BoxSpace(
            low=space.low.tolist(),
            high=space.high.tolist(),
            shape=list(space.shape),
            dtype=str(space.dtype),
        )

    if isinstance(space, spaces.MultiBinary):
        n = space.n
        return MultiBinarySpace(n=int(n) if isinstance(n, (int, np.integer)) else list(n))

    if isinstance(space, spaces.MultiDiscrete):
        return MultiDiscreteSpace(nvec=space.nvec.tolist())

    if isinstance(space, spaces.Tuple):
        return TupleSpace(spaces=[space_to_description(s) for s in space.spaces])

    if isinstance(space, spaces.Dict):
        return DictSpace(spaces={k: space_to_description(v) for k, v in space.spaces.items()})

    if isinstance(space, spaces.Text):
        return TextSpace(
            min_length=space.min_length,
            max_length=space.max_length,
        )

    # Fallback: treat as opaque Box
    raise TypeError(f"Unsupported gymnasium space type: {type(space)}")


def description_to_space(desc: SpaceDescription) -> "spaces.Space":  # type: ignore[name-defined]
    """Reconstruct a Gymnasium space from an rlbridge SpaceDescription."""
    if not _GYM_AVAILABLE:
        raise ImportError("gymnasium is required for description_to_space")

    if isinstance(desc, DiscreteSpace):
        return spaces.Discrete(n=desc.n, start=desc.start)

    if isinstance(desc, BoxSpace):
        return spaces.Box(
            low=np.array(desc.low, dtype=desc.dtype),
            high=np.array(desc.high, dtype=desc.dtype),
            shape=tuple(desc.shape),
            dtype=desc.dtype,
        )

    if isinstance(desc, MultiBinarySpace):
        return spaces.MultiBinary(desc.n)

    if isinstance(desc, MultiDiscreteSpace):
        return spaces.MultiDiscrete(nvec=np.array(desc.nvec))

    if isinstance(desc, TupleSpace):
        return spaces.Tuple(tuple(description_to_space(s) for s in desc.spaces))

    if isinstance(desc, DictSpace):
        return spaces.Dict({k: description_to_space(v) for k, v in desc.spaces.items()})

    if isinstance(desc, TextSpace):
        return spaces.Text(min_length=desc.min_length, max_length=desc.max_length)

    raise TypeError(f"Unsupported SpaceDescription type: {type(desc)}")


# ── Action parsing ────────────────────────────────────────────────────────────

def parse_action(action_json: Any, space: "spaces.Space") -> Any:  # type: ignore[name-defined]
    """Convert a JSON-decoded action value to the type expected by the space."""
    if not _GYM_AVAILABLE:
        return action_json

    if isinstance(space, spaces.Discrete):
        return int(action_json)

    if isinstance(space, spaces.Box):
        arr = np.array(action_json, dtype=space.dtype)
        if arr.shape != space.shape:
            arr = arr.reshape(space.shape)
        return arr

    if isinstance(space, spaces.MultiBinary):
        return np.array(action_json, dtype=np.int8)

    if isinstance(space, spaces.MultiDiscrete):
        return np.array(action_json, dtype=np.int64)

    if isinstance(space, spaces.Tuple):
        if isinstance(action_json, (list, tuple)):
            return tuple(parse_action(a, s) for a, s in zip(action_json, space.spaces))
        return action_json

    if isinstance(space, spaces.Dict):
        if isinstance(action_json, dict):
            return {k: parse_action(action_json[k], space.spaces[k]) for k in space.spaces}
        return action_json

    return action_json


# ── Rendering ────────────────────────────────────────────────────────────────

def rgb_array_to_base64_png(frame: np.ndarray) -> str:
    """Convert an HxWx3 uint8 RGB array to a base64-encoded PNG string."""
    try:
        from PIL import Image  # type: ignore[import]
        img = Image.fromarray(frame.astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        # Minimal raw-bytes fallback (not a valid PNG without Pillow)
        return base64.b64encode(frame.tobytes()).decode("ascii")
