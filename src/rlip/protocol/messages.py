"""
RLIP Protocol Message Models
==============================
All request/response envelopes are JSON-RPC 2.0.
RLIP-specific payload models are defined as Pydantic v2 models.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from .constants import RLIP_PROTOCOL_VERSION, JSONRPC_VERSION


# ── JSON-RPC 2.0 envelopes ───────────────────────────────────────────────────

class RpcRequest(BaseModel):
    """JSON-RPC 2.0 request envelope."""
    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[dict[str, Any]] = None


class RpcError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class RpcResponse(BaseModel):
    """JSON-RPC 2.0 response envelope."""
    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: str
    result: Optional[Any] = None
    error: Optional[RpcError] = None


# ── Space descriptions ────────────────────────────────────────────────────────

class DiscreteSpace(BaseModel):
    type: Literal["Discrete"] = "Discrete"
    n: int
    start: int = 0


class BoxSpace(BaseModel):
    type: Literal["Box"] = "Box"
    low: list[float]
    high: list[float]
    shape: list[int]
    dtype: str


class MultiBinarySpace(BaseModel):
    type: Literal["MultiBinary"] = "MultiBinary"
    n: Union[int, list[int]]


class MultiDiscreteSpace(BaseModel):
    type: Literal["MultiDiscrete"] = "MultiDiscrete"
    nvec: list[int]


class TupleSpace(BaseModel):
    type: Literal["Tuple"] = "Tuple"
    spaces: list[SpaceDescription]


class DictSpace(BaseModel):
    type: Literal["Dict"] = "Dict"
    spaces: dict[str, SpaceDescription]


class TextSpace(BaseModel):
    type: Literal["Text"] = "Text"
    min_length: int = 0
    max_length: Optional[int] = None


SpaceDescription = Union[
    DiscreteSpace,
    BoxSpace,
    MultiBinarySpace,
    MultiDiscreteSpace,
    TupleSpace,
    DictSpace,
    TextSpace,
]

# Resolve forward references
TupleSpace.model_rebuild()
DictSpace.model_rebuild()


# ── Environment metadata ──────────────────────────────────────────────────────

class SuggestedHyperparameters(BaseModel):
    """
    Recommended training hyperparameters for a specific environment.

    Returned by ``factory.env_info.suggested_hyperparameters`` and exposed via
    the ``rl_get_suggested_hyperparameters()`` MCP tool.  All fields are
    optional; absent values mean "use the global default".
    """
    agent_type: str = "tabular_q"
    # Episode counts
    n_episodes: int = 300
    max_steps: int = 200
    # Instruction-matching
    sub_goal_threshold: float = 0.5
    top_k: int = 3
    min_episode_visits: int = 2
    # Tabular-Q hyperparameters
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    # DQN / PPO shared
    hidden_size: int = 64
    lr: float = 1e-3


class EnvironmentInfo(BaseModel):
    env_id: str
    description: str = ""
    version: str = ""
    tags: list[str] = Field(default_factory=list)
    reward_threshold: Optional[float] = None
    max_episode_steps: Optional[int] = None
    namespace: str = "gymnasium"
    render_modes: list[str] = Field(default_factory=list)
    suggested_hyperparameters: Optional[SuggestedHyperparameters] = None


# ── rlip/initialize ───────────────────────────────────────────────────────────

class InitializeParams(BaseModel):
    client_name: str
    client_version: str = "unknown"
    protocol_version: str = RLIP_PROTOCOL_VERSION


class InitializeResult(BaseModel):
    server_name: str
    server_version: str
    protocol_version: str = RLIP_PROTOCOL_VERSION
    capabilities: dict[str, Any] = Field(default_factory=dict)


# ── rlip/environments/list ────────────────────────────────────────────────────

class ListEnvironmentsParams(BaseModel):
    tags: list[str] = Field(default_factory=list)
    namespace: Optional[str] = None


class ListEnvironmentsResult(BaseModel):
    environments: list[EnvironmentInfo]
    total: int


# ── rlip/environment/create ───────────────────────────────────────────────────

class CreateEnvironmentParams(BaseModel):
    env_id: str
    render_mode: Optional[str] = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class CreateEnvironmentResult(BaseModel):
    instance_id: str
    env_id: str
    observation_space: SpaceDescription
    action_space: SpaceDescription


# ── rlip/environment/reset ────────────────────────────────────────────────────

class ResetParams(BaseModel):
    instance_id: str
    seed: Optional[int] = None
    options: dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: Any
    info: dict[str, Any] = Field(default_factory=dict)


# ── rlip/environment/step ─────────────────────────────────────────────────────

class StepParams(BaseModel):
    instance_id: str
    action: Any  # matches the env's action space


class StepResult(BaseModel):
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ── rlip/environment/spaces ───────────────────────────────────────────────────

class SpacesParams(BaseModel):
    instance_id: str


class SpacesResult(BaseModel):
    observation_space: SpaceDescription
    action_space: SpaceDescription


# ── rlip/environment/render ───────────────────────────────────────────────────

class RenderParams(BaseModel):
    instance_id: str


class RenderResult(BaseModel):
    mode: str         # "rgb_array" | "ansi" | "human"
    data: Optional[str] = None   # base64-encoded PNG for rgb_array
    text: Optional[str] = None   # ASCII string for ansi
    width: Optional[int] = None
    height: Optional[int] = None


# ── rlip/environment/close ────────────────────────────────────────────────────

class CloseParams(BaseModel):
    instance_id: str


class CloseResult(BaseModel):
    closed: bool
    instance_id: str


# ── rlip/instances/list ───────────────────────────────────────────────────────

class ListInstancesResult(BaseModel):
    instances: list[dict[str, Any]]
    total: int
