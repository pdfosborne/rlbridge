"""
Local LLM Policy Agent
======================
Query a local OpenAI-compatible model (Ollama, LM Studio, etc.) for each
environment step and use the model response as the action.

This is intended for interactive experiments and debugging, not high-throughput
training. A model call is made for every action, so wall-clock cost can be
significant.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Optional

import httpx

from .._agent_base import AgentBase, TrainResult, _get


class LocalLLMAgent(AgentBase):
    """
    LLM-backed policy that selects actions by prompting a local model.

    Notes
    -----
    - Best results are usually achieved with language-translated observations
      (plain text state descriptions). Raw numeric tensors/arrays often produce
      unstable action choices.
    - The model is called once per action. For long episodes this can be slow.

    Parameters
    ----------
    model:
        Model identifier exposed by the local OpenAI-compatible endpoint.
    base_url:
        Base URL for OpenAI-compatible chat completions.
    api_key:
        API key header value. For local providers this is often ignored.
    system_prompt:
        Optional override for the default action-selection prompt.
    timeout:
        Request timeout in seconds.
    temperature:
        Sampling temperature used when requesting actions.
    max_tokens:
        Max completion tokens per action call.
    """

    name = "local_llm"

    DEFAULT_SYSTEM_PROMPT = (
        "You are a policy model for a reinforcement-learning environment. "
        "Choose exactly one valid action based on the current observation. "
        "Return only JSON with shape {\"action\": <value>} and no extra text."
    )

    def __init__(
        self,
        *,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        system_prompt: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._api_key = api_key
        self._timeout = timeout
        self._warned_obs_shape = False
        self._http = self._build_client()

    def act(self, obs: Any) -> Any:
        """Choose an action from *obs* using only observation context."""
        return self.choose_action(obs)

    def choose_action(
        self,
        obs: Any,
        *,
        action_space: Any = None,
        legal_actions: list[Any] | None = None,
        action_history: list[Any] | None = None,
    ) -> Any:
        """
        Query the local model and return one action.

        Prefer passing language-translated observations (strings). If the
        observation is not a string, a one-time warning is emitted.
        """
        if not isinstance(obs, str) and not self._warned_obs_shape:
            warnings.warn(
                "LocalLLMAgent usually performs best on language-translated states; "
                "raw numeric observations can reduce action quality.",
                stacklevel=2,
            )
            self._warned_obs_shape = True

        prompt = self._build_user_prompt(
            obs=obs,
            action_space=action_space,
            legal_actions=legal_actions,
            action_history=action_history,
        )
        content = self._chat(prompt)
        return self._parse_action(content, action_space=action_space, legal_actions=legal_actions)

    def run_episode(
        self,
        env: Any,
        *,
        max_steps: int = 200,
        seed: Optional[int] = None,
        close_env: bool = False,
    ) -> dict[str, Any]:
        """
        Run one episode by querying the local model for every step.

        Returns a summary dict with total reward and trajectory details.
        """
        reset_out = env.reset(seed=seed)
        obs = _get(reset_out, "observation", reset_out)

        action_space = getattr(env, "action_space", None)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        trajectory: list[dict[str, Any]] = []

        for _ in range(max_steps):
            action = self.choose_action(obs, action_space=action_space)
            step_out = env.step(action)
            next_obs = _get(step_out, "observation", obs)
            reward = float(_get(step_out, "reward", 0.0))
            terminated = bool(_get(step_out, "terminated", False))
            truncated = bool(_get(step_out, "truncated", False))
            info = _get(step_out, "info", {})

            trajectory.append({
                "observation": obs,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            })

            total_reward += reward
            steps += 1
            obs = next_obs

            if terminated or truncated:
                break

        if close_env:
            env.close()

        return {
            "agent": self.name,
            "model": self.model,
            "steps": steps,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "final_observation": obs,
            "trajectory": trajectory,
        }

    def train(self, env: Any, n_episodes: int, **kwargs: Any) -> TrainResult:
        """Training is not supported for this policy-only agent."""
        del env, n_episodes, kwargs
        raise NotImplementedError(
            "LocalLLMAgent does not implement gradient-based training. "
            "Use run_episode() for direct action selection."
        )

    def save(self, path: str | Path) -> None:
        """Save agent configuration to JSON."""
        data = {
            "agent": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Load agent configuration from JSON."""
        data = json.loads(Path(path).read_text())
        self.model = str(data.get("model", self.model))
        self.base_url = str(data.get("base_url", self.base_url)).rstrip("/")
        self.system_prompt = str(data.get("system_prompt", self.system_prompt))
        self.temperature = float(data.get("temperature", self.temperature))
        self.max_tokens = int(data.get("max_tokens", self.max_tokens))
        self._http.close()
        self._http = self._build_client()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LocalLLMAgent":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _build_user_prompt(
        self,
        *,
        obs: Any,
        action_space: Any,
        legal_actions: list[Any] | None,
        action_history: list[Any] | None,
    ) -> str:
        space = self._serialize_action_space(action_space)
        payload = {
            "observation": obs,
            "action_space": space,
            "legal_actions": legal_actions,
            "recent_actions": action_history[-5:] if action_history else [],
            "instruction": "Choose the next action and return strict JSON: {\"action\": ...}",
        }
        return json.dumps(payload, ensure_ascii=True)

    def _build_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )

    def _chat(self, user_prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        response = self._http.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"].get("content", "")).strip()

    def _parse_action(
        self,
        text: str,
        *,
        action_space: Any,
        legal_actions: list[Any] | None,
    ) -> Any:
        candidate = _extract_json_object(text)
        if isinstance(candidate, dict) and "action" in candidate:
            action = candidate["action"]
        else:
            action = _fallback_parse_scalar(text)

        if legal_actions:
            if action in legal_actions:
                return action
            for option in legal_actions:
                if str(option) == str(action):
                    return option
            raise ValueError(f"Model returned illegal action {action!r}; legal actions={legal_actions!r}")

        n = _discrete_n(action_space)
        if n is not None:
            action_int = int(action)
            if action_int < 0 or action_int >= n:
                raise ValueError(f"Model returned out-of-range action {action_int}; expected [0, {n - 1}]")
            return action_int
        return action

    @staticmethod
    def _serialize_action_space(action_space: Any) -> Any:
        if action_space is None:
            return None
        if isinstance(action_space, dict):
            return action_space

        space_type = getattr(action_space, "type", None)
        if isinstance(space_type, str):
            if space_type == "Discrete":
                return {"type": "Discrete", "n": int(getattr(action_space, "n", 0))}
            return {
                "type": space_type,
                "shape": getattr(action_space, "shape", None),
            }

        if hasattr(action_space, "n"):
            return {"type": "Discrete", "n": int(action_space.n)}

        return repr(action_space)


def _extract_json_object(text: str) -> Any:
    """Best-effort parse for JSON object, including fenced code blocks."""
    text = text.strip()
    if not text:
        return None

    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _fallback_parse_scalar(text: str) -> Any:
    """Fallback parser for plain scalar outputs (e.g. '1')."""
    text = text.strip()
    if not text:
        raise ValueError("Model returned an empty action response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    num_match = re.search(r"-?\d+", text)
    if num_match:
        return int(num_match.group(0))

    return text


def _discrete_n(action_space: Any) -> Optional[int]:
    if action_space is None:
        return None
    if isinstance(action_space, dict) and action_space.get("type") == "Discrete":
        n = action_space.get("n")
        return int(n) if n is not None else None

    space_type = getattr(action_space, "type", None)
    if space_type == "Discrete":
        return int(getattr(action_space, "n", 0))
    if hasattr(action_space, "n"):
        return int(action_space.n)
    return None
