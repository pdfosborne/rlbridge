"""
OpenAI Function-Calling Agent for rlbridge
========================================
Works with any OpenAI-compatible endpoint - Ollama, OpenAI, LM Studio, etc.

The agent loop:
  1. User sends a natural-language task.
  2. rlbridge tools are registered as OpenAI "tools" (JSON schema function defs).
  3. The model responds with tool_calls.
  4. Each call is dispatched to the in-process rlbridge dispatcher (or HTTP server).
  5. Results are fed back as tool messages.
  6. Loop until the model returns a plain text response.

Usage
-----
    from rlbridge.adapters.openai_agent import RLIPAgent

    agent = RLIPAgent(base_url="http://localhost:11434/v1", model="llama3.1")
    response = agent.run("Run a CartPole episode with a random policy")
    print(response)

CLI
---
    rlbridge agent --model llama3.1 "Run a CartPole episode"
    rlbridge agent --base-url http://localhost:11434/v1 --model mistral "List available envs"
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ..environments.registry import EnvironmentRegistry, registry as default_registry
from ..server.dispatcher import RLIPDispatcher
from ..server.session import SessionManager

log = logging.getLogger(__name__)

# ── rlbridge tool definitions in OpenAI function-calling schema ──────────────────

RLIP_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rl_list_environments",
            "description": (
                "List all available RL environments registered with rlbridge. "
                "Use to discover what environments are available before creating one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags to filter by, e.g. 'classic-control'.",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Filter by namespace, e.g. 'gymnasium'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_create",
            "description": (
                "Create a new RL environment instance. "
                "Returns an instance_id used for all subsequent calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "env_id": {
                        "type": "string",
                        "description": "Gymnasium environment ID, e.g. 'CartPole-v1'.",
                    },
                    "render_mode": {
                        "type": "string",
                        "description": "Optional: 'rgb_array' or 'ansi'. Leave empty for no rendering.",
                    },
                },
                "required": ["env_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_reset",
            "description": (
                "Reset an environment instance to its initial state. "
                "MUST be called before the first rl_step. Returns the initial observation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": "The instance_id returned by rl_create.",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Optional integer seed for reproducibility.",
                    },
                },
                "required": ["instance_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_step",
            "description": (
                "Execute one action in the environment and advance by one timestep. "
                "Returns observation, reward, terminated, truncated, info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": "The instance_id returned by rl_create.",
                    },
                    "action": {
                        "type": "string",
                        "description": (
                            "Action to execute. "
                            "Discrete: integer string e.g. '1'. "
                            "Box: JSON array e.g. '[0.5, -0.3]'. "
                            "Dict: JSON object."
                        ),
                    },
                },
                "required": ["instance_id", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_sample_action",
            "description": "Sample a uniformly random valid action from the environment's action space.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": "The instance_id returned by rl_create.",
                    },
                },
                "required": ["instance_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_spaces",
            "description": "Get the observation and action space descriptions for an instance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": "The instance_id returned by rl_create.",
                    },
                },
                "required": ["instance_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_render",
            "description": (
                "Render the current state. "
                "Returns a base64 PNG for rgb_array mode, or ASCII text for ansi mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": "The instance_id returned by rl_create.",
                    },
                },
                "required": ["instance_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_close",
            "description": "Close and destroy an environment instance, freeing its resources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": "The instance_id returned by rl_create.",
                    },
                },
                "required": ["instance_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_list_instances",
            "description": "List all currently active environment instances.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rl_run_episode",
            "description": (
                "Convenience: create an env, run a full episode with a random policy, "
                "close it, and return a summary. Good for quick benchmarks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "env_id": {
                        "type": "string",
                        "description": "Gymnasium environment ID, e.g. 'CartPole-v1'.",
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum steps before truncating the episode.",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Optional RNG seed.",
                    },
                },
                "required": ["env_id"],
            },
        },
    },
]


# ── Tool dispatcher (re-uses MCP plugin's tool functions) ────────────────────

def _call_rlip_tool(name: str, arguments: dict[str, Any], dispatcher: RLIPDispatcher) -> str:
    """
    Execute one rlbridge tool call by delegating to the same plugin functions
    used by the MCP adapter - no code duplication.
    """
    # Import the actual tool implementations from the MCP plugin
    import rlbridge.mcp_plugin.plugin as _plugin

    # Patch the plugin's dispatcher reference to the one we were given
    original = _plugin._dispatcher if _plugin._in_process else None
    if _plugin._in_process and original is not dispatcher:
        _plugin._dispatcher = dispatcher

    try:
        fn = getattr(_plugin, name, None)
        if fn is None:
            return f"Unknown tool: {name}"
        result = fn(**arguments)
        return str(result)
    except Exception as exc:
        return f"Tool error ({name}): {exc}"
    finally:
        if _plugin._in_process and original is not None:
            _plugin._dispatcher = original


# ── Agent ─────────────────────────────────────────────────────────────────────

class RLIPAgent:
    """
    OpenAI function-calling agent that connects an Ollama (or any
    OpenAI-compatible) model to rlbridge environments.

    Parameters
    ----------
    base_url:
        OpenAI-compatible chat completions base URL.
        Ollama default: ``http://localhost:11434/v1``
        OpenAI: ``https://api.openai.com/v1``
    model:
        Model name served at *base_url*, e.g. ``llama3.1``, ``mistral``,
        ``gpt-4o``.
    api_key:
        API key (required for OpenAI, ignored by Ollama).
    max_iterations:
        Safety cap on tool-call rounds per ``run()`` call.
    registry:
        Environment registry to use (defaults to the global singleton).
    max_instances:
        Maximum concurrent environment instances.
    system_prompt:
        Override the default system prompt given to the model.
    """

    DEFAULT_SYSTEM = (
        "You are an AI agent with access to reinforcement learning environments "
        "via rlbridge tools. When asked to interact with an RL environment, use the "
        "provided tools in the correct order: rl_create → rl_reset → rl_step (loop) "
        "→ rl_close. Always reset before the first step. Report observations, "
        "rewards, and episode outcomes clearly."
    )

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.1",
        api_key: str = "ollama",
        max_iterations: int = 64,
        registry: EnvironmentRegistry | None = None,
        max_instances: int = 16,
        system_prompt: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self._timeout = timeout

        reg = registry or default_registry
        session = SessionManager(max_instances=max_instances)
        self._dispatcher = RLIPDispatcher(registry=reg, session=session)

        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, task: str) -> str:
        """
        Send *task* to the model and run the tool-calling loop until the model
        produces a final text answer.  Returns that final answer.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        for iteration in range(self.max_iterations):
            response = self._chat(messages)
            message = response["choices"][0]["message"]
            messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                # Model gave a final text response
                return message.get("content") or ""

            # Execute all tool calls and append results
            for tc in tool_calls:
                call_id = tc["id"]
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"].get("arguments", "{}"))
                except json.JSONDecodeError:
                    fn_args = {}

                log.debug("Tool call: %s(%s)", fn_name, fn_args)
                result = _call_rlip_tool(fn_name, fn_args, self._dispatcher)
                log.debug("Tool result: %s", result[:200])

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })

        return "Max iterations reached without a final answer."

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RLIPAgent":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": messages,
            "tools": RLIP_TOOLS,
            "tool_choice": "auto",
            "stream": False,
        }
        resp = self._http.post("/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()
