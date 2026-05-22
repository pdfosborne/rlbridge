# Reinforcement Learning Bridge (rlbridge)

**Reinforcement Learning Bridge** connects LLMs to reinforcement learning (RL) agents.

RL agents can be used to automate complex decision making problems. rlbridge is the first attempt at giving LLMs the ability to use RL directly to train agents. 

LLMs can interact with rl problems directly (e.g. Claude plays Pokemon), using rlbridge massively reduces token costs and learns to optimize the agent for the problem. 

rlbridge automates how LLMs construct RL problems in language, including: building environments, transforming observations to language, and training agents with auto generated sub-tasks without user supervision.
 
LLMs interact with rlbridge through an **MCP plugin** compatible with Claude Code, Claude Desktop, Cursor, Windsurf, Codex CLI and local models through LM Studio and OpenCode.

![interface_example](https://raw.githubusercontent.com/pdfosborne/rlbridge/refs/heads/main/docs/_images/interface_example.png)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Claude Code                                         │
│                                                      │
│  "Run 100 steps of CartPole with a random policy"    │
│         │                                            │
│   MCP layer (stdio)                                  │
└──────────┼───────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────┐
│  RL Bridge MCP Plugin  (rlbridge.mcp_plugin)             │
│  FastMCP tools:  rl_create · rl_reset · rl_step      │
│                 rl_render · rl_close · rl_run_episode│
│         │                                            │
│   In-process dispatcher                              │
└──────────┼────────────────────────────┬──────────────┘
           │                            │
           │     ╌╌ OR ╌╌ (RLIP_SERVER_URL)
           │                            │
┌──────────▼─────────────────┐  ┌──────▼────────────────┐
│  RL Bridge Server           │  │  Live Training        │
│  (HTTP/JSON-RPC 2.0)        │  │  Dashboard            │
│  POST /rpc  ·  GET /docs    │  │  (http://localhost:7860)
└──────────┬─────────────────┘  │  WebSocket/HTTP       │
           │                    │  Real-time reward     │
           │                    │  tracking & policy    │
           │                    │  visualization        │
           │                    └──────────────────────┘
           │
┌──────────▼───────────────────────────────────────────┐
│  Environment Registry + Session Manager              │
│  ┌───────────────┐  ┌───────────────┐                │
│  │ CartPole-v1   │  │ LunarLander   │  + any custom  │
│  │ (Gymnasium)   │  │ (Gymnasium)   │    env …       │
│  └───────────────┘  └───────────────┘                │
└──────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

From PyPI (recommended):

```bash
pip install rlbridge

# Classic Gymnasium envs for examples (CartPole, etc.)
pip install "rlbridge[examples]"
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv pip install rlbridge
# or, in a uv-managed project:
uv add rlbridge
uv add "rlbridge[examples]"
```

From source (development):

```bash
git clone https://github.com/pdfosborne/RL-IP
cd RL-IP
pip install -e ".[dev,examples]"
# or: uv sync --extra dev --extra examples
```

### 2. Add to your AI tool

Run the command for whichever tool(s) you use, then restart the client:

```bash
# Claude Code (CLI)  → ~/.claude.json
rlbridge install-claude

# Claude Desktop (GUI app)
#   macOS   → ~/Library/Application Support/Claude/claude_desktop_config.json
#   Windows → %APPDATA%\Claude\claude_desktop_config.json
#   Linux   → ~/.config/Claude/claude_desktop_config.json
rlbridge install-claude-desktop

# LM Studio  → ~/.lmstudio/mcp.json  (Linux)
#              ~/Library/Application Support/LM Studio/mcp.json  (macOS)
#              %APPDATA%\LM Studio\mcp.json  (Windows)
rlbridge install-lmstudio

# Cursor  → ~/.cursor/mcp.json
rlbridge install-cursor

# Windsurf  → ~/.codeium/windsurf/mcp_config.json
rlbridge install-windsurf

# Codex CLI  → ~/.codex/config.toml
rlbridge install-codex

# OpenCode   → ~/.config/opencode/config.json
rlbridge install-opencode
```

All commands accept `--use-script` (uses the `rlbridge-mcp` console script instead
of `python -m`) and `--config-path` to override the default config location.

### 3. Use it

Open Claude Code and ask:

> *"Run a CartPole episode with a tabular agent and render the optimal policy."*

Claude will call `rl_create`, `rl_reset`, `rl_step` (in a loop), and `rl_close` automatically.

You can try using Claude directly to make actions in the environment:

> *"Run a blackjack episode with an llm agent with language translation and show me the token cost."*

For the default end-to-end training pipeline, ask:

> *"Run the default experiment process on the easy Sailing environment and show me the best agent when it finishes."*

Claude will call `rl_experiment_process` once, then poll `rl_get_training_result(job_id)` until completion.

`rl_experiment_process` runs the full 8-stage workflow in one job:
dashboard, baseline training, language-state training, instruction derivation,
instruction matching, instruction-shaped training, instruction+language training,
and final evaluation/comparison.

The dashboard is viewable at the default localhost url:

http://localhost:7432/

---

## Available MCP Tools

### Environment control

| Tool | Description |
|------|-------------|
| `rl_list_environments` | Browse all registered RL environments |
| `rl_create` | Create an environment instance |
| `rl_reset` | Reset an instance, get the initial observation |
| `rl_step` | Execute one action, get `(obs, reward, terminated, truncated, info)` |
| `rl_sample_action` | Sample a random valid action |
| `rl_spaces` | Inspect observation and action space details |
| `rl_render` | Render current state (PNG or ASCII) |
| `rl_close` | Destroy an instance |
| `rl_list_instances` | List all active instances |
| `rl_run_episode` | Run a complete episode in one call |

### Custom environment builder

Build and register new environments—wrapping any Gymnasium env—with custom
metadata, local caching, and language translation.

| Tool | Description |
|------|-------------|
| `rl_build_environment` | Wrap a Gymnasium env with custom ID, description, and tags; cache to `~/.rlbridge/envs/` and write to user catalog |
| `rl_list_cached_environments` | Browse previously built environments stored in the local cache |
| `rl_load_cached_environments` | Re-register all cached environments at session start |

**Example workflow in Claude Code:**

> *"Build me a FrozenLake environment called FrozenLake-Custom-v0 with tags grid and discrete."*

```
rl_build_environment("FrozenLake-Custom-v0", "FrozenLake-v1",
                     description="Custom FrozenLake.", tags="grid,discrete")
```

### Language translation

Map raw environment observations to natural-language descriptions.  Required
for instruction-following and sub-goal reward shaping.

| Tool | Description |
|------|-------------|
| `rl_sample_states_for_translation` | Randomly explore an environment and display raw observed states so a `translate()` function can be written |
| `rl_set_translator_code` | Compile, validate, and install a Python `translate()` function; saves it to `~/.rlbridge/envs/<env_id>/translator.py` so it reloads automatically |
| `rl_translate_state` | Test a single state → natural-language description round-trip |

**Example workflow in Claude Code:**

> *"Sample some states from FrozenLake-Custom-v0 so I can write a translator."*

```
rl_sample_states_for_translation("FrozenLake-Custom-v0", n_samples=20)
```

> *"Install this translator:"*

```python
def translate(state, *, legal_moves=None, action_history=None):
    row, col = divmod(int(state), 4)
    labels = {0: "start", 15: "goal"}
    return f"Agent at row {row}, column {col}. {labels.get(state, '')}"
```

```
rl_set_translator_code("FrozenLake-Custom-v0", python_code="...")
rl_translate_state("FrozenLake-Custom-v0", state="15")
```

The installed translator is automatically used by `rl_match_instruction` and
`rl_train_agent` with sub-goal shaping.

### Instruction-following

| Tool | Description |
|------|-------------|
| `rl_match_instruction` | Explore an environment, translate states to language, and find the state best matching a natural-language goal |
| `rl_instruction_run_episode` | Run a shaped episode where the matched state provides a bonus reward signal |

For semantic matching with Hugging Face sentence-transformers models:

```bash
pip install -e "."[sentence-transformers]
```

Then choose a model directly in the tool call:

```python
# Uses the default model for sentence-transformers
rl_match_instruction("Sailing-v0", "sail toward the beach", encoder="sentence-transformers")

# Uses a specific Hugging Face model id
rl_match_instruction(
    "Sailing-v0",
    "sail toward the beach",
    encoder="sentence",
    encoder_model="BAAI/bge-small-en-v1.5",
)
```

### RL agent training

| Tool | Description |
|------|-------------|
| `rl_list_agents` | See available agent types (`tabular_q`, `dqn`, `ppo`) with guidance on when to use each |
| `rl_train_agent` | Train an agent; optionally combine with a `match_id` for instruction-shaped rewards |
| `rl_run_agent_episode` | Evaluate a trained agent for one greedy episode |
| `rl_render_policy` | Render the best training episode as an animated GIF |

### Local LLM policy agent (Python API)

RL Bridge also includes a direct `local_llm` policy agent for action selection
without gradient training.

```python
from rlbridge.environments.registry import registry
from rlbridge.language_translation import get_translator
from rlbridge.rl_agents import LocalLLMAgent

env = registry.get("Sailing-v0").create()
translator = get_translator("Sailing-v0")

agent = LocalLLMAgent(base_url="http://localhost:11434/v1", model="llama3.1")

obs = env.reset().observation
obs_text = translator.translate(obs) if translator else str(obs)
action = agent.choose_action(obs_text, action_space=env.action_space)
step = env.step(action)
```

Use this agent primarily with language-translated observations. Calling a model
for every environment action can be expensive in long episodes, so account for
per-step latency and token cost.

---

## Full Example: New Environment from Scratch

```python
# 1. Build and cache the environment
rl_build_environment(
    env_id="FrozenLake-Custom-v0",
    gym_env_id="FrozenLake-v1",
    description="4×4 frozen lake grid-world.",
    tags="grid,discrete",
    namespace="custom",
)

# 2. Sample states to understand the observation format
rl_sample_states_for_translation("FrozenLake-Custom-v0", n_samples=16)
# → STATE 1: 0  STATE 2: 1  STATE 3: 5  …

# 3. Install a translator
rl_set_translator_code("FrozenLake-Custom-v0", python_code="""
def translate(state, *, legal_moves=None, action_history=None):
    row, col = divmod(int(state), 4)
    cell = {0: "start (S)", 5: "hole (H)", 10: "hole (H)", 15: "goal (G)"}.get(state, "frozen (F)")
    return f"Agent is at row {row}, column {col} - {cell}."
""")

# 4. Match an instruction to a goal state
rl_match_instruction("FrozenLake-Custom-v0", "reach the goal")
# → match_id: abc123

# 5. Train an agent with sub-goal shaping
rl_train_agent("dqn", "FrozenLake-Custom-v0", n_episodes=500, match_id="abc123")
# → agent_id: def456

# 6. Render the result
rl_render_policy("FrozenLake-Custom-v0", agent_id="def456")
```

---

## Standalone HTTP Server

Run RL Bridge as a standalone service (useful for multi-process workflows or
connecting non-Python agents):

```bash
rlbridge server --port 8765
```

Then send JSON-RPC requests:

```bash
curl -X POST http://localhost:8765/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "rlbridge/environment/create",
    "params": {"env_id": "CartPole-v1"}
  }'
```

Browse the auto-generated API docs at `http://localhost:8765/docs`.

---

## Remote RL Bridge Server from the MCP Plugin

```bash
# Point the plugin at a remote server instead of running envs in-process
RLIP_SERVER_URL=http://my-gpu-machine:8765 rlbridge mcp
```

---

## Custom Environments (Python API)

For programmatic use, the `EnvironmentBuilder` Python API mirrors the MCP tools:

```python
from rlbridge.environments.builder import EnvironmentBuilder, load_cached_environments

# Build, cache, and register in one call
built = (
    EnvironmentBuilder("FrozenLake-Custom-v0")
    .from_gymnasium("FrozenLake-v1")
    .with_metadata(description="Custom FrozenLake.", tags=["grid", "discrete"])
    .build()   # auto_register=True, update_catalog=True by default
)

# Attach a hand-written translator
built.translator = ...   # any LanguageTranslator instance
built.register(register_translator=True)

# Reload all cached environments at startup
load_cached_environments()
```

For a fully custom (non-Gymnasium) environment, subclass the ABCs directly:

```python
from rlbridge.environments.base import RLIPEnvironment, RLIPEnvironmentFactory
from rlbridge.environments.registry import registry
from rlbridge.protocol.messages import DiscreteSpace, EnvironmentInfo, ResetResult, StepResult, RenderResult

class MyEnv(RLIPEnvironment):
    def reset(self, seed=None, options=None) -> ResetResult: ...
    def step(self, action) -> StepResult: ...
    def close(self): ...
    @property
    def observation_space(self): return DiscreteSpace(n=10)
    @property
    def action_space(self): return DiscreteSpace(n=4)
    def render(self) -> RenderResult: ...

class MyFactory(RLIPEnvironmentFactory):
    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(env_id="MyEnv-v0", description="My custom env", namespace="custom")
    def create(self, render_mode=None, **kwargs) -> MyEnv:
        return MyEnv()

registry.register(MyFactory())
```

---

## Third-Party Environment Plugins

RL Bridge can load additional environments from separately installed pip packages.
Packages register factories via the ``rlbridge.environments`` entry-point group and
optional MCP tools via ``rlbridge.environment_mcp_tools``.

Example: [Flesh and Blood](https://fabtcg.com/) TCG environments live in the
[`flesh-and-blood-rlbridge`](https://github.com/pdfosborne/flesh-and-blood-rlbridge) package (not bundled with RL Bridge):

```bash
# Install RL Bridge from PyPI
pip install rlbridge

# Then install the FaB plugin from GitHub
pip install git+https://github.com/pdfosborne/flesh-and-blood-rlbridge.git
```

After installation, environments appear in the registry automatically:

```python
from rlbridge.environments.registry import registry

env = registry.create("FleshAndBlood-Talishar-v0", format="silver_age")
```

To publish your own plugin, add to ``pyproject.toml``:

```toml
[project.entry-points."rlbridge.environments"]
my-env = "my_package:register_environments"

[project.entry-points."rlbridge.environment_mcp_tools"]
my-env = "my_package:register_mcp_tools"
```

Each callable receives ``registry=`` (environments) or
``mcp=``, ``registry=``, ``log=`` (MCP tools) and returns the number of
items registered.

---

## Protocol

See [docs/protocol_spec.md](docs/protocol_spec.md) for the full JSON-RPC protocol
specification, including all method signatures, space encodings, error codes,
and the session lifecycle diagram.

---

## Project Layout

```
src/rlbridge/
├── __init__.py
├── __main__.py              # CLI (rlbridge server / rlbridge install-claude / …)
├── protocol/
│   ├── constants.py         # Method names, error codes
│   └── messages.py          # Pydantic message models
├── environments/
│   ├── base.py              # RLIPEnvironment / RLIPEnvironmentFactory ABCs
│   ├── builder.py           # EnvironmentBuilder - fluent API for custom envs
│   ├── gymnasium_adapter.py # Gymnasium wrapper
│   ├── registry.py          # EnvironmentRegistry singleton
│   └── utils.py             # NumPy ↔ JSON helpers, space serialisation
├── instruction_matching/
│   ├── base.py              # BaseEncoder ABC
│   ├── tfidf.py             # TFIDFEncoder (default)
│   ├── bm25.py              # BM25Encoder
│   └── sentence_transformer.py  # SentenceEncoder (optional dep)
├── language_translation/
│   ├── base.py              # LanguageTranslator ABC
│   ├── caching.py           # CachingTranslator + translation cache
│   ├── generator.py         # TranslatorGenerator / GeneratedTranslator (LLM)
│   └── sailing.py           # Built-in Sailing translator
├── server/
│   ├── dispatcher.py        # Transport-agnostic JSON-RPC dispatcher
│   ├── rlip_server.py       # FastAPI HTTP server
│   ├── session.py           # Session / instance manager
│   └── exceptions.py        # RLIPError
├── transport/
│   ├── stdio_transport.py   # Stdio (newline-delimited JSON-RPC)
│   └── http_client.py       # Sync + async HTTP clients
└── mcp_plugin/
    └── plugin.py            # FastMCP MCP plugin for Claude Code
examples/
├── cartpole_http.py         # HTTP client episode example
├── custom_env.py            # Registering a custom environment
└── in_process_usage.py      # Using RL Bridge without a server
docs/
└── protocol_spec.md         # Full JSON-RPC protocol specification
```

### Console scripts

| Command | Entry point | Purpose |
|---------|-------------|---------|
| `rlbridge` | `rlbridge.__main__:app` | CLI (server, MCP, install-*, catalog, …) |
| `rlbridge-mcp` | `rlbridge.mcp_plugin.plugin:main` | MCP stdio plugin for AI tools |
| `rlbridge-server` | `rlbridge.__main__:server_app` | Standalone HTTP JSON-RPC server |

### Optional dependency extras

| Extra | Install | Use case |
|-------|---------|----------|
| `examples` / `envs-classic` | `pip install "rlbridge[examples]"` | CartPole and other classic-control envs |
| `envs-box2d` | `pip install "rlbridge[envs-box2d]"` | LunarLander, etc. |
| `envs-atari` | `pip install "rlbridge[envs-atari]"` | Atari games |
| `envs-mujoco` | `pip install "rlbridge[envs-mujoco]"` | MuJoCo envs |
| `envs-all` | `pip install "rlbridge[envs-all]"` | All Gymnasium env groups |
| `torch` | `pip install "rlbridge[torch]"` | DQN / PPO training agents |
| `sentence-transformers` | `pip install "rlbridge[sentence-transformers]"` | Semantic instruction matching |
| `openai-sdk` | `pip install "rlbridge[openai-sdk]"` | Official OpenAI client (optional) |
| `dev` | `pip install "rlbridge[dev]"` | pytest, ruff, mypy, build, twine |

---

## Publishing (maintainers)

Build and upload to PyPI (test first on [TestPyPI](https://test.pypi.org/)):

```bash
# Install build tools
pip install "rlbridge[dev]"
# or: uv sync --extra dev

# Bump version in pyproject.toml (single source of truth; __version__ reads it at runtime)

# Build sdist + wheel
python -m build
# or: uv build

# Upload (requires PyPI account + API token)
twine upload dist/*
# or: uv publish
```

User install after release:

```bash
pip install rlbridge
uv add rlbridge
```

**Note:** The PyPI distribution name is `rlbridge`. The Python import package and console commands are `rlbridge`, `rlbridge-mcp`, and `rlbridge-server`.

**Manual steps before first release:** create a PyPI account, generate an API token, and tag releases on GitHub.

---

## License

Apache License 2.0 - see [LICENSE](LICENSE).

The framework used in this work is currently patent pending with the US Patent and Trademark Office (18/955718). 


### Cite

Please use the following to cite this work

```bibtex
@phdthesis{OsborneThesis2024,
  title        = {Improving Real-World Reinforcement Learning by Self Completing Human Instructions on Rule Defined Language},  
  author       = {Philip Osborne},  
  year         = 2024,  
  month        = {August},  
  address      = {Manchester, UK},  
  note         = {Available at \url{https://research.manchester.ac.uk/en/studentTheses/improving-real-world-reinforcement-learning-by-self-completing-hu}},  
  school       = {The University of Manchester},  
  type         = {PhD thesis}
}

```