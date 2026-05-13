# RLIP — Reinforcement Learning Interaction Protocol

**RLIP** is a JSON-RPC 2.0 protocol—modelled after MCP—for connecting AI agents
to reinforcement learning environments.  It ships as a **Claude Code MCP plugin**,
letting Claude interact with Gymnasium environments directly via natural language.

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
│  RLIP MCP Plugin  (rlip.mcp_plugin)                  │
│  FastMCP tools:  rl_create · rl_reset · rl_step      │
│                 rl_render · rl_close · rl_run_episode│
│         │                                            │
│   In-process RLIP Dispatcher                         │
└──────────┼───────────────────────────────────────────┘
           │           ╌╌ OR ╌╌ (RLIP_SERVER_URL)
┌──────────▼───────────────────────────────────────────┐
│  RLIP Server (HTTP/JSON-RPC 2.0)                     │
│  POST /rpc  ·  GET /environments  ·  GET /health     │
└──────────┬───────────────────────────────────────────┘
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

```bash
git clone https://github.com/your-org/rlip
cd rlip
pip install -e ".[examples]"
```

### 2. Add to Claude Code

```bash
rlip install-claude
```

This writes your `~/.claude.json` MCP configuration.  Restart Claude Code.

### 3. Use it

Open Claude Code and ask:

> *"Run a CartPole episode with a random policy and show me the total reward."*

Claude will call `rl_create`, `rl_reset`, `rl_step` (in a loop), and `rl_close` automatically.

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
| `rl_build_environment` | Wrap a Gymnasium env with custom ID, description, and tags; cache to `~/.rlip/envs/` and write to user catalog |
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
| `rl_set_translator_code` | Compile, validate, and install a Python `translate()` function; saves it to `~/.rlip/envs/<env_id>/translator.py` so it reloads automatically |
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

### RL agent training

| Tool | Description |
|------|-------------|
| `rl_list_agents` | See available agent types (`tabular_q`, `dqn`, `ppo`) with guidance on when to use each |
| `rl_train_agent` | Train an agent; optionally combine with a `match_id` for instruction-shaped rewards |
| `rl_run_agent_episode` | Evaluate a trained agent for one greedy episode |
| `rl_render_policy` | Render the best training episode as an animated GIF |

### Local LLM policy agent (Python API)

RLIP also includes a direct `local_llm` policy agent for action selection
without gradient training.

```python
from rlip.environments.registry import registry
from rlip.language_translation import get_translator
from rlip.rl_agents import LocalLLMAgent

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
    return f"Agent is at row {row}, column {col} — {cell}."
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

Run RLIP as a standalone service (useful for multi-process workflows or
connecting non-Python agents):

```bash
rlip server --port 8765
```

Then send JSON-RPC requests:

```bash
curl -X POST http://localhost:8765/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "rlip/environment/create",
    "params": {"env_id": "CartPole-v1"}
  }'
```

Browse the auto-generated API docs at `http://localhost:8765/docs`.

---

## Remote RLIP Server from the MCP Plugin

```bash
# Point the plugin at a remote server instead of running envs in-process
RLIP_SERVER_URL=http://my-gpu-machine:8765 rlip mcp
```

---

## Custom Environments (Python API)

For programmatic use, the `EnvironmentBuilder` Python API mirrors the MCP tools:

```python
from rlip.environments.builder import EnvironmentBuilder, load_cached_environments

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
from rlip.environments.base import RLIPEnvironment, RLIPEnvironmentFactory
from rlip.environments.registry import registry
from rlip.protocol.messages import DiscreteSpace, EnvironmentInfo, ResetResult, StepResult, RenderResult

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

## Language Translation (Python API)

```python
from rlip.language_translation.generator import TranslatorGenerator, build_translator

def my_llm(prompt: str) -> str:
    ...  # wrap any LLM provider

# Option A: two-stage LLM pipeline (describe → synthesise rules)
translator = build_translator(
    env, llm_fn=my_llm,
    env_context="4×4 grid, state = integer 0–15.",
    n_samples=20,
)

# Option B: hand-written rules with LLM fallback
from rlip.language_translation.generator import GeneratedTranslator

translator = GeneratedTranslator(llm_fn=my_llm, env_id="MyEnv-v0")
translator.rule_code = """
def translate(state, *, legal_moves=None, action_history=None):
    row, col = divmod(int(state), 4)
    return f"row {row}, col {col}"
"""

# Save as a standalone module (no LLM dependency at runtime)
translator.save_code("my_env_translator.py")
```

Generated translators fall back to live LLM calls for states the rule function
cannot handle, cache those answers, and auto-refine the rules once the cache
grows past `refine_threshold`.

---

## Proxy Mode (MCP ↔ Remote RLIP)

```
┌─────────────┐   stdio/MCP   ┌───────────────────┐   HTTP/JSON-RPC   ┌──────────────┐
│ Claude Code │ ────────────► │  RLIP MCP Plugin  │ ────────────────► │ RLIP Server  │
│             │               │ (thin proxy)      │                   │ (GPU box)    │
└─────────────┘               └───────────────────┘                   └──────────────┘
```

Set `RLIP_SERVER_URL` to enable proxy mode.

---

## Protocol

See [docs/protocol_spec.md](docs/protocol_spec.md) for the full RLIP specification,
including all method signatures, space encodings, error codes, and the session
lifecycle diagram.

---

## Project Layout

```
src/rlip/
├── __init__.py
├── __main__.py              # CLI (rlip server / rlip install-claude / rlip list)
├── protocol/
│   ├── constants.py         # Method names, error codes
│   └── messages.py          # Pydantic message models
├── environments/
│   ├── base.py              # RLIPEnvironment / RLIPEnvironmentFactory ABCs
│   ├── builder.py           # EnvironmentBuilder — fluent API for custom envs
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
catalog.json                 # Built-in public environment catalog
examples/
├── cartpole_http.py         # HTTP client episode example
├── custom_env.py            # Registering a custom environment
└── in_process_usage.py      # Using RLIP without a server
docs/
└── protocol_spec.md         # Full RLIP specification
```

---

## License

MIT

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