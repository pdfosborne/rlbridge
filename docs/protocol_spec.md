# RLIP Protocol Specification v0.1

**Reinforcement Learning Interaction Protocol** - a JSON-RPC 2.0-based protocol
for connecting AI agents to reinforcement learning environments.

---

## Overview

RLIP is a stateful session protocol where a **client** (e.g. Claude Code) opens
one or more *environment instances* on a **server**, orchestrates episodes by
calling `reset` / `step`, and closes instances when done.

The protocol is transport-agnostic: it runs over HTTP POST, stdio
(newline-delimited JSON), or any other reliable byte stream.

---

## Transport

### HTTP

- **Endpoint:** `POST /rpc`
- **Content-Type:** `application/json`
- **Batch endpoint:** `POST /rpc/batch` (accepts a JSON array of requests)
- All responses use HTTP 200; errors are communicated in the JSON-RPC `error` field.

### Stdio

- One JSON object per line (newline-delimited).
- Responses are written to stdout; logging goes to stderr.
- Used by the Claude Code MCP plugin.

---

## Message Format

All messages conform to **JSON-RPC 2.0**.

### Request

```json
{
  "jsonrpc": "2.0",
  "id": "<uuid-string>",
  "method": "rlip/environment/step",
  "params": { ... }
}
```

### Success Response

```json
{
  "jsonrpc": "2.0",
  "id": "<uuid-string>",
  "result": { ... }
}
```

### Error Response

```json
{
  "jsonrpc": "2.0",
  "id": "<uuid-string>",
  "error": {
    "code": -32001,
    "message": "Environment instance not found",
    "data": { "instance_id": "abc…" }
  }
}
```

---

## Error Codes

| Code    | Name                  | Description                                    |
|---------|-----------------------|------------------------------------------------|
| -32700  | PARSE_ERROR           | Invalid JSON                                   |
| -32600  | INVALID_REQUEST       | JSON-RPC envelop is malformed                  |
| -32601  | METHOD_NOT_FOUND      | Unknown method name                            |
| -32602  | INVALID_PARAMS        | Missing or wrong parameter types               |
| -32603  | INTERNAL_ERROR        | Unexpected server-side error                   |
| -32000  | ENV_NOT_FOUND         | `env_id` not in registry                       |
| -32001  | INSTANCE_NOT_FOUND    | `instance_id` is not an active instance        |
| -32002  | ENV_NOT_INITIALIZED   | `step` called before `reset`                   |
| -32003  | INVALID_ACTION        | Action incompatible with action space          |
| -32004  | RENDER_UNAVAILABLE    | Environment does not support rendering         |
| -32005  | ENV_CREATION_FAILED   | `gym.make()` or factory threw an exception     |
| -32006  | PROTOCOL_MISMATCH     | Client/server protocol version incompatible    |

---

## Methods

### `rlip/initialize`

Exchange capabilities and verify the protocol version.

**Params**

| Field             | Type   | Required | Description                     |
|-------------------|--------|----------|---------------------------------|
| `client_name`     | string | ✓        | Human-readable client name      |
| `client_version`  | string |          | Client version string           |
| `protocol_version`| string |          | RLIP version (default `"0.1"`)  |

**Result**

| Field              | Type   | Description                   |
|--------------------|--------|-------------------------------|
| `server_name`      | string | e.g. `"RLIP Server"`          |
| `server_version`   | string | Server package version        |
| `protocol_version` | string | Negotiated protocol version   |
| `capabilities`     | object | Feature flags                 |

---

### `rlip/environments/list`

List environments registered with the server.

**Params**

| Field       | Type       | Description                              |
|-------------|------------|------------------------------------------|
| `tags`      | string[]   | Filter by all of these tags (AND match)  |
| `namespace` | string     | Filter by namespace, e.g. `"gymnasium"`  |

**Result**

| Field          | Type              | Description                    |
|----------------|-------------------|--------------------------------|
| `environments` | EnvironmentInfo[] | Metadata for each environment  |
| `total`        | integer           | Total count                    |

**EnvironmentInfo**

```json
{
  "env_id":            "CartPole-v1",
  "description":       "Classic cart-pole balancing",
  "version":           "1",
  "tags":              ["classic-control"],
  "reward_threshold":  475.0,
  "max_episode_steps": 500,
  "namespace":         "gymnasium",
  "render_modes":      ["human", "rgb_array", "ansi"]
}
```

---

### `rlip/environment/create`

Instantiate an environment.  Returns an `instance_id` for subsequent calls.

**Params**

| Field         | Type   | Required | Description                              |
|---------------|--------|----------|------------------------------------------|
| `env_id`      | string | ✓        | Registered environment ID                |
| `render_mode` | string |          | `"rgb_array"`, `"ansi"`, or `null`       |
| `kwargs`      | object |          | Extra kwargs forwarded to `gym.make()`   |

**Result**

| Field               | Type             | Description                    |
|---------------------|------------------|--------------------------------|
| `instance_id`       | string           | UUID for this instance         |
| `env_id`            | string           | Confirmed environment ID       |
| `observation_space` | SpaceDescription | Serialised observation space   |
| `action_space`      | SpaceDescription | Serialised action space        |

---

### `rlip/environment/reset`

Reset an instance to its initial state.  **Must be called before the first step.**

**Params**

| Field         | Type    | Required | Description              |
|---------------|---------|----------|--------------------------|
| `instance_id` | string  | ✓        |                          |
| `seed`        | integer |          | RNG seed                 |
| `options`     | object  |          | Forwarded to `env.reset` |

**Result**

| Field         | Type | Description             |
|---------------|------|-------------------------|
| `observation` | any  | Initial observation     |
| `info`        | object | Auxiliary information |

---

### `rlip/environment/step`

Execute one action and advance the environment by one timestep.

**Params**

| Field         | Type   | Required | Description                      |
|---------------|--------|----------|----------------------------------|
| `instance_id` | string | ✓        |                                  |
| `action`      | any    | ✓        | Must match the action space type |

Action encoding per space type:

| Space          | JSON value example       |
|----------------|--------------------------|
| `Discrete(4)`  | `2`                      |
| `Box([...], …)`| `[0.5, -0.3, 1.0]`       |
| `MultiBinary`  | `[1, 0, 1, 0]`           |
| `MultiDiscrete`| `[2, 0, 3]`              |
| `Dict`         | `{"x": 1, "y": [0.5]}`   |

**Result**

| Field         | Type    | Description                            |
|---------------|---------|----------------------------------------|
| `observation` | any     | Next observation                       |
| `reward`      | float   | Immediate reward                       |
| `terminated`  | boolean | Episode ended naturally (goal/failure) |
| `truncated`   | boolean | Episode ended by time / step limit     |
| `info`        | object  | Auxiliary diagnostic data              |

---

### `rlip/environment/spaces`

Retrieve the space descriptions for an existing instance.

**Params:** `instance_id`

**Result:** `observation_space`, `action_space` (SpaceDescription objects)

---

### `rlip/environment/render`

Render the current state of an environment.

**Params:** `instance_id`

**Result**

| Field   | Type    | Description                              |
|---------|---------|------------------------------------------|
| `mode`  | string  | `"rgb_array"` or `"ansi"`                |
| `data`  | string? | Base64-encoded PNG (rgb_array mode only) |
| `text`  | string? | ASCII string (ansi mode only)            |
| `width` | integer?| Image width in pixels                    |
| `height`| integer?| Image height in pixels                   |

---

### `rlip/environment/close`

Destroy an environment instance and free its resources.

**Params:** `instance_id`

**Result:** `{ "closed": true, "instance_id": "…" }`

---

### `rlip/instances/list`

List all currently active environment instances on the server.

**Params:** *(none)*

**Result:** `{ "instances": [...], "total": N }`

---

## Space Descriptions

All space objects carry a discriminating `"type"` field.

### Discrete

```json
{ "type": "Discrete", "n": 4, "start": 0 }
```

### Box

```json
{
  "type":  "Box",
  "low":   [-4.8, -inf, -0.418, -inf],
  "high":  [ 4.8,  inf,  0.418,  inf],
  "shape": [4],
  "dtype": "float32"
}
```

### MultiBinary

```json
{ "type": "MultiBinary", "n": 8 }
```

### MultiDiscrete

```json
{ "type": "MultiDiscrete", "nvec": [5, 3, 2] }
```

### Tuple

```json
{ "type": "Tuple", "spaces": [ {...}, {...} ] }
```

### Dict

```json
{ "type": "Dict", "spaces": { "velocity": {...}, "position": {...} } }
```

---

## Session Lifecycle

```
client                          server
  │── rlip/initialize ─────────────►│
  │◄─ {server_name, capabilities} ──│
  │                                 │
  │── rlip/environments/list ───────►│
  │◄─ {environments: [...]} ────────│
  │                                 │
  │── rlip/environment/create ──────►│  (returns instance_id)
  │◄─ {instance_id, spaces} ────────│
  │                                 │
  │── rlip/environment/reset ───────►│
  │◄─ {observation, info} ──────────│
  │                                 │
  │   ┌─ episode loop ─────────┐    │
  │   │ rlip/environment/step ─►│   │
  │   │◄─ {obs, reward, done}  │   │
  │   └────────────────────────┘    │
  │                                 │
  │── rlip/environment/close ───────►│
  │◄─ {closed: true} ───────────────│
```

---

## Extending RLIP

To add a custom environment:

1. Subclass `RLIPEnvironment` and implement `reset`, `step`, `close`, `render`,
   `observation_space`, and `action_space`.
2. Subclass `RLIPEnvironmentFactory`, set `env_info`, implement `create`.
3. Call `registry.register(MyFactory())` before the server starts.

See `examples/custom_env.py` for a complete walkthrough.
