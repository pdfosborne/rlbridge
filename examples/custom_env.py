"""
Custom Environment Example
===========================
Shows how to wrap a fully custom environment (no Gymnasium dependency needed)
and register it with the RLIP registry so it appears in Claude Code.

The GridWorld-1D-v0 environment used here is a simple 1D Grid World:
  - State:  integer position on a line [0..9]
  - Action: 0 = left, 1 = right
  - Reward: +1 when reaching position 9, -0.01 otherwise
  - Done:   reaching position 9

GridWorld-1D-v0 is bundled as a predefined environment, so it is already
registered automatically when RLIP starts.  This script demonstrates how
custom environments work and shows a live episode.

Run this script:
    python examples/custom_env.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, "../src")

from rlip.environments.predefined.gridworld import GridWorldEnv, GridWorldFactory
from rlip.environments.registry import registry


# GridWorld-1D-v0 is registered automatically by the RLIP registry at import
# time (via rlip.environments.predefined.gridworld).  The factory and env
# classes live in that module; we import them here for direct use below.

# ── Run a quick demo episode ──────────────────────────────────────────────────

print(f"GridWorld-1D-v0 in registry: {'GridWorld-1D-v0' in registry}")
print(f"Registry size: {len(registry)} environments\n")

env = registry.create("GridWorld-1D-v0")
reset_result = env.reset(seed=1)
print(f"Initial observation: {reset_result.observation}")

for i in range(20):
    action = env.sample_action()
    step = env.step(action)
    render = env.render()
    print(f"  step {i+1:2d}  action={action}  {render.text}  reward={step.reward:+.2f}")
    if step.terminated:
        print("  DONE – reached goal!")
        break

env.close()
print("\nGridWorld-1D-v0 is available automatically in the MCP plugin.")
print("See rlip/environments/predefined/gridworld.py to inspect the implementation.")
