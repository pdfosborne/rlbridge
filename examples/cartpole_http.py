"""
CartPole Example - RLIP HTTP Client
=====================================
Demonstrates using the RLIP HTTP client to run a CartPole episode.

Prerequisites
-------------
1. Install RLIP with examples:
       pip install -e ".[examples]"

2. Start the RLIP server in a separate terminal:
       rlip server

3. Run this script:
       python examples/cartpole_http.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, "../src")  # for running from the examples/ directory

from rlip.transport.http_client import RLIPClient, RLIPClientError


def run_cartpole_episode(
    base_url: str = "http://localhost:8765",
    seed: int = 42,
    max_steps: int = 500,
) -> None:
    with RLIPClient(base_url) as client:
        # ── Handshake ────────────────────────────────────────────────────────
        info = client.initialize(client_name="cartpole-example")
        print(f"Connected to: {info.server_name} v{info.server_version}")
        print(f"Protocol:     RLIP {info.protocol_version}\n")

        # ── Create environment ────────────────────────────────────────────────
        env = client.create_environment("CartPole-v1", render_mode=None)
        print(f"Created CartPole-v1  (instance: {env.instance_id[:8]}…)")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space:      {env.action_space}\n")

        # ── Reset ─────────────────────────────────────────────────────────────
        reset = client.reset(env.instance_id, seed=seed)
        print(f"Initial obs: {reset.observation}")

        total_reward = 0.0
        step_count = 0

        import random
        rng = random.Random(seed)

        # ── Episode loop ──────────────────────────────────────────────────────
        for step in range(max_steps):
            action = rng.randint(0, 1)  # random policy
            result = client.step(env.instance_id, action=action)

            total_reward += result.reward
            step_count += 1

            if step < 5 or result.terminated or result.truncated:
                print(
                    f"  step {step_count:4d}  action={action}  "
                    f"reward={result.reward:+.1f}  "
                    f"terminated={result.terminated}"
                )

            if result.terminated or result.truncated:
                break

        print(f"\nEpisode finished in {step_count} steps, total reward: {total_reward:.1f}")

        # ── Close ─────────────────────────────────────────────────────────────
        client.close_environment(env.instance_id)
        print("Environment closed.")


if __name__ == "__main__":
    try:
        run_cartpole_episode()
    except RLIPClientError as e:
        print(f"RLIP error: {e}")
    except Exception as e:
        print(f"Error: {e}\n\nMake sure the RLIP server is running: rlip server")
