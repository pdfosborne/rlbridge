"""
In-Process Usage Example
=========================
Use rlbridge entirely in-process (no server, no subprocess) - useful for testing
and for building your own agent loop in Python.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "../src")

from rlbridge.environments.registry import registry
from rlbridge.server.dispatcher import rlbridgeDispatcher
from rlbridge.server.session import SessionManager
from rlbridge.protocol.constants import Methods


def main() -> None:
    # Build the dispatcher directly - no HTTP needed
    session = SessionManager(max_instances=4)
    dispatcher = rlbridgeDispatcher(registry=registry, session=session)

    def rpc(method: str, **params) -> dict:
        req = {"jsonrpc": "2.0", "id": "test", "method": method, "params": params}
        resp = dispatcher.dispatch(req)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["result"]

    # Initialize
    init = rpc(Methods.INITIALIZE, client_name="in-process-example")
    print(f"Server: {init['server_name']} v{init['server_version']}")

    # Browse environments
    envs = rpc(Methods.LIST_ENVIRONMENTS, tags=[], namespace=None)
    print(f"\nAvailable environments: {envs['total']}")
    for e in envs["environments"][:5]:
        print(f"  {e['env_id']}")
    print("  …")

    # Create CartPole
    env_info = rpc(Methods.CREATE_ENVIRONMENT, env_id="CartPole-v1", render_mode=None, kwargs={})
    instance_id = env_info["instance_id"]
    print(f"\nCreated CartPole-v1  (instance: {instance_id[:8]}…)")

    # Run one episode with a balance-pole heuristic
    reset = rpc(Methods.RESET, instance_id=instance_id, seed=0, options={})
    obs = reset["observation"]

    total_reward = 0.0
    for step in range(500):
        # Simple heuristic: push in the direction the pole is leaning
        action = 1 if obs[2] > 0 else 0
        result = rpc(Methods.STEP, instance_id=instance_id, action=action)
        obs = result["observation"]
        total_reward += result["reward"]
        if result["terminated"] or result["truncated"]:
            print(f"Episode ended at step {step + 1}, total reward: {total_reward:.1f}")
            break

    # Clean up
    rpc(Methods.CLOSE, instance_id=instance_id)
    print("Done.")


if __name__ == "__main__":
    main()
