"""
RLIP Agents
============
Self-contained RL agent implementations that train on any RLIP environment.

Each agent exposes the same minimal interface::

    agent.train(env, n_episodes)  → TrainResult
    agent.act(obs)                → action
    agent.save(path) / .load(path)

Available agents
----------------
- :class:`~rlip.agents.tabular_q.TabularQAgent`   – tabular Q-learning (ε-greedy)
- :class:`~rlip.agents.dqn.DQNAgent`              – deep Q-network (NumPy, no framework dep)
- :class:`~rlip.agents.ppo.PPOAgent`              – proximal policy optimisation (NumPy)

All three work with discrete action spaces.  DQN and PPO also handle
continuous ``Box`` observations via flat-vector encoding.

Quick start
-----------
::

    from rlip.environments.predefined.sailing import SAILING_V0
    from rlip.rl_agents import TabularQAgent, DQNAgent, PPOAgent

    env = SAILING_V0.create()

    agent = TabularQAgent(n_actions=2, alpha=0.1, gamma=0.99, epsilon=0.1)
    result = agent.train(env, n_episodes=500)
    print(result)
"""

from .tabular_q import TabularQAgent, TabularQTrainResult
try:
    from .dqn import DQNAgent, DQNTrainResult
except ImportError:
    DQNAgent = None  # type: ignore[assignment,misc]
    DQNTrainResult = None  # type: ignore[assignment,misc]
from .ppo import PPOAgent, PPOTrainResult

__all__ = [
    "TabularQAgent",
    "TabularQTrainResult",
    "DQNAgent",
    "DQNTrainResult",
    "PPOAgent",
    "PPOTrainResult",
]
