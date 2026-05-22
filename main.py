"""Interactive RL Bridge training CLI.

Features:
- Select a registered environment.
- Select an existing language translator.
- Provide an instruction and match it to a sub-goal state.
- Train an agent while updating the web dashboard.
- Render policy replay output (GIF when rgb_array is available).
- Generate an evaluation/training report PNG.
"""

from __future__ import annotations

import importlib
import math
import os
import random
import sys
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Allow running from project root without installation.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlip.analysis import create_training_report
from rlip.environments.registry import registry
from rlip.instruction_following import match_instruction, scale_sub_goal_bonus
from rlip.interaction_protocols import GreedyEpisodeProtocol, MultiEpisodeProtocol
from rlip.language_translation import TRANSLATORS, get_translator
from rlip.mcp_plugin._dashboard import dashboard, start_dashboard
from rlip.mcp_plugin._env_wrappers import _LangStateEnv, _ShapedEnv
from rlip.policy_rendering import render_optimal_policy


@dataclass
class TrainConfig:
    agent_type: str
    n_episodes: int
    max_steps: int
    seed: Optional[int]
    use_language_state: bool


@dataclass
class DeckMetaRewardConfig:
    format: str
    inner_agent_type: str
    inner_train_episodes: int
    inner_eval_episodes: int
    matchups_per_deck: int
    inner_max_steps: int


class ProgressEnv:
    """Wrapper that updates the dashboard once per completed episode."""

    def __init__(
        self,
        env: Any,
        *,
        n_episodes: int,
        dashboard_agent_id: str,
        agent_ref: Any,
    ) -> None:
        self._env = env
        self._n_episodes = n_episodes
        self._dashboard_agent_id = dashboard_agent_id
        self._agent_ref = agent_ref
        self._ep_reward = 0.0
        self._reset_calls = 0
        self._completed = 0

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        if self._reset_calls > 0:
            self._completed += 1
            dashboard.update(
                self._dashboard_agent_id,
                completed=min(self._completed, self._n_episodes),
                last_reward=self._ep_reward,
                epsilon=float(getattr(self._agent_ref, "epsilon", 0.0)),
            )
            self._ep_reward = 0.0
        self._reset_calls += 1
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        reward = result.reward if hasattr(result, "reward") else result.get("reward", 0.0)
        self._ep_reward += float(reward)
        return result

    def close(self) -> None:
        if self._reset_calls > 0 and self._completed < self._n_episodes:
            self._completed += 1
            dashboard.update(
                self._dashboard_agent_id,
                completed=min(self._completed, self._n_episodes),
                last_reward=self._ep_reward,
                epsilon=float(getattr(self._agent_ref, "epsilon", 0.0)),
            )
        dashboard.finish(self._dashboard_agent_id)
        self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return getattr(self._env, "env_id", "")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def _ask_int(prompt: str, default: int, minimum: int = 1) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return max(minimum, value)
    except ValueError:
        return default


def _ask_float(prompt: str, default: float, minimum: float = 0.0) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        return max(minimum, value)
    except ValueError:
        return default


def _ask_bool(prompt: str, default: bool = False) -> bool:
    tag = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{tag}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def _select_from_list(title: str, options: list[str], default_idx: int = 0) -> int:
    print(f"\n{title}")
    for i, item in enumerate(options, start=1):
        print(f"  {i:2d}. {item}")
    raw = input(f"Select [default {default_idx + 1}]: ").strip()
    if not raw:
        return default_idx
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return idx
    except ValueError:
        pass
    return default_idx


def _build_agent(agent_type: str, hyperparams: dict[str, Any]) -> Any:
    if agent_type == "tabular_q":
        mod = importlib.import_module("rlip.rl_agents.tabular_q")
        return mod.TabularQAgent(
            alpha=float(hyperparams.get("alpha", 0.1)),
            gamma=float(hyperparams.get("gamma", 0.99)),
            epsilon=float(hyperparams.get("epsilon", 1.0)),
            epsilon_min=float(hyperparams.get("epsilon_min", 0.01)),
            epsilon_decay=float(hyperparams.get("epsilon_decay", 0.995)),
            seed=hyperparams.get("seed"),
        )

    if agent_type == "dqn":
        mod = importlib.import_module("rlip.rl_agents.dqn")
        return mod.DQNAgent(
            hidden_size=int(hyperparams.get("hidden_size", 64)),
            lr=float(hyperparams.get("lr", 1e-3)),
            gamma=float(hyperparams.get("gamma", 0.99)),
            epsilon=float(hyperparams.get("epsilon", 1.0)),
            epsilon_min=float(hyperparams.get("epsilon_min", 0.01)),
            epsilon_decay=float(hyperparams.get("epsilon_decay", 0.995)),
            buffer_size=int(hyperparams.get("buffer_size", 10000)),
            batch_size=int(hyperparams.get("batch_size", 64)),
            target_update_freq=int(hyperparams.get("target_update_freq", 100)),
            seed=hyperparams.get("seed"),
        )

    if agent_type == "ppo":
        mod = importlib.import_module("rlip.rl_agents.ppo")
        return mod.PPOAgent(
            hidden_size=int(hyperparams.get("hidden_size", 64)),
            lr_actor=float(hyperparams.get("lr_actor", 1e-3)),
            lr_critic=float(hyperparams.get("lr_critic", 1e-3)),
            gamma=float(hyperparams.get("gamma", 0.99)),
            lam=float(hyperparams.get("lam", 0.95)),
            clip_eps=float(hyperparams.get("clip_eps", 0.2)),
            n_steps=int(hyperparams.get("n_steps", 256)),
            ppo_epochs=int(hyperparams.get("ppo_epochs", 4)),
            mini_batch_size=int(hyperparams.get("mini_batch_size", 64)),
            seed=hyperparams.get("seed"),
        )

    raise ValueError(f"Unsupported agent type: {agent_type}")


def _run_eval_episode(env: Any, agent: Any, max_steps: int, seed: Optional[int]) -> dict[str, Any]:
    reset_out = env.reset(seed=seed)
    obs = reset_out.observation if hasattr(reset_out, "observation") else reset_out.get("observation", reset_out)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    for step in range(1, max_steps + 1):
        if hasattr(agent, "act_greedy"):
            action = agent.act_greedy(obs)
        else:
            action = agent.act(obs)

        out = env.step(action)
        obs = out.observation if hasattr(out, "observation") else out.get("observation", obs)
        reward = float(out.reward if hasattr(out, "reward") else out.get("reward", 0.0))
        terminated = bool(out.terminated if hasattr(out, "terminated") else out.get("terminated", False))
        truncated = bool(out.truncated if hasattr(out, "truncated") else out.get("truncated", False))
        total_reward += reward
        steps = step
        if terminated or truncated:
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "final_observation": obs,
    }


def _is_fab_env(env_id: str) -> bool:
    return env_id.startswith("FleshAndBlood-")


def _fab_win_probabilities(obs: Any) -> tuple[float, float]:
    """Estimate win probability for each player from a FaB observation.

    Returns ``(agent_win_prob, opponent_win_prob)``.
    """
    if not isinstance(obs, dict):
        return 0.5, 0.5

    agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
    opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}

    agent_life = float(agent.get("life", 0.0))
    opp_life = float(opp.get("life", 0.0))

    # Terminal states should produce deterministic probabilities.
    if opp_life <= 0 < agent_life:
        return 1.0, 0.0
    if agent_life <= 0 < opp_life:
        return 0.0, 1.0

    agent_hand_size = len(agent.get("hand", [])) if isinstance(agent.get("hand"), list) else 0
    opp_hand_size = int(opp.get("hand_size", 0) or 0)

    agent_resources = float(agent.get("resources", 0.0))
    opp_resources = float(opp.get("resources", 0.0))
    agent_ap = float(agent.get("action_points", 0.0))
    opp_ap = float(opp.get("action_points", 0.0))
    agent_deck = float(agent.get("deck", 0.0))
    opp_deck = float(opp.get("deck", 0.0))

    # Base state strength from common FaB signals.
    agent_score = (
        1.8 * agent_life
        + 1.0 * agent_hand_size
        + 0.6 * agent_resources
        + 0.8 * agent_ap
        + 0.05 * agent_deck
    )
    opp_score = (
        1.8 * opp_life
        + 1.0 * opp_hand_size
        + 0.6 * opp_resources
        + 0.8 * opp_ap
        + 0.05 * opp_deck
    )

    # Combat pressure: pending net damage shifts immediate win chances.
    pending = obs.get("pending_combat")
    if isinstance(pending, dict):
        atk = float(pending.get("attack_power", 0.0) or 0.0)
        blk = float(pending.get("total_block", 0.0) or 0.0)
        net = max(0.0, atk - blk)
        attacker = int(pending.get("attacker", 0) or 0)
        if attacker == 0:
            agent_score += 1.5 * net
        else:
            opp_score += 1.5 * net

    # Tempo: active player has initiative.
    active_player = int(obs.get("active_player", 0) or 0)
    if active_player == 0:
        agent_score += 0.4
    else:
        opp_score += 0.4

    diff = (agent_score - opp_score) / 8.0
    agent_p = 1.0 / (1.0 + math.exp(-diff))
    agent_p = max(0.0, min(1.0, agent_p))
    return agent_p, 1.0 - agent_p


def _fab_outcome_score(obs: Any, *, terminated: bool) -> float:
    """Return an agent-centric score in [0, 1] from a final FaB observation."""
    if isinstance(obs, dict):
        agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
        opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}
        agent_life = float(agent.get("life", 0.0) or 0.0)
        opp_life = float(opp.get("life", 0.0) or 0.0)

        if terminated:
            if opp_life <= 0 < agent_life:
                return 1.0
            if agent_life <= 0 < opp_life:
                return 0.0
            if agent_life == opp_life:
                return 0.5

    # For non-terminal or ambiguous outcomes, use the state win-probability estimate.
    p_agent, _ = _fab_win_probabilities(obs)
    return float(p_agent)


def _deck_matchup_key(deck_option: dict[str, Any], matchup_option: dict[str, Any], cfg: DeckMetaRewardConfig) -> tuple[Any, ...]:
    return (
        str(deck_option.get("key", "")),
        str(matchup_option.get("key", "")),
        cfg.format,
        cfg.inner_agent_type,
        cfg.inner_train_episodes,
        cfg.inner_eval_episodes,
        cfg.inner_max_steps,
    )


def _evaluate_deck_vs_matchup(
    *,
    deck_option: dict[str, Any],
    matchup_option: dict[str, Any],
    cfg: DeckMetaRewardConfig,
    play_hyperparams: dict[str, Any],
    seed: Optional[int],
) -> dict[str, Any]:
    """Train a gameplay policy for one deck/matchup pair and report win-rate."""
    env_kwargs = {
        "render_mode": None,
        "format": cfg.format,
        "agent_hero_id": str(deck_option.get("hero_id")),
        "opponent_hero_id": str(matchup_option.get("hero_id")),
        "deck_size": int(deck_option.get("deck_size", 40) or 40),
        "agent_deck_style": str(deck_option.get("style", "balanced")),
        "opponent_deck_style": str(matchup_option.get("style", "balanced")),
    }

    play_agent = _build_agent(cfg.inner_agent_type, dict(play_hyperparams))
    train_env = registry.create("FleshAndBlood-Talishar-v0", **env_kwargs)
    try:
        train_result = play_agent.train(
            train_env,
            n_episodes=cfg.inner_train_episodes,
            max_steps=cfg.inner_max_steps,
            seed=seed,
        )
    finally:
        train_env.close()

    eval_env = registry.create("FleshAndBlood-Talishar-v0", **env_kwargs)
    eval_scores: list[float] = []
    try:
        base_seed = 0 if seed is None else int(seed)
        for ep in range(cfg.inner_eval_episodes):
            ep_seed = base_seed + 10_000 + ep
            out = _run_eval_episode(eval_env, play_agent, max_steps=cfg.inner_max_steps, seed=ep_seed)
            eval_scores.append(
                _fab_outcome_score(
                    out.get("final_observation"),
                    terminated=bool(out.get("terminated", False)),
                )
            )
    finally:
        eval_env.close()

    win_rate = (sum(eval_scores) / len(eval_scores)) if eval_scores else 0.5
    return {
        "win_rate": float(win_rate),
        "train_mean_reward": float(train_result.mean_reward),
        "train_best_reward": float(train_result.best_reward),
    }


class DeckBuildMetaRewardEnv:
    """One-step outer environment: choose deck, receive inner-loop win-rate reward."""

    def __init__(
        self,
        *,
        cfg: DeckMetaRewardConfig,
        play_hyperparams: dict[str, Any],
        seed: Optional[int] = None,
    ) -> None:
        self._cfg = cfg
        self._play_hyperparams = dict(play_hyperparams)
        self._base_seed = seed
        self._episode_idx = 0
        self._rng = random.Random(seed)
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {}

        deck_env = registry.create("FleshAndBlood-DeckBuild-v0", render_mode=None, format=cfg.format)
        try:
            reset_out = deck_env.reset(seed=seed, options={"format": cfg.format, "two_phase_deckbuild": True})
            obs = reset_out.observation if hasattr(reset_out, "observation") else reset_out.get("observation", {})
            options = obs.get("deck_options") if isinstance(obs, dict) else None
            self._deck_options = list(options) if isinstance(options, list) else []
        finally:
            deck_env.close()

        n_actions = max(1, len(self._deck_options))
        self.action_space = {"type": "Discrete", "n": n_actions}
        self.env_id = "FleshAndBlood-DeckBuildMetaReward-v0"

    def _legal_actions(self) -> list[str]:
        return [f"choose_deck {o.get('key', '')}" for o in self._deck_options]

    def reset(self, seed: Any = None, options: Any = None) -> dict[str, Any]:
        if seed is not None:
            self._base_seed = int(seed)
            self._rng.seed(self._base_seed)
        self._episode_idx += 1
        obs = {
            "stage": "deck_selection",
            "format": self._cfg.format,
            "deck_options": self._deck_options,
            "legal_actions": self._legal_actions(),
            "meta_reward_mode": "inner_play_win_rate",
        }
        return {
            "observation": obs,
            "info": {
                "legal_actions": obs["legal_actions"],
                "stage": "deck_selection",
                "meta_reward_mode": "inner_play_win_rate",
            },
        }

    def step(self, action: Any) -> dict[str, Any]:
        if not self._deck_options:
            obs = {
                "stage": "terminal",
                "format": self._cfg.format,
                "selected_deck": None,
                "meta_reward": {"win_rate": 0.5, "error": "No deck options available"},
                "legal_actions": [],
            }
            return {
                "observation": obs,
                "reward": 0.5,
                "terminated": True,
                "truncated": False,
                "info": {"legal_actions": [], "meta_reward": obs["meta_reward"]},
            }

        if isinstance(action, int):
            idx = int(action) % len(self._deck_options)
        elif isinstance(action, str) and action.startswith("choose_deck "):
            key = action.split(" ", 1)[1].strip()
            idx = next((i for i, o in enumerate(self._deck_options) if str(o.get("key")) == key), 0)
        else:
            idx = 0

        deck_choice = dict(self._deck_options[idx])
        opponent_pool = [o for o in self._deck_options if str(o.get("hero_id")) != str(deck_choice.get("hero_id"))]
        if not opponent_pool:
            opponent_pool = list(self._deck_options)

        if self._cfg.matchups_per_deck >= len(opponent_pool):
            sampled = [dict(x) for x in opponent_pool]
        else:
            sampled = [dict(x) for x in self._rng.sample(opponent_pool, self._cfg.matchups_per_deck)]

        details: list[dict[str, Any]] = []
        for i, matchup in enumerate(sampled):
            ep_seed = None if self._base_seed is None else int(self._base_seed) + self._episode_idx * 1000 + i
            cache_key = _deck_matchup_key(deck_choice, matchup, self._cfg)
            cached = self._cache.get(cache_key)
            if cached is None:
                stats = _evaluate_deck_vs_matchup(
                    deck_option=deck_choice,
                    matchup_option=matchup,
                    cfg=self._cfg,
                    play_hyperparams=self._play_hyperparams,
                    seed=ep_seed,
                )
                self._cache[cache_key] = stats
                used_cache = False
            else:
                stats = cached
                used_cache = True

            details.append(
                {
                    "matchup": f"{matchup.get('label', matchup.get('key', 'unknown'))}",
                    "hero_id": matchup.get("hero_id"),
                    "win_rate": float(stats["win_rate"]),
                    "used_cache": used_cache,
                }
            )

        win_rate = (sum(d["win_rate"] for d in details) / len(details)) if details else 0.5
        meta = {
            "win_rate": float(win_rate),
            "selected_deck": deck_choice,
            "matchups_evaluated": len(details),
            "matchup_results": details,
        }
        obs = {
            "stage": "terminal",
            "format": self._cfg.format,
            "selected_deck": deck_choice,
            "meta_reward": meta,
            "legal_actions": [],
        }
        return {
            "observation": obs,
            "reward": float(win_rate),
            "terminated": True,
            "truncated": False,
            "info": {
                "legal_actions": [],
                "meta_reward": meta,
            },
        }

    def close(self) -> None:
        return None

    def sample_action(self) -> int:
        if not self._deck_options:
            return 0
        return self._rng.randrange(len(self._deck_options))


def main() -> int:
    print("RL Bridge Interactive Trainer")
    print("=" * 80)

    env_infos = sorted(registry.list_environments(), key=lambda e: e.env_id.lower())
    env_options = [f"{e.env_id}  |  {e.namespace}  |  {e.description[:80]}" for e in env_infos]
    env_idx = _select_from_list("Available environments", env_options)
    env_info = env_infos[env_idx]
    env_id = env_info.env_id

    agent_options = ["tabular_q", "dqn", "ppo"]
    agent_idx = _select_from_list("Agent type", agent_options)
    agent_type = agent_options[agent_idx]

    n_episodes = _ask_int("Training episodes", 300, minimum=1)
    max_steps = _ask_int("Max steps per episode", 200, minimum=1)
    seed_raw = input("Seed [blank for random]: ").strip()
    seed = int(seed_raw) if seed_raw else None

    # Translator selection (filtered by selected environment)
    default_env_translator = get_translator(env_id)

    translator_labels = ["None"]
    if default_env_translator is not None:
        translator_labels.append(f"Auto for {env_id}")

    default_t_idx = 1 if len(translator_labels) > 1 else 0
    t_idx = _select_from_list(
        "Language translation for instruction matching",
        translator_labels,
        default_idx=default_t_idx,
    )
    translator = None
    translator_label = translator_labels[t_idx]
    if translator_label == "None":
        translator = None
    elif translator_label.startswith("Auto for "):
        translator = default_env_translator
    else:
        translator = TRANSLATORS.get(translator_label)

    instruction = ""
    match = None
    sub_goal_bonus: Optional[float] = None
    sub_goal_threshold = 0.5

    if translator is not None and _ask_bool("Provide instruction for sub-goal matching", default=True):
        instruction = input("Instruction: ").strip()
        if instruction:
            print("Matching instruction to observed states...")
            match_env = registry.create(env_id, render_mode=None)
            try:
                match = match_instruction(
                    instruction=instruction,
                    env=match_env,
                    translator=translator,
                    max_steps=max_steps,
                    seed=seed,
                )
            finally:
                match_env.close()

            print("Matched language state:")
            print(f"  {match.matched_language}")
            print(f"  similarity={match.similarity_score:.4f}")

            sub_goal_threshold = _ask_float("Sub-goal similarity threshold", 0.5, minimum=0.0)
            if _ask_bool("Auto-scale sub-goal bonus", default=True):
                tmp_env = registry.create(env_id)
                try:
                    sub_goal_bonus = scale_sub_goal_bonus(tmp_env, n_instructions=max(1, len(match.matched_states)))
                finally:
                    tmp_env.close()
            else:
                sub_goal_bonus = _ask_float("Sub-goal bonus", 0.01, minimum=0.0)

    use_language_state = translator is not None and _ask_bool("Train agent on translated language observations", default=False)

    cfg = TrainConfig(
        agent_type=agent_type,
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        use_language_state=use_language_state,
    )

    deck_meta_cfg: Optional[DeckMetaRewardConfig] = None
    if env_id == "FleshAndBlood-DeckBuild-v0" and _ask_bool(
        "Use nested deck reward (terminal reward = inner gameplay win-rate)",
        default=True,
    ):
        fmt_raw = input("Meta format [silver_age]: ").strip().lower()
        meta_format = fmt_raw or "silver_age"
        inner_agent_options = ["tabular_q", "dqn", "ppo"]
        inner_idx = _select_from_list("Inner gameplay agent type", inner_agent_options, default_idx=2)
        inner_agent_type = inner_agent_options[inner_idx]
        inner_train_episodes = _ask_int("Inner train episodes per matchup", 60, minimum=1)
        inner_eval_episodes = _ask_int("Inner eval episodes per matchup", 6, minimum=1)
        matchups_per_deck = _ask_int("Matchups sampled per deck", 4, minimum=1)
        inner_max_steps = _ask_int("Inner max steps per episode", max_steps, minimum=1)
        deck_meta_cfg = DeckMetaRewardConfig(
            format=meta_format,
            inner_agent_type=inner_agent_type,
            inner_train_episodes=inner_train_episodes,
            inner_eval_episodes=inner_eval_episodes,
            matchups_per_deck=matchups_per_deck,
            inner_max_steps=inner_max_steps,
        )

    hyperparams: dict[str, Any] = {"seed": seed, "gamma": 0.99}
    if agent_type in {"tabular_q", "dqn"}:
        hyperparams["epsilon"] = _ask_float("epsilon", 1.0, minimum=0.0)
        hyperparams["epsilon_min"] = _ask_float("epsilon_min", 0.01, minimum=0.0)
        hyperparams["epsilon_decay"] = _ask_float("epsilon_decay", 0.995, minimum=0.0)
    if agent_type == "tabular_q":
        hyperparams["alpha"] = _ask_float("alpha", 0.1, minimum=0.0)
    elif agent_type == "dqn":
        hyperparams["hidden_size"] = _ask_int("hidden_size", 64)
        hyperparams["lr"] = _ask_float("learning_rate", 1e-3, minimum=0.0)
        hyperparams["buffer_size"] = _ask_int("buffer_size", 10000)
        hyperparams["batch_size"] = _ask_int("batch_size", 64)
        hyperparams["target_update_freq"] = _ask_int("target_update_freq", 100)
    elif agent_type == "ppo":
        hyperparams["hidden_size"] = _ask_int("hidden_size", 64)
        hyperparams["lr_actor"] = _ask_float("lr_actor", 1e-3, minimum=0.0)
        hyperparams["lr_critic"] = _ask_float("lr_critic", 1e-3, minimum=0.0)
        hyperparams["lam"] = _ask_float("lam", 0.95, minimum=0.0)
        hyperparams["clip_eps"] = _ask_float("clip_eps", 0.2, minimum=0.0)
        hyperparams["n_steps"] = _ask_int("n_steps", 256)
        hyperparams["ppo_epochs"] = _ask_int("ppo_epochs", 4)
        hyperparams["mini_batch_size"] = _ask_int("mini_batch_size", 64)

    dashboard_port = _ask_int("Dashboard port", 7432, minimum=1)
    dashboard_url = start_dashboard(port=dashboard_port)
    print(f"Dashboard running at {dashboard_url}")

    agent = _build_agent(cfg.agent_type, hyperparams)

    run_agent_id = f"cli-{agent_type}-{uuid.uuid4().hex[:8]}"
    dashboard.register(
        agent_id=run_agent_id,
        agent_type=agent_type,
        env_id=env_id,
        n_episodes=n_episodes,
        use_language_state=cfg.use_language_state,
        uses_instructions=bool(instruction and match is not None),
        instructions=[instruction] if instruction else None,
    )

    if deck_meta_cfg is not None:
        train_env = DeckBuildMetaRewardEnv(
            cfg=deck_meta_cfg,
            play_hyperparams={k: v for k, v in hyperparams.items() if v is not None},
            seed=seed,
        )
    else:
        base_train_env = registry.create(env_id, render_mode=None)
        train_env = base_train_env
        if match is not None and translator is not None and instruction:
            train_env = _ShapedEnv(
                env=train_env,
                sub_goal_language=match.matched_language,
                sub_goal_languages=[s[0] for s in match.matched_states],
                bonus=sub_goal_bonus,
                threshold=sub_goal_threshold,
                translator=translator,
                env_id=env_id,
            )
        if cfg.use_language_state and translator is not None:
            train_env = _LangStateEnv(train_env, translator=translator, env_id=env_id)

    progress_env = ProgressEnv(
        train_env,
        n_episodes=n_episodes,
        dashboard_agent_id=run_agent_id,
        agent_ref=agent,
    )

    print("Starting training...\n")
    train_result = agent.train(
        progress_env,
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
    )
    progress_env.close()

    print("Training complete")
    print(f"  mean_reward={train_result.mean_reward:.4f}")
    print(f"  best_reward={train_result.best_reward:.4f}")
    print(f"  final_epsilon={train_result.final_epsilon:.4f}")

    if deck_meta_cfg is not None:
        eval_env = DeckBuildMetaRewardEnv(
            cfg=deck_meta_cfg,
            play_hyperparams={k: v for k, v in hyperparams.items() if v is not None},
            seed=seed,
        )
        try:
            eval_metrics = _run_eval_episode(eval_env, agent, max_steps=1, seed=seed)
        finally:
            eval_env.close()

        renders_dir = Path.home() / ".rlip" / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        safe_env = env_id.replace("/", "_").replace(" ", "_")
        report_path = renders_dir / f"{safe_env}_{agent_type}_report.png"

        comparison_runs = [
            {
                "label": f"{agent_type}:cli",
                "train_result": train_result,
                "metadata": {
                    "env_id": env_id,
                    "agent_id": run_agent_id,
                    "agent_type": agent_type,
                    "meta_reward": "inner_play_win_rate",
                    "inner_agent_type": deck_meta_cfg.inner_agent_type,
                },
                "hyperparameters": {k: v for k, v in hyperparams.items() if v is not None},
                "instruction": None,
                "best_match_observation": None,
                "best_match_similarity": None,
            }
        ]
        fig = create_training_report(
            comparison_runs=comparison_runs,
            output_path=str(report_path),
        )
        fig.clf()

        final_obs = eval_metrics.get("final_observation")
        meta = final_obs.get("meta_reward", {}) if isinstance(final_obs, dict) else {}
        selected = meta.get("selected_deck") or {}
        selected_label = str(selected.get("label", "unknown"))
        win_rate = float(meta.get("win_rate", eval_metrics.get("total_reward", 0.0)) or 0.0)

        print("\n" + "=" * 80)
        print("Deck Meta-Training Summary")
        print("=" * 80)
        print(textwrap.dedent(
            f"""
            Environment:                 {env_id}
            Agent (deckbuilder):         {agent_type}
            Dashboard:                   {dashboard_url}
            Meta reward mode:            inner gameplay win-rate
            Inner agent:                 {deck_meta_cfg.inner_agent_type}
            Inner train episodes:        {deck_meta_cfg.inner_train_episodes}
            Inner eval episodes:         {deck_meta_cfg.inner_eval_episodes}
            Matchups per deck:           {deck_meta_cfg.matchups_per_deck}
            Selected deck (eval):        {selected_label}
            Terminal reward / win-rate:  {win_rate:.2%}
            Report PNG:                  {report_path}
            """
        ).strip())
        return 0

    # Evaluation run
    eval_env = registry.create(env_id, render_mode=None)
    if cfg.use_language_state and translator is not None:
        eval_env = _LangStateEnv(eval_env, translator=translator, env_id=env_id)
    try:
        eval_metrics = _run_eval_episode(eval_env, agent, max_steps=max_steps, seed=seed)
    finally:
        eval_env.close()

    # Render policy replay
    print("Rendering policy replay...")
    render_env = registry.create(env_id, render_mode="rgb_array")
    if cfg.use_language_state and translator is not None:
        render_env = _LangStateEnv(render_env, translator=translator, env_id=env_id)

    try:
        policy_fn = (lambda obs: agent.act_greedy(obs)) if hasattr(agent, "act_greedy") else (lambda obs: agent.act(obs))
        eval_result = MultiEpisodeProtocol(
            GreedyEpisodeProtocol(policy_fn=policy_fn, max_steps=max_steps, seed=seed, record_history=True),
            n_episodes=8,
            base_seed=seed,
        )(render_env)
    finally:
        render_env.close()

    renders_dir = Path.home() / ".rlip" / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    safe_env = env_id.replace("/", "_").replace(" ", "_")
    gif_path = renders_dir / f"{safe_env}_{agent_type}_policy.gif"

    render_env2 = registry.create(env_id, render_mode="rgb_array")
    if cfg.use_language_state and translator is not None:
        render_env2 = _LangStateEnv(render_env2, translator=translator, env_id=env_id)
    try:
        render_result = render_optimal_policy(
            eval_result,
            env=render_env2,
            max_steps=max_steps,
            seed=seed,
            fallback=policy_fn,
            translate=(translator if cfg.use_language_state else False),
            output_gif=str(gif_path),
        )
    finally:
        render_env2.close()

    # Training/evaluation report
    report_path = renders_dir / f"{safe_env}_{agent_type}_report.png"
    comparison_runs = [
        {
            "label": f"{agent_type}:cli",
            "train_result": train_result,
            "metadata": {
                "env_id": env_id,
                "agent_id": run_agent_id,
                "agent_type": agent_type,
                "use_language_state": cfg.use_language_state,
                "uses_instructions": bool(instruction and match is not None),
                "match_id": "cli-match" if match is not None else "-",
            },
            "hyperparameters": {k: v for k, v in hyperparams.items() if v is not None},
            "instruction": instruction or None,
            "best_match_observation": match.matched_language if match is not None else None,
            "best_match_similarity": match.similarity_score if match is not None else None,
        }
    ]

    fig = create_training_report(
        comparison_runs=comparison_runs,
        output_path=str(report_path),
    )
    fig.clf()

    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(textwrap.dedent(
        f"""
        Environment:      {env_id}
        Agent:            {agent_type}
        Dashboard:        {dashboard_url}
        Eval steps:       {eval_metrics['steps']}
        Eval reward:      {eval_metrics['total_reward']:.4f}
        Eval terminated:  {eval_metrics['terminated']}
        Eval truncated:   {eval_metrics['truncated']}
        Render GIF:       {gif_path}
        Report PNG:       {report_path}
        """
    ).strip())

    if _is_fab_env(env_id):
        p_agent, p_opp = _fab_win_probabilities(eval_metrics.get("final_observation"))
        print(f"FaB win probability (agent):    {p_agent:.2%}")
        print(f"FaB win probability (opponent): {p_opp:.2%}")

    if render_result.n_gif_frames == 0:
        print("Note: rgb_array frames were not produced; check environment render support.")

    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
