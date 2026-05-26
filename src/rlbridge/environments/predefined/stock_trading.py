"""
rlbridge Simulated Stock Trading Environment
===========================================
Discrete trading task over a synthetic price series.

State
-----
Observation is a list:
    [day, price_bin, momentum_bin, position, cash_bin]

Actions
-------
    0 - hold
    1 - buy  (if flat and enough cash)
    2 - sell (if currently holding one share)

Goal
----
Maximise final portfolio value over the episode horizon.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from ...protocol.messages import (
    DiscreteSpace,
    EnvironmentInfo,
    MultiDiscreteSpace,
    RenderResult,
    ResetResult,
    StepResult,
    SuggestedHyperparameters,
)
from ..base import rlbridgeEnvironment, rlbridgeEnvironmentFactory


class StockTradingEnvironment(rlbridgeEnvironment):
    """Simple single-asset trading simulator with one-share position limit."""

    def __init__(
        self,
        *,
        max_days: int = 60,
        initial_cash: float = 1000.0,
        start_price: float = 100.0,
        drift: float = 0.0005,
        volatility: float = 0.02,
        fee: float = 1.0,
    ) -> None:
        self._max_days = max_days
        self._initial_cash = float(initial_cash)
        self._start_price = float(start_price)
        self._drift = float(drift)
        self._volatility = float(volatility)
        self._fee = float(fee)

        self._rng = random.Random()
        self._initialized = False

        self._day = 0
        self._prices: list[float] = []
        self._cash = self._initial_cash
        self._position = 0  # 0 or 1 share
        self._entry_price: Optional[float] = None
        self._last_value = self._initial_cash

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        if seed is not None:
            self._rng.seed(seed)

        opts = options or {}
        initial_cash = float(opts.get("initial_cash", self._initial_cash))
        start_price = float(opts.get("start_price", self._start_price))

        self._prices = self._generate_prices(start_price=start_price)
        self._day = 0
        self._cash = initial_cash
        self._position = 0
        self._entry_price = None
        self._last_value = initial_cash
        self._initialized = True

        return ResetResult(observation=self._obs(), info=self._info())

    def step(self, action: Any) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")

        a = int(action)
        if a not in (0, 1, 2):
            raise ValueError(f"Action must be 0..2, got {a}")

        price = self._price_at(self._day)
        invalid_action_penalty = 0.0

        if a == 1:  # buy
            total_cost = price + self._fee
            if self._position == 0 and self._cash >= total_cost:
                self._cash -= total_cost
                self._position = 1
                self._entry_price = price
            else:
                invalid_action_penalty = 0.1
        elif a == 2:  # sell
            if self._position == 1:
                self._cash += price - self._fee
                self._position = 0
                self._entry_price = None
            else:
                invalid_action_penalty = 0.1

        self._day += 1
        done = self._day >= self._max_days

        # Reward is day-over-day portfolio value change, lightly penalising
        # invalid actions.
        current_value = self._portfolio_value(day=self._day)
        reward = (current_value - self._last_value) / 10.0 - invalid_action_penalty
        self._last_value = current_value

        if done:
            self._initialized = False

        return StepResult(
            observation=self._obs(),
            reward=float(reward),
            terminated=False,
            truncated=done,
            info=self._info(),
        )

    def close(self) -> None:
        self._initialized = False

    @property
    def observation_space(self) -> MultiDiscreteSpace:
        # day, price_bin, momentum_bin, position, cash_bin
        return MultiDiscreteSpace(nvec=[self._max_days + 1, 20, 11, 2, 21])

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(n=3, start=0)

    def render(self) -> RenderResult:
        value = self._portfolio_value(day=self._day)
        text = (
            f"day={self._day}/{self._max_days} "
            f"price={self._price_at(self._day):.2f} "
            f"position={self._position} "
            f"cash={self._cash:.2f} "
            f"value={value:.2f}  "
            "actions[0=hold 1=buy 2=sell]"
        )
        return RenderResult(mode="ansi", text=text)

    def sample_action(self) -> int:
        return self._rng.randint(0, 2)

    def _obs(self) -> list[int]:
        price = self._price_at(self._day)
        prev_price = self._price_at(max(0, self._day - 1))
        price_ratio = price / max(1e-6, self._start_price)
        momentum = (price - prev_price) / max(1e-6, prev_price)

        price_bin = self._clip_int(int(price_ratio * 10.0), 0, 19)
        momentum_bin = self._clip_int(int((momentum + 0.05) * 100.0), 0, 10)
        cash_bin = self._clip_int(int((self._cash / self._initial_cash) * 10.0), 0, 20)

        return [self._day, price_bin, momentum_bin, self._position, cash_bin]

    def _info(self) -> dict[str, Any]:
        return {
            "day": self._day,
            "price": self._price_at(self._day),
            "cash": round(self._cash, 4),
            "position": self._position,
            "entry_price": self._entry_price,
            "portfolio_value": round(self._portfolio_value(day=self._day), 4),
            "max_days": self._max_days,
        }

    def _generate_prices(self, *, start_price: float) -> list[float]:
        prices = [max(1.0, start_price)]
        for _ in range(self._max_days):
            prev = prices[-1]
            noise = self._rng.uniform(-self._volatility, self._volatility)
            change = self._drift + noise
            next_price = max(1.0, prev * (1.0 + change))
            prices.append(next_price)
        return prices

    def _portfolio_value(self, *, day: int) -> float:
        return self._cash + self._position * self._price_at(day)

    def _price_at(self, day: int) -> float:
        idx = self._clip_int(day, 0, len(self._prices) - 1)
        return self._prices[idx]

    @staticmethod
    def _clip_int(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))


class StockTradingFactory(rlbridgeEnvironmentFactory):
    """Factory for predefined stock-trading simulators."""

    def __init__(
        self,
        env_id: str = "StockTrading-Sim-v0",
        *,
        max_days: int = 60,
        initial_cash: float = 1000.0,
        start_price: float = 100.0,
        drift: float = 0.0005,
        volatility: float = 0.02,
        fee: float = 1.0,
        tags: Optional[list[str]] = None,
        description: str = "",
    ) -> None:
        self._env_id = env_id
        self._max_days = max_days
        self._initial_cash = initial_cash
        self._start_price = start_price
        self._drift = drift
        self._volatility = volatility
        self._fee = fee
        self._tags = tags or ["trading", "finance", "simulated", "discrete"]
        self._description = description or (
            "Synthetic single-stock trading simulator with hold/buy/sell actions, "
            "transaction fee, and portfolio-value based rewards."
        )

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=self._description,
            tags=self._tags,
            namespace="trading",
            render_modes=["ansi"],
            max_episode_steps=self._max_days,
            reward_threshold=2.0,
            suggested_hyperparameters=SuggestedHyperparameters(
                agent_type="ppo",
                n_episodes=2000,
                max_steps=self._max_days,
                gamma=0.995,
                hidden_size=128,
                lr=3e-4,
                sub_goal_threshold=0.5,
                top_k=3,
                min_episode_visits=2,
            ),
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> StockTradingEnvironment:
        _ = render_mode
        return StockTradingEnvironment(
            max_days=int(kwargs.get("max_days", self._max_days)),
            initial_cash=float(kwargs.get("initial_cash", self._initial_cash)),
            start_price=float(kwargs.get("start_price", self._start_price)),
            drift=float(kwargs.get("drift", self._drift)),
            volatility=float(kwargs.get("volatility", self._volatility)),
            fee=float(kwargs.get("fee", self._fee)),
        )


STOCK_TRADING_SIM_V0 = StockTradingFactory()
ALL_STOCK_TRADING_FACTORIES: list[StockTradingFactory] = [STOCK_TRADING_SIM_V0]
