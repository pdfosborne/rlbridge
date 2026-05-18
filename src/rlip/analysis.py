"""
RLIP Training Analysis & Reporting
==================================
Comparison-first report generation for one or many trained agents.

The report focuses on:
- Reward convergence during training.
- "Optimal policy" reward at breakpoints in training (best-so-far proxy).
- Instruction and best matched observation with similarity percentage.
- Metadata and hyper-parameters used for each run.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

from ._agent_base import TrainResult


@dataclass
class AgentReportRun:
    """Normalized report input for one trained agent run."""

    label: str
    train_result: TrainResult
    metadata: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    instruction: Optional[str] = None
    best_match_observation: Optional[Any] = None
    best_match_similarity: Optional[float] = None
    breakpoints: list[tuple[int, float]] = field(default_factory=list)
    eval_mean: Optional[float] = None
    eval_std: Optional[float] = None


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _stable_eps_for_convergence(values: list[float]) -> float:
    if not values:
        return 0.0
    span = max(values) - min(values)
    return max(1e-6, span * 0.05)


def _estimate_convergence_episode(rolling: list[float], stable_window: int) -> Optional[int]:
    """
    First episode where rolling reward reaches/stays near terminal behavior.

    Uses the mean of the final stable_window as the terminal target and finds
    the first index where the remaining sequence stays within +/-5% reward-span.
    """
    if not rolling:
        return None
    n = len(rolling)
    stable_window = max(3, min(stable_window, n))
    tail = rolling[-stable_window:]
    target = sum(tail) / len(tail)
    eps = _stable_eps_for_convergence(rolling)

    for i in range(n):
        remainder = rolling[i:]
        if all(abs(v - target) <= eps for v in remainder):
            return i + 1
    return None


def _compute_breakpoints_best_so_far(
    rewards: list[float],
    n_breakpoints: int,
) -> list[tuple[int, float]]:
    """
    Breakpoint evaluation as best-so-far reward proxy.

    For breakpoint episode k, reward is max(rewards[:k]).
    Schedule: every 10 episodes for 1-100, then larger intervals (50-episode gaps) for >100.
    """
    if not rewards:
        return []

    n = len(rewards)
    del n_breakpoints

    # Phase 1: every 10 episodes for the first 100 episodes
    points: list[int] = list(range(10, min(101, n + 1), 10))
    
    # Phase 2: larger intervals for episodes > 100 (every 50 episodes)
    if n > 100:
        points.extend(ep for ep in range(150, n + 1, 50))
        if n not in points:
            points.append(n)
    
    points = sorted(set(points))  # Remove duplicates and sort

    out: list[tuple[int, float]] = []
    running_best = float("-inf")
    next_point_idx = 0

    for ep_idx, reward in enumerate(rewards, start=1):
        running_best = max(running_best, reward)
        while next_point_idx < len(points) and ep_idx >= points[next_point_idx]:
            out.append((points[next_point_idx], running_best))
            next_point_idx += 1

    return out


def _truncate(value: Any, max_len: int = 70) -> str:
    s = str(value)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _wrap_text(value: Any, width: int) -> str:
    text = str(value) if value is not None else "-"
    # Keep table cells and side panels readable by wrapping long values.
    return textwrap.fill(text, width=max(8, width), break_long_words=False)


def _flag_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "on"}


def _legend_label(run: AgentReportRun) -> str:
    use_lang = _flag_enabled(run.metadata.get("use_language_state"))
    use_instr = _flag_enabled(run.metadata.get("uses_instructions"))
    if not use_instr:
        match_id = str(run.metadata.get("match_id") or "").strip()
        use_instr = bool(match_id and match_id != "-")

    tags: list[str] = []
    if use_lang:
        tags.append("lang")
    if use_instr:
        tags.append("instr")
    if not tags:
        return run.label
    return f"{run.label} ({'+'.join(tags)})"


def _normalize_runs(
    train_result: Optional[TrainResult],
    comparison_runs: Optional[list[dict[str, Any]]],
    metadata: Optional[dict[str, Any]],
    subgoal_info: Optional[dict[str, Any]],
    n_breakpoints: int,
) -> list[AgentReportRun]:
    runs: list[AgentReportRun] = []

    if comparison_runs:
        for i, raw in enumerate(comparison_runs, start=1):
            tr = raw.get("train_result")
            if tr is None:
                raise ValueError(f"comparison_runs[{i}] missing train_result")
            run = AgentReportRun(
                label=str(raw.get("label") or f"run-{i}"),
                train_result=tr,
                metadata=dict(raw.get("metadata") or {}),
                hyperparameters=dict(raw.get("hyperparameters") or {}),
                instruction=raw.get("instruction"),
                best_match_observation=raw.get("best_match_observation"),
                best_match_similarity=raw.get("best_match_similarity"),
                breakpoints=list(raw.get("breakpoints") or []),
                eval_mean=raw.get("eval_mean"),
                eval_std=raw.get("eval_std"),
            )
            if not run.breakpoints:
                run.breakpoints = _compute_breakpoints_best_so_far(
                    run.train_result.episode_rewards,
                    n_breakpoints=n_breakpoints,
                )
            runs.append(run)
        return runs

    if train_result is None:
        raise ValueError("Provide train_result or comparison_runs")

    legacy_instruction = None
    legacy_best_match_obs = None
    legacy_best_match_sim = None
    if subgoal_info:
        legacy_instruction = subgoal_info.get("instruction")
        legacy_best_match_obs = subgoal_info.get("best_match_observation") or subgoal_info.get("sub_goal_language")
        legacy_best_match_sim = subgoal_info.get("best_match_similarity")

    single = AgentReportRun(
        label=train_result.agent_name,
        train_result=train_result,
        metadata=dict(metadata or {}),
        hyperparameters={},
        instruction=legacy_instruction,
        best_match_observation=legacy_best_match_obs,
        best_match_similarity=legacy_best_match_sim,
        breakpoints=_compute_breakpoints_best_so_far(train_result.episode_rewards, n_breakpoints=n_breakpoints),
    )
    runs.append(single)
    return runs


def create_training_report(
    train_result: Optional[TrainResult] = None,
    render_result: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
    subgoal_steps: Optional[list[tuple[int, float, bool]]] = None,
    subgoal_info: Optional[dict[str, Any]] = None,
    *,
    comparison_runs: Optional[list[dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    rolling_window: Optional[int] = None,
    n_breakpoints: int = 6,
    fig_width: float = 17.0,
    dpi: int = 120,
    n_sample_frames: int = 0,
) -> Any:
    """
    Generate a redesigned training report.

    Backward compatibility:
    - Existing single-agent callers can keep passing train_result.
    - comparison_runs enables side-by-side multi-agent comparison.

    Notes
    -----
    Breakpoint "optimal policy reward" is reported as a best-so-far proxy:
    for each breakpoint episode k, value is max(train_reward[1..k]).
    """
    del render_result
    del subgoal_steps
    del n_sample_frames

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for training reports. Install with: pip install matplotlib"
        ) from exc

    runs = _normalize_runs(
        train_result=train_result,
        comparison_runs=comparison_runs,
        metadata=metadata,
        subgoal_info=subgoal_info,
        n_breakpoints=n_breakpoints,
    )

    max_episodes = max(len(r.train_result.episode_rewards) for r in runs)
    default_window = max(10, int(max_episodes * 0.05))
    window = rolling_window or default_window

    fig = plt.figure(figsize=(fig_width, 12.5), dpi=dpi)
    fig.patch.set_facecolor("#f7f7f4")
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.45,
        wspace=0.25,
        height_ratios=[2.8, 2.2, 2.4],
    )

    ax_conv = fig.add_subplot(gs[0, :])
    _draw_convergence_panel(ax_conv, runs, window)

    ax_eval = fig.add_subplot(gs[1, :])
    _draw_evaluation_panel(ax_eval, runs)

    ax_inst = fig.add_subplot(gs[2, 0])
    _draw_instruction_panel(ax_inst, runs)

    ax_cfg = fig.add_subplot(gs[2, 1])
    _draw_config_panel(ax_cfg, runs)

    env_names = sorted({str(r.metadata.get("env_id", "")) for r in runs if r.metadata.get("env_id")})
    env_label = f" | env={', '.join(env_names)}" if env_names else ""
    fig.suptitle(
        f"RLIP Comparative Training Report ({len(runs)} agent(s)){env_label}",
        fontsize=14,
        fontweight="bold",
        color="#1b1f24",
        y=0.99,
    )

    fig.text(
        0.01,
        0.01,
        "Evaluation: 100 episodes with fixed weights on plain environment (no instruction rewards). Error bars = ±1 std.",
        fontsize=7.5,
        color="#555555",
    )

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def _draw_convergence_panel(ax: Any, runs: list[AgentReportRun], window: int) -> None:
    ax.set_facecolor("#ffffff")
    palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
    ]

    for i, run in enumerate(runs):
        rewards = run.train_result.episode_rewards
        xs = list(range(1, len(rewards) + 1))
        rolling = _rolling_mean(rewards, window)
        color = palette[i % len(palette)]

        label = _legend_label(run)
        ax.plot(xs, rewards, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(xs, rolling, color=color, linewidth=2.0, label=f"{label} rolling")

        conv_ep = _estimate_convergence_episode(rolling, stable_window=max(8, window // 2))
        if conv_ep is not None and conv_ep <= len(rolling):
            conv_y = rolling[conv_ep - 1]
            ax.scatter([conv_ep], [conv_y], color=color, s=36, zorder=5)
            ax.axvline(conv_ep, color=color, linestyle=":", linewidth=1.0, alpha=0.6)
            ax.text(
                conv_ep,
                conv_y,
                f"  {label} conv@{conv_ep}",
                fontsize=7,
                color=color,
                va="bottom",
            )

    ax.set_title("Training Reward Convergence", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_ylabel("Reward", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=7, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)


def _draw_evaluation_panel(ax: Any, runs: list[AgentReportRun]) -> None:
    """
    Bar chart comparing clean evaluation results across agents.

    Each bar shows the mean reward from 100 post-training episodes run with
    fixed weights on the plain environment (no instruction rewards).  Error
    bars show ±1 standard deviation.  Falls back to the best training reward
    when clean evaluation data is unavailable.
    """
    ax.set_facecolor("#ffffff")
    palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
    ]

    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    colors: list[str] = []
    is_fallback: list[bool] = []

    for i, run in enumerate(runs):
        labels.append(_legend_label(run))
        colors.append(palette[i % len(palette)])
        if run.eval_mean is not None:
            means.append(run.eval_mean)
            stds.append(run.eval_std if run.eval_std is not None else 0.0)
            is_fallback.append(False)
        else:
            # Fallback: use best training reward with no error bar
            means.append(run.train_result.best_reward)
            stds.append(0.0)
            is_fallback.append(True)

    xs = list(range(len(labels)))
    bars = ax.bar(xs, means, color=colors, alpha=0.75, width=0.55, zorder=3)
    ax.errorbar(
        xs,
        means,
        yerr=stds,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.4,
        capsize=5,
        zorder=4,
    )

    for bar, mean, std, fallback in zip(bars, means, stds, is_fallback):
        label_text = f"{mean:.4f}" if fallback else f"{mean:.4f}\n±{std:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(stds) * 0.05 + abs(mean) * 0.01,
            label_text,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#1f2933",
        )
        if fallback:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() / 2.0,
                "(train\nbest)",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#ffffff",
                alpha=0.8,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=15 if len(labels) > 3 else 0, ha="right")
    ax.set_title(
        "Clean Evaluation — Mean Reward ± Std  (100 episodes, fixed weights, no instruction rewards)",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_ylabel("Mean reward", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_instruction_panel(ax: Any, runs: list[AgentReportRun]) -> None:
    ax.axis("off")
    ax.set_facecolor("#ffffff")
    ax.set_title("Instruction Match Summary", fontsize=10, fontweight="bold", pad=8)

    rows: list[list[str]] = []
    for run in runs:
        sim = "-"
        if run.best_match_similarity is not None:
            sim = f"{max(0.0, min(1.0, float(run.best_match_similarity))) * 100.0:.2f}%"
        rows.append(
            [
                _truncate(run.label, 24),
                _wrap_text(run.instruction or "-", 42),
                _wrap_text(run.best_match_observation or "-", 34),
                sim,
            ]
        )

    tbl = ax.table(
        cellText=rows,
        colLabels=["Agent", "Instruction", "Best match observation", "Sim %"],
        loc="center",
        cellLoc="left",
        colWidths=[0.16, 0.45, 0.29, 0.10],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.8)
    tbl.scale(1.0, 2.0)

    for col in range(4):
        tbl[(0, col)].set_facecolor("#2c3e50")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")
        tbl[(0, col)].get_text().set_wrap(True)
    for row_i in range(1, len(rows) + 1):
        bg = "#f3f6f8" if row_i % 2 == 0 else "#ffffff"
        for col in range(4):
            tbl[(row_i, col)].set_facecolor(bg)
            tbl[(row_i, col)].set_edgecolor("#d9d9d9")
            tbl[(row_i, col)].get_text().set_wrap(True)


def _draw_config_panel(ax: Any, runs: list[AgentReportRun]) -> None:
    ax.axis("off")
    ax.set_facecolor("#ffffff")
    ax.set_title("Metadata & Hyper-parameters", fontsize=10, fontweight="bold", pad=8)

    lines: list[str] = []
    panel_width = 66
    for idx, run in enumerate(runs, start=1):
        lines.append(f"[{idx}] {run.label}")

        if run.metadata:
            lines.append("  metadata:")
            for k, v in sorted(run.metadata.items()):
                wrapped = _wrap_text(v, 34).splitlines()
                lines.append(f"    {k}: {wrapped[0]}")
                for extra in wrapped[1:]:
                    lines.append(f"      {extra}")
        else:
            lines.append("  metadata: -")

        if run.hyperparameters:
            lines.append("  hyper-parameters:")
            for k, v in sorted(run.hyperparameters.items()):
                wrapped = _wrap_text(v, 34).splitlines()
                lines.append(f"    {k}: {wrapped[0]}")
                for extra in wrapped[1:]:
                    lines.append(f"      {extra}")
        else:
            lines.append("  hyper-parameters: -")

        rewards = run.train_result.episode_rewards
        if rewards:
            summary_parts = [
                f"episodes={len(rewards)}, mean={run.train_result.mean_reward:.4f}, "
                f"best={run.train_result.best_reward:.4f}, last10%={run.train_result.last_n_mean:.4f}"
            ]
            if run.eval_mean is not None:
                eval_str = f"eval_mean={run.eval_mean:.4f}"
                if run.eval_std is not None:
                    eval_str += f" ±{run.eval_std:.4f}"
                eval_str += " (100ep, fixed weights)"
                summary_parts.append(eval_str)
            summary = ", ".join(summary_parts)
            lines.append("  summary:")
            for sline in textwrap.wrap(summary, width=panel_width - 4):
                lines.append(f"    {sline}")
        lines.append("")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines).rstrip(),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        family="monospace",
        color="#1f2933",
    )


__all__ = ["AgentReportRun", "create_training_report"]
