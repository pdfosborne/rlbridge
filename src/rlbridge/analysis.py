"""
rlbridge Training Analysis & Reporting
==================================
Comparison-first report generation for one or many trained agents.

The report focuses on:
- Reward convergence during training.
- "Optimal policy" reward at breakpoints in training (best-so-far proxy).
- Instruction and best matched observation with similarity percentage.
- Metadata and hyper-parameters used for each run.
"""

from __future__ import annotations

import math
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
    eval_rewards: list[float] = field(default_factory=list)
    sub_steps_matches: list[dict[str, Any]] = field(default_factory=list)
    """Per-sub-step match info.  Each entry is a dict with keys:
    ``instruction`` (str), ``matched_language`` (str), ``similarity`` (float),
    ``render_text`` (Optional[str]), ``render_b64`` (Optional[str]).
    """


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
                eval_rewards=list(raw.get("eval_rewards") or []),
                sub_steps_matches=list(raw.get("sub_steps_matches") or []),
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
) -> dict[str, Any]:
    """
    Generate three separate training report figures.

    Returns
    -------
    dict with keys:
        ``"rewards"``       - Figure 1: rolling training reward + summary metrics
                             table (training + clean evaluation).
        ``"instructions"``  - Figure 2: instruction match summary (no text overlap).
        ``"config"``        - Figure 3: metadata and hyper-parameters.

    When *output_path* is given each figure is saved to:
        ``{stem}_rewards.png``, ``{stem}_instructions.png``, ``{stem}_config.png``

    Backward compatibility
    ---------------------
    Single-agent callers can keep passing *train_result*.
    Multi-agent callers use *comparison_runs*.
    """
    del render_result, subgoal_steps, n_sample_frames

    try:
        import matplotlib
        matplotlib.use("Agg")
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

    env_names = sorted({str(r.metadata.get("env_id", "")) for r in runs if r.metadata.get("env_id")})
    env_label = f" | env={', '.join(env_names)}" if env_names else ""
    n_agents = len(runs)

    # ── Figure 1: Reward charts ───────────────────────────────────────────────
    fig_rewards, (ax_conv, ax_eval) = plt.subplots(
        2, 1,
        figsize=(fig_width, 10.5),
        dpi=dpi,
        gridspec_kw={"hspace": 0.45, "height_ratios": [1, 1]},
    )
    # Reduce left outer padding and add more right-side breathing room.
    fig_rewards.subplots_adjust(left=0.06, right=0.86)
    fig_rewards.patch.set_facecolor("#f7f7f4")
    _draw_convergence_panel(ax_conv, runs, window)
    _draw_metrics_summary_panel(ax_eval, runs, window)
    fig_rewards.suptitle(
        f"rlbridge Reward Report ({n_agents} agent(s)){env_label}",
        fontsize=15, fontweight="bold", color="#1b1f24", y=0.99,
    )
    fig_rewards.text(
        0.01, 0.005,
        "Top panel: rolling average reward during training.  "
        "Bottom panel: summary training/evaluation metrics "
        "(evaluation uses fixed weights on plain environment; no instruction rewards).",
        fontsize=9, color="#555555",
    )

    # ── Figure 2: Instructions ────────────────────────────────────────────────
    fig_instructions = _build_instruction_figure(
        runs, fig_width=fig_width, dpi=dpi, env_label=env_label,
    )

    # ── Figure 3: Config / metadata ───────────────────────────────────────────
    fig_config = _build_config_figure(
        runs, fig_width=fig_width, dpi=dpi, env_label=env_label,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    if output_path:
        import pathlib as _pl
        stem = str(_pl.Path(output_path).with_suffix(""))
        fig_rewards.savefig(
            stem + "_rewards.png", dpi=dpi, bbox_inches="tight",
            facecolor=fig_rewards.get_facecolor(),
        )
        fig_instructions.savefig(
            stem + "_instructions.png", dpi=dpi, bbox_inches="tight",
            facecolor=fig_instructions.get_facecolor(),
        )
        fig_config.savefig(
            stem + "_config.png", dpi=dpi, bbox_inches="tight",
            facecolor=fig_config.get_facecolor(),
        )

    return {"rewards": fig_rewards, "instructions": fig_instructions, "config": fig_config}


# ─────────────────────────────────────────────────────────────────────────────
# Panel drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _draw_convergence_panel(ax: Any, runs: list[AgentReportRun], window: int) -> None:
    """Rolling-average training reward only - no raw scatter."""
    ax.set_facecolor("#ffffff")

    for i, run in enumerate(runs):
        rewards = run.train_result.episode_rewards
        if not rewards:
            continue
        xs = list(range(1, len(rewards) + 1))
        rolling = _rolling_mean(rewards, window)
        color = _PALETTE[i % len(_PALETTE)]
        label = _legend_label(run)

        ax.plot(xs, rolling, color=color, linewidth=2.0, label=label)

        conv_ep = _estimate_convergence_episode(rolling, stable_window=max(8, window // 2))
        if conv_ep is not None and conv_ep <= len(rolling):
            conv_y = rolling[conv_ep - 1]
            ax.scatter([conv_ep], [conv_y], color=color, s=40, zorder=5)
            ax.axvline(conv_ep, color=color, linestyle=":", linewidth=1.0, alpha=0.5)
            ax.text(conv_ep, conv_y, f"  conv@{conv_ep}", fontsize=8.5, color=color, va="bottom")

    ax.set_title(
        f"Training Reward - Rolling Average  (window={window} episodes)",
        fontsize=12.5, fontweight="bold",
    )
    ax.set_xlabel("Episode", fontsize=10.5)
    ax.set_ylabel("Reward", fontsize=10.5)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=11, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)


def _draw_metrics_summary_panel(ax: Any, runs: list[AgentReportRun], window: int) -> None:
    """Render a compact table of training and clean-evaluation metrics."""
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    col_labels = [
        "Agent",
        "Episodes",
        "Train mean",
        "Train best",
        "Train last 10%",
        f"Conv ep (w={window})",
        "Eval mean",
        "Eval std",
        "Eval n",
    ]

    cell_text: list[list[str]] = []
    for run in runs:
        train_rewards = run.train_result.episode_rewards
        rolling = _rolling_mean(train_rewards, window)
        conv_ep = _estimate_convergence_episode(rolling, stable_window=max(8, window // 2))
        eval_n = len(run.eval_rewards)

        row = [
            _legend_label(run),
            str(len(train_rewards)),
            f"{run.train_result.mean_reward:.3f}",
            f"{run.train_result.best_reward:.3f}",
            f"{run.train_result.last_n_mean:.3f}",
            str(conv_ep) if conv_ep is not None else "n/a",
            f"{run.eval_mean:.3f}" if run.eval_mean is not None else "n/a",
            f"{run.eval_std:.3f}" if run.eval_std is not None else "n/a",
            str(eval_n) if eval_n else "n/a",
        ]
        cell_text.append(row)

    if not cell_text:
        ax.text(
            0.5, 0.5,
            "No training/evaluation metrics available",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="#999999",
        )
        return

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.65)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#e8edf3")
            cell.set_text_props(weight="bold", color="#1b1f24")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 else "#f8fafc")
        if col_idx == 0:
            cell.set_text_props(ha="left")
        cell.set_edgecolor("#d1d9e0")
        cell.set_linewidth(0.6)

    ax.set_title(
        "Training + Clean Evaluation Summary Metrics",
        fontsize=12.5, fontweight="bold",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone figure builders for instructions and config
# ─────────────────────────────────────────────────────────────────────────────

# Points-to-inches conversion
_PT_TO_IN = 1.0 / 72.0


def _build_instruction_figure(
    runs: list[AgentReportRun],
    fig_width: float,
    dpi: int,
    env_label: str,
) -> Any:
    """
    Full-page instruction match summary.

    Text is placed at deterministic y-positions computed from line counts so
    nothing ever overlaps regardless of instruction or observation length.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    FONT_PT = 10.0
    LINE_H_IN = FONT_PT * _PT_TO_IN * 1.55   # vertical space per text line
    GAP_H_IN = 0.30                           # gap between agent blocks
    STEP_INDENT_H_IN = 0.10                   # extra gap before each sub-step block
    TITLE_H_IN = 0.60                         # reserved at top
    FOOTER_H_IN = 0.20                        # reserved at bottom
    INST_W = 72                               # wrap width for instruction text
    OBS_W  = 68                               # wrap width for matched-lang / render text

    palette_bg   = ["#e8f4fd", "#fef9e7", "#e9f7ef", "#fdf2f8", "#fff3e0", "#f0f4ff"]
    palette_step = ["#d0e8f8", "#fdf0c0", "#d0f0e0", "#f5d8f0", "#ffe8c0", "#dce4ff"]

    # ── Pre-compute text lines per agent block ────────────────────────────────
    # Each entry: (text, is_header, indent_level)
    # indent_level  0 = block header
    #               1 = top-level field row
    #               2 = sub-step header
    #               3 = sub-step body row

    blocks: list[dict[str, Any]] = []
    for idx, run in enumerate(runs, start=1):
        lines: list[tuple[str, int]] = []   # (text, indent_level)

        header = f"[{idx}]  {_legend_label(run)}"
        lines.append((header, 0))

        # Original / top-level instruction
        instr_text = run.instruction or "—"
        instr_wrapped = textwrap.wrap(instr_text, width=INST_W) or ["—"]
        lines.append((f"  Instruction :  {instr_wrapped[0]}", 1))
        for extra in instr_wrapped[1:]:
            lines.append((f"                 {extra}", 1))

        if run.sub_steps_matches:
            # Per-sub-step section - hides the legacy "Matched obs" row
            lines.append((f"  Sub-steps   :  {len(run.sub_steps_matches)} matched", 1))
            for si, sm in enumerate(run.sub_steps_matches, start=1):
                step_instr   = str(sm.get("instruction", ""))
                matched_lang = str(sm.get("matched_language", "") or "—")
                similarity   = sm.get("similarity")
                render_text  = sm.get("render_text") or matched_lang  # text render of matched obs
                render_b64   = sm.get("render_b64")   # base64 PNG (unused here, for future)

                # Sub-step header line
                step_prefix = f"  [{si}] "
                step_wrapped = textwrap.wrap(step_instr, width=INST_W - len(step_prefix)) or [step_instr]
                lines.append((f"{step_prefix}{step_wrapped[0]}", 2))
                for extra in step_wrapped[1:]:
                    lines.append((f"      {'':>{len(step_prefix)-6}}{extra}", 2))

                # Matched observation (= language render of matched state)
                obs_wrapped = textwrap.wrap(matched_lang, width=OBS_W) or ["—"]
                lines.append((f"      Obs match   :  {obs_wrapped[0]}", 3))
                for extra in obs_wrapped[1:]:
                    lines.append((f"                    {extra}", 3))

                # Render (same as matched_language for text envs; kept separate
                # so future visual-env support can substitute render_b64)
                if render_text and render_text != matched_lang:
                    render_wrapped = textwrap.wrap(render_text, width=OBS_W) or [render_text]
                    lines.append((f"      Render      :  {render_wrapped[0]}", 3))
                    for extra in render_wrapped[1:]:
                        lines.append((f"                    {extra}", 3))

                # Similarity
                if similarity is not None:
                    sim_pct = max(0.0, min(1.0, float(similarity))) * 100.0
                    lines.append((f"      Similarity  :  {sim_pct:.2f}%", 3))
                else:
                    lines.append(("      Similarity  :  —", 3))
        else:
            # Legacy single-instruction format
            obs_text = str(run.best_match_observation) if run.best_match_observation else "—"
            obs_wrapped = textwrap.wrap(obs_text, width=OBS_W) or ["—"]
            lines.append((f"  Matched obs :  {obs_wrapped[0]}", 1))
            for extra in obs_wrapped[1:]:
                lines.append((f"                 {extra}", 1))

            if run.best_match_similarity is not None:
                sim_pct = max(0.0, min(1.0, float(run.best_match_similarity))) * 100.0
                lines.append((f"  Similarity  :  {sim_pct:.2f}%", 1))
            else:
                lines.append(("  Similarity  :  —", 1))

        blocks.append({
            "run": run,
            "lines": lines,
            "bg": palette_bg[idx % len(palette_bg)],
            "step_bg": palette_step[idx % len(palette_step)],
        })

    # Compute figure height
    total_content_h = sum(len(b["lines"]) * LINE_H_IN for b in blocks)
    total_gap_h = max(0, len(blocks) - 1) * GAP_H_IN
    fig_h = TITLE_H_IN + total_content_h + total_gap_h + FOOTER_H_IN
    fig_h = max(4.0, min(fig_h + 0.6, 60.0))   # raised cap for many sub-steps

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#f7f7f4")
    ax.set_facecolor("#f7f7f4")
    ax.axis("off")

    fig.suptitle(
        f"rlbridge Instruction Match Summary ({len(runs)} agent(s)){env_label}",
        fontsize=15, fontweight="bold", color="#1b1f24", y=0.995,
    )

    # Available height in figure-fraction units for the axes region
    avail_h_in = fig_h - TITLE_H_IN - FOOTER_H_IN
    avail_frac = avail_h_in / fig_h  # height of axes as fraction of figure

    # Set axes to fill available area
    bottom_frac = FOOTER_H_IN / fig_h
    ax.set_position([0.01, bottom_frac, 0.98, avail_frac])

    # In axes coordinates (0=bottom, 1=top): work top-down
    line_h_ax = LINE_H_IN / avail_h_in      # one line in axes y-units
    gap_h_ax = GAP_H_IN / avail_h_in        # inter-block gap

    y = 1.0  # start at top of axes

    for block in blocks:
        run_lines = block["lines"]
        n_lines = len(run_lines)
        block_h_ax = n_lines * line_h_ax
        step_bg = block["step_bg"]

        # Background rectangle for the whole agent block
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.005, y - block_h_ax - 0.005),
            0.990, block_h_ax + 0.008,
            boxstyle="round,pad=0.005",
            facecolor=block["bg"],
            edgecolor="#cccccc",
            linewidth=0.7,
            transform=ax.transAxes,
            zorder=1,
            clip_on=False,
        ))

        # Draw each line; sub-step body rows (indent 2/3) get a tinted background
        prev_step_start: Optional[float] = None
        prev_indent: int = 0
        for j, (text, indent) in enumerate(run_lines):
            line_y = y - j * line_h_ax

            # Draw a background band for sub-step groups (indent 2 starts a group,
            # indent 3 continues it).  Close the band when indent drops back.
            if indent == 2:
                # Start a new sub-step highlight band
                prev_step_start = line_y
                prev_indent = 2
            elif indent == 3 and prev_step_start is not None:
                prev_indent = 3
            elif prev_step_start is not None and indent < 2:
                # Close the band
                band_top = prev_step_start
                band_bot = line_y  # current line is outside the band
                ax.add_patch(mpatches.FancyBboxPatch(
                    (0.010, band_bot),
                    0.980, band_top - band_bot,
                    boxstyle="round,pad=0.002",
                    facecolor=step_bg,
                    edgecolor="#aaaaaa",
                    linewidth=0.4,
                    transform=ax.transAxes,
                    zorder=1,
                    clip_on=False,
                ))
                prev_step_start = None
                prev_indent = indent

            is_header = indent == 0
            is_step_hdr = indent == 2
            ax.text(
                0.015,
                line_y,
                text,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=FONT_PT + (0.5 if is_header else 0.0),
                fontweight="bold" if (is_header or is_step_hdr) else "normal",
                family="monospace",
                color="#1b1f24",
                zorder=2,
                clip_on=False,
            )

        # Close any open sub-step band at the bottom of the block
        if prev_step_start is not None:
            band_bot = y - n_lines * line_h_ax
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.010, band_bot),
                0.980, prev_step_start - band_bot,
                boxstyle="round,pad=0.002",
                facecolor=step_bg,
                edgecolor="#aaaaaa",
                linewidth=0.4,
                transform=ax.transAxes,
                zorder=1,
                clip_on=False,
            ))

        y -= block_h_ax + gap_h_ax

    return fig


def _build_config_figure(
    runs: list[AgentReportRun],
    fig_width: float,
    dpi: int,
    env_label: str,
) -> Any:
    """
    Full-page metadata and hyper-parameter report.

    Text is placed at deterministic y-positions so nothing overlaps.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    FONT_PT = 9.5
    LINE_H_IN = FONT_PT * _PT_TO_IN * 1.6
    GAP_H_IN = 0.35
    TITLE_H_IN = 0.60
    FOOTER_H_IN = 0.20
    WRAP_W = 60
    palette_bg = ["#e8f4fd", "#fef9e7", "#e9f7ef", "#fdf2f8", "#fff3e0", "#f0f4ff"]

    blocks: list[dict[str, Any]] = []
    for idx, run in enumerate(runs, start=1):
        lines: list[tuple[str, bool]] = []

        lines.append((f"[{idx}]  {run.label}", True))

        # Metadata
        if run.metadata:
            lines.append(("  metadata:", False))
            for k, v in sorted(run.metadata.items()):
                wrapped = textwrap.wrap(f"{k}: {v}", width=WRAP_W) or [f"{k}: -"]
                lines.append((f"    {wrapped[0]}", False))
                for extra in wrapped[1:]:
                    lines.append((f"      {extra}", False))
        else:
            lines.append(("  metadata: —", False))

        # Hyperparameters
        if run.hyperparameters:
            lines.append(("  hyper-parameters:", False))
            for k, v in sorted(run.hyperparameters.items()):
                wrapped = textwrap.wrap(f"{k}: {v}", width=WRAP_W) or [f"{k}: -"]
                lines.append((f"    {wrapped[0]}", False))
                for extra in wrapped[1:]:
                    lines.append((f"      {extra}", False))
        else:
            lines.append(("  hyper-parameters: —", False))

        # Summary stats
        rewards = run.train_result.episode_rewards
        if rewards:
            lines.append(("  training summary:", False))
            summary_parts = [
                f"episodes={len(rewards)}",
                f"mean={run.train_result.mean_reward:.4f}",
                f"best={run.train_result.best_reward:.4f}",
                f"last_10%={run.train_result.last_n_mean:.4f}",
            ]
            if run.eval_mean is not None:
                eval_str = f"eval_mean={run.eval_mean:.4f}"
                if run.eval_std is not None:
                    eval_str += f" ±{run.eval_std:.4f}"
                eval_str += " (100ep, fixed weights)"
                summary_parts.append(eval_str)
            for chunk in textwrap.wrap("  ".join(summary_parts), width=WRAP_W):
                lines.append((f"    {chunk}", False))

        blocks.append({
            "lines": lines,
            "bg": palette_bg[idx % len(palette_bg)],
        })

    # Compute figure height
    total_content_h = sum(len(b["lines"]) * LINE_H_IN for b in blocks)
    total_gap_h = max(0, len(blocks) - 1) * GAP_H_IN
    fig_h = TITLE_H_IN + total_content_h + total_gap_h + FOOTER_H_IN
    fig_h = max(4.0, min(fig_h + 0.6, 32.0))

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#f7f7f4")
    ax.set_facecolor("#f7f7f4")
    ax.axis("off")

    fig.suptitle(
        f"rlbridge Metadata & Hyper-parameters ({len(runs)} agent(s)){env_label}",
        fontsize=15, fontweight="bold", color="#1b1f24", y=0.995,
    )

    avail_h_in = fig_h - TITLE_H_IN - FOOTER_H_IN
    avail_frac = avail_h_in / fig_h
    bottom_frac = FOOTER_H_IN / fig_h
    ax.set_position([0.01, bottom_frac, 0.98, avail_frac])

    line_h_ax = LINE_H_IN / avail_h_in
    gap_h_ax = GAP_H_IN / avail_h_in
    y = 1.0

    for block in blocks:
        n_lines = len(block["lines"])
        block_h_ax = n_lines * line_h_ax

        ax.add_patch(mpatches.FancyBboxPatch(
            (0.005, y - block_h_ax - 0.005),
            0.990, block_h_ax + 0.008,
            boxstyle="round,pad=0.005",
            facecolor=block["bg"],
            edgecolor="#cccccc",
            linewidth=0.7,
            transform=ax.transAxes,
            zorder=1,
            clip_on=False,
        ))

        for j, (text, is_header) in enumerate(block["lines"]):
            ax.text(
                0.015,
                y - j * line_h_ax,
                text,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=FONT_PT + (0.5 if is_header else 0.0),
                fontweight="bold" if is_header else "normal",
                family="monospace",
                color="#1b1f24",
                zorder=2,
                clip_on=False,
            )

        y -= block_h_ax + gap_h_ax

    return fig


__all__ = ["AgentReportRun", "create_training_report"]

