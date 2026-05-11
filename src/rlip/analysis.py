"""
RLIP Training Analysis & Reporting
====================================
Generates a multi-panel training report figure combining:

- **Reward curve** – per-episode reward and a rolling average over the full
  training run.
- **Sample frames** – evenly-spaced RGB frames from the best training episode
  replayed through the policy (requires a ``PolicyRenderResult``).
- **Metadata table** – agent type, hyper-parameters, training statistics, and
  any caller-supplied extra facts.
- **Sub-goal panel** – when instruction-following / sub-goal shaping was used,
  shows the cosine-similarity trajectory over the steps of the best episode
  with markers at each step the threshold was met.

Quick start
-----------
::

    from rlip.analysis import create_training_report

    fig = create_training_report(
        train_result=train_result,          # TrainResult from agent.train()
        render_result=render_result,        # PolicyRenderResult (optional)
        subgoal_steps=subgoal_steps,        # list of (step, sim, reached)
        subgoal_info={"instruction": "...", "sub_goal_language": "...",
                      "threshold": 0.5, "bonus": 1.0},
        metadata={"env_id": "Sailing-v0", "match_id": "abc123"},
        output_path="report.png",
    )
"""

from __future__ import annotations

from typing import Any, Optional

from ._agent_base import TrainResult
from .policy_rendering import PolicyRenderResult


# ── Rolling mean ──────────────────────────────────────────────────────────────

def _rolling_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _sample_evenly(items: list[Any], n: int) -> list[Any]:
    """Return at most *n* evenly-spaced items from *items*."""
    if not items:
        return []
    if len(items) <= n:
        return items
    if n == 1:
        return [items[0]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


# ── Public entry-point ────────────────────────────────────────────────────────

def create_training_report(
    train_result: TrainResult,
    render_result: Optional[PolicyRenderResult] = None,
    metadata: Optional[dict[str, Any]] = None,
    subgoal_steps: Optional[list[tuple[int, float, bool]]] = None,
    subgoal_info: Optional[dict[str, Any]] = None,
    *,
    output_path: Optional[str] = None,
    rolling_window: Optional[int] = None,
    n_sample_frames: int = 6,
    fig_width: float = 16.0,
    dpi: int = 120,
) -> Any:
    """
    Generate a multi-panel RL training report figure.

    Parameters
    ----------
    train_result:
        ``TrainResult`` (or subclass) returned by an agent's ``.train()``
        method.
    render_result:
        Optional ``PolicyRenderResult`` from ``render_optimal_policy()``.
        When provided, evenly-sampled RGB frames from the best episode
        are shown in a strip.
    metadata:
        Extra key-value pairs added to the metadata table (e.g.
        ``{"env_id": "Sailing-v0", "use_language_state": True}``).
        Values are coerced to ``str`` and truncated to 60 characters.
    subgoal_steps:
        Per-step sub-goal similarity data for the sub-goal panel.
        Each element is a 3-tuple ``(step: int, similarity: float,
        reached: bool)``.  Computed externally (e.g. by post-hoc analysis
        of the best training trajectory) and passed here so this module
        stays side-effect-free.
    subgoal_info:
        Descriptive metadata about the sub-goal shown as a text block in
        the sub-goal panel.  Recognised keys:
        ``instruction``, ``sub_goal_language``, ``threshold``, ``bonus``,
        ``n_subgoals``, ``match_id``.
    output_path:
        If given, the figure is saved to this path (format inferred from
        extension: ``.png``, ``.pdf``, ``.svg``, …).  The figure is also
        returned.
    rolling_window:
        Number of episodes for the rolling reward average.  Defaults to
        5 % of total episodes, minimum 10.
    n_sample_frames:
        Maximum number of evenly-spaced frames to show from the best
        episode (only used when *render_result* is provided).
    fig_width:
        Figure width in inches.
    dpi:
        Output resolution (PNG / raster formats only).

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.  Use ``plt.show()`` to display interactively
        or ``fig.savefig(path)`` to export manually.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for training reports.  "
            "Install it with:  pip install matplotlib"
        ) from exc

    episodes = train_result.episode_rewards
    n_ep = len(episodes)
    window = rolling_window or max(10, int(n_ep * 0.05))
    rolling = _rolling_mean(episodes, window)

    # Collect sample frames (RGB frames with PNG data)
    frames_with_png = [
        f for f in (render_result.frames if render_result else [])
        if f.png_data
    ]
    sample_frames = _sample_evenly(frames_with_png, n_sample_frames)
    has_frames = bool(sample_frames)

    has_subgoal = bool(subgoal_steps)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Row 0: reward curve  (always, taller)
    # Row 1: frames strip  (optional)
    # Row 2: metadata table (left)  +  sub-goal panel (right)
    row_heights: list[float] = [3.0]
    if has_frames:
        row_heights.append(2.5)
    row_heights.append(2.5)

    n_rows = len(row_heights)
    fig_height = sum(row_heights) + 0.7 * n_rows
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("#f8f9fa")

    gs = gridspec.GridSpec(
        n_rows, 2,
        figure=fig,
        hspace=0.60,
        wspace=0.30,
        height_ratios=row_heights,
    )

    # Row 0: reward curve (spans both columns)
    ax_reward = fig.add_subplot(gs[0, :])
    _draw_reward_curve(ax_reward, episodes, rolling, window, train_result)

    bottom_row = n_rows - 1

    # Row 1 (optional): frames strip
    if has_frames:
        _draw_frames_strip(fig, gs, 1, sample_frames, render_result)

    # Bottom row left: metadata table
    ax_meta = fig.add_subplot(gs[bottom_row, 0])
    _draw_metadata_table(ax_meta, train_result, render_result, metadata)

    # Bottom row right: sub-goal panel
    ax_sg = fig.add_subplot(gs[bottom_row, 1])
    if has_subgoal:
        _draw_subgoal_panel(ax_sg, subgoal_steps, subgoal_info)
    else:
        _draw_no_subgoal(ax_sg, subgoal_info)

    # Super-title
    env_label = ""
    if render_result:
        env_label = f"  ·  {render_result.env_id}"
    elif metadata and "env_id" in metadata:
        env_label = f"  ·  {metadata['env_id']}"
    fig.suptitle(
        f"RLIP Training Report  ·  {train_result.agent_name}{env_label}",
        fontsize=14,
        fontweight="bold",
        y=1.005,
        color="#1a1a2e",
    )

    if output_path:
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    return fig


# ── Sub-plot renderers ────────────────────────────────────────────────────────

def _draw_reward_curve(
    ax: Any,
    episodes: list[float],
    rolling: list[float],
    window: int,
    result: TrainResult,
) -> None:
    """Per-episode reward + rolling average + best-episode marker."""
    xs = list(range(1, len(episodes) + 1))

    ax.set_facecolor("#ffffff")
    ax.plot(
        xs, episodes,
        color="#b0c4de", linewidth=0.7, alpha=0.55,
        label="Episode reward",
    )
    ax.plot(
        xs, rolling,
        color="#e07b39", linewidth=2.0,
        label=f"Rolling mean (w={window})",
    )

    # Mark best episode
    best_idx = max(range(len(episodes)), key=lambda i: episodes[i])
    ax.axvline(
        best_idx + 1,
        color="#27ae60", linestyle="--", linewidth=1.2,
        label=f"Best ep #{best_idx + 1}  (R={episodes[best_idx]:.2f})",
    )
    ax.scatter([best_idx + 1], [episodes[best_idx]], color="#27ae60", s=60, zorder=5)

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Total Reward", fontsize=10)
    ax.set_title("Training Reward Curve", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Summary stats annotation
    stats = (
        f"mean={result.mean_reward:.3f}   "
        f"best={result.best_reward:.3f}   "
        f"last 10%={result.last_n_mean:.3f}   "
        f"ε_final={result.final_epsilon:.4f}"
    )
    ax.text(
        0.99, 0.04, stats,
        transform=ax.transAxes,
        fontsize=8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.80),
    )


def _draw_frames_strip(
    fig: Any,
    gs: Any,
    row: int,
    frames: list[Any],
    render_result: PolicyRenderResult,
) -> None:
    """Evenly-sampled RGB frames from the best policy episode."""
    from io import BytesIO

    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return  # silently skip if Pillow not available

    n = len(frames)
    if n == 0:
        return

    inner_gs = gs[row, :].subgridspec(1, n, wspace=0.04)

    last_ax = None
    for col, frame in enumerate(frames):
        ax = fig.add_subplot(inner_gs[0, col])
        last_ax = ax
        img = Image.open(BytesIO(frame.png_data)).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        # Title under each frame
        parts = [f"step {frame.step}  r={frame.reward:+.2f}"]
        if frame.language_obs:
            parts.append(frame.language_obs[:40])
        if frame.sub_goal_reached:
            parts.append("◀ sub-goal reached")
        ax.set_title("\n".join(parts), fontsize=6.5, pad=2)

        # Red border for sub-goal frames
        if frame.sub_goal_reached:
            for spine in ax.spines.values():
                spine.set_edgecolor("#e74c3c")
                spine.set_linewidth(2)
                spine.set_visible(True)

    # Section label above the strip
    if last_ax is not None:
        y1 = last_ax.get_subplotspec().get_gridspec().get_subplot_params().top
        fig.text(
            0.5,
            y1 + 0.005,
            f"Sample Frames — Best Episode  "
            f"(R={render_result.best_episode_reward:.3f}, "
            f"{render_result.best_episode_steps} steps)",
            ha="center", fontsize=10, fontweight="bold",
            transform=fig.transFigure,
        )


def _draw_metadata_table(
    ax: Any,
    result: TrainResult,
    render_result: Optional[PolicyRenderResult],
    extra: Optional[dict[str, Any]],
) -> None:
    """Left panel: agent and training metadata as a styled table."""
    ax.axis("off")
    ax.set_facecolor("#ffffff")

    rows: list[tuple[str, str]] = []

    rows.append(("Agent type", result.agent_name))
    rows.append(("Training episodes", str(result.n_episodes)))
    rows.append(("Mean reward", f"{result.mean_reward:.4f}"))
    rows.append(("Best reward", f"{result.best_reward:.4f}"))
    rows.append(("Last 10% mean", f"{result.last_n_mean:.4f}"))
    rows.append(("Final ε", f"{result.final_epsilon:.6f}"))

    # Agent-type specific fields
    for attr, label in [
        ("q_table_size",  "Q-table size"),
        ("obs_dim",       "Obs dimension"),
        ("mean_loss",     "Mean train loss"),
    ]:
        val = getattr(result, attr, None)
        if val is not None:
            fmt = f"{val:.6f}" if isinstance(val, float) else str(val)
            rows.append((label, fmt))

    if render_result:
        rows.append(("Environment", render_result.env_id))
        rows.append(("Best ep reward", f"{render_result.best_episode_reward:.4f}"))
        rows.append(("Best ep steps", str(render_result.best_episode_steps)))
        rows.append(("Policy states", str(render_result.policy_size)))

    if extra:
        for k, v in extra.items():
            display_key = str(k).replace("_", " ").title()
            display_val = str(v)[:60]
            rows.append((display_key, display_val))

    cell_text = [[r[0], r[1]] for r in rows]
    col_labels = ["Parameter", "Value"]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)

    # Header row
    for col in range(2):
        tbl[(0, col)].set_facecolor("#2c3e50")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")
    # Alternating rows
    for row_i in range(1, len(rows) + 1):
        bg = "#f0f4f8" if row_i % 2 == 0 else "#ffffff"
        for col in range(2):
            tbl[(row_i, col)].set_facecolor(bg)
            tbl[(row_i, col)].set_edgecolor("#dddddd")

    ax.set_title("Agent & Training Metadata", fontsize=10, fontweight="bold", pad=8)


def _draw_subgoal_panel(
    ax: Any,
    subgoal_steps: list[tuple[int, float, bool]],
    subgoal_info: Optional[dict[str, Any]],
) -> None:
    """
    Right panel: cosine-similarity trajectory and sub-goal reach events.

    Parameters
    ----------
    subgoal_steps:
        Per-step data as ``(step, similarity, reached)`` tuples.
    subgoal_info:
        Descriptive metadata shown in a text box below the chart.
    """
    steps   = [s for s, _, _ in subgoal_steps]
    sims    = [sim for _, sim, _ in subgoal_steps]
    reached = [(s, sim) for s, sim, hit in subgoal_steps if hit]

    threshold = 0.5
    if subgoal_info:
        threshold = float(subgoal_info.get("threshold", threshold))

    ax.set_facecolor("#ffffff")
    ax.plot(steps, sims, color="#3498db", linewidth=1.5, label="Similarity")
    ax.fill_between(steps, sims, alpha=0.12, color="#3498db")
    ax.axhline(
        threshold, color="#e74c3c", linestyle="--", linewidth=1.0,
        label=f"Threshold ({threshold})",
    )

    if reached:
        r_steps, r_sims = zip(*reached)
        ax.scatter(
            r_steps, r_sims,
            color="#27ae60", s=80, zorder=5, label="Sub-goal reached",
        )
        for s in r_steps:
            ax.axvline(s, color="#27ae60", linestyle=":", linewidth=0.9, alpha=0.6)

    # Summary text
    peak = max(sims) if sims else 0.0
    summary_lines = [
        f"Peak sim: {peak:.4f}",
        (f"Sub-goal reached {len(reached)}× (first: step {reached[0][0]})"
         if reached else "Sub-goal never reached in best episode"),
    ]
    if subgoal_info:
        instr = subgoal_info.get("instruction", "")
        if instr:
            summary_lines.insert(0, f"Instruction: {instr[:55]!r}")
        sg_lang = subgoal_info.get("sub_goal_language", "")
        if sg_lang:
            summary_lines.append(f"Sub-goal: {sg_lang[:55]!r}")
        bonus = subgoal_info.get("bonus")
        if bonus is not None:
            summary_lines.append(f"Bonus: {bonus}  Threshold: {threshold}")

    ax.text(
        0.99, 0.04,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=7, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.85),
    )

    ax.set_xlabel("Step (best training episode)", fontsize=9)
    ax.set_ylabel("Cosine similarity to sub-goal", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Sub-goal / Instruction Similarity", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_no_subgoal(
    ax: Any,
    subgoal_info: Optional[dict[str, Any]],
) -> None:
    """Placeholder when no sub-goal similarity data is available."""
    ax.set_facecolor("#f8f9fa")
    ax.axis("off")
    ax.set_title("Sub-goal / Instruction Similarity", fontsize=10, fontweight="bold")

    if subgoal_info:
        # Sub-goal shaping WAS used but we have no per-step similarity data
        lines = ["Sub-goal shaping was applied during training.\n"]
        for key, label in [
            ("instruction",       "Instruction"),
            ("sub_goal_language", "Matched state"),
            ("threshold",         "Threshold"),
            ("bonus",             "Reward bonus"),
            ("match_id",          "Match ID"),
        ]:
            val = subgoal_info.get(key)
            if val is not None:
                val_str = str(val)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "…"
                lines.append(f"{label}: {val_str}")
        ax.text(
            0.5, 0.55,
            "\n".join(lines),
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=8.5, color="#2c3e50",
            linespacing=1.6,
        )
        ax.text(
            0.5, 0.08,
            "Per-step similarity data not available.\n"
            "Re-train and call rl_create_training_report() to capture it.",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=7.5, color="#888888",
        )
    else:
        ax.text(
            0.5, 0.58,
            "No instruction / sub-goal used.",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=10, color="#666666",
        )
        ax.text(
            0.5, 0.40,
            "Train with  match_id=…  (from rl_match_instruction)\nto enable sub-goal similarity tracking.",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=8, color="#999999",
        )


__all__ = ["create_training_report"]
