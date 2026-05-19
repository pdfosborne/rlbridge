"""
Policy Rendering for RLIP Interaction Protocols
================================================
Given a completed :class:`~rlip.interaction_protocols.InteractionResult`,
this module:

1. **Extracts an optimal policy** – selects the best episode by total reward
   and builds a greedy observation-to-action lookup table from its trajectory.

2. **Replays the policy** – runs a fresh environment episode, consulting the
   lookup table at each step and falling back to a random action for
   unseen observations.

3. **Renders each step** – captures ``rgb_array`` PNG frames or ``ansi`` text
   from the environment's ``render()`` call after every action.

4. **Saves output** – writes individual PNG frames to a directory and/or
   composes them into an animated GIF (requires Pillow, already a core dep).

Quick start
-----------
::

    from rlip.environments.predefined.sailing import SailingFactory, SAILING_V0
    from rlip.interaction_protocols import RandomEpisodeProtocol
    from rlip.policy_rendering import render_optimal_policy

    # 1. Run any interaction protocol to collect episode data.
    env = SAILING_V0.create()
    result = RandomEpisodeProtocol(max_steps=200, seed=0, record_history=True)(env)

    # 2. Render the optimal (best-reward) episode to a GIF.
    render_result = render_optimal_policy(
        result,
        env_factory=SAILING_V0,
        output_gif="sailing_policy.gif",
    )
    print(render_result)

Instruction-following
---------------------
Works with :class:`~rlip.interaction_protocols.InstructionFollowingProtocol`
results too — sub-goal steps are annotated in the frame metadata::

    from rlip.instruction_following import build_sequential_instruction_following_protocol

    protocol = build_sequential_instruction_following_protocol(
        ["sail towards the beach side"], env, seed=0
    )
    result = protocol(env)
    render_optimal_policy(result, env_factory=SAILING_V0, output_gif="if_policy.gif")
"""

from __future__ import annotations

import base64
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .environments.base import RLIPEnvironment, RLIPEnvironmentFactory
from .interaction_protocols import (
    EpisodeResult,
    InteractionResult,
    StepRecord,
    _EnvLike,
    _get,
    _make_sampler,
)


# ── Policy extraction ─────────────────────────────────────────────────────────

def extract_optimal_policy(
    result: InteractionResult,
) -> tuple[EpisodeResult, dict[Any, Any]]:
    """
    Select the best episode from *result* and build a greedy lookup table.

    The "optimal policy" derived here is a **tabular greedy policy**: a dict
    mapping each observed state to the action taken at that state in the
    highest-reward episode.  When multiple steps visit the same state, the
    action from the *last* visit is kept (later steps reflect a more refined
    trajectory).

    Parameters
    ----------
    result:
        A completed :class:`~rlip.interaction_protocols.InteractionResult`
        with ``record_history=True``.

    Returns
    -------
    best_episode : EpisodeResult
        The episode with the highest ``total_reward``.
    policy : dict[obs, action]
        Observation-to-action mapping extracted from the best episode.

    Raises
    ------
    ValueError
        If *result* contains no episodes, or none have recorded history.
    """
    if not result.episodes:
        raise ValueError("InteractionResult contains no episodes.")

    episodes_with_history = [ep for ep in result.episodes if ep.history]
    if not episodes_with_history:
        raise ValueError(
            "No episode history found.  Re-run the protocol with "
            "record_history=True to capture trajectories."
        )

    best_episode = max(episodes_with_history, key=lambda ep: ep.total_reward)

    policy: dict[Any, Any] = {}
    for rec in best_episode.history:
        # Coerce numpy scalars / arrays to plain Python for hashability
        key = _hashable_obs(rec.observation)
        policy[key] = rec.action

    return best_episode, policy


def _hashable_obs(obs: Any) -> Any:
    """Convert an observation to a hashable key (handles lists/numpy arrays)."""
    if isinstance(obs, dict):
        # Deterministic key order so policy lookup is stable across runs.
        return tuple(
            (str(k), _hashable_obs(v))
            for k, v in sorted(obs.items(), key=lambda item: str(item[0]))
        )
    if isinstance(obs, (list, tuple)):
        return tuple(_hashable_obs(v) for v in obs)
    if isinstance(obs, float):
        return round(obs, 6)
    if obs is None:
        return None
    try:
        import numpy as np  # noqa: PLC0415
        if isinstance(obs, np.ndarray):
            return tuple(_hashable_obs(v) for v in obs.flatten().tolist())
    except ImportError:
        pass
    try:
        hash(obs)
        return obs
    except TypeError:
        return repr(obs)


# ── Rendered frame ────────────────────────────────────────────────────────────

@dataclass
class RenderedFrame:
    """One rendered step from a policy replay."""

    step: int
    """Step number (1-based)."""

    action: Any
    """Action executed at this step."""

    observation: Any
    """Raw observation after the action."""

    reward: float
    """Reward received at this step."""

    terminated: bool
    truncated: bool

    mode: str
    """Render mode: ``"rgb_array"``, ``"ansi"``, or ``"none"``."""

    png_data: Optional[bytes] = field(default=None, repr=False)
    """Decoded PNG bytes when mode is ``"rgb_array"``."""

    ansi_text: Optional[str] = field(default=None, repr=False)
    """Text content when mode is ``"ansi"``."""

    language_obs: Optional[str] = field(default=None, repr=False)
    """Language description of the observation (when a translator is active)."""

    sub_goal_reached: bool = False
    """True if this step's language matched the instruction sub-goal."""

    sub_goal_similarity: Optional[float] = None
    """Cosine similarity to the sub-goal language, if available."""


# ── Render result ─────────────────────────────────────────────────────────────

@dataclass
class PolicyRenderResult:
    """Output of :func:`render_optimal_policy`."""

    protocol_name: str
    env_id: str
    best_episode_index: int
    best_episode_reward: float
    best_episode_steps: int
    frames: list[RenderedFrame] = field(default_factory=list, repr=False)
    output_dir: Optional[str] = None
    output_gif: Optional[str] = None
    output_path_image: Optional[str] = None
    output_overlay_image: Optional[str] = None
    n_frames_saved: int = 0
    n_gif_frames: int = 0
    policy_size: int = 0
    """Number of unique states in the extracted policy table."""

    def __str__(self) -> str:
        lines = [
            f"PolicyRenderResult",
            f"  Protocol:          {self.protocol_name}",
            f"  Environment:       {self.env_id}",
            f"  Best episode:      #{self.best_episode_index}  "
            f"reward={self.best_episode_reward:.4f}  steps={self.best_episode_steps}",
            f"  Policy table size: {self.policy_size} unique observations",
            f"  Frames rendered:   {len(self.frames)}",
        ]
        if self.output_dir:
            lines.append(f"  PNG frames saved:  {self.n_frames_saved}  → {self.output_dir}")
        if self.output_gif:
            lines.append(
                f"  GIF saved:         {self.n_gif_frames} frames  → {self.output_gif}"
            )
        if self.output_path_image:
            lines.append(f"  Path image saved:  {self.output_path_image}")
        if self.output_overlay_image:
            lines.append(f"  Overlay image saved: {self.output_overlay_image}")
        return "\n".join(lines)


# ── Core renderer ─────────────────────────────────────────────────────────────

class PolicyRenderer:
    """
    Replay a greedy policy on a fresh environment instance with rendering.

    Parameters
    ----------
    env:
        An RLIP environment that supports ``render()`` (may be the same
        instance used for training — it will be ``reset()`` first).
    policy:
        Observation-to-action lookup table, typically from
        :func:`extract_optimal_policy`.
    fallback:
        Action to execute when the current observation is not in *policy*.
        Accepts:

        * ``"random"`` (default) – sample from the environment's action space.
        * ``"zero"`` – always use action 0.
        * Any callable ``(obs) -> action``.
    render_mode:
        Override the environment's render mode.  If the environment was
        created without a render mode, pass ``"rgb_array"`` or ``"ansi"``
        here to enable rendering via :meth:`inject_render_mode`.
    translate:
        Language translator source (same semantics as interaction protocols).
    """

    def __init__(
        self,
        env: _EnvLike,
        policy: dict[Any, Any],
        fallback: Any = "random",
        translate: Any = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.fallback = fallback
        self.translate = translate

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> list[RenderedFrame]:
        """
        Run one episode using the greedy policy and collect rendered frames.

        Parameters
        ----------
        max_steps:
            Hard cap on episode length.
        seed:
            Seed for the environment reset.

        Returns
        -------
        list[RenderedFrame]
            One frame per step.  ``png_data`` is populated for ``rgb_array``
            environments; ``ansi_text`` for ``ansi`` environments.
        """
        from .language_translation.base import LanguageTranslator  # noqa: PLC0415

        # Resolve language translator
        translator: Optional[LanguageTranslator] = None
        if self.translate is not None and self.translate is not False:
            if isinstance(self.translate, LanguageTranslator):
                translator = self.translate
            elif self.translate is True:
                env_id = getattr(self.env, "env_id", type(self.env).__name__)
                from .language_translation import get_translator  # noqa: PLC0415
                translator = get_translator(env_id)

        # Build fallback action supplier
        sampler = _make_sampler(self.env, seed)
        if callable(self.fallback) and not isinstance(self.fallback, str):
            fallback_fn: Callable[[Any], Any] = self.fallback
        elif self.fallback == "zero":
            fallback_fn = lambda _obs: 0  # noqa: E731
        else:
            fallback_fn = lambda _obs: sampler()  # noqa: E731

        # Reset
        reset_out = self.env.reset(seed=seed)
        obs = _get(reset_out, "observation", reset_out)

        frames: list[RenderedFrame] = []
        action_history: list[Any] = []

        for step_n in range(1, max_steps + 1):
            # Select action from policy or fallback
            key = _hashable_obs(obs)
            action = self.policy.get(key, fallback_fn(obs))
            action_history.append(action)

            step_out = self.env.step(action)
            obs        = _get(step_out, "observation", obs)
            reward     = float(_get(step_out, "reward", 0.0))
            terminated = bool(_get(step_out, "terminated", False))
            truncated  = bool(_get(step_out, "truncated", False))
            info       = dict(_get(step_out, "info", {}) or {})

            # Language translation
            language_obs: Optional[str] = None
            if translator:
                language_obs = translator.translate(obs, action_history=action_history)

            # Render
            frame = self._capture_frame(
                step=step_n,
                action=action,
                observation=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                language_obs=language_obs,
                info=info,
            )
            frames.append(frame)

            if terminated or truncated:
                break

        return frames

    # ── Internal ──────────────────────────────────────────────────────────────

    def _capture_frame(
        self,
        *,
        step: int,
        action: Any,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        language_obs: Optional[str],
        info: dict[str, Any],
    ) -> RenderedFrame:
        """Call env.render() and return a RenderedFrame."""
        png_data: Optional[bytes] = None
        ansi_text: Optional[str] = None
        mode = "none"

        render_fn = getattr(self.env, "render", None)
        if render_fn is not None:
            try:
                render_out = render_fn()
                mode = _get(render_out, "mode", "none") or "none"
                if mode == "rgb_array":
                    b64 = _get(render_out, "data")
                    if b64:
                        png_data = base64.b64decode(b64)
                elif mode == "ansi":
                    ansi_text = _get(render_out, "text")
            except Exception:
                pass  # render not supported / not initialised with a render mode

        return RenderedFrame(
            step=step,
            action=action,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            mode=mode,
            png_data=png_data,
            ansi_text=ansi_text,
            language_obs=language_obs,
            sub_goal_reached=bool(info.get("sub_goal_reached", False)),
            sub_goal_similarity=info.get("sub_goal_similarity"),
        )


# ── Output helpers ────────────────────────────────────────────────────────────

def save_frames_to_dir(
    frames: list[RenderedFrame],
    output_dir: str | os.PathLike,
) -> int:
    """
    Write each ``rgb_array`` frame as a numbered PNG file.

    Parameters
    ----------
    frames:
        Frames from :meth:`PolicyRenderer.run`.
    output_dir:
        Directory path.  Created if it does not exist.

    Returns
    -------
    int
        Number of PNG files written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for frame in frames:
        if frame.png_data:
            fname = out / f"frame_{frame.step:04d}.png"
            fname.write_bytes(frame.png_data)
            saved += 1
    return saved


def save_path_image(
    frames: list[RenderedFrame],
    output_path: str | os.PathLike,
    max_cols: int = 8,
    thumb_width: int = 120,
    annotate: bool = True,
) -> bool:
    """
    Compose all ``rgb_array`` frames into a single static PNG showing the
    full render path of the policy in one compact image.

    Frames are laid out in a grid (left-to-right, top-to-bottom) so the
    entire episode can be seen at a glance without any animation.

    Parameters
    ----------
    frames:
        Frames from :meth:`PolicyRenderer.run`.
    output_path:
        Destination ``.png`` file path.  Parent directory is created if
        needed.
    max_cols:
        Maximum number of thumbnails per row.
    thumb_width:
        Width (px) to scale each thumbnail to (aspect ratio preserved).
    annotate:
        If *True*, draw a small step / reward label at the top-left of
        each thumbnail.

    Returns
    -------
    bool
        ``True`` if the image was saved, ``False`` if there were no RGB
        frames to composite.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for path image output.  Install it with: "
            "pip install pillow"
        ) from exc

    from io import BytesIO  # noqa: PLC0415

    rgb_frames = [f for f in frames if f.png_data]
    if not rgb_frames:
        return False

    # Build thumbnails
    thumbs: list[Image.Image] = []
    thumb_height = thumb_width  # will be updated from the first frame
    for i, frame in enumerate(rgb_frames):
        img = Image.open(BytesIO(frame.png_data)).convert("RGB")
        scale = thumb_width / img.width
        h = max(1, int(img.height * scale))
        img = img.resize((thumb_width, h), Image.LANCZOS)
        if i == 0:
            thumb_height = h

        if annotate:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("DejaVuSansMono.ttf", 10)
            except (IOError, OSError):
                font = ImageFont.load_default()
            label = f"{frame.step} r={frame.reward:+.2f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([1, 1, lw + 4, lh + 4], fill=(0, 0, 0, 180))
            draw.text((2, 2), label, fill=(255, 255, 255), font=font)

            # Mark terminal frames distinctly
            if frame.terminated:
                draw.rectangle([0, 0, thumb_width - 1, h - 1], outline=(255, 80, 80), width=2)
            elif frame.sub_goal_reached:
                draw.rectangle([0, 0, thumb_width - 1, h - 1], outline=(80, 255, 80), width=2)

        thumbs.append(img)

    n = len(thumbs)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    canvas_w = cols * thumb_width
    canvas_h = rows * thumb_height

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = col * thumb_width
        y = row * thumb_height
        canvas.paste(thumb, (x, y))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return True


def save_overlay_image(
    frames: list[RenderedFrame],
    output_path: str | os.PathLike,
    max_width: int = 480,
) -> bool:
    """
    Composite all ``rgb_array`` frames into a single PNG by overlaying every
    frame at equal opacity, producing a "heat-map" view of all visited states
    simultaneously.

    Each frame contributes an equal share (1/n) to the final pixel value,
    equivalent to an arithmetic mean across all frames.  Early frames are
    tinted with a cool blue hue and late frames with a warm red hue so that
    the trajectory direction remains visible in the blended result.

    Parameters
    ----------
    frames:
        Frames from :meth:`PolicyRenderer.run`.
    output_path:
        Destination ``.png`` file path.  Parent directory is created if
        needed.
    max_width:
        Resize frames to at most this width before compositing.  0 = no
        resize.

    Returns
    -------
    bool
        ``True`` if the image was saved, ``False`` if there were no RGB
        frames to composite.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for overlay image output.  "
            "Install it with: pip install pillow"
        ) from exc

    from io import BytesIO  # noqa: PLC0415

    rgb_frames = [f for f in frames if f.png_data]
    if not rgb_frames:
        return False

    n = len(rgb_frames)

    # Load and optionally resize all frames to a common size.
    def _load(frame: RenderedFrame) -> Image.Image:
        img = Image.open(BytesIO(frame.png_data)).convert("RGB")  # type: ignore[arg-type]
        if max_width > 0 and img.width > max_width:
            scale = max_width / img.width
            img = img.resize((max_width, max(1, int(img.height * scale))), Image.LANCZOS)
        return img

    first = _load(rgb_frames[0])
    target_size = first.size

    # Accumulate pixel sums as floats using a running blend.
    # Each frame is tinted slightly along a blue→red gradient so that
    # trajectory direction is encoded in the colour shift.
    # We blend each frame in at weight 1/(i+1) to maintain a running mean.
    composite = first.convert("RGBA")

    for i, frame in enumerate(rgb_frames[1:], start=1):
        img = _load(frame)
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)

        # Per-channel tint: early frames cool (blue +), late frames warm (red +)
        t = i / max(n - 1, 1)  # 0.0 (first) → 1.0 (last)
        tint_r = int(30 * t)        # red grows toward the end
        tint_b = int(30 * (1 - t))  # blue fades toward the end
        r, g, b = img.split()
        r = r.point(lambda x: min(255, x + tint_r))
        b = b.point(lambda x: min(255, x + tint_b))
        img = Image.merge("RGB", (r, g, b))

        # Incremental equal-weight blend: result = result*(i/(i+1)) + frame*(1/(i+1))
        alpha = 1.0 / (i + 1)
        composite = Image.blend(composite.convert("RGB"), img, alpha=alpha).convert("RGBA")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    composite.convert("RGB").save(output_path, format="PNG", optimize=True)
    return True


def save_gif(
    frames: list[RenderedFrame],
    output_path: str | os.PathLike,
    fps: float = 4.0,
    annotate: bool = True,
    max_width: int = 480,
) -> int:
    """
    Compose ``rgb_array`` frames into an animated GIF using Pillow.

    Parameters
    ----------
    frames:
        Frames from :meth:`PolicyRenderer.run`.
    output_path:
        Destination ``.gif`` file path.  Parent directory is created if
        needed.
    fps:
        Frames per second.  Lower values make the animation slower.
    annotate:
        If *True*, overlay a small text annotation on each frame showing
        the step number, reward, and (if available) sub-goal similarity.
    max_width:
        If > 0, downscale each frame so its width does not exceed this
        value (aspect ratio is preserved).  Keeps GIF file sizes small
        enough to embed as a data URL.  Set to 0 to disable resizing.

    Returns
    -------
    int
        Number of frames written to the GIF (may be 0 if no PNG data).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for GIF output.  Install it with: "
            "pip install pillow"
        ) from exc

    from io import BytesIO  # noqa: PLC0415

    images: list[Image.Image] = []
    for frame in frames:
        if not frame.png_data:
            continue
        img = Image.open(BytesIO(frame.png_data)).convert("RGBA")

        # Downscale if wider than max_width
        if max_width > 0 and img.width > max_width:
            scale = max_width / img.width
            new_size = (max_width, max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.LANCZOS)

        if annotate:
            draw = ImageDraw.Draw(img)

            def _normalize_text(value: Optional[str]) -> str:
                if not value:
                    return ""
                return " ".join(str(value).split()).strip()

            def _is_unknown_sailing(value: str) -> bool:
                return value.lower().startswith("unknown sailing state")

            def _clip_for_width(
                text: str,
                prefix: str,
                font_obj: Any,
                pad: int = 12,
                ellipsis: bool = True,
            ) -> str:
                # Clip by rendered width (not just character count).
                available = max(40, img.width - pad)
                full = f"{prefix}{text}"
                if draw.textbbox((0, 0), full, font=font_obj)[2] <= available:
                    return full
                clipped = text
                suffix = "..." if ellipsis else ""
                while clipped and draw.textbbox((0, 0), f"{prefix}{clipped}{suffix}", font=font_obj)[2] > available:
                    clipped = clipped[:-1]
                if clipped:
                    return f"{prefix}{clipped}{suffix}"
                return prefix + ("..." if ellipsis else "")

            def _wrap_two_lines(text: str, prefix: str, font_obj: Any, pad: int = 12) -> list[str]:
                # Return up to two lines for metadata text.
                one_line = _clip_for_width(text, prefix, font_obj, pad=pad, ellipsis=False)
                if one_line == f"{prefix}{text}":
                    return [one_line]

                available = max(40, img.width - pad)
                words = text.split()
                if not words:
                    return [prefix]

                first_words: list[str] = []
                for w in words:
                    cand = " ".join(first_words + [w])
                    if draw.textbbox((0, 0), f"{prefix}{cand}", font=font_obj)[2] <= available:
                        first_words.append(w)
                    else:
                        break

                if not first_words:
                    first = _clip_for_width(text, prefix, font_obj, pad=pad, ellipsis=False)
                    # If even one character cannot fit cleanly, fallback to ellipsis clip.
                    if first == prefix:
                        return [_clip_for_width(text, prefix, font_obj, pad=pad, ellipsis=True)]
                    consumed = first[len(prefix):].strip()
                    rest = text[len(consumed):].strip() if consumed else text
                    if not rest:
                        return [first]
                    second = _clip_for_width(rest, "  ", font_obj, pad=pad, ellipsis=True)
                    return [first, second]

                first_text = " ".join(first_words)
                first = f"{prefix}{first_text}"
                rest = " ".join(words[len(first_words):]).strip()
                if not rest:
                    return [first]
                second = _clip_for_width(rest, "  ", font_obj, pad=pad, ellipsis=True)
                return [first, second]

            # Load fonts: compact metadata lines use a smaller font.
            try:
                font_main = ImageFont.truetype("DejaVuSansMono.ttf", 14)
                font_meta = ImageFont.truetype("DejaVuSansMono.ttf", 11)
            except (IOError, OSError):
                font_main = ImageFont.load_default()
                font_meta = ImageFont.load_default()

            lines: list[tuple[str, Any]] = []
            lines.append((f"step={frame.step}  r={frame.reward:+.3f}", font_main))

            obs_text = _normalize_text(frame.observation if isinstance(frame.observation, str) else "")
            lang_text = _normalize_text(frame.language_obs)

            # Hide noisy placeholder text from translator fallbacks.
            if _is_unknown_sailing(obs_text):
                obs_text = ""
            if _is_unknown_sailing(lang_text):
                lang_text = ""

            # If state and language text are the same, keep only one line.
            if obs_text and lang_text and obs_text.casefold() == lang_text.casefold():
                lang_text = ""

            if obs_text:
                lines.append((_clip_for_width(obs_text, "s:", font_meta), font_meta))
            if lang_text:
                for wrapped in _wrap_two_lines(lang_text, "l:", font_meta):
                    lines.append((wrapped, font_meta))

            if frame.sub_goal_similarity is not None or frame.sub_goal_reached:
                sim_text = f"sim={frame.sub_goal_similarity:.3f}" if frame.sub_goal_similarity is not None else ""
                goal_text = "◀ sub-goal!" if frame.sub_goal_reached else ""
                suffix_line = "  ".join(filter(None, [sim_text, goal_text]))
                if suffix_line:
                    lines.append((suffix_line, font_meta))

            # Calculate compact per-line sizes and draw one background box.
            line_heights: list[int] = []
            max_box_width = 0
            for text, font_obj in lines:
                bbox = draw.textbbox((0, 0), text, font=font_obj)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                max_box_width = max(max_box_width, w)
                line_heights.append(max(11, h + 2))

            total_height = sum(line_heights) + 6
            draw.rectangle([2, 2, max_box_width + 8, total_height + 2], fill=(0, 0, 0, 160))

            y = 4
            for idx, (text, font_obj) in enumerate(lines):
                draw.text((4, y), text, fill=(255, 255, 255, 255), font=font_obj)
                y += line_heights[idx]

        images.append(img.convert("P", palette=Image.ADAPTIVE))

    if not images:
        return 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(1000.0 / fps))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    return len(images)


# ── Convenience entry-point ───────────────────────────────────────────────────

def render_optimal_policy(
    result: InteractionResult,
    env_factory: Optional[RLIPEnvironmentFactory] = None,
    env: Optional[_EnvLike] = None,
    *,
    render_mode: str = "rgb_array",
    max_steps: int = 200,
    seed: Optional[int] = None,
    fallback: Any = "random",
    translate: Any = True,
    output_dir: Optional[str | os.PathLike] = None,
    output_gif: Optional[str | os.PathLike] = None,
    output_path_image: Optional[str | os.PathLike] = None,
    output_overlay_image: Optional[str | os.PathLike] = None,
    gif_fps: float = 4.0,
    gif_annotate: bool = True,
    path_image_max_cols: int = 8,
    path_image_thumb_width: int = 120,
) -> PolicyRenderResult:
    """
    Full pipeline: extract optimal policy → replay on a rendered env →
    optionally save PNG frames and/or animated GIF.

    You must supply either *env_factory* (preferred — creates a fresh
    instance with the requested *render_mode*) or *env* (an already-
    instantiated environment; rendering must have been enabled when it
    was created).

    Parameters
    ----------
    result:
        A completed :class:`~rlip.interaction_protocols.InteractionResult`
        with ``record_history=True``.
    env_factory:
        :class:`~rlip.environments.base.RLIPEnvironmentFactory` used to
        create a fresh rendering-enabled environment.
    env:
        Pre-existing environment instance.  Ignored if *env_factory* is
        given.
    render_mode:
        Render mode passed to *env_factory*.  Has no effect when *env* is
        supplied directly.
    max_steps:
        Maximum steps for the replay episode.
    seed:
        Seed for the replay reset.
    fallback:
        Action to use for states not in the policy table (``"random"``,
        ``"zero"``, or a ``Callable``).
    translate:
        Language translator source.  ``True`` (default) auto-resolves from
        the environment ID.
    output_dir:
        If given, individual PNG frames are saved here (one per step).
    output_gif:
        If given, an animated GIF is written to this path.
    output_path_image:
        If given, a single static PNG showing all frames tiled in a grid
        is written to this path.  Much smaller than a GIF and suitable
        for sharing or embedding.
    output_overlay_image:
        If given, all frames are composited into a single PNG by overlaying
        each frame at equal opacity (arithmetic mean), with a subtle
        blue→red tint gradient encoding trajectory direction.  Useful for
        seeing every visited state at once without animation.
    gif_fps:
        Animation speed (frames per second).
    gif_annotate:
        Overlay step / reward / sub-goal annotations on each GIF frame.
    path_image_max_cols:
        Maximum thumbnail columns in the path image grid (default 8).
    path_image_thumb_width:
        Width of each thumbnail in the path image (default 120 px).

    Returns
    -------
    PolicyRenderResult
    """
    # ── 1. Extract optimal policy ──────────────────────────────────────────────
    best_ep, policy = extract_optimal_policy(result)

    # ── 2. Build rendering environment ────────────────────────────────────────
    if env_factory is not None:
        render_env: _EnvLike = env_factory.create(render_mode=render_mode)
    elif env is not None:
        render_env = env
    else:
        raise ValueError("Supply either env_factory or env.")

    # ── 3. Replay with rendering ───────────────────────────────────────────────
    renderer = PolicyRenderer(
        env=render_env,
        policy=policy,
        fallback=fallback,
        translate=translate,
    )
    frames = renderer.run(max_steps=max_steps, seed=seed)

    # ── 4. Save outputs ────────────────────────────────────────────────────────
    n_png_saved = 0
    n_gif_frames = 0

    if output_dir is not None:
        n_png_saved = save_frames_to_dir(frames, output_dir)

    if output_gif is not None:
        n_gif_frames = save_gif(
            frames,
            output_gif,
            fps=gif_fps,
            annotate=gif_annotate,
        )

    if output_path_image is not None:
        save_path_image(
            frames,
            output_path_image,
            max_cols=path_image_max_cols,
            thumb_width=path_image_thumb_width,
        )

    if output_overlay_image is not None:
        save_overlay_image(frames, output_overlay_image)

    env_id = getattr(render_env, "env_id", type(render_env).__name__)

    return PolicyRenderResult(
        protocol_name=result.protocol_name,
        env_id=env_id,
        best_episode_index=best_ep.episode,
        best_episode_reward=best_ep.total_reward,
        best_episode_steps=best_ep.steps,
        frames=frames,
        output_dir=str(output_dir) if output_dir else None,
        output_gif=str(output_gif) if output_gif else None,
        output_path_image=str(output_path_image) if output_path_image else None,
        output_overlay_image=str(output_overlay_image) if output_overlay_image else None,
        n_frames_saved=n_png_saved,
        n_gif_frames=n_gif_frames,
        policy_size=len(policy),
    )


__all__ = [
    "extract_optimal_policy",
    "RenderedFrame",
    "PolicyRenderResult",
    "PolicyRenderer",
    "save_frames_to_dir",
    "save_gif",
    "save_overlay_image",
    "save_path_image",
    "render_optimal_policy",
]
