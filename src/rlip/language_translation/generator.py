"""
Language Translation Generator
================================
Automatically creates :class:`~base.LanguageTranslator` implementations for
new environments using an LLM in a two-stage pipeline.

Stage 1 — Description
    Sample *n* observed states from the environment and ask the LLM to write
    a natural-language description for each one, given a brief environment
    context string supplied by the caller.

Stage 2 — Rule synthesis
    Show the (state, description) pairs back to the LLM and ask it to write
    a deterministic Python ``translate`` function that encodes the observed
    patterns as explicit conditional logic.

The resulting :class:`GeneratedTranslator` wraps the synthesised rule function
and falls back to a live LLM call for any state it cannot handle, caching
answers and periodically re-running rule synthesis (refinement) when the cache
grows large enough.  Refined rules are tested against all cached observations
so that states the new rules can handle are removed from the live cache.

Quick start
-----------
::

    from rlip.environments.registry import registry
    from rlip.language_translation.generator import TranslatorGenerator

    env = registry.get("MyEnv-v0").create()

    def my_llm(prompt: str) -> str:
        # Wrap any LLM provider here — OpenAI, Anthropic, local Ollama, …
        import openai
        return openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

    gen = TranslatorGenerator(
        env=env,
        llm_fn=my_llm,
        env_context="A 10×10 grid where the agent (A) collects coins (C).",
        n_samples=40,
    )
    translator = gen.build()
    print(translator.translate(some_state))

    # Save a standalone Python module for later import:
    translator.save_code("my_env_translator.py")

⚠️  Security notice
    :class:`TranslatorGenerator` and :class:`GeneratedTranslator` execute
    LLM-generated Python code using ``exec``.  Only use this with trusted LLM
    providers and review generated code before deploying to production systems.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .base import LanguageTranslator

#: Type alias for any LLM callable: accepts a prompt and returns its completion.
LLMCallable = Callable[[str], str]


# ── LLM prompt templates ────────────────────────────────────────────────────

_DESCRIBE_PROMPT = """\
You are helping build a language translation module for a reinforcement \
learning environment.

Environment context:
{env_context}

Below are {n_samples} observed states sampled from the environment, each \
shown with its index, raw value, and the most recent action taken (if any).

{states_block}

For each state write a clear, natural-language description that a human would \
understand.  Include:
  - The agent/object position or configuration
  - Any key features visible in the raw state value
  - What action was most recently taken (when action_history is provided)

Reply in this exact format — one line per state, no preamble, no commentary:
STATE 1: <description>
STATE 2: <description>
…
"""

_RULES_PROMPT = """\
You are writing the body of a Python language-translation function for a \
reinforcement learning environment.

Environment context:
{env_context}

Here are example (state, description) pairs that the function must reproduce:

{pair_block}

Write a complete Python function with exactly this signature:

    def translate(state, *, legal_moves=None, action_history=None):
        ...
        return description_string

Requirements:
  - Parse the raw state value to extract features (position, angle, flags, …).
  - Build a description that matches the vocabulary and style of the examples.
  - If the state format cannot be parsed or a feature is out of range, \
return an empty string "" (the caller handles the fallback).
  - Do NOT import any external modules; use only Python built-ins.
  - Stay within the format of the sample descriptions unless generalising \
clearly improves coverage.

Output ONLY the function code block.  No explanation, no markdown fences.
"""

_REFINE_PROMPT = """\
The translate function below was used in production but fell back to the LLM \
for the states listed below.  These states were NOT handled by the current \
rules.

Current function:
{current_code}

New (state → description) examples to incorporate:
{pair_block}

Rewrite the translate function so that it handles all new examples correctly \
while preserving the existing logic for states that already work.

Output ONLY the updated function code block.  No explanation, no markdown \
fences.
"""

_DESCRIBE_SINGLE_PROMPT = """\
Describe the following reinforcement learning environment state in one clear \
natural-language sentence.

Environment context:
{env_context}

Raw state: {state}
Most recent action history: {action_history}

Reply with a single descriptive sentence only.
"""

_SAVE_TEMPLATE = '''\
"""
{class_name} — auto-generated language translator for {env_id}.

Generated by rlip.language_translation.generator.TranslatorGenerator.
Environment context: {env_context}

⚠️  This module contains auto-generated code.  Review before production use.
"""

from __future__ import annotations
from typing import Any
from rlip.language_translation.base import LanguageTranslator


# ── Generated rule function ────────────────────────────────────────────────


{rule_fn_code}


# ── Translator class ───────────────────────────────────────────────────────


class {class_name}(LanguageTranslator):
    """Auto-generated translator for ``{env_id}``."""

    name = "generated:{env_id}"

    def translate(
        self,
        state: Any,
        *,
        legal_moves: list[Any] | None = None,
        action_history: list[Any] | None = None,
    ) -> str:
        try:
            result = _translate_rule(
                state,
                legal_moves=legal_moves,
                action_history=action_history,
            )
            if result:
                return result
        except Exception:
            pass
        return ""
'''


# ── Internal helpers ────────────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """Remove leading ```python / trailing ``` fences often added by LLMs."""
    lines = text.strip().splitlines()
    if lines and re.match(r"^```", lines[0]):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _compile_rule_fn(code: str) -> Optional[Callable]:
    """
    Compile *code* and return the ``translate`` function it defines.

    Returns *None* if compilation fails or no ``translate`` callable is found.
    """
    try:
        ns: dict[str, Any] = {}
        exec(compile(code, "<generated_translator>", "exec"), ns)  # noqa: S102
        fn = ns.get("translate")
        return fn if callable(fn) else None
    except Exception:
        return None


def _parse_descriptions(response: str, n: int) -> list[str]:
    """
    Parse ``STATE N: <description>`` lines from an LLM response.

    Missing entries are filled with a placeholder so the returned list always
    has exactly *n* elements.
    """
    descriptions: list[str] = [""] * n
    pattern = re.compile(r"STATE\s+(\d+)\s*[:\-]\s*(.+)", re.IGNORECASE)
    for m in pattern.finditer(response):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            descriptions[idx] = m.group(2).strip()
    for i, desc in enumerate(descriptions):
        if not desc:
            descriptions[i] = f"[State {i + 1}: description unavailable]"
    return descriptions


def _to_class_name(env_id: str) -> str:
    """Convert e.g. ``"GridWorld-1D-v0"`` → ``"GridWorld1DV0LanguageTranslator"``."""
    parts = re.split(r"[-_\s/]+", env_id)
    return "".join(p.capitalize() for p in parts) + "LanguageTranslator"


def _to_rule_fn_name(env_id: str) -> str:
    """Return a safe module-level function name for the rule function."""
    safe = re.sub(r"[^a-zA-Z0-9]", "_", env_id).lower()
    return f"_translate_rule"


def _get_obs(step_or_reset_out: Any) -> Any:
    """Extract observation from step/reset output."""
    if isinstance(step_or_reset_out, dict):
        return step_or_reset_out.get("observation", step_or_reset_out)
    if hasattr(step_or_reset_out, "observation"):
        return step_or_reset_out.observation
    return step_or_reset_out


def _get_flag(step_out: Any, name: str) -> bool:
    if isinstance(step_out, dict):
        return bool(step_out.get(name, False))
    return bool(getattr(step_out, name, False))


def _sample_action(env: Any, rng: random.Random) -> Any:
    """Sample a random action from the environment's action space."""
    # RLIP environments expose sample_action() directly — prefer that so
    # text-action envs (chess, sailing, textworld …) return valid moves.
    if hasattr(env, "sample_action"):
        try:
            return env.sample_action()
        except Exception:
            pass
    action_space = getattr(env, "action_space", None)
    if action_space is not None:
        n = getattr(action_space, "n", None)
        if n is not None:
            return rng.randint(0, int(n) - 1)
        if hasattr(action_space, "sample"):
            try:
                return action_space.sample()
            except Exception:
                pass
    return 0


def _sample_states(
    env: Any,
    n_samples: int = 30,
    max_steps_per_episode: int = 50,
    max_episodes: int = 0,
    seed: Optional[int] = None,
) -> list[tuple[Any, list[Any]]]:
    """
    Collect *n_samples* unique observed states from *env* via random exploration.

    Returns a list of ``(observation, action_history)`` pairs where
    *action_history* contains all actions taken up to (and including) the step
    that produced this observation.

    Parameters
    ----------
    max_steps_per_episode:
        Hard step cap per episode (default 50 — enough to explore without
        running a full game).
    max_episodes:
        Maximum number of episodes to run.  0 (default) → ``n_samples * 10``.
        Prevents infinite loops on high-variance or long-horizon environments.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    samples: list[tuple[Any, list[Any]]] = []

    if max_episodes <= 0:
        max_episodes = n_samples * 10

    episode_seed = seed
    episodes_run = 0

    while len(samples) < n_samples and episodes_run < max_episodes:
        reset_out = env.reset(seed=episode_seed)
        obs = _get_obs(reset_out)
        action_history: list[Any] = []
        episode_seed = (episode_seed + 1) if episode_seed is not None else None
        episodes_run += 1

        for _ in range(max_steps_per_episode):
            key = repr(obs)
            if key not in seen:
                seen.add(key)
                samples.append((obs, action_history.copy()))
                if len(samples) >= n_samples:
                    return samples

            action = _sample_action(env, rng)
            action_history.append(action)
            step_out = env.step(action)
            obs = _get_obs(step_out)

            if _get_flag(step_out, "terminated") or _get_flag(step_out, "truncated"):
                break

    return samples


# ── Cached LLM entry ────────────────────────────────────────────────────────

@dataclass
class _CachedEntry:
    """One LLM-answered fallback entry."""
    state: Any
    description: str


# ── GeneratedTranslator ────────────────────────────────────────────────────

class GeneratedTranslator(LanguageTranslator):
    """
    Runtime language translator produced by :class:`TranslatorGenerator`.

    Wraps a synthesised Python rule function with three layers of behaviour:

    1. **Rule function** — deterministic Python code generated by an LLM.
       Fast, reproducible, no network traffic.
    2. **LLM fallback** — for states the rule function cannot handle (returns
       ``""`` or raises), a live LLM call is made.
    3. **Auto-refinement** — once the number of LLM-answered cache entries
       reaches *refine_threshold*, :meth:`refine` is called automatically.
       The rule function is rewritten to incorporate cached examples, and
       entries that the new rules handle are removed from the cache.

    Parameters
    ----------
    llm_fn:
        Callable that accepts a prompt string and returns an LLM completion.
        Any provider can be wrapped: ``lambda p: openai_client.chat(p)``.
    env_id:
        Environment identifier, used for logging and saved-file naming.
    rule_code:
        Python source code of the initial rule ``translate`` function.
        May be empty; if so the translator operates in pure-LLM mode until
        :meth:`refine` is called or rules are injected via :attr:`rule_code`.
    env_context:
        Short description of the environment passed to all prompts.
    refine_threshold:
        Number of unique LLM-answered states that triggers automatic
        refinement.  After refinement the threshold is bumped by the same
        amount to avoid immediate re-triggering.
    """

    def __init__(
        self,
        llm_fn: LLMCallable,
        env_id: str,
        rule_code: str = "",
        env_context: str = "",
        refine_threshold: int = 20,
        base_translator: Optional["LanguageTranslator"] = None,
    ) -> None:
        self._llm_fn = llm_fn
        self._env_id = env_id
        self._env_context = env_context
        self._base_refine_threshold = refine_threshold
        self._refine_threshold = refine_threshold
        self.name = f"generated:{env_id}"
        #: Registered translator used as the first fallback before rule_fn/LLM.
        self._base_translator = base_translator

        # Rule function
        self._rule_code: str = ""
        self._rule_fn: Optional[Callable] = None
        if rule_code:
            self.rule_code = rule_code  # uses the property setter

        # LLM fallback cache: repr(state) → _CachedEntry
        self._llm_cache: dict[str, _CachedEntry] = {}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def rule_code(self) -> str:
        """The Python source of the current rule function."""
        return self._rule_code

    @rule_code.setter
    def rule_code(self, code: str) -> None:
        """Set new rule code and immediately compile it."""
        cleaned = _strip_markdown_fences(code)
        fn = _compile_rule_fn(cleaned)
        if fn is None:
            raise ValueError(
                "Could not compile the provided rule code.  "
                "Ensure it defines a callable named 'translate'."
            )
        self._rule_code = cleaned
        self._rule_fn = fn

    # ── Core translate ────────────────────────────────────────────────────────

    def translate(
        self,
        state: Any,
        *,
        legal_moves: list[Any] | None = None,
        action_history: list[Any] | None = None,
    ) -> str:
        """
        Translate *state* to a natural-language description.

        Attempts the rule function first; falls back to an LLM call on failure,
        caching the result.  Triggers :meth:`refine` automatically when the
        cache reaches the configured threshold.
        """
        # 0. Registered base translator (e.g. SailingLanguageTranslator)
        if self._base_translator is not None:
            try:
                result = self._base_translator.translate(
                    state,
                    legal_moves=legal_moves,
                    action_history=action_history,
                )
                if result:
                    return result
            except Exception:
                pass

        # 1. Rule function (LLM-generated)
        if self._rule_fn is not None:
            try:
                result = self._rule_fn(
                    state,
                    legal_moves=legal_moves,
                    action_history=action_history,
                )
                if result:
                    return result
            except Exception:
                pass

        # 2. LLM fallback (with cache)
        key = repr(state)
        if key in self._llm_cache:
            return self._llm_cache[key].description

        description = self._llm_describe(state, action_history=action_history)
        self._llm_cache[key] = _CachedEntry(state=state, description=description)

        # 3. Auto-refinement check
        if len(self._llm_cache) >= self._refine_threshold:
            self.refine()

        return description

    # ── Refinement ────────────────────────────────────────────────────────────

    def refine(self) -> bool:
        """
        Re-run rule synthesis using cached LLM-answered states as training data.

        The LLM is asked to rewrite (or write from scratch, if no rules exist
        yet) the rule function so it handles all cached examples.  After a
        successful rewrite, entries that the updated rules can now handle are
        removed from the cache.

        Returns
        -------
        bool
            *True* if the rule function was successfully updated.
        """
        if not self._llm_cache:
            return False

        pair_lines: list[str] = []
        for entry in list(self._llm_cache.values())[:50]:
            pair_lines.append(f"state = {repr(entry.state)}")
            pair_lines.append(f"description = {entry.description!r}")
            pair_lines.append("")
        pair_block = "\n".join(pair_lines).strip()

        if self._rule_code:
            prompt = _REFINE_PROMPT.format(
                current_code=self._rule_code,
                pair_block=pair_block,
            )
        else:
            prompt = _RULES_PROMPT.format(
                env_context=self._env_context or "Not provided.",
                pair_block=pair_block,
            )

        try:
            new_code_raw = self._llm_fn(prompt)
            new_code = _strip_markdown_fences(new_code_raw)
            new_fn = _compile_rule_fn(new_code)
            if new_fn is None:
                return False

            self._rule_code = new_code
            self._rule_fn = new_fn

            # Remove cache entries that the updated rules can now handle.
            self._prune_cache()
            # Bump threshold so refinement doesn't immediately re-trigger.
            self._refine_threshold = len(self._llm_cache) + self._base_refine_threshold
            return True

        except Exception:
            return False

    def _prune_cache(self) -> int:
        """
        Remove entries from the LLM cache that the current rule function
        now handles correctly.

        Returns
        -------
        int
            Number of entries pruned.
        """
        if self._rule_fn is None:
            return 0

        pruned = 0
        for key, entry in list(self._llm_cache.items()):
            try:
                result = self._rule_fn(entry.state)
                if result:
                    del self._llm_cache[key]
                    pruned += 1
            except Exception:
                pass
        return pruned

    # ── LLM description ───────────────────────────────────────────────────────

    def _llm_describe(
        self,
        state: Any,
        *,
        action_history: list[Any] | None = None,
    ) -> str:
        """Make a single-state LLM description call."""
        prompt = _DESCRIBE_SINGLE_PROMPT.format(
            env_context=self._env_context or "Not provided.",
            state=repr(state),
            action_history=repr(action_history) if action_history else "None",
        )
        try:
            return self._llm_fn(prompt).strip()
        except Exception as exc:
            return f"[LLM description unavailable: {exc}]"

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_code(
        self,
        path: str | Path,
        class_name: Optional[str] = None,
    ) -> Path:
        """
        Write the generated translator as a standalone Python module.

        The saved file defines a ``LanguageTranslator`` subclass that wraps
        the current rule function and returns ``""`` on failure (no LLM
        dependency at runtime).  Add an LLM fallback by subclassing or
        wrapping with :class:`GeneratedTranslator` again.

        Parameters
        ----------
        path:
            Destination ``.py`` file path.  Parent directories are created
            if they do not exist.
        class_name:
            Class name to use in the generated file.  Defaults to
            ``"{EnvId}LanguageTranslator"`` derived from ``env_id``.

        Returns
        -------
        Path
            The resolved path of the written file.
        """
        out = Path(path)
        if class_name is None:
            class_name = _to_class_name(self._env_id)

        rule_fn_code = self._rule_code
        if not rule_fn_code:
            rule_fn_code = (
                "def _translate_rule(state, *, legal_moves=None, action_history=None):\n"
                "    # No rules generated yet — all calls fall back to LLM.\n"
                "    return \"\"\n"
            )
        else:
            # Rename 'translate' → '_translate_rule' for the module level.
            rule_fn_code = re.sub(
                r"^def translate\s*\(",
                "def _translate_rule(",
                rule_fn_code,
                count=1,
                flags=re.MULTILINE,
            )

        content = _SAVE_TEMPLATE.format(
            class_name=class_name,
            env_id=self._env_id,
            env_context=(self._env_context or "").replace("\n", " "),
            rule_fn_code=rule_fn_code,
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        return out

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def cache_size(self) -> int:
        """Number of states currently answered via LLM fallback."""
        return len(self._llm_cache)

    @property
    def has_rules(self) -> bool:
        """Whether a compiled rule function is currently active."""
        return self._rule_fn is not None

    def __repr__(self) -> str:
        return (
            f"GeneratedTranslator(env_id={self._env_id!r}, "
            f"has_rules={self.has_rules}, "
            f"cache_size={self.cache_size}, "
            f"refine_threshold={self._refine_threshold})"
        )


# ── TranslatorGenerator ────────────────────────────────────────────────────

class TranslatorGenerator:
    """
    Orchestrates the two-stage LLM pipeline (describe → synthesise rules) and
    returns a ready-to-use :class:`GeneratedTranslator`.

    Parameters
    ----------
    env:
        An RLIP environment instance used for state sampling.  It will be
        ``reset()`` and ``step()``-ped to collect observations; it is not
        closed by this class.
    llm_fn:
        Any callable ``(prompt: str) -> str`` wrapping your LLM provider.
    env_context:
        Short English description of the environment given to the LLM so it
        can write meaningful descriptions and rules.  E.g.::

            "A 10×10 grid.  The agent (A) collects coins (C).  "
            "The state is a string 'row_col_coins_remaining'."
    n_samples:
        Number of unique observations to sample for the description stage.
        More samples → richer vocabulary for rule synthesis.  Default 30.
    seed:
        Random seed for reproducible state sampling.
    refine_threshold:
        Passed to :class:`GeneratedTranslator` — number of LLM cache entries
        that trigger automatic refinement.
    """

    def __init__(
        self,
        env: Any,
        llm_fn: LLMCallable,
        env_context: str = "",
        n_samples: int = 30,
        seed: Optional[int] = None,
        refine_threshold: int = 20,
    ) -> None:
        self._env = env
        self._llm_fn = llm_fn
        self._env_context = env_context
        self._n_samples = n_samples
        self._seed = seed
        self._refine_threshold = refine_threshold
        self._env_id: str = getattr(env, "env_id", type(env).__name__)

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> GeneratedTranslator:
        """
        Run the full pipeline and return a :class:`GeneratedTranslator`.

        Steps:

        1. Sample *n_samples* unique observed states from the environment.
        2. Ask the LLM to describe all sampled states (one LLM call).
        3. Ask the LLM to synthesise a rule function from the pairs (one LLM
           call).
        4. Pre-populate the cache with descriptions for any states the initial
           rules cannot yet handle.

        Returns
        -------
        GeneratedTranslator
            Ready to use with :attr:`~GeneratedTranslator.has_rules` ``= True``
            when rule synthesis succeeded.
        """
        samples = self._sample_states()
        descriptions = self._describe_states(samples)
        rule_code = self._synthesise_rules(samples, descriptions)

        translator = GeneratedTranslator(
            llm_fn=self._llm_fn,
            env_id=self._env_id,
            env_context=self._env_context,
            refine_threshold=self._refine_threshold,
        )

        # Set rule code (compiles it internally).
        if rule_code:
            try:
                translator.rule_code = rule_code
            except ValueError:
                pass  # bad code — translator starts in LLM-only mode

        # Pre-populate LLM cache with stage-1 descriptions for states that the
        # rule function still cannot express, so those descriptions are not lost.
        for (state, _ah), desc in zip(samples, descriptions):
            key = repr(state)
            if key in translator._llm_cache:
                continue
            # Check if rule can handle this state
            handled = False
            if translator._rule_fn is not None:
                try:
                    handled = bool(translator._rule_fn(state))
                except Exception:
                    pass
            if not handled:
                translator._llm_cache[key] = _CachedEntry(
                    state=state, description=desc
                )

        return translator

    def sample_states(
        self, n: Optional[int] = None
    ) -> list[tuple[Any, list[Any]]]:
        """
        Sample observed states from the environment.

        Parameters
        ----------
        n:
            Number of unique states to sample.  Defaults to *n_samples*
            passed at construction time.

        Returns
        -------
        list of ``(observation, action_history)`` tuples.
        """
        return self._sample_states(n or self._n_samples)

    def describe_states(
        self, samples: list[tuple[Any, list[Any]]]
    ) -> list[str]:
        """
        Ask the LLM to describe a list of (obs, action_history) samples.

        Parameters
        ----------
        samples:
            List of ``(observation, action_history)`` pairs — the same format
            returned by :meth:`sample_states`.

        Returns
        -------
        list[str]
            One description per sample, in the same order.
        """
        return self._describe_states(samples)

    def synthesise_rules(
        self,
        samples: list[tuple[Any, list[Any]]],
        descriptions: list[str],
    ) -> str:
        """
        Ask the LLM to write a rule function from (state, description) pairs.

        Parameters
        ----------
        samples:
            List of ``(observation, action_history)`` pairs.
        descriptions:
            Parallel list of language descriptions.

        Returns
        -------
        str
            Raw Python source code of the generated ``translate`` function.
        """
        return self._synthesise_rules(samples, descriptions)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sample_states(
        self, n: Optional[int] = None
    ) -> list[tuple[Any, list[Any]]]:
        return _sample_states(
            self._env,
            n_samples=n or self._n_samples,
            seed=self._seed,
        )

    def _describe_states(
        self, samples: list[tuple[Any, list[Any]]]
    ) -> list[str]:
        """One LLM call to produce descriptions for all sampled states."""
        state_lines: list[str] = []
        for i, (obs, ah) in enumerate(samples, 1):
            line = f"STATE {i}: {repr(obs)}"
            if ah:
                line += f"  (last action: {ah[-1]})"
            state_lines.append(line)

        prompt = _DESCRIBE_PROMPT.format(
            env_context=self._env_context or "Not provided.",
            n_samples=len(samples),
            states_block="\n".join(state_lines),
        )
        response = self._llm_fn(prompt)
        return _parse_descriptions(response, len(samples))

    def _synthesise_rules(
        self,
        samples: list[tuple[Any, list[Any]]],
        descriptions: list[str],
    ) -> str:
        """One LLM call to produce the rule translate function."""
        pair_lines: list[str] = []
        for (obs, ah), desc in zip(samples, descriptions):
            pair_lines.append(f"# state = {repr(obs)}")
            if ah:
                pair_lines.append(f"# action_history[-1] = {repr(ah[-1])}")
            pair_lines.append(f"# description → {desc!r}")
            pair_lines.append("")

        prompt = _RULES_PROMPT.format(
            env_context=self._env_context or "Not provided.",
            pair_block="\n".join(pair_lines).strip(),
        )
        raw = self._llm_fn(prompt)
        return _strip_markdown_fences(raw)


# ── Convenience wrapper ────────────────────────────────────────────────────

def build_translator(
    env: Any,
    llm_fn: LLMCallable,
    *,
    env_context: str = "",
    n_samples: int = 30,
    seed: Optional[int] = None,
    refine_threshold: int = 20,
) -> GeneratedTranslator:
    """
    One-shot convenience function: sample states, describe, synthesise rules,
    and return a :class:`GeneratedTranslator`.

    Equivalent to::

        TranslatorGenerator(env, llm_fn, ...).build()

    Parameters
    ----------
    env:
        RLIP environment to sample from.
    llm_fn:
        LLM callable ``(prompt: str) -> str``.
    env_context:
        Short description of the environment for the LLM.
    n_samples:
        Number of unique states to sample.
    seed:
        Random seed for reproducible sampling.
    refine_threshold:
        LLM cache entries before automatic refinement fires.

    Returns
    -------
    GeneratedTranslator
    """
    return TranslatorGenerator(
        env=env,
        llm_fn=llm_fn,
        env_context=env_context,
        n_samples=n_samples,
        seed=seed,
        refine_threshold=refine_threshold,
    ).build()


__all__ = [
    "LLMCallable",
    "GeneratedTranslator",
    "TranslatorGenerator",
    "build_translator",
]
