"""
Instruction Planning Database for rlbridge
=======================================
Persistent per-environment storage of instruction usage and evaluation
outcomes.  Accumulates results across sessions in a JSON file so the LLM
can inspect past runs via ``rl_get_instruction_plan`` and make informed
decisions about which instructions to try next.

Classes
-------
- :class:`InstructionUsageRecord` - one training run's metadata.
- :class:`InstructionPlanEntry`   - all runs for a single instruction string.
- :class:`InstructionPlanDatabase` - the full per-environment database.

Module-level state
------------------
- :data:`_PLAN_DATABASES` - in-memory registry keyed by *env_id*.
- :func:`get_plan_database` - accessor that loads from disk on first call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ── One usage record ──────────────────────────────────────────────────────────

@dataclass
class InstructionUsageRecord:
    """One use of an instruction (one call to rl_match_instruction + training run)."""

    timestamp: str
    """ISO-8601 datetime string when this usage was recorded."""

    match_id: str
    """The match_id generated for this usage."""

    sub_steps: list[str] = field(default_factory=list)
    """Ordered sub-steps the LLM decomposed the instruction into."""

    training_reward: Optional[float] = None
    """Best episode reward achieved during instruction-shaped RL training."""

    eval_reward: Optional[float] = None
    """Total reward from a clean greedy evaluation episode *without* any
    instruction-shaping bonus.  This is the unbiased performance measure."""

    agent_id: Optional[str] = None
    """ID of the trained agent, if one was created."""

    agent_type: Optional[str] = None
    """Agent type used (tabular_q / dqn / ppo)."""

    n_episodes: Optional[int] = None
    """Number of training episodes run."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "match_id": self.match_id,
            "sub_steps": self.sub_steps,
            "training_reward": self.training_reward,
            "eval_reward": self.eval_reward,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "n_episodes": self.n_episodes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstructionUsageRecord":
        return cls(
            timestamp=d.get("timestamp", ""),
            match_id=d.get("match_id", ""),
            sub_steps=d.get("sub_steps") or [],
            training_reward=d.get("training_reward"),
            eval_reward=d.get("eval_reward"),
            agent_id=d.get("agent_id"),
            agent_type=d.get("agent_type"),
            n_episodes=d.get("n_episodes"),
        )


# ── Per-instruction history entry ─────────────────────────────────────────────

@dataclass
class InstructionPlanEntry:
    """All historical data for one instruction string in an environment."""

    instruction: str
    """The instruction text."""

    env_id: str
    """Environment this instruction belongs to."""

    source: str = "user"
    """How this instruction was created: 'user', 'llm', or 'derived'."""

    first_used: str = ""
    """ISO-8601 timestamp of first use."""

    times_used: int = 0
    """Total number of times this instruction has been used."""

    match_similarity: float = 0.0
    """Best cosine similarity achieved when matching this instruction."""

    matched_language: str = ""
    """Language-translated description of the best-matched environment state."""

    sub_steps: list[str] = field(default_factory=list)
    """Most-recent ordered sub-steps derived from this instruction."""

    best_eval_reward: Optional[float] = None
    """Best clean evaluation reward seen across all training runs."""

    derived_score: Optional[float] = None
    """CSR × log(visits) score (only populated for 'derived' instructions)."""

    derived_csr: Optional[float] = None
    """Conditional success rate (only for 'derived' instructions)."""

    usage_history: list[InstructionUsageRecord] = field(default_factory=list)
    """Ordered history of every use, oldest first."""

    def record_use(
        self,
        match_id: str,
        sub_steps: Optional[list[str]] = None,
        similarity: float = 0.0,
        matched_language: str = "",
    ) -> InstructionUsageRecord:
        """Create and append a new usage record; return it for later updating."""
        import datetime as _dt  # noqa: PLC0415
        now = _dt.datetime.now().isoformat(timespec="seconds")
        if not self.first_used:
            self.first_used = now
        self.times_used += 1
        if similarity > self.match_similarity:
            self.match_similarity = similarity
            if matched_language:
                self.matched_language = matched_language
        elif matched_language and not self.matched_language:
            self.matched_language = matched_language
        if sub_steps:
            self.sub_steps = list(sub_steps)
        rec = InstructionUsageRecord(
            timestamp=now,
            match_id=match_id,
            sub_steps=list(sub_steps or self.sub_steps),
        )
        self.usage_history.append(rec)
        return rec

    def update_eval_reward(self, match_id: str, eval_reward: float,
                           training_reward: Optional[float] = None,
                           agent_id: Optional[str] = None,
                           agent_type: Optional[str] = None,
                           n_episodes: Optional[int] = None) -> None:
        """Populate the eval_reward for an existing usage record."""
        for rec in reversed(self.usage_history):
            if rec.match_id == match_id:
                rec.eval_reward = eval_reward
                if training_reward is not None:
                    rec.training_reward = training_reward
                if agent_id is not None:
                    rec.agent_id = agent_id
                if agent_type is not None:
                    rec.agent_type = agent_type
                if n_episodes is not None:
                    rec.n_episodes = n_episodes
                break
        if self.best_eval_reward is None or eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "env_id": self.env_id,
            "source": self.source,
            "first_used": self.first_used,
            "times_used": self.times_used,
            "match_similarity": self.match_similarity,
            "matched_language": self.matched_language,
            "sub_steps": self.sub_steps,
            "best_eval_reward": self.best_eval_reward,
            "derived_score": self.derived_score,
            "derived_csr": self.derived_csr,
            "usage_history": [r.to_dict() for r in self.usage_history],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstructionPlanEntry":
        obj = cls(
            instruction=d.get("instruction", ""),
            env_id=d.get("env_id", ""),
            source=d.get("source", "user"),
            first_used=d.get("first_used", ""),
            times_used=d.get("times_used", 0),
            match_similarity=d.get("match_similarity", 0.0),
            matched_language=d.get("matched_language", ""),
            sub_steps=d.get("sub_steps") or [],
            best_eval_reward=d.get("best_eval_reward"),
            derived_score=d.get("derived_score"),
            derived_csr=d.get("derived_csr"),
        )
        obj.usage_history = [
            InstructionUsageRecord.from_dict(r)
            for r in (d.get("usage_history") or [])
        ]
        return obj


# ── Database ──────────────────────────────────────────────────────────────────

class InstructionPlanDatabase:
    """
    Persistent per-environment database of instruction usage and outcomes.

    Stores every instruction tried in an environment, the sub-steps the LLM
    decomposed it into, and the clean evaluation reward obtained after each
    training run.  Saved as JSON to the rlbridge cache directory so results
    accumulate across sessions.

    The LLM can read this via ``rl_get_instruction_plan`` to plan which
    instructions to try next based on past performance.
    """

    def __init__(self, env_id: str, path: Optional[str] = None) -> None:
        self.env_id = env_id
        self._path: Optional[str] = path
        self._entries: dict[str, InstructionPlanEntry] = {}
        # Ordered list of instruction texts in the sequence they were first used.
        self._use_order: list[str] = []
        if path:
            self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        import json as _json  # noqa: PLC0415
        import os as _os  # noqa: PLC0415
        if not self._path or not _os.path.exists(self._path):
            return
        try:
            raw = _json.loads(open(self._path, encoding="utf-8").read())
            for d in raw.get("entries", []):
                e = InstructionPlanEntry.from_dict(d)
                self._entries[e.instruction] = e
            self._use_order = raw.get("use_order", list(self._entries.keys()))
        except Exception:  # noqa: BLE001
            pass

    def save(self) -> None:
        """Write the database to disk.  Silently no-ops if no path is set."""
        if not self._path:
            return
        import json as _json  # noqa: PLC0415
        import os as _os  # noqa: PLC0415
        _os.makedirs(_os.path.dirname(self._path), exist_ok=True)
        data = {
            "env_id": self.env_id,
            "use_order": self._use_order,
            "entries": [e.to_dict() for e in self._entries.values()],
        }
        with open(self._path, "w", encoding="utf-8") as fh:
            _json.dump(data, fh, indent=2)

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def record_instruction_use(
        self,
        instruction: str,
        match_id: str,
        source: str = "llm",
        sub_steps: Optional[list[str]] = None,
        similarity: float = 0.0,
        matched_language: str = "",
    ) -> InstructionUsageRecord:
        """Log a new use of *instruction* and return the usage record."""
        if instruction not in self._entries:
            self._entries[instruction] = InstructionPlanEntry(
                instruction=instruction,
                env_id=self.env_id,
                source=source,
            )
            self._use_order.append(instruction)
        entry = self._entries[instruction]
        entry.source = source  # update in case it was derived first, now used by LLM
        rec = entry.record_use(
            match_id,
            sub_steps=sub_steps,
            similarity=similarity,
            matched_language=matched_language,
        )
        self.save()
        return rec

    def update_eval_reward(
        self,
        instruction: str,
        match_id: str,
        eval_reward: float,
        training_reward: Optional[float] = None,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        n_episodes: Optional[int] = None,
    ) -> None:
        """Update the eval reward for a specific usage of *instruction*."""
        if instruction not in self._entries:
            return
        self._entries[instruction].update_eval_reward(
            match_id, eval_reward,
            training_reward=training_reward,
            agent_id=agent_id, agent_type=agent_type, n_episodes=n_episodes,
        )
        self.save()

    def add_derived(
        self,
        instruction: str,
        derived_score: float,
        derived_csr: float,
        similarity: float = 1.0,
    ) -> InstructionPlanEntry:
        """Register a derived instruction (from rl_train_and_derive_instructions)."""
        if instruction not in self._entries:
            self._entries[instruction] = InstructionPlanEntry(
                instruction=instruction,
                env_id=self.env_id,
                source="derived",
            )
            self._use_order.append(instruction)
        entry = self._entries[instruction]
        entry.derived_score = derived_score
        entry.derived_csr = derived_csr
        if similarity > entry.match_similarity:
            entry.match_similarity = similarity
        # Preserve source as "derived" only if never actually used by LLM
        if entry.source not in ("user", "llm"):
            entry.source = "derived"
        self.save()
        return entry

    # ── Read helpers ──────────────────────────────────────────────────────────

    def get_entry(self, instruction: str) -> Optional[InstructionPlanEntry]:
        return self._entries.get(instruction)

    def all_entries_ordered(self) -> list[InstructionPlanEntry]:
        """Return entries in the order they were first used."""
        seen: set[str] = set()
        result: list[InstructionPlanEntry] = []
        for instr in self._use_order:
            if instr in self._entries and instr not in seen:
                result.append(self._entries[instr])
                seen.add(instr)
        # Catch anything not in use_order (shouldn't happen, but be safe).
        for e in self._entries.values():
            if e.instruction not in seen:
                result.append(e)
        return result

    def best_instructions(self, n: int = 5) -> list[InstructionPlanEntry]:
        """Top-n entries by best_eval_reward (entries without eval come last)."""
        entries = list(self._entries.values())
        entries.sort(
            key=lambda e: (e.best_eval_reward is not None, e.best_eval_reward or 0.0),
            reverse=True,
        )
        return entries[:n]

    def summary_text(self, max_entries: int = 30) -> str:
        """Human-readable table of all instructions with their outcomes."""
        entries = self.all_entries_ordered()[:max_entries]
        if not entries:
            return f"No instructions recorded yet for '{self.env_id}'."

        lines = [
            f"Instruction Planning Database - {self.env_id}",
            f"  {len(self._entries)} instruction(s) recorded\n",
            f"  {'#':<3}  {'Source':<8}  {'Used':<5}  {'BestEval':>9}  "
            f"{'DrvScore':>9}  {'Sim%':>6}  Instruction  →  Matched State",
            "  " + "-" * 120,
        ]
        for i, e in enumerate(entries, start=1):
            eval_str  = f"{e.best_eval_reward:+.4f}" if e.best_eval_reward is not None else "      —"
            drv_str   = f"{e.derived_score:.4f}" if e.derived_score is not None else "       —"
            sim_pct   = f"{e.match_similarity * 100:.1f}%" if e.match_similarity > 0 else "   —"
            instr_short = e.instruction[:55] + ("…" if len(e.instruction) > 55 else "")
            lang_short  = (e.matched_language[:55] + "…" if len(e.matched_language) > 55
                           else e.matched_language) if e.matched_language else "—"
            lines.append(
                f"  {i:<3}  {e.source:<8}  {e.times_used:<5}  "
                f"{eval_str:>9}  {drv_str:>9}  {sim_pct:>6}  {instr_short}  →  {lang_short}"
            )

        # Best performers
        best = self.best_instructions(3)
        with_eval = [e for e in best if e.best_eval_reward is not None]
        if with_eval:
            lines.append("\nTop instructions by clean evaluation reward:")
            for e in with_eval:
                lang_note = f"  matched: '{e.matched_language[:60]}'" if e.matched_language else ""
                drv_note  = f"  derived_score={e.derived_score:.4f}  csr={e.derived_csr:.1%}" if e.derived_score else ""
                lines.append(
                    f"  • [{e.source}] {e.instruction[:80]}"
                    f"  → eval={e.best_eval_reward:+.4f}  sim={e.match_similarity*100:.1f}%"
                    + lang_note + drv_note
                )

        return "\n".join(lines)


# ── Global in-memory registry ─────────────────────────────────────────────────

_PLAN_DATABASES: dict[str, InstructionPlanDatabase] = {}


def get_plan_database(env_id: str, plan_path: Optional[str] = None) -> InstructionPlanDatabase:
    """
    Return the in-memory :class:`InstructionPlanDatabase` for *env_id*,
    loading from *plan_path* on first access.

    Parameters
    ----------
    env_id:
        Registered environment ID.
    plan_path:
        Filesystem path to the JSON file.  Only used on first load; later
        calls reuse the cached in-memory instance.
    """
    if env_id not in _PLAN_DATABASES:
        _PLAN_DATABASES[env_id] = InstructionPlanDatabase(env_id, path=plan_path)
    return _PLAN_DATABASES[env_id]


__all__ = [
    "InstructionUsageRecord",
    "InstructionPlanEntry",
    "InstructionPlanDatabase",
    "_PLAN_DATABASES",
    "get_plan_database",
]
