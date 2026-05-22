#!/usr/bin/env python3
"""
Standalone instruction matching script.

Scores a natural-language instruction against a list of observed language
descriptions using TF-IDF cosine similarity, optionally adjusted by validated
feedback stored for an environment.

Examples
--------
Match against inline candidate strings::

    python scripts/match_instruction.py \\
        --instruction "sail towards the beach" \\
        --candidates "near the shore" "in open water" "close to harbor"

Record user validation and re-run (feedback boosts/discourages future scores)::

    python scripts/match_instruction.py \\
        --env-id Sailing-v0 \\
        --instruction "sail towards the beach" \\
        --candidates "near the shore" "in open water" \\
        --validate-correct "near the shore"

Reject a bad match::

    python scripts/match_instruction.py \\
        --env-id Sailing-v0 \\
        --instruction "sail towards the beach" \\
        --candidates "near the shore" "in open water" \\
        --validate-incorrect "in open water"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from rlip.instruction_matching import (  # noqa: E402
    TFIDFEncoder,
    get_feedback_layer,
    record_match_feedback,
    score_instruction_against_corpus,
)


def _load_candidates(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "candidates" in data:
        return [str(x) for x in data["candidates"]]
    raise ValueError(f"Unsupported candidates file format: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Match an instruction to language candidates.")
    parser.add_argument("--instruction", required=True, help="Natural-language goal text.")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=[],
        help="Observed language descriptions to rank.",
    )
    parser.add_argument(
        "--candidates-file",
        default="",
        help="JSON file with a list of candidate strings (or {\"candidates\": [...]}).",
    )
    parser.add_argument(
        "--env-id",
        default="",
        help="Environment ID for loading/saving validated feedback.",
    )
    parser.add_argument(
        "--similarity-band",
        type=float,
        default=0.05,
        help="Band around the best score for co-equal sub-goals.",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Disable feedback-layer score adjustments.",
    )
    parser.add_argument(
        "--validate-correct",
        default="",
        help="Record that this state_language is a correct match for the instruction.",
    )
    parser.add_argument(
        "--validate-incorrect",
        default="",
        help="Record that this state_language is an incorrect match for the instruction.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked candidates to print.",
    )
    args = parser.parse_args(argv)

    candidates = list(args.candidates)
    if args.candidates_file:
        candidates.extend(_load_candidates(args.candidates_file))
    if not candidates:
        parser.error("Provide --candidates and/or --candidates-file.")

    env_id = args.env_id.strip()
    if args.validate_correct or args.validate_incorrect:
        if not env_id:
            parser.error("--env-id is required when recording validation feedback.")

    if args.validate_correct:
        record_match_feedback(
            env_id,
            args.instruction,
            args.validate_correct,
            correct=True,
            source="script",
        )
        print(f"Recorded CORRECT feedback: {args.validate_correct!r}", file=sys.stderr)

    if args.validate_incorrect:
        record_match_feedback(
            env_id,
            args.instruction,
            args.validate_incorrect,
            correct=False,
            source="script",
        )
        print(f"Recorded INCORRECT feedback: {args.validate_incorrect!r}", file=sys.stderr)

    pairs = [(lang, None) for lang in candidates]
    feedback = None if args.no_feedback or not env_id else get_feedback_layer(env_id)

    result = score_instruction_against_corpus(
        args.instruction,
        pairs,
        encoder=TFIDFEncoder(),
        feedback_layer=feedback,
        similarity_band=args.similarity_band,
    )

    top_k = max(1, min(args.top_k, len(result.candidates)))
    print(f"Instruction: {result.instruction!r}")
    print(f"Best match:  {result.best_language!r}")
    print(f"Adjusted:    {result.similarity_score:.4f}")
    print(f"Base TF-IDF: {result.base_similarity_score:.4f}")
    print(f"Sub-goals:   {len(result.matched_states)}")
    print()
    print(f"Top {top_k} candidates:")
    for i, cand in enumerate(result.candidates[:top_k], start=1):
        print(
            f"  {i}. adj={cand.adjusted_score:.4f}  base={cand.base_score:.4f}  "
            f"{cand.language[:120]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
