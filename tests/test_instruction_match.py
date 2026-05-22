"""Tests for rlip.instruction_match programmatic API."""

from __future__ import annotations

import json

import pytest

from rlip.instruction_match import (
    format_match_result,
    load_candidates,
    match_candidates,
    record_validation,
    resolve_candidates,
)
from rlip.instruction_matching.tfidf import TFIDFEncoder


def test_load_candidates_list(tmp_path):
    path = tmp_path / "cands.json"
    path.write_text(json.dumps(["near shore", "open water"]), encoding="utf-8")
    assert load_candidates(path) == ["near shore", "open water"]


def test_load_candidates_dict(tmp_path):
    path = tmp_path / "cands.json"
    path.write_text(json.dumps({"candidates": ["a", "b"]}), encoding="utf-8")
    assert load_candidates(path) == ["a", "b"]


def test_load_candidates_bad_format(tmp_path):
    path = tmp_path / "cands.json"
    path.write_text(json.dumps({"other": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported candidates file"):
        load_candidates(path)


def test_resolve_candidates_merges_inline_and_file(tmp_path):
    path = tmp_path / "cands.json"
    path.write_text(json.dumps(["from file"]), encoding="utf-8")
    assert resolve_candidates(["inline"], candidates_file=path) == ["inline", "from file"]


def test_resolve_candidates_requires_input():
    with pytest.raises(ValueError, match="Provide candidates"):
        resolve_candidates()


def test_match_candidates_tfidf_only():
    result = match_candidates(
        "sail towards the beach",
        ["near the shore", "in open water"],
        tfidf_only=True,
    )
    assert result.best_language in {"near the shore", "in open water"}
    assert len(result.candidates) == 2


def test_match_candidates_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        match_candidates("goal", [])


def test_record_validation_requires_env_id():
    with pytest.raises(ValueError, match="env_id is required"):
        record_validation("", "goal", "state", correct=True)


def test_format_match_result_includes_best():
    result = match_candidates(
        "sail to beach",
        ["near shore", "open water"],
        encoder=TFIDFEncoder(),
    )
    text = format_match_result(result, top_k=2)
    assert "Best match:" in text
    assert result.best_language in text
