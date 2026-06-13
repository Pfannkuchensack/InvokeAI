from __future__ import annotations

from pathlib import Path

import pytest

from invokeai.app.util.dynamicprompts import build_wildcard_manager, find_missing_wildcards

DB_WILDCARDS = {"animals": ["cat", "dog", "bird"]}


def test_build_wildcard_manager_from_db_wildcards() -> None:
    wm = build_wildcard_manager(db_wildcards=DB_WILDCARDS)
    assert set(wm.get_collection_names()) == {"animals"}
    assert sorted(wm.get_values("animals").iterate_string_values_weighted()) == ["bird", "cat", "dog"]


def test_build_wildcard_manager_merges_disk_and_db(tmp_path: Path) -> None:
    (tmp_path / "sky.txt").write_text("sun\nmoon\n", encoding="utf-8")
    wm = build_wildcard_manager(db_wildcards=DB_WILDCARDS, disk_path=tmp_path)
    assert set(wm.get_collection_names()) == {"animals", "sky"}


def test_build_wildcard_manager_creates_disk_directory(tmp_path: Path) -> None:
    target = tmp_path / "does-not-exist-yet"
    assert not target.exists()
    build_wildcard_manager(disk_path=target)
    assert target.is_dir()


def test_build_wildcard_manager_empty_has_no_wildcards() -> None:
    wm = build_wildcard_manager()
    assert list(wm.get_collection_names()) == []


def test_find_missing_wildcards_detects_unknown_wildcard_in_variant() -> None:
    # Regression: `__random__` inside a variant is parsed as a wildcard reference. Left unchecked it
    # sends the combinatorial generator into an infinite loop, so it must be reported up front.
    wm = build_wildcard_manager(db_wildcards=DB_WILDCARDS)
    assert find_missing_wildcards("{__random__8chan|fenster|stuff}", wm) == ["random"]


def test_find_missing_wildcards_passes_known_wildcard() -> None:
    wm = build_wildcard_manager(db_wildcards=DB_WILDCARDS)
    assert find_missing_wildcards("a {__animals__|house}", wm) == []


@pytest.mark.parametrize("prompt", ["plain text", "{a|b|c}", "a {2$$x|y|z}"])
def test_find_missing_wildcards_ignores_prompts_without_wildcards(prompt: str) -> None:
    wm = build_wildcard_manager(db_wildcards=DB_WILDCARDS)
    assert find_missing_wildcards(prompt, wm) == []


def test_find_missing_wildcards_dedupes_repeated_unknown_wildcards() -> None:
    wm = build_wildcard_manager(db_wildcards=DB_WILDCARDS)
    assert find_missing_wildcards("__nope__ and __nope__ and __animals__", wm) == ["nope"]
