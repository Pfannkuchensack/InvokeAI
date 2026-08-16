import pytest
import torch

from invokeai.backend.krea2.prompt_weights import (
    MAX_WEIGHT,
    MIN_WEIGHT,
    build_token_weights,
    derive_attention_weights,
    parse_weighted_prompt,
)


def test_prompt_without_markers_is_returned_unchanged() -> None:
    prompt = "A stone bridge over a river (at dawn), photographed on 35mm film."
    clean, spans = parse_weighted_prompt(prompt)

    # A parenthetical with no weight after it is ordinary prose, not a marker.
    assert clean == prompt
    assert spans == []


@pytest.mark.parametrize(
    "prompt",
    ["a woman with (blonde hair:-1) walking", "a woman with (blonde hair)-1 walking"],
)
def test_both_notations_parse_identically(prompt: str) -> None:
    # ComfyUI's colon form and InvokeAI's trailing-number form must be interchangeable, negatives included.
    clean, spans = parse_weighted_prompt(prompt)

    assert clean == "a woman with blonde hair walking"
    assert len(spans) == 1
    assert spans[0].weight == -1.0
    assert spans[0].text == "blonde hair"
    assert clean[spans[0].start : spans[0].end] == "blonde hair"


def test_whitespace_is_preserved_byte_for_byte() -> None:
    # This is the compel regression the bespoke parser exists to avoid: compel flattens a prompt to
    # "Zeile eins. Zeile zwei mit wort drin.", silently changing the unweighted text as well.
    prompt = "Zeile eins.\n\nZeile zwei  mit (wort:1.5) drin."
    clean, spans = parse_weighted_prompt(prompt)

    assert clean == "Zeile eins.\n\nZeile zwei  mit wort drin."
    assert len(spans) == 1
    assert clean[spans[0].start : spans[0].end] == "wort"


def test_escaped_parentheses_survive_and_shift_later_offsets() -> None:
    prompt = r"a sign \(closed\) beside a (red door:1.4)"
    clean, spans = parse_weighted_prompt(prompt)

    assert clean == "a sign (closed) beside a red door"
    assert len(spans) == 1
    # The escapes shortened the clean string by two characters; the span must track that, not the raw index.
    assert clean[spans[0].start : spans[0].end] == "red door"


def test_escaped_parentheses_inside_a_weighted_phrase() -> None:
    clean, spans = parse_weighted_prompt(r"a (neon sign \(open\):1.6) at night")

    assert clean == "a neon sign (open) at night"
    assert clean[spans[0].start : spans[0].end] == "neon sign (open)"
    assert spans[0].weight == 1.6


def test_multiple_spans_are_independent_and_non_overlapping() -> None:
    clean, spans = parse_weighted_prompt("a (cat:0.2) and a cat and a (dog:2.0)")

    assert clean == "a cat and a cat and a dog"
    assert [(s.text, s.weight) for s in spans] == [("cat", 0.2), ("dog", 2.0)]
    # Only the marked occurrence of "cat" is weighted -- the bare one keeps its own offsets.
    assert clean[spans[0].start : spans[0].end] == "cat"
    assert spans[0].end < clean.index("cat", spans[0].end)


def test_neutral_weight_removes_the_marker_without_producing_a_span() -> None:
    clean, spans = parse_weighted_prompt("a (cat:1.0) sits")

    assert clean == "a cat sits"
    assert spans == []


def test_malformed_marker_is_left_as_literal_text() -> None:
    clean, spans = parse_weighted_prompt("a (cat:abc) sits")

    assert clean == "a (cat:abc) sits"
    assert spans == []


@pytest.mark.parametrize("prompt", ["a (clock reading 3:30) on the wall", "a (cat:-9) sits"])
def test_out_of_band_weight_is_emitted_literally(prompt: str) -> None:
    # Prose that happens to end in ":<number>)" must not be reinterpreted as extreme emphasis, and a
    # typo'd weight must not be silently clamped into a huge k_bias.
    clean, spans = parse_weighted_prompt(prompt)

    assert clean == prompt
    assert spans == []


def test_weights_at_the_band_edges_are_accepted() -> None:
    _, spans = parse_weighted_prompt(f"a (cat:{MIN_WEIGHT}) and a (dog:{MAX_WEIGHT})")

    assert [s.weight for s in spans] == [MIN_WEIGHT, MAX_WEIGHT]


def test_empty_phrase_is_dropped() -> None:
    clean, spans = parse_weighted_prompt("a (:1.5) cat")

    assert clean == "a  cat"
    assert spans == []


def _offsets(text: str, words: list[str], *, prefix_len: int = 0) -> torch.Tensor:
    """Build an offset mapping by locating each word in `text`, with two (0, 0) special tokens appended."""
    offsets = []
    cursor = 0
    for word in words:
        start = text.index(word, cursor)
        offsets.append((start + prefix_len, start + len(word) + prefix_len))
        cursor = start + len(word)
    return torch.tensor(offsets + [(0, 0), (0, 0)], dtype=torch.long)


def test_build_token_weights_maps_spans_onto_the_right_tokens() -> None:
    prefix = "SYSTEM PREFIX: "
    clean, spans = parse_weighted_prompt("a red door and a blue (door:0.3) frame")
    words = ["a", " red", " door", " and", " a", " blue", " door", " frame"]
    weights = build_token_weights(_offsets(clean, words, prefix_len=len(prefix)), spans, len(prefix))

    assert weights is not None
    # Only the second "door" -- the marked one -- is weighted; the two trailing specials stay neutral.
    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0])


def test_build_token_weights_covers_every_token_a_span_overlaps() -> None:
    # A BPE merge straddling the phrase boundary still counts: overlap, not containment.
    spans = parse_weighted_prompt("x (bright red:0.5) y")[1]
    offsets = torch.tensor([(0, 2), (2, 8), (8, 12), (12, 14)], dtype=torch.long)
    weights = build_token_weights(offsets, spans, 0)

    # Span covers characters [2, 12); the token at [12, 14) starts outside it and stays neutral.
    assert weights is not None
    assert weights.tolist() == [1.0, 0.5, 0.5, 1.0]


def test_build_token_weights_ignores_padding_and_special_tokens() -> None:
    spans = parse_weighted_prompt("(cat:0.4)")[1]
    offsets = torch.tensor([(0, 0), (0, 3), (0, 0), (0, 0)], dtype=torch.long)
    weights = build_token_weights(offsets, spans, 0)

    assert weights is not None
    assert weights.tolist() == pytest.approx([1.0, 0.4, 1.0, 1.0])


def test_build_token_weights_returns_none_when_a_span_is_truncated_away() -> None:
    # The phrase sits past the tokenizer's truncation limit, so no token carries it. That is a warning,
    # not a failed generation.
    spans = parse_weighted_prompt("x (cat:0.4)")[1]
    offsets = torch.tensor([(0, 1)], dtype=torch.long)

    assert build_token_weights(offsets, spans, 0) is None


def test_build_token_weights_returns_none_without_spans() -> None:
    assert build_token_weights(torch.tensor([(0, 3)], dtype=torch.long), [], 0) is None


def test_build_token_weights_rejects_a_malformed_offset_mapping() -> None:
    with pytest.raises(ValueError, match=r"shape \(seq_len, 2\)"):
        build_token_weights(torch.zeros(4, dtype=torch.long), [], 0)


@pytest.mark.parametrize(
    ("weight", "strength", "expected_scale", "expected_bias"),
    [
        (1.0, 1.0, 1.0, 0.0),
        (0.5, 1.0, 0.5, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 2.0, -1.0, 0.0),  # strength > 1 reaches the subtractive regime without negative syntax
        (-1.0, 1.0, -1.0, 0.0),
        (2.0, 1.0, 1.0, 2.0),
        (2.0, 0.0, 1.0, 0.0),  # strength 0 is the identity for every weight
        (0.0, 0.0, 1.0, 0.0),
    ],
)
def test_derive_attention_weights(weight: float, strength: float, expected_scale: float, expected_bias: float) -> None:
    value_scale, key_bias = derive_attention_weights(torch.tensor([weight]), strength)

    assert value_scale.item() == pytest.approx(expected_scale)
    assert key_bias.item() == pytest.approx(expected_bias)
