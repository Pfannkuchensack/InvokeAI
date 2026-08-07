"""Regression tests for Ideogram 4 GGUF transformer support.

Two things are covered here:

1. **Config probing.** Ideogram 4's GGUFs carry *no* metadata at all (``kv_count == 0``), so
   identification rests entirely on tensor names, and the conditional/unconditional CFG branch
   can only be told apart by filename — the two files are otherwise identical in names, shapes,
   quantization types and byte size.

2. **The GGUF forward path.** ``GGMLTensor`` only presents its true shape and dtype through
   Python-level attribute overrides. ``nn.Linear`` survives that because its ops are intercepted
   by the dispatch table, but anything that validates in C++ first — ``F.rms_norm``,
   ``F.embedding`` — sees the packed buffer, and reading ``weight.dtype`` yields ``uint8``
   instead of the compute dtype. Both bit when the model is driven from a real GGUF.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock

import gguf
import pytest
import torch
import torch.nn as nn

from invokeai.backend.ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
from invokeai.backend.ideogram4.modeling_ideogram4 import (
    Ideogram4Config,
    Ideogram4Transformer,
    _dequantized,
    _resolve_compute_dtype,
)
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.main import (
    Main_GGUF_Ideogram4_Config,
    _detect_ideogram4_gguf_branch,
    _has_ideogram4_keys,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

_OVERRIDE_FIELDS: dict[str, Any] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/ideogram4.gguf",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

# The two fingerprint tensors the config keys off, plus a bit of filler.
_IDEOGRAM4_KEYS = (
    "embed_image_indicator.weight",
    "llm_cond_proj.weight",
    "llm_cond_norm.weight",
    "layers.0.attention.qkv.weight",
)


def _bf16_ggml(value: torch.Tensor) -> GGMLTensor:
    """Wrap a tensor the way ``gguf_sd_loader`` does for a BF16 GGUF tensor.

    BF16 is not in ``TORCH_COMPATIBLE_QTYPES``, so the loader leaves the raw bytes unshaped —
    which is exactly the situation that breaks naive shape/dtype probing.
    """
    packed = value.to(torch.bfloat16).view(torch.int16).flatten().view(torch.uint8)
    return GGMLTensor(
        packed,
        ggml_quantization_type=gguf.GGMLQuantizationType.BF16,
        tensor_shape=value.shape,
        compute_dtype=torch.float32,
    )


def _make_mod(tmpdir: str, filename: str, state_dict: dict[str, Any]) -> MagicMock:
    path = Path(tmpdir) / filename
    path.write_bytes(b"GGUF")
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = state_dict
    return mod


def _gguf_state_dict(keys=_IDEOGRAM4_KEYS) -> dict[str, Any]:
    return {k: _bf16_ggml(torch.zeros(2)) for k in keys}


# --------------------------------------------------------------------------- config probing


@pytest.mark.parametrize(
    "filename,expected_branch",
    [
        ("ideogram4-transformer-q5_0.gguf", "conditional"),
        ("ideogram4-unconditional_transformer-q5_0.gguf", "unconditional"),
        ("ideogram4-transformer-q4_0.gguf", "conditional"),
        ("Ideogram4-UNCONDITIONAL_transformer-Q8_0.gguf", "unconditional"),
    ],
)
def test_branch_detected_from_filename(filename: str, expected_branch: str) -> None:
    """The CFG branch has no in-file signal — only the name distinguishes the two files."""
    with TemporaryDirectory() as tmpdir:
        mod = _make_mod(tmpdir, filename, _gguf_state_dict())
        config = Main_GGUF_Ideogram4_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.branch == expected_branch
        assert config.base.value == "ideogram-4"
        assert config.format.value == "gguf_quantized"


def test_explicit_branch_override_wins() -> None:
    """A user correcting a renamed file must beat the filename heuristic."""
    with TemporaryDirectory() as tmpdir:
        mod = _make_mod(tmpdir, "my-renamed-model.gguf", _gguf_state_dict())
        overrides = dict(_OVERRIDE_FIELDS) | {"branch": "unconditional"}
        config = Main_GGUF_Ideogram4_Config.from_model_on_disk(mod, overrides)
        assert config.branch == "unconditional"


def test_non_gguf_state_dict_is_rejected() -> None:
    with TemporaryDirectory() as tmpdir:
        plain = {k: torch.zeros(2) for k in _IDEOGRAM4_KEYS}
        mod = _make_mod(tmpdir, "ideogram4-transformer-q5_0.gguf", plain)
        with pytest.raises(NotAMatchError, match="GGUF"):
            Main_GGUF_Ideogram4_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_foreign_gguf_is_rejected() -> None:
    """A GGUF from another architecture must not be claimed as Ideogram 4."""
    with TemporaryDirectory() as tmpdir:
        # Z-Image-ish keys.
        foreign = _gguf_state_dict(("cap_embedder.0.weight", "layers.0.attention.qkv.weight"))
        mod = _make_mod(tmpdir, "some-model-q4_0.gguf", foreign)
        with pytest.raises(NotAMatchError, match="Ideogram 4"):
            Main_GGUF_Ideogram4_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


@pytest.mark.parametrize("lone_key", ["embed_image_indicator.weight", "llm_cond_proj.weight"])
def test_single_fingerprint_key_is_not_enough(lone_key: str) -> None:
    """Both fingerprints are required — one alone is too close to other LLM-conditioned DiTs."""
    assert not _has_ideogram4_keys({lone_key: torch.zeros(2)})


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model.", "diffusion_model."])
def test_comfyui_prefixes_are_tolerated(prefix: str) -> None:
    sd = {prefix + k: torch.zeros(2) for k in _IDEOGRAM4_KEYS}
    assert _has_ideogram4_keys(sd)


def test_unmarked_filename_defaults_to_conditional() -> None:
    """Upstream tags only the unconditional branch, so an untagged name is the conditional one."""
    assert _detect_ideogram4_gguf_branch("ideogram4-transformer-q5_0") == "conditional"


# --------------------------------------------------------------------------- GGUF forward path


def test_resolve_compute_dtype_prefers_module_then_tensor() -> None:
    """fp8 puts compute_dtype on the module, GGUF on the tensor; plain weights report it directly."""
    plain = nn.Linear(4, 4)
    assert _resolve_compute_dtype(plain) == plain.weight.dtype

    gguf_layer = nn.Linear(4, 4)
    gguf_layer.weight = nn.Parameter(_bf16_ggml(torch.zeros(4, 4)), requires_grad=False)
    # The packed buffer reports uint8 — the naive probe that this guards against.
    assert gguf_layer.weight.dtype == torch.uint8
    assert _resolve_compute_dtype(gguf_layer) == torch.float32

    fp8_layer = nn.Linear(4, 4)
    fp8_layer.compute_dtype = torch.bfloat16
    assert _resolve_compute_dtype(fp8_layer) == torch.bfloat16


def test_dequantized_passes_plain_tensors_through() -> None:
    plain = torch.ones(3)
    assert _dequantized(plain) is plain


def test_dequantized_restores_shape_and_dtype() -> None:
    """``GGMLTensor`` hides its true shape from C++ ops; ``_dequantized`` materializes it."""
    value = torch.randn(5)
    wrapped = _bf16_ggml(value)
    # Packed: 5 bf16 values = 10 bytes.
    assert wrapped.quantized_data.shape == (10,)
    restored = _dequantized(wrapped)
    assert restored.shape == (5,)
    assert restored.dtype == torch.float32
    assert torch.allclose(restored, value.to(torch.bfloat16).to(torch.float32))


def _tiny_config() -> Ideogram4Config:
    """A structurally faithful but tiny Ideogram 4.

    ``mrope_section`` must sum to ``head_dim / 4`` — the real (24, 20, 20) against head_dim 256.
    """
    return Ideogram4Config(
        emb_dim=64,
        num_layers=2,
        num_heads=2,
        intermediate_size=128,
        adanln_dim=16,
        in_channels=8,
        llm_features_dim=32,
        mrope_section=(4, 2, 2),
    )


def test_forward_pass_with_gguf_weights() -> None:
    """End-to-end guard for the GGUF path: RMSNorm, the indicator embedding and the Linears.

    Before the shape/dtype handling was added this raised, first in ``input_proj`` ("mat1 and
    mat2 must have the same dtype, but got Byte and BFloat16") and then in ``llm_cond_norm``
    ("Expected weight to be of same shape as normalized_shape").
    """
    config = _tiny_config()
    torch.manual_seed(0)
    reference = Ideogram4Transformer(config)

    model = Ideogram4Transformer(config)
    model.load_state_dict({k: _bf16_ggml(v) for k, v in reference.state_dict().items()}, assign=True)
    model.eval()

    assert any(isinstance(p.data, GGMLTensor) for p in model.parameters())

    batch, n_text, n_image = 1, 3, 4
    seq = n_text + n_image
    indicator = torch.full((batch, seq), OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
    indicator[:, :n_text] = LLM_TOKEN_INDICATOR
    position_ids = torch.zeros(batch, seq, 3, dtype=torch.long)
    position_ids[0, :, 0] = torch.arange(seq)

    with torch.no_grad():
        out = model(
            llm_features=torch.randn(batch, seq, config.llm_features_dim),
            x=torch.randn(batch, seq, config.in_channels),
            t=torch.full((batch,), 0.5),
            position_ids=position_ids,
            segment_ids=torch.zeros(batch, seq, dtype=torch.long),
            indicator=indicator,
        )

    assert out.shape == (batch, seq, config.in_channels)
    assert torch.isfinite(out).all()


def test_forward_matches_unquantized_reference() -> None:
    """Dequantized GGUF weights must reproduce the plain model within bf16 rounding."""
    config = _tiny_config()
    torch.manual_seed(0)
    reference = Ideogram4Transformer(config).to(torch.bfloat16).to(torch.float32)
    reference.eval()

    quantized = Ideogram4Transformer(config)
    quantized.load_state_dict({k: _bf16_ggml(v) for k, v in reference.state_dict().items()}, assign=True)
    quantized.eval()

    batch, seq = 1, 6
    indicator = torch.full((batch, seq), OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
    indicator[:, :2] = LLM_TOKEN_INDICATOR
    position_ids = torch.zeros(batch, seq, 3, dtype=torch.long)
    position_ids[0, :, 0] = torch.arange(seq)
    kwargs = {
        "llm_features": torch.randn(batch, seq, config.llm_features_dim),
        "x": torch.randn(batch, seq, config.in_channels),
        "t": torch.full((batch,), 0.5),
        "position_ids": position_ids,
        "segment_ids": torch.zeros(batch, seq, dtype=torch.long),
        "indicator": indicator,
    }

    with torch.no_grad():
        expected = reference(**kwargs)
        actual = quantized(**kwargs)

    assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
