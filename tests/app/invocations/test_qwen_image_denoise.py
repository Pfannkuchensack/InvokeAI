"""Tests for the Qwen Image denoise invocation."""

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from invokeai.app.invocations.fields import LatentsField
from invokeai.app.invocations.qwen_image import qwen_image_denoise
from invokeai.app.invocations.qwen_image.qwen_image_denoise import QwenImageDenoiseInvocation
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    QwenImageConditioningInfo,
)


class TestPrepareCfgScale:
    """Test _prepare_cfg_scale utility method."""

    def test_scalar_cfg_scale(self):
        inv = QwenImageDenoiseInvocation.model_construct(cfg_scale=4.0)
        result = inv._prepare_cfg_scale(5)
        assert result == [4.0, 4.0, 4.0, 4.0, 4.0]

    def test_list_cfg_scale(self):
        inv = QwenImageDenoiseInvocation.model_construct(cfg_scale=[1.0, 2.0, 3.0])
        result = inv._prepare_cfg_scale(3)
        assert result == [1.0, 2.0, 3.0]

    def test_list_cfg_scale_length_mismatch(self):
        inv = QwenImageDenoiseInvocation.model_construct(cfg_scale=[1.0, 2.0])
        with pytest.raises(AssertionError):
            inv._prepare_cfg_scale(3)

    def test_invalid_cfg_scale_type(self):
        inv = QwenImageDenoiseInvocation.model_construct(cfg_scale="invalid")
        with pytest.raises(ValueError, match="Invalid CFG scale type"):
            inv._prepare_cfg_scale(3)


class TestPackUnpackLatents:
    """Test latent packing and unpacking roundtrip."""

    def test_pack_unpack_roundtrip(self):
        """Packing then unpacking should restore the original tensor."""
        import torch

        latents = torch.randn(1, 16, 128, 128)
        packed = QwenImageDenoiseInvocation._pack_latents(latents, 1, 16, 128, 128)
        assert packed.shape == (1, 64 * 64, 64)  # (B, H/2*W/2, C*4)

        unpacked = QwenImageDenoiseInvocation._unpack_latents(packed, 128, 128)
        assert unpacked.shape == (1, 16, 128, 128)
        assert torch.allclose(latents, unpacked)

    def test_pack_shape(self):
        """Pack should produce the correct shape."""
        import torch

        latents = torch.randn(1, 16, 140, 118)
        packed = QwenImageDenoiseInvocation._pack_latents(latents, 1, 16, 140, 118)
        assert packed.shape == (1, 70 * 59, 64)

    def test_unpack_shape(self):
        """Unpack should produce the correct shape."""
        import torch

        packed = torch.randn(1, 70 * 59, 64)
        unpacked = QwenImageDenoiseInvocation._unpack_latents(packed, 140, 118)
        assert unpacked.shape == (1, 16, 140, 118)


class TestAlignRefLatentDims:
    """Test reference latent dim alignment for 2x2 packing."""

    def test_even_dims_unchanged(self):
        assert QwenImageDenoiseInvocation._align_ref_latent_dims(96, 64) == (96, 64)

    def test_odd_dims_trimmed_to_even(self):
        assert QwenImageDenoiseInvocation._align_ref_latent_dims(97, 65) == (96, 64)
        assert QwenImageDenoiseInvocation._align_ref_latent_dims(150, 151) == (150, 150)

    def test_minimum_aligned_dims(self):
        assert QwenImageDenoiseInvocation._align_ref_latent_dims(2, 2) == (2, 2)
        assert QwenImageDenoiseInvocation._align_ref_latent_dims(3, 2) == (2, 2)

    def test_raises_on_zero_dim(self):
        with pytest.raises(ValueError, match="spatial dims must be >= 2"):
            QwenImageDenoiseInvocation._align_ref_latent_dims(0, 64)
        with pytest.raises(ValueError, match="spatial dims must be >= 2"):
            QwenImageDenoiseInvocation._align_ref_latent_dims(64, 0)

    def test_raises_on_one_dim(self):
        """A 1-pixel latent aligns to 0 and must be rejected."""
        with pytest.raises(ValueError, match="spatial dims must be >= 2"):
            QwenImageDenoiseInvocation._align_ref_latent_dims(1, 64)
        with pytest.raises(ValueError, match="spatial dims must be >= 2"):
            QwenImageDenoiseInvocation._align_ref_latent_dims(64, 1)


class TestMaybeClampRefLatentSize:
    """Test the diffusers-style VAE_IMAGE_SIZE clamp applied to reference latents
    before packing. This is defense-in-depth for backend callers (direct API,
    older graph JSON) that wire qwen_image_i2l without explicit width/height —
    without the clamp, the transformer receives an out-of-distribution sequence
    length and VRAM usage spikes on large reference images."""

    def test_in_budget_latent_unchanged(self):
        """A 1024² ref image → 128x128 latent → exactly the budget. Pass through."""
        import torch

        ref = torch.randn(1, 16, 128, 128)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        assert result.shape == (1, 16, 128, 128)
        assert result is ref  # identity, no copy

    def test_small_latent_unchanged(self):
        """A 512² ref → 64x64 latent (4x under budget). Pass through unchanged."""
        import torch

        ref = torch.randn(1, 16, 64, 64)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        assert result.shape == (1, 16, 64, 64)
        assert result is ref

    def test_native_resolution_landscape_clamped(self):
        """A native 1600x1200 image → 200x150 latents. Should clamp to the same
        dims diffusers produces (1184x896 pixels → 148x112 latents)."""
        import torch

        ref = torch.randn(1, 16, 150, 200)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        assert result.shape == (1, 16, 112, 148)

    def test_native_resolution_portrait_clamped(self):
        """1200x1600 → 150x200 latents → diffusers target 896x1184 → 112x148."""
        import torch

        ref = torch.randn(1, 16, 200, 150)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        assert result.shape == (1, 16, 148, 112)

    def test_huge_latent_clamped(self):
        """A 4096x4096 image → 512x512 latents (16x budget). Clamp to 128x128
        latents (= 1024² pixels), well within model's trained distribution."""
        import torch

        ref = torch.randn(1, 16, 512, 512)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        assert result.shape == (1, 16, 128, 128)

    def test_clamp_preserves_aspect_ratio_within_rounding(self):
        """Aspect ratio of the clamped latent should match the input to within
        the 32-pixel snapping granularity used by diffusers."""
        import torch

        # 1920x1080 (16:9, ~2M pixels)
        ref = torch.randn(1, 16, 135, 240)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        # diffusers: calculate_dimensions(1024², 16/9) → (1376, 768) px → (172, 96) latent
        assert result.shape == (1, 16, 96, 172)

    def test_clamp_output_is_packable(self):
        """The clamped latent must have even spatial dims (required by 2x2 packing)
        before _align_ref_latent_dims is called. Because the clamp snaps to 32px
        in pixel space and vae_scale_factor=8, every clamp output is a multiple
        of 4 in latent space (and therefore even)."""
        import torch

        for h, w in [(150, 200), (200, 150), (135, 240), (512, 512)]:
            ref = torch.randn(1, 16, h, w)
            result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
            _, _, rh, rw = result.shape
            assert rh % 2 == 0, f"clamp produced odd height {rh} for input ({h},{w})"
            assert rw % 2 == 0, f"clamp produced odd width {rw} for input ({h},{w})"


class TestBuildImgShapes:
    """Test img_shapes construction. Regression test for the ghosting/doubling bug
    where ref and noisy segments shared identical spatial RoPE positions."""

    def test_txt2img_single_segment(self):
        """No reference latent → single segment for the noisy latent only."""
        result = QwenImageDenoiseInvocation._build_img_shapes(64, 64)
        assert result == [[(1, 32, 32)]]

    def test_edit_uses_distinct_ref_dims(self):
        """Edit-mode img_shapes must place ref segment at the ref's OWN dims, not
        the noisy dims. Identical dims caused the ghosting artifact."""
        noisy_h, noisy_w = 64, 64
        ref_h, ref_w = 96, 64
        result = QwenImageDenoiseInvocation._build_img_shapes(noisy_h, noisy_w, ref_h, ref_w)
        assert result == [[(1, 32, 32), (1, 48, 32)]]
        # The bug was that both segments had the same shape:
        assert result[0][0] != result[0][1]

    def test_edit_matches_diffusers_layout(self):
        """Structure must match diffusers QwenImageEditPipeline (single batch,
        nested list of (frame, h//2, w//2) tuples)."""
        result = QwenImageDenoiseInvocation._build_img_shapes(80, 112, 128, 96)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 2
        assert result[0][0] == (1, 40, 56)
        assert result[0][1] == (1, 64, 48)


@pytest.mark.parametrize(
    ("width", "height", "mu"),
    [
        # One token per 16x16 pixels (8x VAE, 2x2 patches). The stock fit is mu = 0.5 + 0.4 * (tokens - 256) / 7936.
        (16, 16, 0.5 - 0.4 * 255 / 7936),  # 1 token: below the lower anchor the line keeps running
        (1024, 1024, 0.6935483870967742),  # 4096 tokens: 0.5 + 0.4 * 3840 / 7936
        (2048, 1008, 0.8935483870967742),  # 8064 tokens: 0.5 + 0.4 * 7808 / 7936; area decides, not a side
        (2048, 1024, 0.9),  # 8192 tokens: the fit's end
        (4096, 4096, 0.9),  # 65536 tokens: 3.790 extrapolated
    ],
)
def test_without_a_scheduler_folder_mu_follows_the_stock_fit_and_holds_at_its_end(
    tmp_path: Path, width: int, height: int, mu: float
) -> None:
    # A GGUF or single-file transformer has no scheduler/ folder.
    _, actual = QwenImageDenoiseInvocation._schedule(tmp_path / "qwen-image.gguf", width, height, 2, None)

    assert actual == pytest.approx(mu, abs=1e-12)


def _with_scheduler_config(model_path: Path) -> Path:
    (model_path / "scheduler").mkdir(parents=True)
    (model_path / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "use_dynamic_shifting": True,
                "base_image_seq_len": 256,
                "max_image_seq_len": 1024,
                "base_shift": 0.3,
                "max_shift": 1.5,
                "shift_terminal": 0.02,
                "time_shift_type": "exponential",
            }
        )
    )
    return model_path


@pytest.mark.parametrize(
    ("size", "mu"),
    [
        (384, 0.8),  # 576 tokens: 0.3 + 1.2 * 320 / 768
        (2048, 1.5),  # 16384 tokens: held at this config's max_shift
    ],
)
def test_a_checkpoints_own_scheduler_config_sets_the_fit(tmp_path: Path, size: int, mu: float) -> None:
    scheduler, actual = QwenImageDenoiseInvocation._schedule(_with_scheduler_config(tmp_path), size, size, 2, None)

    assert actual == pytest.approx(mu, abs=1e-12)
    assert scheduler.config.max_shift == 1.5


def test_an_explicit_shift_sets_mu_at_any_size_with_the_checkpoints_scheduler(tmp_path: Path) -> None:
    scheduler, mu = QwenImageDenoiseInvocation._schedule(_with_scheduler_config(tmp_path), 4096, 4096, 2, 3.0)

    assert mu == pytest.approx(1.0986122886681098, abs=1e-12)  # ln 3
    assert scheduler.config.max_shift == 1.5


class _StopAtTimesteps(Exception):
    """Ends a run once the scheduler has been handed its mu."""


class _TinyQwenImageTransformer:
    """Stands in for `QwenImageTransformer2DModel`: the config the node reads before it sets the schedule."""

    config = SimpleNamespace(patch_size=2, out_channels=16)


def _run_to_scheduler(monkeypatch, model_path: Path, width: int, height: int, shift: float | None = None) -> float:
    """Run `_run_diffusion` with the real scheduler up to `set_timesteps` and return the mu it was handed."""
    monkeypatch.setattr(qwen_image_denoise, "QwenImageTransformer2DModel", _TinyQwenImageTransformer)
    monkeypatch.setattr(qwen_image_denoise.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    captured: list[float] = []

    def capture_mu(self, *, sigmas, mu, device) -> None:
        captured.append(mu)
        # A frame of real size need not be denoised to know the mu it was handed.
        raise _StopAtTimesteps

    monkeypatch.setattr(FlowMatchEulerDiscreteScheduler, "set_timesteps", capture_mu)
    conditioning = ConditioningFieldData(conditionings=[QwenImageConditioningInfo(prompt_embeds=torch.zeros(1, 4, 8))])
    context = SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: SimpleNamespace(model=_TinyQwenImageTransformer()),
            get_config=lambda _identifier: SimpleNamespace(),
            get_absolute_path=lambda _config: model_path,
        ),
        conditioning=SimpleNamespace(load=lambda _name: conditioning),
    )
    node = QwenImageDenoiseInvocation.model_construct(
        latents=None,
        reference_latents=None,
        denoise_mask=None,
        denoising_start=0.0,
        denoising_end=1.0,
        transformer=SimpleNamespace(transformer=SimpleNamespace(), loras=[]),
        positive_conditioning=SimpleNamespace(conditioning_name="positive"),
        negative_conditioning=None,
        cfg_scale=1.0,
        width=width,
        height=height,
        steps=40,
        seed=0,
        shift=shift,
    )

    with pytest.raises(_StopAtTimesteps):
        node._run_diffusion(context)

    assert len(captured) == 1
    return captured[0]


def test_an_explicit_shift_reaches_the_scheduler(monkeypatch, tmp_path) -> None:
    mu = _run_to_scheduler(monkeypatch, tmp_path / "qwen-image.gguf", 4096, 4096, shift=3.0)

    assert mu == pytest.approx(1.0986122886681098, abs=1e-12)  # ln 3


def test_the_first_preview_is_sent_inside_the_transformer_lock(monkeypatch, tmp_path) -> None:
    """A Tiny AutoEncoder preview places its TAE next to the locked model; before the lock there is none to fit."""
    monkeypatch.setattr(qwen_image_denoise, "QwenImageTransformer2DModel", _TinyQwenImageTransformer)
    monkeypatch.setattr(qwen_image_denoise.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    events: list[str] = []

    class _Previewed(Exception):
        """Ends the run at the first preview: what follows needs a real transformer."""

    class _Transformer(_TinyQwenImageTransformer):
        def modules(self):
            return iter(())

    @contextmanager
    def lock(**_kwargs):
        events.append("lock")
        try:
            yield ({}, _Transformer())
        finally:
            events.append("unlock")

    def preview(*_args) -> None:
        events.append("preview")
        raise _Previewed

    conditioning = ConditioningFieldData(conditionings=[QwenImageConditioningInfo(prompt_embeds=torch.zeros(1, 4, 8))])
    context = SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: SimpleNamespace(model=_Transformer(), model_on_device=lock),
            get_config=lambda _identifier: SimpleNamespace(),
            get_absolute_path=lambda _config: tmp_path / "qwen-image.gguf",
        ),
        conditioning=SimpleNamespace(load=lambda _name: conditioning),
        util=SimpleNamespace(sd_step_callback=preview),
    )
    node = QwenImageDenoiseInvocation.model_construct(
        latents=None,
        reference_latents=None,
        denoise_mask=None,
        control=None,
        denoising_start=0.0,
        denoising_end=1.0,
        transformer=SimpleNamespace(transformer=SimpleNamespace(), loras=[]),
        positive_conditioning=SimpleNamespace(conditioning_name="positive"),
        negative_conditioning=None,
        cfg_scale=1.0,
        width=64,
        height=64,
        steps=2,
        seed=0,
        shift=None,
    )

    with pytest.raises(_Previewed):
        node._run_diffusion(context)

    assert events == ["lock", "preview", "unlock"]


def test_a_shift_of_zero_sets_the_same_mu_as_no_shift(monkeypatch, tmp_path) -> None:
    # The workflow editor sends 0 for an optional float field left alone; log(0) would crash the run.
    model_path = tmp_path / "qwen-image.gguf"
    auto = _run_to_scheduler(monkeypatch, model_path, 1024, 1024, shift=None)
    zero = _run_to_scheduler(monkeypatch, model_path, 1024, 1024, shift=0.0)

    assert zero == pytest.approx(0.5 + 0.4 * 3840 / 7936, abs=1e-12)  # the stock fit at 4096 tokens
    assert zero == auto


def test_a_negative_shift_sets_the_same_mu_as_no_shift(monkeypatch, tmp_path) -> None:
    # The field has no lower bound; log() of a negative shift would crash the run.
    model_path = tmp_path / "qwen-image.gguf"
    auto = _run_to_scheduler(monkeypatch, model_path, 1024, 1024, shift=None)
    negative = _run_to_scheduler(monkeypatch, model_path, 1024, 1024, shift=-1.0)

    assert negative == pytest.approx(0.5 + 0.4 * 3840 / 7936, abs=1e-12)
    assert negative == auto


class TestBuildImgShapesMultipleReferences:
    def test_one_row_per_reference_in_order(self):
        result = QwenImageDenoiseInvocation._build_img_shapes(64, 64, ref_latent_dims=[(128, 96), (64, 160)])
        assert result == [[(1, 32, 32), (1, 64, 48), (1, 32, 80)]]


class _EditTransformer(_TinyQwenImageTransformer):
    """An Edit-style transformer stub: records the reference tokens it is fed and predicts from its input."""

    def __init__(self, zero_cond_t: bool = True) -> None:
        self.zero_cond_t = zero_cond_t
        self.config = SimpleNamespace(patch_size=2, out_channels=16, zero_cond_t=zero_cond_t)
        self.calls: list[dict] = []

    def modules(self):
        return iter(())

    def __call__(self, *, hidden_states, img_shapes, **_kwargs):
        noisy_seq_len = (64 // 16) ** 2
        self.calls.append(
            {
                "seq_len": hidden_states.shape[1],
                "img_shapes": img_shapes,
                "reference_tokens": hidden_states[:, noisy_seq_len:].clone(),
            }
        )
        return (hidden_states * 0.5,)


def _run_edit_denoise(
    monkeypatch, tmp_path: Path, reference_latents, saved: dict[str, torch.Tensor], *, zero_cond_t: bool = True
):
    """Run the node end to end on CPU against a recording stub; return (output, transformer, wm bytes)."""
    transformer = _EditTransformer(zero_cond_t)
    working_memory: list[int] = []
    monkeypatch.setattr(qwen_image_denoise, "QwenImageTransformer2DModel", _EditTransformer)
    monkeypatch.setattr(qwen_image_denoise.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(qwen_image_denoise, "requires_sidecar_patching", lambda *_a: False)
    monkeypatch.setattr(qwen_image_denoise, "peak_dequant_transient_bytes", lambda *_a: 0)
    monkeypatch.setattr(qwen_image_denoise, "sage_attention_scope", lambda: nullcontext())
    monkeypatch.setattr(qwen_image_denoise.LayerPatcher, "apply_smart_model_patches", lambda **_kwargs: nullcontext())

    @contextmanager
    def lock(*, working_mem_bytes):
        working_memory.append(working_mem_bytes)
        yield ({}, transformer)

    conditioning = ConditioningFieldData(conditionings=[QwenImageConditioningInfo(prompt_embeds=torch.zeros(1, 4, 8))])
    context = SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: SimpleNamespace(model=transformer, model_on_device=lock),
            get_config=lambda _identifier: SimpleNamespace(format=None),
            get_absolute_path=lambda _config: tmp_path / "qwen-image.gguf",
        ),
        conditioning=SimpleNamespace(load=lambda _name: conditioning),
        tensors=SimpleNamespace(load=lambda name: saved[name]),
        util=SimpleNamespace(sd_step_callback=lambda *_a: None),
    )
    node = QwenImageDenoiseInvocation.model_construct(
        latents=None,
        reference_latents=reference_latents,
        denoise_mask=None,
        control=None,
        denoising_start=0.0,
        denoising_end=1.0,
        transformer=SimpleNamespace(transformer=SimpleNamespace(), loras=[]),
        positive_conditioning=SimpleNamespace(conditioning_name="positive"),
        negative_conditioning=None,
        cfg_scale=1.0,
        width=64,
        height=64,
        steps=2,
        seed=0,
        shift=None,
    )
    return node._run_diffusion(context), transformer, working_memory


def _pack(latents: torch.Tensor) -> torch.Tensor:
    """Independent 2x2 packing of (1, C, 1, h, w) or (1, C, h, w) latents to (1, h/2*w/2, C*4)."""
    latents = latents.to(torch.bfloat16).reshape(latents.shape[0], latents.shape[1], *latents.shape[-2:])
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, h * w // 4, c * 4)


def _field(name: str) -> LatentsField:
    return LatentsField(latents_name=name)


class TestMultipleReferenceLatents:
    @staticmethod
    def _saved() -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(1)
        return {
            "a": torch.randn(1, 16, 1, 16, 12, generator=gen),
            "b": torch.randn(1, 16, 1, 8, 24, generator=gen),
        }

    def test_list_feeds_the_references_packed_in_the_given_order(self, monkeypatch, tmp_path):
        saved = self._saved()
        shapes = {"a": (1, 8, 6), "b": (1, 4, 12)}
        for order in (["a", "b"], ["b", "a"]):
            _, transformer, _ = _run_edit_denoise(monkeypatch, tmp_path, [_field(n) for n in order], saved)
            call = transformer.calls[0]
            assert call["img_shapes"] == [[(1, 4, 4), *(shapes[n] for n in order)]]
            expected = torch.cat([_pack(saved[n]) for n in order], dim=1)
            torch.testing.assert_close(call["reference_tokens"], expected, rtol=0, atol=0)
            assert call["seq_len"] == 16 + expected.shape[1]

    def test_working_memory_counts_every_reference(self, monkeypatch, tmp_path):
        saved = self._saved()
        _, _, one = _run_edit_denoise(monkeypatch, tmp_path, [_field("a")], saved)
        _, _, two = _run_edit_denoise(monkeypatch, tmp_path, [_field("a"), _field("b")], saved)
        # 0.25 MiB per token (bf16): the second reference adds 4 x 12 = 48 tokens.
        assert two[0] - one[0] == 48 * 256 * 1024

    def test_list_of_one_feeds_what_the_single_field_feeds(self, monkeypatch, tmp_path):
        saved = self._saved()
        single, t_single, wm_single = _run_edit_denoise(monkeypatch, tmp_path, _field("a"), saved)
        listed, t_list, wm_list = _run_edit_denoise(monkeypatch, tmp_path, [_field("a")], saved)

        assert t_list.calls[0]["img_shapes"] == t_single.calls[0]["img_shapes"]
        torch.testing.assert_close(
            t_list.calls[0]["reference_tokens"], t_single.calls[0]["reference_tokens"], rtol=0, atol=0
        )
        assert wm_list == wm_single
        torch.testing.assert_close(listed, single, rtol=0, atol=0)

    def test_single_field_keeps_the_clamp_and_zero_cond_t_gate(self, monkeypatch, tmp_path):
        big = {"big": torch.zeros(1, 16, 1, 256, 256)}  # 2048x2048 pixels
        _, single, _ = _run_edit_denoise(monkeypatch, tmp_path, _field("big"), big)
        assert single.calls[0]["img_shapes"] == [[(1, 4, 4), (1, 64, 64)]]

        # Without zero_cond_t the single field is dropped, as it always was (the loader fix is a separate change).
        _, plain, _ = _run_edit_denoise(monkeypatch, tmp_path, _field("a"), self._saved(), zero_cond_t=False)
        assert plain.calls[0]["img_shapes"] == [[(1, 4, 4)]]
        assert plain.calls[0]["seq_len"] == 16

    def test_list_feeds_references_even_without_zero_cond_t(self, monkeypatch, tmp_path):
        saved = self._saved()
        _, transformer, _ = _run_edit_denoise(
            monkeypatch, tmp_path, [_field("a"), _field("b")], saved, zero_cond_t=False
        )
        assert transformer.calls[0]["img_shapes"] == [[(1, 4, 4), (1, 8, 6), (1, 4, 12)]]
        torch.testing.assert_close(
            transformer.calls[0]["reference_tokens"],
            torch.cat([_pack(saved["a"]), _pack(saved["b"])], dim=1),
            rtol=0,
            atol=0,
        )

    def test_without_references_the_edit_transformer_still_gets_a_zero_segment(self, monkeypatch, tmp_path):
        _, transformer, _ = _run_edit_denoise(monkeypatch, tmp_path, None, {})
        assert transformer.calls[0]["img_shapes"] == [[(1, 4, 4), (1, 4, 4)]]

    @pytest.mark.parametrize(
        ("latents", "message"),
        [
            (torch.zeros(1, 16, 1, 136, 136), "above the one megapixel"),  # 1088x1088
            (torch.zeros(1, 16, 1, 256, 256), "above the one megapixel"),  # 2048x2048
            (torch.zeros(1, 4, 1, 16, 16), "expected"),
            (torch.zeros(1, 16, 2, 16, 16), "expected"),
            (torch.zeros(2, 16, 1, 16, 16), "expected"),
        ],
    )
    def test_list_rejects_unusable_references(self, monkeypatch, tmp_path, latents, message):
        with pytest.raises(ValueError, match=message):
            _run_edit_denoise(monkeypatch, tmp_path, [_field("x")], {"x": latents})

    @pytest.mark.parametrize("ratio", [16 / 9, 9 / 16, 4 / 3, 3 / 4, 1.0, 3 / 2, 21 / 9, 1 / 3, 7 / 5])
    def test_list_accepts_every_calculate_dimensions_size(self, monkeypatch, tmp_path, ratio):
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import VAE_IMAGE_SIZE, calculate_dimensions

        width, height = calculate_dimensions(VAE_IMAGE_SIZE, ratio)
        latents = {"x": torch.zeros(1, 16, 1, height // 8, width // 8)}
        _, transformer, _ = _run_edit_denoise(monkeypatch, tmp_path, [_field("x")], latents)
        assert transformer.calls[0]["img_shapes"] == [[(1, 4, 4), (1, height // 16, width // 16)]]

    def test_empty_list_on_an_edit_transformer_raises(self, monkeypatch, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            _run_edit_denoise(monkeypatch, tmp_path, [], {})
