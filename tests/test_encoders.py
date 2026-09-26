"""Tests for `anometa.features.encoders`: frozen DINOv3 and SigLIP2 encoders."""

import numpy as np
import pytest
import torch
from conftest import FakeEncoder
from httpx import Request, Response
from huggingface_hub.errors import GatedRepoError

from anometa.config import Paths
from anometa.data.ad2 import load_rgb
from anometa.features import encoders
from anometa.features.encoders import (
    DIMS,
    dinov3_size,
    feature_tag,
    load_encoder,
    split_dinov3_tokens,
    split_siglip2_tokens,
)


def _gated_repo_error(message: str) -> GatedRepoError:
    """Build a `GatedRepoError` with the minimal response `httpx` object it reads.

    Args:
        message: The error message.

    Returns:
        A `GatedRepoError` usable without a real HTTP exchange.
    """
    response = Response(403, request=Request("GET", "https://huggingface.co/gated"))
    return GatedRepoError(message, response=response)


@pytest.mark.parametrize(
    ("hw", "expected"),
    [
        ((1056, 4224), (512, 2048)),  # sheet_metal
        ((1900, 1400), (688, 512)),  # vial
        ((1024, 2232), (512, 1120)),  # can
        ((2048, 2448), (512, 608)),  # fabric, rice, wallplugs, walnuts
    ],
)
def test_dinov3_size(hw: tuple[int, int], expected: tuple[int, int]) -> None:
    """Short side 512, aspect kept, both sides multiples of 16."""
    assert dinov3_size(*hw) == expected


def test_dinov3_size_sheet_metal() -> None:
    """Sheet Metal gives a 32x128 patch grid."""
    h, w = dinov3_size(1056, 4224)
    assert (h // 16, w // 16) == (32, 128)


def test_split_dinov3_tokens_layout() -> None:
    """CLS is token 0, 4 registers follow, patches are row-major after them."""
    h, w, d = 2, 3, 5
    hidden = torch.arange((1 + 4 + h * w) * d, dtype=torch.float32).reshape(1, -1, d)
    enc = split_dinov3_tokens(hidden, num_register_tokens=4, grid=(h, w))
    assert torch.equal(enc.cls, hidden[0, 0])
    assert enc.patches.shape == (h, w, d)
    assert torch.equal(enc.patches[0, 0], hidden[0, 5])
    assert torch.equal(enc.patches[1, 0], hidden[0, 5 + w])


@pytest.mark.parametrize("wrapped", [False, True], ids=["direct", "wrapped"])
def test_auto_backend_falls_back_to_timm_when_gated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, wrapped: bool
) -> None:
    """A gated-repo error, raw or wrapped in OSError, yields the timm encoder and a warning."""

    def gated(name: object, device: object) -> encoders.Encoder:
        """Fake transformers loader failing as transformers 5.17 does on a gated repo."""
        if wrapped:
            raise OSError("gated") from _gated_repo_error("gated")
        raise _gated_repo_error("gated")

    monkeypatch.setattr(encoders, "_load_transformers_dinov3", gated)
    monkeypatch.setattr(
        encoders, "_load_timm_dinov3", lambda name, device: FakeEncoder(backend="timm")
    )
    with caplog.at_level("WARNING"):
        enc = load_encoder("dinov3_s", "cpu")
    assert enc.backend == "timm"
    assert "gated" in caplog.text.lower()
    assert feature_tag("dinov3_s", "timm") == "r512-timm"
    assert feature_tag("siglip2", "transformers") == "p1024"


def test_auto_backend_reraises_non_gated_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OSError with no gated-repository cause propagates instead of falling back."""

    def offline(name: object, device: object) -> encoders.Encoder:
        """Fake transformers loader that fails for a non-gated reason."""
        raise OSError("connection refused")

    def timm_loader(name: object, device: object) -> encoders.Encoder:
        """Fake timm loader that must never be reached."""
        raise AssertionError("fell back to timm")

    monkeypatch.setattr(encoders, "_load_transformers_dinov3", offline)
    monkeypatch.setattr(encoders, "_load_timm_dinov3", timm_loader)
    with pytest.raises(OSError, match="connection refused"):
        load_encoder("dinov3_s", "cpu")


def test_timm_backend_is_dinov3_only() -> None:
    """SigLIP2 has no timm backend."""
    with pytest.raises(ValueError, match="siglip2"):
        load_encoder("siglip2", "cpu", backend="timm")


def test_split_siglip2_tokens_drops_padding() -> None:
    """Only the first h*w tokens are patches; the pooled vector is the CLS slot."""
    hidden = torch.randn(10, 4)
    pooled = torch.randn(4)
    enc = split_siglip2_tokens(hidden, pooled, (2, 3))
    assert enc.patches.shape == (2, 3, 4)
    assert torch.equal(enc.patches.reshape(6, 4), hidden[:6])
    assert torch.equal(enc.cls, pooled)


@pytest.mark.models
@pytest.mark.parametrize(("name", "grid"), [("dinov3_s", (32, 128)), ("siglip2", None)])
def test_real_encoder_on_sheet_metal_shape(
    name: encoders.EncoderName, grid: tuple[int, int] | None
) -> None:
    """Real weights: a 1056x4224 image encodes to the expected grid and width."""
    enc = load_encoder(name, "cpu").encode(np.zeros((1056, 4224, 3), np.uint8))
    assert enc.cls.shape == (DIMS[name],)
    h, w, _ = enc.patches.shape
    assert (h, w) == grid if grid else h * w <= 1024


@pytest.mark.gpu
@pytest.mark.models
@pytest.mark.data
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("name", ["dinov3_s", "dinov3_l"])
def test_dinov3_cuda_bf16_matches_cpu_fp32(name) -> None:
    """Real Vial image: DINOv3 CLS in CUDA bf16 is finite and within cosine 0.99 of CPU fp32.

    dinov3_l overflows to NaN in fp16, which is why CUDA runs in bf16.
    """
    image = load_rgb(sorted((Paths().data / "vial" / "train/good").glob("*.png"))[0])
    bf16 = load_encoder(name, "cuda").encode(image).cls.float().cpu()
    fp32 = load_encoder(name, "cpu").encode(image).cls
    assert bool(torch.isfinite(bf16).all())
    assert torch.nn.functional.cosine_similarity(bf16, fp32, dim=0).item() >= 0.99
