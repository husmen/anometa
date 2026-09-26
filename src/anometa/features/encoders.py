"""Frozen vision-foundation-model encoders: DINOv3 (ViT-S/L) and SigLIP2.

`load_encoder` returns an `Encoder` that turns an HxWx3 uint8 image into
`Encoded` features (a pooled `cls` vector and an `(h, w, dim)` patch grid),
backed by pinned Hugging Face Hub revisions (see PLAN_1 § Licences and
access). DINOv3 loads through `transformers` by default (`DINOv3ViTModel`,
the gated `facebook/dinov3-*` repositories) and falls back to timm's ungated
re-upload of the same weights on a gated-repository error. SigLIP2 exists
only in `transformers` (`Siglip2VisionModel`, ungated).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import torch
from huggingface_hub.errors import GatedRepoError
from numpy.typing import NDArray
from torchvision.transforms import v2

from anometa.config import EncoderName, resolve_device

logger = logging.getLogger(__name__)

_Dinov3Name = Literal["dinov3_s", "dinov3_l"]
_Device = Literal["cuda", "mps", "cpu"]

HF_IDS: dict[EncoderName, str] = {
    "dinov3_s": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "siglip2": "google/siglip2-so400m-patch16-naflex",
}
"""Hugging Face Hub repository id per encoder, transformers backend."""

TIMM_IDS: dict[_Dinov3Name, str] = {
    "dinov3_s": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_l": "vit_large_patch16_dinov3.lvd1689m",
}
"""timm model name per encoder, ungated fallback backend (DINOv3 only)."""

HF_REVISIONS: dict[EncoderName, str] = {
    "dinov3_s": "114c1379950215c8b35dfcd4e90a5c251dde0d32",
    "dinov3_l": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    "siglip2": "cc24074f717b612951c2dead130904ab9b65a81e",
}
"""Pinned commit per transformers repository (PLAN_1 § Licences and access)."""

TIMM_REVISIONS: dict[_Dinov3Name, str] = {
    "dinov3_s": "3bf4720a82ec2066db88137180ff1f83a675cef0",
    "dinov3_l": "30c1109559f65dea34316b0d4842d35c5771fe11",
}
"""Pinned commit per timm fallback repository."""

DIMS: dict[EncoderName, int] = {"dinov3_s": 384, "dinov3_l": 1024, "siglip2": 1152}
"""Output embedding width per encoder."""

RESOLUTION_TAGS: dict[EncoderName, str] = {
    "dinov3_s": "r512",
    "dinov3_l": "r512",
    "siglip2": "p1024",
}
"""Feature-cache tag per encoder, before any backend suffix."""

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Encoded:
    """One image's frozen-encoder features, on the encoder's device.

    Attributes:
        cls: Pooled embedding, shape `(dim,)`.
        patches: Per-patch embeddings, shape `(h, w, dim)`.
    """

    cls: torch.Tensor
    patches: torch.Tensor


class Encoder(Protocol):
    """A frozen vision encoder that turns an image into `Encoded` features."""

    name: str
    backend: Literal["transformers", "timm"]
    dim: int
    resolution_tag: str
    revisions: dict[str, str]

    def encode(self, image: NDArray[np.uint8]) -> Encoded:
        """Encode one HxWx3 uint8 image.

        Args:
            image: HxWx3 uint8 array.

        Returns:
            The image's pooled and per-patch features.
        """
        ...


def feature_tag(name: EncoderName, backend: Literal["transformers", "timm"]) -> str:
    """Build the feature-cache tag for an encoder and backend.

    Args:
        name: Encoder name.
        backend: Loading backend.

    Returns:
        `RESOLUTION_TAGS[name]`, with a `-timm` suffix for the timm backend.
    """
    tag = RESOLUTION_TAGS[name]
    return f"{tag}-timm" if backend == "timm" else tag


def dinov3_size(h: int, w: int, short: int = 512, patch: int = 16) -> tuple[int, int]:
    """Resize an HxW image so its short side is `short`, kept a patch multiple.

    Args:
        h: Source height.
        w: Source width.
        short: Target short-side length.
        patch: Patch size; both output sides are rounded to a multiple of it.

    Returns:
        `(new_h, new_w)`, each a multiple of `patch`.
    """
    scale = short / min(h, w)
    return round(h * scale / patch) * patch, round(w * scale / patch) * patch


def split_dinov3_tokens(
    hidden: torch.Tensor, num_register_tokens: int, grid: tuple[int, int]
) -> Encoded:
    """Split a DINOv3 token sequence into its CLS token and patch grid.

    Args:
        hidden: `(1, 1 + num_register_tokens + h*w, dim)` last hidden state.
        num_register_tokens: Register tokens following the CLS token.
        grid: `(h, w)` patch grid shape.

    Returns:
        `Encoded` with the CLS token and the patch tokens reshaped to `(h, w, dim)`.
    """
    h, w = grid
    tokens = hidden[0]
    cls = tokens[0]
    patches = tokens[1 + num_register_tokens :].reshape(h, w, -1)
    return Encoded(cls=cls, patches=patches)


def split_siglip2_tokens(
    hidden: torch.Tensor, pooled: torch.Tensor, spatial_shape: tuple[int, int]
) -> Encoded:
    """Split a padded SigLIP2 NaFlex token sequence into CLS and patch grid.

    Args:
        hidden: `(n, dim)` token sequence, real patches first, padding after.
        pooled: `(dim,)` pooled embedding, used as the CLS slot.
        spatial_shape: `(h, w)` patch grid shape; the first `h*w` tokens are real.

    Returns:
        `Encoded` with `pooled` as `cls` and the first `h*w` tokens reshaped to `(h, w, dim)`.
    """
    h, w = spatial_shape
    patches = hidden[: h * w].reshape(h, w, -1)
    return Encoded(cls=pooled, patches=patches)


def _dinov3_pixel_values(image: NDArray[np.uint8]) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize and normalise an image for DINOv3, shared by both backends.

    Args:
        image: HxWx3 uint8 array.

    Returns:
        `(pixel_values, grid)`: a `(1, 3, h, w)` batch and the resulting
        `(h // 16, w // 16)` patch grid shape.
    """
    h, w, _ = image.shape
    new_h, new_w = dinov3_size(h, w)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((new_h, new_w), antialias=True),
            v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
    pixel_values: torch.Tensor = transform(image).unsqueeze(0)
    return pixel_values, (new_h // 16, new_w // 16)


@dataclass
class _LoadedEncoder:
    """Concrete `Encoder` wrapping a closure that runs the loaded model.

    The closure captures its model, processor and device, so this one class
    serves every backend: only the private `_load_*` functions differ.
    """

    name: str
    backend: Literal["transformers", "timm"]
    dim: int
    resolution_tag: str
    revisions: dict[str, str]
    _encode: Callable[[NDArray[np.uint8]], Encoded]

    def encode(self, image: NDArray[np.uint8]) -> Encoded:
        """Encode one HxWx3 uint8 image.

        Args:
            image: HxWx3 uint8 array.

        Returns:
            The image's pooled and per-patch features.
        """
        return self._encode(image)


def _load_transformers_dinov3(name: _Dinov3Name, device: _Device) -> Encoder:
    """Load a DINOv3 encoder through `transformers`, from its gated repository.

    Args:
        name: `"dinov3_s"` or `"dinov3_l"`.
        device: Resolved device to run on.

    Returns:
        The loaded encoder.

    Raises:
        huggingface_hub.errors.GatedRepoError: If this host lacks access.
    """
    from transformers import AutoModel, DINOv3ViTModel

    repo_id = HF_IDS[name]
    revision = HF_REVISIONS[name]
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model: DINOv3ViTModel = AutoModel.from_pretrained(repo_id, revision=revision)
    model = model.to(device=device, dtype=dtype).eval()
    num_register_tokens: int = model.config.num_register_tokens

    def encode(image: NDArray[np.uint8]) -> Encoded:
        pixel_values, grid = _dinov3_pixel_values(image)
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        with torch.inference_mode():
            hidden: torch.Tensor = model(pixel_values).last_hidden_state
        return split_dinov3_tokens(hidden, num_register_tokens, grid)

    return _LoadedEncoder(
        name=name,
        backend="transformers",
        dim=DIMS[name],
        resolution_tag=feature_tag(name, "transformers"),
        revisions={repo_id: revision},
        _encode=encode,
    )


def _load_timm_dinov3(name: _Dinov3Name, device: _Device) -> Encoder:
    """Load a DINOv3 encoder through timm's ungated re-upload of the weights.

    Args:
        name: `"dinov3_s"` or `"dinov3_l"`.
        device: Resolved device to run on.

    Returns:
        The loaded encoder.

    Raises:
        TypeError: If timm builds something other than its `Eva` ViT.
    """
    import timm
    from timm.models import Eva

    timm_id = TIMM_IDS[name]
    revision = TIMM_REVISIONS[name]
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = timm.create_model(
        timm_id,
        pretrained=True,
        num_classes=0,
        pretrained_cfg_overlay={"hf_hub_id": f"timm/{timm_id}@{revision}"},
    )
    if not isinstance(model, Eva):
        raise TypeError(f"{timm_id} loaded as {type(model).__name__}, expected timm Eva")
    model = model.to(device=device, dtype=dtype).eval()
    num_register_tokens = model.num_prefix_tokens - 1

    def encode(image: NDArray[np.uint8]) -> Encoded:
        pixel_values, grid = _dinov3_pixel_values(image)
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        with torch.inference_mode():
            hidden = model.forward_features(pixel_values)
        return split_dinov3_tokens(hidden, num_register_tokens, grid)

    return _LoadedEncoder(
        name=name,
        backend="timm",
        dim=DIMS[name],
        resolution_tag=feature_tag(name, "timm"),
        revisions={f"timm/{timm_id}": revision},
        _encode=encode,
    )


def _load_siglip2(device: _Device) -> Encoder:
    """Load the SigLIP2 encoder through `transformers` (ungated).

    Args:
        device: Resolved device to run on.

    Returns:
        The loaded encoder.
    """
    from transformers import AutoImageProcessor, Siglip2VisionModel
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.models.siglip2.image_processing_siglip2 import Siglip2ImageProcessor

    repo_id = HF_IDS["siglip2"]
    revision = HF_REVISIONS["siglip2"]
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model: Siglip2VisionModel = Siglip2VisionModel.from_pretrained(repo_id, revision=revision)
    model = model.to(device=device, dtype=dtype).eval()
    processor: Siglip2ImageProcessor = AutoImageProcessor.from_pretrained(
        repo_id, revision=revision
    )

    def encode(image: NDArray[np.uint8]) -> Encoded:
        inputs = processor.preprocess(image, max_num_patches=1024, return_tensors="pt")
        pixel_values: torch.Tensor = inputs["pixel_values"].to(device=device, dtype=dtype)
        pixel_attention_mask: torch.Tensor = inputs["pixel_attention_mask"].to(device)
        spatial_shapes: torch.Tensor = inputs["spatial_shapes"].to(device)
        with torch.inference_mode():
            output: BaseModelOutputWithPooling = model(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
        h, w = (int(x) for x in spatial_shapes[0].tolist())
        hidden = output.last_hidden_state
        pooled = output.pooler_output
        if hidden is None or pooled is None:
            raise ValueError("Siglip2VisionModel returned no last_hidden_state or pooler_output")
        return split_siglip2_tokens(hidden[0], pooled[0], (h, w))

    return _LoadedEncoder(
        name="siglip2",
        backend="transformers",
        dim=DIMS["siglip2"],
        resolution_tag=RESOLUTION_TAGS["siglip2"],
        revisions={repo_id: revision},
        _encode=encode,
    )


def _is_gated(exc: BaseException) -> bool:
    """Tell whether an exception is, or was caused by, a gated-repository error.

    transformers wraps hub errors in `OSError(...) from e`, so the
    `__cause__` chain is walked rather than checking `exc` alone.

    Args:
        exc: The exception raised while loading from the Hub.

    Returns:
        Whether `exc` or any exception in its `__cause__` chain is a
        `GatedRepoError`.
    """
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, GatedRepoError):
            return True
        cause = cause.__cause__
    return False


def load_encoder(
    name: EncoderName,
    device: str = "auto",
    backend: Literal["auto", "transformers", "timm"] = "auto",
) -> Encoder:
    """Load a frozen encoder, resolving its device and backend.

    Args:
        name: Encoder to load.
        device: `"auto"`, `"cuda"`, `"mps"` or `"cpu"` (see `resolve_device`).
        backend: `"transformers"`, `"timm"`, or `"auto"` (transformers, falling
            back to timm on a gated-repository error, raised directly or as
            the cause of an `OSError`; DINOv3 only).

    Returns:
        The loaded encoder.

    Raises:
        ValueError: If `backend="timm"` is requested for `"siglip2"`, which
            has no timm backend.
    """
    resolved_device = resolve_device(device)
    if name == "siglip2":
        if backend == "timm":
            raise ValueError("siglip2 has no timm backend")
        return _load_siglip2(resolved_device)
    if backend == "timm":
        return _load_timm_dinov3(name, resolved_device)
    if backend == "transformers":
        return _load_transformers_dinov3(name, resolved_device)
    try:
        return _load_transformers_dinov3(name, resolved_device)
    except OSError as exc:  # GatedRepoError subclasses OSError
        if not _is_gated(exc):
            raise
        logger.warning("gated repository %s; falling back to timm: %s", HF_IDS[name], exc)
        return _load_timm_dinov3(name, resolved_device)
