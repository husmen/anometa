"""Feature extraction and cache: one `.npz` per (encoder, scenario, backend).

`extract_scenario` runs a loaded `Encoder` over every image of an AD2
scenario, builds a leave-one-image-out `PatchBank` from its `train` images
and scores every image's patches against that bank, then writes the result
and its provenance (encoder, backend, revision, device, dtype, git commit)
to `cache_path`. `load_features` reads a cache back as `Features`; Track B,
the search layer and the demo all read only this cache, never the encoder.
"""

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from anometa.artifacts import git_info
from anometa.config import EncoderName, Paths, Scenario
from anometa.data.ad2 import index_scenario, load_rgb
from anometa.data.download import read_sha256sums
from anometa.features.encoders import Encoded, Encoder, feature_tag
from anometa.features.novelty import build_bank, novelty_stats, patch_distances

_ENCODER_NAMES: tuple[EncoderName, ...] = ("dinov3_s", "dinov3_l", "siglip2")
PROVENANCE_KEYS: tuple[str, ...] = (
    "encoder",
    "backend",
    "revision",
    "device",
    "dtype",
    "git_commit",
)
"""Keys of `Features.provenance`, each stored as a `provenance_<key>` scalar."""


def _encoder_name(name: str) -> EncoderName:
    """Narrow an `Encoder.name` string to the closed `EncoderName` literal.

    `Encoder.name` is a plain `str` in the protocol, but every concrete
    encoder sets it to one of `EncoderName`'s values; this checks that at
    runtime instead of casting past the static mismatch.

    Args:
        name: An encoder's `.name` value.

    Returns:
        `name`, narrowed to `EncoderName`.

    Raises:
        ValueError: If `name` isn't one of `EncoderName`'s values.
    """
    for candidate in _ENCODER_NAMES:
        if candidate == name:
            return candidate
    raise ValueError(f"unknown encoder name: {name!r}")


def cache_path(
    encoder_name: EncoderName,
    scenario: Scenario,
    paths: Paths,
    backend: Literal["transformers", "timm"] = "transformers",
) -> Path:
    """Build a scenario's feature-cache file path for one encoder and backend.

    Args:
        encoder_name: Encoder whose features are cached.
        scenario: Scenario being cached.
        paths: Run paths; the pinned archive checksum is read from
            `paths.configs / "data" / "sha256sums.txt"`.
        backend: Encoder loading backend, folded into the cache tag by
            `feature_tag`.

    Returns:
        `paths.cache / "features" / f"{encoder_name}-{tag}" /
        f"{scenario}-{sha[:12]}.npz"`, where `tag` is `feature_tag(encoder_name,
        backend)` and `sha` is the scenario archive's pinned sha256 digest, or
        `"unpinned"` when it isn't pinned yet.
    """
    sums_path = paths.configs / "data" / "sha256sums.txt"
    sha = "unpinned"
    if sums_path.is_file():
        sha = read_sha256sums(sums_path).get(f"{scenario}.tar.gz", "unpinned")
    tag = feature_tag(encoder_name, backend)
    return paths.cache / "features" / f"{encoder_name}-{tag}" / f"{scenario}-{sha[:12]}.npz"


@dataclass(frozen=True)
class Features:
    """One scenario's cached per-image encoder features.

    Attributes:
        image_id: Image ids, in cache row order, shape `(N,)`.
        cls: Pooled embeddings, shape `(N, D)`.
        mean_patch: Mean patch embeddings, shape `(N, D)`.
        novelty: `(max, mean)` leave-one-out patch novelty, shape `(N, 2)`.
        distmap: Per-patch nearest-neighbour distance maps, shape `(N, h, w)`.
        provenance: How the cache was produced, keyed by `PROVENANCE_KEYS`:
            encoder name, backend, `repo@revision` pins, device, dtype and the
            anometa git commit; `"unknown"` for a value that wasn't recorded.
    """

    image_id: NDArray[np.str_]
    cls: NDArray[np.float32]
    mean_patch: NDArray[np.float32]
    novelty: NDArray[np.float32]
    distmap: NDArray[np.float16]
    provenance: dict[str, str]

    def rows(self, ids: Sequence[str]) -> NDArray[np.int64]:
        """Look up row positions for a sequence of image ids.

        Args:
            ids: Image ids to look up.

        Returns:
            Integer row positions into every field, one per id in `ids`, in
            the same order as `ids`.

        Raises:
            KeyError: If an id in `ids` isn't in `image_id`.
        """
        positions = {image_id: i for i, image_id in enumerate(self.image_id)}
        return np.array([positions[image_id] for image_id in ids], dtype=np.int64)


def _encode_finite(encoder: Encoder, image_id: str, path: str) -> Encoded:
    """Encode one image and reject non-finite features (e.g. fp16 overflow).

    Args:
        encoder: The loaded encoder to run.
        image_id: The image's id, named in the error.
        path: The image file to load.

    Returns:
        The image's features.

    Raises:
        ValueError: If `cls` or `patches` holds a NaN or infinity.
    """
    enc = encoder.encode(load_rgb(Path(path)))
    if not (bool(torch.isfinite(enc.cls).all()) and bool(torch.isfinite(enc.patches).all())):
        raise ValueError(f"{image_id}: encoder produced non-finite features")
    return enc


def extract_scenario(encoder: Encoder, scenario: Scenario, paths: Paths) -> Path:
    """Encode a scenario's images and cache their features.

    Encodes `train` images first and builds a `PatchBank` from their patch
    grids alone, on the encoder's device. A `train` image's novelty and
    distance map are then scored with its own patches left out of the bank
    (leave-one-out); `validation` and `test_public` images are scored
    against the full `train` bank. A no-op when the destination cache file
    already exists.

    Args:
        encoder: The loaded encoder to run.
        scenario: The scenario to extract.
        paths: Run paths; images are read from `paths.data`, the cache is
            written under `paths.cache`.

    Returns:
        The cache file path (`cache_path(encoder.name, scenario, paths,
        encoder.backend)`), whether it was just written or already existed.

    Raises:
        ValueError: If images in `scenario` encode to different patch grid
            shapes, or an image encodes to non-finite values.
    """
    cache = cache_path(_encoder_name(encoder.name), scenario, paths, encoder.backend)
    if cache.exists():
        return cache

    index = index_scenario(paths.data, scenario)
    train = index[index["source"] == "train"]
    rest = index[index["source"] != "train"]

    encoded: dict[str, Encoded] = {}
    for image_id, path in zip(train["image_id"], train["path"], strict=True):
        encoded[image_id] = _encode_finite(encoder, image_id, path)
    train_ids: list[str] = list(train["image_id"])
    bank = build_bank([encoded[i].patches for i in train_ids], train_ids)

    for image_id, path in zip(rest["image_id"], rest["path"], strict=True):
        encoded[image_id] = _encode_finite(encoder, image_id, path)

    grids: set[tuple[int, int]] = {
        (encoded[i].patches.shape[0], encoded[i].patches.shape[1]) for i in index["image_id"]
    }
    if len(grids) > 1:
        raise ValueError(f"{scenario}: inconsistent patch grid shapes {sorted(grids)}")
    h, w = next(iter(grids))
    dim = encoder.dim
    n = len(index)

    cls = np.empty((n, dim), dtype=np.float32)
    mean_patch = np.empty((n, dim), dtype=np.float32)
    novelty = np.empty((n, 2), dtype=np.float32)
    distmap = np.empty((n, h, w), dtype=np.float16)

    train_ids_set = set(train_ids)
    for row, image_id in enumerate(index["image_id"]):
        enc = encoded[image_id]
        exclude = image_id if image_id in train_ids_set else None
        dmap = patch_distances(enc, bank, exclude=exclude)
        cls[row] = enc.cls.detach().float().cpu().numpy()
        mean_patch[row] = enc.patches.reshape(-1, dim).float().mean(dim=0).detach().cpu().numpy()
        novelty[row] = novelty_stats(dmap.reshape(-1))
        distmap[row] = dmap.detach().cpu().numpy()

    first = encoded[train_ids[0]].cls
    revisions = ",".join(f"{repo}@{rev}" for repo, rev in sorted(encoder.revisions.items()))
    features = Features(
        image_id=np.asarray(index["image_id"], dtype=str),
        cls=cls,
        mean_patch=mean_patch,
        novelty=novelty,
        distmap=distmap,
        provenance={
            "encoder": encoder.name,
            "backend": encoder.backend,
            "revision": revisions or "unknown",
            "device": first.device.type,
            "dtype": str(first.dtype).removeprefix("torch."),
            "git_commit": git_info()[0] or "unknown",
        },
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=cache.parent, suffix=".npz")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(
            tmp_path,
            image_id=features.image_id,
            cls=features.cls,
            mean_patch=features.mean_patch,
            novelty=features.novelty,
            distmap=features.distmap,
            allow_pickle=False,
            **{f"provenance_{k}": np.str_(v) for k, v in features.provenance.items()},
        )
        os.replace(tmp_path, cache)
    except BaseException:
        # A partial write must never sit at `cache`'s path (see cache.exists() above).
        tmp_path.unlink(missing_ok=True)
        raise
    return cache


def load_features(
    encoder_name: EncoderName,
    scenario: Scenario,
    paths: Paths,
    backend: Literal["transformers", "timm"] = "transformers",
) -> Features:
    """Load a scenario's cached features for one encoder and backend.

    Args:
        encoder_name: Encoder whose cached features to load.
        scenario: Scenario to load.
        paths: Run paths; see `cache_path`.
        backend: Encoder loading backend that produced the cache.

    Returns:
        The cached `Features`; caches written before provenance was
        recorded load with every `provenance` value `"unknown"`.

    Raises:
        FileNotFoundError: If no cache file exists for `backend`. If the
            other backend's cache exists instead, the message names it as
            the `encoder_backend` value to use.
    """
    path = cache_path(encoder_name, scenario, paths, backend)
    if not path.is_file():
        other: Literal["transformers", "timm"] = (
            "timm" if backend == "transformers" else "transformers"
        )
        if cache_path(encoder_name, scenario, paths, other).is_file():
            raise FileNotFoundError(
                f"{path} not found; found a {other} cache instead, set encoder_backend={other!r}"
            )
        raise FileNotFoundError(f"{path} not found; run `anometa extract` first")
    with np.load(path) as data:
        return Features(
            image_id=data["image_id"],
            cls=data["cls"],
            mean_patch=data["mean_patch"],
            novelty=data["novelty"],
            distmap=data["distmap"],
            provenance={
                k: str(data[f"provenance_{k}"]) if f"provenance_{k}" in data else "unknown"
                for k in PROVENANCE_KEYS
            },
        )
