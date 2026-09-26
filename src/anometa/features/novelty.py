"""Nearest-neighbour patch distances, novelty stats and distance maps.

`build_bank` flattens a scenario's `train`-normal patch grids into one
`PatchBank`. `nn_distances` scores query patches against it (optionally
leaving one bank image out, for leave-one-image-out train novelty; see
PLAN_1 § Review focus). `novelty_stats` reduces a distance vector to the two
scalars used as features, and `patch_distances` reshapes an image's own
distances back into its patch grid, for the Track A patch-distance map.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from anometa.features.encoders import Encoded


@dataclass(frozen=True)
class PatchBank:
    """A flattened bank of reference patch embeddings.

    Attributes:
        feats: All images' patch embeddings, concatenated, shape `(M, D)`,
            float16 (halved to keep the bank small; distances upcast per
            chunk, see `nn_distances`).
        offsets: Row offset of each image's block in `feats`; image `i`
            occupies `feats[offsets[i]:offsets[i + 1]]`. Length
            `len(image_ids) + 1`.
        image_ids: One id per bank image, in `feats` order.
    """

    feats: torch.Tensor
    offsets: list[int]
    image_ids: list[str]


def build_bank(patches: Sequence[torch.Tensor], image_ids: Sequence[str]) -> PatchBank:
    """Flatten per-image patch grids into one bank of patch embeddings.

    Args:
        patches: One `(h, w, D)` tensor per image; grids may differ in
            `h, w` but must share `D` and device.
        image_ids: One id per image, matching `patches` in order.

    Returns:
        A `PatchBank` with all patches concatenated, in the input device, and
        stored as float16.
    """
    flattened = [p.reshape(-1, p.shape[-1]) for p in patches]
    feats = torch.cat(flattened, dim=0).half()
    offsets = [0]
    for block in flattened:
        offsets.append(offsets[-1] + block.shape[0])
    return PatchBank(feats=feats, offsets=offsets, image_ids=list(image_ids))


def nn_distances(
    queries: torch.Tensor,
    bank: PatchBank,
    exclude: str | None = None,
    chunk: int = 65536,
) -> torch.Tensor:
    """Compute each query's Euclidean distance to its nearest bank row.

    Runs on whatever device `bank.feats` is on (CPU, MPS or CUDA), moving
    `queries` there if needed. Processes the bank in row chunks, each cast to
    float32 before the squared-distance formula: float16 overflows it, as
    patch token norms reach the hundreds.

    Args:
        queries: `(Q, D)` query patch embeddings.
        bank: The reference `PatchBank`.
        exclude: If given, an `image_ids` entry whose rows are left out of
            the bank for this call (leave-one-image-out train novelty).
        chunk: Bank rows processed per step.

    Returns:
        `(Q,)` float32 distances to the nearest (non-excluded) bank row.
    """
    feats = bank.feats
    if exclude is not None:
        idx = bank.image_ids.index(exclude)
        start, end = bank.offsets[idx], bank.offsets[idx + 1]
        feats = torch.cat([feats[:start], feats[end:]])
    queries = queries.to(device=feats.device, dtype=torch.float32)
    query_sq = queries.pow(2).sum(dim=1, keepdim=True)
    best = torch.full((queries.shape[0],), float("inf"), device=feats.device)
    for i in range(0, feats.shape[0], chunk):
        block = feats[i : i + chunk].float()
        block_sq = block.pow(2).sum(dim=1)
        sq_dist = (query_sq + block_sq.unsqueeze(0) - 2 * queries @ block.T).clamp_min(0)
        best = torch.minimum(best, sq_dist.min(dim=1).values)
    return best.sqrt()


def novelty_stats(d: torch.Tensor) -> tuple[float, float]:
    """Reduce a distance vector to the two patch-novelty scalars.

    Args:
        d: `(Q,)` distances, e.g. from `nn_distances`.

    Returns:
        `(max, mean)` of the top `ceil(0.01 * Q)` distances.
    """
    k = math.ceil(0.01 * d.shape[0])
    top = torch.topk(d, k).values
    return float(top.max()), float(top.mean())


def patch_distances(encoded: Encoded, bank: PatchBank, exclude: str | None = None) -> torch.Tensor:
    """Score one image's patch grid against a bank, as a distance map.

    Args:
        encoded: The image's encoder output; `encoded.patches` is `(h, w, D)`.
        bank: The reference `PatchBank`.
        exclude: Forwarded to `nn_distances` (leave-one-image-out).

    Returns:
        `(h, w)` float32 nearest-neighbour distance map.
    """
    h, w, d = encoded.patches.shape
    distances = nn_distances(encoded.patches.reshape(-1, d), bank, exclude=exclude)
    return distances.reshape(h, w)
