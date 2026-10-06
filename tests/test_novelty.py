"""Tests for `anometa.features.novelty`: patch bank, novelty and distance maps."""

import torch

from anometa.features.novelty import PatchBank, build_bank, nn_distances, novelty_stats


def make_bank(seed: int = 0) -> tuple[list[torch.Tensor], PatchBank]:
    """Build a 3-image, 4x8-dim fake patch bank.

    Args:
        seed: Seed for the random patch grids.

    Returns:
        The three `(3, 4, 8)` source patch grids and the `PatchBank` built
        from them under image ids `["A", "B", "C"]`.
    """
    g = torch.Generator().manual_seed(seed)
    patches = [torch.randn(3, 4, 8, generator=g) for _ in range(3)]
    return patches, build_bank(patches, ["A", "B", "C"])


def test_nn_distances_matches_cdist() -> None:
    """Chunked distances equal torch.cdist minima."""
    _patches, bank = make_bank()
    q = torch.randn(50, 8)
    ref = torch.cdist(q, bank.feats.float()).min(1).values
    assert torch.allclose(nn_distances(q, bank, chunk=7), ref, atol=1e-2)


def test_train_novelty_is_leave_one_out() -> None:
    """A bank image scores zero against itself, and exactly its distance to the others with LOO."""
    patches, bank = make_bank()
    qa = patches[0].reshape(-1, 8)
    assert torch.allclose(nn_distances(qa, bank), torch.zeros(12), atol=1e-2)
    others = torch.cat([p.reshape(-1, 8) for p in patches[1:]]).half().float()
    ref = torch.cdist(qa.half().float(), others).min(1).values
    assert torch.allclose(nn_distances(qa, bank, exclude="A"), ref, atol=1e-2)


def test_novelty_stats() -> None:
    """Max and top-1% mean of 200 distances."""
    assert novelty_stats(torch.arange(1, 201, dtype=torch.float32)) == (200.0, 199.5)
