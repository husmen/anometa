"""Pixel-level anomaly detection metrics for MVTec AD 2.

AU-PRO wraps the vendored MVTec AD reference implementation
(`anometa.metrics._pro_reference`); SegF1 and ClassF1 follow the pooled and
any-pixel definitions of the MVTec AD 2 paper (docs: metrics). Every metric here is pooled over
a full scenario, including its good images (which have all-zero masks and
drive the false-positive rate).
"""

from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from anometa.metrics._pro_reference import compute_pro, trapezoid
from anometa.metrics.image import auroc

# Anomaly maps and masks arrive at whatever float/int dtype the caller has on
# hand (float16 at full resolution in production, plain float64 in tests).
_Map = NDArray[np.float16] | NDArray[np.float32] | NDArray[np.float64]
_Mask = NDArray[np.int64] | NDArray[np.float64]
_Labels = NDArray[np.int64] | NDArray[np.float64]


def au_pros(maps: Sequence[_Map], masks: Sequence[_Mask], limits: Sequence[float]) -> list[float]:
    """Compute the area under the PRO curve at several false-positive-rate limits.

    The PRO curve, the expensive part, is computed once and integrated at
    every limit.

    Args:
        maps: Anomaly maps, one per image, pooled over one scenario.
        masks: Ground-truth masks, one per image, same shape as the
            corresponding map; 0 anomaly-free, nonzero anomalous.
        limits: Integration limits on the global pixel false positive rate.

    Returns:
        AU-PRO per limit, each normalized to `[0, 1]` at its limit.
    """
    fprs, pros = compute_pro(list(maps), list(masks))
    return [float(trapezoid(fprs, pros, x_max=limit) / limit) for limit in limits]


def au_pro(maps: Sequence[_Map], masks: Sequence[_Mask], limit: float = 0.05) -> float:
    """Compute the area under the PRO curve up to a false-positive-rate limit.

    Args:
        maps: Anomaly maps, one per image, pooled over one scenario.
        masks: Ground-truth masks, one per image, same shape as the
            corresponding map; 0 anomaly-free, nonzero anomalous.
        limit: Integration limit on the global pixel false positive rate.

    Returns:
        AU-PRO, normalized to `[0, 1]` at `limit`.
    """
    return au_pros(maps, masks, [limit])[0]


def upsample(map_: _Map, size_hw: tuple[int, int]) -> NDArray[np.float16]:
    """Bilinearly upsample a 2D anomaly map to `size_hw`.

    Uses `torch.nn.functional.interpolate` with `align_corners=False`, the
    resize the MVTec AD 2 server reports using (docs: metrics).

    Args:
        map_: Anomaly map, shape `(h, w)`.
        size_hw: Target `(height, width)`.

    Returns:
        Upsampled map, shape `size_hw`, dtype float16.
    """
    tensor = torch.from_numpy(np.asarray(map_, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=size_hw, mode="bilinear", align_corners=False)
    result: NDArray[np.float16] = resized[0, 0].numpy().astype(np.float16)
    return result


def validation_threshold(val_maps: Iterable[_Map]) -> float:
    """Compute the SegF1/ClassF1 threshold from defect-free validation maps.

    Threshold is mean + 3 std over every pixel of every map, accumulated in
    float64 (the MVTec AD 2 evaluation checker's rule; docs: metrics).

    Args:
        val_maps: Anomaly maps of defect-free validation images.

    Returns:
        The scalar threshold.
    """
    pixels = np.concatenate([np.asarray(m, dtype=np.float64).ravel() for m in val_maps])
    return float(pixels.mean() + 3.0 * pixels.std())


def seg_f1(maps: Sequence[_Map], masks: Sequence[_Mask], thr: float) -> float:
    """Compute the pooled pixel-level F1 score across a set of images.

    Ground truth is `mask > 0`; predictions are `map > thr`. Pixel counts are
    pooled across every image before precision/recall are computed.

    Args:
        maps: Anomaly maps, one per image.
        masks: Ground-truth masks, one per image, same shape as `maps`.
        thr: Score threshold, from `validation_threshold`.

    Returns:
        Pooled pixel F1 score.
    """
    tp = fp = fn = 0
    for map_, mask in zip(maps, masks, strict=True):
        pred = np.asarray(map_) > thr
        gt = np.asarray(mask) > 0
        tp += int(np.sum(pred & gt))
        fp += int(np.sum(pred & ~gt))
        fn += int(np.sum(~pred & gt))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def class_f1(maps: Sequence[_Map], labels: _Labels, thr: float) -> float:
    """Compute the image-level F1 score from each image's max pixel score.

    An image is predicted anomalous iff `max(map) > thr`; anomalous is the
    positive class.

    Args:
        maps: Anomaly maps, one per image.
        labels: Ground-truth image labels in `{0, 1}`, shape `(n,)`.
        thr: Score threshold, from `validation_threshold`.

    Returns:
        F1 score with anomalous as the positive class, or 0.0 when there are
        no predicted and no true positives.
    """
    pred = np.array([bool(np.max(m) > thr) for m in maps])
    true = np.asarray(labels) > 0
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def pixel_metrics(
    maps: Sequence[_Map],
    masks: Sequence[_Mask],
    labels: _Labels,
    thr: float,
) -> dict[str, float]:
    """Compute the MVTec AD 2 pixel-level metric set for one scenario.

    The image score for `auroc` is each map's maximum pixel value. Maps are
    expected at full resolution; good images must supply an all-zero mask.

    Args:
        maps: Anomaly maps, one per image, full resolution.
        masks: Ground-truth masks, one per image, same shape as `maps`.
        labels: Ground-truth image labels in `{0, 1}`, shape `(n,)`.
        thr: Score threshold, from `validation_threshold`.

    Returns:
        Dict with `au_pro_005`, `au_pro_030`, `seg_f1`, `class_f1` and
        `auroc`.
    """
    image_scores = np.array([np.max(m) for m in maps], dtype=np.float64)
    au_pro_005, au_pro_030 = au_pros(maps, masks, [0.05, 0.30])
    return {
        "au_pro_005": au_pro_005,
        "au_pro_030": au_pro_030,
        "seg_f1": seg_f1(maps, masks, thr),
        "class_f1": class_f1(maps, labels, thr),
        "auroc": auroc(np.asarray(labels, dtype=np.float64), image_scores),
    }
