"""anomalib PatchCore and EfficientAD-S: model settings, fit and raw anomaly maps.

`build_model` holds the Track A settings (256x256 input without
center crop, no post-processing, so maps stay raw). `fit_predict_anomalib`
fits one model on a scenario's `train` images and returns its raw maps for
the `validation` and `test_public` images, at model resolution.
"""

from pathlib import Path

import numpy as np
import torch
from anomalib.data import ImageBatch, MVTecAD2
from anomalib.engine import Engine
from anomalib.models import AnomalibModule, EfficientAd, Patchcore
from numpy.typing import NDArray

from anometa.config import Paths, Scenario, TrackAConfig, resolve_device


def build_model(cfg: TrackAConfig) -> AnomalibModule:
    """Build the anomalib model a Track A config names, with the Track A settings (docs: tracks).

    Post-processing, evaluation and visualisation are disabled: the
    post-processor would min-max normalise maps with all-normal validation
    statistics, and metrics are computed by `anometa.metrics.pixel`.

    Args:
        cfg: A Track A config with `model` `"patchcore"` or `"efficientad_s"`.
            EfficientAD's Imagenette folder is `cfg.paths.data.parent /
            "imagenette"` (`data/imagenette` by default).

    Returns:
        The unfitted model.

    Raises:
        ValueError: If `cfg.model` is `"patch_distance"`, which has no
            anomalib model.
    """
    if cfg.model == "patchcore":
        return Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=0.01,
            num_neighbors=9,
            pre_processor=Patchcore.configure_pre_processor(image_size=cfg.image_size),
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
    if cfg.model == "efficientad_s":
        return EfficientAd(
            imagenet_dir=cfg.paths.data.parent / "imagenette",
            model_size="small",
            lr=1e-4,
            weight_decay=1e-5,
            pre_processor=EfficientAd.configure_pre_processor(image_size=cfg.image_size),
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
    raise ValueError(f"{cfg.model} is not an anomalib model")


def fit_predict_anomalib(
    cfg: TrackAConfig, scenario: Scenario, paths: Paths
) -> dict[str, NDArray[np.float32]]:
    """Fit an anomalib model on a scenario's `train` images and predict raw maps.

    PatchCore runs one epoch (it only fills its memory bank); EfficientAD-S
    trains for `cfg.max_steps` steps at batch size 1 and validates once, after
    the last step, which sets the map-normalisation quantiles from the final
    weights (Lightning's default would validate after every epoch, which
    costs time and nothing else). Torch is seeded with
    `cfg.seeds[0]` first. Lightning runs on `cfg.device` (resolved) and keeps
    its working files under `paths.cache / "anomalib"`.

    Args:
        cfg: A Track A config with an anomalib `model`.
        scenario: The scenario to fit and predict.
        paths: Run paths; the AD2 tree is read from `paths.data`.

    Returns:
        Image id (relative to the scenario folder, without extension, as in
        `index_scenario`) to raw anomaly map at model resolution, for every
        `validation` and `test_public` image.
    """
    seed = cfg.seeds[0]
    torch.manual_seed(seed)
    model = build_model(cfg)
    efficientad = cfg.model == "efficientad_s"
    dm = MVTecAD2(
        root=paths.data,
        category=scenario,
        train_batch_size=1 if efficientad else 32,
        eval_batch_size=32,
        num_workers=4,
        seed=seed,
    )
    dm.setup()
    device = resolve_device(cfg.device)
    accelerator = "gpu" if device == "cuda" else device
    root_dir = paths.cache / "anomalib"
    engine = (
        Engine(
            max_steps=cfg.max_steps,
            accelerator=accelerator,
            devices=1,
            default_root_dir=root_dir,
            # anomalib's max-steps progress bar holds a local class the checkpoint can't pickle.
            enable_progress_bar=False,
            # Validate once, after the last step: each pass only refreshes the map quantiles.
            check_val_every_n_epoch=None,
            val_check_interval=cfg.max_steps,
            num_sanity_val_steps=0,
        )
        if efficientad
        else Engine(max_epochs=1, accelerator=accelerator, devices=1, default_root_dir=root_dir)
    )
    engine.fit(model, datamodule=dm)

    scenario_dir = (paths.data / scenario).resolve()
    maps: dict[str, NDArray[np.float32]] = {}
    for loader in (dm.val_dataloader(), dm.test_dataloader()):
        for batch in engine.predict(model, dataloaders=loader) or []:
            assert isinstance(batch, ImageBatch)
            assert batch.anomaly_map is not None
            assert batch.image_path is not None
            arrays: NDArray[np.float32] = batch.anomaly_map.detach().cpu().float().numpy()
            for image_path, array in zip(batch.image_path, arrays, strict=True):
                image_id = (
                    Path(image_path).resolve().relative_to(scenario_dir).with_suffix("").as_posix()
                )
                maps[image_id] = array.reshape(array.shape[-2:])
    return maps
