# Anometa

Research on whether TabPFN-3.5, applied to frozen vision-foundation-model features, is useful for few-shot industrial anomaly detection.

## Language

### Benchmark

**Scenario**:
One of the eight MVTec AD 2 object/product categories (e.g. Can, Walnuts). The unit at which models are trained and evaluated.
_Avoid_: category, class, object

**Scene**:
One physical setup in `test_public` (same object, same defect or lack of one, same position) captured under every lighting condition. Identified by label and the 3-digit filename index, e.g. `bad/007`. The unit of the dev/lock split.
_Avoid_: sample, instance, group

**Anomaly map**:
A per-pixel anomaly score image for one input image. Required for pixel-level metrics and for the official evaluation server.
_Avoid_: heatmap, segmentation

**Image score**:
A single anomaly score per image. For Track B it is the classifier's anomaly probability; for Track A it is the maximum of the anomaly map.
_Avoid_: image prediction, confidence

**Lighting condition**:
The illumination under which an image was captured: regular, or a shifted condition (under-exposed, over-exposed, extra light source). Training data is regular only.
_Avoid_: exposure, illumination setting

**Shifted lighting**:
Any lighting condition other than regular. Unseen by every model during training and few-shot sampling.
_Avoid_: unseen lighting, OOD lighting

**Robustness gap**:
A metric on regular-lit lock images minus the same metric on shifted-lit lock images.
_Avoid_: degradation, domain gap

### Splits

**Dev split**:
The half of a scenario's public test set used for few-shot sampling and all model selection.
_Avoid_: validation set, public test

**Lock split**:
The half of a scenario's public test set evaluated once, after the final configuration is frozen.
_Avoid_: test set, holdout

**Private test**:
MVTec's unlabelled test sets, scored only by the official evaluation server. Out of scope for Track B.
_Avoid_: test set

**Label budget**:
The number k of labelled anomalous images per scenario given to a Track B classifier. Normal images are not counted.
_Avoid_: shot count, sample size

### Tracks

**Track A**:
Unsupervised anomaly detection: models learn from defect-free images only and output anomaly maps.
_Avoid_: baseline track, classical AD

**Track B**:
Supervised few-shot adaptation: a classifier trained on frozen foundation-model features from labelled normal and anomalous images, outputting an image score.
_Avoid_: TabPFN track, supervised AD

**Track C**:
Mask-supervised few-shot patch classification: a classifier trained on frozen foundation-model patch features, labelled from the ground-truth masks of the k few-shot anomalies, outputting an anomaly map.
_Avoid_: patch-level Track B, supervised segmentation

**Patch label**:
The normal/anomalous label of one patch, derived from how much of it a ground-truth mask covers.
_Avoid_: pixel label, patch target

**Patch distance map**:
An anomaly map made of each patch's distance to the nearest normal patch of the same scenario. It learns from normal images only, so it belongs to Track A.
_Avoid_: DINOv3 PatchCore, novelty map

**Patch novelty score**:
An image-level feature summarising how far an image's patch features lie from the nearest normal patches of the same scenario (e.g. max, top-1% mean).
_Avoid_: PatchCore score, anomaly feature

**One-class control**:
A Track B reference that scores images using normal images only (label budget zero), to test whether labelled anomalies add value.
_Avoid_: unsupervised baseline, zero-shot
