# Anometa explainer: narration

One block per scene, read in order. `build.py` turns each block into one audio clip; the clip's length sets the scene's length in `scenes.py`. Numbers come from `reports/lock/results.md` and the experiment log (`docs/experiment-log.md`).

## question

Factories need to catch defects, but defects are rare. Often you have a handful of labelled examples, sometimes just one. Can a model learn to spot new defects from that little? Anometa puts TabPFN-3.5, a foundation model for tables, to this test.

## data

The test bed is MVTec AD 2: eight industrial scenarios, from vials and walnuts to sheet metal and fabric. The training images show only good parts under regular light. The test images add defects, and lighting the model has never seen.

## protocol

To keep the results honest, every scenario's labelled test images are split by scene into two halves. All experiments and model choices use the dev half. The configuration is then frozen, and the lock half is evaluated exactly once.

## pipeline

Each image goes through a frozen DINOv3 vision transformer. Its global token and its average patch become a compact summary, compressed to sixteen numbers. Two more numbers measure how far the image's most unusual patches sit from their nearest normal patch. That gives one row of eighteen numbers per image.

## tabpfn

TabPFN reads a table instead of being trained. The context holds about three hundred normal rows and a few defect rows. In one forward pass, it predicts the probability that each new image is defective. There are no gradient updates and no tuning.

## results

On the lock half, with a single labelled defect, TabPFN reaches an AUROC of zero point seven six, against zero point six eight for logistic regression. With five defects it reaches zero point seven nine. Its probabilities are also far better calibrated: half the log loss of logistic regression. A good label-free novelty score is just as strong on ranking, though, so a few labels add little.

## maps

The same idea works at patch level. Each sixteen by sixteen pixel patch becomes a row, and TabPFN scores every patch to draw a defect map, shown here for one image per scenario. On the dev split, averaged over all eight scenarios, these maps beat the DINOv3 distance map. They also beat PatchCore, but PatchCore ran with default settings and was not tuned, so that gap is not conclusive.

## takeaways

TabPFN needs no training: fitting takes milliseconds, and one scenario runs end to end in about fourteen seconds on an RTX 3090. Results match within a fraction of a point on an Apple M4 Pro and on a plain CPU. Anometa is open source, with every run recorded and reproducible.
