# UNet Pet Segmentation — Portfolio Refactor Plan

**Audience:** engineering hiring at Nvidia (Isaac/Robotics), Wayve, Figure, 1X, Agile, PAL, Boston Dynamics, BMW/Bosch R&D. Target: summer 2027 internships.

**Anchor:** match the structure of `~/projects/resnet-cifar10` (same author's higher-bar sibling) — `src/unet_pet_seg/` package, `configs/*.yaml`, `scripts/`, `tests/`, `pyproject.toml`, `Makefile`.

**Biggest credibility gaps today:** no augmentation, no qualitative viz, no training curves, no tests, 128px toy resolution, no AMP, no baseline comparison. mIoU 0.7422 is an unverifiable scalar.

---

## Phase 1 — Packaging & layout  `[mechanical · sonnet]`
Move flat files into the sibling's layout so the repo reads as a package, not a script dump.
- [x] Create `src/unet_pet_seg/{__init__,model,dataset,trainer,evaluator,logger,config}.py`
- [x] Move `train.py` / `evaluate.py` → `scripts/`; update imports
- [x] Add `pyproject.toml` (package, pinned deps, entry points), pin `requirements.txt`
- [x] Add `Makefile` (`install`, `train`, `eval`, `test`, `lint`, `viz`)
- [x] `.gitignore`: `data/`, `runs/`, `artifacts/`, `*.pth`

## Phase 2 — Config & CLI  `[structural · sonnet]`
Replace the global-constants `config.py` with yaml + dataclass; kills "tutorial" smell.
- [x] `configs/unet_base.yaml` + `unet_256_aug.yaml`; dataclass loader with type hints
- [x] `argparse` in `scripts/train.py`: `--config`, `--resume`, `--out-dir`, `--seed`
- [x] Full seeding helper (torch, numpy, python, `cudnn.deterministic`) — document non-determinism honestly

## Phase 3 — Training loop lift  `[structural · sonnet]`
Extract a reusable `Trainer`; add the engineering signals a reviewer scans for first.
- [x] `Trainer` class: fit/validate/checkpoint/resume, best-mIoU tracking, early stop
- [x] AMP (`torch.amp.autocast` + `GradScaler`) — cheap Nvidia-facing signal
- [x] `grad_clip`, `torch.compile` toggle, cosine LR option
- [x] TensorBoard logger: loss, per-class IoU, LR, grad-norm, sample pred grid every N epochs
- [x] Checkpoint `{epoch, model, opt, scaler, best_miou, config_hash}`

## Phase 4 — Data pipeline credibility  `[structural · sonnet]`
Current dataset has zero augmentation — the #1 tutorial tell for a segmentation repo.
- [ ] Albumentations (or `v2` transforms): hflip, affine, color jitter, RandomResizedCrop — applied identically to img+mask
- [ ] Bump default `IMAGE_SIZE` 128 → 256 (see Phase 7 decision)
- [ ] Dataset unit test: mask dtype/range `{0,1,2}`, alignment under aug

## Phase 5 — Tests  `[structural · sonnet]`
Mirror sibling's `tests/`: `test_model.py`, `test_dataset.py`, `test_training.py`.
- [ ] Model: forward shape, param count sanity, gradient flow
- [ ] Dataset: shapes, label range, deterministic seed
- [ ] Training: 1-step overfit-one-batch smoke test (catches silent bugs pre-run)

## Phase 6 — Results & visualization  `[content · opus]`
A segmentation repo without pictures is uncheckable. This is the biggest ROI visible change.
- [ ] `scripts/visualize.py`: save input/GT/pred/overlay grid for N test images → `artifacts/preds/`
- [ ] Parse TensorBoard logs → `artifacts/curves.png` (loss + per-class IoU)
- [ ] Baseline row: torchvision `fcn_resnet50` or `deeplabv3_resnet50` fine-tuned same schedule — makes 0.7422 interpretable
- [ ] `artifacts/` committed (small PNGs only), referenced from README

## Phase 7 — Scope & recipe decisions  `[decision · advisor]`
Call advisor once before Phase 8 re-train; budget is a GTX 1060.
- [ ] Final image size (256 vs 384), epochs, aug recipe, loss (CE vs CE+Dice)
- [ ] Which baseline(s) actually run — pick one, not three
- [ ] Stretch: ONNX export + latency note, Dockerfile, mixed-precision vs fp32 benchmark table
- [ ] Framing hook: one paragraph linking dense prediction → robotics perception (free-space / semantic occupancy)

## Phase 8 — Re-train & measure  `[mechanical · sonnet]` *(separate budget: ~4–8h wall)*
Do not fold into Phase 4. Own phase so the time cost is visible.
- [ ] Train base + aug config; log curves; save best checkpoint
- [ ] Final test mIoU (per-class + mean), latency ms/img on 1060, param count / FLOPs

## Phase 9 — README rewrite  `[content · opus]`
2-minute read for a hiring engineer. Results table with *numbers + pictures + baseline*, honest limits.
- [ ] Hook, arch diagram, dataset, training recipe, results table, qualitative grid, limitations, repro commands, robotics-framing paragraph
- [ ] Delete "Key Concepts" tutorial section

---

## Triage summary
| Phase          | Type       | Model   | Est.      |
| -------------- | ---------- | ------- | --------- |
| 1 Packaging    | mechanical | sonnet  | 1h        |
| 2 Config/CLI   | structural | sonnet  | 1h        |
| 3 Trainer+AMP  | structural | sonnet  | 2h        |
| 4 Data aug     | structural | sonnet  | 1h        |
| 5 Tests        | structural | sonnet  | 1h        |
| 6 Viz+baseline | content    | opus    | 2h        |
| 7 Decisions    | decision   | advisor | 20m       |
| 8 Re-train     | mechanical | sonnet  | 4–8h wall |
| 9 README       | content    | opus    | 1h        |
