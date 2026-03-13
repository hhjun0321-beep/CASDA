# CASDA Quick Start Guide

> **CASDA** — Context-Aware Data Augmentation Framework.
> Reproducing the full experiment pipeline on Google Colab.

This document provides a step-by-step guide for running the CASDA pipeline in a Google Colab environment. Adjust the shell variables at the top to match your local setup, then execute the scripts in order.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Global Variables](#global-variables)
- [Pipeline Overview](#pipeline-overview)
- [Step 0. Initial Setup](#step-0-initial-setup)
- [Stage A: Data Preprocessing (CPU, one-time)](#stage-a-data-preprocessing-cpu-one-time)
- [Stage B: ControlNet Training + Generation (GPU)](#stage-b-controlnet-training--generation-gpu)
- [Stage C: Post-processing + Quality Control (CPU)](#stage-c-post-processing--quality-control-cpu)
- [Stage D: Evaluation (GPU)](#stage-d-evaluation-gpu)

---

## Prerequisites

| Requirement | Details |
|---|---|
| Environment | Google Colab (GPU runtime recommended) |
| Dataset | [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/data) |
| Storage | Google Drive mounted at `/content/drive/MyDrive/` |
| GPU | T4 or higher (for ControlNet training and benchmark) |

---

## Global Variables

Define reusable path variables to avoid hard-coded paths across all steps.  
Run this **once** at the top of your Colab notebook.

```bash
# ── Project root ──
PROJ=/content/CASDA
SCRIPTS=${PROJ}/scripts
CONFIG=${PROJ}/configs/benchmark_experiment.yaml

# ── Google Drive data root ──
DRIVE_DATA=/content/drive/MyDrive/data/Severstal

# ── Source data ──
TRAIN_IMAGES=${DRIVE_DATA}/train_images
TRAIN_CSV=${DRIVE_DATA}/train.csv

# ── Stage A outputs ──
ROI_DIR=${DRIVE_DATA}/roi_patches
CN_DATASET=${DRIVE_DATA}/controlnet_dataset

# ── Stage B outputs ──
CN_TRAINING=${DRIVE_DATA}/controlnet_training
CN_VALIDATION=${DRIVE_DATA}/controlnet_validation
BEST_MODEL=${CN_TRAINING}/best_model
AUG_IMAGES=${DRIVE_DATA}/augmented_images

# ── Stage C outputs ──
AUG_DATASET=${DRIVE_DATA}/augmented_dataset
CASDA_COMPOSED=${AUG_DATASET}/casda_composed
COPYPASTE_DIR=${AUG_DATASET}/copypaste_baseline
CASDA_NO_BLEND=${AUG_DATASET}/casda_no_blend
BG_CACHE=${DRIVE_DATA}/cache/bg_types.json

# ── Stage D outputs ──
FID_RESULTS=${DRIVE_DATA}/fid_results
SPLIT_RESULTS=${DRIVE_DATA}/split_experiment
BENCHMARK_RESULTS=${DRIVE_DATA}/benchmark_results
YOLO_DATASETS=${DRIVE_DATA}/yolo_datasets
REFERENCE_RESULTS=${DRIVE_DATA}/casda/benchmark_results.json

# ── Local disk (Colab I/O optimization) ──
LOCAL_IMAGES=/content/dataset_local/train_images
```

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage A: Data Preprocessing (CPU, one-time)                              │
├──────────────────────────────────────────────────────────────────────────┤
│ Step 1.  extract_rois.py              — Extract ROI patches              │
│ Step 2.  prepare_controlnet_data.py   — Prepare ControlNet train data    │
├──────────────────────────────────────────────────────────────────────────┤
│ Stage B: ControlNet Training + Generation (GPU)                          │
├──────────────────────────────────────────────────────────────────────────┤
│ Step 3.  train_controlnet.py          — Train ControlNet                 │
│ Step 4.  run_validation_phases.py     — Validate ControlNet (optional)   │
│ Step 5.  test_controlnet.py           — Generate synthetic images        │
├──────────────────────────────────────────────────────────────────────────┤
│ Stage C: Post-processing + Quality Control (CPU)                         │
├──────────────────────────────────────────────────────────────────────────┤
│ Step 6.  compose_casda_images.py      — Poisson Blending composition     │
│ Step 7.  create_copypaste_baseline.py — CopyPaste baseline generation    │
│ Step 8.  compose_casda_images.py      — w/o Blending ablation variant    │
│ Step 9.  score_casda_quality.py       — Compute synthesis quality scores │
│ Step 10. validate_augmented_quality.py— Quality validation               │
├──────────────────────────────────────────────────────────────────────────┤
│ Stage D: Evaluation (GPU)                                                │
├──────────────────────────────────────────────────────────────────────────┤
│ Step 11. run_fid.py                   — FID evaluation                   │
│ Step 12. run_split_experiment.py      — Optimal split ratio search       │
│ Step 13. run_benchmark.py             — Model training + evaluation      │
│ Step 14. run_benchmark.py             — Ablation study (yolo_mfd)        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Step 0. Initial Setup

### Download Dataset

1. Download from https://www.kaggle.com/c/severstal-steel-defect-detection/data
2. Upload the zip file to Google Drive and unzip

### Clone Repository

```bash
!git clone https://github.com/hhjun0321-beep/CASDA.git
```

---

## Stage A: Data Preprocessing (CPU, one-time)

### Step 1. extract_rois.py — Extract ROI Patches

```bash
!python ${SCRIPTS}/extract_rois.py \
  --image_dir      ${TRAIN_IMAGES} \
  --train_csv      ${TRAIN_CSV} \
  --output_dir     ${ROI_DIR} \
  --roi_size 256 \
  --grid_size 64 \
  --min_suitability 0.5 \
  --num_workers 8
```

### Step 2. prepare_controlnet_data.py — Prepare ControlNet Training Data

```bash
!python ${SCRIPTS}/prepare_controlnet_data.py \
  --roi_metadata         ${ROI_DIR}/roi_metadata.csv \
  --train_images         ${TRAIN_IMAGES} \
  --train_csv            ${TRAIN_CSV} \
  --output_dir           ${CN_DATASET} \
  --per_class_cap 1200 \
  --rare_class_threshold 200 \
  --class_edge_override "4:0.05,0.0" \
  --skip_validation
```

---

## Stage B: ControlNet Training + Generation (GPU)

### Step 3. train_controlnet.py — Train ControlNet

```bash
!python ${SCRIPTS}/train_controlnet.py \
  --data_dir                        ${CN_DATASET} \
  --output_dir                      ${CN_TRAINING} \
  --pretrained_model_name_or_path   runwayml/stable-diffusion-v1-5 \
  --controlnet_model_name_or_path   lllyasviel/sd-controlnet-canny \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --mixed_precision fp16 \
  --num_train_epochs 20 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine \
  --lr_warmup_steps 50 \
  --controlnet_conditioning_scale 1.0 \
  --snr_gamma 5.0 \
  --early_stopping_patience 20 \
  --validation_steps 200 \
  --logging_steps 10 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 3 \
  --save_fp16 \
  --skip_save_pipeline \
  --seed 42
```

### Step 4. run_validation_phases.py — Validate ControlNet Model (Optional)

```bash
!python ${SCRIPTS}/run_validation_phases.py \
  --model_path          ${BEST_MODEL} \
  --jsonl_path          ${CN_DATASET}/train.jsonl \
  --image_root          ${CN_DATASET} \
  --roi_metadata_path   ${ROI_DIR}/roi_metadata.csv \
  --training_log_path   ${CN_TRAINING}/training_log.json \
  --output_base         ${CN_VALIDATION} \
  --phases 1 2 3 4 \
  --controlnet_conditioning_scale 0.7 \
  --resolution 512 \
  --workers 8
```

### Step 5. test_controlnet.py — Generate Synthetic Images

```bash
!python ${SCRIPTS}/test_controlnet.py \
  --model_path    ${BEST_MODEL} \
  --jsonl_path    ${CN_DATASET}/train.jsonl \
  --output_dir    ${AUG_IMAGES} \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --controlnet_conditioning_scale 0.7 \
  --num_images_per_class '{"1":2,"2":10,"3":1,"4":2}' \
  --resolution 512
```

---

## Stage C: Post-processing + Quality Control (CPU)

### Step 6. compose_casda_images.py — Poisson Blending Composition

```bash
!python ${SCRIPTS}/compose_casda_images.py \
  --generated-dir       ${AUG_IMAGES}/generated \
  --hint-dir            ${CN_DATASET}/hints \
  --metadata-csv        ${CN_DATASET}/packaged_roi_metadata.csv \
  --summary-json        ${AUG_IMAGES}/generation_summary.json \
  --clean-images-dir    ${TRAIN_IMAGES} \
  --train-csv           ${TRAIN_CSV} \
  --output-dir          ${CASDA_COMPOSED} \
  --workers 8 \
  --bg-cache            ${BG_CACHE} \
  --compositions-per-roi 5
```

### Step 7. create_copypaste_baseline.py — CopyPaste Baseline Generation

> Baseline comparison: paste original ROIs directly onto clean backgrounds without Poisson Blending.

```bash
!python ${SCRIPTS}/create_copypaste_baseline.py \
  --roi-dir            ${ROI_DIR} \
  --metadata-csv       ${ROI_DIR}/roi_metadata.csv \
  --clean-images-dir   ${TRAIN_IMAGES} \
  --train-csv          ${TRAIN_CSV} \
  --output-dir         ${COPYPASTE_DIR} \
  --workers 8 \
  --bg-cache           ${BG_CACHE}
```

### Step 8. compose_casda_images.py — w/o Blending Ablation Variant

> Ablation study: generate a no-blend variant to isolate the effect of Poisson Blending.

```bash
!python ${SCRIPTS}/compose_casda_images.py \
  --generated-dir       ${AUG_IMAGES}/generated \
  --hint-dir            ${CN_DATASET}/hints \
  --metadata-csv        ${CN_DATASET}/packaged_roi_metadata.csv \
  --summary-json        ${AUG_IMAGES}/generation_summary.json \
  --clean-images-dir    ${TRAIN_IMAGES} \
  --train-csv           ${TRAIN_CSV} \
  --output-dir          ${CASDA_NO_BLEND} \
  --no-blend \
  --workers 8 \
  --bg-cache            ${BG_CACHE}
```

### Step 9. score_casda_quality.py — Compute Synthesis Quality Scores

```bash
!python ${SCRIPTS}/score_casda_quality.py \
  --casda-dir  ${CASDA_COMPOSED} \
  --workers 10
```

### Step 10. validate_augmented_quality.py — Quality Validation

```bash
!python ${SCRIPTS}/validate_augmented_quality.py \
  --augmented_dir       ${CASDA_COMPOSED} \
  --min_quality_score   0.7 \
  --workers 12
```

---

## Stage D: Evaluation (GPU)

### Step 11. run_fid.py — FID Evaluation

```bash
!python ${SCRIPTS}/run_fid.py \
  --config             ${CONFIG} \
  --data-dir           ${TRAIN_IMAGES} \
  --csv                ${TRAIN_CSV} \
  --casda-dir          ${AUG_DATASET} \
  --casda-roi-dir      ${AUG_IMAGES}/generated \
  --roi-metadata-csv   ${ROI_DIR}/roi_metadata.csv \
  --output-dir         ${FID_RESULTS} \
  --workers 12
```

### Step 12. run_split_experiment.py — Optimal Split Ratio Search

> Find the best train/val/test split ratio using DeepLabV3+ as the reference model.

```bash
!python ${SCRIPTS}/run_split_experiment.py \
  --csv          ${TRAIN_CSV} \
  --data-dir     ${TRAIN_IMAGES} \
  --config       ${CONFIG} \
  --output-dir   ${SPLIT_RESULTS} \
  --ratios "60/20/20,70/15/15,80/10/10" \
  --epochs 100 \
  --seed 42
```

### Step 13. run_benchmark.py — Model Training + Evaluation

> Run all 3 models across 5 dataset groups: `baseline_raw`, `baseline_trad`, `casda_composed`, `casda_composed_pruning`, `copypaste`.

```bash
!python ${SCRIPTS}/run_benchmark.py \
  --config              ${CONFIG} \
  --data-dir            ${LOCAL_IMAGES} \
  --groups              baseline_raw baseline_trad casda_composed casda_composed_pruning copypaste \
  --casda-dir           ${AUG_DATASET} \
  --yolo-dir            ${YOLO_DATASETS} \
  --output-dir          ${BENCHMARK_RESULTS} \
  --no-fid \
  --reference-results   ${REFERENCE_RESULTS}
```

### Step 14. run_benchmark.py — Ablation Study (yolo_mfd only)

> Run `ablation_no_pruning` and `ablation_no_blending` groups with the yolo_mfd model only.

```bash
!python ${SCRIPTS}/run_benchmark.py \
  --config              ${CONFIG} \
  --data-dir            ${LOCAL_IMAGES} \
  --models              yolo_mfd \
  --groups              ablation_no_pruning ablation_no_blending \
  --casda-dir           ${AUG_DATASET} \
  --yolo-dir            ${YOLO_DATASETS} \
  --output-dir          ${BENCHMARK_RESULTS} \
  --no-fid \
  --reference-results   ${REFERENCE_RESULTS}
```
