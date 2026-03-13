# CASDA

**CASDA : Context-Aware Data Augmentation Framework **

## Abstract

Surface defect detection in steel manufacturing remains challenging due to severe class imbalance, limited defect samples, and insufficient background diversity in benchmark datasets. To address these limitations, this study proposes CASDA (Context-Aware Steel Defect Augmentation), a generative data augmentation framework that integrates geometric defect characterization with ControlNet-based conditional image synthesis to generate physically plausible defect images.

CASDA is a data augmentation pipeline that uses **ControlNet** (conditioned on multi-channel hint images) to generate physically plausible synthetic steel defect images. Generated defects are composited onto real defect-free backgrounds via **Poisson Blending**, then quality-filtered through a suitability scoring system. The pipeline is evaluated on the [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/data) dataset across three model architectures.

---

## Highlights

- **ControlNet-based defect synthesis** with multi-channel hints (R=defect shape, G=structure, B=texture) and hybrid text prompts
- **Poisson Blending composition** for seamless integration of generated 512x512 ROIs into real 1600x256 steel images
- **Quality-aware pruning** via suitability scoring (color consistency, artifact detection, sharpness)
- **Three benchmark models**: YOLO-MFD, EB-YOLOv8, DeepLabV3+
- **Seven dataset groups** including ablation studies isolating the contribution of each pipeline component
- **Statistical hypothesis testing** for rigorous validation of augmentation effects
- **FID evaluation** at both ROI and full-image levels with granular per-class and per-subtype breakdowns

---

## Repository Structure

```
CASDA/
├── configs/
│   └── benchmark_experiment.yaml   # Experiment configuration (models, datasets, metrics, hypotheses)
├── scripts/
│   ├── extract_rois.py             # Stage A — Extract ROI patches from Severstal images
│   ├── prepare_controlnet_data.py  # Stage A — Prepare multi-channel hints + train.jsonl
│   ├── train_controlnet.py         # Stage B — Train ControlNet on SD v1.5
│   ├── run_validation_phases.py    # Stage B — 4-phase ControlNet validation
│   ├── test_controlnet.py          # Stage B — Generate synthetic defect images
│   ├── compose_casda_images.py     # Stage C — Poisson Blending composition
│   ├── score_casda_quality.py      # Stage C — Suitability scoring
│   ├── validate_augmented_quality.py # Stage C — Quality validation
│   ├── create_copypaste_baseline.py  # Stage C — CopyPaste baseline
│   ├── run_fid.py                  # Stage D — FID evaluation
│   ├── run_benchmark.py            # Stage D — Model training + evaluation
│   ├── run_split_experiment.py     # Stage D — Optimal split ratio search
│   ├── analyze_benchmark_results.py  # Stage D — Tables, charts, statistical tests
│   └── ...                         # Visualization, verification, and utility scripts
├── src/
│   ├── models/
│   │   ├── yolo_mfd.py             # YOLO-MFD: YOLOv8 + Multi-scale Edge Feature Enhancement
│   │   ├── eb_yolov8.py            # EB-YOLOv8: YOLOv8 + BiFPN weighted fusion
│   │   └── deeplabv3plus.py        # DeepLabV3+: ResNet-101 + ASPP segmentation
│   ├── training/                   # Datasets, trainer, evaluators
│   ├── preprocessing/              # ROI extraction, ControlNet data preparation
│   ├── analysis/                   # FID, benchmark analysis, statistical testing
│   └── utils/                      # I/O, visualization, configuration helpers
├── QUICKSTART.md                   # Step-by-step pipeline execution guide
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Models

| Model | Type | Base | Key Enhancement |
|---|---|---|---|
| **YOLO-MFD** | Detection | YOLOv8s | MEFE module — multi-scale Sobel edge extraction with channel attention for micro-defect enhancement |
| **EB-YOLOv8** | Detection | YOLOv8s | BiFPN — bi-directional feature pyramid with fast normalized weighted fusion via depthwise separable convolutions |
| **DeepLabV3+** | Segmentation | ResNet-101 | ASPP (atrous rates 6/12/18) encoder-decoder with output stride 16 |

---

## Pipeline Overview

```
Stage A: Data Preprocessing (CPU)
  ├── Extract ROI patches with suitability scoring
  └── Prepare multi-channel ControlNet hints + hybrid text prompts
          │
Stage B: ControlNet Training + Generation (GPU)
  ├── Train ControlNet (SD v1.5 + sd-controlnet-canny)
  ├── Multi-phase validation (optional)
  └── Generate synthetic defect images
          │
Stage C: Post-processing + Quality Control (CPU)
  ├── Poisson Blending composition onto clean backgrounds
  ├── Suitability scoring (color / artifact / sharpness)
  ├── Quality-aware pruning (stratified top-k)
  └── CopyPaste baseline generation
          │
Stage D: Evaluation (GPU)
  ├── FID evaluation (ROI-level + full-image)
  ├── Optimal split ratio search
  ├── Benchmark: 3 models × 7 dataset groups
  └── Statistical hypothesis testing + result analysis
```

See [QUICKSTART.md](QUICKSTART.md) for the full step-by-step execution guide with CLI commands.

---

## Installation

```bash
git clone https://github.com/hhjun0321-beep/CASDA.git
cd CASDA
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA-capable GPU (T4 or higher recommended).

---

## Quick Start

The full pipeline is documented in **[QUICKSTART.md](QUICKSTART.md)**, designed for Google Colab with GPU runtime. The guide covers:

1. **Stage A** — Data preprocessing (ROI extraction, ControlNet data preparation)
2. **Stage B** — ControlNet training and synthetic image generation
3. **Stage C** — Poisson Blending composition, quality scoring, and baseline generation
4. **Stage D** — FID evaluation, benchmark training, and ablation studies

---

## Dataset Groups

All experiments use the [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/data) dataset (4 defect classes, 256x1600 images).

| Group | Description |
|---|---|
| `baseline_raw` | Original Severstal images only, no augmentation |
| `baseline_trad` | Original + traditional geometric augmentations (flip, rotation, scale, brightness, contrast) |
| `casda_composed` | Original + all Poisson Blending composed CASDA images |
| `casda_composed_pruning` | Original + quality-pruned CASDA images (stratified top-k) |
| `copypaste` | Original ROIs pasted onto clean backgrounds (no ControlNet, no blending) |
| `ablation_no_blending` | ControlNet ROIs directly pasted (no Poisson Blending) |
| `ablation_no_pruning` | No suitability filtering, quantity limit only |

---

## Evaluation Metrics

**Detection:** mAP@0.5, per-class AP, precision-recall curves

**Segmentation:** Dice coefficient (overall + per-class), IoU

**Synthesis quality:** FID score (ROI-level and full-image), suitability score (color consistency, artifact detection, sharpness)

**Statistical testing:** Formal hypothesis tests (H3: architecture independence, H4: minority class improvement, H5: physical plausibility via FID, H6: optimal augmentation ratio)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{casda2025,
  title   = {[PAPER TITLE]},
  author  = {[AUTHORS]},
  journal = {[VENUE]},
  year    = {[YEAR]},
  url     = {https://github.com/hhjun0321-beep/CASDA}
}
```

---

## Acknowledgments

- **Dataset:** [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) (Kaggle)
- **Diffusion backbone:** [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) (Runway ML) + [ControlNet](https://huggingface.co/lllyasviel/sd-controlnet-canny) (Lvmin Zhang)
- **Detection framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
