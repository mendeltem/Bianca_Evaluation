# BIANCA WMH Segmentation

**Manuscript:** *Stroke lesion preprocessing for robust WMH segmentation with BIANCA*
**Journal:** NeuroImage: Clinical
**Author:** Uchralt Temuulen

---

## Overview

This repository contains the analysis pipeline for evaluating BIANCA (Brain Intensity AbNormality Classification Algorithm) WMH segmentation under three FLAIR preprocessing conditions in stroke patients:

- **Non-removed** — original bias-corrected FLAIR
- **Removed** — stroke lesion voxels zero-filled
- **Inpainted** — stroke lesion voxels replaced with NAWM intensities (FSL `lesion_filling`)

The study uses a two-phase design: Phase I (n=89) optimizes parameters via stratified 5-fold cross-validation (10 seeds, scanner-balanced), while Phase II assesses preprocessing effects with ground truth (n=89) and volume-based robustness (n=211).

## Repository Structure

```
├── Phase_1/                          # Phase I: parameter optimization & result analysis
│   ├── 5FCV_SET/                     #   Threshold, cluster, train/test condition analyses
│   └── GE_COMPARE/                   #   GE scanner inclusion/exclusion analysis
├── Phase_2_a/                        # Phase II-A: Dice-based assessment (n=89)
├── Phase_2_b/                        # Phase II-B: Volume-based assessment (n=211)
├── 0_Severity_definitions.py         # WMH severity classification
├── 1_preprare_locate.py              # LOCATE adaptive threshold training
├── 3_threshold_analysis_*.py         # Threshold sweep (1–99%)
├── 4_cluster_size_grid_search.py     # Cluster-level grid search & metrics
├── 5_create_5_fold_cross_validation_sets.py  # Stratified 5-fold CV pipeline
├── 6_train_bianca.py                 # Final model training (inpainted condition)
├── 7_run_removal.py                  # Phase II inference
├── 8_histogramm_removal.py           # FLAIR histogram analysis
└── preprocess.py                     # Preprocessing utilities
```

## Analysis

The pipeline covers threshold optimization (B0.85, B0.90, LOCATE), cluster-level evaluation (26-connectivity, minimum cluster size grid search), preprocessing condition equivalence testing (Friedman, ICC, Bland–Altman, Cliff's Delta), feature importance via SHAP, and correlation analyses (Spearman with Bonferroni correction).

## Installation

### Prerequisites

- Python 3.12 (conda/miniforge)
- FSL (BIANCA, LOCATE, `lesion_filling`)
- ANTsPy, HD-BET, TrueNet

### Setup

```bash
pip install -r requirements.txt
```

All scripts read from a `.env` file (not tracked):

```env
DATASET=BELOVE_BIDS_WMH_BIDS
CONDA_ENV=/path/to/bianca_env
STANDARD_SPACE_T1=/path/to/MNI152_T1_1mm_Brain.nii.gz
LOCATE_PATH=/path/to/LOCATE-BIANCA
SLURM_ACCOUNT=sc-users
SLURM_CPUS=64
SLURM_MEM=128G
SLURM_TIME=48:00:00
```

## License

This project is part of the BeLOVE study (Charité – Universitätsmedizin Berlin).
