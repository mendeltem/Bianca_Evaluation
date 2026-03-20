# Pipeline Documentation
## BIANCA WMH Segmentation — Major Revision (NeuroImage: Clinical)

**Author:** Uchralt Temuulen | **Generated:** March 2026

---

## 1. Project Overview

This document describes the complete analysis pipeline for the BIANCA WMH (White Matter Hyperintensity) segmentation study, submitted to NeuroImage: Clinical. The pipeline addresses all reviewer concerns from the major revision, organized across two directory levels within the `1_Project_all_code_all_data` project root.

The pipeline is divided into two layers: (1) a `DATASETS/` subdirectory containing preprocessing scripts that convert raw BIDS-formatted neuroimaging data into analysis-ready derivatives, and (2) the main project directory containing the numbered analysis scripts (Steps 0–8) that perform severity classification, cross-validation, statistical analysis, and robustness assessment.

All scripts are designed to run on the Charité HPC cluster via SLURM, with configuration managed through a shared `.env` file. The conda environment (`bianca_env`) provides all Python dependencies, while FSL must be installed separately on the cluster.

---

## 2. Directory Structure

```
1_Project_all_code_all_data/
├── .env                                    # Shared environment configuration
├── DATASETS/                               # Preprocessing layer
│   ├── BELOVE_BIDS_WMH_BIDS/              # BeLOVE dataset (BIDS format)
│   ├── CHALLENGE_BIDS_WMH_BIDS/           # WMH Challenge dataset (BIDS)
│   ├── REMOVAL_BELOVE_DATASET_BIDS/       # Removal subset (Phase II-B)
│   ├── librarys/                           # Shared Python utility modules
│   ├── preprocess.py                       # Main preprocessing pipeline
│   ├── lesion_level_classier.py            # Severity classification
│   ├── create_cross_validation.py          # CV split creation
│   ├── run_slurm_preprocess.py             # SLURM launcher
│   ├── environment.yml / INSTALL.md        # Environment setup
│   └── .env                                # Dataset-specific config
├── Phase_1/                                # Phase I outputs (CV, LOCATE)
├── Phase_2_a/                              # Phase II-A outputs (Dice metrics)
├── Phase_2_b/                              # Phase II-B outputs (volume analysis)
├── 0_Severity_definitions.py               # Step 0: Severity cutoffs
├── 0_severity_population.py                # Step 0: Population tables
├── 1_preprare_locate.py                    # Step 1: LOCATE training
├── 2_compare_GE_analyse.py                 # Step 2: GE comparison
├── 3_threshold_analysis_all_seeds.py       # Step 3: Threshold sweep
├── 3_threshold_analysis_parallel_temp.py
├── 4_cluster_size_grid_search.py           # Step 4: Cluster analysis
├── 4_Analyze_cluster_metrics.py
├── 4_plot_cluster_grid_search.py
├── 5_create_5_fold_cross_validation_sets.py  # Step 5: Main CV
├── 5_population_5fcv.py
├── 6_train_bianca.py                       # Step 6: Final model
├── 7_run_removal.py                        # Step 7: Phase II
├── 8_histogramm_removal.py                 # Step 8: Histograms
├── 02_run_slurm_compare_GE_analyse.py      # SLURM launchers
├── 03_runs_slurm.py
├── 04_run_slurm_cluster_analysis.py
├── 05_run_slurm_cv_5.py
└── 07_run_slurm_removal.py
```

---

## 3. Environment Configuration

All scripts read their configuration from a shared `.env` file in the project root:

| Variable | Value | Description |
|----------|-------|-------------|
| `DATASET` | `CHALLENGE_BIDS_WMH_BIDS` / `BELOVE_BIDS_WMH_BIDS` / `REMOVAL_BELOVE_DATASET_BIDS` | Active dataset to process |
| `SLURM_ACCOUNT` | `sc-users` | SLURM billing account |
| `SLURM_CPUS` | `64` | CPUs per task |
| `SLURM_MEM` | `128G` | Memory allocation |
| `SLURM_TIME` | `48:00:00` | Wall time limit |
| `CONDA_ENV` | `/home/temuuleu/.../bianca_env` | Full path to conda environment |
| `STANDARD_SPACE_T1` | `.../MNI152_T1_1mm_Brain.nii.gz` | MNI standard space template |
| `LOCATE_PATH` | `.../LOCATE-BIANCA` | LOCATE software directory |
| `SHUFFLE_DATA_SET` | `True` | Shuffle subjects in CV splits |
| `FORCE` | `False` | Force reprocessing of existing outputs |

---

## 4. Preprocessing Layer (DATASETS/)

The `DATASETS/` directory contains all preprocessing code that transforms raw BIDS-formatted neuroimaging data into analysis-ready derivatives. This layer handles three datasets: CHALLENGE_BIDS_WMH_BIDS (WMH Segmentation Challenge, N=60), BELOVE_BIDS_WMH_BIDS (BeLOVE cohort with expert masks, N=89), and REMOVAL_BELOVE_DATASET_BIDS (full BeLOVE cohort for Phase II-B, N=211).

| File | Purpose | Description |
|------|---------|-------------|
| `preprocess.py` | Preprocessing Pipeline | HD-BET brain extraction, ANTs N4 bias correction, FSL lesion_filling inpainting, lesion mask creation, QC PNG generation. Processes T1 and FLAIR for each subject. |
| `lesion_level_classier.py` | Severity Classification | Trains a Decision Tree (max_depth=2) on Challenge WMH volumes (N=40) to derive severity cutoffs: LOW ≤ 6.96 mL, MIDDLE 6.96–27.40 mL, HIGH > 27.40 mL. Applies learned thresholds to BeLOVE dataset. |
| `create_cross_validation.py` | Cross-Validation Splits | Merges BeLOVE and Challenge datasets (excluding GE Signa), creates stratified 5-fold CV splits balanced by scanner type and WMH severity level. |
| `run_slurm_preprocess.py` | SLURM Job Launcher | Generates and submits a SLURM batch script for preprocess.py. Reads cluster configuration from .env. |
| `environment.yml` | Conda Environment | Defines bianca_env: Python 3.12, nibabel, scikit-learn, ANTsPy, HD-BET, TrueNet. |
| `INSTALL.md` | Installation Guide | Step-by-step setup instructions for conda environment, HD-BET, ANTsPy, TrueNet, and FSL dependency. |
| `.env` | Environment Config | Dataset paths, SLURM parameters, conda environment path, FSL standard space template, LOCATE path, processing flags. |

### 4.1 Preprocessing Pipeline (preprocess.py)

The preprocessing pipeline performs the following steps for each subject:

1. **T1 processing:** TrueNet-based preprocessing (`prepare_truenet_data`) for skull stripping and MNI registration.
2. **FLAIR brain extraction:** HD-BET neural network-based skull stripping.
3. **Bias field correction:** ANTs N4BiasFieldCorrection applied to brain-extracted FLAIR.
4. **Lesion masking:** For subjects with stroke lesions, creates binary lesion masks and applies dilation for safety margins.
5. **Lesion inpainting:** FSL `lesion_filling` replaces lesion voxels with locally sampled NAWM intensities (inpainted/filled condition).
6. **Lesion removal:** Zero-intensity replacement of lesion voxels (removed condition).
7. **QC image generation:** PNG slices for visual quality control.

Three FLAIR variants are produced per subject: `non_removed` (original bias-corrected), `removed` (zero-filled), and `inpainted`/`filled` (FSL lesion_filling). These three conditions are carried through the entire analysis pipeline.

### 4.2 Severity Classification

WMH severity levels are derived from a Decision Tree classifier (max_depth=2) trained on the Challenge dataset (N=40). The tree learns volume-based cutoffs that partition subjects into three severity groups: LOW (≤ 6.96 mL), MIDDLE (6.96–27.40 mL), and HIGH (> 27.40 mL). These cutoffs are applied to the BeLOVE dataset for consistent severity labeling across all analyses.

---

## 5. Analysis Pipeline (Steps 0–8)

The numbered scripts in the main project directory implement the full analysis pipeline for the manuscript. Each step builds on the preprocessed data from the DATASETS layer and produces results used in subsequent steps or directly in the paper.

| Step | Script(s) | Phase | Description |
|------|-----------|-------|-------------|
| 0 | `0_Severity_definitions.py`, `0_severity_population.py` | Severity Definitions & Population Tables | Assigns WMH severity levels using fixed Decision Tree cutoffs. Updates source Excel files in-place. Generates summary Excel and JPG table images for Phase I, LOCATE pool, and Phase II-A. |
| 1 | `1_preprare_locate.py` | LOCATE Training Pipeline | Holds out 21 subjects (7 per scanner) for LOCATE threshold training. LOO BIANCA for unbiased LPMs. Trains LOCATE random forest. Builds models without GE (n=21) and with GE (n=26). |
| 2 | `2_compare_GE_analyse.py` | GE Scanner Comparison | Nested training design (45 without GE vs. 60 with GE). Cliff's Delta, Wilcoxon signed-rank, Bonferroni correction, Bland–Altman agreement. Addresses R1C4; R5#8. |
| 3 | `3_threshold_analysis_all_seeds.py`, `3_threshold_analysis_parallel_temp.py` | Threshold Optimization | Threshold sweep (1–99%) across 10 seeds × 5 folds. Confirms 0.85 as optimal for inpainted condition. Addresses R1C1; R5#7. |
| 4 | `4_cluster_size_grid_search.py`, `4_Analyze_cluster_metrics.py`, `4_plot_cluster_grid_search.py` | Cluster-Level Analysis | Grid search MCS (1–100 voxels), 26-connectivity, any-overlap criterion (MICCAI 2017). Lesion-level Precision, Recall, F1. Addresses R1C2; R5#17. |
| 5 | `5_create_5_fold_cross_validation_sets.py`, `5_population_5fcv.py` | 5-Fold Stratified CV | Main evaluation: 10 seeds × 5 folds, 3×3 condition matrix, voxel + lesion-level metrics. Three thresholds (0.85, 0.90, LOCATE). Addresses R1C1–3; R5#5,7,9,17. |
| 6 | `6_train_bianca.py` | Final BIANCA Model Training | Trains on inpainted FLAIR. Excludes Phase II test + LOCATE subjects. Single model, three test conditions. |
| 7 | `7_run_removal.py` | Phase II Robustness Assessment | Phase II-A (n=86): Dice with expert masks. Phase II-B (n=211): volume-based robustness. Three conditions. PV vs. deep WMH stratification. Addresses R1C1; R2C2; R5#9. |
| 8 | `8_histogramm_removal.py` | Histogram Intensity Analysis | FLAIR intensity distribution comparison (non_removed vs. removed). Demonstrates right-tail distortion from stroke lesions. Addresses R5C18. |

### 5.1 Step 0: Severity Definitions & Population Tables

Two scripts handle severity classification at the project level. `0_Severity_definitions.py` applies the fixed Decision Tree cutoffs (LOW ≤ 6.96 mL, MIDDLE 6.96–27.40 mL, HIGH > 27.40 mL) to update `severity_level` columns in the Phase I pool (`bianca_pool_wihtouth_ge.xlsx`) and the LOCATE training pool (`locate_pool.xlsx`). It then generates a summary Excel workbook with severity distributions per cohort and scanner × severity cross-tabulations, plus publication-ready JPG table images.

`0_severity_population.py` performs the same in-place updates and generates an additional Excel report (`2_severity_definitions.xlsx`) with detailed statistics including sample sizes, volume ranges, medians, and interquartile ranges for each severity level within each cohort (Phase I, LOCATE pool, Phase II-A).

### 5.2 Step 1: LOCATE Training

To prevent data leakage through the thresholding step, 21 subjects are held out from the main cross-validation for LOCATE training. These subjects are stratified by scanner type (n=7 each for Philips, Siemens Tim Trio, Siemens Prisma fit). For each of the 21 subjects, a leave-one-out BIANCA model is trained on the remaining 20 to generate unbiased lesion probability maps (LPMs). The LPMs, together with FLAIR, T1, white matter masks, brain masks, and ventricle distance maps, train the LOCATE random forest for locally adaptive threshold estimation.

Two LOCATE models are produced: one trained on 21 subjects (without GE, for main analyses) and one on 26 subjects (21 + 5 GE Signa, for scanner generalization analyses in Step 2).

### 5.3 Step 2: GE Scanner Comparison

This analysis addresses **R1 Comment 4** and **R5 #8** regarding the exclusion of GE Signa scans. A nested training design is used where the "without GE" training set (45 subjects: 15 Prisma + 15 Trio + 15 Philips) is a strict subset of the "with GE" set (60 subjects: same 45 + 15 GE). This ensures that any performance difference is attributable solely to the GE data.

Statistical analysis includes Cliff's Delta effect sizes with bootstrap 95% confidence intervals (1000 iterations), Wilcoxon signed-rank tests with Bonferroni correction, scanner-stratified evaluation, and Bland–Altman agreement analysis. The term "robustness" is qualified to apply specifically to Siemens and Philips platforms.

### 5.4 Step 3: Threshold Optimization

The threshold sweep evaluates all integer thresholds from 1% to 99% across all 10 seeds of the stratified 5-fold CV pipeline. For each threshold, precision, sensitivity, and Dice are computed on the inpainted condition. The analysis confirms that 0.85 is the optimal threshold, replicating findings from Ferris et al. (2023) and demonstrating convergence independent of random seed.

Two script variants exist: `3_threshold_analysis_parallel_temp.py` combines 5-fold CV creation with threshold analysis in a single run (for the inpainted condition only), while `3_threshold_analysis_all_seeds.py` reads pre-existing LPMs from the full pipeline and performs the sweep post-hoc across all seeds.

### 5.5 Step 4: Cluster-Level Analysis

This step addresses **R1 Comment 2** and **R5 #17**, which raised concerns that global Dice alone cannot distinguish genuine sensitivity gains from spatially distributed false positives. Three scripts work together:

- **`4_cluster_size_grid_search.py`:** Grid search over minimum cluster sizes (1–100 voxels) across 10 seeds × 5 folds to determine the optimal filtering threshold that maximizes lesion-level F1.
- **`4_plot_cluster_grid_search.py`:** Visualization of grid search results with best minimum cluster size annotations.
- **`4_Analyze_cluster_metrics.py`:** Full cluster-level analysis including voxel vs. cluster dissociation, cluster precision vs. recall, R vs. NR condition comparison, Cliff's Delta, post-hoc Wilcoxon with Bonferroni, and per-dataset breakdowns.

Connected-component labeling uses 26-connectivity. The overlap criterion follows MICCAI 2017: a predicted cluster is a true positive if it overlaps with any ground truth cluster by at least one voxel. Filtering is applied only to predictions; ground truth clusters are retained regardless of size.

### 5.6 Step 5: 5-Fold Stratified Cross-Validation

This is the main evaluation pipeline, addressing **R1 Comments 1–3** and **R5 #5, #7, #9, #17**. The design uses 10 random seeds × 5 folds, stratified by scanner type (~1/3 each: Philips, Tim Trio, Prisma fit) and WMH severity level. All 9 combinations of training and test preprocessing conditions (non_removed, removed, inpainted × non_removed, removed, inpainted) are evaluated per fold.

Metrics computed include voxel-level Dice coefficient, sensitivity, and precision, as well as lesion-level F1 from cluster analysis. Three thresholds are compared: fixed 0.85, fixed 0.90, and LOCATE adaptive thresholding. Results demonstrate convergence across all conditions (differences in the 4th decimal place), supporting the choice of inpainted as the primary condition.

### 5.7 Step 6: Final BIANCA Model Training

Based on the 5-fold CV results showing equivalent performance across all three conditions, the inpainted (FSL lesion_filling) condition is selected for the final BIANCA model. The training pool excludes all Phase II test subjects and LOCATE training subjects to maintain strict separation. This single model is applied to all three FLAIR variants at inference time in Step 7, isolating the preprocessing effect while keeping the classifier constant.

### 5.8 Step 7: Phase II Robustness Assessment

Phase II-A (n=86) evaluates subjects with expert-delineated WMH masks, providing ground-truth-referenced accuracy metrics (Dice, sensitivity, precision) for whole, deep, and periventricular WMH. Phase II-B (n=211) extends to the full BeLOVE cohort for volume-based robustness assessment with increased statistical power to detect scanner effects.

Three preprocessing conditions are evaluated. BIANCA inference + LOCATE adaptive thresholding is applied to all conditions. Segmentation output is split into periventricular and deep WMH using anatomical masks for region-specific evaluation. Scanner-stratified analysis leverages increased Philips representation in Phase II-B.

### 5.9 Step 8: Histogram Intensity Analysis

This script directly addresses **R5 Comment 18**, which noted that the argument about stroke lesions distorting global histograms was plausible but not empirically demonstrated. It compares FLAIR intensity distributions between non_removed and removed conditions across the Phase II-B cohort (n=211), demonstrating that stroke lesions produce a right-tail distortion that is eliminated by lesion removal.

NIfTI loading and intensity extraction is parallelized across multiple workers. Summary statistics are computed in the first pass; raw arrays are loaded only for 3 representative subjects and group KDE sampling. Output includes 300 dpi publication figures and a statistics Excel file.

---

## 6. SLURM Job Management

Each computationally intensive analysis step has a corresponding SLURM launcher script (prefixed with the step number and `_run_slurm_`). These launchers read cluster configuration from the `.env` file and generate a bash script that is submitted via `sbatch`. All launchers follow the same pattern: load conda via miniforge, activate `bianca_env`, set `MPLBACKEND=Agg` for non-interactive matplotlib, and execute the target Python script.

| SLURM Launcher | Target Script | Task |
|----------------|---------------|------|
| `02_run_slurm_compare_GE_analyse.py` | `2_compare_GE_analyse.py` | GE comparison analysis |
| `03_runs_slurm.py` | `3_threshold_analysis_*.py` | Threshold sweep |
| `04_run_slurm_cluster_analysis.py` | `4_cluster_size_grid_search.py` | Cluster grid search |
| `05_run_slurm_cv_5.py` | `5_create_5_fold_cross_validation_sets.py` | 5-fold CV pipeline |
| `07_run_slurm_removal.py` | `7_run_removal.py` | Phase II removal |

Job logs are written to a `log/` subdirectory. Email notifications are sent on job completion or failure to the address specified in `SLURM_MAIL_USER`.

---

## 7. Reviewer Comment Mapping

The following table maps each reviewer concern to the scripts that address it. This mapping is also documented in the docstrings of each script for traceability.

| Reviewer Comment | Concern | Resolution (Script) |
|------------------|---------|---------------------|
| R1 Comment 1 / R5 #9 | Zero-filling non-physiological; inpainting not tested | Scripts 3, 5, 7, 8: Three conditions (non_removed, removed, inpainted) evaluated |
| R1 Comment 2 / R5 #17 | Dice alone insufficient; precision not contextualized | Scripts 4, 5: Cluster-based lesion-level F1, grid search for MCS |
| R1 Comment 3 / R5 #7 | LOO-CV inadequate; scanner leakage concerns | Scripts 3, 5: Stratified 5-fold CV, 10 seeds, scanner-balanced |
| R1 Comment 4 / R5 #8 | GE exclusion weakens robustness claim | Script 2: Nested training design, empirical GE comparison |
| R2 Comment 2 | Phase II-A vs II-B distinction unclear | Script 7: Phase II-A (Dice with masks) vs II-B (volume-based) |
| R5 #5 | Multiple comparisons not transparent | Scripts 2, 4, 5: Bonferroni correction families documented |
| R5 #18 | Histogram distortion not empirically shown | Script 8: FLAIR intensity distribution comparison |

---

## 8. Recommended Execution Order

The pipeline should be executed in the following order, respecting data dependencies between steps:

1. **Preprocessing:** Run `preprocess.py` (via `run_slurm_preprocess.py`) for each dataset: `CHALLENGE_BIDS_WMH_BIDS`, `BELOVE_BIDS_WMH_BIDS`, `REMOVAL_BELOVE_DATASET_BIDS`.
2. **Severity classification:** Run `lesion_level_classier.py` to assign severity levels.
3. **Step 0:** Run `0_Severity_definitions.py` and `0_severity_population.py` for population tables.
4. **Step 1:** Run `1_preprare_locate.py` to train LOCATE models (requires preprocessed data).
5. **Step 3:** Run threshold analysis to confirm optimal threshold at 0.85 (requires LOCATE from Step 1).
6. **Step 4:** Run cluster grid search and analysis (requires CV outputs).
7. **Step 5:** Run the full 5-fold stratified CV pipeline (requires LOCATE from Step 1, threshold from Step 3).
8. **Step 2:** Run GE comparison analysis (can run independently once preprocessing is complete).
9. **Step 6:** Train the final BIANCA model on inpainted data (requires CV results from Step 5).
10. **Step 7:** Run Phase II removal/robustness assessment (requires trained model from Step 6).
11. **Step 8:** Run histogram intensity analysis (requires Phase II-B preprocessed data).

---

## 9. Software Dependencies

The pipeline requires the following software stack:

- **Python 3.12** (via conda/miniforge)
- **FSL** (`FSLDIR` must be set; provides BIANCA, LOCATE, lesion_filling, fsl_anat)
- **ANTsPy** (N4 bias field correction)
- **HD-BET** (neural network brain extraction)
- **TrueNet** (T1 preprocessing via `prepare_truenet_data`)
- **nibabel, scikit-learn, scipy, numpy, pandas, matplotlib, openpyxl, Pillow**
- **python-dotenv** (environment variable management)
- **SLURM workload manager** (Charité HPC cluster)
