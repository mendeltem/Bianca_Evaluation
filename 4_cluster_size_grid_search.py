#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Contained Cluster Size Grid Search Pipeline
=================================================

Revision context (NeuroImage: Clinical, Major Revision)
-------------------------------------------------------
This script addresses two grouped reviewer concerns:

  (1) Lesion-level metrics and Dice validity (R1 Comment 2; R5 #17)
      R1 criticized that global Dice alone cannot confirm whether higher
      WMH volumes after lesion removal reflect true positives or spatially
      distributed false positives ("boundary thickening"). R5 noted that
      precision values (~0.64) were not sufficiently interrogated.

      Resolution: We added cluster-based lesion-level Precision, Recall,
      and F1 scores using connected-component labeling (26-connectivity).
      The minimum cluster size (MCS) was determined empirically via this
      grid search (MCS 1-100 voxels) to avoid arbitrary thresholds.
      Overlap criterion: any-overlap (a predicted cluster touching at
      least one ground truth voxel counts as TP), consistent with the
      MICCAI 2017 WMH Segmentation Challenge (Kuijf et al., 2019).
      Filtering is applied ONLY to predictions; expert-delineated ground
      truth clusters are retained regardless of size.

  (2) Cross-validation design (R1 Comment 3; R5 #7)
      Both reviewers noted that LOO-CV on a mixed multi-scanner dataset
      does not address generalization to unseen scanner types and may
      induce subtle data leakage of scanner-specific intensity patterns.

      Resolution: LOO-CV was replaced with stratified 5-fold CV, balanced
      by scanner type (~1/3 each: Philips, Siemens Tim Trio, Siemens
      Prisma fit) and WMH severity. To ensure robustness of the MCS
      estimate, the grid search is repeated across 10 random seeds
      (5 folds each), and the optimal MCS is the value maximizing mean
      lesion-level F1 averaged over all 50 train/test splits.

Paper changes
-------------
  Methods section 2.5 (new): "Cluster-based lesion-level evaluation"
    - Describes connected-component labeling, any-overlap criterion,
      and empirically determined MCS from this grid search.
    - MCS values are dataset-specific (BeLOVE vs Challenge) and
      threshold-specific (0.85, 0.90, LOCATE), reported in Supplemental.

  Methods section 2.4 (revised): "Cross-validation design"
    - LOO-CV replaced with stratified 5-fold CV.
    - Scanner and severity stratification described.

  Results section 3.1 (revised):
    - Lesion-level Precision, Recall, F1 reported alongside voxel-level
      Dice, sensitivity, precision for all three conditions
      (non_removed, removed, inpainted).
    - Language revised: "confirmed true positives" replaced with
      "increase in sensitivity with concurrent marginal reduction in
      precision, supported by both voxel-level and cluster-level evidence."

  Supplemental material (new):
    - Table: Optimal MCS per dataset and threshold with F1/P/R values.
    - Grid search curves (MCS vs F1) for transparency.

  Response to Reviewers:
    - R1 Comment 2: Cluster-based metrics added; language revised.
    - R5 #17: Precision contextualized against published benchmarks;
      cluster-level metrics provide complementary boundary-independent
      evaluation.
    - R1 Comment 3 / R5 #7: LOO-CV replaced with stratified 5-fold CV.

Technical design
----------------
This script is self-contained: it creates the 5-fold splits, trains
BIANCA (inpainted condition only), runs inference, applies thresholds
(0.85, 0.90, LOCATE), and sweeps MCS 1-100. Outputs land in
Phase_1/5FCV_SET/ in the SAME directory structure as the main CV
pipeline (Script 5), so all intermediate files (models, LPMs,
thresholded images) are reusable without re-computation.

The optimal MCS values are saved as optimal_mcs_lookup.json for direct
import into the main pipeline's MCS_LOOKUP dictionary.

Usage:
  # Local (all seeds, all folds):
  python 4_cluster_size_grid_search.py

  # SLURM array (one fold per task, all seeds):
  sbatch --array=0-4 run_grid_search.sh

  # Merge only:
  python 4_cluster_size_grid_search.py --merge

References:
  Kuijf et al. (2019). Standardized assessment of automatic
    segmentation of white matter hyperintensities. IEEE TMI, 38(11).
  Griffanti et al. (2016). BIANCA: Brain Intensity AbNormality
    Classification Algorithm. NeuroImage: Clinical, 9, 235-242.
  Sundaresan et al. (2019). Automated lesion segmentation with BIANCA.
    NeuroImage, 202, 116056.
    
    
"""

import os
import sys
import json
import glob
import gzip
import subprocess
import tempfile
import time
from typing import List

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage import label as scipy_label, generate_binary_structure
from joblib import Parallel, delayed

from DATASETS.librarys.directory_lib import (
    fsl_copy, get_volume, threshold_lpm,
    apply_bianca_mask, fsl_dice_score,
)

# =============================================================
# CONFIG
# =============================================================
SEEDS = list(range(1, 11))
N_FOLDS = 5
SCANNER_NAMES = ["Philips", "Tim Trio", "Prisma_fit"]

# Condition: inpainted only
CONDITION_NAME = "filled"           # used in combo_name and directory structure
FLAIR_COL = "FLAIR_filled_path"
FLAIR_FALLBACK = "FLAIR_non_removed_path"

# Thresholds for grid search
THRESHOLDS = ["85", "90", "locate"]

# Cluster size sweep
CLUSTER_SIZES_ALL = list(range(1, 101))

# Paths
BELOVE_BASE = "DATASETS/BELOVE_BIDS_WMH_BIDS"
CHALLENGE_BASE = "DATASETS/CHALLENGE_BIDS_WMH_BIDS"
BIANCA_POOL_PATH = "Phase_1/LOCATE_SET/bianca_pool_wihtouth_ge.xlsx"

# Output directories (SAME structure as main 5FCV pipeline)
CV_SET_BASE_DIR = "Phase_1/5FCV_SET"
CLUSTER_DIR = "Phase_1/Cluster_analysis_CC/"

# LOCATE
LOCATE_TRAIN_DIR = (
    "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/"
    "DATASET_SC/PROJECT_NULL_07_04_2025/1_Project_all_code_all_data/"
    "Phase_1/LOCATE_SET/locate_train"
)
LOCATE_PATH = (
    "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/"
    "Projects/BIANCA/LOCATE/LOCATE-BIANCA"
)
LOCATE_FEATURE_SELECT = [1, 1, 1, 1]

# Processing
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
MAX_RETRIES = 30
RETRY_DELAY = 2

# NIfTI label saving (optional, for inspection)
SAVE_NIFTI_LABELS = True
NIFTI_SAVE_STEPS = [10, 16, 20]


# =============================================================
# UTILITY FUNCTIONS
# =============================================================

def ensure_valid_gzip(src, dst):
    """Check if dst is a valid gzip; recreate from src if not."""
    if not str(dst).endswith(".gz"):
        return True
    if os.path.exists(dst):
        try:
            with gzip.open(dst, "rb") as f:
                f.read(2)
            return True
        except (gzip.BadGzipFile, OSError, EOFError):
            pass
    cmd = ["fslmaths", src, "-mul", "1", dst]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  fslmaths failed: {e}")
        return False


def get_valid_flair(dataset_base, row):
    """Resolve FLAIR path: inpainted first, fallback to non_removed."""
    for col in [FLAIR_COL, FLAIR_FALLBACK]:
        if col in row.index and pd.notna(row[col]):
            p = os.path.join(dataset_base, str(row[col]))
            if os.path.isfile(p):
                return p
    return ""


def get_valid_path(base, primary_col, fallback_col, row):
    """Resolve a file path with fallback."""
    p1 = os.path.join(base, str(row[primary_col])) if pd.notna(row.get(primary_col)) else ""
    if p1 and os.path.isfile(p1):
        return p1
    return os.path.join(base, str(row[fallback_col])) if pd.notna(row.get(fallback_col)) else ""


def run_locate_testing(
    train_image_directory_path: str,
    test_image_directory_name: str,
    locate_path: str,
    verbose: int = 1,
    feature_select: List[int] = [1, 1, 1, 1],
) -> None:
    """Run LOCATE testing via MATLAB."""
    matlab_script = f"""
    addpath(genpath('{locate_path}'));
    feature_select = {feature_select};
    verbose = {verbose};
    try
        LOCATE_testing('{test_image_directory_name}', '{train_image_directory_path}', feature_select, verbose);
    catch ME
        fprintf('Error: %s\\n', ME.message);
        exit(1);
    end
    exit(0);
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".m") as tmp:
        script_path = tmp.name
        tmp.write(matlab_script)
    try:
        rc = os.system(f"matlab -batch \"run('{script_path}');\"")
        if rc != 0:
            raise RuntimeError(f"LOCATE testing failed (rc={rc})")
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


# =============================================================
# GRID SEARCH CORE (from original Script 4)
# =============================================================

def evaluate_all_mcs_ultra_fast(pred_data, gt_data, mcs_list):
    """Evaluate all MCS values in a single pass (ultra-optimized)."""
    struct = generate_binary_structure(3, 3)

    pred_labeled, n_pred_raw = scipy_label(pred_data, structure=struct)
    gt_labeled, n_gt = scipy_label(gt_data, structure=struct)

    pred_sizes = np.bincount(pred_labeled.ravel()) if n_pred_raw > 0 else np.array([])

    overlap_mask = (pred_labeled > 0) & (gt_labeled > 0)
    if overlap_mask.any():
        overlapping_pairs = np.unique(
            np.c_[pred_labeled[overlap_mask], gt_labeled[overlap_mask]], axis=0
        )
        hit_pred_ids = np.unique(overlapping_pairs[:, 0])
    else:
        overlapping_pairs = np.empty((0, 2), dtype=int)
        hit_pred_ids = np.array([], dtype=int)

    results = {}
    for mcs in mcs_list:
        if n_pred_raw == 0 or n_gt == 0:
            results[mcs] = {
                "lesion_precision": 0.0, "lesion_recall": 0.0, "lesion_f1": 0.0,
                "n_pred_clusters": 0, "n_gt_clusters": n_gt,
            }
            continue

        valid_pred_ids = np.where(pred_sizes >= mcs)[0]
        valid_pred_ids = valid_pred_ids[valid_pred_ids > 0]
        n_pred = len(valid_pred_ids)

        if n_pred == 0:
            results[mcs] = {
                "lesion_precision": 0.0, "lesion_recall": 0.0, "lesion_f1": 0.0,
                "n_pred_clusters": 0, "n_gt_clusters": n_gt,
            }
            continue

        tp_pred = len(np.intersect1d(valid_pred_ids, hit_pred_ids))
        valid_pairs = overlapping_pairs[np.isin(overlapping_pairs[:, 0], valid_pred_ids)]
        tp_gt = len(np.unique(valid_pairs[:, 1]))

        prec = tp_pred / n_pred
        rec = tp_gt / n_gt
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results[mcs] = {
            "lesion_precision": round(prec, 4),
            "lesion_recall": round(rec, 4),
            "lesion_f1": round(f1, 4),
            "n_pred_clusters": n_pred,
            "n_gt_clusters": n_gt,
        }

    return results, pred_labeled, gt_labeled, pred_sizes


def generate_clean_nifti_mask(pred_labeled, pred_sizes, mcs):
    """Generate a filtered label mask for a given MCS."""
    struct = generate_binary_structure(3, 3)
    valid_labels = pred_sizes >= mcs
    valid_labels[0] = False
    filtered_mask = valid_labels[pred_labeled]
    labeled_clean, _ = scipy_label(filtered_mask, structure=struct)
    return labeled_clean


# =============================================================
# DATA LOADING
# =============================================================

def load_data():
    """Load bianca pool and full file dataframes."""
    bianca_pool_df = pd.read_excel(BIANCA_POOL_PATH)
    meta_columns = ["subject", "dataset_base", "scanner", "severity_level"]
    bianca_pool_meta = bianca_pool_df[meta_columns].copy()

    belove_df = pd.read_excel(os.path.join(BELOVE_BASE, "derivatives/preprocessed_files.xlsx"))
    challenge_df = pd.read_excel(os.path.join(CHALLENGE_BASE, "derivatives/preprocessed_files.xlsx"))
    all_file_df = pd.concat([belove_df, challenge_df], ignore_index=True)

    all_file_df["manual_mask"] = np.where(
        all_file_df["WMH_removed_path"].notna(),
        all_file_df["WMH_removed_path"],
        all_file_df["WMH_path"],
    )

    if FLAIR_COL not in all_file_df.columns:
        all_file_df[FLAIR_COL] = all_file_df.get(FLAIR_FALLBACK, all_file_df["FLAIR_brain_biascorr"])
    if FLAIR_FALLBACK not in all_file_df.columns:
        all_file_df[FLAIR_FALLBACK] = all_file_df["FLAIR_brain_biascorr"]
    else:
        all_file_df[FLAIR_FALLBACK] = all_file_df[FLAIR_FALLBACK].fillna(
            all_file_df["FLAIR_brain_biascorr"]
        )

    base_mapping = bianca_pool_meta[["subject", "dataset_base"]].drop_duplicates()
    if "dataset_base" in all_file_df.columns:
        all_file_df = all_file_df.drop(columns=["dataset_base"])
    all_file_df = all_file_df.merge(base_mapping, on="subject", how="left")
    all_file_df = all_file_df.drop_duplicates(subset=["subject"])

    pool_subjects = set(bianca_pool_meta["subject"])
    all_file_df = all_file_df[all_file_df["subject"].isin(pool_subjects)].copy()

    scanner_dfs = {}
    for sn in SCANNER_NAMES:
        scanner_dfs[sn] = bianca_pool_meta[bianca_pool_meta["scanner"] == sn].reset_index(drop=True)

    return bianca_pool_df, bianca_pool_meta, all_file_df, scanner_dfs


# =============================================================
# FOLD CREATION
# =============================================================

def create_stratified_folds(scanner_dfs, seed):
    """Create scanner-stratified folds for one seed."""
    scanner_folds = {}
    for sn in SCANNER_NAMES:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        X = scanner_dfs[sn]["subject"].values
        y = scanner_dfs[sn]["severity_level"].values
        folds = {}
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            folds[fold_idx] = {"train": X[train_idx].tolist(), "test": X[test_idx].tolist()}
        scanner_folds[sn] = folds
    return scanner_folds


def get_fold_subjects(scanner_folds, fold_idx):
    """Assemble train/test lists from per-scanner folds."""
    train_all, test_all = [], []
    for sn in SCANNER_NAMES:
        train_all.extend(scanner_folds[sn][fold_idx]["train"])
        test_all.extend(scanner_folds[sn][fold_idx]["test"])
    assert len(set(train_all) & set(test_all)) == 0, "DATA LEAK!"
    return train_all, test_all


# =============================================================
# BIANCA TRAINING
# =============================================================

def train_bianca_model(fold_dir, train_all, all_file_df):
    """Train BIANCA model for inpainted condition if not already done."""
    n_train = len(train_all)
    combo = f"train_{CONDITION_NAME}__test_{CONDITION_NAME}"

    master_lines = []
    for subj in train_all:
        rows = all_file_df[all_file_df["subject"] == subj]
        if rows.empty:
            raise ValueError(f"Subject {subj} missing from all_file_df!")
        row = rows.iloc[0]
        dbase = row["dataset_base"]

        flair = get_valid_flair(dbase, row)
        t1 = os.path.join(dbase, row["T1"])
        mat = os.path.join(dbase, row["mni_mat_path"])
        wmh = os.path.join(dbase, row["manual_mask"])

        for path, name in [(flair, FLAIR_COL), (t1, "T1"), (mat, "mni_mat"), (wmh, "manual_mask")]:
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"Missing {name} for {subj}: {path}")

        master_lines.append(f"{flair} {t1} {mat} {wmh}")

    master_file = os.path.join(
        fold_dir, f"bianca_n_{n_train}_balanced_train_master_file_{CONDITION_NAME}.txt"
    )
    with open(master_file, "w") as f:
        f.write("\n".join(master_lines))

    model_path = os.path.join(fold_dir, f"BIANCA_MODEL_N_{n_train}_{CONDITION_NAME.upper()}")

    if not os.path.isfile(model_path):
        trainstring = ",".join(str(r) for r in range(1, n_train + 1))
        cmd = [
            "bianca",
            f"--singlefile={master_file}",
            "--brainmaskfeaturenum=1", "--matfeaturenum=3",
            "--featuresubset=1,2", "--labelfeaturenum=4",
            "--trainingpts=2000", "--nonlespts=10000",
            f"--trainingnums={trainstring}",
            f"--saveclassifierdata={model_path}",
            f"--querysubjectnum={n_train}", "-v",
        ]
        print(f"    Training BIANCA ({CONDITION_NAME}, N={n_train})...")
        subprocess.run(cmd, check=True)
        print(f"    Model saved: {model_path}")
    else:
        print(f"    Model exists: {model_path}")

    return model_path, n_train


# =============================================================
# PER-SUBJECT PROCESSING: inference + thresholding + grid search
# =============================================================

def process_test_subject(test_subject, fold_dir, model_path, n_train,
                         all_file_df, bianca_pool_df):
    """
    For one test subject:
      1. BIANCA inference → LPM
      2. Lesion correction (if applicable)
      3. Threshold at 85, 90 + LOCATE
      4. Cluster grid search on each threshold
    Returns list of result dicts.
    """
    combo_name = f"train_{CONDITION_NAME}__test_{CONDITION_NAME}"

    rows = all_file_df[all_file_df["subject"] == test_subject]
    if rows.empty:
        print(f"    {test_subject}: missing from all_file_df, skipping")
        return []

    test_row = rows.iloc[0]
    dbase = test_row["dataset_base"]

    # Resolve paths
    test_FLAIR = get_valid_flair(dbase, test_row)
    test_T1 = os.path.join(dbase, test_row["T1"])
    test_mat = os.path.join(dbase, test_row["mni_mat_path"])
    test_wmh_roi = os.path.join(dbase, test_row["manual_mask"])
    test_WMmask = os.path.join(dbase, test_row["WMmask"])
    test_brainmask = os.path.join(dbase, test_row["brainmask"])
    test_ventdistmap = os.path.join(dbase, test_row["ventdistmap"])

    required = [test_FLAIR, test_T1, test_mat, test_wmh_roi]
    if not all(p and os.path.isfile(p) for p in required):
        print(f"    {test_subject}: missing required files, skipping")
        return []

    # Directory structure matching main pipeline
    sub_dir = os.path.join(fold_dir, "test", test_subject)
    sub_result_dir = os.path.join(sub_dir, f"bianca_result_{combo_name}")
    os.makedirs(sub_result_dir, exist_ok=True)

    # Copy manual mask (needed by grid search later)
    manual_mask_dst = os.path.join(sub_result_dir, f"{test_subject}_manual_mask.nii.gz")
    if not os.path.isfile(manual_mask_dst):
        fsl_copy(test_wmh_roi, manual_mask_dst)

    # --- 1. BIANCA Inference ---
    BIANCA_LPM = os.path.join(
        sub_result_dir, f"{test_subject}_BIANCA_LPM_{combo_name}.nii.gz"
    )
    if not os.path.isfile(BIANCA_LPM):
        test_masterfile = os.path.join(sub_result_dir, f"{test_subject}_masterfile_test.txt")
        with open(test_masterfile, "w") as f:
            f.write(f"{test_FLAIR} {test_T1} {test_mat} {test_wmh_roi}")

        cmd = [
            "bianca",
            f"--singlefile={test_masterfile}",
            "--brainmaskfeaturenum=1", "--matfeaturenum=3",
            "--featuresubset=1,2",
            f"--loadclassifierdata={model_path}",
            "--querysubjectnum=1",
            "-o", BIANCA_LPM, "-v",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"    {test_subject}: LPM created")

    if not os.path.isfile(BIANCA_LPM):
        print(f"    {test_subject}: LPM missing after inference")
        return []

    # --- 2. Lesion correction (if subject has stroke lesion) ---
    lesion_path = test_row.get("lesion_path", None)
    has_lesion = pd.notna(lesion_path) and str(lesion_path).strip() != ""

    LPM_to_use = BIANCA_LPM
    if has_lesion:
        lesion_full = (
            os.path.join(dbase, str(lesion_path))
            if not os.path.isabs(str(lesion_path))
            else str(lesion_path)
        )
        if os.path.isfile(lesion_full):
            corrected = os.path.join(
                sub_result_dir,
                f"{test_subject}_BIANCA_LPM_{combo_name}_corrected.nii.gz",
            )
            if not os.path.isfile(corrected):
                cmd = f"fslmaths {lesion_full} -binv -mul {BIANCA_LPM} {corrected}"
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            LPM_to_use = corrected

            # Copy lesion mask
            les_dst = os.path.join(sub_result_dir, f"{test_subject}_lesion.nii.gz")
            if not os.path.isfile(les_dst):
                fsl_copy(lesion_full, les_dst)

    # --- 3. Threshold + WM mask + LOCATE ---
    subject_results = []

    for threshold in THRESHOLDS:
        pred_path = None

        if threshold == "locate":
            # --- LOCATE ---
            LOCATE_sub_dir = os.path.join(sub_result_dir, "locate")
            os.makedirs(LOCATE_sub_dir, exist_ok=True)
            results_dir = os.path.join(LOCATE_sub_dir, "LOCATE_results_directory")

            existing = (
                [f for f in os.listdir(results_dir) if "BIANCA_LOCATE_binarylesionmap" in f]
                if os.path.exists(results_dir) else []
            )

            if not existing:
                # Prepare LOCATE input files
                for src, suffix in [
                    (test_FLAIR, "feature_FLAIR"),
                    (LPM_to_use, "BIANCA_LPM"),
                    (test_T1, "feature_t1w"),
                    (test_WMmask, "biancamask"),
                    (test_brainmask, "brainmask"),
                    (test_ventdistmap, "ventdistmap"),
                ]:
                    dst = os.path.join(LOCATE_sub_dir, f"{test_subject}_{suffix}.nii.gz")
                    if src and os.path.isfile(src) and not os.path.isfile(dst):
                        fsl_copy(src, dst)

                try:
                    run_locate_testing(
                        train_image_directory_path=LOCATE_TRAIN_DIR,
                        test_image_directory_name=os.path.abspath(LOCATE_sub_dir),
                        locate_path=LOCATE_PATH,
                        verbose=1,
                        feature_select=LOCATE_FEATURE_SELECT,
                    )
                except Exception as e:
                    print(f"    {test_subject}: LOCATE failed: {e}")
                    continue

                existing = (
                    [f for f in os.listdir(results_dir) if "BIANCA_LOCATE_binarylesionmap" in f]
                    if os.path.exists(results_dir) else []
                )

            if existing:
                pred_path = os.path.join(results_dir, existing[0])
            else:
                print(f"    {test_subject}: no LOCATE result")
                continue

        else:
            # --- Numeric threshold (85 or 90) ---
            thresh_val = int(threshold) / 100.0
            thresh_dir = os.path.join(sub_result_dir, threshold)
            os.makedirs(thresh_dir, exist_ok=True)

            bianca_thresh = os.path.join(
                thresh_dir, f"{test_subject}_BIANCA_LPM_thresh_{threshold}.nii.gz"
            )
            wm_corrected = os.path.join(
                thresh_dir, f"{test_subject}_BIANCA_LPM_thresh_{threshold}_wm_corrected.nii.gz"
            )

            if not os.path.isfile(bianca_thresh):
                for attempt in range(MAX_RETRIES):
                    try:
                        threshold_lpm(LPM_to_use, bianca_thresh, thresh_val)
                        if os.path.isfile(bianca_thresh):
                            break
                    except Exception as e:
                        print(f"    Attempt {attempt+1} thresholding {test_subject}: {e}")
                        time.sleep(RETRY_DELAY)

            if not os.path.isfile(wm_corrected):
                if os.path.isfile(test_WMmask):
                    for attempt in range(MAX_RETRIES):
                        try:
                            apply_bianca_mask(bianca_thresh, test_WMmask, wm_corrected)
                            if os.path.isfile(wm_corrected):
                                break
                        except Exception as e:
                            print(f"    Attempt {attempt+1} WM masking {test_subject}: {e}")
                            time.sleep(RETRY_DELAY)

            pred_path = wm_corrected if os.path.isfile(wm_corrected) else bianca_thresh

        if pred_path is None or not os.path.isfile(pred_path):
            print(f"    {test_subject}: no prediction for threshold {threshold}")
            continue

        # --- 4. Cluster grid search ---
        ensure_valid_gzip(pred_path, pred_path)
        ensure_valid_gzip(test_wmh_roi, test_wmh_roi)

        pred_data = (nib.load(pred_path).get_fdata() > 0).astype(np.int32)
        gt_img = nib.load(test_wmh_roi)
        gt_data = (gt_img.get_fdata() > 0).astype(np.int32)

        results, pred_labeled, gt_labeled, pred_sizes = evaluate_all_mcs_ultra_fast(
            pred_data, gt_data, CLUSTER_SIZES_ALL
        )

        gt_labels_saved = False
        for mcs in CLUSTER_SIZES_ALL:
            metrics = results[mcs]

            # Optional: save NIfTI for inspection
            if SAVE_NIFTI_LABELS and mcs in NIFTI_SAVE_STEPS:
                if not gt_labels_saved:
                    gt_out = os.path.join(sub_result_dir, f"{test_subject}_gt_labeled_clusters.nii.gz")
                    if not os.path.exists(gt_out):
                        nib.save(
                            nib.Nifti1Image(gt_labeled.astype(np.int32), gt_img.affine, gt_img.header),
                            gt_out,
                        )
                    gt_labels_saved = True

                pred_out = os.path.join(
                    sub_result_dir,
                    f"{test_subject}_thresh_{threshold}_mcs_{mcs}_pred_labeled_clusters.nii.gz",
                )
                if not os.path.exists(pred_out):
                    clean = generate_clean_nifti_mask(pred_labeled, pred_sizes, mcs)
                    nib.save(
                        nib.Nifti1Image(clean.astype(np.int32), nib.load(pred_path).affine),
                        pred_out,
                    )

            metrics.update({
                "subject": test_subject,
                "train_condition": CONDITION_NAME,
                "test_condition": CONDITION_NAME,
                "threshold": threshold,
                "min_cluster_size": mcs,
            })
            subject_results.append(metrics)

    return subject_results


# =============================================================
# RUN ONE FOLD (for one seed)
# =============================================================

def _checkpoint_path(seed, fold_idx):
    return os.path.join(CLUSTER_DIR, f"checkpoint_gridsearch_seed_{seed}_fold_{fold_idx + 1}.xlsx")


def fold_already_done(seed, fold_idx):
    p = _checkpoint_path(seed, fold_idx)
    if os.path.isfile(p):
        print(f"  [SKIP] Seed {seed} | Fold {fold_idx + 1} already completed.")
        return True
    return False


def run_fold(seed, fold_idx, scanner_dfs, all_file_df, bianca_pool_df):
    """Run one fold for one seed: train → infer → threshold → grid search."""
    if fold_already_done(seed, fold_idx):
        return pd.read_excel(_checkpoint_path(seed, fold_idx)).to_dict("records")

    scanner_folds = create_stratified_folds(scanner_dfs, seed)
    train_all, test_all = get_fold_subjects(scanner_folds, fold_idx)

    print(f"\n{'=' * 60}")
    print(f"Seed {seed} | Fold {fold_idx + 1}/{N_FOLDS}: "
          f"Train={len(train_all)}, Test={len(test_all)}, Jobs={N_JOBS}")
    print("=" * 60)

    fold_dir = os.path.join(CV_SET_BASE_DIR, f"seed_{seed}", f"fold_{fold_idx + 1}")
    os.makedirs(os.path.join(fold_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "test"), exist_ok=True)

    # Save subject lists
    with open(os.path.join(fold_dir, "train_subjects.txt"), "w") as f:
        f.write("\n".join(sorted(train_all)))
    with open(os.path.join(fold_dir, "test_subjects.txt"), "w") as f:
        f.write("\n".join(sorted(test_all)))

    # --- Train BIANCA ---
    model_path, n_train = train_bianca_model(fold_dir, train_all, all_file_df)

    # --- Process each test subject ---
    all_results = []
    for ti, test_subject in enumerate(test_all):
        print(f"\n  [{ti + 1}/{len(test_all)}] {test_subject}")
        try:
            subj_results = process_test_subject(
                test_subject, fold_dir, model_path, n_train,
                all_file_df, bianca_pool_df,
            )
            # Add seed/fold info
            for r in subj_results:
                r["seed"] = seed
                r["fold"] = fold_idx + 1
            all_results.extend(subj_results)
        except Exception as e:
            print(f"    ERROR {test_subject}: {e}")
            continue

    # Save checkpoint
    os.makedirs(CLUSTER_DIR, exist_ok=True)
    pd.DataFrame(all_results).to_excel(_checkpoint_path(seed, fold_idx), index=False)
    print(f"\n  Checkpoint saved: seed {seed}, fold {fold_idx + 1} ({len(all_results)} rows)")

    return all_results


# =============================================================
# MERGE + REPORT
# =============================================================

def merge_checkpoints():
    """Merge all checkpoint files and report optimal MCS per threshold."""
    os.makedirs(CLUSTER_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CLUSTER_DIR, "checkpoint_gridsearch_seed_*_fold_*.xlsx")))
    if not files:
        print("No checkpoint files found.")
        return

    results_df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)

    # Separate by dataset prefix for dataset-specific MCS
    results_df["dataset"] = results_df["subject"].apply(
        lambda x: "belove" if x.startswith("belove") else "challenge"
    )

    # --- Global best (averaged over all seeds and folds) ---
    avg_global = (
        results_df.groupby(["threshold", "min_cluster_size"])[
            ["lesion_f1", "lesion_precision", "lesion_recall"]
        ]
        .mean()
        .reset_index()
    )

    print(f"\n{'=' * 60}")
    print(f"GLOBAL BEST MCS (averaged over {len(files)} checkpoints)")
    print("=" * 60)
    for th in sorted(avg_global["threshold"].unique()):
        sub = avg_global[avg_global["threshold"] == th]
        best = sub.loc[sub["lesion_f1"].idxmax()]
        print(
            f"  {th}: MCS={int(best['min_cluster_size'])}v, "
            f"F1={best['lesion_f1']:.4f}, P={best['lesion_precision']:.4f}, "
            f"R={best['lesion_recall']:.4f}"
        )

    # --- Per-dataset best ---
    avg_dataset = (
        results_df.groupby(["dataset", "threshold", "min_cluster_size"])[
            ["lesion_f1", "lesion_precision", "lesion_recall"]
        ]
        .mean()
        .reset_index()
    )

    print(f"\n{'=' * 60}")
    print("DATASET-SPECIFIC BEST MCS")
    print("=" * 60)
    mcs_lookup = {}
    for ds in ["belove", "challenge"]:
        for th in sorted(avg_dataset["threshold"].unique()):
            sub = avg_dataset[(avg_dataset["dataset"] == ds) & (avg_dataset["threshold"] == th)]
            if sub.empty:
                continue
            best = sub.loc[sub["lesion_f1"].idxmax()]
            mcs_val = int(best["min_cluster_size"])
            mcs_lookup[(ds, th)] = mcs_val
            print(
                f"  ({ds}, {th}): MCS={mcs_val}v, "
                f"F1={best['lesion_f1']:.4f}, P={best['lesion_precision']:.4f}, "
                f"R={best['lesion_recall']:.4f}"
            )

    # --- Save ---
    output_path = os.path.join(CLUSTER_DIR, "cluster_size_grid_search_all_seeds.xlsx")
    results_df.to_excel(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(results_df)} rows)")

    # Save MCS lookup as JSON for easy import
    mcs_json = {f"{k[0]}_{k[1]}": v for k, v in mcs_lookup.items()}
    mcs_json_path = os.path.join(CLUSTER_DIR, "optimal_mcs_lookup.json")
    with open(mcs_json_path, "w") as f:
        json.dump(mcs_json, f, indent=2)
    print(f"Saved MCS lookup: {mcs_json_path}")

    return mcs_lookup


# =============================================================
# MAIN
# =============================================================

def main():
    os.makedirs(CLUSTER_DIR, exist_ok=True)
    os.makedirs(CV_SET_BASE_DIR, exist_ok=True)

    if "--merge" in sys.argv:
        merge_checkpoints()
        return

    # Load data once
    bianca_pool_df, bianca_pool_meta, all_file_df, scanner_dfs = load_data()
    print(f"Bianca pool: {len(bianca_pool_meta)} subjects")
    for sn in SCANNER_NAMES:
        print(f"  {sn}: {len(scanner_dfs[sn])}")

    # SLURM array mode: one fold per task, all seeds
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_task is not None:
        fold_idx = int(array_task)
        print(f"SLURM mode: Fold {fold_idx + 1}, all {len(SEEDS)} seeds")
        for seed in SEEDS:
            run_fold(seed, fold_idx, scanner_dfs, all_file_df, bianca_pool_df)
        return

    # Standalone mode: all seeds, all folds
    print(f"Standalone mode: {len(SEEDS)} seeds x {N_FOLDS} folds")
    for seed in SEEDS:
        for fold_idx in range(N_FOLDS):
            run_fold(seed, fold_idx, scanner_dfs, all_file_df, bianca_pool_df)

    merge_checkpoints()


if __name__ == "__main__":
    main()