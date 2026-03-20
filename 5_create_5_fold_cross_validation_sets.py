#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

5-Fold Stratified Cross-Validation Pipeline
============================================

Revision context (NeuroImage: Clinical, Major Revision)
-------------------------------------------------------
This is the main evaluation pipeline. It addresses the following
grouped reviewer concerns:

  (1) Zero-filling / lesion replacement (R1 Comment 1; R5 #9)
      R1 criticized zero-intensity replacement as non-physiological.
      R5 noted inpainting alternatives were acknowledged but not tested.

      Resolution: Three preprocessing conditions are evaluated:
        - non_removed: no lesion preprocessing
        - removed: zero-intensity lesion replacement
        - inpainted (filled): NAWM-based inpainting (FSL lesion_filling)
      All train x test combinations (3 x 3 = 9) are computed per fold.

  (2) Cross-validation design (R1 Comment 3; R5 #7)
      LOO-CV replaced with stratified 5-fold CV, balanced by scanner
      type (~1/3 each: Philips, Tim Trio, Prisma fit) and WMH severity.
      Repeated across 10 random seeds for robustness assessment.

  (3) Lesion-level metrics (R1 Comment 2; R5 #17)
      Cluster-based Precision, Recall, F1 added alongside voxel-level
      Dice, sensitivity, precision. Minimum cluster size (MCS) determined
      empirically via grid search (Script 4), dataset- and threshold-
      specific. Any-overlap criterion, consistent with MICCAI 2017
      (Kuijf et al., 2019).

  (4) Multiple comparisons transparency (R5 #5)
      Bonferroni correction families documented in supplemental material.

Paper changes
-------------
  Methods section 2.4 (revised): "Cross-validation design"
    - LOO-CV replaced with stratified 5-fold CV (10 seeds).
    - Scanner and severity stratification described.

  Methods section 2.5 (new): "Cluster-based lesion-level evaluation"
    - Connected-component labeling, any-overlap criterion, empirically
      determined MCS from grid search.

  Methods section 2.7 (revised): "Algorithm configuration optimization"
    - Three conditions (non_removed, removed, inpainted) evaluated.
    - Three thresholds (0.85, 0.90, LOCATE) compared.

  Results section 3.1 (revised):
    - Voxel-level AND lesion-level metrics for all conditions.
    - Language revised per reviewer feedback (no "confirmed true
      positives", no "clinical relevance").

  Response to Reviewers:
    - R1 Comment 1: Inpainted condition added; convergence across
      all three conditions demonstrated.
    - R1 Comment 2: Cluster-based metrics added; language revised.
    - R1 Comment 3 / R5 #7: LOO-CV replaced with stratified 5-fold CV.
    - R5 #9: Inpainting now tested alongside zero-filling.
    - R5 #17: Precision contextualized; cluster-level metrics added.

Technical design
----------------
Per seed (10 seeds) x per fold (5 folds):
  1. Stratified scanner-balanced splits
  2. BIANCA training for each condition (non_removed, removed, filled)
  3. Inference on held-out test subjects
  4. Lesion correction (stroke lesion masking)
  5. Thresholding (0.85, 0.90) + LOCATE adaptive thresholding
  6. Voxel-level metrics (Dice, sensitivity, precision)
  7. Cluster-based metrics (lesion-level F1, Precision, Recall)
  8. Per-subject CSV caching + DONE/IN_PROGRESS locking for SLURM

MCS values from Script 4 (grid search) are imported via MCS_LOOKUP.
Directory structure is shared with Script 4 to avoid re-computation.

Terminology (consistent across all scripts):
  - non_removed: no lesion preprocessing
  - removed: zero-intensity lesion replacement
  - inpainted / filled: NAWM-based inpainting (FSL lesion_filling)

References:
  Kuijf et al. (2019). IEEE TMI, 38(11).
  Griffanti et al. (2016). NeuroImage: Clinical, 9, 235-242.
  Sundaresan et al. (2019). NeuroImage, 202, 116056.
  Ferris et al. (2023). BIANCA in chronic stroke populations.




"""

import os
import json
import pandas as pd
import numpy as np
import subprocess
import time
import glob
import nibabel as nib
import gzip
import subprocess
import os
from typing import List
import tempfile

from cliffs_delta import cliffs_delta
from sklearn.model_selection import StratifiedKFold
from DATASETS.librarys.directory_lib import fsl_copy, get_volume, threshold_lpm
from DATASETS.librarys.directory_lib import apply_bianca_mask, fsl_dice_score
from DATASETS.librarys.directory_lib import create_panel_plot_with_multi_overlay, get_top_slices
from DATASETS.librarys.directory_lib import get_files_from_dir


def ensure_valid_gzip(src, dst):
    """
    Checks if 'dst' is a valid gzip file. 
    If it is invalid or missing, it runs fslmaths to recreate it from 'src'.
    """
    # GUARD: Only process .gz files. If it's a .mat or .nii, it's not compressed.
    if not str(dst).endswith('.gz'):
        return True 

    is_valid = False
    
    # 1. Check if the destination file exists and is a valid gzip
    if os.path.exists(dst):
        try:
            with gzip.open(dst, 'rb') as f:
                f.read(2)
            is_valid = True
        except (gzip.BadGzipFile, OSError, EOFError):
            is_valid = False
            
    # If it's valid, we are done
    if is_valid:
        # print(f"Valid gzip found at: {dst}") # You can uncomment this if you want the logs back
        return True

    # 2. If not valid (or missing), replace/create it using fslmaths
    print(f"Invalid or missing file at {dst}. Recreating with fslmaths...")
    cmd = ["fslmaths", src, "-mul", "1", dst]
    
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"fslmaths failed with error:\n{e.stderr}")
        return False
        
    except FileNotFoundError:
        print("Error: 'fslmaths' command not found. Is FSL sourced in your environment?")
        return False


# =============================================================
# MINIMUM CLUSTER SIZE LOOKUP (from grid search)
# =============================================================
# Determined via grid search over MCS 1-100, averaged across 10 seeds
# and 5-fold CV. Optimal = MCS maximizing lesion-level F1.
# Any-overlap criterion, consistent with MICCAI 2017 (Kuijf et al., 2019).
# Filtering applied ONLY to predictions; GT retained without filtering.
# =============================================================
MCS_LOOKUP = {
    # (dataset_prefix, threshold_str) -> optimal minimum cluster size (voxels)
    ("belove",    "85"):     3,
    ("belove",    "90"):     2,
    ("belove",    "locate"): 16,
    ("challenge", "85"):     4,
    ("challenge", "90"):     2,
    ("challenge", "locate"): 8,
}

MCS_FALLBACK = {"85": 3, "90": 2, "locate": 16}


def get_mcs(subject, threshold_str):
    """Look up optimal minimum cluster size for a subject and threshold."""
    dataset = "belove" if subject.startswith("belove") else "challenge"
    return MCS_LOOKUP.get((dataset, threshold_str), MCS_FALLBACK.get(threshold_str, 3))


def cluster_based_metrics(pred_path, gt_path, min_cluster_size=10, overlap_thresh=1):
    """
    Lesion-level Precision, Recall, F1 using connected components.

    Uses 26-connectivity (full 3D neighborhood), consistent with the
    grid search used to determine optimal minimum cluster size.

    Filtering is applied ONLY to prediction. Ground truth clusters are
    retained regardless of size (expert-delineated lesions).

    overlap_thresh=1 (default): any-overlap (MICCAI 2017).
    overlap_thresh in (0,1): fractional overlap.
    """
    from scipy.ndimage import label as scipy_label, generate_binary_structure

    ensure_valid_gzip(pred_path, pred_path)
    ensure_valid_gzip(gt_path, gt_path)

    pred_bin = (nib.load(pred_path).get_fdata() > 0).astype(np.int32)
    gt_bin   = (nib.load(gt_path).get_fdata() > 0).astype(np.int32)

    # 26-connectivity (full 3D), consistent with grid search
    struct = generate_binary_structure(3, 3)

    # Label prediction and filter small clusters
    pred_labeled, n_pred_raw = scipy_label(pred_bin, structure=struct)
    for i in range(1, n_pred_raw + 1):
        if (pred_labeled == i).sum() < min_cluster_size:
            pred_labeled[pred_labeled == i] = 0
    pred_labeled, n_pred = scipy_label(pred_labeled > 0, structure=struct)

    # Label GT (NO filtering)
    gt_labeled, n_gt = scipy_label(gt_bin, structure=struct)

    def _is_tp(cluster_mask, reference_bin, thresh):
        cluster_size = cluster_mask.sum()
        if cluster_size == 0:
            return False
        overlap_count = (reference_bin[cluster_mask] > 0).sum()
        if isinstance(thresh, float) and thresh < 1.0:
            return (overlap_count / cluster_size) >= thresh
        return overlap_count >= thresh

    # Precision side: pred clusters overlapping GT
    tp_pred = 0
    for i in range(1, n_pred + 1):
        if _is_tp(pred_labeled == i, gt_bin, overlap_thresh):
            tp_pred += 1

    # Recall side: GT clusters overlapping filtered prediction
    pred_filtered_bin = (pred_labeled > 0).astype(np.int32)
    tp_gt = 0
    for i in range(1, n_gt + 1):
        if _is_tp(gt_labeled == i, pred_filtered_bin, overlap_thresh):
            tp_gt += 1

    lesion_precision = round(tp_pred / n_pred, 4) if n_pred > 0 else 0.0
    lesion_recall    = round(tp_gt / n_gt, 4) if n_gt > 0 else 0.0
    lesion_f1 = (
        round(2 * lesion_precision * lesion_recall /
              (lesion_precision + lesion_recall), 4)
        if (lesion_precision + lesion_recall) > 0 else 0.0
    )

    return {
        'lesion_precision': lesion_precision,
        'lesion_recall':    lesion_recall,
        'lesion_f1':        lesion_f1,
        'n_pred_clusters':  n_pred,
        'n_gt_clusters':    n_gt,
        'n_tp_pred':        tp_pred,
        'n_tp_gt':          tp_gt,
        'n_fp_clusters':    n_pred - tp_pred,
        'n_fn_clusters':    n_gt - tp_gt,
        'min_cluster_size': min_cluster_size,
    }


def get_valid_path(base, primary_col, fallback_col, row):
    p1 = os.path.join(base, str(row[primary_col])) if pd.notna(row[primary_col]) else ""
    if p1 and os.path.isfile(p1):
        return p1
    return os.path.join(base, str(row[fallback_col])) if pd.notna(row[fallback_col]) else ""


def run_locate_testing(
    train_image_directory_path: str,
    test_image_directory_name: str,
    locate_path: str,
    verbose: int = 1,
    feature_select: List[int] = [1, 1, 1, 1]
) -> None:
    matlab_script_content = f"""
    addpath(genpath('{locate_path}'));
    feature_select = {feature_select};
    verbose = {verbose};
    train_image_directory_path = '{train_image_directory_path}';
    test_image_directory_name = '{test_image_directory_name}';
    try
        LOCATE_testing(test_image_directory_name, train_image_directory_path, feature_select, verbose);
    catch ME
        fprintf('Error: %s\\n', ME.message);
        exit(1);
    end
    exit(0);
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".m") as temp_file:
        script_path = temp_file.name
        temp_file.write(matlab_script_content)

    try:
        return_code = os.system(f"matlab -batch \"run('{script_path}');\"")
        if return_code != 0:
            raise RuntimeError(f"LOCATE testing failed with return code {return_code}")
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def reconstruct_thresh_data(sub_bianca_result_dir, test_subject, thresholds_list):
    thresh_paths = {}
    thresh_dice  = {}
    for thresh in thresholds_list:
        if thresh == 'locate':
            results_dir = os.path.join(sub_bianca_result_dir, "locate", "LOCATE_results_directory")
            if os.path.exists(results_dir):
                loc_files = [f for f in os.listdir(results_dir) if "BIANCA_LOCATE_binarylesionmap" in f]
                if loc_files:
                    thresh_paths['locate'] = os.path.join(results_dir, loc_files[0])
            csv_p = os.path.join(sub_bianca_result_dir, "locate", f"{test_subject}_metrics_locate.csv")
            if os.path.isfile(csv_p):
                try:
                    thresh_dice['locate'] = pd.read_csv(csv_p).iloc[0].get('dice_score', None)
                except:
                    pass
        else:
            ts = str(int(thresh * 100))
            corr_path = os.path.join(sub_bianca_result_dir, ts,
                f"{test_subject}_BIANCA_LPM_thresh_{ts}_wm_corrected.nii.gz")
            if os.path.isfile(corr_path):
                thresh_paths[ts] = corr_path
            csv_p = os.path.join(sub_bianca_result_dir, ts, f"{test_subject}_metrics_{ts}.csv")
            if os.path.isfile(csv_p):
                try:
                    thresh_dice[ts] = pd.read_csv(csv_p).iloc[0].get('dice_score', None)
                except:
                    pass
    return thresh_paths, thresh_dice


def make_panel_plot(test_FLAIR_path, test_wmh_roi_file, lesion_path_full,
                    thresh_paths, thresh_dice, plot_output_path,
                    test_subject, combo_name):
    if not (os.path.isfile(test_FLAIR_path) and os.path.isfile(test_wmh_roi_file) and os.path.isfile(lesion_path_full)):
        return
    top_wmh_slices_list = get_top_slices(lesion_path_full, top_n=1)
    images = [
        (test_FLAIR_path, []),
        (test_FLAIR_path, [test_wmh_roi_file, lesion_path_full]),
    ]
    titles_list = ["FLAIR ", "GT and Lesion", "85", "90", "locate"]
    for tkey in ['85', '90', 'locate']:
        if tkey in thresh_paths and os.path.isfile(thresh_paths[tkey]):
            images.append((test_FLAIR_path, [thresh_paths[tkey]]))
            titles_list.append(f"locate" if tkey == 'locate' else f"{tkey}")
    if len(images) > 2:
        create_panel_plot_with_multi_overlay(
            images=images, out_path=plot_output_path, subject_name=test_subject,
            title=f"", titles_list=titles_list, slice_indices=top_wmh_slices_list, dpi=120,
        )
        print(f"    Plot saved: {plot_output_path}")


# =============================================================
# CONFIG
# =============================================================
DEBUG = True
DEBUG = False

DELETE = False


PLOT = False
SAVE_METRICS = True
MAX_RETRIES = 30
RETRY_DELAY = 2


seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
if DEBUG:
    seeds = [1]

N_FOLDS = 5
SCANNER_NAMES = ['Philips', 'Tim Trio', 'Prisma_fit']
thresholds_list = [0.85, 0.90, 'locate']

train_image_directory_path = '/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/DATASET_SC/PROJECT_NULL_07_04_2025/1_Project_all_code_all_data/Phase_1/LOCATE_SET/locate_train'
LOCATE_PATH                = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Projects/BIANCA/LOCATE/LOCATE-BIANCA"
feature_select = [1, 1, 1, 1]

bianca_pool       = "Phase_1/LOCATE_SET/bianca_pool_wihtouth_ge.xlsx"
bianca_pool_df    = pd.read_excel(bianca_pool)
CV_SET_base_dir = "Phase_1/5FCV_SET"
all_metrics_result_dir = os.path.join(CV_SET_base_dir, "bianca_result_dir")
os.makedirs(all_metrics_result_dir, exist_ok=True)
png_images = os.path.join(CV_SET_base_dir, "png_images")
os.makedirs(png_images, exist_ok=True)

splits_json_path = os.path.join(CV_SET_base_dir, "experiment_splits.json")
if os.path.exists(splits_json_path):
    with open(splits_json_path, 'r') as f:
        all_splits = json.load(f)
else:
    all_splits = {}

BELOVE_BASE    = "DATASETS/BELOVE_BIDS_WMH_BIDS"
CHALLENGE_BASE = "DATASETS/CHALLENGE_BIDS_WMH_BIDS"
belove_file_path    = os.path.join(BELOVE_BASE, "derivatives/preprocessed_files.xlsx")
challenge_file_path = os.path.join(CHALLENGE_BASE, "derivatives/preprocessed_files.xlsx")
belove_df    = pd.read_excel(belove_file_path)
challenge_df = pd.read_excel(challenge_file_path)
all_file_df = pd.concat([belove_df, challenge_df], ignore_index=True)

all_file_df['manual_mask'] = np.where(
    all_file_df['WMH_removed_path'].notna(),
    all_file_df['WMH_removed_path'],
    all_file_df['WMH_path']
)
if 'FLAIR_non_removed_path' not in all_file_df.columns:
    all_file_df['FLAIR_non_removed_path'] = all_file_df['FLAIR_brain_biascorr']
else:
    all_file_df['FLAIR_non_removed_path'] = all_file_df['FLAIR_non_removed_path'].fillna(all_file_df['FLAIR_brain_biascorr'])

bianca_pool_df['manual_mask'] = np.where(
    bianca_pool_df['WMH_removed_path'].notna(),
    bianca_pool_df['WMH_removed_path'],
    bianca_pool_df['wmh_roi_file']
)
meta_columns = ["subject", "dataset_base", "scanner", "sex", "age",
                "lesion_type", "has_lesion", "severity_level"]
bianca_pool_df = bianca_pool_df[meta_columns]
base_mapping = bianca_pool_df[['subject', 'dataset_base']].drop_duplicates()
all_file_df = all_file_df.merge(base_mapping, on='subject', how='left')
all_file_df = all_file_df.drop_duplicates(subset=['subject'])

scanner_dfs = {}
for scanner_name in SCANNER_NAMES:
    scanner_dfs[scanner_name] = bianca_pool_df[bianca_pool_df['scanner'] == scanner_name].reset_index(drop=True)
    print(f"  {scanner_name}: {len(scanner_dfs[scanner_name])} subjects")
meta_df = bianca_pool_df[["subject", "scanner", "severity_level"]]


def stratified_downsample(pool_df, n_sample, rng):
    severity_counts = pool_df['severity_level'].value_counts()
    total = len(pool_df)
    allocation = {}
    for sev in severity_counts.index:
        allocation[sev] = int(np.floor(n_sample * severity_counts[sev] / total))
    remainder = n_sample - sum(allocation.values())
    sevs = sorted(list(severity_counts.index))
    rng.shuffle(sevs)
    for i in range(remainder):
        allocation[sevs[i % len(sevs)]] += 1
    sampled = []
    for sev, n in allocation.items():
        pool = pool_df[pool_df['severity_level'] == sev]['subject'].tolist()
        chosen = rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()
        sampled.extend(chosen)
    return sampled


def make_fold_summary(seed, fold_idx, train_all, test_all, df, scanner_dfs):
    train_df = df[df['subject'].isin(train_all)]
    test_df  = df[df['subject'].isin(test_all)]
    lesion_counts = train_df['lesion_type'].value_counts().to_dict()
    has_lesion_n = train_df['has_lesion'].sum() if 'has_lesion' in train_df.columns else "N/A"
    print(f"  Fold {fold_idx+1}: ... | Lesion types: {lesion_counts} | has_lesion={has_lesion_n}")
    tr_scan = train_df['scanner'].value_counts().to_dict()
    te_scan = test_df['scanner'].value_counts().to_dict()
    tr_sev  = train_df['severity_level'].value_counts().to_dict()
    te_sev  = test_df['severity_level'].value_counts().to_dict()
    tr_cross = pd.crosstab(train_df['scanner'], train_df['severity_level'])
    te_cross = pd.crosstab(test_df['scanner'], test_df['severity_level'])
    n_train, n_test = len(train_all), len(test_all)
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  seed_{seed} / fold_{fold_idx+1}")
    lines.append(f"{'='*60}")
    lines.append("")
    lines.append(f"TOTALS:  Train={n_train}  |  Test={n_test}  |  Total={n_train+n_test}")
    lines.append("")
    lines.append("SCANNER DISTRIBUTION")
    lines.append(f"  {'':15s} {'Train':>8s} {'Test':>8s} {'Total':>8s}")
    lines.append(f"  {'-'*41}")
    for s in SCANNER_NAMES:
        tr, te = tr_scan.get(s, 0), te_scan.get(s, 0)
        lines.append(f"  {s:15s} {tr:>8d} {te:>8d} {tr+te:>8d}")
    lines.append(f"  {'-'*41}")
    lines.append(f"  {'Total':15s} {n_train:>8d} {n_test:>8d} {n_train+n_test:>8d}")
    lines.append("")
    lines.append("SEVERITY DISTRIBUTION")
    lines.append(f"  {'':15s} {'Train':>8s} {'Test':>8s} {'Total':>8s}")
    lines.append(f"  {'-'*41}")
    for s in ['high', 'middle', 'low']:
        tr, te = tr_sev.get(s, 0), te_sev.get(s, 0)
        lines.append(f"  {s:15s} {tr:>8d} {te:>8d} {tr+te:>8d}")
    lines.append(f"  {'-'*41}")
    lines.append(f"  {'Total':15s} {n_train:>8d} {n_test:>8d} {n_train+n_test:>8d}")
    lines.append("")
    lines.append("TRAIN - Scanner x Severity")
    lines.append(f"  {'':15s} {'high':>8s} {'middle':>8s} {'low':>8s} {'Total':>8s}")
    lines.append(f"  {'-'*49}")
    for s in SCANNER_NAMES:
        vals = [tr_cross.loc[s, c] if s in tr_cross.index and c in tr_cross.columns else 0 for c in ['high', 'middle', 'low']]
        lines.append(f"  {s:15s} {vals[0]:>8d} {vals[1]:>8d} {vals[2]:>8d} {sum(vals):>8d}")
    lines.append("")
    lines.append("TEST - Scanner x Severity")
    lines.append(f"  {'':15s} {'high':>8s} {'middle':>8s} {'low':>8s} {'Total':>8s}")
    lines.append(f"  {'-'*49}")
    for s in SCANNER_NAMES:
        vals = [te_cross.loc[s, c] if s in te_cross.index and c in te_cross.columns else 0 for c in ['high', 'middle', 'low']]
        lines.append(f"  {s:15s} {vals[0]:>8d} {vals[1]:>8d} {vals[2]:>8d} {sum(vals):>8d}")
    lines.append("")
    for label, subj_list in [("TRAIN", train_all), ("TEST", test_all)]:
        lines.append(f"{label} SUBJECTS ({len(subj_list)})")
        for scanner_name in SCANNER_NAMES:
            scanner_subjects = sorted([s for s in subj_list if s in scanner_dfs[scanner_name]['subject'].values])
            lines.append(f"  {scanner_name} ({len(scanner_subjects)}): {', '.join(scanner_subjects)}")
        lines.append("")
    return '\n'.join(lines)


def create_stratified_folds(scanner_df, n_folds, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X = scanner_df['subject'].values
    y = scanner_df['severity_level'].values
    folds = {}
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        folds[fold_idx] = {'train': X[train_idx].tolist(), 'test': X[test_idx].tolist()}
    return folds


train_conditions = [
    {"name": "non_removed", "primary_col": "FLAIR_non_removed_path"},
    {"name": "removed",     "primary_col": "FLAIR_removed_path"},
    {"name": "filled",      "primary_col": "FLAIR_filled_path"},
]
test_conditions = [
    {"name": "non_removed", "primary_col": "FLAIR_non_removed_path"},
    {"name": "removed",     "primary_col": "FLAIR_removed_path"},
    {"name": "filled",      "primary_col": "FLAIR_filled_path"},
]
# =============================================================
# MAIN LOOP
# =============================================================


if DELETE:
    import sys
    for seed in seeds:
        seed_dir = os.path.join(CV_SET_base_dir, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            continue
        
        seed_excel = os.path.join(all_metrics_result_dir, f"bianca_metrics_seed_{seed}.xlsx")
        if os.path.isfile(seed_excel):
            os.remove(seed_excel)
            print(f"Deleted: {seed_excel}")
        
        deleted_done = 0
        deleted_prog = 0
        
        for dirpath, dirnames, filenames in os.walk(seed_dir):
            # IN_PROGRESS löschen
            if "IN_PROGRESS.txt" in filenames:
                try:
                    os.remove(os.path.join(dirpath, "IN_PROGRESS.txt"))
                    deleted_prog += 1
                except FileNotFoundError:
                    pass
            
            # DONE.txt nur löschen wenn < 3 valide CSVs
            if "DONE.txt" in filenames:
                csv_count = 0
                for fn in filenames:
                    if "_metrics_" in fn and fn.endswith(".csv"):
                        fp = os.path.join(dirpath, fn)
                        if os.path.getsize(fp) > 0:
                            csv_count += 1
                # Auch in Unterordnern (85/, 90/, locate/) checken
                for sd in dirnames:
                    sd_path = os.path.join(dirpath, sd)
                    if os.path.isdir(sd_path):
                        for fn in os.listdir(sd_path):
                            if "_metrics_" in fn and fn.endswith(".csv"):
                                fp = os.path.join(sd_path, fn)
                                if os.path.getsize(fp) > 0:
                                    csv_count += 1
                
                if csv_count < 3:
                    try:
                        os.remove(os.path.join(dirpath, "DONE.txt"))
                        deleted_done += 1
                    except FileNotFoundError:
                        pass
        
        print(f"Seed {seed}: deleted {deleted_done} DONE.txt, {deleted_prog} IN_PROGRESS.txt")
    
    print("\nCleanup done. Set DELETE = False and rerun.")
    sys.exit(0)




summary_rows = []

for seed in seeds:
    all_metrics_results = []

    if not DEBUG:
        seed_excel_path = os.path.join(all_metrics_result_dir, f"bianca_metrics_seed_{seed}.xlsx")
        if os.path.isfile(seed_excel_path):
            print(f"Seed {seed} already done, skipping")
            continue

    print(f"\n{'='*60}")
    print(f"=== Seed {seed} ===")
    print(f"{'='*60}")

    seed_splits = {}
    scanner_folds = {}
    for scanner_name in SCANNER_NAMES:
        scanner_folds[scanner_name] = create_stratified_folds(scanner_dfs[scanner_name], N_FOLDS, seed)

    N_FOLDS_list = list(range(N_FOLDS))
    if DEBUG:
        N_FOLDS_list = N_FOLDS_list[:1]

    for fold_idx in N_FOLDS_list:
        rng = np.random.default_rng(seed * 1000 + fold_idx)

        test_per_scanner = {}
        train_pool_per_scanner = {}
        for scanner_name in SCANNER_NAMES:
            test_per_scanner[scanner_name] = scanner_folds[scanner_name][fold_idx]['test']
            train_pool_per_scanner[scanner_name] = scanner_folds[scanner_name][fold_idx]['train']

        train_per_scanner = {}
        for scanner_name in SCANNER_NAMES:
            train_per_scanner[scanner_name] = train_pool_per_scanner[scanner_name]

        train_all = []
        test_all = []
        for scanner_name in SCANNER_NAMES:
            train_all.extend(train_per_scanner[scanner_name])
            test_all.extend(test_per_scanner[scanner_name])
            

        n_train_total = len(train_all)
        assert len(set(train_all) & set(test_all)) == 0, "DATA LEAK!"

        train_df_fold = bianca_pool_df[bianca_pool_df['subject'].isin(train_all)]
        sev = train_df_fold['severity_level'].value_counts().to_dict()
        lesion_counts = train_df_fold['lesion_type'].value_counts().to_dict()
        has_lesion_n = int(train_df_fold['has_lesion'].sum())

        scanner_train_str = "+".join([f"{len(train_per_scanner[s])}{s[0]}" for s in SCANNER_NAMES])
        scanner_test_str = "+".join([f"{len(test_per_scanner[s])}{s[0]}" for s in SCANNER_NAMES])

        print(f"  Fold {fold_idx+1}: Train={len(train_all)} ({scanner_train_str}) | "
              f"Test={len(test_all)} ({scanner_test_str}) | "
              f"Sev: h={sev.get('high',0)} m={sev.get('middle',0)} l={sev.get('low',0)}")

        fold_dir = os.path.join(CV_SET_base_dir, f"seed_{seed}", f"fold_{fold_idx+1}")
        os.makedirs(os.path.join(fold_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "test"), exist_ok=True)

        with open(os.path.join(fold_dir, "train_subjects.txt"), 'w') as f:
            f.write('\n'.join(sorted(train_all)))
        with open(os.path.join(fold_dir, "test_subjects.txt"), 'w') as f:
            f.write('\n'.join(sorted(test_all)))

        summary_text = make_fold_summary(seed, fold_idx, train_all, test_all, bianca_pool_df, scanner_dfs)
        with open(os.path.join(fold_dir, "fold_summary.txt"), 'w') as f:
            f.write(summary_text)

        # =====================================================
        # 1. BIANCA TRAINING
        # =====================================================
        for train_cond in train_conditions:
            cond_name = train_cond["name"]
            primary_col = train_cond["primary_col"]
            master_file_text_lines = []

            for train_subject in train_all:
                subject_rows = all_file_df[all_file_df['subject'] == train_subject]
                if subject_rows.empty:
                    raise ValueError(f"Subject {train_subject} missing from all_file_df!")
                train_row = subject_rows.iloc[0]
                dataset_base = train_row['dataset_base']

                train_FLAIR        = get_valid_path(dataset_base, primary_col, 'FLAIR_non_removed_path', train_row)
                train_T1           = os.path.join(dataset_base, train_row['T1'])
                train_mni_mat_path = os.path.join(dataset_base, train_row['mni_mat_path'])
                train_wmh_roi_file = os.path.join(dataset_base, train_row['manual_mask'])

                train_FLAIR        = train_FLAIR if os.path.isfile(train_FLAIR) else ""
                train_T1           = train_T1 if os.path.isfile(train_T1) else ""
                train_mni_mat_path = train_mni_mat_path if os.path.isfile(train_mni_mat_path) else ""
                train_wmh_roi_file = train_wmh_roi_file if os.path.isfile(train_wmh_roi_file) else ""

                if not all([train_FLAIR, train_T1, train_mni_mat_path, train_wmh_roi_file]):
                    missing = []
                    if not train_FLAIR: missing.append(primary_col)
                    if not train_T1: missing.append("T1")
                    if not train_mni_mat_path: missing.append("mni_mat_path")
                    if not train_wmh_roi_file: missing.append("manual_mask")
                    raise FileNotFoundError(f"Missing for {train_subject} in '{cond_name}': {', '.join(missing)}")

                master_file_text_lines.append(f"{train_FLAIR} {train_T1} {train_mni_mat_path} {train_wmh_roi_file}")

            train_master_file = os.path.join(fold_dir, f'bianca_n_{n_train_total}_balanced_train_master_file_{cond_name}.txt')
            with open(train_master_file, 'w') as f:
                f.write('\n'.join(master_file_text_lines))

            model_path = os.path.join(fold_dir, f"BIANCA_MODEL_N_{n_train_total}_{cond_name.upper()}")

            if not os.path.isfile(model_path):
                row_number = len(master_file_text_lines)
                trainstring = ",".join([str(r) for r in range(1, row_number + 1)])
                train_bianca_commands = [
                    "bianca",
                    f"--singlefile={train_master_file}",
                    "--brainmaskfeaturenum=1", "--matfeaturenum=3",
                    "--featuresubset=1,2", "--labelfeaturenum=4",
                    "--trainingpts=2000", "--nonlespts=10000",
                    f"--trainingnums={trainstring}",
                    f"--saveclassifierdata={model_path}",
                    f"--querysubjectnum={row_number}", "-v"
                ]
                try:
                    print(f"    BIANCA Training: {cond_name} (N={n_train_total})")
                    subprocess.run(train_bianca_commands, check=True)
                    print(f"    Training done")
                except subprocess.CalledProcessError as e:
                    print(f"    Training error: seed {seed}, fold {fold_idx+1}, condition {cond_name}")
                    raise

        # =====================================================
        # 2. TESTING LOOP
        # =====================================================
        for test_subject in test_all:
            sub_dir = os.path.join(fold_dir, "test", test_subject)
            os.makedirs(sub_dir, exist_ok=True)

            subject_rows = all_file_df[all_file_df['subject'] == test_subject]
            if subject_rows.empty:
                print(f"  {test_subject}: missing from all_file_df, skipping")
                continue
            test_row = subject_rows.iloc[0]
            dataset_base = test_row['dataset_base']

            lesion_path = test_row.get('lesion_path', None)
            has_lesion = pd.notna(lesion_path) and str(lesion_path).strip() != ""

            subject_meta_rows = bianca_pool_df[bianca_pool_df['subject'] == test_subject]
            if subject_meta_rows.empty:
                print(f"  {test_subject}: missing from bianca_pool_df, skipping")
                continue

            subject_meta_row = subject_meta_rows.iloc[0]
            scanner_group = subject_meta_row["scanner"]
            severity_level = subject_meta_row["severity_level"]

            for train_cond in train_conditions:
                train_cond_name = train_cond["name"]

                for test_cond in test_conditions:
                    test_cond_name = test_cond["name"]
                    test_primary_col = test_cond["primary_col"]
                    combo_name = f"train_{train_cond_name}__test_{test_cond_name}"

                    sub_bianca_result_dir = os.path.join(sub_dir, f"bianca_result_{combo_name}")
                    os.makedirs(sub_bianca_result_dir, exist_ok=True)

                    # --- IN_PROGRESS: nur hier skippen ---
                    progress_file = os.path.join(sub_bianca_result_dir, "IN_PROGRESS.txt")
                    if os.path.exists(progress_file):
                        file_age_seconds = time.time() - os.path.getmtime(progress_file)
                        if file_age_seconds > 2 * 3600:
                            print(f"  {test_subject} ({combo_name}): Stale lock, removing...")
                            os.remove(progress_file)
                        else:
                            print(f"  {test_subject} ({combo_name}): in progress, skipping...")
                            continue  # NUR hier continue

                    done_file = os.path.join(sub_bianca_result_dir, "DONE.txt")

                    # =================================================
                    # ALREADY DONE: load cached metrics + generate plot
                    # =================================================
                    if os.path.exists(done_file):
                        print(f"  {test_subject} ({combo_name}): already done, loading cache...")

                        if SAVE_METRICS:
                            for thresh in thresholds_list:
                                if thresh == 'locate':
                                    csv_path = os.path.join(sub_bianca_result_dir, "locate",
                                                            f"{test_subject}_metrics_locate.csv")
                                else:
                                    thresh_str = str(int(thresh * 100))
                                    csv_path = os.path.join(sub_bianca_result_dir, thresh_str,
                                                            f"{test_subject}_metrics_{thresh_str}.csv")
                                if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
                                    try:
                                        cached = pd.read_csv(csv_path).iloc[0].to_dict()
                                        all_metrics_results.append(cached)
                                    except pd.errors.EmptyDataError:
                                        pass

                        if PLOT and has_lesion:
                            masks_png_dir = os.path.join(png_images, f"seed_{seed}_fold_{fold_idx}")
                            os.makedirs(masks_png_dir, exist_ok=True)
                            plot_output_path = os.path.join(masks_png_dir, f"{test_subject}_{combo_name}_FLAIR_masks.png")
                            if not os.path.isfile(plot_output_path):
                                test_FLAIR_path   = get_valid_path(dataset_base, test_primary_col, 'FLAIR_non_removed_path', test_row)
                                test_wmh_roi_file = os.path.join(dataset_base, test_row['manual_mask'])
                                lesion_path_full  = (os.path.join(dataset_base, str(lesion_path))
                                    if not os.path.isabs(str(lesion_path)) else str(lesion_path))
                                thresh_paths, thresh_dice = reconstruct_thresh_data(sub_bianca_result_dir, test_subject, thresholds_list)
                                make_panel_plot(test_FLAIR_path, test_wmh_roi_file, lesion_path_full,
                                    thresh_paths, thresh_dice, plot_output_path, test_subject, combo_name)

                    # =================================================
                    # NEW PROCESSING (nur wenn NICHT done)
                    # =================================================
                    else:
                        try:
                            with open(progress_file, 'w') as f:
                                f.write(f"Started: {pd.Timestamp.now()}\nSubject: {test_subject}\nCombo: {combo_name}")

                            test_FLAIR_path       = get_valid_path(dataset_base, test_primary_col, 'FLAIR_non_removed_path', test_row)
                            test_T1               = os.path.join(dataset_base, test_row['T1'])
                            test_mni_mat_path     = os.path.join(dataset_base, test_row['mni_mat_path'])
                            test_wmh_roi_file     = os.path.join(dataset_base, test_row['manual_mask'])
                            test_WMmask_path      = os.path.join(dataset_base, test_row['WMmask'])
                            test_brainmask_path   = os.path.join(dataset_base, test_row['brainmask'])
                            test_ventdistmap_path = os.path.join(dataset_base, test_row['ventdistmap'])

                            test_FLAIR_path       = test_FLAIR_path if os.path.isfile(test_FLAIR_path) else ""
                            test_T1               = test_T1 if os.path.isfile(test_T1) else ""
                            test_mni_mat_path     = test_mni_mat_path if os.path.isfile(test_mni_mat_path) else ""
                            test_wmh_roi_file     = test_wmh_roi_file if os.path.isfile(test_wmh_roi_file) else ""
                            test_WMmask_path      = test_WMmask_path if os.path.isfile(test_WMmask_path) else ""
                            test_brainmask_path   = test_brainmask_path if os.path.isfile(test_brainmask_path) else ""
                            test_ventdistmap_path = test_ventdistmap_path if os.path.isfile(test_ventdistmap_path) else ""

                            if not all([test_FLAIR_path, test_T1, test_mni_mat_path, test_wmh_roi_file]):
                                missing = []
                                if not test_FLAIR_path: missing.append(test_primary_col)
                                if not test_T1: missing.append("T1")
                                if not test_mni_mat_path: missing.append("mni_mat_path")
                                if not test_wmh_roi_file: missing.append("manual_mask")
                                raise FileNotFoundError(f"Missing for {test_subject} in '{combo_name}': {', '.join(missing)}")

                            manual_mask = os.path.join(sub_bianca_result_dir, f"{test_subject}_manual_mask.nii.gz")
                            fsl_copy(test_wmh_roi_file, manual_mask)

                            test_masterfile_line = f"{test_FLAIR_path} {test_T1} {test_mni_mat_path} {test_wmh_roi_file}"
                            test_masterfile_path = os.path.join(sub_bianca_result_dir, f"{test_subject}_masterfile_test.txt")
                            with open(test_masterfile_path, 'w') as f:
                                f.write(test_masterfile_line)

                            model_path = os.path.join(fold_dir, f"BIANCA_MODEL_N_{n_train_total}_{train_cond_name.upper()}")
                            BIANCA_LPM = os.path.join(sub_bianca_result_dir, f"{test_subject}_BIANCA_LPM_{combo_name}.nii.gz")

                            if not os.path.isfile(BIANCA_LPM):
                                test_bianca_commands = [
                                    "bianca",
                                    f"--singlefile={test_masterfile_path}",
                                    "--brainmaskfeaturenum=1", "--matfeaturenum=3",
                                    "--featuresubset=1,2",
                                    f"--loadclassifierdata={model_path}",
                                    "--querysubjectnum=1",
                                    "-o", BIANCA_LPM, "-v"
                                ]
                                subprocess.run(test_bianca_commands, check=True)
                                print(f"    {test_subject} ({combo_name}): LPM created")

                            # --- Lesion correction ---
                            BIANCA_LPM_corrected = os.path.join(
                                sub_bianca_result_dir, f"{test_subject}_BIANCA_LPM_{combo_name}_corrected.nii.gz")
                            if has_lesion:
                                lesion_path_full = os.path.join(dataset_base, str(lesion_path)) if not os.path.isabs(str(lesion_path)) else str(lesion_path)
                                command = f"fslmaths {lesion_path_full} -binv -mul {BIANCA_LPM} {BIANCA_LPM_corrected}"
                                subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
                                BIANCA_LPM = BIANCA_LPM_corrected
                                _lesion_path_ = os.path.join(sub_bianca_result_dir, f"{test_subject}_lesion.nii.gz")
                                fsl_copy(lesion_path_full, _lesion_path_)

                            thresh_paths = {}
                            thresh_dice  = {}

                            # --- Post-Processing: Threshold & Evaluation ---
                            for thresh in thresholds_list:
                                if thresh == 'locate':
                                    LOCATE_sub_dir = os.path.join(sub_bianca_result_dir, "locate")
                                    os.makedirs(LOCATE_sub_dir, exist_ok=True)
                                    results_dir = os.path.join(LOCATE_sub_dir, "LOCATE_results_directory")

                                    existing_results = (
                                        [f for f in os.listdir(results_dir) if "BIANCA_LOCATE_binarylesionmap" in f]
                                        if os.path.exists(results_dir) else [])

                                    if not existing_results:
                                        print(f"  {test_subject} ({combo_name}): Starting LOCATE")
                                        feature_FLAIR_out = os.path.join(LOCATE_sub_dir, f"{test_subject}_feature_FLAIR.nii.gz")
                                        BIANCA_LPM_out    = os.path.join(LOCATE_sub_dir, f"{test_subject}_BIANCA_LPM.nii.gz")
                                        feature_t1w_out   = os.path.join(LOCATE_sub_dir, f"{test_subject}_feature_t1w.nii.gz")
                                        biancamask_out    = os.path.join(LOCATE_sub_dir, f"{test_subject}_biancamask.nii.gz")
                                        brainmask_out     = os.path.join(LOCATE_sub_dir, f"{test_subject}_brainmask.nii.gz")
                                        ventdistmap_out   = os.path.join(LOCATE_sub_dir, f"{test_subject}_ventdistmap.nii.gz")

                                        fsl_copy(test_FLAIR_path, feature_FLAIR_out)
                                        fsl_copy(BIANCA_LPM, BIANCA_LPM_out)
                                        fsl_copy(test_T1, feature_t1w_out)
                                        fsl_copy(test_WMmask_path, biancamask_out)
                                        fsl_copy(test_brainmask_path, brainmask_out)
                                        fsl_copy(test_ventdistmap_path, ventdistmap_out)

                                        run_locate_testing(
                                            train_image_directory_path=train_image_directory_path,
                                            test_image_directory_name=os.path.abspath(LOCATE_sub_dir),
                                            locate_path=LOCATE_PATH, verbose=1,
                                            feature_select=feature_select,
                                        )
                                    else:
                                        print(f"  {test_subject} ({combo_name}): LOCATE results already exist.")

                                    existing_results = (
                                        [f for f in os.listdir(results_dir) if "BIANCA_LOCATE_binarylesionmap" in f]
                                        if os.path.exists(results_dir) else [])
                                    if not existing_results:
                                        raise FileNotFoundError(f"No LOCATE results for {test_subject} ({combo_name})")

                                    locate_result_path = os.path.join(results_dir, existing_results[0])
                                    thresh_paths['locate'] = locate_result_path

                                    if SAVE_METRICS:
                                        locate_eval_metrics = fsl_dice_score(locate_result_path, test_wmh_roi_file)
                                        locate_dice_val    = locate_eval_metrics if not isinstance(locate_eval_metrics, dict) else locate_eval_metrics.get('dice_score', None)
                                        locate_sensitivity = locate_eval_metrics if not isinstance(locate_eval_metrics, dict) else locate_eval_metrics.get('sensitivity', None)
                                        locate_precision   = locate_eval_metrics if not isinstance(locate_eval_metrics, dict) else locate_eval_metrics.get('precision', None)
                                        locate_mcs = get_mcs(test_subject, 'locate')
                                        locate_cluster_metrics = cluster_based_metrics(
                                            locate_result_path, test_wmh_roi_file,
                                            min_cluster_size=locate_mcs)
                                        thresh_dice['locate'] = locate_dice_val

                                        row_dict = {
                                            'subject': test_subject, 'seed': seed, 'fold': fold_idx,
                                            'threshold': 'locate', 'scanner': scanner_group,
                                            'severity_level': severity_level,
                                            'train_condition': train_cond_name, 'test_condition': test_cond_name,
                                            'dice_score': locate_dice_val,
                                            'sensitivity': locate_sensitivity, 'precision': locate_precision,
                                            'lesion_f1': locate_cluster_metrics['lesion_f1'],
                                            'lesion_precision': locate_cluster_metrics['lesion_precision'],
                                            'lesion_recall': locate_cluster_metrics['lesion_recall'],
                                            'n_pred_clusters': locate_cluster_metrics['n_pred_clusters'],
                                            'n_gt_clusters': locate_cluster_metrics['n_gt_clusters'],
                                            'n_tp_pred': locate_cluster_metrics['n_tp_pred'],
                                            'n_tp_gt': locate_cluster_metrics['n_tp_gt'],
                                            'n_fp_clusters': locate_cluster_metrics['n_fp_clusters'],
                                            'n_fn_clusters': locate_cluster_metrics['n_fn_clusters'],
                                            'min_cluster_size': locate_mcs,
                                        }
                                        all_metrics_results.append(row_dict)

                                else:
                                    thresh_str = str(int(thresh * 100))
                                    thresh_dir = os.path.join(sub_bianca_result_dir, thresh_str)
                                    os.makedirs(thresh_dir, exist_ok=True)

                                    bianca_thresh_path = os.path.join(thresh_dir, f"{test_subject}_BIANCA_LPM_thresh_{thresh_str}.nii.gz")
                                    thresh_corrected_path = os.path.join(thresh_dir, f"{test_subject}_BIANCA_LPM_thresh_{thresh_str}_wm_corrected.nii.gz")

                                    if not os.path.isfile(bianca_thresh_path):
                                        for attempt in range(MAX_RETRIES):
                                            try:
                                                threshold_lpm(BIANCA_LPM, bianca_thresh_path, thresh)
                                                if os.path.isfile(bianca_thresh_path):
                                                    break
                                            except Exception as e:
                                                print(f"  Attempt {attempt+1} failed for thresholding {test_subject}: {e}")
                                                time.sleep(RETRY_DELAY)
                                        else:
                                            raise RuntimeError(f"Failed to create {bianca_thresh_path} after {MAX_RETRIES} attempts.")

                                    if not os.path.isfile(thresh_corrected_path):
                                        for attempt in range(MAX_RETRIES):
                                            try:
                                                apply_bianca_mask(bianca_thresh_path, test_WMmask_path, thresh_corrected_path)
                                                if os.path.isfile(thresh_corrected_path):
                                                    break
                                            except Exception as e:
                                                print(f"  Attempt {attempt+1} failed for masking {test_subject}: {e}")
                                                time.sleep(RETRY_DELAY)
                                        else:
                                            raise FileNotFoundError(f"Could not create {thresh_corrected_path} after {MAX_RETRIES} attempts.")

                                    thresh_paths[thresh_str] = thresh_corrected_path
                                    metrics_csv_path = os.path.join(thresh_dir, f"{test_subject}_metrics_{thresh_str}.csv")

                                    if SAVE_METRICS:
                                        th_eval_metrics = fsl_dice_score(thresh_corrected_path, test_wmh_roi_file)
                                        th_dice_val    = th_eval_metrics if not isinstance(th_eval_metrics, dict) else th_eval_metrics.get('dice_score', None)
                                        th_sensitivity = th_eval_metrics if not isinstance(th_eval_metrics, dict) else th_eval_metrics.get('sensitivity', None)
                                        th_precision   = th_eval_metrics if not isinstance(th_eval_metrics, dict) else th_eval_metrics.get('precision', None)
                                        th_mcs = get_mcs(test_subject, thresh_str)
                                        th_cluster_metrics = cluster_based_metrics(
                                            thresh_corrected_path, test_wmh_roi_file,
                                            min_cluster_size=th_mcs)
                                        thresh_dice[thresh_str] = th_dice_val

                                        row_dict = {
                                            'subject': test_subject, 'seed': seed, 'fold': fold_idx,
                                            'threshold': thresh_str, 'scanner': scanner_group,
                                            'severity_level': severity_level,
                                            'train_condition': train_cond_name, 'test_condition': test_cond_name,
                                            'dice_score': th_dice_val,
                                            'sensitivity': th_sensitivity, 'precision': th_precision,
                                            'lesion_f1': th_cluster_metrics['lesion_f1'],
                                            'lesion_precision': th_cluster_metrics['lesion_precision'],
                                            'lesion_recall': th_cluster_metrics['lesion_recall'],
                                            'n_pred_clusters': th_cluster_metrics['n_pred_clusters'],
                                            'n_gt_clusters': th_cluster_metrics['n_gt_clusters'],
                                            'n_tp_pred': th_cluster_metrics['n_tp_pred'],
                                            'n_tp_gt': th_cluster_metrics['n_tp_gt'],
                                            'n_fp_clusters': th_cluster_metrics['n_fp_clusters'],
                                            'n_fn_clusters': th_cluster_metrics['n_fn_clusters'],
                                            'min_cluster_size': th_mcs,
                                        }
                                        pd.DataFrame([row_dict]).to_csv(metrics_csv_path, index=False)
                                        all_metrics_results.append(row_dict)

                            # --- PLOT for newly processed subjects ---
                            if PLOT and has_lesion:
                                masks_png_dir = os.path.join(png_images, f"seed_{seed}_fold_{fold_idx}")
                                os.makedirs(masks_png_dir, exist_ok=True)
                                plot_output_path = os.path.join(masks_png_dir, f"{test_subject}_{combo_name}_FLAIR_masks.png")
                                if not os.path.isfile(plot_output_path):
                                    lesion_path_full = (os.path.join(dataset_base, str(lesion_path))
                                        if not os.path.isabs(str(lesion_path)) else str(lesion_path))
                                    make_panel_plot(test_FLAIR_path, test_wmh_roi_file, lesion_path_full,
                                        thresh_paths, thresh_dice, plot_output_path, test_subject, combo_name)

                        except Exception as e:
                            print(f"  ERROR {test_subject} ({combo_name}): {e}")
                            # Kein DONE.txt → wird beim naechsten Lauf erneut versucht

                        else:
                            # Nur bei Erfolg DONE.txt schreiben
                            with open(done_file, 'w') as f:
                                f.write(f"Finished: {pd.Timestamp.now()}\nSubject: {test_subject}\nCombo: {combo_name}")

                        finally:
                            if os.path.exists(progress_file):
                                os.remove(progress_file)

        # --- Store for JSON ---
        seed_splits[f"fold_{fold_idx+1}"] = {
            "train_lesion_types": lesion_counts,
            "train_has_lesion": int(has_lesion_n) if has_lesion_n != "N/A" else 0,
            "train_subjects": sorted(train_all),
            "test_subjects": sorted(test_all),
            **{f"train_{s}": sorted(train_per_scanner[s]) for s in SCANNER_NAMES},
            **{f"test_{s}": sorted(test_per_scanner[s]) for s in SCANNER_NAMES},
            "train_severity": sev,
            "n_train": len(train_all), "n_test": len(test_all),
        }

        summary_rows.append({
            'train_has_lesion': has_lesion_n,
            'train_lesion_types': str(lesion_counts),
            'seed': f'seed_{seed}', 'fold': f'fold_{fold_idx+1}',
            **{f'train_{s}': len(train_per_scanner[s]) for s in SCANNER_NAMES},
            'train_total': n_train_total,
            **{f'test_{s}': len(test_per_scanner[s]) for s in SCANNER_NAMES},
            'test_total': len(test_all),
            'train_high': sev.get('high', 0),
            'train_mid': sev.get('middle', 0),
            'train_low': sev.get('low', 0),
        })

    all_splits[f"seed_{seed}"] = seed_splits

    if all_metrics_results:
        current_seed_metrics = [m for m in all_metrics_results if m['seed'] == seed]
        if current_seed_metrics and not DEBUG:
            seed_df = pd.DataFrame(current_seed_metrics)
            seed_excel_path = os.path.join(all_metrics_result_dir, f"bianca_metrics_seed_{seed}.xlsx")
            seed_df.to_excel(seed_excel_path, index=False)
            print(f"  Saved metrics for seed {seed} to {seed_excel_path}")


# =============================================================
# COMBINE ALL SEED RESULTS
# =============================================================
all_seed_files = glob.glob(os.path.join(all_metrics_result_dir, "bianca_metrics_seed_*.xlsx"))
metrics_output_path = os.path.join(CV_SET_base_dir, "bianca_evaluation_metrics.xlsx")

#here i want to check if all subject and methods etc are run

if not DEBUG and all_seed_files:
    combined_df = pd.concat([pd.read_excel(f) for f in all_seed_files], ignore_index=True)

    # --- Completeness Check ---
    expected_seeds = set(seeds)
    expected_folds = set(range(N_FOLDS))
    expected_thresholds = set(['85', '90', 'locate'])
    expected_train_conds = set([c['name'] for c in train_conditions])
    expected_test_conds = set([c['name'] for c in test_conditions])

    found_seeds = set(combined_df['seed'].unique())
    missing_seeds = expected_seeds - found_seeds
    if missing_seeds:
        print(f"WARNING: Missing seeds: {sorted(missing_seeds)}")

    for s in sorted(found_seeds):
        seed_df = combined_df[combined_df['seed'] == s]
        found_folds = set(seed_df['fold'].unique())
        
        
        missing_folds = expected_folds - found_folds
        if missing_folds:
            print(f"  WARNING: Seed {s} missing folds: {sorted(missing_folds)}")

        for f in sorted(found_folds):
            fold_df = seed_df[seed_df['fold'] == f]
            combos = fold_df.groupby(['train_condition', 'test_condition', 'threshold'])['subject'].nunique()
            n_test_expected = len(fold_df['subject'].unique())

            for tc in expected_train_conds:
                for te in expected_test_conds:
                    for th in expected_thresholds:
                        try:
                            n = combos.loc[(tc, te, th)]
                        except KeyError:
                            n = 0
                        if n < n_test_expected:
                            print(f"  WARNING: Seed {s} Fold {f} train_{tc}__test_{te} thresh={th}: "
                                  f"{n}/{n_test_expected} subjects")

    total_expected = len(seeds) * N_FOLDS * len(expected_train_conds) * len(expected_test_conds) * len(expected_thresholds)
    # expected per combo = avg test subjects per fold
    print(f"\nCompleteness: {len(combined_df)} rows total")
    print(f"Seeds found: {sorted(found_seeds)} / expected: {sorted(expected_seeds)}")
    print(f"Seed files: {len(all_seed_files)} / {len(seeds)}")