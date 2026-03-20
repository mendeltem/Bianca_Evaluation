#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOCATE Training Pipeline for Locally Adaptive Thresholding

To prevent data leakage through the thresholding step, a dedicated
LOCATE training set (n=21) was held out prior to cross-validation.
These subjects were selected from the combined BeLOVE and WMH
Segmentation Challenge datasets, excluding subjects with stroke
lesions and GE Signa scans. Selection was stratified by scanner type
(n=7 each for Philips, Siemens Tim Trio, and Siemens Prisma fit) to
ensure balanced scanner representation. LOCATE training subjects
included low and moderate WMH burden cases, which is sufficient given
the algorithm's reliance on spatial rather than intensity-dependent
features (Sundaresan et al., 2019).

For each of the 21 subjects, a leave-one-out BIANCA model was trained
on the remaining 20 subjects to generate an unbiased lesion probability
map (LPM). These LPMs, together with FLAIR, T1, white matter masks,
brain masks, and ventricle distance maps, were used to train the LOCATE
random forest regression model for locally adaptive threshold estimation.

The trained LOCATE model was subsequently applied as the thresholding
method within the main 5-fold cross-validation pipeline. All 21 LOCATE
training subjects were excluded from the cross-validation folds to
maintain strict separation between threshold optimization and
segmentation evaluation.

An additional LOCATE model was trained on a combined set of 21 + 5 GE
Signa subjects (n=26) for scanner generalization analyses. GE LPMs
were generated using a BIANCA model trained on all 21 LOO subjects,
ensuring no leakage between training and inference.

Pipeline steps:
  1. Merge BeLOVE and Challenge datasets (N=130)
  2. Select 21 LOCATE subjects, stratified by scanner type
  3. LOO BIANCA training and LPM generation for each subject
  4. LOCATE model training on 21 subjects (without GE)
  5. LOCATE model training on 26 subjects (with GE)

Reference:
    Sundaresan et al. (2019). Automated lesion segmentation with
    BIANCA: Impact of population-level features, classification
    algorithm and locally adaptive thresholding. NeuroImage 202, 116056.
    
    
    
Paper:

LOCATE training required unbiased lesion probability maps generated via leave-one-out cross-validation. 
A dedicated set of 21 subjects, stratified by scanner type (n=7 per scanner), 
was held out prior to the main cross-validation to prevent data leakage through 
the thresholding step (Sundaresan et al., 2019).    



"""

import os
import json
import glob
import tempfile
import subprocess
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from dotenv import load_dotenv

from DATASETS.librarys.directory_lib import (
    fsl_copy,
    get_volume,
    threshold_lpm,
    apply_bianca_mask,
    fsl_dice_score,
    create_panel_plot_with_multi_overlay,
    get_top_slices,
)

load_dotenv()

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
LOCATE_PATH = os.getenv("LOCATE_PATH")

# ─────────────────────────────────────────────────────────────────────────────
# LOCATE helper functions
# ────────────────────────────────────────────────────────────────────────────


def run_locate_training(
    train_image_directory_path: str,
    locate_path: str,
    verbose: int = 1,
    feature_select: List[int] = [1, 1, 1, 1],
) -> None:
    """
    Run LOCATE training via MATLAB with proper error handling.

    Args:
        train_image_directory_path: Path to directory containing training images.
        locate_path: Path to the LOCATE toolbox.
        verbose: Verbosity level (0 or 1).
        feature_select: Feature selection flags [1,1,1,1].

    Raises:
        RuntimeError: If LOCATE training fails.
        FileNotFoundError: If required paths don't exist.
    """
    if not os.path.exists(train_image_directory_path):
        raise FileNotFoundError(
            f"Training directory not found: {train_image_directory_path}"
        )
    if not os.path.exists(locate_path):
        raise FileNotFoundError(f"LOCATE path not found: {locate_path}")

    matlab_script_content = f"""
    try
        addpath(genpath('{locate_path}'));
        train_image_directory_name = '{train_image_directory_path}';
        feature_select = {feature_select};
        verbose = {verbose};
        fprintf('Starting LOCATE training...\\n');
        fprintf('Training directory: %s\\n', train_image_directory_name);
        LOCATE_training(train_image_directory_name, feature_select, verbose);
    catch ME
        fprintf('Error: %s\\n', ME.message);
        fprintf('File: %s\\n', ME.stack(1).file);
        fprintf('Line: %d\\n', ME.stack(1).line);
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\\n', ME.stack(i).name, ME.stack(i).line);
        end
        exit(1);
    end
    fprintf('LOCATE training completed successfully.\\n');
    exit(0);
    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".m") as tmp:
        script_path = tmp.name
        tmp.write(matlab_script_content)

    try:
        print(f"Running LOCATE training on {train_image_directory_path}")
        return_code = os.system(f"matlab -batch \"run('{script_path}');\"")
        if return_code != 0:
            raise RuntimeError(
                f"LOCATE training failed with return code {return_code}"
            )
        print("LOCATE training completed successfully")
    except Exception as e:
        raise RuntimeError(f"Error during LOCATE training: {str(e)}")
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def prepare_locate_subject_dir(
    locate_dir,
    subject,
    flair_path,
    t1_path,
    wmh_path,
    WMmask_path,
    brainmask_path,
    ventdistmap_path,
    bianca_lpm_path,
):
    """
    Prepare a LOCATE-compatible subject directory with all required files.

    Copies each source file into locate_dir using the LOCATE naming convention.
    Skips files that already exist or whose source path is missing.
    """
    os.makedirs(locate_dir, exist_ok=True)

    file_map = {
        f"{subject}_feature_FLAIR.nii.gz": flair_path,
        f"{subject}_feature_t1w.nii.gz": t1_path,
        f"{subject}_manualmask.nii.gz": wmh_path,
        f"{subject}_biancamask.nii.gz": WMmask_path,
        f"{subject}_brainmask.nii.gz": brainmask_path,
        f"{subject}_ventdistmap.nii.gz": ventdistmap_path,
        f"{subject}_BIANCA_LPM.nii.gz": bianca_lpm_path,
    }

    for dst_name, src_path in file_map.items():
        dst_path = os.path.join(locate_dir, dst_name)
        if (
            not os.path.isfile(dst_path)
            and src_path is not None
            and os.path.isfile(src_path)
        ):
            fsl_copy(src_path, dst_path)


def resolve_path(dataset_base, relative_path):
    """
    Build absolute path from dataset base and relative path.
    Returns the path if the file exists, otherwise an empty string.
    """
    if pd.isna(relative_path) or relative_path == "":
        return ""
    full = os.path.join(dataset_base, relative_path)
    return full if os.path.isfile(full) else ""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and merge datasets
# ─────────────────────────────────────────────────────────────────────────────

BELOVE_BASE = "DATASETS/BELOVE_BIDS_WMH_BIDS"
CHALLENGE_BASE = "DATASETS/CHALLENGE_BIDS_WMH_BIDS"

base_dir = "Phase_1/LOCATE_SET"
os.makedirs(base_dir, exist_ok=True)

# BeLOVE
belove_file_df = pd.read_excel(
    os.path.join(BELOVE_BASE, "derivatives/preprocessed_files.xlsx")
)
belove_file_df["dataset_base"] = BELOVE_BASE

belove_df = pd.read_excel(
    os.path.join(BELOVE_BASE, "derivatives/BELOVE_BIDS_WMH_file_locations.xlsx")
)
belove_merged = pd.merge(belove_file_df, belove_df, on="subject", how="outer")

# Challenge
challenge_file_df = pd.read_excel(
    os.path.join(CHALLENGE_BASE, "derivatives/preprocessed_files.xlsx")
)
challenge_file_df["dataset_base"] = CHALLENGE_BASE

challenge_df = pd.read_excel(
    os.path.join(CHALLENGE_BASE, "derivatives/CHALLENGE_BIDS_WMH_file_locations.xlsx")
)
challenge_merged = pd.merge(challenge_file_df, challenge_df, on="subject", how="outer")

# Combine both datasets
all_file_df = pd.concat([belove_merged, challenge_merged], ignore_index=True)
assert all_file_df["subject"].is_unique, "Duplicate subjects found!"

results_path = os.path.join(base_dir, "all_files.xlsx")
all_file_df.to_excel(results_path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Select 20 LOCATE subjects: stratified by severity and scanner
#    Target distribution: 6 Philips, 7 Tim Trio, 7 Prisma_fit
# ─────────────────────────────────────────────────────────────────────────────
scanner_targets = {
    "Philips": 7,
    "Tim Trio": 7,
    "Prisma_fit": 7,
}

# Filter: exclude subjects with stroke lesions and GE scanner
locate_pool = all_file_df[all_file_df["has_lesion"] != True].copy()
locate_pool = locate_pool[locate_pool["scanner"] != "GE Signa"].copy()

all_available_pool_path = os.path.join(base_dir, "all_available_pool.xlsx")
locate_pool.to_excel(all_available_pool_path)

# Filter: nur low und middle für LOCATE
locate_pool = locate_pool[locate_pool["severity_level"].isin(["low", "middle"])].copy()

# Assign clean scanner labels AFTER all filtering
locate_pool["scanner_clean"] = locate_pool["scanner"].apply(
    lambda x: x if x in ["Prisma_fit", "Tim Trio"] else "Philips"
)

selected_parts = []
for scanner_name, n_target in scanner_targets.items():
    pool = locate_pool[locate_pool["scanner_clean"] == scanner_name]

    if len(pool) < n_target:
        print(
            f"⚠️  {scanner_name}: only {len(pool)} available, using all"
        )
        selected_parts.append(pool)
    elif pool["severity_level"].nunique() < 2:
        # Cannot stratify with only one severity level
        sel, _ = train_test_split(
            pool, train_size=n_target, random_state=RANDOM_STATE
        )
        selected_parts.append(sel)
    else:
        sel, _ = train_test_split(
            pool,
            train_size=n_target,
            stratify=pool["severity_level"],
            random_state=RANDOM_STATE,
        )
        selected_parts.append(sel)

locate_df = pd.concat(selected_parts)

print(f"\nSelected {len(locate_df)} subjects for LOCATE LOO:")
print(locate_df["scanner_clean"].value_counts())
print(locate_df["severity_level"].value_counts())
print(locate_df["scanner"].value_counts())

# Label subsets in the master dataframe
all_file_df["subset"] = "bianca_pool"
all_file_df.loc[
    all_file_df["subject"].isin(locate_df["subject"]), "subset"
] = "locate_loo"
print(all_file_df["subset"].value_counts())

# Save subset spreadsheets
bianca_pool_path = os.path.join(base_dir, "bianca_pool.xlsx")
all_file_df[all_file_df["subset"] == "bianca_pool"].to_excel(
    bianca_pool_path, index=False
)

bianca_pool_path = os.path.join(base_dir, "bianca_pool.xlsx")
bianca_pool_df =  all_file_df[all_file_df["subset"] == "bianca_pool"]

bianca_pool_df.to_excel(
    bianca_pool_path, index=False
)

bianca_pool_wihtouth_ge_path = os.path.join(base_dir, "bianca_pool_wihtouth_ge.xlsx")
bianca_pool_df = bianca_pool_df[bianca_pool_df["scanner"] != "GE Signa"].copy()

bianca_pool_df.to_excel(
    bianca_pool_wihtouth_ge_path, index=False
)

must_meta_columns = ["subject",
                "scanner",
                "severity_level"]

meta_df = bianca_pool_df[must_meta_columns]


bianca_pool_wihtouth_ge_path = os.path.join(base_dir, "bianca_pool_wihtouth_ge_meta.xlsx")


meta_df.to_excel(
    bianca_pool_wihtouth_ge_path, index=False
)


locate_pool_path = os.path.join(base_dir, "locate_pool.xlsx")
locate_loo_df = all_file_df[all_file_df["subset"] == "locate_loo"]
locate_loo_df.to_excel(locate_pool_path, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Leave-One-Out BIANCA training + inference for each LOCATE subject
# ─────────────────────────────────────────────────────────────────────────────

subject_list = list(locate_loo_df["subject"])
locate_train_dir = os.path.join(base_dir, "locate_train")
os.makedirs(locate_train_dir, exist_ok=True)

#i want to remove all ürevisous subject from the direcotry i want to create new trainings

for si, subject in enumerate(tqdm(subject_list, desc="LOO subjects")):

    print(f"\n--- [{si+1}/{len(subject_list)}] Subject: {subject} ---")
    train_subjects = [s for s in subject_list if s != subject]

    # Resolve test subject paths
    row = locate_loo_df[locate_loo_df["subject"] == subject].iloc[0]
    ds_base = row["dataset_base"]

    FLAIR_brain_biascorr = resolve_path(ds_base, row["FLAIR_brain_biascorr"])
    T1 = resolve_path(ds_base, row["T1"])
    mni_mat_path = resolve_path(ds_base, row["mni_mat_path"])
    wmh_roi_file = resolve_path(ds_base, row["wmh_roi_file"])
    ventdistmap = resolve_path(ds_base, row["ventdistmap"])
    brainmask = resolve_path(ds_base, row["brainmask"])
    WMmask = resolve_path(ds_base, row["WMmask"])

    subject_dir = os.path.join(base_dir, subject)
    os.makedirs(subject_dir, exist_ok=True)

    # ── Build training masterfile (all subjects except current) ──────────
    master_file_lines = []
    for train_subject in train_subjects:
        tr = locate_loo_df[locate_loo_df["subject"] == train_subject].iloc[0]
        tr_base = tr["dataset_base"]

        tr_flair = resolve_path(tr_base, tr["FLAIR_brain_biascorr"])
        tr_t1 = resolve_path(tr_base, tr["T1"])
        tr_mat = resolve_path(tr_base, tr["mni_mat_path"])
        tr_wmh = resolve_path(tr_base, tr["wmh_roi_file"])

        master_file_lines.append(f"{tr_flair} {tr_t1} {tr_mat} {tr_wmh}")

    train_master_file = os.path.join(
        subject_dir, f"{subject}_train_master_file.txt"
    )
    with open(train_master_file, "w") as f:
        f.write("\n".join(master_file_lines))

    # ── BIANCA model training ───────────────────────────────────────────
    BIANCA_MODEL = os.path.join(subject_dir, "BIANCA_MODEL_LOO_LOCATE")

    if not os.path.isfile(BIANCA_MODEL):
        row_number = len(master_file_lines)
        trainingpts = 2000
        trainstring = ",".join(str(r) for r in range(1, row_number + 1))

        train_cmd = [
            "bianca",
            f"--singlefile={train_master_file}",
            "--brainmaskfeaturenum=1",
            "--matfeaturenum=3",
            "--featuresubset=1,2",
            "--labelfeaturenum=4",
            f"--trainingpts={trainingpts}",
            "--nonlespts=10000",
            f"--trainingnums={trainstring}",
            f"--saveclassifierdata={BIANCA_MODEL}",
            f"--querysubjectnum={row_number}",
            "-v",
        ]

        try:
            print(f"  Training BIANCA model (n_train={row_number})")
            subprocess.run(train_cmd, check=True)
            print(f"  ✅ Training done for {subject}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Training error for {subject}: {e}")
            break

    # ── Test masterfile ──────────────────────────────────────────────────
    test_masterfile_path = os.path.join(
        subject_dir, f"{subject}_masterfile_test.txt"
    )
    if not os.path.isfile(test_masterfile_path):
        with open(test_masterfile_path, "w") as f:
            f.write(
                f"{FLAIR_brain_biascorr} {T1} {mni_mat_path} {wmh_roi_file}"
            )

    # ── BIANCA inference (generate LPM) ─────────────────────────────────
    BIANCA_LPM = os.path.join(subject_dir, f"{subject}_BIANCA_LPM.nii.gz")

    if not os.path.isfile(BIANCA_LPM):
        test_cmd = [
            "bianca",
            f"--singlefile={test_masterfile_path}",
            "--brainmaskfeaturenum=1",
            "--matfeaturenum=3",
            "--featuresubset=1,2",
            f"--loadclassifierdata={BIANCA_MODEL}",
            "--querysubjectnum=1",
            "-o",
            BIANCA_LPM,
            "-v",
        ]
        try:
            print(f"  Generating BIANCA LPM for {subject}")
            subprocess.run(test_cmd, check=True, capture_output=True, text=True)
            print(f"  ✅ Inference done for {subject}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Inference error for {subject}: {e}")
            break

    # ── Prepare LOCATE directories ──────────────────────────────────────
    local_locate = os.path.join(subject_dir, "locate")

    prepare_locate_subject_dir(
        locate_dir=local_locate,
        subject=subject,
        flair_path=FLAIR_brain_biascorr,
        t1_path=T1,
        wmh_path=wmh_roi_file,
        WMmask_path=WMmask,
        brainmask_path=brainmask,
        ventdistmap_path=ventdistmap,
        bianca_lpm_path=BIANCA_LPM,
    )

    prepare_locate_subject_dir(
        locate_dir=locate_train_dir,
        subject=subject,
        flair_path=FLAIR_brain_biascorr,
        t1_path=T1,
        wmh_path=wmh_roi_file,
        WMmask_path=WMmask,
        brainmask_path=brainmask,
        ventdistmap_path=ventdistmap,
        bianca_lpm_path=BIANCA_LPM,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run LOCATE training on the full training directory
# ─────────────────────────────────────────────────────────────────────────────

locate_train_abs = os.path.abspath(locate_train_dir)

print(f"\nLOCATE training directory: {locate_train_abs}")
print(f"LOCATE toolbox path: {LOCATE_PATH}")
assert os.path.isdir(LOCATE_PATH), f"LOCATE_PATH not found: {LOCATE_PATH}"


trainted_model = os.path.join(locate_train_abs,"LOCATE_training_files","RF_regression_model_LOCATE.mat")

if not os.path.isfile(trainted_model):

    run_locate_training(
        train_image_directory_path=locate_train_abs,
        locate_path=LOCATE_PATH,
        verbose=1,
        feature_select=[1, 1, 1, 1],
    )

print("\n✅ Pipeline completed.")



# ─────────────────────────────────────────────────────────────────────────────
# 5. LOCATE Training WITH GE subjects
#    - Load 5 leftover GE subjects
#    - Train BIANCA on all 21 LOO subjects (no leakage: GE not in LOO set)
#    - Generate LPMs for 5 GE subjects
#    - Copy all 21 + 5 = 26 into locate_train_with_ge
#    - Run LOCATE training on combined set
# ─────────────────────────────────────────────────────────────────────────────

ge_leftover_path = os.path.join(base_dir, "ge_data_that_not_in_trainset.xlsx")
ge_leftover_df = pd.read_excel(ge_leftover_path)

ge_leftover_df.columns

"""
Index(['subject', 'WMH_path', 'lesion_path', 'WMmask', 'WMmask_exists',
       'ventdistmap', 'ventdistmap_exists', 'T1', 'T1_exists', 'brainmask',
       'brainmask_exists', 'FLAIR_brain', 'FLAIR_brain_biascorr',
       'biascorr_method', 'FLAIR_mni_mat_path', 'mni_mat_path',
       'WMH_removed_path', 'FLAIR_non_removed_path', 'FLAIR_removed_path',
       'FLAIR_filled_path', 'brainmask_path', 'dataset_base', 'original_name',
       'scanner', 'sex', 'age', 'lesion_type', 'Wahlund', 'ROI_Volume',
       'Lesion_Volume', 'flair_file', 't1_file', 'wmh_roi_file', 'lesion_file',
       'has_roi', 'has_lesion', 'severity_level', 'anat_file_count',
       'manual_mask'],
      dtype='str')
"""

print(f"\nLoaded {len(ge_leftover_df)} GE subjects from {ge_leftover_path}")

# ── 5a. Train BIANCA on ALL 21 LOO subjects (used as training for GE inference)
ge_model_dir = os.path.join(base_dir, "ge_bianca_model")
os.makedirs(ge_model_dir, exist_ok=True)

master_file_lines_all = []
for train_subject in subject_list:
    tr = locate_loo_df[locate_loo_df["subject"] == train_subject].iloc[0]
    tr_base = tr["dataset_base"]

    tr_flair = resolve_path(tr_base, tr["FLAIR_brain_biascorr"])
    tr_t1 = resolve_path(tr_base, tr["T1"])
    tr_mat = resolve_path(tr_base, tr["mni_mat_path"])
    tr_wmh = resolve_path(tr_base, tr["wmh_roi_file"])

    if not all([tr_flair, tr_t1, tr_mat, tr_wmh]):
        raise FileNotFoundError(f"Missing files for LOO subject {train_subject}")

    master_file_lines_all.append(f"{tr_flair} {tr_t1} {tr_mat} {tr_wmh}")

all_21_masterfile = os.path.join(ge_model_dir, "all_21_loo_masterfile.txt")
with open(all_21_masterfile, "w") as f:
    f.write("\n".join(master_file_lines_all))

BIANCA_MODEL_ALL_21 = os.path.join(ge_model_dir, "BIANCA_MODEL_ALL_21_LOO")

if not os.path.isfile(BIANCA_MODEL_ALL_21):
    n = len(master_file_lines_all)
    trainstring = ",".join(str(r) for r in range(1, n + 1))
    train_cmd = [
        "bianca",
        f"--singlefile={all_21_masterfile}",
        "--brainmaskfeaturenum=1",
        "--matfeaturenum=3",
        "--featuresubset=1,2",
        "--labelfeaturenum=4",
        "--trainingpts=2000",
        "--nonlespts=10000",
        f"--trainingnums={trainstring}",
        f"--saveclassifierdata={BIANCA_MODEL_ALL_21}",
        f"--querysubjectnum={n}",
        "-v",
    ]
    try:
        print(f"  Training BIANCA on all 21 LOO subjects...")
        subprocess.run(train_cmd, check=True)
        print(f"  ✅ Model saved: {BIANCA_MODEL_ALL_21}")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Training error: {e}")
        raise

# ── 5b. Generate LPMs for 5 GE subjects + prepare LOCATE dirs

# Handle manual_mask column (same logic as ge_compare script)
import numpy as np

ge_leftover_df['manual_mask'] = np.where(
    ge_leftover_df['WMH_removed_path'].notna(),
    ge_leftover_df['WMH_removed_path'],
    ge_leftover_df['WMH_path']
)

ge_subject_list = list(ge_leftover_df["subject"])

for gi, ge_subject in enumerate(ge_subject_list):
    print(f"\n--- GE [{gi+1}/{len(ge_subject_list)}] Subject: {ge_subject} ---")

    row = ge_leftover_df[ge_leftover_df["subject"] == ge_subject].iloc[0]
    ds_base = row["dataset_base"]

    ge_FLAIR = resolve_path(ds_base, row["FLAIR_brain_biascorr"])
    ge_T1 = resolve_path(ds_base, row["T1"])
    ge_mni = resolve_path(ds_base, row["mni_mat_path"])
    ge_wmh = resolve_path(ds_base, row["manual_mask"])
    ge_ventdist = resolve_path(ds_base, row["ventdistmap"])
    ge_brainmask = resolve_path(ds_base, row["brainmask"])
    ge_WMmask = resolve_path(ds_base, row["WMmask"])

    ge_subject_dir = os.path.join(base_dir, ge_subject)
    os.makedirs(ge_subject_dir, exist_ok=True)

    # Test masterfile for this GE subject
    ge_test_masterfile = os.path.join(ge_subject_dir, f"{ge_subject}_masterfile_test.txt")
    with open(ge_test_masterfile, "w") as f:
        f.write(f"{ge_FLAIR} {ge_T1} {ge_mni} {ge_wmh}")

    # BIANCA inference -> LPM
    ge_LPM = os.path.join(ge_subject_dir, f"{ge_subject}_BIANCA_LPM.nii.gz")

    if not os.path.isfile(ge_LPM):
        test_cmd = [
            "bianca",
            f"--singlefile={ge_test_masterfile}",
            "--brainmaskfeaturenum=1",
            "--matfeaturenum=3",
            "--featuresubset=1,2",
            f"--loadclassifierdata={BIANCA_MODEL_ALL_21}",
            "--querysubjectnum=1",
            "-o", ge_LPM,
            "-v",
        ]
        try:
            print(f"  Generating BIANCA LPM for {ge_subject}")
            subprocess.run(test_cmd, check=True, capture_output=True, text=True)
            print(f"  ✅ LPM created: {ge_LPM}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Inference error for {ge_subject}: {e}")
            continue

    # Prepare LOCATE dir for this GE subject (local + shared)
    local_locate = os.path.join(ge_subject_dir, "locate")

    prepare_locate_subject_dir(
        locate_dir=local_locate,
        subject=ge_subject,
        flair_path=ge_FLAIR,
        t1_path=ge_T1,
        wmh_path=ge_wmh,
        WMmask_path=ge_WMmask,
        brainmask_path=ge_brainmask,
        ventdistmap_path=ge_ventdist,
        bianca_lpm_path=ge_LPM,
    )

# ── 5c. Build locate_train_with_ge: copy 21 LOO + 5 GE subjects

locate_train_with_ge_dir = os.path.join(base_dir, "locate_train_with_ge")
os.makedirs(locate_train_with_ge_dir, exist_ok=True)

# Copy 21 LOO subjects from existing locate_train
for subject in subject_list:
    prepare_locate_subject_dir(
        locate_dir=locate_train_with_ge_dir,
        subject=subject,
        flair_path=os.path.join(locate_train_dir, f"{subject}_feature_FLAIR.nii.gz"),
        t1_path=os.path.join(locate_train_dir, f"{subject}_feature_t1w.nii.gz"),
        wmh_path=os.path.join(locate_train_dir, f"{subject}_manualmask.nii.gz"),
        WMmask_path=os.path.join(locate_train_dir, f"{subject}_biancamask.nii.gz"),
        brainmask_path=os.path.join(locate_train_dir, f"{subject}_brainmask.nii.gz"),
        ventdistmap_path=os.path.join(locate_train_dir, f"{subject}_ventdistmap.nii.gz"),
        bianca_lpm_path=os.path.join(locate_train_dir, f"{subject}_BIANCA_LPM.nii.gz"),
    )

# Copy 5 GE subjects from their local locate dirs
for ge_subject in ge_subject_list:
    ge_local = os.path.join(base_dir, ge_subject, "locate")
    prepare_locate_subject_dir(
        locate_dir=locate_train_with_ge_dir,
        subject=ge_subject,
        flair_path=os.path.join(ge_local, f"{ge_subject}_feature_FLAIR.nii.gz"),
        t1_path=os.path.join(ge_local, f"{ge_subject}_feature_t1w.nii.gz"),
        wmh_path=os.path.join(ge_local, f"{ge_subject}_manualmask.nii.gz"),
        WMmask_path=os.path.join(ge_local, f"{ge_subject}_biancamask.nii.gz"),
        brainmask_path=os.path.join(ge_local, f"{ge_subject}_brainmask.nii.gz"),
        ventdistmap_path=os.path.join(ge_local, f"{ge_subject}_ventdistmap.nii.gz"),
        bianca_lpm_path=os.path.join(ge_local, f"{ge_subject}_BIANCA_LPM.nii.gz"),
    )

# Verify
n_subjects_with_ge = len(glob.glob(os.path.join(locate_train_with_ge_dir, "*_BIANCA_LPM.nii.gz")))
print(f"\nlocate_train_with_ge: {n_subjects_with_ge} subjects (expected {len(subject_list) + len(ge_subject_list)})")

# ── 5d. Run LOCATE training on the combined 21+5 set

locate_train_with_ge_abs = os.path.abspath(locate_train_with_ge_dir)

print(f"\nLOCATE training (with GE) directory: {locate_train_with_ge_abs}")

run_locate_training(
    train_image_directory_path=locate_train_with_ge_abs,
    locate_path=LOCATE_PATH,
    verbose=1,
    feature_select=[1, 1, 1, 1],
)

print("\n✅ LOCATE training WITH GE completed.")


