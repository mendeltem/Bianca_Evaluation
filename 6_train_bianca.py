#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIANCA Model Training for Phase II (Inpainted)
===============================================

Revision context (NeuroImage: Clinical, Major Revision)
-------------------------------------------------------
This script trains the BIANCA model used in Phase II (Script 6).

  (1) Training condition choice
      The 5-fold stratified CV (Script 5, 10 seeds) showed that all
      three training conditions (non_removed, removed, inpainted)
      produce virtually identical segmentation performance when applied
      to all three test conditions (Dice 0.567, Sensitivity 0.687,
      Lesion-level F1 0.873, differences in the 4th decimal place).

      We selected inpainted (FSL lesion_filling) as the training
      condition because:
        (a) It addresses R1 Comment 1 and R5 #9 by avoiding the
            non-physiological zero-intensity values of the removed
            condition.
        (b) It provides the most physiologically representative FLAIR
            intensity distribution for training, as stroke lesion voxels
            are replaced with locally sampled NAWM intensities.
        (c) The equivalent performance across conditions confirms that
            this choice does not sacrifice accuracy.

  (2) Data separation (R1 Comment 3; R5 #7; R5 #13)
      The training pool excludes all Phase II test subjects and LOCATE
      training subjects to maintain strict separation.

  (3) Single model, three test conditions
      Script 6 applies this one model to all three FLAIR variants
      (non_removed, removed, inpainted) at inference time. This design
      isolates the preprocessing effect on test images while keeping
      the trained classifier constant.

Paper changes
-------------
  Methods section 2.6 (revised): "BIANCA model training"
    - Training on inpainted FLAIR images justified by 5-fold CV results.
    - Training pool: N subjects from bianca_pool (without GE, without
      LOCATE subjects, without Phase II test subjects).
    - For subjects without stroke lesions, FLAIR_filled_path falls back
      to the original bias-corrected FLAIR (identical to non_removed).

Terminology (consistent across all scripts):
  - non_removed: no lesion preprocessing (FLAIR_brain_biascorr)
  - removed: zero-intensity lesion replacement (FLAIR_removed_path)
  - inpainted / filled: NAWM-based inpainting (FLAIR_filled_path)

@author: temuuleu
"""

import os
import numpy as np
import pandas as pd
import subprocess

# =============================================================
# CONFIG
# =============================================================
TRAIN_CONDITION = "filled"  # inpainted FLAIR

model_dir = "Phase_1/BIANCA_MODELS"
os.makedirs(model_dir, exist_ok=True)

base_dir = "Phase_1/LOCATE_SET"
os.makedirs(base_dir, exist_ok=True)

# =============================================================
# 1. LOAD BIANCA POOL (without GE, without LOCATE subjects)
# =============================================================
bianca_pool_path = os.path.join(base_dir, "bianca_pool_wihtouth_ge.xlsx")
bianca_pool_df = pd.read_excel(bianca_pool_path)

# =============================================================
# 2. EXCLUDE PHASE II TEST SUBJECTS
# =============================================================
bianca_scanner_pool_meta = os.path.join(model_dir, "bianca_scanner_pool_meta.xlsx")
bianca_train_df = pd.read_excel(bianca_scanner_pool_meta)
phase2_test_subjects = list(
    bianca_train_df[bianca_train_df["TEST"] == "TEST"]["subject"]
)

filtered_df = bianca_pool_df[
    ~bianca_pool_df["subject"].isin(phase2_test_subjects)
].copy()

# manual_mask: prefer WMH_removed_path if available, else wmh_roi_file
filtered_df["manual_mask"] = np.where(
    filtered_df["WMH_removed_path"].notna(),
    filtered_df["WMH_removed_path"],
    filtered_df["wmh_roi_file"],
)

# Save filtered pool for reference
filtered_df.to_excel(os.path.join(model_dir, "filtered_df.xlsx"), index=False)

meta_columns = ["subject", "scanner", "severity_level"]
meta_filtered_df = filtered_df[meta_columns]
meta_filtered_df.to_excel(
    os.path.join(model_dir, "filtered_meta_df.xlsx"), index=False
)

N = len(meta_filtered_df)
print(f"Training pool: N={N} subjects")
print(meta_filtered_df["scanner"].value_counts().to_string())


# =============================================================
# 3. RESOLVE FLAIR PATH
# =============================================================
def resolve_flair(row):
    """
    Use FLAIR_filled_path if available, otherwise FLAIR_brain_biascorr.

    Subjects with stroke lesions will have FLAIR_filled_path (inpainted).
    Subjects without lesions only have FLAIR_brain_biascorr, which is
    equivalent since there are no lesions to fill.
    """
    dbase = row["dataset_base"]

    # Try inpainted first
    if "FLAIR_filled_path" in row.index and pd.notna(row["FLAIR_filled_path"]):
        p = os.path.join(dbase, str(row["FLAIR_filled_path"]))
        if os.path.isfile(p):
            return p

    # Fallback: original bias-corrected FLAIR
    return os.path.join(dbase, str(row["FLAIR_brain_biascorr"]))


# =============================================================
# 4. BUILD BIANCA MASTER FILE
# =============================================================
BIANCA_MODEL = os.path.join(model_dir, f"BIANCA_MODEL_N_{N}_{TRAIN_CONDITION.upper()}")
train_subjects = list(meta_filtered_df["subject"])
master_file_lines = []

for train_subject in train_subjects:
    row = filtered_df[filtered_df["subject"] == train_subject].iloc[0]
    dbase = row["dataset_base"]

    flair = resolve_flair(row)
    t1 = os.path.join(dbase, row["T1"])
    mat = os.path.join(dbase, row["mni_mat_path"])
    wmh = os.path.join(dbase, row["manual_mask"])

    # Validate all files exist
    paths = {"FLAIR": flair, "T1": t1, "mni_mat": mat, "manual_mask": wmh}
    missing = [k for k, v in paths.items() if not v or not os.path.isfile(v)]
    if missing:
        raise FileNotFoundError(
            f"Missing files for {train_subject}: {', '.join(missing)}"
        )

    master_file_lines.append(f"{flair} {t1} {mat} {wmh}")

train_master_file = os.path.join(
    model_dir, f"bianca_n_{N}_{TRAIN_CONDITION}_train_master_file.txt"
)
with open(train_master_file, "w") as f:
    f.write("\n".join(master_file_lines))

print(f"Master file: {train_master_file} ({len(master_file_lines)} subjects)")
print(f"Train condition: {TRAIN_CONDITION}")


# =============================================================
# 5. TRAIN BIANCA k-NN CLASSIFIER
# =============================================================
if not os.path.isfile(BIANCA_MODEL):
    row_number = len(master_file_lines)
    trainstring = ",".join(str(r) for r in range(1, row_number + 1))

    cmd = [
        "bianca",
        f"--singlefile={train_master_file}",
        "--brainmaskfeaturenum=1",
        "--matfeaturenum=3",
        "--featuresubset=1,2",
        "--labelfeaturenum=4",
        "--trainingpts=2000",
        "--nonlespts=10000",
        f"--trainingnums={trainstring}",
        f"--saveclassifierdata={BIANCA_MODEL}",
        f"--querysubjectnum={row_number}",
        "-v",
    ]

    try:
        print(f"BIANCA Training: N={N}, condition={TRAIN_CONDITION}")
        subprocess.run(cmd, check=True)
        print(f"  Model saved: {BIANCA_MODEL}")
    except subprocess.CalledProcessError as e:
        print(f"  Training error: {e}")
        raise
else:
    print(f"Model already exists: {BIANCA_MODEL}")