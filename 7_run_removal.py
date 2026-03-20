#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase II Removal Pipeline: Robustness Assessment
=================================================

Revision context (NeuroImage: Clinical, Major Revision)
-------------------------------------------------------
This script implements the Phase II-A (n=86) and Phase II-B (n=211)
robustness assessment and addresses the following reviewer concerns:

  (1) Zero-filling / lesion replacement (R1 Comment 1; R5 #9)
      R1 criticized zero-intensity replacement as non-physiological.
      R5 noted inpainting alternatives were acknowledged but not tested.

      Resolution: Three preprocessing conditions are evaluated:
        - non_removed: no lesion preprocessing
        - removed: zero-intensity lesion replacement
        - inpainted (filled): NAWM-based inpainting (FSL lesion_filling)
      BIANCA inference + LOCATE adaptive thresholding applied to all
      three conditions. Convergence across conditions demonstrates that
      effects are attributable to lesion removal, not filling strategy.

  (2) Phase II-A vs Phase II-B distinction (R2 Comment 2)
      R2 asked for clarification of the conceptual difference between
      Phase II-A and Phase II-B beyond sample size.

      Resolution: Phase II-A (n=86) provides ground-truth-referenced
      accuracy metrics (Dice, sensitivity, precision) for subjects with
      expert-delineated WMH masks. Phase II-B (n=211) extends to the
      full BeLOVE cohort for volume-based robustness assessment with
      increased statistical power to detect scanner effects.
      Results are saved separately (Phase_2_a with Dice, Phase_2_b all).

  (3) Periventricular vs deep WMH stratification
      Segmentation output is split into periventricular and deep WMH
      using anatomical masks, enabling region-specific evaluation as
      requested by reviewers for more granular assessment.

Paper changes
-------------
  Results section 3.2 (Phase II-A, revised):
    - Three conditions reported: non_removed, removed, inpainted.
    - Dice, sensitivity, precision for whole/deep/periventricular WMH.
    - Language revised per reviewer feedback.

  Results section 3.3 (Phase II-B, revised):
    - Volume comparisons across three conditions.
    - Scanner-stratified analysis with increased Philips representation.

  Supplemental material:
    - Panel plots per subject showing FLAIR + masks for all conditions.

  Response to Reviewers:
    - R1 Comment 1: Inpainted condition added.
    - R5 #9: Inpainting tested alongside zero-filling.
    - R2 Comment 2: Phase distinction clarified.

Technical design
----------------
For each BeLOVE removal subject:
  1. BIANCA inference using inpainted-trained model (Script 7,
     BIANCA_MODEL_N_{N}_FILLED). Training condition selected based on
     5-fold CV results showing equivalent performance across all three
     training conditions (Dice differences in 4th decimal place).
     Inpainted chosen as methodologically most conservative option
     (R1 Comment 1; R5 #9).
  2. LOCATE adaptive thresholding
  3. Lesion correction (subtract stroke lesion mask from output)
  4. Split into periventricular / deep WMH regions
  5. Volume computation (whole, deep, periventricular)
  6. Dice evaluation against ground truth (Phase II-A subjects only)
  7. Panel plot generation

Output structure:
  Phase_2_a/RESULTS/       - Subjects with ground truth (Dice scores)
  Phase_2_b/RESULTS/       - All subjects (volumes for all, Dice where available)
  Phase_2_a/RESULTS/plots/ - Panel plots for all subjects

Terminology (consistent across all scripts):
  - non_removed: no lesion preprocessing
  - removed: zero-intensity lesion replacement
  - inpainted / filled: NAWM-based inpainting (FSL lesion_filling)

"""

import os
import subprocess
import logging
import tempfile
import pandas as pd
from typing import List

from DATASETS.librarys.directory_lib import (
    create_panel_plot_with_multi_overlay,
    create_panel_plot_with_multi_overlay_paper,
    get_top_slices,
    get_files_from_dir,
    find_one_element,
    find_elements,
    fsl_copy,
    get_volume,
    fsl_dice_score,
)


# ==========================================
# FUNCTIONS
# ==========================================

def run_bianca(flair_path, t1_path, mni_mat_path, model_path, output_dir, subject, force=False):
    """Run BIANCA inference using a pre-trained model."""
    empty_file = os.path.join(output_dir, f"{subject}_empty.nii.gz")
    if not os.path.isfile(empty_file):
        subprocess.run(["fslmaths", flair_path, "-mul", "0", empty_file], check=True)

    master_file = os.path.join(output_dir, f"{subject}_master_file_test.txt")
    with open(master_file, 'w') as f:
        f.write(f"{flair_path} {t1_path} {mni_mat_path} {empty_file}")

    lpm_path = os.path.join(output_dir, f"{subject}_BIANCA_LPM.nii.gz")

    if force or not os.path.isfile(lpm_path):
        cmd = [
            "bianca",
            f"--singlefile={master_file}",
            "--brainmaskfeaturenum=1",
            "--matfeaturenum=3",
            "--featuresubset=1,2",
            f"--loadclassifierdata={model_path}",
            "--querysubjectnum=1",
            "-o", lpm_path,
            "-v"
        ]
        subprocess.run(cmd, check=True)

    return lpm_path


def run_locate_testing(
    train_image_directory_path: str,
    test_image_directory_name: str,
    locate_path: str,
    verbose: int = 1,
    feature_select: List[int] = [1, 1, 1, 1]
) -> None:
    """Run LOCATE testing with proper error handling."""
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


# ==========================================
# SETUP & PATHS
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_dir = "Phase_1/BIANCA_MODELS"
BIANCA_MODEL = os.path.join(model_dir, "BIANCA_MODEL_N_59_FILLED")

REMOVAL_BELOVE_DATASET_BIDS = "DATASETS/REMOVAL_BELOVE_DATASET_BIDS"
BELOVE_BIDS_WMH_BIDS        = "DATASETS/BELOVE_BIDS_WMH_BIDS"

# Output directories:
#   Phase_2_a: subjects with ground truth (Dice scores)
#   Phase_2_b: all subjects (volumes, + Dice where available)
#   PLOT_DIR:  panel plots for all subjects
Phase_2_a_results = "Phase_2_a/RESULTS"
Phase_2_b_results = "Phase_2_b/RESULTS"

PLOT_DIR = os.path.join(Phase_2_a_results, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(Phase_2_a_results, exist_ok=True)
os.makedirs(Phase_2_b_results, exist_ok=True)


train_image_directory_path = (
    "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/"
    "DATASET_SC/PROJECT_NULL_07_04_2025/1_Project_all_code_all_data/"
    "Phase_1/LOCATE_SET/locate_train"
)
LOCATE_PATH = (
    "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/"
    "Projects/BIANCA/LOCATE/LOCATE-BIANCA"
)
feature_select = [1, 1, 1, 1]
dice_round = 4


BELOVE_BIDS_WMH_preprocessed_files   = os.path.join(BELOVE_BIDS_WMH_BIDS, "derivatives/preprocessed_files.xlsx")
BELOVE_BIDS_WMH_file_locations       = os.path.join(BELOVE_BIDS_WMH_BIDS, "derivatives/BELOVE_BIDS_WMH_file_locations.xlsx")
BELOVE_BIDS_WMH_meta_locations       = os.path.join(BELOVE_BIDS_WMH_BIDS, "derivatives/BELOVE_K_META_DATA.xlsx")

BELOVE_BIDS_WMH_preprocessed_df      = pd.read_excel(BELOVE_BIDS_WMH_preprocessed_files)
BELOVE_BIDS_WMH_file_df              = pd.read_excel(BELOVE_BIDS_WMH_file_locations)
BELOVE_BIDS_WMH_meta_df              = pd.read_excel(BELOVE_BIDS_WMH_meta_locations)



BELOVE_BIDS_WMH_DF                  = BELOVE_BIDS_WMH_preprocessed_df.merge(BELOVE_BIDS_WMH_file_df, on='subject', how='inner')
BELOVE_BIDS_WMH_meta_df['subject']  = BELOVE_BIDS_WMH_meta_df['subject'].str.replace('belove_kersten-', 'belove_k-')
BELOVE_BIDS_WMH_meta_df             = BELOVE_BIDS_WMH_meta_df[['subject', 'name', 'flair_file_name', 'sex', 'wahlund', 'age']]

BELOVE_BIDS_WMH_DF                  = BELOVE_BIDS_WMH_DF.merge(BELOVE_BIDS_WMH_meta_df, on='subject', how='inner')


#this ist the removal data

REMOVAL_BELOVE_preprocessed_files   = os.path.join(REMOVAL_BELOVE_DATASET_BIDS, "derivatives/preprocessed_files.xlsx")
REMOVAL_BELOVE_file_locations       = os.path.join(REMOVAL_BELOVE_DATASET_BIDS, "derivatives/REMOVAL_BELOVE_file_locations.xlsx")


# --- Load and Merge Data ---
REMOVAL_BELOVE_preprocessed_df = pd.read_excel(REMOVAL_BELOVE_preprocessed_files)
REMOVAL_BELOVE_removal_df      = pd.read_excel(REMOVAL_BELOVE_file_locations)
REMOVAL_BELOVE_DF              = REMOVAL_BELOVE_preprocessed_df.merge(REMOVAL_BELOVE_removal_df, on='subject', how='inner')

subjects_removal = list(REMOVAL_BELOVE_DF['subject'])

data_excel = "Phase_1/BIANCA_MODELS/bianca_scanner_pool.xlsx"
meta_excel = "Phase_1/BIANCA_MODELS/bianca_scanner_pool_meta.xlsx"


belove_k_meta_df = pd.read_excel(meta_excel)
belove_k_subjects = belove_k_meta_df[belove_k_meta_df['TEST'] == 'TEST']['subject'].tolist()


subjects = belove_k_subjects + subjects_removal


# Challenge-derived severity cutoffs (Decision Tree, N=40)
SEVERITY_CUTOFF_LOW_MID = 6.96    # mL
SEVERITY_CUTOFF_MID_HIGH = 27.40  # mL


DEBUG = True
DEBUG = False
if DEBUG:
    subjects = subjects[:10]

# ==========================================
# MAIN PIPELINE (SUBJECT LOOP)
# ==========================================
from typing import List, Dict, Optional

subjects_with_source = []
for s in belove_k_subjects:
    subjects_with_source.append((s, "belove_k"))
for s in subjects_removal:
    subjects_with_source.append((s, "removal"))
    
    
all_result_concat = pd.DataFrame()

for si, (subject, source) in enumerate(subjects_with_source[:]):
    print(f"\n[{si}] Processing {subject} (source={source})...")
    
    
    severity_level = ""
    
    # --- Resolve subject row from the correct DataFrame ---
    if source == "belove_k":
        subject_row = BELOVE_BIDS_WMH_DF[BELOVE_BIDS_WMH_DF['subject'] == subject].iloc[0]
        
    else:
        subject_row = REMOVAL_BELOVE_DF[REMOVAL_BELOVE_DF['subject'] == subject].iloc[0]
        
    ROI_Volume = subject_row.get('ROI_Volume', "")
    severity_level = subject_row.get('severity_level', "")
    

    # Assign severity from BeLOVE Cohort A cutoffs (N=130) if missing
    if (not severity_level or pd.isna(severity_level)) and ROI_Volume and not pd.isna(ROI_Volume):
        roi = float(ROI_Volume)
        if roi < SEVERITY_CUTOFF_LOW_MID:
            severity_level = "low"
        elif roi < SEVERITY_CUTOFF_MID_HIGH:
            severity_level = "middle"
        else:
            severity_level = "high"


    has_roi = subject_row['has_roi']
    # --- Resolve all paths via unified helper ---

    # All processing goes to Phase_2_b; Phase_2_a gets Dice-filtered copy at the end
    sub_2_a_dir = os.path.join(Phase_2_a_results, subject)
    sub_2_b_dir = os.path.join(Phase_2_b_results, subject)
    
    if "belove_k" in subject:
        base = BELOVE_BIDS_WMH_BIDS
        work_dir = sub_2_a_dir
    else:
        base = REMOVAL_BELOVE_DATASET_BIDS
        work_dir = sub_2_b_dir


    T1_path = os.path.join(base, str(subject_row['T1']))
    WMmask_path = os.path.join(base, str(subject_row['WMmask']))
    brainmask_path = os.path.join(base, str(subject_row['brainmask']))
    ventdistmap_path = os.path.join(base, str(subject_row['ventdistmap']))
    wmh_roi_file = os.path.join(base, str(subject_row['WMH_removed_path']))

    #check with os.path.isfile

    infarct_path = ""
    if 'lesion_path' in subject_row and not pd.isna(subject_row['lesion_path']):
        infarct_path = os.path.join(base, str(subject_row['lesion_path']))

    masks_dir = os.path.join(
        base, "derivatives", "fsl_anat", subject, "ses-01", "masks"
    )
    
    deepWMmask_path = os.path.join(masks_dir, f"{subject}_desc-deepWMmask.nii.gz")
    periventmask_path = os.path.join(masks_dir, f"{subject}_desc-periventmask.nii.gz")

    has_deepWMmask = os.path.isfile(deepWMmask_path)
    has_periventmask = os.path.isfile(periventmask_path)

    if not has_deepWMmask:
        print(f"  Warning: {subject}: deepWMmask not found: {deepWMmask_path}")
    if not has_periventmask:
        print(f"  Warning: {subject}: periventmask not found: {periventmask_path}")


    if pd.isna(subject_row['T1']) or not os.path.exists(T1_path):
        raise FileNotFoundError(f"Subject {subject}: T1 not found at {T1_path}")

    if pd.isna(subject_row['mni_mat_path']):
        raise ValueError(f"Subject {subject}: column 'mni_mat_path' is NaN.")

    mni_mat = os.path.join(base, str(subject_row['mni_mat_path']))
    if not os.path.exists(mni_mat):
        raise FileNotFoundError(f"Subject {subject}: mni_mat not found at {mni_mat}")

    # ==========================================
    # GROUND TRUTH PREPARATION (Phase 2a only)
    # ==========================================
    manualmask = ""
    manual_per_WMH = ""
    manual_deep_WMH = ""

    if has_roi:
        gt_dir = os.path.join(sub_2_a_dir, "ground_truth")
        os.makedirs(gt_dir, exist_ok=True)

        manualmask = os.path.join(gt_dir, f"{subject}_manualmask.nii.gz")

        if os.path.exists(wmh_roi_file) and not os.path.exists(manualmask):
            fsl_copy(wmh_roi_file, manualmask)

        manual_per_WMH = os.path.join(gt_dir, f"{subject}_manual_periventricular.nii.gz")
        manual_deep_WMH = os.path.join(gt_dir, f"{subject}_manual_deepWMH.nii.gz")

        if os.path.exists(manualmask):
            if os.path.exists(periventmask_path) and not os.path.exists(manual_per_WMH):
                subprocess.run(["fslmaths", manualmask, "-mul", periventmask_path, manual_per_WMH], check=True)
            if os.path.exists(deepWMmask_path) and not os.path.exists(manual_deep_WMH):
                subprocess.run(["fslmaths", manualmask, "-mul", deepWMmask_path, manual_deep_WMH], check=True)

    # ==========================================
    # INFERENCE LOOP OVER CONDITIONS
    # ==========================================
    subject_metrics = {}

    conditions = [
        {"name": "non_removed", "primary_col": "FLAIR_non_removed_path"},
        {"name": "removed",     "primary_col": "FLAIR_removed_path"},
        {"name": "filled",      "primary_col": "FLAIR_filled_path"},
    ]

    for cond in conditions:
        
        cond_name = cond["name"]
        primary_col = cond["primary_col"]

        raw_val = subject_row[primary_col]
        if pd.isna(raw_val):
            print(f"  Warning: Subject {subject}: column '{primary_col}' is NaN. Skipping condition {cond_name}.")
            continue

        flair_source = os.path.join(base, str(raw_val))
        if not os.path.exists(flair_source):
            raise FileNotFoundError(f"Subject {subject}: FLAIR not found at {flair_source}")

        # --- BIANCA & LOCATE ---

        work_dir = sub_2_a_dir if source == "belove_k" else sub_2_b_dir
        dir_b = os.path.join(work_dir, cond_name)
        

        os.makedirs(dir_b, exist_ok=True)
        flair_b = os.path.join(dir_b, f"{subject}_FLAIR_{cond_name}.nii.gz")
        fsl_copy(flair_source, flair_b)

        print(f"  Running BIANCA for '{cond_name}'...")
        lpm_b = run_bianca(flair_b, T1_path, mni_mat, BIANCA_MODEL, dir_b, subject)

        LOCATE_sub_dir = os.path.join(dir_b, "locate")
        os.makedirs(LOCATE_sub_dir, exist_ok=True)

        progress_file = os.path.join(LOCATE_sub_dir, "IN_PROGRESS.txt")
        results_dir = os.path.join(LOCATE_sub_dir, "LOCATE_results_directory")

        existing_results = (
            [f for f in os.listdir(results_dir) if "BIANCA_LOCATE_binarylesionmap" in f]
            if os.path.exists(results_dir) else []
        )

        if os.path.exists(progress_file):
            print(f"  Warning: {subject}: LOCATE in progress. Skipping...")
            continue

        if not existing_results:
            try:
                with open(progress_file, 'w') as f:
                    f.write(f"Started: {pd.Timestamp.now()}\nSubject: {subject}")

                print(f"  Starting LOCATE for {subject} ({cond_name})...")

                fsl_copy(flair_b, os.path.join(LOCATE_sub_dir, f"{subject}_feature_FLAIR.nii.gz"))
                fsl_copy(lpm_b, os.path.join(LOCATE_sub_dir, f"{subject}_BIANCA_LPM.nii.gz"))
                fsl_copy(T1_path, os.path.join(LOCATE_sub_dir, f"{subject}_feature_t1w.nii.gz"))
                fsl_copy(WMmask_path, os.path.join(LOCATE_sub_dir, f"{subject}_biancamask.nii.gz"))
                fsl_copy(brainmask_path, os.path.join(LOCATE_sub_dir, f"{subject}_brainmask.nii.gz"))
                fsl_copy(ventdistmap_path, os.path.join(LOCATE_sub_dir, f"{subject}_ventdistmap.nii.gz"))

                run_locate_testing(
                    train_image_directory_path=train_image_directory_path,
                    test_image_directory_name=os.path.abspath(LOCATE_sub_dir),
                    locate_path=LOCATE_PATH,
                    verbose=1,
                    feature_select=feature_select,
                )
            finally:
                if os.path.exists(progress_file):
                    os.remove(progress_file)
        else:
            print(f"  {subject} ({cond_name}): LOCATE results already exist.")

        existing_results = (
            [f for f in os.listdir(results_dir) if "BIANCA_LOCATE_binarylesionmap" in f]
            if os.path.exists(results_dir) else []
        )
        if not existing_results:
            raise FileNotFoundError(
                f"Subject {subject} ({cond_name}): No LOCATE results found in {results_dir}"
            )

        locate_result_path = os.path.join(results_dir, existing_results[0])

        # --- Lesion correction ---
        lesion_corrected = os.path.join(results_dir, f"{subject}_lesion_corrected.nii.gz")
        if infarct_path and os.path.exists(infarct_path):
            #command = f"fslmaths {locate_result_path} -sub {infarct_path} -bin {lesion_corrected}"
            command = f"fslmaths {locate_result_path} -sub {infarct_path} -thr 0 -bin {lesion_corrected}"
            subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        else:
            fsl_copy(locate_result_path, lesion_corrected)


        # --- Separate into Periventricular and Deep WMH ---
        out_per_WMH = os.path.join(dir_b, f"{subject}_{cond_name}_periventricular.nii.gz")
        out_deep_WMH = os.path.join(dir_b, f"{subject}_{cond_name}_deepWMH.nii.gz")

        if os.path.exists(periventmask_path):
            subprocess.run(["fslmaths", lesion_corrected, "-mul", periventmask_path, out_per_WMH], check=True)
            print(f"    Created periventricular mask for {subject} ({cond_name})")

        if os.path.exists(deepWMmask_path):
            subprocess.run(["fslmaths", lesion_corrected, "-mul", deepWMmask_path, out_deep_WMH], check=True)
            print(f"    Created deep WMH mask for {subject} ({cond_name})")

        # --- Volumes ---
        _, _, vol_whole = get_volume(lesion_corrected)
        _, _, vol_deep = get_volume(out_deep_WMH) if os.path.exists(out_deep_WMH) else (0, 0, 0)
        _, _, vol_per = get_volume(out_per_WMH) if os.path.exists(out_per_WMH) else (0, 0, 0)

        cond_metrics = {
            'whole_vol': vol_whole,
            'deep_vol': vol_deep,
            'per_vol': vol_per,
        }

        # --- Dice evaluation (only if has_roi) ---
        if has_roi and os.path.exists(manualmask):
            eval_whole = fsl_dice_score(lesion_corrected, manualmask)
            cond_metrics['whole_dice'] = round(eval_whole["dice_score"], dice_round)
            cond_metrics['whole_sens'] = round(eval_whole["sensitivity"], dice_round)
            cond_metrics['whole_prec'] = round(eval_whole["precision"], dice_round)

            if os.path.exists(out_deep_WMH) and os.path.exists(manual_deep_WMH):
                eval_deep = fsl_dice_score(out_deep_WMH, manual_deep_WMH)
                cond_metrics['deep_dice'] = round(eval_deep["dice_score"], dice_round)
                cond_metrics['deep_sens'] = round(eval_deep["sensitivity"], dice_round)
                cond_metrics['deep_prec'] = round(eval_deep["precision"], dice_round)

            if os.path.exists(out_per_WMH) and os.path.exists(manual_per_WMH):
                eval_per = fsl_dice_score(out_per_WMH, manual_per_WMH)
                cond_metrics['per_dice'] = round(eval_per["dice_score"], dice_round)
                cond_metrics['per_sens'] = round(eval_per["sensitivity"], dice_round)
                cond_metrics['per_prec'] = round(eval_per["precision"], dice_round)

        subject_metrics[cond_name] = cond_metrics

    # ==========================================
    # BUILD DATAFRAME FOR THIS SUBJECT
    # ==========================================

    _, _, brain_volume_ml = get_volume(T1_path)
    if infarct_path and os.path.exists(infarct_path):
        _, _, infarct_volume_ml = get_volume(infarct_path)
    else:
        infarct_volume_ml = 0.0

    df_temp = pd.DataFrame(index=[si])

    name_col = "name" if source == "belove_k" else "original_patien_name"
    df_temp['original_patien_name'] = str(subject_row.get(name_col, ""))

    df_temp['subject'] = subject
    df_temp['scanner'] = str(subject_row.get("scanner", ""))
    df_temp['Wahlund'] = str(subject_row.get("Wahlund", ""))
    df_temp['sex'] = str(subject_row.get("sex", ""))
    df_temp['age'] = str(subject_row.get("age", ""))
    df_temp['lesion_type'] = str(subject_row.get("lesion_type", ""))
    df_temp['threshold'] = "locate"

    roi_vol = subject_row.get("ROI_Volume", 0.0)
    df_temp['ROI_Volume'] = float(roi_vol) if pd.notna(roi_vol) else 0.0
    df_temp['brain_volume_ml'] = brain_volume_ml
    df_temp['infarct_volume_ml'] = infarct_volume_ml
    df_temp['source'] = source
    
    df_temp['severity_level'] = severity_level
    

    for cond in conditions:
        c_name = cond["name"]
        metrics = subject_metrics.get(c_name, {})

        df_temp[f'WMH_{c_name}_volume_ml'] = metrics.get('whole_vol', 0.0)
        df_temp[f'deepWMH_{c_name}_volume_ml'] = metrics.get('deep_vol', 0.0)
        df_temp[f'perWMH_{c_name}_volume_ml'] = metrics.get('per_vol', 0.0)

        if has_roi:
            df_temp[f'WMH_{c_name}_dice'] = metrics.get('whole_dice', "")
            df_temp[f'WMH_{c_name}_sens'] = metrics.get('whole_sens', "")
            df_temp[f'WMH_{c_name}_prec'] = metrics.get('whole_prec', "")

            df_temp[f'deepWMH_{c_name}_dice'] = metrics.get('deep_dice', "")
            df_temp[f'deepWMH_{c_name}_sens'] = metrics.get('deep_sens', "")
            df_temp[f'deepWMH_{c_name}_prec'] = metrics.get('deep_prec', "")

            df_temp[f'perWMH_{c_name}_dice'] = metrics.get('per_dice', "")
            df_temp[f'perWMH_{c_name}_sens'] = metrics.get('per_sens', "")
            df_temp[f'perWMH_{c_name}_prec'] = metrics.get('per_prec', "")

    all_result_concat = pd.concat([all_result_concat, df_temp], ignore_index=True)

    # ==========================================
    # PLOT GENERATION
    # ==========================================

    infarct_path = os.path.join(base, str(subject_row['lesion_path']))
    manual_mask_path = os.path.join(base, str(subject_row["WMH_removed_path"]))


    dir_nr = os.path.join(work_dir, "non_removed")
    dir_f = os.path.join(work_dir, "filled")
    dir_r = os.path.join(work_dir, "removed")

    flair_nr = os.path.join(dir_nr, f"{subject}_FLAIR_non_removed.nii.gz")
    flair_f = os.path.join(dir_f, f"{subject}_FLAIR_filled.nii.gz")
    flair_r = os.path.join(dir_r, f"{subject}_FLAIR_removed.nii.gz")

    wmh_nr_deep = os.path.join(dir_nr, f"{subject}_non_removed_deepWMH.nii.gz")
    wmh_nr_per = os.path.join(dir_nr, f"{subject}_non_removed_periventricular.nii.gz")
    wmh_f_deep = os.path.join(dir_f, f"{subject}_filled_deepWMH.nii.gz")
    wmh_f_per = os.path.join(dir_f, f"{subject}_filled_periventricular.nii.gz")
    wmh_r_deep = os.path.join(dir_r, f"{subject}_removed_deepWMH.nii.gz")
    wmh_r_per = os.path.join(dir_r, f"{subject}_removed_periventricular.nii.gz")

    wmh_nr_total = os.path.join(dir_nr, "locate", "LOCATE_results_directory", f"{subject}_lesion_corrected.nii.gz")
    wmh_r_total = os.path.join(dir_r, "locate", "LOCATE_results_directory", f"{subject}_lesion_corrected.nii.gz")

    # --- Difference masks ---
    diff_R_gt_NR = os.path.join(work_dir, f"{subject}_diff_R_gt_NR.nii.gz")
    diff_NR_gt_R = os.path.join(work_dir, f"{subject}_diff_NR_gt_R.nii.gz")

    if os.path.exists(wmh_r_total) and os.path.exists(wmh_nr_total):
        subprocess.run(["fslmaths", wmh_r_total, "-sub", wmh_nr_total, "-thr", "0.5", "-bin", diff_R_gt_NR], check=True)
        subprocess.run(["fslmaths", wmh_nr_total, "-sub", wmh_r_total, "-thr", "0.5", "-bin", diff_NR_gt_R], check=True)
    else:
        print(f"  Warning: {subject}: Lesion corrected masks missing. Diff masks not created.")

    # --- Panel plot (only for subjects with lesion + ROI) ---
    if infarct_path and os.path.exists(infarct_path) and has_roi:
        top_slices = get_top_slices(infarct_path, top_n=1)

        manual_mask_path = (
            os.path.join(base, str(subject_row["WMH_removed_path"]))
            if has_roi else None
        )

        images_config = [
            (flair_nr, [p for p in [infarct_path, None, None, manual_mask_path] if p is None or os.path.exists(p)]),
            (flair_nr, [p for p in [None, wmh_nr_deep, wmh_nr_per] if p is None or os.path.exists(str(p))]),
            (flair_f,  [p for p in [None, wmh_f_deep, wmh_f_per] if p is None or os.path.exists(str(p))]),
            (flair_r,  [p for p in [None, wmh_r_deep, wmh_r_per] if p is None or os.path.exists(str(p))]),
        ]

        titles = [
            "A\n Lesion &\n WMH Manual Mask",
            "B\n Not Removed",
            "C\n Inpainted",
            "D\n Removed",
        ]

        out_file = os.path.join(PLOT_DIR, f"{subject}_final_analysis.png")

        create_panel_plot_with_multi_overlay_paper(
            images_config,
            out_file,
            subject_name=subject,
            titles_list=titles,
            slice_indices=top_slices,
            title=""
        )
        
        print(f"  Plot saved: {out_file}")

# ==========================================
# SAVE RESULTS
# ==========================================
print("\nProcessing complete!")
print(all_result_concat.head())

# 1. Save complete dataset (Phase 2b: all subjects)
output_excel_all = os.path.join(Phase_2_b_results, "LOCATE_Results_Metrics_ALL.xlsx")
all_result_concat.to_excel(output_excel_all, index=False)
print(f"All results saved: {output_excel_all}")

# 2. Filter: only subjects with Dice scores (Phase 2a)
df_with_dice = all_result_concat[
    (all_result_concat['WMH_non_removed_dice'] != "") &
    (all_result_concat['WMH_non_removed_dice'].notna())
].copy()

print(f"\nFiltered: {len(df_with_dice)} of {len(all_result_concat)} subjects have Dice scores.")

# 3. Save filtered dataset (Phase 2a)
output_excel_filtered = os.path.join(Phase_2_a_results, "LOCATE_Results_Metrics_DICE_ONLY.xlsx")
df_with_dice.to_excel(output_excel_filtered, index=False)
print(f"Filtered results saved: {output_excel_filtered}")