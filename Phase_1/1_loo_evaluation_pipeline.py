"""
================================================================================
Leave-One-Out Cross-Validation Data Generation Pipeline
================================================================================

Paper Reference: "Robustness and Error Susceptibility of BIANCA for White 
Matter Hyperintensity Segmentation"

This script generates the data for:
- Figure 2: Threshold optimization analysis (precision-sensitivity curves)
- Supplemental Figure 1: DICE Score comparison across threshold strategies

Methodology (Section 2.4):
"To assess whether the presence of lesions affects WMH segmentation accuracy, 
we employed leave-one-out cross-validation (LOO) on the full cohort (n=103), 
as originally proposed by Anbeek et al. (2004) and implemented within BIANCA 
(Griffanti et al., 2016). For each subject, BIANCA's k-NN classifier was 
trained on manually labeled WMH voxels from the remaining 102 subjects and 
then applied to segment WMH in the test subject."

Threshold Analysis (Section 2.7):
"A systematic threshold analysis across the full range of thresholds (0.0-1.0) 
was performed generating precision-sensitivity curves and Dice coefficient 
optimization plots to empirically determine the optimal threshold."

Metrics Computed:
- Dice coefficient (primary outcome)
- Hausdorff Distance 95th percentile (HD95)
- Sensitivity (recall)
- Precision

Output:
- Excel file with per-subject, per-threshold metrics for downstream analysis

Author: Uchralt Temuulen
================================================================================
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('bianca_loo_evaluation')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define paths relative to project root (modify for your environment)
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '.')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'loo_evaluation')

# LOO results directories
LOO_WITH_REMOVAL_DIR = os.path.join(DATA_DIR, 'LOO_WITH_REMOVAL')
LOO_WITHOUT_REMOVAL_DIR = os.path.join(DATA_DIR, 'LOO_WITHOUT_REMOVAL')

# Output file
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, 'loo_results_ROC.xlsx')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Threshold range for systematic analysis
# Paper Section 2.7: "threshold analysis across the full range (0.0-1.0)"
THRESHOLD_START = 0.01
THRESHOLD_END = 1.00
THRESHOLD_STEP = 0.01


# ==============================================================================
# METRIC COMPUTATION FUNCTIONS
# ==============================================================================

def compute_dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Dice similarity coefficient.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Paper Reference (Section 2.1):
    "Throughout this study, we define optimal segmentation performance as 
    maximal spatial overlap between automated and manual expert segmentation, 
    quantified by the Dice similarity coefficient."
    
    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask
    gt : np.ndarray
        Binary ground truth mask
        
    Returns
    -------
    float
        Dice coefficient (0-1)
    """
    pred_binary = (pred > 0).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    sum_masks = np.sum(pred_binary) + np.sum(gt_binary)
    
    if sum_masks == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0
    
    return 2.0 * intersection / sum_masks


def compute_sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute sensitivity (true positive rate / recall).
    
    Sensitivity = TP / (TP + FN)
    
    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask
    gt : np.ndarray
        Binary ground truth mask
        
    Returns
    -------
    float
        Sensitivity (0-1)
    """
    pred_binary = (pred > 0).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    true_positives = np.sum(pred_binary * gt_binary)
    actual_positives = np.sum(gt_binary)
    
    if actual_positives == 0:
        return 1.0 if np.sum(pred_binary) == 0 else 0.0
    
    return true_positives / actual_positives


def compute_precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute precision (positive predictive value).
    
    Precision = TP / (TP + FP)
    
    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask
    gt : np.ndarray
        Binary ground truth mask
        
    Returns
    -------
    float
        Precision (0-1)
    """
    pred_binary = (pred > 0).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    true_positives = np.sum(pred_binary * gt_binary)
    predicted_positives = np.sum(pred_binary)
    
    if predicted_positives == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0
    
    return true_positives / predicted_positives


def compute_hd95(pred: np.ndarray, gt: np.ndarray, voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """
    Compute 95th percentile Hausdorff Distance (HD95).
    
    HD95 is more robust to outliers than maximum Hausdorff distance.
    
    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask
    gt : np.ndarray
        Binary ground truth mask
    voxel_spacing : tuple
        Voxel dimensions in mm
        
    Returns
    -------
    float
        HD95 in mm
    """
    pred_binary = (pred > 0).astype(np.uint8)
    gt_binary = (gt > 0).astype(np.uint8)
    
    # Handle empty masks
    if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0:
        return 0.0
    if np.sum(pred_binary) == 0 or np.sum(gt_binary) == 0:
        return np.inf
    
    # Extract surface points using erosion
    pred_surface = pred_binary ^ binary_erosion(pred_binary)
    gt_surface = gt_binary ^ binary_erosion(gt_binary)
    
    # Get coordinates of surface points
    pred_coords = np.array(np.where(pred_surface)).T
    gt_coords = np.array(np.where(gt_surface)).T
    
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.inf
    
    # Apply voxel spacing to convert to mm
    pred_coords_mm = pred_coords * np.array(voxel_spacing)
    gt_coords_mm = gt_coords * np.array(voxel_spacing)
    
    # Compute pairwise distances
    distances_pred_to_gt = cdist(pred_coords_mm, gt_coords_mm).min(axis=1)
    distances_gt_to_pred = cdist(gt_coords_mm, pred_coords_mm).min(axis=1)
    
    # Combine and get 95th percentile
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    hd95 = np.percentile(all_distances, 95)
    
    return hd95


def compute_volume_ml(mask: np.ndarray, voxel_volume_mm3: float) -> float:
    """
    Compute lesion volume in milliliters.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    voxel_volume_mm3 : float
        Volume of single voxel in mm³
        
    Returns
    -------
    float
        Volume in mL (mm³ / 1000)
    """
    voxel_count = np.sum(mask > 0)
    volume_mm3 = voxel_count * voxel_volume_mm3
    return volume_mm3 / 1000.0  # Convert to mL


# ==============================================================================
# IMAGE PROCESSING FUNCTIONS
# ==============================================================================

def load_nifti(filepath: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load NIfTI image and return data array and image object.
    
    Parameters
    ----------
    filepath : str
        Path to NIfTI file
        
    Returns
    -------
    tuple
        (data_array, nib_image)
    """
    img = nib.load(filepath)
    data = img.get_fdata()
    return data, img


def match_orientation_and_compare(gt_path: str, pred_path: str) -> Dict[str, float]:
    """
    Load masks, match orientations, and compute all metrics.
    
    Paper Reference (Section 2.5):
    "Subsequent processing included rigid registration of FLAIR to T1-weighted 
    images followed by affine normalization of both modalities to the 1 mm 
    MNI-152 template space."
    
    Parameters
    ----------
    gt_path : str
        Path to ground truth mask
    pred_path : str
        Path to prediction mask
        
    Returns
    -------
    dict
        Dictionary with all computed metrics
    """
    # Load images
    gt_data, gt_img = load_nifti(gt_path)
    pred_data, pred_img = load_nifti(pred_path)
    
    # Resample prediction to ground truth space if needed
    if gt_img.shape != pred_img.shape:
        logger.info(f"Resampling prediction from {pred_img.shape} to {gt_img.shape}")
        pred_resampled = resample_from_to(pred_img, gt_img, order=0)
        pred_data = pred_resampled.get_fdata()
    
    # Get voxel spacing
    voxel_dims = gt_img.header.get_zooms()[:3]
    voxel_volume = np.prod(voxel_dims)
    
    # Compute metrics
    metrics = {
        'dice': compute_dice_coefficient(pred_data, gt_data),
        'sensitivity': compute_sensitivity(pred_data, gt_data),
        'precision': compute_precision(pred_data, gt_data),
        'hausdorff_distance': compute_hd95(pred_data, gt_data, voxel_dims)
    }
    
    return metrics


def apply_white_matter_mask(input_path: str, wm_mask_path: str, output_path: str):
    """
    Apply white matter mask to restrict segmentation to WM regions.
    
    Paper Reference (Section 2.5):
    "A white matter mask, which combines tissue segmentation with nonlinear 
    registration fields to MNI space, applied during post-processing to 
    eliminate extracerebral false positives."
    
    Parameters
    ----------
    input_path : str
        Path to input segmentation
    wm_mask_path : str
        Path to white matter mask
    output_path : str
        Path for masked output
    """
    cmd = ["fslmaths", input_path, "-mas", wm_mask_path, output_path]
    subprocess.run(cmd, check=True)


def threshold_probability_map(lpm_path: str, threshold: float, output_path: str):
    """
    Threshold BIANCA lesion probability map to create binary mask.
    
    Paper Reference (Section 2.7):
    "A systematic threshold analysis across the full range of thresholds 
    (0.0-1.0) was performed."
    
    Parameters
    ----------
    lpm_path : str
        Path to lesion probability map
    threshold : float
        Threshold value (0-1)
    output_path : str
        Path for binary output
    """
    cmd = [
        "fslmaths", lpm_path,
        "-thr", str(threshold),
        "-bin",
        output_path
    ]
    subprocess.run(cmd, check=True)


# ==============================================================================
# FILE ORGANIZATION FUNCTIONS
# ==============================================================================

def build_subject_dict(file_list: List[str]) -> Dict[str, Dict]:
    """
    Organize files into subject-based dictionary structure.
    
    Parameters
    ----------
    file_list : list
        List of file paths
        
    Returns
    -------
    dict
        Nested dictionary: subject_id -> file_type -> path
    """
    subject_dict = {}
    
    # Define file type identifiers
    file_types = {
        'feature_FLAIR': ['feature_flair', 'flair'],
        'feature_T1': ['feature_t1w', 't1w'],
        'manualmask': ['manualmask', 'manual_mask', 'gt_mask'],
        'biancamask': ['biancamask', 'wm_mask'],
        'BIANCA_LPM': ['bianca_lpm', 'lpm', 'probability_map'],
        'ventdistmap': ['ventdistmap', 'vent_dist']
    }
    
    for filepath in file_list:
        filename = os.path.basename(filepath).lower()
        
        # Extract subject ID (assumes format: subject_xxx_type.nii.gz)
        parts = filename.replace('.nii.gz', '').replace('.nii', '').split('_')
        if len(parts) >= 2:
            subject_id = '_'.join(parts[:2]) if parts[0] in ['sub', 'subject', 'phillips', 'challenge'] else parts[0]
        else:
            subject_id = parts[0]
        
        # Initialize subject entry
        if subject_id not in subject_dict:
            subject_dict[subject_id] = {'normal': {}}
        
        # Categorize file
        for file_type, keywords in file_types.items():
            if any(kw in filename for kw in keywords):
                subject_dict[subject_id]['normal'][file_type] = filepath
                break
    
    return subject_dict


# ==============================================================================
# MAIN EVALUATION PIPELINE
# ==============================================================================

def evaluate_loo_subject(subject_id: str,
                          subject_data: Dict,
                          output_dir: str,
                          train_set_label: str,
                          thresholds: np.ndarray) -> List[Dict]:
    """
    Evaluate single subject across all thresholds.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    subject_data : dict
        Dictionary with file paths for this subject
    output_dir : str
        Directory for intermediate outputs
    train_set_label : str
        Label indicating training configuration
    thresholds : np.ndarray
        Array of threshold values to test
        
    Returns
    -------
    list
        List of result dictionaries (one per threshold)
    """
    results = []
    
    # Get required paths
    normal_dict = subject_data.get('normal', {})
    
    flair_path = normal_dict.get('feature_FLAIR')
    manual_mask_path = normal_dict.get('manualmask')
    wm_mask_path = normal_dict.get('biancamask')
    lpm_path = normal_dict.get('BIANCA_LPM')
    
    # Validate required files exist
    if not all([manual_mask_path, wm_mask_path, lpm_path]):
        logger.warning(f"Missing required files for {subject_id}, skipping")
        return results
    
    # Compute ground truth volume
    gt_data, gt_img = load_nifti(manual_mask_path)
    voxel_dims = gt_img.header.get_zooms()[:3]
    voxel_volume = np.prod(voxel_dims)
    volume_ml = compute_volume_ml(gt_data, voxel_volume)
    
    # Create subject output directory
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # Evaluate each threshold
    for threshold in thresholds:
        threshold_int = int(round(threshold * 100))
        
        # Generate thresholded binary mask
        thresh_output = os.path.join(
            subject_output_dir, 
            f"{subject_id}_bianca_thresh_{threshold_int}.nii.gz"
        )
        threshold_probability_map(lpm_path, threshold, thresh_output)
        
        # Apply white matter mask
        corrected_path = os.path.join(
            subject_output_dir,
            f"{subject_id}_bianca_thresh_{threshold_int}_wm.nii.gz"
        )
        apply_white_matter_mask(thresh_output, wm_mask_path, corrected_path)
        
        # Compute metrics
        metrics = match_orientation_and_compare(manual_mask_path, corrected_path)
        
        # Record results
        results.append({
            'subject_id': subject_id,
            'volume_ml': round(volume_ml, 2),
            'dice_score': round(metrics['dice'], 4),
            'hausdorff_distance': round(metrics['hausdorff_distance'], 2),
            'threshold': threshold_int,
            'sensitivity': round(metrics['sensitivity'], 4),
            'precision': round(metrics['precision'], 4),
            'TRAIN_SET': train_set_label
        })
    
    return results


def run_full_evaluation(subject_dict: Dict,
                         output_dir: str,
                         train_set_label: str,
                         output_excel: str):
    """
    Run evaluation for all subjects and save results.
    
    Parameters
    ----------
    subject_dict : dict
        Dictionary with subject data
    output_dir : str
        Output directory for intermediate files
    train_set_label : str
        Label for training configuration
    output_excel : str
        Path for output Excel file
    """
    # Generate threshold array
    thresholds = np.round(np.arange(THRESHOLD_START, THRESHOLD_END + THRESHOLD_STEP, THRESHOLD_STEP), 2)
    
    logger.info(f"Starting evaluation for {len(subject_dict)} subjects")
    logger.info(f"Threshold range: {thresholds[0]} to {thresholds[-1]} (step: {THRESHOLD_STEP})")
    logger.info(f"Total evaluations: {len(subject_dict) * len(thresholds)}")
    
    all_results = []
    
    for i, (subject_id, subject_data) in enumerate(subject_dict.items()):
        logger.info(f"Processing {subject_id} ({i+1}/{len(subject_dict)})")
        
        subject_results = evaluate_loo_subject(
            subject_id=subject_id,
            subject_data=subject_data,
            output_dir=output_dir,
            train_set_label=train_set_label,
            thresholds=thresholds
        )
        
        all_results.extend(subject_results)
        
        # Save intermediate results
        if (i + 1) % 10 == 0:
            df = pd.DataFrame(all_results)
            df.to_excel(output_excel, index=False)
            logger.info(f"Intermediate save: {len(all_results)} results")
    
    # Final save
    df = pd.DataFrame(all_results)
    df.to_excel(output_excel, index=False)
    logger.info(f"Evaluation complete. {len(all_results)} results saved to {output_excel}")
    
    return df


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main pipeline for LOO cross-validation evaluation.
    
    Paper Reference:
    - Section 2.4: Leave-one-out cross-validation methodology
    - Section 2.7: Algorithm configuration optimization
    - Figure 2: Threshold optimization analysis
    - Supplemental Figure 1: DICE Score comparison
    """
    print("=" * 80)
    print("LEAVE-ONE-OUT CROSS-VALIDATION EVALUATION")
    print("Paper: Robustness and Error Susceptibility of BIANCA")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Threshold range: {THRESHOLD_START} to {THRESHOLD_END}")
    print(f"  Threshold step: {THRESHOLD_STEP}")
    print(f"  Output: {OUTPUT_EXCEL}")
    
    # -------------------------------------------------------------------------
    # Validate input directories
    # -------------------------------------------------------------------------
    print("\n[1] Checking input directories...")
    
    if not os.path.exists(LOO_WITH_REMOVAL_DIR):
        raise FileNotFoundError(
            f"LOO with removal directory not found: {LOO_WITH_REMOVAL_DIR}\n"
            f"Please update the DATA_DIR path."
        )
    
    # -------------------------------------------------------------------------
    # Build subject dictionary
    # -------------------------------------------------------------------------
    print("\n[2] Building subject dictionary...")
    
    # Collect all NIfTI files
    all_files = []
    for root, dirs, files in os.walk(LOO_WITH_REMOVAL_DIR):
        for f in files:
            if f.endswith('.nii.gz'):
                all_files.append(os.path.join(root, f))
    
    subject_dict = build_subject_dict(all_files)
    print(f"  Found {len(subject_dict)} subjects")
    
    # -------------------------------------------------------------------------
    # Run evaluation
    # -------------------------------------------------------------------------
    print("\n[3] Running evaluation...")
    
    results_df = run_full_evaluation(
        subject_dict=subject_dict,
        output_dir=OUTPUT_DIR,
        train_set_label="BIANCA trained with removal",
        output_excel=OUTPUT_EXCEL
    )
    
    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal results: {len(results_df)}")
    print(f"Unique subjects: {results_df['subject_id'].nunique()}")
    print(f"Thresholds tested: {results_df['threshold'].nunique()}")
    
    # Find optimal threshold based on mean Dice
    mean_dice_by_thresh = results_df.groupby('threshold')['dice_score'].mean()
    optimal_thresh = mean_dice_by_thresh.idxmax()
    optimal_dice = mean_dice_by_thresh.max()
    
    print(f"\nOptimal threshold (by mean Dice): {optimal_thresh}")
    print(f"Mean Dice at optimal: {optimal_dice:.4f}")
    
    # Paper-relevant thresholds
    for thresh in [85, 90]:
        if thresh in results_df['threshold'].values:
            thresh_dice = results_df[results_df['threshold'] == thresh]['dice_score'].mean()
            print(f"Mean Dice at {thresh/100:.2f}: {thresh_dice:.4f}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return results_df


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    results_df = main()