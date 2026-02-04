#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIANCA Leave-One-Out Cross-Validation Pipeline

This module implements the leave-one-out (LOO) cross-validation framework for
BIANCA (Brain Intensity AbNormality Classification Algorithm) white matter
hyperintensity (WMH) segmentation as described in the paper:

    "Robustness and Error Susceptibility of BIANCA for White Matter 
     Hyperintensity Segmentation: The Roles of Lesion Volume and Scanner 
     Heterogeneity"

The pipeline automates the complete WMH segmentation workflow for Phase I 
performance optimization (n=103) including:
    - Automated training data preparation (excluding test subject)
    - BIANCA classifier training and testing
    - Probability map thresholding and binarization
    - White matter mask application
    - Periventricular and deep WMH stratification
    - Quantitative evaluation (Dice coefficient, volume metrics)
    - Multi-slice visualization panels

Paper Sections:
    - 2.2: Multi-phase validation study design
    - 2.4: Lesion removal assessment
    - 2.5: Preprocessing pipeline
    - 2.6: Lesion delineation and removal
    - 2.7: Algorithm configuration optimization

Author: Mendula (Center for Stroke Research Berlin)
License: MIT
"""

import os
import sys
import pandas as pd
import numpy as np
import subprocess
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ============================================================================
# CONFIGURATION - Environment Variables
# ============================================================================
# All paths should be set via environment variables for reproducibility.
# Example: export BIANCA_LOO_DIR="/path/to/derivatives/LOO"

def get_config() -> Dict[str, str]:
    """
    Load configuration from environment variables.
    
    Required environment variables:
        BIANCA_LOCATION_FILE: Excel file with subject metadata
        BIANCA_LOO_DIR: Output directory for LOO results
        BIANCA_IMAGES_DIR: Directory for visualization outputs
    
    Returns:
        dict: Configuration with paths
    
    Raises:
        EnvironmentError: If required variables not set
    """
    required_vars = [
        'BIANCA_LOCATION_FILE',
        'BIANCA_LOO_DIR',
        'BIANCA_IMAGES_DIR'
    ]
    
    config = {}
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            raise EnvironmentError(
                f"Required environment variable not set: {var}\n"
                f"Please set: export {var}='/path/to/file'"
            )
        config[var] = value
    
    return config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_non_empty_lines(file_path: str) -> int:
    """
    Count non-empty lines in file.
    
    Useful for determining the number of training subjects in BIANCA
    master files after excluding the test subject.
    
    Args:
        file_path: Path to file
    
    Returns:
        int: Number of non-empty lines
    """
    try:
        with open(file_path, 'r') as f:
            count = sum(1 for line in f if line.strip())
        return count
    except FileNotFoundError:
        print(f"    ✗ File not found: {file_path}")
        return 0
    except Exception as e:
        print(f"    ✗ Error reading file: {e}")
        return 0


def run_fsl_command(command: str, description: str) -> bool:
    """
    Execute FSL command with error handling and reporting.
    
    Args:
        command: FSL command as string (can use piping and redirects)
        description: Human-readable description of operation
    
    Returns:
        bool: True if successful, False otherwise
    
    Example:
        >>> run_fsl_command(
        ...     "fslmaths input.nii.gz -thr 10 -bin output.nii.gz",
        ...     "Thresholding image"
        ... )
    """
    print(f"  {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"    ✓ Success")
            return True
        else:
            print(f"    ✗ Failed with code {result.returncode}")
            if result.stderr:
                print(f"       Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"    ✗ Command timed out (>120s)")
        return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def apply_bianca_mask(
    input_path: str,
    mask_path: str,
    output_path: str
) -> bool:
    """
    Apply white matter mask to BIANCA binary segmentation.
    
    Uses FSL's -mas operator to eliminate false positives outside white
    matter. This is standard post-processing for BIANCA segmentations.
    See Paper Section 2.5 for preprocessing pipeline details.
    
    Args:
        input_path: Path to binary BIANCA segmentation (before masking)
        mask_path: Path to white matter binary mask
        output_path: Path for output masked segmentation
    
    Returns:
        bool: True if successful, False otherwise
    
    Raises:
        FileNotFoundError: If input or mask file doesn't exist
    """
    print(f"  Applying WM mask...")
    
    if not os.path.exists(input_path):
        print(f"    ✗ Input file not found: {input_path}")
        return False
    
    if not os.path.exists(mask_path):
        print(f"    ✗ WM mask file not found: {mask_path}")
        return False
    
    try:
        # Verify dimensions match
        input_img = nib.load(input_path)
        mask_img = nib.load(mask_path)
        
        if input_img.shape != mask_img.shape:
            print(f"    ⚠ WARNING: Shape mismatch: {input_img.shape} vs {mask_img.shape}")
        
        # Apply mask: output = input * mask (voxel-wise multiplication)
        result = subprocess.run(
            ["fslmaths", input_path, "-mas", mask_path, output_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"    ✗ FSL failed with code {result.returncode}")
            if result.stderr:
                print(f"       STDERR: {result.stderr[:200]}")
            return False
        
        if os.path.exists(output_path):
            print(f"    ✓ WM mask applied successfully")
            return True
        else:
            print(f"    ✗ Output file not created")
            return False
            
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def threshold_lpm(lpm_path: str, output_path: str, threshold: float = 0.85) -> bool:
    """
    Threshold BIANCA probability map to binary segmentation.
    
    Converts BIANCA's continuous probability output to binary segmentation
    using a fixed threshold. Paper Section 2.7 determined optimal threshold
    to be 0.85 based on precision-sensitivity analysis.
    
    Args:
        lpm_path: Path to BIANCA likelihood probability map
        output_path: Path for binary output
        threshold: Probability threshold (default: 0.85, Paper Section 2.7)
    
    Returns:
        bool: True if successful, False otherwise
    
    Note:
        Paper Figure 2 shows threshold optimization analysis.
        Threshold 0.85 was selected for optimal Dice coefficient (0.61).
    """
    try:
        result = subprocess.run(
            ["fslmaths", lpm_path, "-thr", str(threshold), "-bin", output_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        print(f"    ✗ Thresholding failed: {e}")
        return False


def get_volume(path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate voxel count and volume from binary segmentation.
    
    Uses FSL's fslstats to compute volume metrics. Automatically converts
    from mm³ to mL for reporting.
    
    Args:
        path: Path to binary segmentation image
    
    Returns:
        tuple: (voxel_count, volume_mm3, volume_ml)
               or (None, None, None) if calculation fails
    
    Example:
        >>> voxel_count, vol_mm3, vol_ml = get_volume("segmentation.nii.gz")
        >>> print(f"Total WMH volume: {vol_ml:.2f} mL")
    """
    try:
        output = subprocess.run(
            ["fslstats", path, "-V"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30
        )
        
        result_str = output.stdout.strip().split()
        voxel_count = float(result_str[0])
        volume_mm3 = float(result_str[1])
        volume_ml = volume_mm3 / 1000  # Convert mm³ to mL
        
        return voxel_count, volume_mm3, volume_ml
        
    except Exception as e:
        print(f"    ✗ Volume calculation failed: {e}")
        return None, None, None


def calculate_dice_metrics(
    prediction_path: str,
    reference_path: str
) -> Optional[Dict]:
    """
    Calculate Dice coefficient, sensitivity, and precision.
    
    Computes voxel-wise overlap metrics between automated prediction and
    manual reference segmentation. Used throughout Paper Section 3 to
    evaluate BIANCA performance.
    
    Metrics computed:
        - Dice coefficient: 2*TP / (2*TP + FP + FN)
        - Sensitivity (recall): TP / (TP + FN)
        - Precision: TP / (TP + FP)
    
    Args:
        prediction_path: Path to automated segmentation (binary)
        reference_path: Path to manual reference segmentation (binary)
    
    Returns:
        dict: Metrics including dice_score, sensitivity, precision, and
              confusion matrix values (TP, FP, FN, TN)
        None: If calculation fails
    
    Paper Reference:
        Primary optimization criterion (Paper Section 2.1):
        "maximal spatial overlap between automated and manual expert
         segmentation, quantified by the Dice similarity coefficient"
    """
    try:
        pred = nib.load(prediction_path).get_fdata()
        ref = nib.load(reference_path).get_fdata()
        
        # Binarize in case of floating point artifacts
        pred = (pred > 0.5).astype(int)
        ref = (ref > 0.5).astype(int)
        
        # Compute confusion matrix
        tp = np.sum((pred == 1) & (ref == 1))  # True positives
        fp = np.sum((pred == 1) & (ref == 0))  # False positives
        fn = np.sum((pred == 0) & (ref == 1))  # False negatives
        tn = np.sum((pred == 0) & (ref == 0))  # True negatives
        
        # Calculate metrics with division-by-zero protection
        dice = (2 * tp) / (2 * tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return {
            "dice_score": round(dice, 4),
            "sensitivity": round(sensitivity, 4),
            "precision": round(precision, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }
        
    except Exception as e:
        print(f"  ✗ Dice calculation failed: {e}")
        return None


def load_slice(image_path: str, slice_idx: int) -> np.ndarray:
    """
    Load single axial slice from NIfTI image.
    
    Args:
        image_path: Path to NIfTI file
        slice_idx: Slice index (z-axis)
    
    Returns:
        np.ndarray: 2D image array (rotated for display)
    """
    try:
        img = nib.load(image_path)
        data = img.get_fdata()
        return np.rot90(data[:, :, slice_idx])
    except Exception as e:
        print(f"    ✗ Failed to load slice {slice_idx}: {e}")
        return np.zeros((256, 256))


def overlay_multi_channel(
    base_image: np.ndarray,
    mask_list: List[np.ndarray]
) -> np.ndarray:
    """
    Create RGB overlay of multiple masks on base image.
    
    Creates publication-quality visualization with color-coded masks.
    Used for Paper Figures to display prediction vs manual segmentations.
    
    Args:
        base_image: 2D base image (grayscale, normalized)
        mask_list: List of 2D binary masks
    
    Returns:
        np.ndarray: RGB image array (H x W x 3)
    
    Color mapping:
        - Red: Prediction/BIANCA output
        - Green: Manual reference segmentation
        - Blue: Additional lesion masks (if provided)
    """
    # Normalize base image
    base_image = base_image / np.max(base_image) if np.max(base_image) > 0 else base_image
    rgb = np.stack([base_image] * 3, axis=-1)
    
    # Define colors [R, G, B]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
    
    # Overlay each mask with assigned color
    for i, mask in enumerate(mask_list):
        if mask is None or np.max(mask) == 0:
            continue
            
        mask = mask / np.max(mask) if np.max(mask) > 0 else mask
        color = colors[i % len(colors)]
        
        for c in range(3):
            rgb[:, :, c] = np.where(
                mask > 0,
                color[c] * mask + (1 - mask) * rgb[:, :, c],
                rgb[:, :, c]
            )
    
    return rgb


def get_best_slices(
    image_path: str,
    n_slices: int = 3
) -> List[int]:
    """
    Select best slices for visualization based on content.
    
    Automatically identifies slices with maximum lesion/WMH content for
    publication-quality figures.
    
    Args:
        image_path: Path to binary segmentation image
        n_slices: Number of slices to return (default: 3)
    
    Returns:
        list: Sorted slice indices
    """
    try:
        img = nib.load(image_path)
        data = img.get_fdata()
        
        # Compute content per slice (sum of non-zero voxels)
        slice_sums = [np.sum(data[:, :, i] > 0) for i in range(data.shape[2])]
        
        # Return indices of slices with most content
        best_indices = np.argsort(slice_sums)[-n_slices:]
        return sorted(best_indices)
        
    except Exception as e:
        print(f"    ✗ Failed to compute best slices: {e}")
        return []


def create_visualization_panel(
    images: List[Tuple[str, List[str]]],
    output_path: str,
    subject_name: str = "Subject",
    title: Optional[str] = None,
    titles_list: Optional[List[str]] = None,
    slice_indices: Optional[List[int]] = None,
    dpi: int = 150
) -> None:
    """
    Create multi-panel visualization comparing predictions and reference.
    
    Generates publication-quality figure panels showing:
        - Multiple imaging slices (rows)
        - Multiple contrasts/overlays (columns)
        - Color-coded segmentations (red=prediction, green=manual)
    
    Paper Reference:
        Figure 3: Representative white matter hyperintensity segmentation
        examples. Shows range of preprocessing effects with multi-slice
        visualization.
    
    Args:
        images: List of (background_path, [mask_paths]) tuples
        output_path: Path for output PNG file
        subject_name: Subject identifier for plot title
        title: Custom title (if None, auto-generated)
        titles_list: Column titles (if None, auto-generated)
        slice_indices: Z-indices for slices to display
        dpi: Output DPI (default: 150 for screen, use 300+ for publication)
    
    Returns:
        None (saves figure to disk)
    """
    if slice_indices is None:
        slice_indices = [0]
    if titles_list is None:
        titles_list = [f"Panel {i+1}" for i in range(len(images))]
    if title is None:
        title = f"{subject_name}"
    
    n_panels = len(images)
    n_slices = len(slice_indices)
    
    # Create figure grid (rows=slices, cols=panels)
    fig, axes = plt.subplots(
        n_slices, n_panels,
        figsize=(5 * n_panels, 5 * n_slices)
    )
    
    # Handle single row/column edge cases
    if n_slices == 1:
        axes = [axes]
    if n_panels == 1:
        axes = [[ax] for ax in axes]
    
    # Populate grid
    for i, slice_idx in enumerate(slice_indices):
        for j, (bg_path, mask_paths) in enumerate(images):
            ax = axes[i][j]
            
            if bg_path is None or not os.path.isfile(bg_path):
                ax.set_title(f"{titles_list[j]}\n[Missing]", fontsize=10)
                ax.axis("off")
                continue
            
            # Load base image
            base = load_slice(bg_path, slice_idx)
            
            # Load all masks for this panel
            overlays = []
            for mask_path in mask_paths:
                if mask_path and os.path.isfile(mask_path):
                    overlays.append(load_slice(mask_path, slice_idx))
            
            # Display with overlays
            if overlays:
                rgb_img = overlay_multi_channel(base, overlays)
                ax.imshow(rgb_img)
            else:
                ax.imshow(base, cmap="gray")
            
            ax.set_title(
                f"{titles_list[j]}\nSlice {slice_idx}",
                fontsize=10
            )
            ax.axis("off")
    
    # Format and save
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close()
    
    print(f"  ✓ Visualization saved: {output_path}")


# ============================================================================
# BIANCA TRAINING AND TESTING
# ============================================================================

def build_training_master_file(
    test_subject: str,
    location_df: pd.DataFrame,
    output_path: str
) -> Tuple[int, List[str]]:
    """
    Build BIANCA master file excluding test subject (leave-one-out).
    
    BIANCA requires a master file with paths to training data:
        Line format: FLAIR_path T1_path transformation_matrix manual_mask
    
    For LOO validation, each iteration trains on n-1 subjects and tests
    on the held-out subject. This ensures unbiased performance estimates.
    
    Paper Reference (Section 2.4 & 2.2):
        "leave-one-out cross-validation (LOO) on the full cohort (n=103),
         as originally proposed by Anbeek et al. (2004) and implemented
         within BIANCA"
    
    Args:
        test_subject: Subject ID to exclude from training
        location_df: DataFrame with all subject metadata
        output_path: Path for output master file
    
    Returns:
        tuple: (number_of_training_subjects, list_of_training_subject_ids)
    
    Raises:
        KeyError: If required columns not in DataFrame
    """
    master_lines = []
    training_subjects = []
    
    required_columns = [
        'FLAIR_brain_corrected_path',
        'T1_brain_corrected_path',
        'FLAIR_TO_mni_mat_path',
        'manualmask_path'
    ]
    
    for col in required_columns:
        if col not in location_df.columns:
            raise KeyError(f"Missing required column: {col}")
    
    # Iterate over all subjects except test subject
    for _, row in location_df.iterrows():
        if row['subject'] == test_subject:
            continue
        
        training_subjects.append(row['subject'])
        
        # Build BIANCA master file line
        line = (
            f"{row['FLAIR_brain_corrected_path']} "
            f"{row['T1_brain_corrected_path']} "
            f"{row['FLAIR_TO_mni_mat_path']} "
            f"{row['manualmask_path']}"
        )
        master_lines.append(line)
    
    # Write master file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(master_lines))
    
    return len(training_subjects), training_subjects


def run_bianca_training(
    master_file: str,
    output_prefix: str,
    num_training_subjects: int,
    force_retrain: bool = False
) -> bool:
    """
    Train BIANCA classifier on training subjects.
    
    BIANCA uses k-nearest neighbor (kNN) classification with features
    from T1 and FLAIR images. This function trains the classifier on
    the training set and saves the model for testing.
    
    Paper Reference (Section 2.7):
        Algorithm configuration details and feature specifications.
    
    Args:
        master_file: Path to BIANCA master file
        output_prefix: Prefix for output files (saved_train_data)
        num_training_subjects: Number of training subjects (for -trainingnums)
        force_retrain: Skip if classifier exists (unless True)
    
    Returns:
        bool: True if training successful
    
    Note:
        Training parameters fixed based on Paper Section 2.7:
        - trainingpts: 2000 random voxels per training subject
        - nonlespts: 10000 non-lesion voxels per training subject
        - spatialweight: 1 (spatial smoothing)
        - Features: FLAIR (1) and T1 (2); matrix (3) for registration
    """
    classifier_path = output_prefix
    
    # Check if classifier already exists
    if os.path.isfile(classifier_path) and not force_retrain:
        print(f"  ✓ Classifier exists (skipped)\n")
        return True
    
    print(f"Step 2: BIANCA training...")
    
    # Build training points string: "1,2,3,...,N"
    training_nums = ",".join(str(i) for i in range(1, num_training_subjects + 1))
    
    # BIANCA training command
    bianca_cmd = [
        "bianca",
        f"--singlefile={master_file}",
        "--brainmaskfeaturenum=1",      # Feature 1: brain mask
        "--matfeaturenum=3",             # Feature 3: transformation matrix
        "--featuresubset=1,2",           # Use FLAIR (1) and T1 (2)
        "--labelfeaturenum=4",           # Feature 4: manual mask
        "--trainingpts=2000",            # 2000 training voxels per subject
        "--querysubjectnum=" + str(num_training_subjects),
        "--nonlespts=10000",             # 10000 non-lesion voxels
        "--trainingnums=" + training_nums,
        "--selectpts=noborder",          # Exclude border voxels
        "--spatialweight=1",             # Spatial smoothing
        "--saveclassifierdata=" + classifier_path,
        "-v"  # Verbose output
    ]
    
    try:
        result = subprocess.run(
            bianca_cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0 and os.path.isfile(classifier_path):
            print(f"  ✓ Training completed\n")
            return True
        else:
            print(f"  ✗ Training failed (return code: {result.returncode})\n")
            if result.stderr:
                print(f"     Error: {result.stderr[:300]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Training timeout (>5 minutes)\n")
        return False
    except Exception as e:
        print(f"  ✗ Training error: {e}\n")
        return False


def run_bianca_testing(
    master_file: str,
    classifier_path: str,
    output_lpm: str,
    force_retest: bool = False
) -> bool:
    """
    Apply trained BIANCA classifier to test subject.
    
    Generates probability map (likelihood map) for test subject based on
    the trained classifier. Output ranges from 0-1, where 1 indicates
    voxels most likely to be WMH.
    
    Paper Reference (Section 2.7):
        Testing phase where trained classifier is applied to test subject
        to generate probability map.
    
    Args:
        master_file: Test subject master file (single subject)
        classifier_path: Path to trained classifier
        output_lpm: Path for output likelihood probability map
        force_retest: Skip if output exists (unless True)
    
    Returns:
        bool: True if testing successful
    """
    if os.path.isfile(output_lpm) and not force_retest:
        print(f"  ✓ LPM exists (skipped)\n")
        return True
    
    print(f"Step 3: BIANCA testing (segmentation)...")
    
    bianca_cmd = [
        "bianca",
        f"--singlefile={master_file}",
        "--brainmaskfeaturenum=1",
        "--matfeaturenum=3",
        "--featuresubset=1,2",
        "--loadclassifierdata=" + classifier_path,
        "--querysubjectnum=1",
        "-o", output_lpm,
        "-v"
    ]
    
    try:
        result = subprocess.run(
            bianca_cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0 and os.path.isfile(output_lpm):
            print(f"  ✓ Segmentation completed\n")
            return True
        else:
            print(f"  ✗ Segmentation failed (return code: {result.returncode})\n")
            return False
            
    except Exception as e:
        print(f"  ✗ Segmentation error: {e}\n")
        return False


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_subject_loo(
    subject_id: str,
    location_df: pd.DataFrame,
    config: Dict[str, str],
    subject_idx: int,
    total_subjects: int,
    threshold: float = 0.85
) -> None:
    """
    Complete LOO processing for single subject.
    
    Implements full pipeline:
        1. Build training data (excluding test subject)
        2. Train BIANCA classifier
        3. Test on held-out subject
        4. Threshold probability map
        5. Apply white matter mask
        6. Stratify into periventricular and deep WMH
        7. Calculate Dice metrics
        8. Create visualizations
    
    Paper Reference (Sections 2.2, 2.4-2.7, 3.1):
        Complete Phase I optimization workflow.
    
    Args:
        subject_id: Subject identifier
        location_df: DataFrame with all metadata
        config: Configuration dict with paths
        subject_idx: Current subject index (for progress)
        total_subjects: Total number of subjects
        threshold: BIANCA probability threshold (default: 0.85 from Paper Fig 2)
    
    Returns:
        None (saves outputs to subject directory)
    """
    print("=" * 80)
    print(f"PROCESSING: {subject_id} [{subject_idx+1}/{total_subjects}]")
    print("=" * 80 + "\n")
    
    # Create subject output directory
    subject_row = location_df[location_df['subject'] == subject_id].iloc[0]
    subject_dir = os.path.join(config['BIANCA_LOO_DIR'], subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    
    # Get auxiliary paths from metadata
    wm_mask_path = subject_row['WMmask_path']
    ventricle_distmap_path = subject_row['ventdistmap_path']
    
    # ========================================================================
    # Step 1: Build Training Master File (Leave-One-Out)
    # ========================================================================
    print("Step 1: Building training master file (LOO)...")
    
    master_file_path = os.path.join(subject_dir, "master_file.txt")
    num_training, training_subjects = build_training_master_file(
        subject_id,
        location_df,
        master_file_path
    )
    
    print(f"  ✓ {num_training} training subjects (excluding {subject_id})\n")
    
    # ========================================================================
    # Step 2: BIANCA Training
    # ========================================================================
    saved_classifier_path = os.path.join(subject_dir, f"{subject_id}_saved_train_data")
    
    if not run_bianca_training(
        master_file_path,
        saved_classifier_path,
        num_training
    ):
        print(f"  ✗ Training failed, skipping subject {subject_id}\n")
        return
    
    # ========================================================================
    # Step 3: BIANCA Testing
    # ========================================================================
    # Create test master file with only test subject
    flair_path = subject_row['FLAIR_brain_corrected_path']
    t1_path = subject_row['T1_brain_corrected_path']
    mat_path = subject_row['FLAIR_TO_mni_mat_path']
    manual_mask_path = subject_row['manualmask_path']
    
    test_master_path = os.path.join(subject_dir, f"{subject_id}_master_file_test.txt")
    with open(test_master_path, 'w') as f:
        f.write(f"{flair_path} {t1_path} {mat_path} {manual_mask_path}")
    
    output_lpm_path = os.path.join(subject_dir, f"{subject_id}_BIANCA_LPM.nii.gz")
    
    if not run_bianca_testing(
        test_master_path,
        saved_classifier_path,
        output_lpm_path
    ):
        print(f"  ✗ Testing failed, skipping subject {subject_id}\n")
        return
    
    # ========================================================================
    # Step 4: Threshold Probability Map
    # ========================================================================
    print("Step 4: Thresholding probability map...")
    
    binary_path = os.path.join(
        subject_dir,
        f"{subject_id}_BIANCA_LPM_th_{threshold}.nii.gz"
    )
    
    if not threshold_lpm(output_lpm_path, binary_path, threshold):
        print(f"  ✗ Thresholding failed\n")
        return
    
    voxel_count, vol_mm3, vol_ml = get_volume(binary_path)
    if vol_ml is not None:
        print(f"  ✓ Thresholded at {threshold}")
        print(f"  ✓ Volume (before WM mask): {vol_ml:.2f} mL\n")
    
    # ========================================================================
    # Step 5: Apply White Matter Mask
    # ========================================================================
    print("Step 5: Applying white matter mask...")
    
    binary_wm_path = os.path.join(
        subject_dir,
        f"{subject_id}_BIANCA_LPM_th_{threshold}_wm.nii.gz"
    )
    
    if not apply_bianca_mask(binary_path, wm_mask_path, binary_wm_path):
        print(f"  ✗ WM mask application failed\n")
        return
    
    vol_wm = get_volume(binary_wm_path)
    if vol_wm[2] is not None:
        print(f"  ✓ Volume (after WM mask): {vol_wm[2]:.2f} mL\n")
    
    # ========================================================================
    # Step 6: Stratify into Periventricular and Deep WMH
    # ========================================================================
    print("Step 6: Stratifying WMH by location...")
    
    # Deep WMH: distance > 10mm from ventricles
    deepwmh_mask_path = os.path.join(subject_dir, f"{subject_id}_desc-deepWMmask.nii.gz")
    if not os.path.isfile(deepwmh_mask_path):
        cmd = f"fslmaths {ventricle_distmap_path} -thr 10 -bin {deepwmh_mask_path}"
        run_fsl_command(cmd, "Creating deep WMH mask (distance > 10mm)")
    
    # Periventricular WMH: distance <= 10mm from ventricles
    peri_mask_path = os.path.join(subject_dir, f"{subject_id}_desc-periventmask.nii.gz")
    if not os.path.isfile(peri_mask_path):
        cmd = f"fslmaths {ventricle_distmap_path} -uthr 10 -bin {peri_mask_path}"
        run_fsl_command(cmd, "Creating periventricular WMH mask (distance <= 10mm)")
    
    # Apply masks and compute volumes
    deepwmh_seg_path = os.path.join(subject_dir, f"{subject_id}_BIANCA_deepWMH_wm.nii.gz")
    peri_seg_path = os.path.join(subject_dir, f"{subject_id}_BIANCA_periventWMH_wm.nii.gz")
    
    cmd_deep = f"fslmaths {binary_wm_path} -mas {deepwmh_mask_path} {deepwmh_seg_path}"
    cmd_peri = f"fslmaths {binary_wm_path} -mas {peri_mask_path} {peri_seg_path}"
    
    run_fsl_command(cmd_deep, "Applying deep WMH mask")
    run_fsl_command(cmd_peri, "Applying periventricular WMH mask")
    
    print()
    
    # ========================================================================
    # Step 7: Evaluate Performance (Dice Metrics)
    # ========================================================================
    print("Step 7: Calculating Dice metrics...")
    
    metrics_before = calculate_dice_metrics(binary_path, manual_mask_path)
    metrics_after = calculate_dice_metrics(binary_wm_path, manual_mask_path)
    
    if metrics_before:
        print(f"  Before WM mask: Dice={metrics_before['dice_score']:.4f}, "
              f"Sensitivity={metrics_before['sensitivity']:.4f}, "
              f"Precision={metrics_before['precision']:.4f}")
    
    if metrics_after:
        print(f"  After WM mask:  Dice={metrics_after['dice_score']:.4f}, "
              f"Sensitivity={metrics_after['sensitivity']:.4f}, "
              f"Precision={metrics_after['precision']:.4f}\n")
    
    # ========================================================================
    # Step 8: Create Visualizations
    # ========================================================================
    print("Step 8: Creating visualizations...")
    
    os.makedirs(config['BIANCA_IMAGES_DIR'], exist_ok=True)
    
    # Select best slices for visualization
    best_slices = get_best_slices(manual_mask_path, n_slices=3)
    print(f"  ✓ Best slices: {best_slices}\n")
    
    # Panel 1: Before WM mask application
    plot_before_path = os.path.join(
        config['BIANCA_IMAGES_DIR'],
        f"{subject_id}_panel_before_wm.png"
    )
    
    images_before = [
        (flair_path, []),
        (flair_path, [binary_path]),
        (flair_path, [manual_mask_path])
    ]
    
    titles_before = [
        "FLAIR",
        f"Prediction\n(Dice: {metrics_before['dice_score']:.4f})",
        "Manual Reference"
    ]
    
    try:
        create_visualization_panel(
            images=images_before,
            output_path=plot_before_path,
            subject_name=subject_id,
            title=f"{subject_id} - Before WM Mask",
            titles_list=titles_before,
            slice_indices=best_slices,
            dpi=150
        )
    except Exception as e:
        print(f"  ✗ Panel creation failed: {e}")
    
    # Panel 2: After WM mask application
    plot_after_path = os.path.join(
        config['BIANCA_IMAGES_DIR'],
        f"{subject_id}_panel_after_wm.png"
    )
    
    images_after = [
        (flair_path, []),
        (flair_path, [binary_wm_path]),
        (flair_path, [manual_mask_path])
    ]
    
    titles_after = [
        "FLAIR",
        f"Prediction\n(Dice: {metrics_after['dice_score']:.4f})",
        "Manual Reference"
    ]
    
    try:
        create_visualization_panel(
            images=images_after,
            output_path=plot_after_path,
            subject_name=subject_id,
            title=f"{subject_id} - After WM Mask",
            titles_list=titles_after,
            slice_indices=best_slices,
            dpi=150
        )
    except Exception as e:
        print(f"  ✗ Panel creation failed: {e}")
    
    print("=" * 80)
    print("✓ SUBJECT PROCESSING COMPLETE")
    print("=" * 80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point - processes all subjects in LOO cross-validation.
    
    Sets up configuration from environment variables, loads subject list,
    and iterates through leave-one-out validation for each subject.
    
    Usage:
        export BIANCA_LOCATION_FILE="/path/to/location_data.xlsx"
        export BIANCA_LOO_DIR="/path/to/derivatives/LOO"
        export BIANCA_IMAGES_DIR="/path/to/derivatives/LOO_images"
        python bianca_loo_crossvalidation.py
    """
    print("\n" + "=" * 80)
    print("BIANCA LEAVE-ONE-OUT CROSS-VALIDATION PIPELINE")
    print("Phase I: Performance Optimization (Paper Section 2.2)")
    print("=" * 80 + "\n")
    
    # Load configuration
    try:
        config = get_config()
        print("✓ Configuration loaded successfully\n")
    except EnvironmentError as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)
    
    # Load subject data
    try:
        location_df = pd.read_excel(config['BIANCA_LOCATION_FILE'])
        subject_list = list(location_df['subject'])
        print(f"✓ Loaded {len(subject_list)} subjects\n")
    except Exception as e:
        print(f"✗ Error loading subject data: {e}")
        sys.exit(1)
    
    print("=" * 80 + "\n")
    
    # Process each subject
    for idx, subject_id in enumerate(subject_list):
        try:
            process_subject_loo(
                subject_id,
                location_df,
                config,
                subject_idx=idx,
                total_subjects=len(subject_list),
                threshold=0.85  # Paper Section 2.7: optimal threshold
            )
        except Exception as e:
            print(f"\n✗ Error processing {subject_id}: {e}\n")
            continue
    
    print("=" * 80)
    print("✓ ALL SUBJECTS PROCESSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
