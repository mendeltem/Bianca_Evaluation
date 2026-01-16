#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness Evaluation Module for BIANCA WMH Segmentation

This module implements the Phase II robustness assessment comparing
R (removed) vs NR (non-removed) preprocessing strategies for the study:

"Robustness and Error Susceptibility of BIANCA for White Matter 
Hyperintensity Segmentation: The Roles of Lesion Volume and Scanner Heterogeneity"

Author: Uchralt Temuulen
Affiliation: Center for Stroke Research Berlin, Charité-Universitätsmedizin Berlin

Paper Sections Implemented:
    - Section 2.4: Lesion removal assessment
        "We compared two preprocessing strategies:
        (1) R (removed): lesions were masked from FLAIR images prior to 
            BIANCA training and segmentation
        (2) NR (not removed): BIANCA segmentation was performed on original 
            FLAIR images, with lesion voxels subtracted post-hoc from the 
            resulting WMH maps"
    
    - Section 3.2: Results - Robustness Phase II-A (n=86)
        "Bland-Altman analysis demonstrated systematic directional differences 
        between R and NR segmentation approaches"
    
    - Section 3.3: Results - Robustness Phase II-B (n=211)
        "R segmentation consistently produced larger WMH volumes than NR 
        conditions with magnitude varying according to lesion etiology"
    
    - Table 1: Comparison of WMH segmentation volumes by lesion type
    - Table 2: Volume differences by scanner type
    - Figure 4: Bland-Altman agreement analysis
    - Figure 5: Correlation between lesion volume and WMH differences

Key Methodology:
    - NR Condition: BIANCA processes original FLAIR, lesions subtracted post-hoc
    - R Condition: Lesions masked from FLAIR before BIANCA processing
    - Metrics: Dice coefficient, sensitivity, precision, volume differences
    - Regional analysis: Total, periventricular (≤10mm), deep (>10mm) WMH

Environment Variables Required:
    - STANDARD_SPACE_T1: Path to MNI-152 T1 template
    - DATA_SET: Path to input dataset directory
    - BIANCA_MODEL: Path to trained BIANCA classifier model
    - THRESHHOLD_BIANCA: Probability threshold (default: 0.85)
    - SUBJECT_DATA_PATH: Path to subject metadata Excel file
    - CLINICAL_DATA_PATH: Path to clinical data Excel file (optional)
    - OUTPUT_DIR: Base directory for outputs

Dependencies:
    - FSL 6.0+ (https://fsl.fmrib.ox.ac.uk/fsl/)
    - HD-BET (https://github.com/MIC-DKFZ/HD-BET)
    - pandas, numpy, nibabel
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import nibabel as nib


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_robustness_config() -> Dict[str, Any]:
    """
    Load configuration for robustness evaluation from environment variables.
    
    Retrieves all necessary paths and parameters, ensuring reproducibility
    across different computing environments without hardcoded paths.
    
    Returns:
        Dict with configuration parameters
        
    Raises:
        EnvironmentError: If required environment variables are missing
        
    Paper Reference:
        Section 2.2 - Multi-phase validation study design requires
        consistent configuration across all experimental phases
    """
    config = {
        # Required paths
        'standard_space_t1': os.getenv('STANDARD_SPACE_T1'),
        'bianca_model': os.getenv('BIANCA_MODEL'),
        'subject_data_path': os.getenv('SUBJECT_DATA_PATH'),
        'output_dir': os.getenv('OUTPUT_DIR'),
        
        # Optional paths
        'clinical_data_path': os.getenv('CLINICAL_DATA_PATH'),
        'fsldir': os.getenv('FSLDIR', '/usr/local/fsl'),
        
        # Parameters
        'threshold': float(os.getenv('THRESHHOLD_BIANCA', '0.85')),
        'ventricle_distance_threshold': float(os.getenv('VENT_DIST_THRESHOLD', '10.0')),
    }
    
    # Validate required variables
    required = ['standard_space_t1', 'bianca_model', 'subject_data_path', 'output_dir']
    missing = [k for k in required if not config.get(k)]
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please configure these in your .env file."
        )
    
    return config


# Default threshold validated in Section 2.7
DEFAULT_THRESHOLD = 0.85

# Lesion type mapping (Section 2.6)
LESION_TYPE_MAP = {
    1: 'infratentorial',
    2: 'lacune',
    3: 'infarct',
    4: 'mixed',
    5: 'ICH'
}

# Sex mapping
SEX_MAP = {
    0: 'Women',
    1: 'Men'
}

# Scanner mapping (Section 2.3)
SCANNER_MAP = {
    1: 'Siemens',
    2: 'Philips'
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_fsl_command(
    command: str,
    description: str = "",
    check: bool = True
) -> bool:
    """
    Execute an FSL command with error handling.
    
    Args:
        command: FSL command string to execute
        description: Human-readable description for logging
        check: Whether to raise exception on failure
        
    Returns:
        bool: True if command succeeded, False otherwise
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if description:
            logging.info(f"✓ {description}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ {description}: {e.stderr}")
        return False


def fsl_copy(source: str, destination: str) -> bool:
    """
    Copy a NIfTI file using FSL's imcp (handles .nii.gz properly).
    
    Args:
        source: Path to source image
        destination: Path to destination
        
    Returns:
        bool: True if copy succeeded
    """
    return run_fsl_command(
        f"imcp {source} {destination}",
        f"Copy {Path(source).name}"
    )


def get_volume(nifti_path: str) -> Tuple[float, float, float]:
    """
    Calculate volume of non-zero voxels using fslstats.
    
    Args:
        nifti_path: Path to NIfTI image
        
    Returns:
        Tuple of (voxel_count, volume_mm3, volume_ml)
        
    Paper Reference:
        Table 1 - Volume measurements reported in mL for
        NR and R segmentation conditions
    """
    try:
        # Get voxel count
        result = subprocess.run(
            f"fslstats {nifti_path} -V",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        parts = result.stdout.strip().split()
        voxel_count = float(parts[0])
        volume_mm3 = float(parts[1])
        volume_ml = volume_mm3 / 1000.0
        
        return voxel_count, volume_mm3, volume_ml
        
    except (subprocess.CalledProcessError, IndexError, ValueError) as e:
        logging.error(f"Volume calculation failed for {nifti_path}: {e}")
        return 0.0, 0.0, 0.0


def calculate_dice_coefficient(
    segmentation_path: str,
    reference_path: str
) -> Dict[str, float]:
    """
    Calculate Dice coefficient and related metrics between two binary masks.
    
    The Dice coefficient was chosen as the primary optimization criterion
    because it balances precision and sensitivity in a single metric.
    
    Args:
        segmentation_path: Path to automated segmentation
        reference_path: Path to manual reference mask
        
    Returns:
        Dict with dice_score, sensitivity, precision
        
    Paper Reference:
        Section 2.1 - "We define optimal segmentation performance as maximal 
        spatial overlap between automated and manual expert segmentation, 
        quantified by the Dice similarity coefficient"
        
        Dice = 2|A ∩ B| / (|A| + |B|)
        Sensitivity = |A ∩ B| / |B| (true positive rate)
        Precision = |A ∩ B| / |A| (positive predictive value)
    """
    try:
        seg_img = nib.load(segmentation_path)
        ref_img = nib.load(reference_path)
        
        seg_data = (seg_img.get_fdata() > 0).astype(np.float32)
        ref_data = (ref_img.get_fdata() > 0).astype(np.float32)
        
        intersection = np.sum(seg_data * ref_data)
        seg_sum = np.sum(seg_data)
        ref_sum = np.sum(ref_data)
        
        # Dice coefficient
        if (seg_sum + ref_sum) > 0:
            dice = (2.0 * intersection) / (seg_sum + ref_sum)
        else:
            dice = 0.0
        
        # Sensitivity (recall, true positive rate)
        if ref_sum > 0:
            sensitivity = intersection / ref_sum
        else:
            sensitivity = 0.0
        
        # Precision (positive predictive value)
        if seg_sum > 0:
            precision = intersection / seg_sum
        else:
            precision = 0.0
        
        return {
            'dice_score': dice,
            'sensitivity': sensitivity,
            'precision': precision
        }
        
    except Exception as e:
        logging.error(f"Dice calculation failed: {e}")
        return {'dice_score': 0.0, 'sensitivity': 0.0, 'precision': 0.0}


# =============================================================================
# REGIONAL MASK CREATION
# =============================================================================

def create_deep_wmh_mask(
    ventricle_distance_map: str,
    output_path: str,
    distance_threshold: float = 10.0
) -> bool:
    """
    Create deep white matter mask from ventricle distance map.
    
    Deep WMH are defined as lesions located >10mm from the ventricles,
    which appear more often as small punctuated lesions.
    
    Args:
        ventricle_distance_map: Path to ventricle distance map
        output_path: Path for output deep WM mask
        distance_threshold: Distance threshold in mm (default: 10mm)
        
    Returns:
        bool: True if mask creation succeeded
        
    Paper Reference:
        Introduction - "Deep (subcortical) WMH which appear more often 
        as small punctuated lesions (Sundaresan et al., 2021)"
        
    Command:
        fslmaths ventdist -thr 10 -bin deep_mask
    """
    return run_fsl_command(
        f"fslmaths {ventricle_distance_map} -thr {distance_threshold} -bin {output_path}",
        "Create deep WMH mask (>10mm from ventricles)"
    )


def create_periventricular_wmh_mask(
    ventricle_distance_map: str,
    output_path: str,
    distance_threshold: float = 10.0
) -> bool:
    """
    Create periventricular white matter mask from ventricle distance map.
    
    Periventricular WMH are defined as lesions located ≤10mm from ventricles,
    which appear brighter and tend to form confluent lesions.
    
    Args:
        ventricle_distance_map: Path to ventricle distance map
        output_path: Path for output periventricular mask
        distance_threshold: Distance threshold in mm (default: 10mm)
        
    Returns:
        bool: True if mask creation succeeded
        
    Paper Reference:
        Introduction - "Periventricular WMHs are reported to appear brighter 
        and tend to form confluent lesions alongside the ventricles"
        
    Command:
        fslmaths ventdist -uthr 10 -bin peri_mask
    """
    return run_fsl_command(
        f"fslmaths {ventricle_distance_map} -uthr {distance_threshold} -bin {output_path}",
        "Create periventricular WMH mask (≤10mm from ventricles)"
    )


def create_combined_brain_mask(
    deep_mask_path: str,
    periventricular_mask_path: str,
    output_path: str
) -> bool:
    """
    Create combined mask (union of deep and periventricular regions).
    
    This mask represents the total white matter region for WMH analysis.
    
    Args:
        deep_mask_path: Path to deep WM mask
        periventricular_mask_path: Path to periventricular mask
        output_path: Path for combined output mask
        
    Returns:
        bool: True if mask creation succeeded
        
    Command:
        fslmaths deep_mask -add peri_mask -bin combined_mask
    """
    return run_fsl_command(
        f"fslmaths {deep_mask_path} -add {periventricular_mask_path} -bin {output_path}",
        "Create combined WM mask"
    )


# =============================================================================
# LESION REMOVAL FOR R CONDITION
# =============================================================================

def remove_lesion_from_flair(
    flair_path: str,
    lesion_mask_path: str,
    output_path: str
) -> bool:
    """
    Remove lesion regions from FLAIR image by zeroing out lesion voxels.
    
    This implements the "R" (removed) preprocessing strategy where
    lesion voxels are set to zero intensity before BIANCA processing.
    
    Args:
        flair_path: Path to input FLAIR brain image
        lesion_mask_path: Path to binary lesion mask (1 = lesion region)
        output_path: Path for output image with lesions removed
        
    Returns:
        bool: True if removal succeeded
        
    Paper Reference:
        Section 2.4 - "R (removed): lesions were masked from FLAIR 
        images prior to BIANCA training and segmentation"
        
        Section 2.6 - "For lesion removal, the resulting ROIs were 
        binarized and multiplied with the corresponding FLAIR image 
        using fslmaths, effectively replacing lesion voxel intensities 
        with zero values and preventing their inclusion in training 
        or testing datasets."
        
    Method:
        1. Invert lesion mask (0 -> 1, 1 -> 0)
        2. Multiply FLAIR by inverted mask
        This zeros out lesion regions while preserving other voxels.
    """
    temp_inverted = output_path.replace('.nii.gz', '_temp_inv.nii.gz')
    
    try:
        # Step 1: Invert the lesion mask
        subprocess.run(
            ['fslmaths', lesion_mask_path, '-binv', temp_inverted],
            check=True,
            capture_output=True
        )
        
        # Step 2: Multiply FLAIR by inverted mask (zeros out lesions)
        subprocess.run(
            ['fslmaths', flair_path, '-mul', temp_inverted, output_path],
            check=True,
            capture_output=True
        )
        
        # Cleanup temporary file
        if os.path.exists(temp_inverted):
            os.remove(temp_inverted)
        
        logging.info(f"✓ Lesion removal completed: {Path(output_path).name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ Lesion removal failed: {e}")
        return False


def subtract_lesion_from_segmentation(
    segmentation_path: str,
    lesion_mask_path: str,
    output_path: str
) -> bool:
    """
    Subtract lesion mask from WMH segmentation (post-hoc correction for NR).
    
    In the NR condition, lesions are subtracted post-hoc from the
    BIANCA output to remove any incorrectly segmented lesion voxels.
    
    Args:
        segmentation_path: Path to BIANCA WMH segmentation
        lesion_mask_path: Path to lesion mask
        output_path: Path for corrected output
        
    Returns:
        bool: True if subtraction succeeded
        
    Paper Reference:
        Section 2.4 - "NR (not removed): BIANCA segmentation was performed 
        on original FLAIR images, with lesion voxels subtracted post-hoc 
        from the resulting WMH maps"
        
    Command:
        fslmaths segmentation -sub lesion_mask -bin output
    """
    return run_fsl_command(
        f"fslmaths {segmentation_path} -sub {lesion_mask_path} -bin {output_path}",
        "Subtract lesion from segmentation (NR post-hoc correction)"
    )


# =============================================================================
# BIANCA SEGMENTATION
# =============================================================================

def create_bianca_masterfile(
    flair_path: str,
    t1_path: str,
    mni_matrix_path: str,
    output_dir: str,
    subject_id: str,
    condition: str = "nr"
) -> str:
    """
    Create BIANCA master file for single subject processing.
    
    The master file specifies input images and transformation matrix
    for BIANCA's k-NN classifier.
    
    Args:
        flair_path: Path to FLAIR brain image
        t1_path: Path to T1 brain image
        mni_matrix_path: Path to FLAIR-to-MNI transformation matrix
        output_dir: Directory for output files
        subject_id: Subject identifier
        condition: Processing condition ('nr' or 'r')
        
    Returns:
        str: Path to created master file
        
    Paper Reference:
        Section 2 Methods - BIANCA requires input as FLAIR images with
        optional T1-MPRAGE for improved performance, using voxel intensities
        from both modalities as features for the k-NN classifier.
    """
    # Placeholder for empty mask (BIANCA requirement)
    empty_file = os.path.join(output_dir, f"{subject_id}_empty.nii.gz")
    
    # Master file content: FLAIR T1 matrix empty_mask
    content = f"{flair_path} {t1_path} {mni_matrix_path} {empty_file}"
    
    master_file_path = os.path.join(
        output_dir,
        f"{subject_id}_masterfile_{condition}.txt"
    )
    
    with open(master_file_path, 'w') as f:
        f.write(content)
    
    return master_file_path


def run_bianca_segmentation(
    master_file_path: str,
    classifier_path: str,
    output_path: str,
    verbose: bool = True
) -> bool:
    """
    Run BIANCA segmentation using pre-trained classifier.
    
    BIANCA (Brain Intensity AbNormality Classification Algorithm) uses
    a k-nearest neighbor (kNN) classification approach with FLAIR and
    T1 intensity features.
    
    Args:
        master_file_path: Path to BIANCA master file
        classifier_path: Path to trained BIANCA model
        output_path: Path for output lesion probability map (LPM)
        verbose: Whether to run in verbose mode
        
    Returns:
        bool: True if segmentation succeeded
        
    Paper Reference:
        Section 2 - "BIANCA is a fully supervised automated WMH segmentation 
        machine learning-based method and relies on a k-nearest neighbour 
        (kNN) classification approach with flexible features"
        
        --featuresubset=1,2 uses FLAIR and T1 voxel intensities as features
        --brainmaskfeaturenum=1 indicates FLAIR provides brain mask
        --matfeaturenum=3 indicates transformation matrix position
    """
    bianca_cmd = [
        "bianca",
        f"--singlefile={master_file_path}",
        "--brainmaskfeaturenum=1",
        "--matfeaturenum=3",
        "--featuresubset=1,2",
        f"--loadclassifierdata={classifier_path}",
        "--querysubjectnum=1",
        "-o", output_path,
    ]
    
    if verbose:
        bianca_cmd.append("-v")
    
    try:
        subprocess.run(bianca_cmd, check=True, capture_output=True)
        logging.info(f"✓ BIANCA segmentation: {Path(output_path).name}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ BIANCA failed: {e.stderr}")
        return False


def apply_white_matter_mask(
    lpm_path: str,
    wm_mask_path: str,
    output_path: str
) -> bool:
    """
    Apply white matter mask to BIANCA output to remove false positives.
    
    The WM mask eliminates extracerebral false positives by restricting
    the segmentation to white matter regions only.
    
    Args:
        lpm_path: Path to BIANCA lesion probability map
        wm_mask_path: Path to white matter mask
        output_path: Path for masked output
        
    Returns:
        bool: True if masking succeeded
        
    Paper Reference:
        Section 2.5 - "A white matter mask, which combines tissue segmentation 
        with nonlinear registration fields to MNI space, applied during 
        post-processing to eliminate extracerebral false positives"
    """
    return run_fsl_command(
        f"fslmaths {lpm_path} -mul {wm_mask_path} {output_path}",
        "Apply white matter mask"
    )


def threshold_probability_map(
    lpm_path: str,
    output_path: str,
    threshold: float = 0.85
) -> bool:
    """
    Apply probability threshold to BIANCA output.
    
    The 0.85 threshold was validated as optimal through systematic
    analysis across the full range of thresholds (0.0-1.0).
    
    Args:
        lpm_path: Path to lesion probability map
        output_path: Path for thresholded binary output
        threshold: Probability threshold (default: 0.85)
        
    Returns:
        bool: True if thresholding succeeded
        
    Paper Reference:
        Section 2.7 - "A systematic threshold analysis across the full range 
        of thresholds (0.0-1.0) was performed... We compared three literature-based 
        segmentation configurations: BIANCA thresholded at 0.9 (B_0.9) as proposed 
        by the developers (Griffanti et al., 2016), BIANCA thresholded at 0.85 
        (B_0.85) as suggested in a different study (Ferris et al., 2023)"
        
        Figure 2 - "At the 0.85 threshold, the algorithm achieved balanced 
        performance (precision=0.64, sensitivity=0.67)"
    """
    return run_fsl_command(
        f"fslmaths {lpm_path} -thr {threshold} -bin {output_path}",
        f"Apply threshold {threshold}"
    )


def extract_regional_wmh(
    wmh_path: str,
    region_mask_path: str,
    output_path: str
) -> bool:
    """
    Extract WMH within a specific region (deep or periventricular).
    
    Args:
        wmh_path: Path to total WMH segmentation
        region_mask_path: Path to regional mask
        output_path: Path for regional WMH output
        
    Returns:
        bool: True if extraction succeeded
        
    Paper Reference:
        Table 1 - Reports separate metrics for "Total", "Peri", 
        and "deep" WMH regions
    """
    return run_fsl_command(
        f"fslmaths {wmh_path} -mul {region_mask_path} {output_path}",
        f"Extract regional WMH: {Path(output_path).stem}"
    )


# =============================================================================
# NR (NON-REMOVED) PIPELINE
# =============================================================================

def run_nr_pipeline(
    flair_brain_path: str,
    t1_path: str,
    mni_matrix_path: str,
    lesion_path: str,
    wm_mask_path: str,
    output_dir: str,
    subject_id: str,
    classifier_path: str,
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, str]:
    """
    Run complete NR (non-removed) segmentation pipeline.
    
    In the NR condition, BIANCA processes the original FLAIR image
    and lesion voxels are subtracted post-hoc from the result.
    
    Args:
        flair_brain_path: Path to original FLAIR brain image
        t1_path: Path to T1 brain image
        mni_matrix_path: Path to FLAIR-to-MNI transformation matrix
        lesion_path: Path to lesion mask
        wm_mask_path: Path to white matter mask
        output_dir: Directory for NR outputs
        subject_id: Subject identifier
        classifier_path: Path to trained BIANCA model
        threshold: Probability threshold (default: 0.85)
        
    Returns:
        Dict with paths to all NR output files
        
    Paper Reference:
        Section 2.4 - "NR (not removed): BIANCA segmentation was performed 
        on original FLAIR images, with lesion voxels subtracted post-hoc 
        from the resulting WMH maps"
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}
    
    threshold_str = str(int(threshold * 100))
    
    # Copy FLAIR to output directory
    flair_dest = os.path.join(output_dir, f"{subject_id}_FLAIR_brain.nii.gz")
    fsl_copy(flair_brain_path, flair_dest)
    outputs['flair'] = flair_dest
    
    # Create master file
    master_file = create_bianca_masterfile(
        flair_dest, t1_path, mni_matrix_path,
        output_dir, subject_id, "nr"
    )
    outputs['masterfile'] = master_file
    
    # Run BIANCA
    lpm_path = os.path.join(output_dir, f"{subject_id}_BIANCA_LPM.nii.gz")
    if not os.path.isfile(lpm_path):
        if not run_bianca_segmentation(master_file, classifier_path, lpm_path):
            return outputs
    outputs['lpm'] = lpm_path
    
    # Apply white matter mask
    wm_corrected = os.path.join(output_dir, f"{subject_id}_BIANCA_LPM_wm.nii.gz")
    apply_white_matter_mask(lpm_path, wm_mask_path, wm_corrected)
    outputs['wm_corrected'] = wm_corrected
    
    # Apply threshold
    thresh_path = os.path.join(
        output_dir,
        f"{subject_id}_BIANCA_thresh_{threshold_str}.nii.gz"
    )
    threshold_probability_map(wm_corrected, thresh_path, threshold)
    outputs['thresholded'] = thresh_path
    
    # Subtract lesion post-hoc (NR strategy)
    final_path = os.path.join(
        output_dir,
        f"{subject_id}_BIANCA_lesion_corrected.nii.gz"
    )
    subtract_lesion_from_segmentation(thresh_path, lesion_path, final_path)
    outputs['final'] = final_path
    
    return outputs


# =============================================================================
# R (REMOVED) PIPELINE
# =============================================================================

def run_r_pipeline(
    flair_brain_path: str,
    t1_path: str,
    mni_matrix_path: str,
    lesion_path: str,
    wm_mask_path: str,
    output_dir: str,
    subject_id: str,
    classifier_path: str,
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, str]:
    """
    Run complete R (removed) segmentation pipeline.
    
    In the R condition, lesions are masked from the FLAIR image
    before BIANCA processing.
    
    Args:
        flair_brain_path: Path to original FLAIR brain image
        t1_path: Path to T1 brain image
        mni_matrix_path: Path to FLAIR-to-MNI transformation matrix
        lesion_path: Path to lesion mask
        wm_mask_path: Path to white matter mask
        output_dir: Directory for R outputs
        subject_id: Subject identifier
        classifier_path: Path to trained BIANCA model
        threshold: Probability threshold (default: 0.85)
        
    Returns:
        Dict with paths to all R output files
        
    Paper Reference:
        Section 2.4 - "R (removed): lesions were masked from FLAIR images 
        prior to BIANCA training and segmentation"
        
        Section 4.1 - "Removing lesions prior to segmentation improved 
        overall accuracy, yielding systematically higher WMH volume 
        estimates that scaled with lesion size"
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}
    
    threshold_str = str(int(threshold * 100))
    
    # Remove lesions from FLAIR (R strategy)
    flair_removed = os.path.join(output_dir, f"{subject_id}_FLAIR_removed.nii.gz")
    remove_lesion_from_flair(flair_brain_path, lesion_path, flair_removed)
    outputs['flair_removed'] = flair_removed
    
    # Create master file
    master_file = create_bianca_masterfile(
        flair_removed, t1_path, mni_matrix_path,
        output_dir, subject_id, "r"
    )
    outputs['masterfile'] = master_file
    
    # Run BIANCA
    lpm_path = os.path.join(output_dir, f"{subject_id}_BIANCA_LPM.nii.gz")
    if not os.path.isfile(lpm_path):
        if not run_bianca_segmentation(master_file, classifier_path, lpm_path):
            return outputs
    outputs['lpm'] = lpm_path
    
    # Apply white matter mask
    wm_corrected = os.path.join(output_dir, f"{subject_id}_BIANCA_LPM_wm.nii.gz")
    apply_white_matter_mask(lpm_path, wm_mask_path, wm_corrected)
    outputs['wm_corrected'] = wm_corrected
    
    # Apply threshold
    final_path = os.path.join(
        output_dir,
        f"{subject_id}_BIANCA_thresh_{threshold_str}.nii.gz"
    )
    threshold_probability_map(wm_corrected, final_path, threshold)
    outputs['thresholded'] = final_path
    outputs['final'] = final_path  # No post-hoc subtraction needed
    
    return outputs


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_robustness_metrics(
    nr_outputs: Dict[str, str],
    r_outputs: Dict[str, str],
    manual_mask_path: str,
    lesion_path: str,
    deep_mask_path: str,
    peri_mask_path: str,
    combined_mask_path: str,
    subject_metadata: Dict[str, Any],
    output_dir: str,
    subject_id: str
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics comparing R vs NR segmentation.
    
    Computes Dice coefficients, volume differences, and regional metrics
    for the robustness assessment.
    
    Args:
        nr_outputs: Dict with NR output paths
        r_outputs: Dict with R output paths
        manual_mask_path: Path to manual reference mask
        lesion_path: Path to lesion mask
        deep_mask_path: Path to deep WM mask
        peri_mask_path: Path to periventricular mask
        combined_mask_path: Path to combined WM mask
        subject_metadata: Dict with subject clinical information
        output_dir: Output directory for regional files
        subject_id: Subject identifier
        
    Returns:
        Dict with all computed metrics
        
    Paper Reference:
        Section 3.2.2 - "Bland-Altman analysis demonstrated systematic 
        directional differences between R and NR segmentation approaches"
        
        Table 1 - "Comparison of WMH segmentation volumes lesions 
        not removed (NR) and removed (R)"
    """
    metrics = dict(subject_metadata)
    
    # --- Extract regional segmentations ---
    # NR regional
    nr_peri = os.path.join(output_dir, f"{subject_id}_nr_perWMH.nii.gz")
    nr_deep = os.path.join(output_dir, f"{subject_id}_nr_deepWMH.nii.gz")
    nr_whole = os.path.join(output_dir, f"{subject_id}_nr_wholeWMH.nii.gz")
    
    extract_regional_wmh(nr_outputs.get('final', ''), peri_mask_path, nr_peri)
    extract_regional_wmh(nr_outputs.get('final', ''), deep_mask_path, nr_deep)
    extract_regional_wmh(nr_outputs.get('final', ''), combined_mask_path, nr_whole)
    
    # R regional
    r_peri = os.path.join(output_dir, f"{subject_id}_r_perWMH.nii.gz")
    r_deep = os.path.join(output_dir, f"{subject_id}_r_deepWMH.nii.gz")
    r_whole = os.path.join(output_dir, f"{subject_id}_r_wholeWMH.nii.gz")
    
    extract_regional_wmh(r_outputs.get('final', ''), peri_mask_path, r_peri)
    extract_regional_wmh(r_outputs.get('final', ''), deep_mask_path, r_deep)
    extract_regional_wmh(r_outputs.get('final', ''), combined_mask_path, r_whole)
    
    # Manual mask regional
    manual_peri = os.path.join(output_dir, f"{subject_id}_manual_perWMH.nii.gz")
    manual_deep = os.path.join(output_dir, f"{subject_id}_manual_deepWMH.nii.gz")
    
    extract_regional_wmh(manual_mask_path, peri_mask_path, manual_peri)
    extract_regional_wmh(manual_mask_path, deep_mask_path, manual_deep)
    
    # --- Calculate Dice coefficients ---
    # NR Dice
    metrics['dice_whole_nr'] = calculate_dice_coefficient(nr_whole, manual_mask_path)['dice_score']
    metrics['dice_peri_nr'] = calculate_dice_coefficient(nr_peri, manual_peri)['dice_score']
    metrics['dice_deep_nr'] = calculate_dice_coefficient(nr_deep, manual_deep)['dice_score']
    
    # R Dice
    metrics['dice_whole_r'] = calculate_dice_coefficient(r_whole, manual_mask_path)['dice_score']
    metrics['dice_peri_r'] = calculate_dice_coefficient(r_peri, manual_peri)['dice_score']
    metrics['dice_deep_r'] = calculate_dice_coefficient(r_deep, manual_deep)['dice_score']
    
    # --- Calculate volumes (mL) ---
    # NR volumes
    _, _, metrics['vol_whole_nr'] = get_volume(nr_whole)
    _, _, metrics['vol_peri_nr'] = get_volume(nr_peri)
    _, _, metrics['vol_deep_nr'] = get_volume(nr_deep)
    
    # R volumes
    _, _, metrics['vol_whole_r'] = get_volume(r_whole)
    _, _, metrics['vol_peri_r'] = get_volume(r_peri)
    _, _, metrics['vol_deep_r'] = get_volume(r_deep)
    
    # Manual and lesion volumes
    _, _, metrics['vol_manual'] = get_volume(manual_mask_path)
    _, _, metrics['vol_lesion'] = get_volume(lesion_path)
    
    # --- Volume differences (R - NR) ---
    # Paper: "R segmentation consistently produced larger WMH volumes"
    metrics['vol_diff_whole'] = metrics['vol_whole_r'] - metrics['vol_whole_nr']
    metrics['vol_diff_peri'] = metrics['vol_peri_r'] - metrics['vol_peri_nr']
    metrics['vol_diff_deep'] = metrics['vol_deep_r'] - metrics['vol_deep_nr']
    
    # Dice differences
    metrics['dice_diff_whole'] = metrics['dice_whole_r'] - metrics['dice_whole_nr']
    metrics['dice_diff_peri'] = metrics['dice_peri_r'] - metrics['dice_peri_nr']
    metrics['dice_diff_deep'] = metrics['dice_deep_r'] - metrics['dice_deep_nr']
    
    # Store output paths
    metrics['path_nr_whole'] = nr_whole
    metrics['path_r_whole'] = r_whole
    
    return metrics


# =============================================================================
# MAIN ROBUSTNESS EVALUATION PIPELINE
# =============================================================================

def process_single_subject(
    subject_id: str,
    subject_data: Dict[str, Any],
    clinical_data: Optional[Dict[str, Any]],
    output_base_dir: str,
    classifier_path: str,
    standard_space_path: str,
    threshold: float = DEFAULT_THRESHOLD
) -> Optional[Dict[str, Any]]:
    """
    Process a single subject through both R and NR pipelines.
    
    Args:
        subject_id: Subject identifier
        subject_data: Dict with subject file paths
        clinical_data: Optional dict with clinical information
        output_base_dir: Base output directory
        classifier_path: Path to BIANCA classifier
        standard_space_path: Path to MNI template
        threshold: Probability threshold
        
    Returns:
        Dict with all metrics or None if processing failed
        
    Paper Reference:
        Section 2.4 - "This procedure was executed twice: first with lesions 
        retained in the FLAIR images (NR), and then with lesions masked 
        prior to processing (R)"
    """
    logging.info(f"Processing {subject_id}")
    
    try:
        # Create output directories
        subject_dir = os.path.join(output_base_dir, subject_id)
        nr_dir = os.path.join(subject_dir, "non_removed")
        r_dir = os.path.join(subject_dir, "removed")
        
        os.makedirs(subject_dir, exist_ok=True)
        
        # Get required paths
        flair_brain = subject_data.get('flair_brain')
        t1_brain = subject_data.get('t1_brain') or subject_data.get('t1_path')
        lesion_path = subject_data.get('lesion_path') or subject_data.get('flair_infarct_path')
        manual_mask = subject_data.get('manualmask_path') or subject_data.get('manual_mask_null')
        wm_mask = subject_data.get('WMmask')
        ventdist = subject_data.get('ventdistmap')
        
        # Copy auxiliary files
        wm_dest = os.path.join(subject_dir, f"{subject_id}_WMmask.nii.gz")
        vent_dest = os.path.join(subject_dir, f"{subject_id}_ventdistmap.nii.gz")
        t1_dest = os.path.join(subject_dir, f"{subject_id}_T1.nii.gz")
        lesion_dest = os.path.join(subject_dir, f"{subject_id}_lesion.nii.gz")
        
        fsl_copy(wm_mask, wm_dest)
        fsl_copy(ventdist, vent_dest)
        fsl_copy(t1_brain, t1_dest)
        fsl_copy(lesion_path, lesion_dest)
        
        # Create regional masks
        deep_mask = os.path.join(subject_dir, f"{subject_id}_deep_mask.nii.gz")
        peri_mask = os.path.join(subject_dir, f"{subject_id}_peri_mask.nii.gz")
        combined_mask = os.path.join(subject_dir, f"{subject_id}_combined_mask.nii.gz")
        
        create_deep_wmh_mask(vent_dest, deep_mask)
        create_periventricular_wmh_mask(vent_dest, peri_mask)
        create_combined_brain_mask(deep_mask, peri_mask, combined_mask)
        
        # Register FLAIR to MNI (if not already done)
        mni_mat = os.path.join(subject_dir, f"{subject_id}_flair2mni.mat")
        mni_out = os.path.join(subject_dir, f"{subject_id}_flair_mni.nii.gz")
        
        if not os.path.exists(mni_mat):
            flirt_cmd = [
                "flirt",
                "-in", flair_brain,
                "-ref", standard_space_path,
                "-out", mni_out,
                "-omat", mni_mat,
                "-dof", "12"
            ]
            subprocess.run(flirt_cmd, check=True, capture_output=True)
            logging.info(f"  ✓ FLAIR-to-MNI registration")
        
        # Run NR pipeline
        nr_outputs = run_nr_pipeline(
            flair_brain, t1_dest, mni_mat, lesion_dest, wm_dest,
            nr_dir, subject_id, classifier_path, threshold
        )
        
        # Run R pipeline
        r_outputs = run_r_pipeline(
            flair_brain, t1_dest, mni_mat, lesion_dest, wm_dest,
            r_dir, subject_id, classifier_path, threshold
        )
        
        # Build metadata
        metadata = {
            'subject': subject_id,
            'scanner': subject_data.get('scanner', 'unknown'),
        }
        
        if clinical_data:
            metadata['lesion_type'] = LESION_TYPE_MAP.get(
                clinical_data.get('lesion_type'), 'unknown'
            )
            metadata['sex'] = SEX_MAP.get(clinical_data.get('sex'), 'unknown')
            metadata['age'] = clinical_data.get('age')
            metadata['wahlund'] = clinical_data.get('Wahlund')
        
        # Calculate metrics
        metrics = calculate_robustness_metrics(
            nr_outputs, r_outputs,
            manual_mask, lesion_dest,
            deep_mask, peri_mask, combined_mask,
            metadata, subject_dir, subject_id
        )
        
        logging.info(f"  Dice NR: {metrics.get('dice_whole_nr', 0):.3f}, "
                    f"R: {metrics.get('dice_whole_r', 0):.3f}")
        logging.info(f"  Vol diff: {metrics.get('vol_diff_whole', 0):.3f} mL")
        
        return metrics
        
    except Exception as e:
        logging.error(f"  ✗ Failed: {e}")
        return None


def run_robustness_evaluation(
    subject_data_df: pd.DataFrame,
    output_dir: str,
    classifier_path: str,
    standard_space_path: str,
    clinical_data_df: Optional[pd.DataFrame] = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_subjects: Optional[int] = None,
    shuffle: bool = False
) -> pd.DataFrame:
    """
    Run complete robustness evaluation comparing R vs NR preprocessing.
    
    Processes all subjects through both pipelines and computes
    comprehensive metrics for statistical analysis.
    
    Args:
        subject_data_df: DataFrame with subject paths
        output_dir: Base output directory
        classifier_path: Path to BIANCA classifier
        standard_space_path: Path to MNI template
        clinical_data_df: Optional DataFrame with clinical data
        threshold: Probability threshold (default: 0.85)
        max_subjects: Optional limit on subjects to process
        shuffle: Whether to randomize subject order
        
    Returns:
        DataFrame with all metrics for all subjects
        
    Paper Reference:
        Section 2.2 - Multi-phase validation study design
        
        Phase II-A (n=86): "In 86 representative cases with diverse lesion 
        types, we compared R versus NR segmentation accuracy stratified 
        by lesion etiology and scanner"
        
        Phase II-B (n=211): "The second validation phase included the 
        application of the optimized algorithm in a real-world high 
        vascular risk cohort of n=211"
    """
    logging.info("=" * 60)
    logging.info("BIANCA ROBUSTNESS EVALUATION")
    logging.info("Comparing R (removed) vs NR (non-removed) strategies")
    logging.info("=" * 60)
    logging.info(f"Threshold: {threshold}")
    logging.info(f"Output: {output_dir}")
    logging.info("=" * 60)
    
    all_metrics = []
    
    # Get subject list
    subjects = list(subject_data_df['subject_name'])
    
    if shuffle:
        import random
        random.shuffle(subjects)
    
    if max_subjects:
        subjects = subjects[:max_subjects]
    
    logging.info(f"Processing {len(subjects)} subjects")
    
    for idx, subject_id in enumerate(subjects):
        logging.info(f"\n[{idx+1}/{len(subjects)}] {subject_id}")
        
        # Get subject data
        subj_row = subject_data_df[
            subject_data_df['subject_name'] == subject_id
        ].iloc[0]
        subject_data = subj_row.to_dict()
        
        # Get clinical data if available
        clinical_data = None
        if clinical_data_df is not None:
            original_name = subject_data.get('original_subject_name_s')
            if original_name:
                matches = clinical_data_df[
                    clinical_data_df['BLV_ID'] == original_name
                ]
                if len(matches) > 0:
                    clinical_data = matches.iloc[0].to_dict()
        
        # Process subject
        metrics = process_single_subject(
            subject_id, subject_data, clinical_data,
            output_dir, classifier_path, standard_space_path,
            threshold
        )
        
        if metrics:
            all_metrics.append(metrics)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_metrics)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(output_dir, f"robustness_results_{timestamp}.xlsx")
    results_df.to_excel(excel_path, index=False)
    
    # Print summary
    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Subjects processed: {len(results_df)}")
    
    if len(results_df) > 0:
        logging.info(f"Mean Dice NR: {results_df['dice_whole_nr'].mean():.3f} "
                    f"± {results_df['dice_whole_nr'].std():.3f}")
        logging.info(f"Mean Dice R: {results_df['dice_whole_r'].mean():.3f} "
                    f"± {results_df['dice_whole_r'].std():.3f}")
        logging.info(f"Mean vol diff (R-NR): {results_df['vol_diff_whole'].mean():.3f} mL")
    
    logging.info(f"\nResults saved: {excel_path}")
    
    return results_df


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """
    Command line interface for robustness evaluation.
    
    Usage:
        python robustness_evaluation.py --config .env
        python robustness_evaluation.py --subjects sub-001 sub-002 -n 10
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BIANCA Robustness Evaluation: R vs NR Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Paper Reference:
    "Robustness and Error Susceptibility of BIANCA for White Matter 
    Hyperintensity Segmentation: The Roles of Lesion Volume and 
    Scanner Heterogeneity"
    
    Section 2.4 - Lesion removal assessment
    Section 3.2-3.3 - Robustness results

Example:
    python robustness_evaluation.py --config .env --threshold 0.85
        """
    )
    
    parser.add_argument('--config', '-c', default='.env',
                       help='Path to .env configuration file')
    parser.add_argument('--threshold', '-t', type=float, default=0.85,
                       help='BIANCA threshold (default: 0.85)')
    parser.add_argument('--max-subjects', '-n', type=int,
                       help='Maximum number of subjects to process')
    parser.add_argument('--shuffle', action='store_true',
                       help='Randomize subject processing order')
    parser.add_argument('--subjects', nargs='+',
                       help='Specific subject IDs to process')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Load environment if config file exists
    if os.path.exists(args.config):
        from dotenv import load_dotenv
        load_dotenv(args.config)
    
    # Load configuration
    try:
        config = load_robustness_config()
    except EnvironmentError as e:
        logging.error(str(e))
        return 1
    
    # Load subject data
    subject_df = pd.read_excel(config['subject_data_path'])
    
    # Filter subjects if specified
    if args.subjects:
        subject_df = subject_df[
            subject_df['subject_name'].isin(args.subjects)
        ]
    
    # Load clinical data if available
    clinical_df = None
    if config.get('clinical_data_path') and os.path.exists(config['clinical_data_path']):
        clinical_df = pd.read_excel(config['clinical_data_path'])
    
    # Run evaluation
    results = run_robustness_evaluation(
        subject_data_df=subject_df,
        output_dir=config['output_dir'],
        classifier_path=config['bianca_model'],
        standard_space_path=config['standard_space_t1'],
        clinical_data_df=clinical_df,
        threshold=args.threshold,
        max_subjects=args.max_subjects,
        shuffle=args.shuffle
    )
    
    print(f"\n✓ Evaluation complete. {len(results)} subjects processed.")
    return 0


if __name__ == "__main__":
    exit(main())