#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Module for BIANCA WMH Segmentation Analysis

This module implements the preprocessing pipeline for the study:
"Robustness and Error Susceptibility of BIANCA for White Matter 
Hyperintensity Segmentation: The Roles of Lesion Volume and Scanner Heterogeneity"

Author: Uchralt Temuulen
Affiliation: Center for Stroke Research Berlin, Charité-Universitätsmedizin Berlin

Paper Reference:
    - Section 2.5: Preprocessing pipeline
    - Section 2.6: Lesion delineation and removal

The preprocessing pipeline includes:
    1. Brain extraction using HD-BET (Isensee et al., 2019)
    2. N4 bias field correction (Tustison et al., 2010)
    3. Registration to MNI-152 template space using FSL FLIRT
    4. White matter mask generation using FSL tools
    5. Ventricle distance map creation for LOCATE module

Environment Variables Required:
    - STANDARD_SPACE_T1: Path to MNI-152 T1 template
    - DATA_SET: Path to input dataset directory
    - BIANCA_MODEL: Path to trained BIANCA classifier model
    - THRESHHOLD_BIANCA: Probability threshold for WMH segmentation (default: 0.85)
    - FSLDIR: Path to FSL installation

Dependencies:
    - FSL 6.0+ (https://fsl.fmrib.ox.ac.uk/fsl/)
    - HD-BET (https://github.com/MIC-DKFZ/HD-BET)
    - nibabel, numpy, scipy
"""

import os
import subprocess
import warnings
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import nibabel as nib
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config_from_env() -> dict:
    """
    Load configuration from environment variables.
    
    This function retrieves all necessary paths and parameters from environment
    variables, ensuring reproducibility across different computing environments.
    
    Returns:
        dict: Configuration dictionary with all required paths and parameters
        
    Raises:
        EnvironmentError: If required environment variables are not set
        
    Paper Reference:
        Section 2.5 - Standardized preprocessing pipeline requiring
        consistent configuration across all subjects
    """
    config = {
        'standard_space_t1': os.getenv('STANDARD_SPACE_T1'),
        'data_set': os.getenv('DATA_SET'),
        'bianca_model': os.getenv('BIANCA_MODEL'),
        'threshold_bianca': float(os.getenv('THRESHHOLD_BIANCA', '0.85')),
        'fsldir': os.getenv('FSLDIR', '/usr/local/fsl'),
        'max_processing_attempts': int(os.getenv('MAX_ATTEMPTS', '10')),
    }
    
    # Validate required variables
    required_vars = ['standard_space_t1', 'data_set']
    missing = [var for var in required_vars if not config.get(var)]
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please set these in your .env file or environment."
        )
    
    return config


# =============================================================================
# BRAIN EXTRACTION
# =============================================================================

def hd_bet(
    input_path: str,
    output_path: str,
    device: str = "auto",
    disable_tta: bool = False,
    save_mask: bool = False,
    verbose: bool = False,
    force: bool = False,
    overwrite: Optional[bool] = None
) -> bool:
    """
    Perform brain extraction using HD-BET.
    
    HD-BET is a convolutional neural network-based tool for robust skull 
    stripping, as described in Isensee et al. (2019).
    
    Args:
        input_path: Path to input NIfTI image (T1w or FLAIR)
        output_path: Output base path (without .nii.gz extension)
        device: Computing device - "auto", "cuda", or "cpu"
            "auto" detects CUDA availability and falls back to CPU
        disable_tta: Disable test-time augmentation for faster processing
        save_mask: Save binary brain mask alongside extracted brain
        verbose: Enable verbose output for debugging
        force: Overwrite existing output files
        overwrite: Deprecated alias for force parameter
        
    Returns:
        bool: True if brain extraction succeeded
        
    Paper Reference:
        Section 2.5 - "Brain extraction was performed using HD-BET, 
        a convolutional neural network-based tool optimized for robust 
        skull stripping (Isensee et al., 2019)"
        
    Example:
        >>> success = hd_bet("subject_T1w.nii.gz", "subject_T1w_brain")
        >>> if success:
        ...     print("Brain extraction completed successfully")
    """
    # Handle deprecated parameter
    if overwrite is not None:
        warnings.warn(
            "'overwrite' parameter is deprecated, use 'force' instead",
            DeprecationWarning,
            stacklevel=2
        )
        force = overwrite
    
    # Define expected output files
    output_files = [f"{output_path}.nii.gz"]
    if save_mask:
        output_files.append(f"{output_path}_mask.nii.gz")
    
    # Check for existing outputs
    existing_files = [f for f in output_files if os.path.exists(f)]
    
    if existing_files and not force:
        if verbose:
            print(f"Output files already exist (use force=True to overwrite):")
            for f in existing_files:
                print(f"  - {f}")
        return True  # Consider existing files as success
    
    # Remove existing files if force is True
    if force and existing_files:
        for f in existing_files:
            try:
                os.remove(f)
                if verbose:
                    print(f"Removed existing file: {f}")
            except OSError as e:
                print(f"Error removing {f}: {e}")
                return False
    
    # Determine computing device
    final_device = device.lower()
    if final_device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                final_device = "cuda"
                if verbose:
                    print("CUDA available, using GPU acceleration")
            else:
                final_device = "cpu"
                if verbose:
                    print("CUDA not available, falling back to CPU")
        except ImportError:
            final_device = "cpu"
            if verbose:
                print("PyTorch not available, defaulting to CPU")
    
    # Validate device choice
    if final_device not in ("cuda", "cpu"):
        warnings.warn(f"Invalid device '{final_device}', defaulting to 'cpu'")
        final_device = "cpu"
    
    # Build HD-BET command
    cmd = [
        "hd-bet",
        "-i", input_path,
        "-o", output_path,
        "-device", final_device
    ]
    
    if disable_tta:
        cmd.append("--disable_tta")
    if save_mask:
        cmd.append("--save_mask")
    if verbose:
        cmd.append("--verbose")
    
    # Execute brain extraction
    try:
        if verbose:
            print(f"Running HD-BET: {' '.join(cmd)}")
        
        subprocess.run(cmd, check=True, capture_output=not verbose)
        
        if verbose:
            print(f"HD-BET completed. Output: {output_path}.nii.gz")
            if save_mask:
                print(f"Brain mask saved: {output_path}_mask.nii.gz")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"HD-BET failed with error: {e}")
        return False
    except FileNotFoundError:
        print("Error: hd-bet command not found. Please install HD-BET first.")
        print("Installation: pip install hd-bet")
        return False


def run_fsl_anat_preprocessing(
    input_t1_path: str,
    output_dir: Optional[str] = None,
    image_type: str = "1",
    verbose: bool = True
) -> str:
    """
    Process T1 anatomical image using FSL's fsl_anat pipeline.
    
    This function runs the complete FSL anatomical preprocessing pipeline,
    which includes bias correction, brain extraction, tissue segmentation,
    and registration to MNI space.
    
    Args:
        input_t1_path: Path to input T1-weighted image
        output_dir: Output directory (default: same directory as input)
        image_type: FSL anat type parameter ("1" for T1)
        verbose: Enable verbose output
        
    Returns:
        str: Path to created fsl_anat.anat directory containing all outputs
        
    Raises:
        FileNotFoundError: If input path doesn't exist
        RuntimeError: If fsl_anat processing fails
        
    Paper Reference:
        Section 2.5 - FSL tools are used for subsequent processing including
        tissue segmentation and nonlinear registration to MNI space
        
    Outputs Generated:
        - T1_biascorr.nii.gz: Bias-corrected T1 image
        - T1_biascorr_brain.nii.gz: Brain-extracted bias-corrected image
        - T1_biascorr_brain_mask.nii.gz: Binary brain mask
        - T1_fast_pve_*.nii.gz: Tissue probability maps (CSF, GM, WM)
        - MNI_to_T1_nonlin_field.nii.gz: Warp field from MNI to T1 space
    """
    if not os.path.exists(input_t1_path):
        raise FileNotFoundError(f"Input T1 file not found: {input_t1_path}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_t1_path)
    
    # Create FSL anat output directory
    fsl_anat_dir = os.path.join(output_dir, "fsl_anat.anat")
    os.makedirs(fsl_anat_dir, exist_ok=True)
    
    # Copy input image to expected location
    dest_image = os.path.join(fsl_anat_dir, "T1.nii.gz")
    img = nib.load(input_t1_path)
    nib.save(img, dest_image)
    
    # Build fsl_anat command
    fsl_anat_cmd = [
        "fsl_anat",
        "-d", fsl_anat_dir,
        "-t", image_type,
        "--clobber",  # Overwrite existing outputs
        "--nocrop"    # Don't crop FOV
    ]
    
    if verbose:
        fsl_anat_cmd.append("-v")
    
    # Execute fsl_anat
    try:
        if verbose:
            print(f"Running fsl_anat on {input_t1_path}")
            print(f"Output directory: {fsl_anat_dir}")
        
        result = subprocess.run(
            fsl_anat_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        if verbose:
            print("fsl_anat completed successfully")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FSL preprocessing failed with exit code {e.returncode}\n"
            f"stderr: {e.stderr}"
        )
    
    return fsl_anat_dir


# =============================================================================
# BIAS FIELD CORRECTION
# =============================================================================

def apply_bias_correction(
    input_path: str,
    output_path: str,
    mask_path: Optional[str] = None,
    verbose: bool = False
) -> bool:
    """
    Apply N4 bias field correction using FSL's FAST or ANTs N4.
    
    Bias field correction removes intensity inhomogeneities caused by
    RF field variations and coil sensitivity profiles.
    
    Args:
        input_path: Path to input image
        output_path: Path for bias-corrected output
        mask_path: Optional brain mask to restrict correction
        verbose: Enable verbose output
        
    Returns:
        bool: True if correction succeeded
        
    Paper Reference:
        Section 2.5 - "N4 bias field correction to correct for 
        intensity inhomogeneities (Tustison et al., 2010)"
    """
    # Try FSL's FAST for bias correction
    cmd = ["fast", "-B", "-o", output_path.replace('.nii.gz', ''), input_path]
    
    try:
        subprocess.run(cmd, check=True, capture_output=not verbose)
        
        # FAST outputs with _restore suffix
        restore_path = output_path.replace('.nii.gz', '_restore.nii.gz')
        if os.path.exists(restore_path):
            os.rename(restore_path, output_path)
        
        if verbose:
            print(f"Bias correction completed: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Bias correction failed: {e}")
        return False


# =============================================================================
# IMAGE REGISTRATION
# =============================================================================

def register_to_mni(
    input_path: str,
    reference_path: str,
    output_image_path: str,
    output_matrix_path: str,
    dof: int = 12,
    verbose: bool = False
) -> bool:
    """
    Register image to MNI-152 template space using FSL FLIRT.
    
    Performs affine registration (default 12 DOF) from native space
    to standard MNI-152 template space.
    
    Args:
        input_path: Path to input brain-extracted image
        reference_path: Path to MNI-152 template
        output_image_path: Path for registered output image
        output_matrix_path: Path for transformation matrix
        dof: Degrees of freedom (6=rigid, 12=affine)
        verbose: Enable verbose output
        
    Returns:
        bool: True if registration succeeded
        
    Paper Reference:
        Section 2.5 - "Affine normalization of both modalities to the 
        1 mm MNI-152 template space using FSL tools"
    """
    flirt_cmd = [
        "flirt",
        "-in", input_path,
        "-ref", reference_path,
        "-out", output_image_path,
        "-omat", output_matrix_path,
        "-dof", str(dof)
    ]
    
    try:
        if verbose:
            print(f"Running FLIRT registration to MNI space")
            print(f"  Input: {input_path}")
            print(f"  Reference: {reference_path}")
        
        subprocess.run(flirt_cmd, check=True, capture_output=not verbose)
        
        if verbose:
            print(f"Registration completed: {output_image_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"FLIRT registration failed: {e}")
        return False


def apply_transform(
    input_path: str,
    reference_path: str,
    transform_path: str,
    output_path: str,
    interpolation: str = "trilinear",
    verbose: bool = False
) -> bool:
    """
    Apply existing transformation matrix to register an image.
    
    Args:
        input_path: Path to input image
        reference_path: Path to reference space image
        transform_path: Path to transformation matrix
        output_path: Path for transformed output
        interpolation: Interpolation method (trilinear, nearestneighbour, etc.)
        verbose: Enable verbose output
        
    Returns:
        bool: True if transformation succeeded
    """
    flirt_cmd = [
        "flirt",
        "-in", input_path,
        "-ref", reference_path,
        "-applyxfm",
        "-init", transform_path,
        "-interp", interpolation,
        "-out", output_path
    ]
    
    try:
        subprocess.run(flirt_cmd, check=True, capture_output=not verbose)
        
        if verbose:
            print(f"Transformation applied: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Transformation failed: {e}")
        return False


# =============================================================================
# LESION REMOVAL
# =============================================================================

def mask_out_lesion(
    flair_path: str,
    lesion_mask_path: str,
    output_path: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Remove lesion voxels from FLAIR image by setting them to zero.
    
    This implements the "R" (removed) preprocessing strategy where
    lesion voxels are masked from FLAIR images prior to BIANCA 
    training and segmentation.
    
    Args:
        flair_path: Path to input FLAIR image
        lesion_mask_path: Path to binary lesion mask
        output_path: Path for output FLAIR with lesions removed
        logger: Optional logger for status messages
        
    Returns:
        bool: True if lesion removal succeeded
        
    Paper Reference:
        Section 2.4 - "R (removed): lesions were masked from FLAIR images 
        prior to BIANCA training and segmentation"
        
        Section 2.6 - "For lesion removal, the resulting ROIs were binarized 
        and multiplied with the corresponding FLAIR image using fslmaths, 
        effectively replacing lesion voxel intensities with zero values"
    """
    if logger:
        logger.info(f"Removing lesions from FLAIR image")
        logger.info(f"  FLAIR: {flair_path}")
        logger.info(f"  Lesion mask: {lesion_mask_path}")
    
    # Create inverted lesion mask (1 - lesion_mask)
    # Then multiply with FLAIR to zero out lesion voxels
    cmd = [
        "fslmaths", lesion_mask_path,
        "-binv",  # Binary invert: lesion=0, other=1
        "-mul", flair_path,  # Multiply to zero out lesions
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        if logger:
            logger.info(f"Lesion removal completed: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"Lesion removal failed: {e}")
        return False


# =============================================================================
# WHITE MATTER MASK CREATION
# =============================================================================

def create_bianca_mask(
    t1_biascorr_path: str,
    pve0_path: str,
    warp_field_path: str,
    keep_intermediate: bool = True,
    verbose: bool = False
) -> bool:
    """
    Create BIANCA white matter mask using FSL's make_bianca_mask.
    
    This mask combines tissue segmentation with nonlinear registration
    fields to MNI space, used during post-processing to eliminate
    extracerebral false positives.
    
    Args:
        t1_biascorr_path: Path to bias-corrected T1 image
        pve0_path: Path to CSF partial volume estimate from FAST
        warp_field_path: Path to MNI-to-T1 nonlinear warp field
        keep_intermediate: Keep intermediate processing files
        verbose: Enable verbose output
        
    Returns:
        bool: True if mask creation succeeded
        
    Paper Reference:
        Section 2.5 - "A white matter mask, which combines tissue 
        segmentation with nonlinear registration fields to MNI space, 
        applied during post-processing to eliminate extracerebral 
        false positives"
    """
    cmd = [
        "make_bianca_mask",
        t1_biascorr_path,
        pve0_path,
        warp_field_path,
        "1" if keep_intermediate else "0"
    ]
    
    try:
        if verbose:
            print(f"Creating BIANCA mask")
        
        subprocess.run(cmd, check=True, capture_output=not verbose)
        
        if verbose:
            print("BIANCA mask created successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"BIANCA mask creation failed: {e}")
        return False


def create_ventricle_distance_map(
    ventricle_mask_path: str,
    output_path: str,
    verbose: bool = False
) -> bool:
    """
    Create ventricle distance map for LOCATE adaptive thresholding.
    
    The distance map is required for the LOCATE module's adaptive
    thresholding operations, which determine optimal local thresholds
    based on distance from ventricles.
    
    Args:
        ventricle_mask_path: Path to binary ventricle mask
        output_path: Path for output distance map
        verbose: Enable verbose output
        
    Returns:
        bool: True if distance map creation succeeded
        
    Paper Reference:
        Section 2.5 - "A ventricle-distance map derived from the 
        ventricle mask, required for the LOCATE module's adaptive 
        thresholding operations (Sundaresan et al., 2019)"
    """
    cmd = [
        "distancemap",
        "-i", ventricle_mask_path,
        "-o", output_path
    ]
    
    try:
        if verbose:
            print(f"Creating ventricle distance map")
        
        subprocess.run(cmd, check=True, capture_output=not verbose)
        
        if verbose:
            print(f"Distance map created: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Distance map creation failed: {e}")
        return False


# =============================================================================
# VOLUME CALCULATION
# =============================================================================

def get_volume(image_path: str) -> Tuple[float, float, float]:
    """
    Calculate volume from a segmentation image using FSL's fslstats.
    
    Computes the number of non-zero voxels and converts to physical
    volume units (mm³ and mL).
    
    Args:
        image_path: Path to binary or thresholded segmentation image
        
    Returns:
        Tuple containing:
            - nonzero_voxels (float): Number of non-zero voxels
            - volume_mm3 (float): Volume in cubic millimeters
            - volume_ml (float): Volume in milliliters
            
    Raises:
        subprocess.CalledProcessError: If fslstats fails
        
    Paper Reference:
        Section 2.8 - Volume calculations for comparing R vs NR 
        segmentation outputs (Table 1, Table 2)
    """
    cmd = f"fslstats {image_path} -V"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Parse fslstats output: "nonzero_voxels volume_mm3"
        parts = result.stdout.split()
        nonzero_voxels = float(parts[0])
        volume_mm3 = float(parts[1])
        volume_ml = volume_mm3 / 1000.0  # Convert mm³ to mL
        
        return nonzero_voxels, volume_mm3, volume_ml
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error calculating volume for {image_path}: {e.stderr}")
        raise


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def fsl_copy(
    source_path: str,
    dest_path: str,
    force: bool = False
) -> bool:
    """
    Copy NIfTI file using FSL's imcp (preserves header information).
    
    Args:
        source_path: Path to source image
        dest_path: Path for destination copy
        force: Overwrite existing destination file
        
    Returns:
        bool: True if copy succeeded
    """
    if os.path.exists(dest_path) and not force:
        return True
    
    # Remove existing file if force is True
    if os.path.exists(dest_path):
        os.remove(dest_path)
    
    cmd = ["imcp", source_path, dest_path]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        # Fallback to nibabel copy
        try:
            img = nib.load(source_path)
            nib.save(img, dest_path)
            return True
        except Exception as e:
            print(f"Copy failed: {e}")
            return False


def validate_nifti_exists(path: str, description: str = "") -> bool:
    """
    Validate that a NIfTI file exists and is readable.
    
    Args:
        path: Path to NIfTI file
        description: Optional description for error messages
        
    Returns:
        bool: True if file exists and is valid
    """
    if not os.path.exists(path):
        logging.error(f"File not found: {path} ({description})")
        return False
    
    try:
        nib.load(path)
        return True
    except Exception as e:
        logging.error(f"Invalid NIfTI file {path}: {e}")
        return False


def log_file_exists(path: str, description: str = "") -> None:
    """
    Log whether a file exists (for pipeline debugging).
    
    Args:
        path: Path to check
        description: Description of the file for logging
    """
    if os.path.exists(path):
        logging.info(f"✓ Found {description}: {path}")
    else:
        logging.warning(f"✗ Missing {description}: {path}")


def run_command(cmd: str, description: str = "", verbose: bool = False) -> bool:
    """
    Execute a shell command with error handling.
    
    Args:
        cmd: Command string to execute
        description: Description for logging
        verbose: Print command output
        
    Returns:
        bool: True if command succeeded
    """
    try:
        if verbose:
            logging.info(f"Running: {cmd}")
        
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        if verbose and result.stdout:
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed: {e.stderr}")
        return False


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Basic module test
    print("BIANCA Preprocessing Module")
    print("=" * 50)
    print("\nThis module provides preprocessing functions for")
    print("the BIANCA WMH segmentation analysis pipeline.")
    print("\nRequired environment variables:")
    print("  - STANDARD_SPACE_T1: MNI template path")
    print("  - DATA_SET: Input dataset directory")
    print("  - BIANCA_MODEL: Trained classifier path")
    print("  - THRESHHOLD_BIANCA: Probability threshold (default: 0.85)")
    print("\nSee docstrings for detailed function documentation.")