#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:57:28 2025

@author: temuuleu
"""

import os
import subprocess
import re
import logging
from pathlib import Path
from typing import List, Union

import ants  # For ANTs bias field correction
import nibabel as nib  # For handling NIfTI images

from nipype.interfaces.fsl import BET  # For brain extraction using FSL's BET
from nipype import Node  # To run BET as a node


def get_files_from_dir(path, endings=[".nii", ".nii.gz"], session_basename=False, max_depth=None):
    """
    Recursively fetches all files from a given directory that have specific endings 
    and contain a specific basename, up to a specified directory depth.

    Parameters:
    - path (str): The directory path from where files need to be fetched.
    - session_basename (str or bool): The string that the filename should contain.
    - endings (list of str): A list of file extensions to be matched.
    - max_depth (int or None): The maximum directory depth to search. None means no limit.

    Returns:
    - list of str: A list containing paths to all files matching the given conditions.
    """

    if not os.path.isdir(path):
        print(f"Provided path: {path} is not a directory.")
        return []

    base_depth = path.rstrip(os.sep).count(os.sep)

    def is_within_depth(root, max_depth):
        if max_depth is None:
            return True
        current_depth = root.rstrip(os.sep).count(os.sep)
        return (current_depth - base_depth) < max_depth

    matching_files = [
        os.path.join(root, filename)
        for root, dirs, files in os.walk(path)
        if is_within_depth(root, max_depth)
        for filename in files
        if any(filename.endswith(ending) for ending in endings) and 
        (not session_basename or session_basename in os.path.basename(filename))
    ]

    return matching_files


def get_subdirectories(path, index=False, basename=False, only_num=True, verbose=False):
    """
    Retrieves subdirectories within a given directory.

    Args:
        path (str): Path to the directory.
        index (bool): If True, returns a list of tuples with (index, path).
        basename (bool): If True, returns only the basenames of subdirectories.
        only_num (bool): If True, includes only subdirectories that have at least one digit in their name.
        verbose (bool): If True, prints additional information.

    Returns:
        list: List of subdirectory paths or tuples.
    """
    # Check if path exists and is a directory
    if not os.path.isdir(path):
        print(f"Provided path: {path} does not exist or is not a directory.")
        return []

    # Get the list of all subdirectories
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    # Filter subdirectories that contain a number if only_num is True
    if only_num:
        subdirs = [d for d in subdirs if re.search(r'\d', d)]
    
    # Return basenames or full paths
    if basename:
        data_paths_list = subdirs
    else:
        data_paths_list = [os.path.join(path, d) for d in subdirs]
    
    # Sort the list by basename
    data_paths_list.sort(key=os.path.basename)
    
    if verbose:
        print(f"path: {path} exists")
        print(f"number of paths: {len(data_paths_list)}")
    
    # If index flag is True, return a tuple (index, path)
    if index:
        return [(i, p) for i, p in enumerate(data_paths_list)]
    
    return data_paths_list


def find_elements(file_list, include=[], exclude=[]):
    """
    Filters the list of files based on inclusion and exclusion criteria.
    
    Args:
        file_list (list): List of file paths to be filtered.
        include (list): List of substrings to include.
        exclude (list): List of substrings to exclude.
    
    Returns:
        list: Filtered list of files.
    """
    filtered_files = []
    for file in file_list:
        basename = os.path.basename(file).lower()  # Convert to lowercase
        # Check if at least one of the include strings is in the basename,
        # or pass if include is empty.
        include_condition = any(inc.lower() in basename for inc in include) if include else True
        
        # Check that none of the exclude strings is in the basename.
        exclude_condition = not any(exc.lower() in basename for exc in exclude)
        
        if include_condition and exclude_condition:
            filtered_files.append(file)
    
    return filtered_files



def get_subdirectories(path, index=False, basename=False, only_num=True, verbose=False):
    """
    Retrieves subdirectories within a given directory.

    Args:
        path (str): Path to the directory.
        index (bool): If True, returns a list of tuples with (index, path).
        basename (bool): If True, returns only the basenames of subdirectories.
        only_num (bool): If True, includes only subdirectories that have at least one digit in their name.
        verbose (bool): If True, prints additional information.

    Returns:
        list: List of subdirectory paths or tuples.
    """
    # Check if path exists and is a directory
    if not os.path.isdir(path):
        print(f"Provided path: {path} does not exist or is not a directory.")
        return []

    # Get the list of all subdirectories
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    # Filter subdirectories that contain a number if only_num is True
    if only_num:
        subdirs = [d for d in subdirs if re.search(r'\d', d)]
    
    # Return basenames or full paths
    if basename:
        data_paths_list = subdirs
    else:
        data_paths_list = [os.path.join(path, d) for d in subdirs]
    
    # Sort the list by basename
    data_paths_list.sort(key=os.path.basename)
    
    if verbose:
        print(f"path: {path} exists")
        print(f"number of paths: {len(data_paths_list)}")
    
    # If index flag is True, return a tuple (index, path)
    if index:
        return [(i, p) for i, p in enumerate(data_paths_list)]
    
    return data_paths_list


def run_fsl_anat_preprocessing(derivatives_t1_path: str, t: str = "1", verbose: bool = True) -> str:
   """
   Process T1 anatomical image using FSL's fsl_anat.
   
   Args:
       derivatives_t1_path: Path to input T1 image
       t: Type parameter for fsl_anat (default "1")
       verbose: Whether to run in verbose mode (default True)
   
   Returns:
       str: Path to the created fsl_anat directory
       
   Raises:
       FileNotFoundError: If input path doesn't exist
       RuntimeError: If fsl_anat processing fails
   """
   if not os.path.exists(derivatives_t1_path):
       raise FileNotFoundError(f"Input file not found: {derivatives_t1_path}")
       
   # Get directory from input path
   deriv_subject_dir = os.path.dirname(derivatives_t1_path)
   
   # Create FSL anat directory
   subject_fsl_anat_dir = os.path.join(deriv_subject_dir, "fsl_anat.anat")
   os.makedirs(subject_fsl_anat_dir, exist_ok=True)
   
   # Define destination path and copy using nibabel
   dest_image = os.path.join(subject_fsl_anat_dir, "T1.nii.gz")
   img = nib.load(derivatives_t1_path)
   nib.save(img, dest_image)
   
   # Prepare and run fsl_anat command
   fsl_anat_cmd = [
       "fsl_anat", 
       "-d", subject_fsl_anat_dir,
       "-t", t,
       "--clobber",
       "--nocrop"
   ]
   
   if verbose:
       fsl_anat_cmd.append("-v")
       
   try:
       subprocess.run(fsl_anat_cmd, check=True, capture_output=True, text=True)
   except subprocess.CalledProcessError as e:
       raise RuntimeError(f"FSL preprocessing failed with exit code {e.returncode}")
       
   return subject_fsl_anat_dir


def apply_bias_correction(input_path: Path, output_path: Path) -> Path:
    """
    Perform N4 Bias Field Correction using ANTsPy.

    Parameters:
    - input_path: Path, path to the input NIfTI image.
    - output_path: Path, path to save the bias-corrected image.

    Returns:
    - Path: Path to the saved corrected image.
    """
    
    if not os.path.isfile(output_path):
        try:
            # Load the image with ANTsPy
            image = ants.image_read(str(input_path))
            logging.info(f"Loaded image for bias correction: {input_path}")
    
            # Apply N4 Bias Field Correction
            corrected_image = ants.n4_bias_field_correction(image)
            logging.info("Applied N4 Bias Field Correction.")
    
            # Save the corrected image
            ants.image_write(corrected_image, str(output_path))
            logging.info(f"Bias correction completed and saved to {output_path}")
    
            return output_path
        except AttributeError as ae:
            logging.error(f"AttributeError in bias correction: {ae}")
            raise RuntimeError(f"AttributeError in bias correction: {ae}") from ae
        except Exception as e:
            logging.error(f"Error in bias correction: {e}")
            raise RuntimeError(f"Error in bias correction: {e}") from e





def fsl_bet(input_path: Union[Path, str], output_path: Union[Path, str], frac: float = 0.5, force: bool = False) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if output_path.exists() and not force:
        logging.info(f"Brain-extracted file already exists: {output_path}")
        return output_path
    
    if force and output_path.exists():
        logging.info(f"Removing existing file due to force=True: {output_path}")
        output_path.unlink()
    
    try:
        skullstrip = Node(BET(mask=True, frac=frac), name="skullstrip")
        skullstrip.inputs.in_file = str(input_path)
        skullstrip.inputs.out_file = str(output_path)
        skullstrip.run()
        logging.info(f"Brain extraction completed and saved to {output_path}")
    except Exception as e:
        logging.error(f"Error during brain extraction: {e}")
        raise RuntimeError(f"Error during brain extraction: {e}") from e
    
    return output_path



