#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 09:17:06 2025

@author: temuuleu
"""

import os
from dotenv import load_dotenv
import subprocess
import random 
import logging
import traceback
import sys
import nibabel as nib
import pandas as pd
import shutil


from librarys.library import (
    get_subdirectories,
    get_files_from_dir,
    find_elements,
    run_fsl_anat_preprocessing,
    apply_bias_correction,
    fsl_bet,
    mask_out_infarct,
    create_brain_mask,
    process_flair_registration,
    match_orientation_and_compare
)

from librarys.lib_plots import (
    create_five_panel_plot,
    plot_flair_mni_image,
    plot_overlay_4cols,
    create_4_panel_plot_affine_aware,
    plot_two_nifti_images_2cols
)
MAX_ATTEMPTS = 10

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("neuroimaging_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_file_exists(file_path, description):
    """Helper to log whether a file exists and provide a descriptive message."""
    if os.path.isfile(file_path):
        logger.info(f"✓ {description} exists: {file_path}")
        return True
    else:
        logger.warning(f"✗ {description} does not exist: {file_path}")
        return False

def run_command(command, description):
    """Run a shell command with proper logging."""
    logger.info(f"Running command: {description}")
    logger.debug(f"Command details: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        logger.info(f"✓ Command succeeded: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Command failed: {description}")
        logger.error(f"Error details: {e.stderr}")
        return False

# Log startup information
logger.info("=" * 80)
logger.info("NEUROIMAGING PIPELINE STARTED")
logger.info("=" * 80)



#i want both if it start with slumr this should run but i

# Load environment variables from .env file
if len(sys.argv) > 1:
    env_file = sys.argv[1]
    load_dotenv(dotenv_path=env_file)
    logger.info(f"Environment loaded from: {env_file}")
else:
    raise ValueError("Please provide an .env file path as a command-line argument.")
    
    

    #so decei here 
    
    # You can hardcode, use a dropdown, or make it interactive
    AVAILABLE_ENV_FILES = {
        "challenge": "envs/challenge.env",
        "belove":    "envs/belove.env",
        "kersten":   "/envs/kersten.env"
    }
    
    # Choose one manually or interactively
    DATASET_CHOICE = "challenge"  # Change this to "belove", etc.
    
    env_file = AVAILABLE_ENV_FILES[DATASET_CHOICE]
    load_dotenv(dotenv_path=env_file)
    logger.info(f"Environment loaded from: {env_file}")
    
    
    





logger.info("Environment variables loaded from .env file")



# Assign paths dynamically
STANDARD_SPACE_T1 = os.getenv("STANDARD_SPACE_T1")
DATA_SET          = os.getenv("DATA_SET")
SHUFFLE_DATA_SET  = os.getenv("SHUFFLE")

# Check if variables are loaded correctly
if STANDARD_SPACE_T1 is None:
    logger.error("STANDARD_SPACE_T1 environment variable is missing. Check your .env file.")
    raise ValueError("STANDARD_SPACE_T1 environment variable is missing. Check your .env file.")
else:
    logger.info(f"STANDARD_SPACE_T1 path: {STANDARD_SPACE_T1}")
    if not os.path.exists(STANDARD_SPACE_T1):
        logger.error(f"STANDARD_SPACE_T1 file does not exist at: {STANDARD_SPACE_T1}")
        raise FileNotFoundError(f"STANDARD_SPACE_T1 file does not exist at: {STANDARD_SPACE_T1}")

if DATA_SET is None:
    logger.error("DATA_SET environment variable is missing. Check your .env file.")
    raise ValueError("DATA_SET environment variable is missing. Check your .env file.")
else:
    logger.info(f"DATA_SET path: {DATA_SET}")
    if not os.path.exists(DATA_SET):
        logger.error(f"DATA_SET directory does not exist at: {DATA_SET}")
        raise FileNotFoundError(f"DATA_SET directory does not exist at: {DATA_SET}")

# Get list of subject directories from the dataset
try:
    DATA_SET_directorys = get_subdirectories(DATA_SET)
    logger.info(f"Found {len(DATA_SET_directorys)} subject directories.")
except Exception as e:
    logger.error(f"Failed to get subdirectories from {DATA_SET}: {str(e)}")
    raise

# Create derivative directories
logger.info("Creating derivative directories...")
derivatives = os.path.join(DATA_SET, "derivatives")

fsl_anat = os.path.join(derivatives, "fsl_anat")
os.makedirs(fsl_anat, exist_ok=True)
logger.info(f"Created fsl_anat directory: {fsl_anat}")


fsl_anat_flair = os.path.join(derivatives, "fsl_anat_flair")
os.makedirs(fsl_anat_flair, exist_ok=True)
logger.info(f"Created fsl_anat directory: {fsl_anat_flair}")


prepare_flair = os.path.join(derivatives, "prepare_flair")
os.makedirs(prepare_flair, exist_ok=True)




logger.info(f"Created prepare_flair directory: {prepare_flair}")

prepare_t1w = os.path.join(derivatives, "prepare_t1w")
os.makedirs(prepare_t1w, exist_ok=True)
logger.info(f"Created prepare_t1w directory: {prepare_t1w}")

prepare_removed = os.path.join(derivatives, "removed")
os.makedirs(prepare_removed, exist_ok=True)

FLAIR_SPACE_IMAGES = os.path.join(derivatives, "FLAIR_SPACE_IMAGES")
os.makedirs(FLAIR_SPACE_IMAGES, exist_ok=True)

NORMAL_SPACE_IMAGES = os.path.join(derivatives, "NORMAL_SPACE_IMAGES")
os.makedirs(NORMAL_SPACE_IMAGES, exist_ok=True)

# --- Optional SHUFFLING of subject list ---
# Read 'SHUFFLE' setting from .env file, default to False
# Accepts values like 'true', 'True', '1' (case-insensitive)
shuffle = os.getenv("SHUFFLE", "false").strip().lower() in ["true", "1"]


shuffle = 1

if shuffle:
    logger.info("Shuffling subject list before processing...")
    random.shuffle(DATA_SET_directorys)
else:
    logger.info("Processing subjects in original order.")
    
    
DATA_LOCATION = []
skipped_subjects = []

# Process each subject
for di, sub_dir in enumerate(DATA_SET_directorys[:]):
    print(di,sub_dir)
    
    infarct_path           = ""
    infarct_bool           = False
    
    FLAIR_mul_path          = ""
    updated_manualmask_path = ""
    
    try:
        logger.info("=" * 60)
        logger.info(f"PROCESSING SUBJECT {di+1}/{len(DATA_SET_directorys[:1])}: {sub_dir}")
        logger.info("=" * 60)
        
        subject = os.path.basename(sub_dir)
        
        if "challenge-29" == subject:
            reason = "Subject ID contains 'sub-5'"
            logger.info(f"Skipping subject {subject}: {reason}")
            skipped_subjects.append((subject, reason))
            continue
                
              
        logger.info(f"Subject ID: {subject}")

        sub_anat_dir = os.path.join(sub_dir, "anat")
        if not os.path.exists(sub_anat_dir):
            logger.error(f"Subject anat directory not found: {sub_anat_dir}")
            
            reason = f"Subject anat directory not found: {sub_anat_dir}"
            logger.error(reason)
            skipped_subjects.append((subject, reason))
            continue
                        


        logger.info(f"Getting files from subject anat directory: {sub_anat_dir}")
        files = get_files_from_dir(sub_anat_dir, endings=[".nii", ".nii.gz"], max_depth=1)
        logger.info(f"Found {len(files)} .nii/.nii.gz files in the subject anat directory")

        try:
            original_t1_path = find_elements(
                files, 
                include=["T1w"], 
                exclude=["mask"]
            )[0]
            logger.info(f"Found T1w image: {original_t1_path}")
        except (IndexError, Exception) as e:
            logger.error(f"Failed to find T1w image for subject {subject}: {str(e)}")
            
            reason = f"Failed to find T1w image: {str(e)}"
            logger.error(f"Skipping subject {subject}: {reason}")
            skipped_subjects.append((subject, reason))
            continue
                        
    
        ############################################################################


        ########################### FSL_ANAT SECTION ###############################
        logger.info("=" * 40)
        logger.info("STARTING FSL_ANAT SECTION")
        logger.info("=" * 40)

        deriv_subject_dir = os.path.join(fsl_anat, subject, "anat")
        os.makedirs(deriv_subject_dir, exist_ok=True)
        logger.info(f"Created subject derivatives directory: {deriv_subject_dir}")

        derivatives_path = os.path.join(deriv_subject_dir, f"{subject}_T1w.nii")
        
        fsl_anat_dir = os.path.join(deriv_subject_dir, "fsl_anat.anat")
        
        # Check if fsl_anat directory exists
        if os.path.exists(fsl_anat_dir):
            logger.info(f"fsl_anat directory exists: {fsl_anat_dir}")
            fsl_anat_dir_files_list = get_files_from_dir(fsl_anat_dir, max_depth=1)
            logger.info(f"Found {len(fsl_anat_dir_files_list)} files in fsl_anat directory")
        else:
            logger.info(f"fsl_anat directory does not exist yet, will be created during processing: {fsl_anat_dir}")
            fsl_anat_dir_files_list = []


        # Find other necessary files - loop up to 5 times if files are not found
        max_attempts = 10
        attempts = 0
        files_found = False
        
        while attempts < max_attempts and not files_found:
            attempts += 1
            logger.info(f"Attempt {attempts}/{max_attempts} to find critical FSL files")
            
            try:
                warp_file_MNI2structural = find_elements(fsl_anat_dir_files_list,
                                                include=["MNI_to_T1_nonlin_field"],
                                                exclude=["nu"])[0]
                
                log_file_exists(warp_file_MNI2structural, "MNI_to_T1_nonlin_field")
                
                T1_fast_pve_0_path = find_elements(fsl_anat_dir_files_list,
                                                include=["T1_fast_pve_0"],
                                                exclude=["nu"])[0]
                
                
                log_file_exists(T1_fast_pve_0_path, "T1_fast_pve_0_path")
                
                T1_fast_pve_1_path = find_elements(fsl_anat_dir_files_list,
                                                include=["T1_fast_pve_1.nii"],
                                                exclude=["nu"])[0]
                
                
                log_file_exists(T1_fast_pve_1_path, "T1_fast_pve_1_path (GM)")
                

      
                
                
                T1_biascorr_brain = find_elements(fsl_anat_dir_files_list,
                                                include=["T1_biascorr_brain"],
                                                exclude=["mask"])[0]
                
                
                
                
                T1_biascorr_path = find_elements(fsl_anat_dir_files_list,
                                                include=["T1_biascorr."],
                                                exclude=["nu"])[0]
                
                
                
                T1_biascorr_brain_mask = find_elements(fsl_anat_dir_files_list,
                                                include=["brain_mask"],
                                                exclude=["nu"])[0]
                
                


                # If we reach here, all files were found
                files_found = True
                logger.info(f"Successfully found all critical FSL files on attempt {attempts}")
                
            except (IndexError, Exception) as e:
                logger.error(f"Attempt {attempts}/{max_attempts}: Failed to find critical FSL files: {str(e)}")
                
                if attempts < max_attempts:
                    logger.info(f"Re-running fsl_anat_preprocessing and trying again...")
                    
                    os.system(f"fslmaths {original_t1_path} {derivatives_path}")
                    gz_path = os.path.join(deriv_subject_dir, f"{subject}_T1w.nii.gz")

                    if os.path.isfile(gz_path):
                        
                        t1_img   = nib.load(gz_path)
                        t1_data = t1_img.get_fdata()
                        t1_affine = t1_img.affine
                        t1_header = t1_img.header
                        nib.save(t1_img, derivatives_path)
                    
                    run_fsl_anat_preprocessing(derivatives_path)
                    
                    # Refresh the file list after processing
                    if os.path.exists(fsl_anat_dir):
                        fsl_anat_dir_files_list = get_files_from_dir(fsl_anat_dir, max_depth=1)
                        logger.info(f"Found {len(fsl_anat_dir_files_list)} files in fsl_anat directory after processing")
                else:

                    reason = f"Maximum attempts ({max_attempts}) reached. Unable to find critical FSL files."
                    logger.error(f"Skipping subject {subject}: {reason}")
                    skipped_subjects.append((subject, reason))
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    continue
        
        if not files_found:

            print(reason)
            
            reason = "Failed to find all required FSL files after maximum attempts. Skipping subject."
            logger.error(f"Skipping subject {subject}: {reason}")
            skipped_subjects.append((subject, reason))

            
            continue
        


        # Check for bianca_mask
        fsl_anat_dir_files_list = get_files_from_dir(fsl_anat_dir, max_depth=2)
        
        bianca_mask_path_list = find_elements(fsl_anat_dir_files_list,
                                            include=["bianca_mask.nii"],
                                            exclude=["nu"])
        
        if not bianca_mask_path_list:
            logger.info("bianca_mask not found, creating it...")
            
            #warp_file_MNI2structural = invwarpvol  # The inverted warp volume from the previous step.
            keep_intermediate_files = "1"
            
            # Create the make_bianca_mask command
            make_bianca_mask_cmd = f"make_bianca_mask {T1_biascorr_path} {T1_fast_pve_0_path} {warp_file_MNI2structural} {keep_intermediate_files}"
            logger.info(f"Running make_bianca_mask: {make_bianca_mask_cmd}")
            
            if run_command(make_bianca_mask_cmd, "make_bianca_mask"):
                logger.info("✓ make_bianca_mask completed successfully")
            else:
                logger.error("✗ make_bianca_mask failed")
        
            # Refresh file list after creating bianca mask
            fsl_anat_dir_files_list = get_files_from_dir(fsl_anat_dir, max_depth=2)
        
        try:
            bianca_mask_path = find_elements(fsl_anat_dir_files_list,
                                            include=["bianca_mask.nii"],
                                            exclude=["nu"])[0]
            log_file_exists(bianca_mask_path, "bianca_mask")
        except (IndexError, Exception) as e:
            logger.error(f"Failed to find bianca_mask after creation: {str(e)}")
            
                        
            reason = f"Failed to find bianca_mask after creation: {str(e)}"
            logger.error(f"Skipping subject {subject}: {reason}")
            skipped_subjects.append((subject, reason))

            continue
            
        # Check for ventricle mask and distance map
        try:
            ventdistmap_list = find_elements(fsl_anat_dir_files_list,
                                            include=["ventdistmap.nii"],
                                            exclude=["nu"])
            
            ventmask_path = find_elements(fsl_anat_dir_files_list,
                                            include=["_ventmask.nii"],
                                            exclude=["non"])[0]
            log_file_exists(ventmask_path, "ventmask")
            
            ventdistmap_path = os.path.join(fsl_anat_dir, "ventdistmap.nii.gz")
            
            if not os.path.isfile(ventdistmap_path):
                logger.info("ventdistmap not found, creating it...")
                distance_map_cmd = f"distancemap -i {ventmask_path} -o {ventdistmap_path}"
                logger.info(f"Running distancemap: {distance_map_cmd}")
                
                if run_command(distance_map_cmd, "distancemap"):
                    logger.info("✓ distancemap completed successfully")
                else:
                    logger.error("✗ distancemap failed")
            else:
                logger.info(f"ventdistmap already exists: {ventdistmap_path}")
        except (IndexError, Exception) as e:

            reason = f"Failed to find or process ventricle masks: {str(e)}"
            logger.error(f"Skipping subject {subject}: {reason}")
            skipped_subjects.append((subject, reason))

        
            continue

        ########################### FSL_ANAT END ########################################
        
        
        
        ###########################  FLAIR FSL_ANAT SECTION ########################################
        logger.info("=" * 40)
        logger.info("STARTING FLAIR SECTION")
        logger.info("=" * 40)
        
        
        FLAIR_brain_correcteds = find_elements(
            files, 
            include=["FLAIR_brain_corrected"], 
            exclude=["mask", "T1", "mni"]
        )
        
        try:
            original_flair_path = find_elements(
                files, 
                include=["FLAIR"], 
                exclude=["mask", "T1", "mni","brain"]
            )[0]
            logger.info(f"Found FLAIR image: {original_flair_path}")
            
            
        except (IndexError, Exception) as e:
            reason = f"Failed to find FLAIR image for subject {subject}: {str(e)}"
            logger.error(f"Skipping subject {subject}: {reason}")
            skipped_subjects.append((subject, reason))

            continue
        
    
        prepare_flair_sub_dir = os.path.join(fsl_anat_flair, subject,"ses-01" ,"anat")
        os.makedirs(prepare_flair_sub_dir, exist_ok=True)
        logger.info(f"Created prepare_flair subject directory: {prepare_flair_sub_dir}")
        
        
        FLAIR_brain_corrected = process_flair_registration(
                                    subject=subject,
                                    files=files,
                                    original_flair_path=original_flair_path,
                                    deriv_subject_dir=prepare_flair_sub_dir,
                                    MAX_ATTEMPTS=MAX_ATTEMPTS,
                                    logger=logger
                                )
                                                
        
        if FLAIR_brain_corrected is None:
            skipped_subjects.append((subject, "FLAIR fsl_anat preprocessing failed"))
            continue
            

        ################################### FLAIR FSL_ANAT  END ###########################################
        
        
        
        
        
        

        
        ########################### FLAIR SECTION ########################################

            

        prepare_flair_sub_dir = os.path.join(prepare_flair, subject, "anat")
        os.makedirs(prepare_flair_sub_dir, exist_ok=True)
        logger.info(f"Created prepare_flair subject directory: {prepare_flair_sub_dir}")
        

        if not FLAIR_brain_corrected:
            
            FLAIR_brain = os.path.join(prepare_flair_sub_dir, f"{subject}_FLAIR_brain.nii.gz")
            FLAIR_brain_mask_path = os.path.join(prepare_flair_sub_dir, f"{subject}_FLAIR_brain_mask.nii.gz")
            
            if not os.path.isfile(FLAIR_brain):
                logger.info(f"FLAIR brain extraction not found, running BET with frac=0.6...")
                try:
                    fsl_bet(original_flair_path, FLAIR_brain, frac=0.3)
                    logger.info(f"✓ BET completed successfully: {FLAIR_brain}")
                except Exception as e:
                    logger.error(f"✗ BET failed: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            else:
                reason = f"FLAIR brain extraction already exists: {FLAIR_brain}"
                logger.error(f"Skipping subject {subject}: {reason}")
    
    
    
            FLAIR_brain_corrected = os.path.join(prepare_flair_sub_dir, f"{subject}_FLAIR_biascorr.nii.gz")
        
            if not os.path.isfile(FLAIR_brain_corrected):
                logger.info(f"FLAIR bias correction not found, running bias correction...")
                try:
                    apply_bias_correction(FLAIR_brain, FLAIR_brain_corrected)
                    logger.info(f"✓ Bias correction completed successfully: {FLAIR_brain_corrected}")
                except Exception as e:
                    reason = f"✗ Bias correction failed: {str(e)}"
                    logger.error(f"Skipping subject {subject}: {reason}")
                    skipped_subjects.append((subject, reason))
    
                    continue
            else:
                logger.info(f"FLAIR bias correction already exists: {FLAIR_brain_corrected}")
            
        
   
        preproc_T1w = os.path.join(prepare_flair_sub_dir, f"{subject}_space-FLAIR_desc-t1w_brain.nii.gz")
        T1_brain_FLAIR_mat_path = os.path.join(prepare_flair_sub_dir, f"{subject}_from-t1w_to-FLAIR.mat")
        

        if not os.path.isfile(preproc_T1w) or not os.path.isfile(T1_brain_FLAIR_mat_path):
            logger.info("T1w to FLAIR registration not found, running FLIRT...")
            flirt_matrix_command = [
                "flirt",
                "-in", T1_biascorr_brain,
                "-ref", FLAIR_brain_corrected,
                "-out", preproc_T1w,
                "-omat", T1_brain_FLAIR_mat_path,
                "-dof", "12"
            ]
            
            flirt_cmd = " ".join(flirt_matrix_command)
            logger.info(f"Running FLIRT T1w to FLAIR: {flirt_cmd}")
            
            try:
                subprocess.run(flirt_matrix_command, check=True)
                logger.info("✓ FLIRT T1w to FLAIR completed successfully")
                log_file_exists(preproc_T1w, "Registered T1w to FLAIR")
                log_file_exists(T1_brain_FLAIR_mat_path, "T1w to FLAIR transformation matrix")
            except subprocess.CalledProcessError as e:

                reason = f"✗ FLIRT T1w to FLAIR failed: {str(e)}"
                logger.error(f"Skipping subject {subject}: {reason}")
                skipped_subjects.append((subject, reason))
                
                continue
        else:
            logger.info(f"T1w to FLAIR registration already exists: {preproc_T1w}")
            
     
        
        # Register ventricle distance map to FLAIR space
        vendistmap_to_FLAIR_path = os.path.join(prepare_flair_sub_dir, f"{subject}_space-FLAIR_desc-distanceVent.nii.gz")
        
        if not os.path.isfile(vendistmap_to_FLAIR_path):
            logger.info("Ventricle distance map to FLAIR registration not found, running FLIRT...")
            flirt_command = [
                "flirt",
                "-in", ventdistmap_path,
                "-ref", FLAIR_brain_corrected,
                "-out", vendistmap_to_FLAIR_path,
                "-applyxfm",
                "-init", T1_brain_FLAIR_mat_path
            ]
            
            flirt_cmd = " ".join(flirt_command)
            logger.info(f"Running FLIRT ventricle distance map to FLAIR: {flirt_cmd}")
            
            try:
                subprocess.run(flirt_command, check=True)
                logger.info("✓ FLIRT ventricle distance map to FLAIR completed successfully")
                log_file_exists(vendistmap_to_FLAIR_path, "Ventricle distance map in FLAIR space")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ FLIRT ventricle distance map to FLAIR failed: {str(e)}")
                continue
        else:
            logger.info(f"Ventricle distance map in FLAIR space already exists: {vendistmap_to_FLAIR_path}")



        # Register ventricle distance map to FLAIR space
        brain_mask_to_FLAIR_path = os.path.join(prepare_flair_sub_dir, f"{subject}_space-FLAIR_brain_mask.nii.gz")
        
        if not os.path.isfile(brain_mask_to_FLAIR_path):
            logger.info("Ventricle distance map to FLAIR registration not found, running FLIRT...")
            flirt_command = [
                "flirt",
                "-in", T1_biascorr_brain_mask,
                "-ref", FLAIR_brain_corrected,
                "-out", brain_mask_to_FLAIR_path,
                "-applyxfm",
                "-init", T1_brain_FLAIR_mat_path
            ]
            
            flirt_cmd = " ".join(flirt_command)
            logger.info(f"Running FLIRT ventricle distance map to FLAIR: {flirt_cmd}")
            
            try:
                subprocess.run(flirt_command, check=True)
                logger.info("✓ FLIRT ventricle distance map to FLAIR completed successfully")
                log_file_exists(vendistmap_to_FLAIR_path, "Ventricle distance map in FLAIR space")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ FLIRT ventricle distance map to FLAIR failed: {str(e)}")
                continue
        else:
            logger.info(f"Ventricle distance map in FLAIR space already exists: {brain_mask_to_FLAIR_path}")

      

        # Register white matter mask to FLAIR space
        wmmask_FLAIR = os.path.join(prepare_flair_sub_dir, f"{subject}_space-FLAIR_desc-wmmask.nii.gz")
        
        if not os.path.isfile(wmmask_FLAIR):
            logger.info("WM mask to FLAIR registration not found, running FLIRT...")
            flirt_command = [
                "flirt",
                "-in", bianca_mask_path,
                "-ref", FLAIR_brain_corrected,
                "-out", wmmask_FLAIR,
                "-applyxfm",
                "-init", T1_brain_FLAIR_mat_path
            ]
            
            flirt_cmd = " ".join(flirt_command)
            logger.info(f"Running FLIRT WM mask to FLAIR: {flirt_cmd}")
            
            try:
                subprocess.run(flirt_command, check=True)
                logger.info("✓ FLIRT WM mask to FLAIR completed successfully")
                log_file_exists(wmmask_FLAIR, "WM mask in FLAIR space")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ FLIRT WM mask to FLAIR failed: {str(e)}")
                continue
        else:
            logger.info(f"WM mask in FLAIR space already exists: {wmmask_FLAIR}")

        # Register gray matter mask to FLAIR space
        gmmask_FLAIR = os.path.join(prepare_flair_sub_dir, f"{subject}_space-FLAIR_desc-gmmask.nii.gz")
        
        if not os.path.isfile(gmmask_FLAIR):
            logger.info("GM mask to FLAIR registration not found, running FLIRT...")
            flirt_command = [
                "flirt",
                "-in", T1_fast_pve_1_path,
                "-ref", FLAIR_brain_corrected,
                "-out", gmmask_FLAIR,
                "-applyxfm",
                "-init", T1_brain_FLAIR_mat_path
            ]
            
            flirt_cmd = " ".join(flirt_command)
            logger.info(f"Running FLIRT GM mask to FLAIR: {flirt_cmd}")
            
            try:
                subprocess.run(flirt_command, check=True)
                logger.info("✓ FLIRT GM mask to FLAIR completed successfully")
                log_file_exists(gmmask_FLAIR, "GM mask in FLAIR space")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ FLIRT GM mask to FLAIR failed: {str(e)}")
                continue
        else:
            logger.info(f"GM mask in FLAIR space already exists: {gmmask_FLAIR}")

        # Register CSF mask to FLAIR space
        csbfmask_FLAIR = os.path.join(prepare_flair_sub_dir, f"{subject}_space-FLAIR_desc-csbfmask.nii.gz")
        
        if not os.path.isfile(csbfmask_FLAIR):
            logger.info("CSF mask to FLAIR registration not found, running FLIRT...")
            flirt_command = [
                "flirt",
                "-in", T1_fast_pve_0_path,
                "-ref", FLAIR_brain_corrected,
                "-out", csbfmask_FLAIR,
                "-applyxfm",
                "-init", T1_brain_FLAIR_mat_path
            ]
            
            flirt_cmd = " ".join(flirt_command)
            logger.info(f"Running FLIRT CSF mask to FLAIR: {flirt_cmd}")
            
            try:
                subprocess.run(flirt_command, check=True)
                logger.info("✓ FLIRT CSF mask to FLAIR completed successfully")
                log_file_exists(csbfmask_FLAIR, "CSF mask in FLAIR space")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ FLIRT CSF mask to FLAIR failed: {str(e)}")
                continue
        else:
            logger.info(f"CSF mask in FLAIR space already exists: {csbfmask_FLAIR}")
        

    


        
        # Copy ventricle distance map for BIANCA
        bianca_ventdistmap = os.path.join(prepare_flair_sub_dir, f"{subject}_desc-bianca_ventdistmap.nii.gz")
        copy_cmd = f"cp {vendistmap_to_FLAIR_path} {bianca_ventdistmap}"
        run_command(copy_cmd, "Copy ventricle distance map for BIANCA")
        log_file_exists(bianca_ventdistmap, "BIANCA ventricle distance map")
        
        
        # Create deep WMH mask
        deepWMH_mask_path = os.path.join(prepare_flair_sub_dir, f"{subject}_desc-deepWMmask.nii.gz")
        
        if not os.path.isfile(deepWMH_mask_path):
            logger.info("Deep WMH mask not found, creating it...")
            command = f"fslmaths {vendistmap_to_FLAIR_path} -thr 10 -bin {deepWMH_mask_path}"
            
            if run_command(command, "Create deep WMH mask"):
                logger.info("✓ Deep WMH mask created successfully")
                log_file_exists(deepWMH_mask_path, "Deep WMH mask")
            else:
                logger.error("✗ Failed to create deep WMH mask")
                continue
        else:
            logger.info(f"Deep WMH mask already exists: {deepWMH_mask_path}")
            
        # Create periventricular WMH mask
        perWMH_mask_path = os.path.join(prepare_flair_sub_dir, f"{subject}_desc-periventmask.nii.gz")
        
        if not os.path.isfile(perWMH_mask_path):
            logger.info("Periventricular WMH mask not found, creating it...")
            command = f"fslmaths {vendistmap_to_FLAIR_path} -uthr 10 -bin {perWMH_mask_path}"
            
            if run_command(command, "Create periventricular WMH mask"):
                logger.info("✓ Periventricular WMH mask created successfully")
                log_file_exists(perWMH_mask_path, "Periventricular WMH mask")
            else:
                logger.error("✗ Failed to create periventricular WMH mask")
                continue
        else:
            logger.info(f"Periventricular WMH mask already exists: {perWMH_mask_path}")
            
        # Create combined mask (union of deep and periventricular)
        bianca_ventmask_path = os.path.join(prepare_flair_sub_dir, f"{subject}_desc-bianca_ventmask_inverted.nii.gz")
        
        if not os.path.isfile(bianca_ventmask_path):
            logger.info("Combined ventricle mask not found, creating it...")
            command = f"fslmaths {deepWMH_mask_path} -add {perWMH_mask_path} -bin {bianca_ventmask_path}"
            
            if run_command(command, "Create combined ventricle mask"):
                logger.info("✓ Combined ventricle mask created successfully")
                log_file_exists(bianca_ventmask_path, "Combined ventricle mask")
            else:
                logger.error("✗ Failed to create combined ventricle mask")
                continue
        else:
            logger.info(f"Combined ventricle mask already exists: {bianca_ventmask_path}")
            
        # Create inverted mask
        inverted_mask_path = os.path.join(prepare_flair_sub_dir, f"{subject}_desc-bianca_ventmask.nii.gz")
        
        if not os.path.isfile(inverted_mask_path):
            logger.info("Inverted ventricle mask not found, creating it...")
            command = f"fslmaths {bianca_ventmask_path} -mul -1 -add 1 -bin {inverted_mask_path}"
            
            if run_command(command, "Create inverted ventricle mask"):
                logger.info("✓ Inverted ventricle mask created successfully")
                log_file_exists(inverted_mask_path, "Inverted ventricle mask")
            else:
                logger.error("✗ Failed to create inverted ventricle mask")
                continue
        else:
            logger.info(f"Inverted ventricle mask already exists: {inverted_mask_path}")
            
        logger.info("FLAIR processing completed successfully")
        
        
            
        #t1_brain_mask = os.path.join(prepare_flair_sub_dir, f"{subject}_flair_space_t1_brain_mask.nii.gz")
        #brain_mask   =   create_brain_mask(preproc_T1w, t1_brain_mask)
        

        # Apply T1 brain mask to FLAIR image to create better skull-stripped FLAIR
        logger.info(f"Applying T1 brain mask to FLAIR image...")

        #here i want to use the brain mask from t1 to remove the skull from the  FLAIR_biascorr_brain_path
        # FLAIR_brain_corrected_path = os.path.join(prepare_flair_sub_dir, f"{subject}_FLAIR_brain_corrected.nii.gz")
        
        # mask_application_cmd = f"fslmaths {FLAIR_biascorr_brain_path} -mas {brain_mask_to_FLAIR_path} {FLAIR_brain_corrected_path}"
    
        # if run_command(mask_application_cmd, "Apply T1 brain mask to FLAIR"):
        #     logger.info(f"✓ Successfully applied T1 brain mask to FLAIR: {FLAIR_brain_corrected_path}")
        # else:
        #     logger.error(f"✗ Failed to apply T1 brain mask to FLAIR")
        #     continue


        ############################# FLAIR END ######################################     


        # Register FLAIR to MNI space
        FLAIR_mni_mat_path = os.path.join(prepare_flair_sub_dir, f"{subject}_space-MNI_desc-12dof_FLAIR.nii.gz")
        mni_mat_path = os.path.join(prepare_flair_sub_dir, f"{subject}_desc-12dof_from-FLAIR_to-MNI.mat")
        
        if not os.path.isfile(FLAIR_mni_mat_path) or not os.path.isfile(mni_mat_path):
            logger.info("FLAIR to MNI registration not found, running FLIRT...")
            flirt_matrix_command = [
                "flirt",
                "-in", FLAIR_brain_corrected,
                "-ref", STANDARD_SPACE_T1,
                "-out", FLAIR_mni_mat_path,
                "-omat", mni_mat_path,
                "-dof", "12"
            ]
            
            flirt_cmd = " ".join(flirt_matrix_command)
            logger.info(f"Running FLIRT FLAIR to MNI: {flirt_cmd}")
            
            try:
                subprocess.run(flirt_matrix_command, check=True)
                logger.info("✓ FLIRT FLAIR to MNI completed successfully")
                log_file_exists(FLAIR_mni_mat_path, "FLAIR in MNI space")
                log_file_exists(mni_mat_path, "FLAIR to MNI transformation matrix")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ FLIRT FLAIR to MNI failed: {str(e)}")
                continue
        else:
            logger.info(f"FLAIR to MNI registration already exists: {FLAIR_mni_mat_path}")
            
            

        # Register preproc_T1w to MNI space using same matrix
        preproc_T1w_mni_path = os.path.join(prepare_flair_sub_dir, f"{subject}_space-MNI_desc-t1w_brain.nii.gz")
        
        if not os.path.isfile(preproc_T1w_mni_path):
            logger.info("Registering preproc_T1w to MNI space...")
            flirt_apply_cmd = [
                "flirt",
                "-in", preproc_T1w,
                "-ref", STANDARD_SPACE_T1,
                "-applyxfm",
                "-init", mni_mat_path,
                "-out", preproc_T1w_mni_path
            ]
            try:
                subprocess.run(flirt_apply_cmd, check=True)
                logger.info("✓ preproc_T1w successfully registered to MNI space")
                log_file_exists(preproc_T1w_mni_path, "preproc_T1w in MNI space")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ Failed to register preproc_T1w to MNI space: {str(e)}")
        else:
            logger.info(f"preproc_T1w MNI space image already exists: {preproc_T1w_mni_path}")
        
        
        # plot_output_path = os.path.join(NORMAL_SPACE_IMAGES, f"{subject}_t1w_MNI_space.png")
        # plot_flair_mni_image( preproc_T1w_mni_path    ,plot_output_path, f"{subject}  T1 MNI Soace " )
        

        save_path_image          = os.path.join(NORMAL_SPACE_IMAGES, f"{subject}_t1w_FLAIR_MNI_space.png")
        plot_two_nifti_images_2cols(FLAIR_mni_mat_path, 
                                    preproc_T1w_mni_path, 
                                    save_path=save_path_image, 
                                    title1="FLAIR MNI ",
                                    title2="T1 Brain MNI")
        
        
        
        ############################# INFACRT REMOVAL  ######################################                       
                        

        manualmask_paths = find_elements(
            files, 
            include=["manualmask"]
        )

    
        if manualmask_paths:
            manualmask_path = manualmask_paths[0]      
            
            new_manualmask_path = os.path.join(prepare_flair_sub_dir, f"{subject}_manualmask.nii.gz")
            
            aligned_mask_path = match_orientation_and_compare(
                gt_path=FLAIR_brain_corrected,
                pred_path=manualmask_path,
                output_dir=os.path.join(prepare_flair_sub_dir, "orientation_check")
            )
            
            if aligned_mask_path:
                shutil.move(aligned_mask_path, new_manualmask_path)  # overwrite with aligned version
                logger.info(f"✓ Overwrote mask with aligned version: {new_manualmask_path}")
            else:
                logger.warning("No aligned mask created.")
                        
                        
        infarct_paths = find_elements(
            files, 
            include=["infarct"], 
            exclude=["mask"]
        )
            
        
        if infarct_paths and new_manualmask_path:
            logger.info(f"Found infarct image: {infarct_paths}")
            
            infarct_path = infarct_paths[0]
            
            
            deriv_subject_dir_removed   = os.path.join(prepare_removed, subject, "anat")
            os.makedirs(deriv_subject_dir_removed, exist_ok=True)
            
        
            FLAIR_mul_path = os.path.join(deriv_subject_dir_removed, f"{subject}_FLAIR_mul.nii.gz")
            
            #here i want to use the infarct_path  to remove parts from FLAIR_biascorr_brain_path
            mask_out_infarct(
                flair_path=FLAIR_brain_corrected,
                infarct_path=infarct_path,
                output_path=FLAIR_mul_path,
                logger=logger  # Optional
            )
            
            os.path.isfile(FLAIR_mul_path) # False
            
            logger.info(f"created mul image {FLAIR_mul_path}")
            
            infarct_bool = True
            
            updated_manualmask_path = os.path.join(deriv_subject_dir_removed, f"{subject}_manualmask_noinfarct.nii.gz")
            
            logger.info("Removing infarct from manual mask...")
 
        
            mask_out_infarct(
                flair_path=new_manualmask_path,
                infarct_path=infarct_path,
                output_path=updated_manualmask_path,
                logger=logger  # Optional
            )
            
        ############################# INFACRT REMOVAL  END ######################################   
    
        DATA_LOCATION.append({
                "subject": subject,
                "mul": infarct_bool,
                "T1_biascorr_brain": preproc_T1w if os.path.isfile(preproc_T1w) else "",
                "FLAIR_biascorr_brain": FLAIR_brain_corrected if os.path.isfile(FLAIR_brain_corrected) else "",
                "FLAIR_brain_mask": brain_mask_to_FLAIR_path if os.path.isfile(brain_mask_to_FLAIR_path) else "",
                "ventdistmap_FLAIR": vendistmap_to_FLAIR_path if os.path.isfile(vendistmap_to_FLAIR_path) else "",
                "bianca_mask": wmmask_FLAIR if os.path.isfile(wmmask_FLAIR) else "",
                "infarct_path": infarct_path if 'infarct_path' in locals() and os.path.isfile(infarct_path) else "",
                "FLAIR_no_infarct": FLAIR_mul_path if 'FLAIR_mul_path' in locals() and os.path.isfile(FLAIR_mul_path) else "",
                "manualmask": new_manualmask_path if 'new_manualmask_path' in locals() and os.path.isfile(new_manualmask_path) else "",
                "manualmask_noinfarct": updated_manualmask_path if 'updated_manualmask_path' in locals() and os.path.isfile(updated_manualmask_path) else "",
                "perWMH_mask": perWMH_mask_path if os.path.isfile(perWMH_mask_path) else "",
                "deepWMH_mask": deepWMH_mask_path if os.path.isfile(deepWMH_mask_path) else "",
                "FLAIR_mni_mat_path": mni_mat_path if os.path.isfile(mni_mat_path) else ""
            })
                 
        
        #here i want to create a plot with  preproc_T1w ,  FLAIR_brain_corrected , 
        #FLAIR_no_infarct , FLAIR_brain_corrected overlay  manualmask, FLAIR_brain_corrected overlay  updated_manualmask_path, 
        
        plot_output_path = os.path.join(FLAIR_SPACE_IMAGES, f"{subject}_FLAIR_space.png")
        
        if infarct_paths and new_manualmask_path:
        
            create_five_panel_plot(
                t1_path=preproc_T1w,
                flair_path=FLAIR_brain_corrected,
                flair_no_infarct_path=FLAIR_mul_path if infarct_bool else "",
                manualmask_path=infarct_path,
                updated_manualmask_path=updated_manualmask_path,
                out_path=plot_output_path,
                subject_name=subject
            )
            
        else:
            create_4_panel_plot_affine_aware(
                t1_path=preproc_T1w,
                flair_path=FLAIR_brain_corrected,
                manualmask_path= new_manualmask_path,
                out_path=plot_output_path,
                subject_name=subject
            )
        
        plot_output_path = os.path.join(FLAIR_SPACE_IMAGES, f"{subject}_Ventrickles.png")

        plot_overlay_4cols(
            flair_brain_path=FLAIR_brain_corrected,
            perwmh_mask_path=perWMH_mask_path,
            deepwmh_mask_path=deepWMH_mask_path,
            wmmask_path=wmmask_FLAIR,
            out_path=plot_output_path,
            title=f"{subject}"
        )
        

        ############################# T1w SECTION ######################################
        logger.info("=" * 40)
        logger.info("STARTING T1w SECTION")
        logger.info("=" * 40)
        
        prepare_t1w_sub_dir = os.path.join(prepare_t1w, subject, "anat")
        os.makedirs(prepare_t1w_sub_dir, exist_ok=True)
        logger.info(f"Created prepare_t1w subject directory: {prepare_t1w_sub_dir}")
        
        # Copy T1 brain to prepare_t1w directory
        brain_T1w = os.path.join(prepare_t1w_sub_dir, f"{subject}_space-tpl_desc-brain_T1w.nii.gz")
        
        if not os.path.isfile(brain_T1w):
            logger.info(f"Copying T1 brain to prepare_t1w directory...")
            copy_cmd = f"cp {T1_biascorr_brain} {brain_T1w}"
            
            if run_command(copy_cmd, "Copy T1 brain to prepare_t1w"):
                logger.info("✓ T1 brain copied successfully")
                log_file_exists(brain_T1w, "T1 brain in template space")
            else:
                logger.error("✗ Failed to copy T1 brain")
                continue
        else:
            logger.info(f"T1 brain in template space already exists: {brain_T1w}")
            
            
        # Copy T1 bias-corrected to prepare_t1w directory
        tpl_T1w = os.path.join(prepare_t1w_sub_dir, f"{subject}_space-tpl_T1w.nii.gz")
        
        if not os.path.isfile(tpl_T1w):
            logger.info(f"Copying bias-corrected T1 to prepare_t1w directory...")
            copy_cmd = f"cp {T1_biascorr_path} {tpl_T1w}"
            
            if run_command(copy_cmd, "Copy bias-corrected T1 to prepare_t1w"):
                logger.info("✓ Bias-corrected T1 copied successfully")
                log_file_exists(tpl_T1w, "Bias-corrected T1 in template space")
            else:
                logger.error("✗ Failed to copy bias-corrected T1")
                continue
        else:
            logger.info(f"Bias-corrected T1 in template space already exists: {tpl_T1w}")
            
        logger.info("T1w processing completed successfully")
        
        ############################# T1w END ######################################
        
        logger.info("=" * 60)
        logger.info(f"SUBJECT {subject} PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
    

    except Exception as e:
        logger.error(f"Unexpected error processing subject {subject}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        continue
    
    
#save the data location in here as excel
df_data_location = pd.DataFrame(DATA_LOCATION)

excel_output_path = os.path.join(derivatives, "data_location_summary.xlsx")
df_data_location.to_excel(excel_output_path, index=False)


df_data_location["subject"].sort_values()
len(df_data_location["subject"])
len(DATA_SET_directorys)


# Log the skipped subjects
if skipped_subjects:
    logger.info("The following subjects were skipped:")
    for sub, reason in skipped_subjects:
        logger.info(f" - {sub}: {reason}")
else:
    logger.info("No subjects were skipped.")

logger.info("=" * 80)
logger.info("NEUROIMAGING PIPELINE COMPLETED")
logger.info("=" * 80)

logger.info("=" * 80)
logger.info("NEUROIMAGING PIPELINE COMPLETED")
logger.info("=" * 80)
                        
                        