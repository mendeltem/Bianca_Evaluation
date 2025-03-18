#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 09:17:06 2025

@author: temuuleu
"""

import os
from dotenv import load_dotenv
import subprocess

from library import (
    get_subdirectories,
    get_files_from_dir,
    find_elements,
    run_fsl_anat_preprocessing,
    apply_bias_correction,
    fsl_bet,
)

# Load environment variables from .env file
load_dotenv()

# Assign paths dynamically
STANDARD_SPACE_T1 = os.getenv("STANDARD_SPACE_T1")
DATA_SET = os.getenv("DATA_SET")

# Check if variables are loaded correctly
if STANDARD_SPACE_T1 is None or DATA_SET is None:
    raise ValueError("One or more environment variables are missing. Check your .env file.")


DATA_SET_directorys = get_subdirectories(DATA_SET)

derivatives  = os.path.join(DATA_SET,"derivatives")

fsl_anat  = os.path.join(derivatives,"fsl_anat")
os.makedirs(fsl_anat, exist_ok=True)


prepare_flair     = os.path.join(derivatives,"prepare_flair")
os.makedirs(prepare_flair, exist_ok=True)


# prepare_template  = os.path.join(derivatives,"prepare_template")
prepare_t1w       = os.path.join(derivatives,"prepare_t1w")
os.makedirs(prepare_t1w, exist_ok=True)

# bianca            = os.path.join(derivatives,"bianca")

for di, sub_dir in enumerate(DATA_SET_directorys[:]):
    
    print(di, sub_dir)
    subject     = os.path.basename(sub_dir)

    sub_anat_dir = os.path.join(sub_dir,"anat")

    files     = get_files_from_dir(sub_anat_dir, endings=[".nii", ".nii.gz"], max_depth=1)
    
    original_t1_path = find_elements(
        files, 
        include=["T1w"], 
        exclude=[ "mask"]
    )[0]
    
    print(f"{di} {subject} ")
    print(f"{di} original_t1_path {original_t1_path} ")
    
    
    ########################### FLSANAT BEGINN ########################################

    deriv_subject_dir = os.path.join(fsl_anat, subject,"anat")
    os.makedirs(deriv_subject_dir, exist_ok=True)

    derivatives_path = os.path.join(deriv_subject_dir, f"{subject}_T1w.nii")

    fsl_anat_dir = os.path.join(deriv_subject_dir,"fsl_anat.anat" )
    fsl_anat_dir_files_list = get_files_from_dir(fsl_anat_dir,max_depth=1)
    


    original_t1_paths_list = find_elements(fsl_anat_dir_files_list,
                                           include=["T1_biascorr_brain.nii"],
                                           exclude=["nu"])
    
    
    T1_biascorr_brain = ""
    
    if len(original_t1_paths_list) == 1:
        T1_biascorr_brain = original_t1_paths_list[0]
    else:
        print( "T1_biascorr_brain ")
 

    if not T1_biascorr_brain:
    
        print("run fsl_anat")
        os.system(f"cp {original_t1_path} {derivatives_path}")
        
        try:
            run_fsl_anat_preprocessing(derivatives_path)
        except:
            print("failed fsl_run")
            
            
        if len(original_t1_paths_list) == 1:
            T1_biascorr_brain = original_t1_paths_list[0]
        else:
            print( "T1_biascorr_brain ")
            
            
            
    warp_file_MNI2structural = find_elements(fsl_anat_dir_files_list,
                                           include=["MNI_to_T1_nonlin_field"],
                                           exclude=["nu"])[0]
    
    
    T1_fast_pve_0_path = find_elements(fsl_anat_dir_files_list,
                                           include=["T1_fast_pve_0"],
                                           exclude=["nu"])[0]
    
    #gm
    T1_fast_pve_1_path = find_elements(fsl_anat_dir_files_list,
                                           include=["T1_fast_pve_1.nii"],
                                           exclude=["nu"])[0]
    
    T1_biascorr_path = find_elements(fsl_anat_dir_files_list,
                                           include=["T1_biascorr.nii"],
                                           exclude=["nu"])[0]
    

    T1_biascorr_path = find_elements(fsl_anat_dir_files_list,
                                           include=["T1_biascorr.nii"],
                                           exclude=["nu"])[0]
    
    
    fsl_anat_dir_files_list = get_files_from_dir(fsl_anat_dir,max_depth=2)


    bianca_mask_path = find_elements(fsl_anat_dir_files_list,
                                           include=["bianca_mask.nii"],
                                           exclude=["nu"])[0]
    
    
    ventdistmap_path = find_elements(fsl_anat_dir_files_list,
                                           include=["ventdistmap.nii"],
                                           exclude=["nu"])[0]
    
    
    if not os.path.isfile(bianca_mask_path)  or   not os.path.isfile(ventdistmap_path):


        #warp_file_MNI2structural = invwarpvol  # The inverted warp volume from the previous step.
        keep_intermediate_files = "1"
        
        
        # # Create the make_bianca_mask command.
        print("start make_bianca_mask ")
        make_bianca_mask_cmd = f"make_bianca_mask {T1_biascorr_path} {T1_fast_pve_0_path} {warp_file_MNI2structural} {keep_intermediate_files}"
        # print(f"start make bianca maske: {make_bianca_mask_cmd}")
        os.system(make_bianca_mask_cmd) 
    
    
    
    ########################### FLSANAT END ########################################
    
    

    
    ########################### FLAIR BEGINN ########################################
    
    
    original_flair_path = find_elements(
        files, 
        include=["FLAIR"], 
        exclude=[ "mask","T1","mni"]
    )[0]
        
    
    prepare_flair_sub_dir = os.path.join(prepare_flair, subject,"anat")
    os.makedirs(prepare_flair_sub_dir, exist_ok=True)


    FLAIR_brain_path = os.path.join(prepare_flair_sub_dir, f"{subject}_FLAIR_brain.nii.gz")
    FLAIR_brain_mask_path = os.path.join(prepare_flair_sub_dir, f"{subject}_FLAIR_brain_mask.nii.gz")
    
    if not os.path.isfile(FLAIR_brain_path):
        fsl_bet(original_flair_path, FLAIR_brain_path, frac=0.6) 
        
        
    brainmask_path = os.path.join(prepare_flair_sub_dir, f"{subject}_space-FLAIR_desc-brainmask.nii.gz")
    os.system(f"cp {FLAIR_brain_mask_path}  {brainmask_path}")
        
    
    FLAIR_biascorr_brain_path = os.path.join( prepare_flair_sub_dir,f"{subject}_FLAIR_biascorr.nii.gz")
    apply_bias_correction(FLAIR_brain_path,FLAIR_biascorr_brain_path)
        
    preproc_T1w = os.path.join( prepare_flair_sub_dir,f"{subject}_space-FLAIR_desc-t1w_brain.nii.gz")
    
    T1_brain_FLAIR_mat_path   =  os.path.join( prepare_flair_sub_dir,f"{subject}_from-t1w_to-FLAIR.mat")
    
    
    if not os.path.isfile(preproc_T1w)  or  not os.path.isfile(T1_brain_FLAIR_mat_path):
    
        flirt_matrix_command = [
            "flirt",
            "-in", T1_biascorr_brain,
            "-ref", FLAIR_biascorr_brain_path,
            "-out", preproc_T1w,
            "-omat", T1_brain_FLAIR_mat_path,  # Save transformation matrix
            "-dof", "12"  # Rigid body registration
            ]
    
        subprocess.run(flirt_matrix_command)
        
    
    vendistmap_to_FLAIR_path    =   os.path.join(prepare_flair_sub_dir,f"{subject}_space-FLAIR_desc-distanceVent.nii.gz")
    
    if not os.path.isfile(vendistmap_to_FLAIR_path):
    
        print("run  flirt ventricle")
        
        flirt_command = [
            "flirt",
            "-in", ventdistmap_path,
            "-ref", FLAIR_biascorr_brain_path,
            "-out", vendistmap_to_FLAIR_path,
            "-applyxfm",
            "-init", T1_brain_FLAIR_mat_path
        ]
        subprocess.run(flirt_command)
        


    wmmask_FLAIR    =   os.path.join(prepare_flair_sub_dir,f"{subject}_space-FLAIR_desc-wmmask.nii.gz")
    
    if not os.path.isfile(wmmask_FLAIR) :
        print("run  flirt biancamask")
        flirt_command = [
            "flirt",
            "-in", bianca_mask_path,
            "-ref", FLAIR_biascorr_brain_path,
            "-out", wmmask_FLAIR,
            "-applyxfm",
            "-init", T1_brain_FLAIR_mat_path
        ]
        subprocess.run(flirt_command)
    
    gmmask_FLAIR    =   os.path.join(prepare_flair_sub_dir,f"{subject}_space-FLAIR_desc-gmmask.nii.gz")

    if not os.path.isfile(gmmask_FLAIR):
        print("run  flirt GM")
        flirt_command = [
            "flirt",
            "-in", T1_fast_pve_1_path,
            "-ref", FLAIR_biascorr_brain_path,
            "-out", gmmask_FLAIR,
            "-applyxfm",
            "-init", T1_brain_FLAIR_mat_path
        ]
        subprocess.run(flirt_command)


    csbfmask_FLAIR    =   os.path.join(prepare_flair_sub_dir,f"{subject}_space-FLAIR_desc-csbfmask.nii.gz")
    
    if not os.path.isfile(csbfmask_FLAIR):
        print("run  flirt CSBF")
        flirt_command = [
            "flirt",
            "-in", T1_fast_pve_0_path,
            "-ref", FLAIR_biascorr_brain_path,
            "-out", csbfmask_FLAIR,
            "-applyxfm",
            "-init", T1_brain_FLAIR_mat_path
        ]
        subprocess.run(flirt_command)
        
    
    FLAIR_mni_mat_path   =  os.path.join( prepare_flair_sub_dir,f"{subject}_space-MNI_desc-12dof_FLAIR.nii.gz")

    mni_mat_path   =  os.path.join( prepare_flair_sub_dir,f"{subject}_desc-12dof_from-FLAIR_to-MNI.mat")

    if not os.path.isfile(preproc_T1w)  or  not os.path.isfile(T1_brain_FLAIR_mat_path):
    
        flirt_matrix_command = [
            "flirt",
            "-in",  FLAIR_biascorr_brain_path,
            "-ref", STANDARD_SPACE_T1,
            "-out", FLAIR_mni_mat_path,
            "-omat", mni_mat_path,  # Save transformation matrix
            "-dof", "12"  # Rigid body registration
            ]
    
        subprocess.run(flirt_matrix_command)

    
    bianca_ventdistmap = os.path.join( prepare_flair_sub_dir,f"{subject}_desc-bianca_ventdistmap.nii.gz")
    os.system(f"cp {vendistmap_to_FLAIR_path} {bianca_ventdistmap}")
    
    deepWMH_mask_path = os.path.join( prepare_flair_sub_dir,f"{subject}_desc-deepWMmask.nii.gz")   # _desc-deepWMmask.nii.gz

    # Construct the fslmaths command
    command = f"fslmaths {vendistmap_to_FLAIR_path} -thr 10 -bin {deepWMH_mask_path}"
    # Execute the command using subprocess.run
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

    perWMH_mask_path  = os.path.join( prepare_flair_sub_dir,f"{subject}_desc-periventmask.nii.gz")  
    
    # Construct the fslmaths command
    command = f"fslmaths {vendistmap_to_FLAIR_path} -uthr 10 -bin {perWMH_mask_path}"
    
    # Execute the command using subprocess.run
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    bianca_ventmask_path   = os.path.join(prepare_flair_sub_dir, f"{subject}_desc-bianca_ventmask_inverted.nii.gz")
    
    
    # Construct the fslmaths command to create the union of the two masks
    command = f"fslmaths {deepWMH_mask_path} -add {perWMH_mask_path} -bin {bianca_ventmask_path}"
    
    # Execute the command using subprocess.run
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)  
    
    # First, get your original mask (assuming it's the bianca_ventmask you created)
    original_mask_path = bianca_ventmask_path
    
    # Define the output path for the inverted mask
    inverted_mask_path = os.path.join(prepare_flair_sub_dir, f"{subject}_desc-bianca_ventmask.nii.gz")
    
    # Construct the fslmaths command to invert the mask
    # The -mul -1 -add 1 operation inverts a binary mask (1 becomes 0, 0 becomes 1)
    command = f"fslmaths {original_mask_path} -mul -1 -add 1 -bin {inverted_mask_path}"
    
    # Execute the command using subprocess.run
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    
    print(f"Inverted mask created at: {inverted_mask_path}")
    
    ############################# FLAIR END ######################################
    
    
    
    ############################# T1w BEGINN ######################################
    
    prepare_t1w_sub_dir = os.path.join(prepare_t1w, subject,"anat")
    os.makedirs(prepare_t1w_sub_dir, exist_ok=True)
    
    
    brain_T1w = os.path.join( prepare_t1w_sub_dir,f"{subject}_space-tpl_desc-brain_T1w.nii.gz")
    os.system(f"cp {T1_biascorr_brain} {brain_T1w}")
    
    
    tpl_T1w   = os.path.join( prepare_t1w_sub_dir,f"{subject}_space-tpl_T1w.nii.gz")
    os.system(f"cp {T1_biascorr_path} {tpl_T1w}")
    

    ############################# T1w END ######################################
    
    
    
    
 
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





    
    
    








