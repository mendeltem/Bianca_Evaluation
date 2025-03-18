#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:57:15 2025

@author: temuuleu
"""

import os
from dotenv import load_dotenv
import subprocess
import nibabel as nib
import numpy as np

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
BIANCA_MODEL = os.getenv("BIANCA_MODEL")


THRESHHOLD_BIANCA = os.getenv("THRESHHOLD_BIANCA")


# Check if variables are loaded correctly
if BIANCA_MODEL is None or DATA_SET is None:
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


bianca            = os.path.join(derivatives,"bianca")
os.makedirs(bianca, exist_ok=True)



for di, sub_dir in enumerate(DATA_SET_directorys[:10]):
    
    print(di, sub_dir)
    subject     = os.path.basename(sub_dir)
    

    sub_anat_flair_dir = os.path.join(sub_dir,"anat")
    
    files            = get_files_from_dir(sub_anat_flair_dir, 
                                          endings=[".nii", ".nii.gz"], 
                                          max_depth=1)
    
    
    sub_prepare_flair_dir = os.path.join(prepare_flair,subject,"anat")
    
    flair_files    = get_files_from_dir(sub_prepare_flair_dir, endings=[".nii", ".nii.gz"], max_depth=1)
    mat_files     = get_files_from_dir(sub_prepare_flair_dir, endings=[".mat"], max_depth=1)
    
    try:
        
        manual_mask_path = find_elements(
            files, 
            include=["manual_mask"]
        )[0]
        
        t1_path = find_elements(
            flair_files, 
            include=["T1w"], 
            exclude=[ "mask"]
        )[0]
        
        flair_path = find_elements(
            flair_files, 
            include=["FLAIR_biascorr"], 
            exclude=[ "mask"]
        )[0]
        
        mni_mat_path = find_elements(
            mat_files, 
            include=["from-t1w_to-FLAIR.mat"], 
            exclude=[ "mask"]
        )[0]
        
        wmmask = find_elements(
            flair_files, 
            include=["FLAIR_desc-wmmask"], 
            exclude=[ "#"]
        )[0]
        
        
        bianca_ventmask_inverted = find_elements(
            flair_files, 
            include=["bianca_ventmask_inverted"], 
            exclude=[ "#"]
        )[0]
        
        
        deepWMmask = find_elements(
            flair_files, 
            include=["deepWMmask"], 
            exclude=[ "#"]
        )[0]
        
        periventmask = find_elements(
            flair_files, 
            include=["periventmask"], 
            exclude=[ "#"]
        )[0]

    except:
        continue
    
    
    
    master_file_t2_t1_text_lines = []             
    master_file_line = flair_path  + " " +    t1_path  + " " +  mni_mat_path   + " " +  mni_mat_path
    master_file_t2_t1_text_lines.append(master_file_line)
    
    sub_bianca_dir = os.path.join(bianca,subject,"anat")
    os.makedirs(sub_bianca_dir, exist_ok=True)
    master_file_t2_t1_test_path   = os.path.join(sub_bianca_dir, f"{subject}_master_file_test.txt")
    
    with open(master_file_t2_t1_test_path, 'w') as f:
        f.write('\n'.join(master_file_t2_t1_text_lines))   
        


    output_mask_path = os.path.join(sub_bianca_dir, f"{subject}_BIANCA_LPM.nii.gz")
    
    if not os.path.isfile(output_mask_path):
    
        # Construct the BIANCA command to use the saved model
        test_bianca_commands = [
            "bianca",
            "--singlefile=" + master_file_t2_t1_test_path,
            "--brainmaskfeaturenum=1",
            "--matfeaturenum=3",
            "--featuresubset=1,2",
            "--loadclassifierdata="+BIANCA_MODEL,
            "--querysubjectnum=1" ,
            "-o", output_mask_path,
            "-v"
        ]
        # # Execute the command
        try:
            subprocess.run(test_bianca_commands, check=True)
        except subprocess.CalledProcessError as e:
            print("Error in running BIANCA:", e)
            


    bianca_lpm_masked = os.path.join(sub_bianca_dir, f"{subject}_FLAIR_desc-biancamasked.nii.gz")
    
    # Apply mask
    mask_cmd = [
        "fslmaths",
        output_mask_path,
        "-mas",
        wmmask,
        bianca_lpm_masked
    ]
    subprocess.run(mask_cmd, check=True)



    threshold_str = round(float(THRESHHOLD_BIANCA)*100)
    thresh_output = os.path.join(sub_bianca_dir, f"{subject}_FLAIR_desc-thresh{threshold_str}_biancaLPMmaskedThrBin.nii.gz")
  
    # Apply threshold
    threshold_cmd = [
        "fslmaths",
        output_mask_path,
        "-thr", str(THRESHHOLD_BIANCA),
        "-bin",
        thresh_output
    ]
    subprocess.run(threshold_cmd, check=True)  
    
    bianca_ventmask_inverted = find_elements(
        flair_files, 
        include=["bianca_ventmask_inverted"], 
        exclude=[ "#"]
    )[0]
    
    deepWMmask = find_elements(
        flair_files, 
        include=["deepWMmask"], 
        exclude=[ "#"]
    )[0]
    
    periventmask = find_elements(
        flair_files, 
        include=["periventmask"], 
        exclude=[ "#"]
    )[0]
    
    
    perWMH_controll_path  = os.path.join( sub_bianca_dir,f"{subject}_perWMH_thresh{threshold_str}.nii.gz")
    deepWMH_controll_path = os.path.join( sub_bianca_dir,f"{subject}_deepWMH_thresh{threshold_str}.nii.gz")
    wholeWMH_controll_path = os.path.join( sub_bianca_dir,f"{subject}_wholeWMH_thresh{threshold_str}.nii.gz")
    
    
    # Construct the fslmaths command for multiplication
    command = f"fslmaths {thresh_output} -mul {periventmask} {perWMH_controll_path}"
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    
    # Construct the fslmaths command for multiplication
    command = f"fslmaths {thresh_output} -mul {deepWMmask} {deepWMH_controll_path}"
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)  
    
    # Construct the fslmaths command for multiplication
    command = f"fslmaths {thresh_output} -mul {bianca_ventmask_inverted} {wholeWMH_controll_path}"
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)  
    
    
    bianca_flair_path = os.path.join( sub_bianca_dir,f"{subject}_FLAIR_biascorr.nii.gz")
    os.system(f"cp {flair_path} {bianca_flair_path} ")        
    







    
    
    
    