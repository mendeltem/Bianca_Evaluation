#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:18:40 2025

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
DATA_SET = os.getenv("DATA_SET")

# Check if variables are loaded correctly
if DATA_SET is None:
    raise ValueError("One or more environment variables are missing. Check your .env file.")

masterfile_path = os.getenv("MATERFILE")

print("MATERFILE:", masterfile_path)

DATA_SET_directorys = get_subdirectories(DATA_SET)

derivatives  = os.path.join(DATA_SET,"derivatives")

fsl_anat  = os.path.join(derivatives,"fsl_anat")
os.makedirs(fsl_anat, exist_ok=True)

prepare_flair     = os.path.join(derivatives,"prepare_flair")
os.makedirs(prepare_flair, exist_ok=True)

master_file_t2_t1_text_lines  = []

for di, sub_dir in enumerate(DATA_SET_directorys[:]):
    
    print(di, sub_dir)
    subject     = os.path.basename(sub_dir)
    

    sub_anat_flair_dir = os.path.join(sub_dir,"anat")
    
    files            = get_files_from_dir(sub_anat_flair_dir, 
                                          endings=[".nii", ".nii.gz"], 
                                          max_depth=1)
    
    sub_prepare_flair_dir = os.path.join(prepare_flair,subject,"anat")
    
    flair_files     = get_files_from_dir(sub_prepare_flair_dir, endings=[".nii", ".nii.gz"], max_depth=1)
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
        
    except:
        continue
    
    master_file_line = flair_path  + " " +    t1_path  + " " +  mni_mat_path   + " " +  manual_mask_path
    master_file_t2_t1_text_lines.append(master_file_line)
    
    
with open(masterfile_path, 'w') as f:
    f.write('\n'.join(master_file_t2_t1_text_lines))   
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




