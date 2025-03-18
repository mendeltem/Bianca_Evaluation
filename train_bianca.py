#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:19:00 2025

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
DATA_SET     = os.getenv("DATA_SET")
BIANCA_MODEL = os.getenv("BIANCA_MODEL")

# Check if variables are loaded correctly
if DATA_SET is None:
    raise ValueError("One or more environment variables are missing. Check your .env file.")

masterfile_path = os.getenv("MATERFILE")

print("MATERFILE:", masterfile_path)

# Check if master file exists
if not os.path.isfile(masterfile_path):
    raise FileNotFoundError(f"Master file not found at {masterfile_path}")


# Read master file to count lines
with open(masterfile_path, 'r') as f:
    master_file_t2_t1_text_lines = f.readlines()

row_number = len(master_file_t2_t1_text_lines)

# Training points
trainingpts = 2000
trainstring = ",".join([str(r) for r in range(1, row_number)])

row_number  = len(master_file_t2_t1_text_lines) 
trainstring = ",".join([str(r) for r in range(1, row_number)])

print("start train bianca")


# Check if model file already exists
if os.path.isfile(BIANCA_MODEL):
    overwrite = input(f"Model file '{BIANCA_MODEL}' already exists. Overwrite? (yes/no): ").strip().lower()
    if overwrite != "yes":
        print("Skipping training. Using existing model.")
        exit(0)

print("Starting BIANCA training...")


train_bianca_commands = [
    "bianca", 
    "--singlefile=" + masterfile_path,
    "--brainmaskfeaturenum=1",
    "--matfeaturenum=3",
    "--featuresubset=1,2",
    "--labelfeaturenum=4", 
    f"--trainingpts={trainingpts}",
    "--nonlespts=10000",
    "--trainingnums=" + trainstring,
    "--saveclassifierdata=" +BIANCA_MODEL,
    "--querysubjectnum=" + str(row_number),
    "-v"
]

# # Execute the command
try:
    print("start bianca training")
    subprocess.run(train_bianca_commands, check=True)
except subprocess.CalledProcessError as e:
    print("Error in running BIANCA:", e)

    