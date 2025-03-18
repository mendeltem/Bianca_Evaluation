#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 20:19:56 2025

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
    run_locate_testing
)
# Load environment variables from .env file
load_dotenv()

# Assign paths dynamically
DATA_SET = os.getenv("DATA_SET")
LOCATE_PATH = os.getenv("LOCATE_PATH")
LOCATE_TRAIN_PATH = os.getenv("LOCATE_TRAIN_PATH")
LOCATE_TEST_PATH = os.getenv("LOCATE_TEST_PATH")

# Check if variables are loaded correctly
if LOCATE_PATH is None or DATA_SET is None:
    raise ValueError("One or more environment variables are missing. Check your .env file.")

derivatives  = os.path.join(DATA_SET,"derivatives")
locate            = os.path.join(derivatives,"locate")
os.makedirs(locate, exist_ok=True)

# Define parameters
feature_select = [1, 1, 1, 1]  # Use all features
verbose = 1  # Show detailed output

run_locate_testing(
    train_image_directory_path=LOCATE_TRAIN_PATH,
    test_image_directory_name=LOCATE_TEST_PATH,
    locate_path=LOCATE_PATH,
    verbose=verbose,
    feature_select=feature_select
    )










