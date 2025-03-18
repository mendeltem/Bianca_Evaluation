# **BIANCA Processing Pipeline**

This repository contains a set of Python scripts for processing structural MRI and FLAIR images, including skull stripping, bias correction, and lesion segmentation using **BIANCA** (Brain Intensity AbNormality Classification Algorithm) from **FSL**.

## **Table of Contents**
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Environment Variables](#environment-variables)
- [License](#license)

---

## **Overview**
This pipeline processes **T1-weighted (T1w) and FLAIR** MRI images and performs the following steps:

1. **Preprocessing**  
   - Skull stripping using `BET`
   - N4 Bias Field Correction (ANTS)
   - FSL ANAT processing
   - Template registration

2. **FLAIR Image Processing**  
   - Brain extraction
   - Bias correction
   - Registration to T1w
   - Ventricle distance map computation

3. **Lesion Segmentation (BIANCA)**  
   - White matter mask generation
   - Periventricular and deep lesion masks

4. **MNI Registration**
   - 12 DOF linear registration from FLAIR to MNI
   - Transformation of lesion masks to MNI space

---

## **Requirements**
Before running the scripts, ensure that the following dependencies are installed:

- **Python 3.8+**
- **FSL** (for `flirt`, `bet`, `fsl_anat`, `fslmaths`)
- **ANTsPy** (for bias correction)
- **Nipype** (for interfacing with FSL)
- **Nibabel** (for handling NIfTI images)
- **dotenv** (for managing environment variables)

### **Install Dependencies**
You can install the required Python libraries with:

```sh
pip install nibabel antsibp nipype python-dotenv



## **Input Data Structure**
The pipeline expects the input data to be structured as follows:


```plaintext
DATASET_STANDARD/
├── sub-000/
│   └── anat/
│       ├── sub-000_FLAIR.nii.gz
│       ├── sub-000_infarct.nii.gz
│       ├── sub-000_manual_mask.nii.gz
│       ├── sub-000_T1w.nii.gz
│
├── sub-001/
│   └── anat/
│       ├── sub-001_FLAIR.nii.gz
│       ├── sub-001_infarct.nii.gz
│       ├── sub-001_manual_mask.nii.gz
│       ├── sub-001_T1w.nii.gz
│
derivatives/


## **Intermediate Results - prepare_flair**

The `prepare_flair` directory contains intermediate processing results for FLAIR images:

```plaintext
DATASET_STANDARD/
├── derivatives/
│   ├── prepare_flair/
│   │   ├── sub-000/
│   │   │   └── anat/
│   │   │       ├── sub-000_desc-bianca_ventdistmap.nii.gz
│   │   │       ├── sub-000_desc-bianca_ventmask_inverted.nii.gz
│   │   │       ├── sub-000_desc-bianca_ventmask.nii.gz
│   │   │       ├── sub-000_desc-deepWMmask.nii.gz
│   │   │       ├── sub-000_desc-periventmask.nii.gz
│   │   │       ├── sub-000_FLAIR_biascorr.nii.gz
│   │   │       ├── sub-000_FLAIR_brain_mask.nii.gz
│   │   │       ├── sub-000_FLAIR_brain.nii.gz
│   │   │       ├── sub-000_from-t1w_to-FLAIR.mat
│   │   │       ├── sub-000_space-FLAIR_desc-brainmask.nii.gz
│   │   │       ├── sub-000_space-FLAIR_desc-csfbmask.nii.gz
│   │   │       ├── sub-000_space-FLAIR_desc-distanceVent.nii.gz
│   │   │       ├── sub-000_space-FLAIR_desc-gmmask.nii.gz
│   │   │       ├── sub-000_space-FLAIR_desc-t1w_brain.nii.gz
│   │   │       ├── sub-000_space-FLAIR_desc-wmmask.nii.gz


## **Intermediate Results - prepare_template**

The `prepare_template` directory contains template-based intermediate processing results:

```plaintext
derivatives/
├── prepare_template/
│   ├── sub-000/
│   │   └── anat/
│   │       ├── sub-000_desc-bianca_ventdistmap.nii.gz
│   │       ├── sub-000_desc-bianca_ventmask.nii.gz
│   │       ├── sub-000_desc-bianca_wmmask.nii.gz
│   │       ├── sub-000_desc-brain_mask.nii.gz
│   │       ├── sub-000_desc-deepWMmask.nii.gz
│   │       ├── sub-000_desc-periventmask.nii.gz
│   │       ├── sub-000_space-FLAIR_desc-wmmask.nii.gz


## **Intermediate Results - prepare_t1w**

The `prepare_t1w` directory contains processed T1-weighted images mapped to template space:

```plaintext
DATASET_STANDARD/
├── derivatives/
│   ├── prepare_t1w/
│   │   ├── sub-000/
│   │   │   └── anat/
│   │   │       ├── sub-000_space-tpl_desc-brain_T1w.nii.gz
│   │   │       ├── sub-000_space-tpl_T1w.nii.gz



The Bianca Output:

```plaintext
DATASET_STANDARD/
├── derivatives/
│   ├── bianca/
│   │   ├── sub-000/
│   │   │   └── anat/
│   │   │       ├── sub-000_BIANCA_LPM.nii.gz
│   │   │       ├── sub-000_deepWMH_thresh90.nii.gz
│   │   │       ├── sub-000_FLAIR_biascorr.nii.gz
│   │   │       ├── sub-000_FLAIR_desc-biancamasked.nii.gz
│   │   │       ├── sub-000_FLAIR_desc-thresh90_biancaLPMmaskedThrBin.nii.gz
│   │   │       ├── sub-000_master_file_test.txt
│   │   │       ├── sub-000_perWMH_thresh90.nii.gz
│   │   │       ├── sub-000_wholeWMH_thresh90.nii.gz



