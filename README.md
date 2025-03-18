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








