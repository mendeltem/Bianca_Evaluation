# **BIANCA Processing Pipeline**

This repository contains a set of Python scripts for processing structural MRI and FLAIR images, including **skull stripping**, **bias correction**, and **lesion segmentation** using **BIANCA** (**Brain Intensity AbNormality Classification Algorithm**) from **FSL**.

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Pipeline Workflow](#pipeline-workflow)
- [Environment Variables](#environment-variables)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Credits](#credits)

---

## **Overview**
The **BIANCA Processing Pipeline** is designed for automated processing of **T1-weighted (T1w) and FLAIR** MRI images. The pipeline performs the following key operations:

1. **Preprocessing**
   - Skull stripping using `BET` (Brain Extraction Tool)
   - N4 Bias Field Correction (`ANTs`)
   - FSL `ANAT` preprocessing
   - Template registration

2. **FLAIR Image Processing**
   - Brain extraction
   - Bias correction
   - Registration to T1w space
   - Ventricle distance map computation

3. **Lesion Segmentation using BIANCA**
   - White matter mask generation
   - Periventricular and deep lesion masks
   - Lesion probability map computation

4. **MNI Registration**
   - 12 DOF linear registration from FLAIR to MNI space
   - Transformation of lesion masks to MNI space

---

## **Features**
- Fully automated pipeline for **lesion segmentation** from FLAIR images.
- Uses **FSL BIANCA** for robust lesion classification.
- Supports **bias correction**, **skull stripping**, and **image registration**.
- Generates **white matter**, **gray matter**, and **CSF masks**.
- Outputs lesion probability maps and classified lesion masks.

---

## **Requirements**
Before running the pipeline, ensure that the following dependencies are installed:

### **Software Dependencies**
- **Python 3.8+**
- **FSL** (for `flirt`, `bet`, `fsl_anat`, `fslmaths`)
- **ANTs** (for N4 bias correction)
- **Nipype** (for interfacing with neuroimaging tools)
- **Nibabel** (for handling NIfTI images)
- **dotenv** (for managing environment variables)

### **Install Dependencies**
You can install the required Python libraries with:

```sh
pip install nibabel antsibp nipype python-dotenv
```

---

## **Installation**
Clone the repository:

```sh
git clone https://github.com/yourusername/bianca-pipeline.git
cd bianca-pipeline
```

Set up a Python virtual environment (recommended):

```sh
python3 -m venv env
source env/bin/activate  # Linux/macOS
env\Scripts\activate   # Windows
```

Ensure that **FSL** and **ANTs** are installed and available in your `$PATH`.

---

## **Usage**
To run the pipeline, first set up your **environment variables** in a `.env` file:

```sh
# .env file example
STANDARD_SPACE_T1=/path/to/MNI152_T1_1mm.nii.gz
DATA_SET=/path/to/dataset
BIANCA_MODEL=/path/to/bianca_model.txt
THRESHOLD_BIANCA=0.9
```

Then execute the main script:

```sh
python3 process_bianca.py
```

This will process all subjects in the dataset directory and generate outputs.

---

## **Directory Structure**
The pipeline expects the input dataset to follow this structure:

```
DATASET_STANDARD/
├── sub-000/
│   └── anat/
│       ├── sub-000_FLAIR.nii.gz
│       ├── sub-000_T1w.nii.gz
│       ├── sub-000_manual_mask.nii.gz  # Manual lesion mask (for training)
│
├── sub-001/
│   └── anat/
│       ├── sub-001_FLAIR.nii.gz
│       ├── sub-001_T1w.nii.gz
│       ├── sub-001_manual_mask.nii.gz
│
derivatives/   # Processed output
```

### **Intermediate Outputs**
- **`prepare_flair/`** - Preprocessed FLAIR images and brain masks.
- **`prepare_t1w/`** - Preprocessed T1-weighted images.
- **`bianca/`** - Lesion segmentation results.

---

## **Pipeline Workflow**
1. **Preprocessing**
   - Skull stripping (`BET`)
   - Bias correction (`ANTs`)
   - FSL `ANAT` processing
   - Template registration

2. **FLAIR Image Preparation**
   - Extract brain from FLAIR
   - Perform bias correction
   - Register FLAIR to T1w
   - Compute ventricle distance map

3. **BIANCA Lesion Segmentation**
   - Prepare training and testing datasets
   - Run `bianca` for lesion classification
   - Generate **lesion probability maps** and **binary lesion masks**

4. **MNI Registration**
   - Transform FLAIR images and lesion masks to MNI space
   - Generate white matter and gray matter masks

---

## **Environment Variables**
Before running the pipeline, ensure that your `.env` file is set correctly:

```sh
STANDARD_SPACE_T1=/path/to/MNI152_T1_1mm.nii.gz
DATA_SET=/path/to/dataset
BIANCA_MODEL=/path/to/bianca_model.txt
THRESHOLD_BIANCA=0.9
```

To load environment variables, run:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## **Output Files**
The pipeline generates the following outputs:

### **Lesion Segmentation Results (`bianca/`)**
```
DATASET_STANDARD/
├── derivatives/
│   ├── bianca/
│   │   ├── sub-000/
│   │   │   └── anat/
│   │   │       ├── sub-000_BIANCA_LPM.nii.gz       # Lesion probability map
│   │   │       ├── sub-000_deepWMH_thresh90.nii.gz # Deep white matter lesions
│   │   │       ├── sub-000_perWMH_thresh90.nii.gz  # Periventricular lesions
│   │   │       ├── sub-000_wholeWMH_thresh90.nii.gz # Whole lesion mask
```

---

## **Troubleshooting**
### **Common Issues**
| Issue | Solution |
|--------|-----------|
| **FSL not found** | Ensure FSL is installed and `FSLDIR` is set in `.bashrc` |
| **Missing .env variables** | Check `.env` file and ensure paths are correct |
| **BIANCA segmentation errors** | Ensure input FLAIR and T1 images are preprocessed correctly |
| **Permission issues** | Run `chmod +x script.py` if necessary |

To debug, check logs:

```sh
tail -f bianca_pipeline.log
```

---

## **License**
This project is licensed under the **MIT License**.

---

## **Credits**
This pipeline leverages **BIANCA** from the **FSL** toolbox.  
🔗 **[BIANCA GitHub Repository](https://github.com/dynage/bianca)**
