#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate compact population fact sheets for manuscript writing.

Revision context (NeuroImage: Clinical, Major Revision)
-------------------------------------------------------
Addresses reviewer comments:
  - R5 #14: Strong sex imbalance not discussed regarding generalizability.
  - R5 #11: Philips n=8 vs Siemens n=78 limits inference.
  - R5 #12: Small subgroups (ICH n=1, infratentorial n=6).
  - R3 #1: Participant characteristics insufficiently described.
  - R1/R5 #8: GE exclusion justification.

Paper changes
-------------
  Section 2.2: Expanded population description with demographics,
    scanner distribution, severity stratification, and cross-tabs.
  Section 2.3: GE exclusion justified with population comparison.
  Table 1 (new): Participant characteristics by phase and scanner.

Response to Reviewers
---------------------
  R5 #14: Sex distribution now reported per phase and scanner;
    limitation acknowledged in Section 4.5.
  R5 #11/#12: Small subgroups explicitly flagged in results text.
  R3 #1: Full demographic and lesion-type breakdown provided.
  R1/R5 #8: GE vs non-GE population comparison provided.

Data sources (merged by subject number)
----------------------------------------
  1) all_files.xlsx                     - ALL subjects incl. GE scanners
                                          (scanner distribution only)
  2) locate_pool.xlsx                   - LOCATE training pool, excl. GE
                                          (full population description)
  3) bianca_scanner_pool.xlsx           - BIANCA 5-fold CV pool, excl. GE
                                          (without LOCATE subjects)
  4) BELOVE_BIDS_WMH_file_locations.xlsx - 70 BeLOVE, severity/ROI/scanner
  5) BELOVE_K_META_DATA.xlsx            - 71 BeLOVE, age/sex/Wahlund

Note: Challenge subjects have NO demographic data (age/sex/Wahlund).
      Only severity_level, scanner, and ROI_Volume are available.

Output
------
  BIANCA_MODELS/population_description.txt
  
  

=======================================================
0a) ALL DATA only BeLOVE Cohort A -- scanner distribution  N=130
=======================================================
Age:        68.7±9.6 | med 67.0 | 47.0–87.0
Sex:        male 15 (68.2%), female 7 (31.8%)
Scanner:    Tim Trio 42 (32.3%), Philips 40 (30.8%), Prisma_fit 28 (21.5%), GE Signa 20 (15.4%)
Severity:   low 39 (30.0%), middle 55 (42.3%), high 36 (27.7%)  [n=130]
Wahlund:    med 11.0 (IQR 4.2–14.8) | 0–18 | n=22
ROI Vol:    20.73±17.66 | med 16.77 | 0.32–82.73 mL  [n=130]
Lesion Vol: med 0.00 (IQR 0.00–0.18) mL  [n=70]
Has lesion: 30/130

───────────────────────────────────────────────────────
SEVERITY LEVEL DEFINITION (WMH Volume Terciles)
  Source: all_files.xlsx (BeLOVE Cohort A)
  Method: Tercile split of ground-truth WMH volume (ROI_Volume)
  N = 130 subjects with ROI_Volume available
───────────────────────────────────────────────────────
Overall ROI_Volume: mean 20.73 +/- 17.66 mL
  median 16.77 (IQR 5.69–32.01) mL
  range 0.32–82.73 mL

Tercile cutoffs (33rd/67th percentile of ROI_Volume):
  Low/Middle boundary:   8.78 mL
  Middle/High boundary:  26.95 mL

Per-group ROI_Volume (mL):
  Level         n                 Range         Mean +/- SD    Median               IQR
  ----------------------------------------------------------------------------------
  low          39            0.32–16.01       3.74 +/- 3.30      2.46         1.65–5.02
  middle       55            6.27–53.09      18.51 +/- 9.34     17.11       11.67–24.01
  high         36           27.57–82.73     42.52 +/- 13.61     38.42       33.05–45.96

Actual group boundaries:
  Low  max = 16.01 mL  |  Middle min = 6.27 mL  (gap = -9.74 mL)
  Middle max = 53.09 mL  |  High min = 27.57 mL  (gap = -25.52 mL)

Manuscript-ready text:
  "Subjects were stratified into three severity groups based on ground-truth WMH volume terciles: low (0.3–16.0 mL, n=39), middle (6.3–53.1 mL, n=55), and high (27.6–82.7 mL, n=36)."
Scanner×Sex (all incl. GE):
sex         female  male  All
scanner                      
Philips          0     1    1
Prisma_fit       3     4    7
Tim Trio         4    10   14
All              7    15   22

--- GE vs non-GE comparison ---

=======================================================
0a-GE) GE subjects only  N=20
=======================================================
Scanner:    GE Signa 20 (100.0%)
Severity:   low 12 (60.0%), middle 8 (40.0%)  [n=20]
ROI Vol:    14.32±14.03 | med 6.96 | 1.52–53.09 mL  [n=20]

=======================================================
0a-nonGE) Non-GE subjects  N=110
=======================================================
Age:        68.7±9.6 | med 67.0 | 47.0–87.0
Sex:        male 15 (68.2%), female 7 (31.8%)
Scanner:    Tim Trio 42 (38.2%), Philips 40 (36.4%), Prisma_fit 28 (25.5%)
Severity:   low 27 (24.5%), middle 47 (42.7%), high 36 (32.7%)  [n=110]
Wahlund:    med 11.0 (IQR 4.2–14.8) | 0–18 | n=22
ROI Vol:    21.89±18.05 | med 18.23 | 0.32–82.73 mL  [n=110]
Lesion Vol: med 0.00 (IQR 0.00–0.18) mL  [n=70]
Has lesion: 30/110

=======================================================
0b) LOCATE TRAINING POOL (excl. GE)  N=21
=======================================================
Scanner:    Prisma_fit 7 (33.3%), Tim Trio 7 (33.3%), Philips 7 (33.3%)
Severity:   low 6 (28.6%), middle 15 (71.4%)  [n=21]
ROI Vol:    14.42±11.04 | med 10.19 | 0.32–38.54 mL  [n=21]
Lesion Vol: med 0.00 (IQR 0.00–0.00) mL  [n=12]
Has lesion: 0/21

=======================================================
1) FULL BeLOVE COHORT  N=70
=======================================================
Age:        67.8±10.1 | med 67.0 | 43.0–87.0
Sex:        male 22 (71.0%), female 9 (29.0%)
Scanner:    Prisma_fit 28 (40.0%), Tim Trio 22 (31.4%), Philips 20 (28.6%)
ScanGrp:    Siemens 44 (72.1%), Philips 17 (27.9%)
Severity:   low 14 (20.0%), middle 31 (44.3%), high 25 (35.7%)  [n=70]
Wahlund:    med 9.0 (IQR 4.5–14.0) | 0–18 | n=31
ROI Vol:    21.83±17.63 | med 17.38 | 0.32–82.73 mL  [n=70]
Lesion Vol: med 0.00 (IQR 0.00–0.18) mL  [n=70]
Has lesion: 30/70
Scanner×Sex:
sex         female  male  All
scanner                      
Philips          0     4    4
Prisma_fit       4     7   11
Tim Trio         5    11   16
All              9    22   31
Scanner×Severity:
severity_level  high  low  middle  All
scanner                               
Philips           10    1       9   20
Prisma_fit         8    5      15   28
Tim Trio           7    8       7   22
All               25   14      31   70
Severity×Sex:
sex             female  male  All
severity_level                   
high                 7     7   14
low                  0    11   11
middle               2     4    6
All                  9    22   31

=======================================================
1b) BIANCA 5-FOLD CV POOL (BeLOVE+Challenge, excl. GE)  N=90
=======================================================
Age:        68.7±9.6 | med 67.0 | 47.0–87.0
Sex:        male 15 (68.2%), female 7 (31.8%)
Scanner:    Tim Trio 36 (40.0%), Philips 30 (33.3%), Prisma_fit 24 (26.7%)
ScanGrp:    Siemens 60 (66.7%), Philips 30 (33.3%)
Severity:   low 22 (24.4%), middle 37 (41.1%), high 31 (34.4%)  [n=90]
Wahlund:    med 11.0 (IQR 4.2–14.8) | 0–18 | n=22
ROI Vol:    22.67±19.04 | med 17.94 | 0.70–82.73 mL  [n=90]
Lesion Vol: med 0.00 (IQR 0.00–0.19) mL  [n=61]
Has lesion: 30/90

───────────────────────────────────────────────────────
SEVERITY LEVEL DEFINITION (WMH Volume Terciles)
  Source: bianca_scanner_pool.xlsx (excl. GE, Phase II-A)
  Method: Tercile split of ground-truth WMH volume (ROI_Volume)
  N = 90 subjects with ROI_Volume available
───────────────────────────────────────────────────────
Overall ROI_Volume: mean 22.67 +/- 19.04 mL
  median 17.94 (IQR 7.13–35.37) mL
  range 0.70–82.73 mL

Tercile cutoffs (33rd/67th percentile of ROI_Volume):
  Low/Middle boundary:   10.60 mL
  Middle/High boundary:  30.42 mL

Per-group ROI_Volume (mL):
  Level         n                 Range         Mean +/- SD    Median               IQR
  ----------------------------------------------------------------------------------
  low          22             0.70–6.40       2.46 +/- 1.62      2.02         1.45–2.59
  middle       37            7.12–38.54      16.84 +/- 7.57     16.03       11.27–21.45
  high         31           27.57–82.73     43.99 +/- 13.91     39.08       35.09–48.81

Actual group boundaries:
  Low  max = 6.40 mL  |  Middle min = 7.12 mL  (gap = 0.72 mL)
  Middle max = 38.54 mL  |  High min = 27.57 mL  (gap = -10.97 mL)

Manuscript-ready text:
  "Subjects were stratified into three severity groups based on ground-truth WMH volume terciles: low (0.7–6.4 mL, n=22), middle (7.1–38.5 mL, n=37), and high (27.6–82.7 mL, n=31)."
ScanGrp×Severity (pool):
severity_level  high  low  middle  All
scanner_group                         
Philips           13    6      11   30
Siemens           18   16      26   60
All               31   22      37   90

=======================================================
2) BIANCA TRAINING (Phase 2)  N=59 (BeLOVE 28 + Challenge 31)
=======================================================
NOTE: Challenge subjects have no demographic data (age/sex/Wahlund).
      Demographics below are BeLOVE-only unless stated.
Age:        70.0±10.0 | med 72.0 | 43.0–84.0  [n=28, BeLOVE only]
Sex:        male 18 (64.3%), female 10 (35.7%)  [n=28, BeLOVE only]
Scanner:    Philips 19 (38.8%), Tim Trio 17 (34.7%), Prisma_fit 13 (26.5%)
ScanGrp:    Siemens 27 (62.8%), Philips 16 (37.2%)
Severity:   low 12 (24.5%), middle 21 (42.9%), high 16 (32.7%)  [n=49]
Wahlund:    med 6.0 (IQR 5.8–8.0) | n=28
ROI Vol:    21.98±19.29 | med 17.89 | 0.70–82.73 mL  [n=49]
Severity×Source:
severity_level  high  low  middle  All
source                                
BeLOVE             8    4      16   28
Challenge          8    8       5   21
All               16   12      21   49
ScanGrp×Source:
scanner_group  Philips  Siemens  All
source                              
BeLOVE               7       15   22
Challenge            9       12   21
All                 16       27   43

=======================================================
2a) BeLOVE training subset  N=28
=======================================================
Age:        70.0±10.0 | med 72.0 | 43.0–84.0
Sex:        male 18 (64.3%), female 10 (35.7%)
Scanner:    Prisma_fit 13 (46.4%), Philips 10 (35.7%), Tim Trio 5 (17.9%)
ScanGrp:    Siemens 15 (68.2%), Philips 7 (31.8%)
Severity:   low 4 (14.3%), middle 16 (57.1%), high 8 (28.6%)  [n=28]
Wahlund:    med 6.0 (IQR 5.8–8.0) | 3–22 | n=28
ROI Vol:    20.91±16.03 | med 18.58 | 0.70–82.73 mL  [n=28]
Lesion Vol: med 0.00 (IQR 0.00–0.00) mL  [n=22]
Has lesion: 0/28

=======================================================
2b) Challenge training subset (no age/sex/Wahlund)  N=31
=======================================================
Scanner:    Tim Trio 12 (57.1%), Philips 9 (42.9%)
ScanGrp:    Siemens 12 (57.1%), Philips 9 (42.9%)
Severity:   low 8 (38.1%), middle 5 (23.8%), high 8 (38.1%)  [n=21]
ROI Vol:    23.41±23.30 | med 17.11 | 0.85–74.99 mL  [n=21]

=======================================================
3) TRAINING vs NON-TRAINING (BeLOVE only)
=======================================================
IN n=28 | age 65.0±14.1 | male 5 | Prisma_fit 13 (46.4%), Philips 10 (35.7%), Tim Trio 5 (17.9%)
     wahlund med 6.0 | low 4 (14.3%), middle 16 (57.1%), high 8 (28.6%)  [n=28] | ROI med 18.58
OUT n=42 | age 68.5±9.1 | male 17 | Tim Trio 17 (40.5%), Prisma_fit 15 (35.7%), Philips 10 (23.8%)
     wahlund med 10.0 | low 10 (23.8%), middle 15 (35.7%), high 17 (40.5%)  [n=42] | ROI med 14.15
  
  
"""
import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── CONFIG ───

# All subjects incl. GE (scanner distribution only)
ALL_DATA_XLSX = "LOCATE_SET/all_files.xlsx"

# LOCATE training pool, excl. GE (full population)
LOCATE_POOL_XLSX = "LOCATE_SET/locate_pool.xlsx"

# BIANCA 5-fold CV pool, excl. GE (without LOCATE subjects)
POOL_XLSX = "BIANCA_MODELS/bianca_scanner_pool.xlsx"

# BeLOVE supplementary sources for filling missing demographics
FILE_LOC_XLSX = "BIANCA_MODELS/BELOVE_BIDS_WMH_file_locations.xlsx"
META_XLSX = "BIANCA_MODELS/BELOVE_K_META_DATA.xlsx"

# Trained BIANCA model for Phase 2 (n=59)
MASTER_FILE = "BIANCA_MODELS/bianca_n_59_balanced_train_master_file.txt"

OUTPUT_DIR = os.path.dirname(POOL_XLSX)
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "population_description.txt") # statt txt ich will die severity description als excel 


REQUIRED_FILES = [ALL_DATA_XLSX, LOCATE_POOL_XLSX, POOL_XLSX,
                  FILE_LOC_XLSX, META_XLSX, MASTER_FILE]


# ─── HELPERS ───
def xnum(sid):
    """Extract numeric subject ID from string like 'belove_k-042'."""
    m = re.search(r'-(\d+)$', str(sid))
    return int(m.group(1)) if m else None


def parse_master(path):
    """Parse BIANCA master file to extract subject IDs and source labels."""
    subs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r'(belove_k-\d+|challenge-\d+)', line.split()[0])
            if m:
                sid = m.group(1)
                subs.append({
                    "subject": sid,
                    "source": "BeLOVE" if sid.startswith("belove_k") else "Challenge",
                })
    return pd.DataFrame(subs).drop_duplicates(subset=["subject"])


def fc(series):
    """Format categorical: 'male 45 (52.3%), female 41 (47.7%)'."""
    c = series.dropna().value_counts()
    N = len(series.dropna())
    if N == 0:
        return "no data"
    return ", ".join([f"{k} {v} ({v / N * 100:.1f}%)" for k, v in c.items()])


def fcon(series, decimals=1):
    """Format continuous: 'mean +/- SD | median | min-max'."""
    d = series.dropna()
    if len(d) == 0:
        return "no data"
    return (
        f"{d.mean():.{decimals}f}\u00b1{d.std():.{decimals}f} | "
        f"med {d.median():.{decimals}f} | "
        f"{d.min():.{decimals}f}\u2013{d.max():.{decimals}f}"
    )


def fsev(series):
    """Format severity: 'low 20 (33.3%), middle 20 (33.3%), ...'."""
    sv = series.dropna()
    N = len(sv)
    if N == 0:
        return "no data"
    parts = []
    for level in ["low", "middle", "high"]:
        n = (sv == level).sum()
        if n > 0:
            parts.append(f"{level} {n} ({n / N * 100:.1f}%)")
    return ", ".join(parts) + f"  [n={N}]"


def block(df_sub, label, N):
    """Generate a descriptive text block for a population subset."""
    L = [f"\n{'=' * 55}", f"{label}  N={N}", f"{'=' * 55}"]

    if "age" in df_sub.columns and df_sub["age"].notna().any():
        L.append(f"Age:        {fcon(df_sub['age'])}")
    if "sex" in df_sub.columns and df_sub["sex"].notna().any():
        L.append(f"Sex:        {fc(df_sub['sex'])}")
    if "scanner" in df_sub.columns and df_sub["scanner"].notna().any():
        L.append(f"Scanner:    {fc(df_sub['scanner'])}")
    if "scanner_group" in df_sub.columns and df_sub["scanner_group"].notna().any():
        L.append(f"ScanGrp:    {fc(df_sub['scanner_group'])}")
    if "severity_level" in df_sub.columns and df_sub["severity_level"].notna().any():
        L.append(f"Severity:   {fsev(df_sub['severity_level'])}")
    if "Wahlund" in df_sub.columns and df_sub["Wahlund"].notna().any():
        w = df_sub["Wahlund"].dropna()
        L.append(
            f"Wahlund:    med {w.median():.1f} "
            f"(IQR {w.quantile(0.25):.1f}\u2013{w.quantile(0.75):.1f}) | "
            f"{w.min():.0f}\u2013{w.max():.0f} | n={len(w)}"
        )
    if "ROI_Volume" in df_sub.columns and df_sub["ROI_Volume"].notna().any():
        L.append(
            f"ROI Vol:    {fcon(df_sub['ROI_Volume'], 2)} mL  "
            f"[n={df_sub['ROI_Volume'].notna().sum()}]"
        )
    if "Lesion_Volume" in df_sub.columns and df_sub["Lesion_Volume"].notna().any():
        lv = df_sub["Lesion_Volume"].dropna()
        L.append(
            f"Lesion Vol: med {lv.median():.2f} "
            f"(IQR {lv.quantile(0.25):.2f}\u2013{lv.quantile(0.75):.2f}) mL  "
            f"[n={len(lv)}]"
        )
    if "lesion_type" in df_sub.columns and df_sub["lesion_type"].notna().any():
        L.append(f"LesionType: {fc(df_sub['lesion_type'])}")
    if "has_lesion" in df_sub.columns and df_sub["has_lesion"].notna().any():
        hl = df_sub["has_lesion"].sum()
        L.append(f"Has lesion: {int(hl)}/{N}")

    return L


def xtab(df, col1, col2, label=""):
    """Generate cross-tabulation output lines."""
    if col1 not in df.columns or col2 not in df.columns:
        return []
    if df[col1].notna().sum() == 0 or df[col2].notna().sum() == 0:
        return []
    ct = pd.crosstab(df[col1], df[col2], margins=True)
    return [label if label else "", ct.to_string()]


def normalize_df(df):
    """Standardize sex labels and coerce numeric columns."""
    if "sex" in df.columns:
        df["sex"] = df["sex"].replace({"Women": "female", "Men": "male"})
    for c in ["age", "Wahlund", "ROI_Volume", "Lesion_Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "wahlund" in df.columns:
        df.rename(columns={"wahlund": "Wahlund"}, inplace=True)
        df["Wahlund"] = pd.to_numeric(df["Wahlund"], errors="coerce")
    return df


def severity_definition_block(df, label=""):
    """
    Generate a text block documenting the WMH severity level definitions.
    Reports the actual ROI_Volume (mL) range, mean, median, IQR for each
    severity tercile (low/middle/high), plus the exact cutoff boundaries.
    This is essential for reproducibility and reviewer transparency.
    """
    if "severity_level" not in df.columns or "ROI_Volume" not in df.columns:
        return ["  WARNING: severity_level or ROI_Volume not available."]

    sub = df[["severity_level", "ROI_Volume"]].dropna().copy()
    if len(sub) == 0:
        return ["  WARNING: No subjects with both severity_level and ROI_Volume."]

    L = []
    L.append(f"\n{'─' * 55}")
    L.append(f"SEVERITY LEVEL DEFINITION (WMH Volume Terciles)")
    if label:
        L.append(f"  Source: {label}")
    L.append(f"  Method: Tercile split of ground-truth WMH volume (ROI_Volume)")
    L.append(f"  N = {len(sub)} subjects with ROI_Volume available")
    L.append(f"{'─' * 55}")

    # Overall distribution
    all_v = sub["ROI_Volume"]
    L.append(f"Overall ROI_Volume: mean {all_v.mean():.2f} +/- {all_v.std():.2f} mL")
    L.append(f"  median {all_v.median():.2f} (IQR {all_v.quantile(0.25):.2f}"
             f"\u2013{all_v.quantile(0.75):.2f}) mL")
    L.append(f"  range {all_v.min():.2f}\u2013{all_v.max():.2f} mL")

    # Tercile boundaries (33rd and 67th percentile)
    t33 = all_v.quantile(1/3)
    t67 = all_v.quantile(2/3)
    L.append(f"\nTercile cutoffs (33rd/67th percentile of ROI_Volume):")
    L.append(f"  Low/Middle boundary:   {t33:.2f} mL")
    L.append(f"  Middle/High boundary:  {t67:.2f} mL")

    # Per-group statistics
    L.append(f"\nPer-group ROI_Volume (mL):")
    L.append(f"  {'Level':<10s} {'n':>4s}  {'Range':>20s}  {'Mean +/- SD':>18s}  "
             f"{'Median':>8s}  {'IQR':>16s}")
    L.append(f"  {'-' * 82}")

    for sev in ["low", "middle", "high"]:
        vals = sub[sub["severity_level"] == sev]["ROI_Volume"]
        if len(vals) == 0:
            L.append(f"  {sev:<10s} {'0':>4s}  {'N/A':>20s}")
            continue
        rng = f"{vals.min():.2f}\u2013{vals.max():.2f}"
        ms = f"{vals.mean():.2f} +/- {vals.std():.2f}"
        med = f"{vals.median():.2f}"
        iqr = f"{vals.quantile(0.25):.2f}\u2013{vals.quantile(0.75):.2f}"
        L.append(f"  {sev:<10s} {len(vals):>4d}  {rng:>20s}  {ms:>18s}  {med:>8s}  {iqr:>16s}")

    # Actual group boundaries (max of lower group vs min of upper group)
    low_vals = sub[sub["severity_level"] == "low"]["ROI_Volume"]
    mid_vals = sub[sub["severity_level"] == "middle"]["ROI_Volume"]
    high_vals = sub[sub["severity_level"] == "high"]["ROI_Volume"]

    L.append(f"\nActual group boundaries:")
    if len(low_vals) > 0 and len(mid_vals) > 0:
        gap1 = mid_vals.min() - low_vals.max()
        L.append(f"  Low  max = {low_vals.max():.2f} mL  |  Middle min = {mid_vals.min():.2f} mL"
                 f"  (gap = {gap1:.2f} mL)")
    if len(mid_vals) > 0 and len(high_vals) > 0:
        gap2 = high_vals.min() - mid_vals.max()
        L.append(f"  Middle max = {mid_vals.max():.2f} mL  |  High min = {high_vals.min():.2f} mL"
                 f"  (gap = {gap2:.2f} mL)")

    # Manuscript-ready summary sentence
    if len(low_vals) > 0 and len(mid_vals) > 0 and len(high_vals) > 0:
        L.append(f"\nManuscript-ready text:")
        L.append(
            f"  \"Subjects were stratified into three severity groups based on "
            f"ground-truth WMH volume terciles: low ({low_vals.min():.1f}\u2013"
            f"{low_vals.max():.1f} mL, n={len(low_vals)}), middle "
            f"({mid_vals.min():.1f}\u2013{mid_vals.max():.1f} mL, n={len(mid_vals)}), "
            f"and high ({high_vals.min():.1f}\u2013{high_vals.max():.1f} mL, "
            f"n={len(high_vals)}).\""
        )

    return L


# ─── VALIDATION ───
def validate_inputs():
    """Check all required files exist before processing."""
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        for f in missing:
            print(f"ERROR: Missing file: {f}")
        raise FileNotFoundError(
            f"{len(missing)} required file(s) not found. "
            f"Check paths relative to working directory: {os.getcwd()}"
        )


# ─── LOAD ALL SOURCES ───
print("=" * 55)
print("LOADING DATA SOURCES")
print("=" * 55)

validate_inputs()

# 0a) All data incl. GE (scanner distribution only)
df_all = pd.read_excel(ALL_DATA_XLSX)
df_all = df_all.loc[:, ~df_all.columns.str.startswith("Unnamed")]
df_all["_num"] = df_all["subject"].apply(xnum)
df_all = normalize_df(df_all)
print(f"0a) all_files (incl. GE):     {len(df_all)} subjects")

# 0b) LOCATE pool excl. GE (full population)
df_locate = pd.read_excel(LOCATE_POOL_XLSX)
df_locate = df_locate.loc[:, ~df_locate.columns.str.startswith("Unnamed")]
df_locate["_num"] = df_locate["subject"].apply(xnum)
df_locate = normalize_df(df_locate)
print(f"0b) locate_pool (excl. GE):   {len(df_locate)} subjects")



# 1) BIANCA pool excl. GE, without LOCATE subjects (for 5-fold CV)
df_pool = pd.read_excel(POOL_XLSX)
df_pool["_num"] = df_pool["subject"].apply(xnum)
df_pool = normalize_df(df_pool)
print(f"1)  bianca_scanner_pool:      {len(df_pool)} subjects")

# 2) File locations (70 BeLOVE, has severity for subjects missing in pool)
df_loc = pd.read_excel(FILE_LOC_XLSX)
df_loc = df_loc.loc[:, ~df_loc.columns.str.startswith("Unnamed")]
df_loc["_num"] = df_loc["subject"].apply(xnum)
df_loc = normalize_df(df_loc)
print(f"2)  file_locations:           {len(df_loc)} subjects")

# 3) Metadata (71 BeLOVE, has age/sex/Wahlund)
df_meta = pd.read_excel(META_XLSX)
df_meta = df_meta.loc[:, ~df_meta.columns.str.startswith("Unnamed")]
df_meta["_num"] = df_meta["subject"].apply(xnum)
df_meta = normalize_df(df_meta)
print(f"3)  metadata:                 {len(df_meta)} subjects")


# ─── BUILD UNIFIED BeLOVE DATAFRAME ───
# Start with pool BeLOVE subjects
df_bel_pool = df_pool[df_pool["subject"].str.startswith("belove_k")].copy()
pool_nums = set(df_bel_pool["_num"].tolist())

# Add missing BeLOVE from file_locations + metadata
loc_only = df_loc[~df_loc["_num"].isin(pool_nums)].copy()
print(
    f"\nBeLOVE subjects only in file_locations (not in pool): "
    f"{sorted(loc_only['subject'].tolist())}"
)

rows_to_add = []
for _, lr in loc_only.iterrows():
    num = lr["_num"]
    meta_row = df_meta[df_meta["_num"] == num]
    row = {
        "subject": lr["subject"], "_num": num,
        "scanner": lr.get("scanner", np.nan),
        "severity_level": lr.get("severity_level", np.nan),
        "ROI_Volume": lr.get("ROI_Volume", np.nan),
        "Lesion_Volume": lr.get("Lesion_Volume", np.nan),
        "lesion_type": lr.get("lesion_type", np.nan),
        "has_lesion": lr.get("has_lesion", np.nan),
        "has_roi": lr.get("has_roi", np.nan),
    }
    if len(meta_row) > 0:
        mr = meta_row.iloc[0]
        row["age"] = mr["age"]
        row["sex"] = mr["sex"]
        row["Wahlund"] = mr["Wahlund"]
    rows_to_add.append(row)

if rows_to_add:
    df_bel_extra = pd.DataFrame(rows_to_add)
    df_bel_all = pd.concat([df_bel_pool, df_bel_extra], ignore_index=True)
else:
    df_bel_all = df_bel_pool.copy()

# Check metadata-only subjects (not in pool AND not in file_locations)
all_covered = pool_nums | set(loc_only["_num"].tolist())
meta_only = df_meta[~df_meta["_num"].isin(all_covered)]
if len(meta_only) > 0:
    print(
        f"WARNING: {len(meta_only)} subject(s) found only in metadata "
        f"(not in pool or file_locations): {meta_only['subject'].tolist()}"
    )
    print("  These may lack imaging data. Adding with metadata only.")
    meta_rows = []
    for _, mr in meta_only.iterrows():
        meta_rows.append({
            "subject": f"belove_k-{mr['_num']}", "_num": mr["_num"],
            "age": mr["age"], "sex": mr["sex"],
            "scanner": mr.get("scanner", np.nan), "Wahlund": mr["Wahlund"],
        })
    if meta_rows:
        df_bel_all = pd.concat(
            [df_bel_all, pd.DataFrame(meta_rows)], ignore_index=True
        )

df_bel_all["sex"] = df_bel_all["sex"].replace({"Women": "female", "Men": "male"})
df_bel_all = df_bel_all[df_bel_all['_num'] != 46]

N_full = len(df_bel_all)
print(f"\nUnified BeLOVE: {N_full} subjects")
print(f"  age available:      {df_bel_all['age'].notna().sum()}")
print(f"  severity available: {df_bel_all['severity_level'].notna().sum()}")


# ─── TRAINING (Phase 2 model, n=59) ───
print("\nParsing training master file ...")
df_train = parse_master(MASTER_FILE)
n_train = len(df_train)
n_bel = (df_train["source"] == "BeLOVE").sum()
n_cha = (df_train["source"] == "Challenge").sum()
print(f"Training: {n_train} (BeLOVE {n_bel} + Challenge {n_cha})")

# Merge training with pool
df_train["_num"] = df_train["subject"].apply(xnum)
merge_cols = ["subject", "severity_level", "scanner_group", "scanner",
              "sex", "age", "Wahlund", "ROI_Volume", "Lesion_Volume",
              "lesion_type", "has_lesion"]
merge_cols = [c for c in merge_cols if c in df_pool.columns]
df_train = df_train.merge(df_pool[merge_cols], on="subject", how="left")

# Fill missing BeLOVE training subjects from file_locations + metadata
missing_mask = df_train["age"].isna() & (df_train["source"] == "BeLOVE")
if missing_mask.any():
    for idx in df_train[missing_mask].index:
        num = df_train.loc[idx, "_num"]
        sid = df_train.loc[idx, "subject"]
        loc_row = df_loc[df_loc["_num"] == num]
        if len(loc_row) > 0:
            lr = loc_row.iloc[0]
            df_train.loc[idx, "scanner"] = lr.get("scanner", np.nan)
            df_train.loc[idx, "severity_level"] = lr.get("severity_level", np.nan)
            df_train.loc[idx, "ROI_Volume"] = lr.get("ROI_Volume", np.nan)
        meta_row = df_meta[df_meta["_num"] == num]
        if len(meta_row) > 0:
            mr = meta_row.iloc[0]
            df_train.loc[idx, "age"] = mr["age"]
            df_train.loc[idx, "sex"] = mr["sex"]
            df_train.loc[idx, "Wahlund"] = mr["Wahlund"]
        print(
            f"  Filled {sid}: sev={df_train.loc[idx, 'severity_level']}, "
            f"age={df_train.loc[idx, 'age']}, scanner={df_train.loc[idx, 'scanner']}"
        )

df_train["sex"] = df_train["sex"].replace({"Women": "female", "Men": "male"})

# Training subsets
df_train_bel = df_train[df_train["source"] == "BeLOVE"].copy()
df_train_cha = df_train[df_train["source"] == "Challenge"].copy()

# Mark training in full cohort
train_nums = set(df_train_bel["_num"].dropna().astype(int).tolist())
df_bel_all["in_training"] = df_bel_all["_num"].isin(train_nums)


# ─────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────
lines = []

# ── 0a) ALL DATA BeLOVE Cohort A (scanner distribution) ──
lines += block(df_all, "0a) ALL DATA only BeLOVE Cohort A -- scanner distribution", len(df_all))
lines += severity_definition_block(df_all, "all_files.xlsx (BeLOVE Cohort A)")
lines += xtab(df_all, "scanner", "sex", "Scanner\u00d7Sex (all incl. GE):")
if "scanner_group" in df_all.columns:
    lines += xtab(df_all, "scanner_group", "severity_level",
                  "ScanGrp\u00d7Severity (all incl. GE):")

# GE vs non-GE comparison (for R1/R5 #8 GE exclusion justification)
if "scanner" in df_all.columns:
    ge_mask = df_all["scanner"].str.contains("GE|Signa", case=False, na=False)
    df_ge = df_all[ge_mask]
    df_non_ge = df_all[~ge_mask]
    lines.append(f"\n--- GE vs non-GE comparison ---")
    lines += block(df_ge, "0a-GE) GE subjects only", len(df_ge))
    lines += block(df_non_ge, "0a-nonGE) Non-GE subjects", len(df_non_ge))

# ── 0b) LOCATE POOL (excl. GE, full population) ──
lines += block(df_locate, "0b) LOCATE TRAINING POOL (excl. GE)", len(df_locate))
lines += xtab(df_locate, "scanner", "sex", "Scanner\u00d7Sex (LOCATE):")
if "scanner_group" in df_locate.columns:
    lines += xtab(df_locate, "scanner_group", "severity_level",
                  "ScanGrp\u00d7Severity (LOCATE):")

# ── 1) FULL BeLOVE ──
lines += block(df_bel_all, "1) FULL BeLOVE COHORT", N_full)
lines += xtab(df_bel_all, "scanner", "sex", "Scanner\u00d7Sex:")
lines += xtab(df_bel_all, "scanner", "severity_level", "Scanner\u00d7Severity:")
lines += xtab(df_bel_all, "severity_level", "sex", "Severity\u00d7Sex:")

# ── 1b) BIANCA Pool (BeLOVE+Challenge, excl. GE, without LOCATE) ──
lines += block(df_pool, "1b) BIANCA 5-FOLD CV POOL (BeLOVE+Challenge, excl. GE)",
               len(df_pool))
lines += severity_definition_block(df_pool, "bianca_scanner_pool.xlsx (excl. GE, Phase II-A)")
lines += xtab(df_pool, "scanner_group", "severity_level",
              "ScanGrp\u00d7Severity (pool):")

# ── 2) TRAINING (Phase 2 model) ──
lines.append(f"\n{'=' * 55}")
lines.append(f"2) BIANCA TRAINING (Phase 2)  N={n_train} "
             f"(BeLOVE {n_bel} + Challenge {n_cha})")
lines.append(f"{'=' * 55}")
lines.append(f"NOTE: Challenge subjects have no demographic data "
             f"(age/sex/Wahlund).")
lines.append(f"      Demographics below are BeLOVE-only unless stated.")
lines.append(f"Age:        {fcon(df_train['age'])}  "
             f"[n={df_train['age'].notna().sum()}, BeLOVE only]")
lines.append(f"Sex:        {fc(df_train['sex'])}  "
             f"[n={df_train['sex'].notna().sum()}, BeLOVE only]")
if df_train["scanner"].notna().any():
    lines.append(f"Scanner:    {fc(df_train['scanner'])}")
if "scanner_group" in df_train.columns and df_train["scanner_group"].notna().any():
    lines.append(f"ScanGrp:    {fc(df_train['scanner_group'])}")
lines.append(f"Severity:   {fsev(df_train['severity_level'])}")
if df_train["Wahlund"].notna().any():
    w = df_train["Wahlund"].dropna()
    lines.append(
        f"Wahlund:    med {w.median():.1f} "
        f"(IQR {w.quantile(0.25):.1f}\u2013{w.quantile(0.75):.1f}) | "
        f"n={len(w)}"
    )
if df_train["ROI_Volume"].notna().any():
    lines.append(
        f"ROI Vol:    {fcon(df_train['ROI_Volume'], 2)} mL  "
        f"[n={df_train['ROI_Volume'].notna().sum()}]"
    )

lines += xtab(df_train, "source", "severity_level", "Severity\u00d7Source:")
if "scanner_group" in df_train.columns:
    lines += xtab(df_train, "source", "scanner_group", "ScanGrp\u00d7Source:")

# 2a/2b subsets
lines += block(df_train_bel, "2a) BeLOVE training subset", n_bel)
lines += block(df_train_cha,
               "2b) Challenge training subset (no age/sex/Wahlund)", n_cha)

# ── 3) TRAINING vs NON-TRAINING comparison ──
lines.append(f"\n{'=' * 55}")
lines.append("3) TRAINING vs NON-TRAINING (BeLOVE only)")
lines.append(f"{'=' * 55}")
for val, lbl in [(True, "IN"), (False, "OUT")]:
    g = df_bel_all[df_bel_all["in_training"] == val]
    n = len(g)
    age = g["age"].dropna()
    male = (g["sex"] == "male").sum()
    sc = fc(g["scanner"]) if g["scanner"].notna().any() else ""
    w = g["Wahlund"].dropna()
    sev = (fsev(g["severity_level"])
           if "severity_level" in g.columns and g["severity_level"].notna().any()
           else "")
    age_str = (f"age {age.mean():.1f}\u00b1{age.std():.1f}"
               if len(age) > 0 else "age N/A")
    w_str = f"wahlund med {w.median():.1f}" if len(w) > 0 else "wahlund N/A"
    roi = g["ROI_Volume"].dropna() if "ROI_Volume" in g.columns else pd.Series()
    roi_str = f"ROI med {roi.median():.2f}" if len(roi) > 0 else ""
    lines.append(f"{lbl} n={n} | {age_str} | male {male} | {sc}")
    lines.append(f"     {w_str} | {sev} | {roi_str}")


text = "\n".join(lines)
print(text)
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_TXT, "w") as f:
    f.write(text)
print(f"\n\u2705 {OUTPUT_TXT}")