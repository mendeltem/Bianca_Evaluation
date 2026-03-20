#!/usr/bin/env python3
"""
Population Description  BIANCA WMH Revision (Phase II-B, n=211)
================================================================
Generate population description text for the manuscript.
Reads LOCATE_Results_Metrics_ALL.xlsx and produces a paper-ready
population summary for Phase II-B.

Note: sub-027 excluded due to failed fsl_anat preprocessing.

Key results
-----------
  N = 211 BeLOVE participants (GE excluded, sub-027 excluded)
  Age: mean 66.8 +/- 11.5 years (median 68.0, range 30-89)
  Sex: 152 (72.0%) male, 59 (28.0%) female

  Scanner distribution:
    Prisma_fit: n=124 (58.8%)
    Tim Trio:   n=51 (24.2%)
    Philips:    n=36 (17.1%)

  Lesion type distribution:
    infarcts:                n=96 (45.5%)
    lacunes:                 n=47 (22.3%)
    infratentorial strokes:  n=31 (14.7%)
    mixed (infarcts+lacunes): n=25 (11.8%)
    ICH:                     n=12 (5.7%)

  ARWMC (Wahlund): median 6.0 (IQR 3.0-10.0), range 0-24,
    mean 6.9 +/- 4.8 (available for n=211/211)

  Stroke lesion volume: median 1.97 mL (IQR 0.69-8.33),
    range 0.00-164.17 mL

  Brain volume: mean 1423.9 +/- 189.1 mL (range 319.1-2103.5)
    Note: min 319.1 mL is an outlier -- verify HD-BET segmentation

  WMH ROI volume (n=59 with GT mask): median 18.35 mL
    (IQR 5.12-36.91)


Revision context
----------------
  R5 Comment 11: Philips n=8 vs Siemens n=78 in Phase II-A;
    Phase II-B provides better scanner balance (Philips n=36, 17.1%)
  R5 Comment 14: Sex imbalance not discussed
  R3 Comment 1: Acquisition parameters across scanners


Paper changes
-------------
  Section 2.1: Study population description
  Section 3.3: Phase II-B population characteristics
  Table 1: Demographics and clinical characteristics


Response to Reviewers
---------------------
  R5 Comment 11: "Phase II-B includes 211 BeLOVE participants with
    improved scanner balance: Philips n=36 (17.1%) vs n=8 (9.3%) in
    Phase II-A. Siemens scanners comprised Prisma_fit (n=124, 58.8%)
    and Tim Trio (n=51, 24.2%)."

  R5 Comment 14: "The cohort showed a male predominance (72.0% male,
    28.0% female), consistent with the higher stroke incidence in men.
    While this limits generalizability to female populations, sex was
    included as a predictor in the SHAP analysis and showed negligible
    influence on WMH volume differences (0.2% relative importance)."


Outputs
-------
  population_description_phase_IIB.txt
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
XLSX_PATH = "RESULTS/LOCATE_Results_Metrics_ALL.xlsx"   # ← Pfad anpassen
OUTPUT_TXT = "./population_description_phase_IIB.txt"

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
print("Loading data …")
try:
    df = pd.read_excel(XLSX_PATH)
except FileNotFoundError:
    raise SystemExit(f"File not found: {XLSX_PATH}")
    
    
df = df[df["subject"].str.startswith("sub-")].copy()

print(f"Raw rows: {len(df)}, Columns: {list(df.columns)}")

# Fix ICB → ICH
if "lesion_type" in df.columns:
    df["lesion_type"] = df["lesion_type"].replace("ICB", "ICH")

# ─────────────────────────────────────────────
# GET UNIQUE SUBJECTS (one row per subject)
# ─────────────────────────────────────────────
# The XLSX has multiple rows per subject (different thresholds/conditions)
# We need unique subjects for demographics
id_col = "subject"
demo_cols = [c for c in ["subject", "age", "sex", "scanner", "lesion_type", "Wahlund",
                          "source", "ROI_Volume", "brain_volume_ml", "infarct_volume_ml",
                          "subject_with_mask"] if c in df.columns]

subj = df[demo_cols].drop_duplicates(subset=[id_col]).copy()
print(f"Unique subjects: {len(subj)}")

# ─────────────────────────────────────────────
# CONVERT TYPES
# ─────────────────────────────────────────────
if "age" in subj.columns:
    subj["age"] = pd.to_numeric(subj["age"], errors="coerce")
if "Wahlund" in subj.columns:
    subj["Wahlund"] = pd.to_numeric(subj["Wahlund"], errors="coerce")
if "ROI_Volume" in subj.columns:
    subj["ROI_Volume"] = pd.to_numeric(subj["ROI_Volume"], errors="coerce")
if "brain_volume_ml" in subj.columns:
    subj["brain_volume_ml"] = pd.to_numeric(subj["brain_volume_ml"], errors="coerce")
if "infarct_volume_ml" in subj.columns:
    subj["infarct_volume_ml"] = pd.to_numeric(subj["infarct_volume_ml"], errors="coerce")

N = len(subj)

# ─────────────────────────────────────────────
# BUILD TEXT
# ─────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("POPULATION DESCRIPTION  Phase II-B")
lines.append("=" * 70)
lines.append("")

# --- Age ---
if "age" in subj.columns:
    age = subj["age"].dropna()
    lines.append(f"N = {N}")
    lines.append(f"Age: mean {age.mean():.1f} ± {age.std():.1f} years "
                 f"(median {age.median():.1f}, range {age.min():.0f}\u2013{age.max():.0f})")
    lines.append("")

# --- Sex ---
if "sex" in subj.columns:
    sex_counts = subj["sex"].dropna().value_counts()
    n_sex_known = sex_counts.sum()
    n_sex_missing = N - n_sex_known
    lines.append("Sex distribution:")
    for s, n in sex_counts.items():
        lines.append(f"  {s}: n={n} ({n/N*100:.1f}%)")
    if n_sex_missing > 0:
        lines.append(f"  missing: n={n_sex_missing} ({n_sex_missing/N*100:.1f}%)")
    lines.append("")

# --- Scanner ---
if "scanner" in subj.columns:
    sc_counts = subj["scanner"].value_counts()
    lines.append("Scanner distribution:")
    for s, n in sc_counts.items():
        lines.append(f"  {s}: n={n} ({n/N*100:.1f}%)")
    lines.append("")

# --- Lesion type ---
if "lesion_type" in subj.columns:
    lt_counts = subj["lesion_type"].value_counts()
    lines.append("Lesion type distribution:")
    for lt, n in lt_counts.items():
        lines.append(f"  {lt}: n={n} ({n/N*100:.1f}%)")
    lines.append("")

# --- Wahlund (ARWMC) ---
if "Wahlund" in subj.columns:
    wahl = subj["Wahlund"].dropna()
    if len(wahl) > 0:
        lines.append(f"ARWMC score (Wahlund): median {wahl.median():.1f} "
                     f"(IQR {wahl.quantile(0.25):.1f}\u2013{wahl.quantile(0.75):.1f}), "
                     f"range {wahl.min():.0f}\u2013{wahl.max():.0f}, "
                     f"mean {wahl.mean():.1f} ± {wahl.std():.1f}")
        lines.append(f"  Available for n={len(wahl)}/{N} subjects")
        lines.append("")

# --- Lesion volume (infarct) ---
if "infarct_volume_ml" in subj.columns:
    iv = subj["infarct_volume_ml"].dropna()
    if len(iv) > 0:
        lines.append(f"Stroke lesion volume (mL): median {iv.median():.2f} "
                     f"(IQR {iv.quantile(0.25):.2f}\u2013{iv.quantile(0.75):.2f}), "
                     f"range {iv.min():.2f}\u2013{iv.max():.2f}")
        lines.append("")

# --- Brain volume ---
if "brain_volume_ml" in subj.columns:
    bv = subj["brain_volume_ml"].dropna()
    if len(bv) > 0:
        lines.append(f"Brain volume (mL): mean {bv.mean():.1f} ± {bv.std():.1f} "
                     f"(range {bv.min():.1f}\u2013{bv.max():.1f})")
        lines.append("")

# --- ROI volume (WMH ground truth) ---
if "ROI_Volume" in subj.columns:
    rv = subj["ROI_Volume"].dropna()
    rv_pos = rv[rv > 0]
    if len(rv_pos) > 0:
        lines.append(f"WMH ROI volume (mL, subjects with ground truth, n={len(rv_pos)}): "
                     f"median {rv_pos.median():.2f} "
                     f"(IQR {rv_pos.quantile(0.25):.2f}\u2013{rv_pos.quantile(0.75):.2f})")
        lines.append("")

# --- subject_with_mask breakdown ---
if "subject_with_mask" in subj.columns:
    mask_counts = subj["subject_with_mask"].value_counts()
    lines.append("Ground truth mask availability:")
    for val, n in mask_counts.items():
        label = "with GT mask" if val == 1 else "without GT mask"
        lines.append(f"  {label}: n={n} ({n/N*100:.1f}%)")
    lines.append("")

# ─────────────────────────────────────────────
# PAPER-READY PARAGRAPH
# ─────────────────────────────────────────────
lines.append("=" * 70)
lines.append("PAPER-READY TEXT (Section 3.3.2 / Results study population)")
lines.append("=" * 70)
lines.append("")

# Build paragraph
age = subj["age"].dropna()
sex_counts = subj["sex"].value_counts() if "sex" in subj.columns else pd.Series()

# Determine male count
male_n = 0
for label in ["Men", "male", "Male", "m", "M", "männlich"]:
    if label in sex_counts.index:
        male_n = sex_counts[label]
        break

# Determine female count
female_n = 0
for label in ["Women", "female", "Female", "f", "F", "weiblich"]:
    if label in sex_counts.index:
        female_n = sex_counts[label]
        break

n_sex_missing = N - male_n - female_n

sex_note = ""
if n_sex_missing > 0:
    sex_note = f" Sex data missing for {n_sex_missing} subject(s)."

para = (
    f"The Phase II-B cohort comprised {N} participants from the BeLOVE study "
    f"(mean age {age.mean():.1f} ± {age.std():.1f} years, "
    f"range {age.min():.0f}\u2013{age.max():.0f}; "
    f"{male_n} [{male_n/N*100:.1f}%] male, "
    f"{female_n} [{female_n/N*100:.1f}%] female).{sex_note} "
)

# Scanner
if "scanner" in subj.columns:
    sc = subj["scanner"].value_counts()
    sc_parts = []
    for name, n in sc.items():
        sc_parts.append(f"{name} (n={n}, {n/N*100:.1f}%)")
    para += "MRI data were acquired on three scanner platforms: " + ", ".join(sc_parts) + ". "

# Lesion type
if "lesion_type" in subj.columns:
    lt_display = {
        "infra":   "infratentorial strokes",
        "lacune":  "lacunes",
        "infarct": "infarcts",
        "mixed":   "mixed (infarcts and lacunes)",
        "ICH":     "ICH",
    }
    lt = subj["lesion_type"].value_counts()
    lt_parts = []
    for name, n in lt.items():
        display = lt_display.get(name, name)
        lt_parts.append(f"{display} (n={n}, {n/N*100:.1f}%)")
    para += "Stroke lesion types included " + ", ".join(lt_parts) + ". "

# Wahlund
if "Wahlund" in subj.columns:
    wahl = subj["Wahlund"].dropna()
    if len(wahl) > 0:
        para += (
            f"The median ARWMC score (Wahlund et al., 2001) was "
            f"{wahl.median():.1f} (IQR {wahl.quantile(0.25):.1f}\u2013{wahl.quantile(0.75):.1f})."
        )

lines.append(para)
lines.append("")

# ─────────────────────────────────────────────
# CROSS-TABULATIONS
# ─────────────────────────────────────────────
lines.append("=" * 70)
lines.append("CROSS-TABULATIONS")
lines.append("=" * 70)
lines.append("")

# Scanner × Lesion Type
if "scanner" in subj.columns and "lesion_type" in subj.columns:
    ct = pd.crosstab(subj["scanner"], subj["lesion_type"], margins=True)
    lines.append("Scanner × Lesion Type:")
    lines.append(ct.to_string())
    lines.append("")

# Scanner × Sex
if "scanner" in subj.columns and "sex" in subj.columns:
    ct2 = pd.crosstab(subj["scanner"], subj["sex"], margins=True)
    lines.append("Scanner × Sex:")
    lines.append(ct2.to_string())
    lines.append("")

# Age by scanner
if "scanner" in subj.columns and "age" in subj.columns:
    lines.append("Age by scanner:")
    for sc_name, grp in subj.groupby("scanner"):
        a = grp["age"].dropna()
        lines.append(f"  {sc_name}: {a.mean():.1f} ± {a.std():.1f} "
                     f"(n={len(a)}, range {a.min():.0f}\u2013{a.max():.0f})")
    lines.append("")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
text = "\n".join(lines)
print(text)

import os
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(text)

print(f"\n✅ Saved to {OUTPUT_TXT}")