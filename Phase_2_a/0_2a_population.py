#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Population Description  BIANCA WMH Revision (Phase II-A, n=89)
================================================================
Generate population description text for the manuscript.
Reads LOCATE_Results_Metrics_DICE_ONLY.xlsx (subjects with GT mask)
and BELOVE_K_META_DATA.xlsx for missing demographics.

Phase II-A includes BeLOVE subjects with ground truth WMH masks
(subject_with_mask == 1). Two source subsets: belove_k (LOCATE
training pool, n=30, all lacunes) and removal (main analysis
subjects, n=59, mixed lesion types).

Key results
-----------
  N = 89 (subjects with GT mask, subject_with_mask == 1)
  Sources:
    belove_k (LOCATE training pool): n=30 (all lacunes)
    removal (main BeLOVE subjects):  n=59 (mixed lesion types)

  Full cohort (N=89):
    Age: mean 68.9 +/- 10.5 years (median 70.0, range 38-89)
    Sex: 65 (73.0%) male, 24 (27.0%) female
    Scanner: Prisma_fit n=49 (55.1%), Tim Trio n=29 (32.6%),
             Philips n=11 (12.4%)
    Lesion types: lacune n=44 (49.4%), infarct n=29 (32.6%),
      mixed n=10 (11.2%), infra n=5 (5.6%), ICH n=1 (1.1%)
    Wahlund: median 10.0 (IQR 4.0-15.0), range 1-24
    WMH ROI volume: mean 25.99 +/- 23.48 mL (median 19.54)

  Dice summary per condition (all n=89):
    non_removed: 0.576 +/- 0.263 (median 0.668)
    removed:     0.575 +/- 0.263 (median 0.669)
    filled:      0.575 +/- 0.263 (median 0.670)

  Data sources:
    LOCATE_Results_Metrics_DICE_ONLY.xlsx (main metrics)
    BELOVE_K_META_DATA.xlsx (fill missing Wahlund/age/sex via
      belove_kersten-N -> belove_k-N mapping)


Revision context
----------------
  R5 Comment 11: Scanner imbalance in Phase II-A
    (Philips n=11 overall, but only n=3 in removal subset)
  R5 Comment 12: Small subgroups (ICH n=1, infra n=5)
  R3 Comment 1: Acquisition parameters across scanners


Paper changes
-------------
  Section 2.1: Study population description
  Section 3.2: Phase II-A population characteristics
  Table 1: Demographics and clinical characteristics


Response to Reviewers
---------------------
  R5 Comment 11: "Phase II-A comprised 89 subjects with ground truth
    WMH masks (59 BeLOVE removal subjects, 30 LOCATE training subjects).
    Scanner balance was limited (Philips n=3 [5.1%] in the removal
    subset), motivating the larger Phase II-B analysis (n=212,
    Philips n=36 [17.0%])."

  R5 Comment 12: "We acknowledge the small subgroup sizes for ICH
    (n=1) and infratentorial strokes (n=5) in Phase II-A. These are
    reported descriptively without inferential statistics. Phase II-B
    provides larger subgroups (ICH n=12, infra n=31)."


Outputs
-------
  2a_population_description.txt
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(SCRIPT_DIR, "RESULTS", "LOCATE_Results_Metrics_DICE_ONLY.xlsx")
META_XLSX_PATH = os.path.join(SCRIPT_DIR, "RESULTS", "BELOVE_K_META_DATA.xlsx")

# --- Load main data ---
print("Loading main data …")
df = pd.read_excel(XLSX_PATH)

# --- Load meta data ---
print("Loading meta data …")
meta_df = pd.read_excel(META_XLSX_PATH)

# --- Filter subjects with mask ---
if "subject_with_mask" in df.columns:
    df = df[df["subject_with_mask"] == 1].copy()

# --- Normalize lesion type ---
if "lesion_type" in df.columns:
    df["lesion_type"] = df["lesion_type"].replace("ICB", "ICH")

# --- Fill missing Wahlund from meta_df ---
meta_df["subject_key"] = meta_df["subject"].str.replace("belove_kersten-", "belove_k-")
wahlund_map = meta_df.set_index("subject_key")["wahlund"]
df["Wahlund"] = df["Wahlund"].fillna(df["subject"].map(wahlund_map))

# --- Also fill missing age/sex from meta_df ---
age_map = meta_df.set_index("subject_key")["age"]
sex_map = meta_df.set_index("subject_key")["sex"]
if "age" in df.columns:
    df["age"] = df["age"].fillna(df["subject"].map(age_map))
if "sex" in df.columns:
    df["sex"] = df["sex"].fillna(df["subject"].map(sex_map))

# --- Normalize sex labels ---
df["sex"] = df["sex"].replace({"Women": "female", "Men": "male"})

print(f"Wahlund missing after fill: {df['Wahlund'].isna().sum()}")
print(f"Age missing after fill:     {df['age'].isna().sum()}")
print(f"Sex missing after fill:     {df['sex'].isna().sum()}")

# ─── HELPERS ───
def fc(series):
    c = series.dropna().value_counts()
    N = len(series.dropna())
    if N == 0:
        return "no data"
    return ", ".join([f"{k} {v} ({v/N*100:.1f}%)" for k, v in c.items()])

def fcon(series, decimals=1):
    d = series.dropna()
    if len(d) == 0:
        return "no data"
    return (f"{d.mean():.{decimals}f}\u00b1{d.std():.{decimals}f} | "
            f"med {d.median():.{decimals}f} | "
            f"{d.min():.{decimals}f}\u2013{d.max():.{decimals}f}")

def fsev(series):
    sv = series.dropna()
    N = len(sv)
    if N == 0:
        return "no data"
    parts = []
    for level in ["low", "middle", "high"]:
        n = (sv == level).sum()
        if n > 0:
            parts.append(f"{level} {n} ({n/N*100:.1f}%)")
    return ", ".join(parts) + f"  [n={N}]"

def block(df_sub, label, N):
    L = [f"\n{'='*55}", f"{label}  N={N}", f"{'='*55}"]
    if "age" in df_sub.columns and df_sub["age"].notna().any():
        L.append(f"Age:        {fcon(df_sub['age'])}")
    if "sex" in df_sub.columns and df_sub["sex"].notna().any():
        L.append(f"Sex:        {fc(df_sub['sex'])}")
    if "scanner" in df_sub.columns and df_sub["scanner"].notna().any():
        L.append(f"Scanner:    {fc(df_sub['scanner'])}")
    if "severity_level" in df_sub.columns and df_sub["severity_level"].notna().any():
        L.append(f"Severity:   {fsev(df_sub['severity_level'])}")
    if "Wahlund" in df_sub.columns and df_sub["Wahlund"].notna().any():
        w = df_sub["Wahlund"].dropna()
        L.append(f"Wahlund:    med {w.median():.1f} "
                 f"(IQR {w.quantile(0.25):.1f}\u2013{w.quantile(0.75):.1f}) | "
                 f"{w.min():.0f}\u2013{w.max():.0f} | n={len(w)}")
    if "ROI_Volume" in df_sub.columns and df_sub["ROI_Volume"].notna().any():
        L.append(f"ROI Vol:    {fcon(df_sub['ROI_Volume'], 2)} mL  "
                 f"[n={df_sub['ROI_Volume'].notna().sum()}]")
    if "infarct_volume_ml" in df_sub.columns and df_sub["infarct_volume_ml"].notna().any():
        iv = df_sub["infarct_volume_ml"].dropna()
        L.append(f"Infarct Vol: med {iv.median():.2f} "
                 f"(IQR {iv.quantile(0.25):.2f}\u2013{iv.quantile(0.75):.2f}) mL  [n={len(iv)}]")
    if "brain_volume_ml" in df_sub.columns and df_sub["brain_volume_ml"].notna().any():
        L.append(f"Brain Vol:  {fcon(df_sub['brain_volume_ml'], 1)} mL  "
                 f"[n={df_sub['brain_volume_ml'].notna().sum()}]")
    if "lesion_type" in df_sub.columns and df_sub["lesion_type"].notna().any():
        L.append(f"LesionType: {fc(df_sub['lesion_type'])}")
    return L

# ─── POPULATION DESCRIPTION ───
lines = []

# 1) Full cohort
lines += block(df, "1) FULL LOCATE COHORT", len(df))

# 2) By source (BeLOVE vs Challenge)
if "source" in df.columns:
    for src in df["source"].dropna().unique():
        g = df[df["source"] == src]
        lines += block(g, f"2) {src} subset", len(g))

# 3) By scanner
if "scanner" in df.columns:
    for sc in sorted(df["scanner"].dropna().unique()):
        g = df[df["scanner"] == sc]
        lines += block(g, f"3) Scanner: {sc}", len(g))

# 4) Cross-tabs
if "scanner" in df.columns and "sex" in df.columns:
    ct = pd.crosstab(df["scanner"], df["sex"], margins=True)
    lines += [f"\nScanner\u00d7Sex:", ct.to_string()]

if "scanner" in df.columns and "lesion_type" in df.columns:
    ct = pd.crosstab(df["scanner"], df["lesion_type"], margins=True)
    lines += [f"\nScanner\u00d7LesionType:", ct.to_string()]

# 5) Dice summary per condition
lines.append(f"\n{'='*55}")
lines.append("DICE SUMMARY PER CONDITION")
lines.append(f"{'='*55}")
for cond in ["non_removed", "removed", "filled"]:
    col = f"WMH_{cond}_dice"
    if col in df.columns:
        d = df[col].dropna()
        lines.append(f"{cond:15s}: {d.mean():.3f}\u00b1{d.std():.3f} | "
                     f"med {d.median():.3f} | n={len(d)}")

text = "\n".join(lines)
print(text)

# Save
OUTPUT_TXT = os.path.join(SCRIPT_DIR, "2a_population_description.txt")
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
with open(OUTPUT_TXT, "w") as f:
    f.write(text)
print(f"\n✅ {OUTPUT_TXT}")