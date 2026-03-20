#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Feature Importance Analysis  BIANCA WMH Revision (Phase II-B, n=211)
===========================================================================
Random Forest model predicting absolute WMH volume differences between
Non Removed and Removed conditions. SHAP TreeExplainer for feature
importance with 95% CI.

Single comparison: Non Removed vs Removed

Rationale: Phase II-A demonstrated that removed and inpainted conditions
yield equivalent segmentation accuracy (all Cliff's delta < 0.005) and
equivalent WMH volumes (convergence comparison: all p_bonf = 1.0,
all |delta| negligible). Therefore, Phase II-B SHAP analysis uses
the non_removed vs removed comparison only.

Features:
  - Stroke Lesion Volume (infarct_volume_ml)
  - Wahlund ARWMC score
  - Age
  - Brain Volume
  - Scanner Type (Siemens=1, Philips=2)
  - Lesion Type (infra=1, lacune=2, infarct=3, mixed=4, ICH=5)
  - Sex (Women=0, Men=1)

Model: RandomForestRegressor (n_estimators=100, max_depth=10)
  R² reported on training data (descriptive, not predictive).

Key results
-----------
Non Removed vs Removed (n=211, in-sample R²=0.930, OOB R²=0.477,
  MAE=0.087 mL):
  Stroke Lesion Volume highest-ranked predictor: 0.359 (64.8%),
  confirming size-dependent scaling of preprocessing effects.

  Ranking:
    1. Stroke Lesion Volume  0.359  64.8%
    2. Wahlund (ARWMC)       0.069  12.4%
    3. Scanner Type           0.054   9.8%
    4. Brain Volume           0.033   5.9%
    5. Lesion Type            0.022   3.9%
    6. Age                    0.016   2.9%
    7. Sex                    0.001   0.2%

  Key changes vs Phase II-A (n=89):
    - Scanner Type rose from 0.1% (rank 7) to 9.8% (rank 3),
      a ~98-fold increase in relative importance. This is consistent
      with improved Philips representation (17.0% vs 12.4%).
    - Stroke Lesion Volume remains highest-ranked but decreased from
      76.3% to 64.8%, redistributing influence to other predictors.
    - OOB R² = 0.477 (vs 0.357 in Phase II-A), indicating improved
      generalization with the larger sample. The discrepancy between
      in-sample (0.930) and OOB (0.477) reflects the explanatory
      purpose of the surrogate model (Molnar, 2022).

  Summary plot interpretation:
    - High lesion volume (red) → large positive SHAP values
      (larger volume differences)
    - Scanner Type shows clear Philips cluster (red, high value)
      with positive SHAP, consistent with scanner-stratified analysis
    - ARWMC shows expected pattern: higher WMH burden → larger
      preprocessing effects


Revision context
----------------
  R5 Comment 10: SHAP magnitude depends on model scaling; absolute
    predictive performance (R², MAE) must be reported.
  R5 Comment 4: systematic bias despite negligible effect size


Paper changes
-------------
  Section 3.4: SHAP feature importance (Phase II-B)
  Figure X: SHAP importance with 95% CI


Response to Reviewers
---------------------
  R5 Comment 10: "We now report in-sample R²=0.930, OOB R²=0.477,
    and MAE=0.087 mL alongside SHAP values (Phase II-B, n=211).
    Stroke lesion volume remained the highest-ranked predictor (64.8%),
    while scanner type increased from 0.1% to 9.8% relative importance
    with improved Philips representation (17.0% vs 12.4% in Phase II-A).
    This increase confirms that the limited scanner balance in Phase II-A
    obscured scanner-specific effects."


Outputs
-------
  shap_importance_ci_non_removed_vs_removed.png
  shap_summary_non_removed_vs_removed.png
  shap_feature_importance.xlsx
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import sem

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(SCRIPT_DIR, "RESULTS", "LOCATE_Results_Metrics_ALL.xlsx")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots", "2b_SHAP_Analysis")
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURE_COLS = ["ARWMC", "sex", "age", "infarct_volume",
                "brain_volume", "scanner", "lesion_type"]

FEATURE_DISPLAY = {
    "ARWMC": "Wahlund (ARWMC)",
    "sex": "Sex",
    "age": "Age",
    "infarct_volume": "Stroke Lesion Volume",
    "brain_volume": "Brain Volume",
    "scanner": "Scanner Type",
    "lesion_type": "Lesion Type",
}

COND_LABELS = {
    "non_removed": "Non Removed",
    "removed":     "Removed",
}

COMPARISONS = [
    ("non_removed", "removed"),
]

SCANNER_MAPPING = {"Prisma_fit": 1, "Tim Trio": 1, "Philips": 2}
LESION_TYPE_MAPPING = {"infra": 1, "lacune": 2, "infarct": 3, "mixed": 4, "ICH": 5}
SEX_MAPPING = {"Women": 0, "Men": 1, "female": 0, "male": 1, "F": 0, "M": 1}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "oob_score": True,
    "n_jobs": -1,
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

print("Loading data …")
try:
    df = pd.read_excel(XLSX_PATH)
except FileNotFoundError:
    raise SystemExit(f"File not found: {XLSX_PATH}")

# Phase II-B: BeLOVE subjects only (exclude Challenge subjects)
df = df[df["subject"].str.startswith("sub-")].copy()

# Exclude sub-027: fsl_anat preprocessing failed
df = df[df["subject"] != "sub-027"].copy()

# ─────────────────────────────────────────────
# FEATURE PREPARATION
# ─────────────────────────────────────────────

df["ARWMC"] = df["Wahlund"]
df["scanner"] = df["scanner"].map(SCANNER_MAPPING)
df["lesion_type"] = df["lesion_type"].replace("ICB", "ICH").map(LESION_TYPE_MAPPING)
df["sex"] = df["sex"].map(SEX_MAPPING)
df["infarct_volume"] = df["infarct_volume_ml"]
df["brain_volume"] = df["brain_volume_ml"]

df_feat = df.dropna(subset=FEATURE_COLS).copy()
print(f"Subjects after feature prep: {len(df_feat)}")

# ─────────────────────────────────────────────
# SHAP FUNCTIONS
# ─────────────────────────────────────────────

def calculate_shap_statistics(shap_values, feature_names):
    abs_shap = np.abs(shap_values)
    stats = {}
    for i, feat in enumerate(feature_names):
        vals = abs_shap[:, i]
        mean_val = vals.mean()
        stats[feat] = {
            "mean": mean_val,
            "std": vals.std(),
            "sem": sem(vals),
            "ci_95": 1.96 * sem(vals),
            "values": vals,
        }
    return stats


def train_and_explain(df_clean, target_col, comparison_label):
    print(f"\n  Training model for: {comparison_label}")
    X = df_clean[FEATURE_COLS].copy()
    y = df_clean[target_col].copy()
    valid = y.notna()
    X, y = X[valid], y[valid]
    print(f"    Features: {X.shape}, Target: {y.shape}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS, index=X.index)

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_scaled, y)
    r2 = model.score(X_scaled, y)
    r2_oob = model.oob_score_
    from sklearn.metrics import mean_absolute_error
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    print(f"    R² (in-sample) = {r2:.4f}")
    print(f"    R² (OOB)       = {r2_oob:.4f}")
    print(f"    MAE            = {mae:.4f} mL")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    stats = calculate_shap_statistics(shap_values, FEATURE_COLS)

    return {
        "model": model, "shap_values": shap_values,
        "X_scaled": X_scaled, "r2": r2, "r2_oob": r2_oob,
        "mae": mae, "stats": stats,
    }


def plot_importance_with_ci(stats, comparison_label, save_path, r2):
    features = list(stats.keys())
    means = np.array([stats[f]["mean"] for f in features])
    cis = np.array([stats[f]["ci_95"] for f in features])
    pcts = 100 * means / means.sum()

    idx = np.argsort(means)[::-1]
    features_s = [FEATURE_DISPLAY.get(features[i], features[i]) for i in idx]
    means_s, cis_s, pcts_s = means[idx], cis[idx], pcts[idx]

    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(features_s)))
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features_s))
    ax.barh(y_pos, means_s, xerr=cis_s, color=colors,
            capsize=5, error_kw={"linewidth": 2, "elinewidth": 2})
    for i, (m, c, p) in enumerate(zip(means_s, cis_s, pcts_s)):
        ax.text(m + c + 0.005, i, f"{m:.3f} ({p:.1f}%)",
                va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_s)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_title(f"SHAP Feature Importance (Phase II-B)\n{comparison_label}"
                 f"\n(R² = {r2:.4f}, with 95% CI)",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, (means_s + cis_s).max() * 1.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Plot saved: {save_path}")


def plot_shap_summary(X_scaled, shap_values, save_path):
    display_names = [FEATURE_DISPLAY.get(c, c) for c in FEATURE_COLS]
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled, feature_names=display_names, show=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Summary plot saved: {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SHAP Feature Importance Analysis (Phase II-B)")
    print("=" * 60)

    all_tables = {}
    all_results = {}

    for cond1, cond2 in COMPARISONS:
        l1, l2 = COND_LABELS[cond1], COND_LABELS[cond2]
        comp_label = f"{l1} vs {l2}"
        comp_key = f"{cond1}_vs_{cond2}"

        # Target: absolute WMH volume difference
        target_col = f"WMH_abs_diff_{comp_key}"
        df_feat[target_col] = (df_feat[f"WMH_{cond1}_volume_ml"]
                               - df_feat[f"WMH_{cond2}_volume_ml"]).abs()

        res = train_and_explain(df_feat, target_col, comp_label)
        all_results[comp_key] = res

        # Plots
        plot_importance_with_ci(
            res["stats"], comp_label,
            os.path.join(PLOT_DIR, f"shap_importance_ci_{comp_key}.png"),
            r2=res["r2"])
        plot_shap_summary(
            res["X_scaled"], res["shap_values"],
            os.path.join(PLOT_DIR, f"shap_summary_{comp_key}.png"))

        # Results table
        features = list(res["stats"].keys())
        means = np.array([res["stats"][f]["mean"] for f in features])
        pcts = 100 * means / means.sum()
        idx = np.argsort(means)[::-1]

        rows = []
        for i in idx:
            f = features[i]
            s = res["stats"][f]
            rows.append({
                "Feature": FEATURE_DISPLAY.get(f, f),
                "Mean |SHAP|": round(s["mean"], 4),
                "95% CI": f"±{s['ci_95']:.4f}",
                "% Influence": round(pcts[i], 1),
                "R² (in-sample)": round(res["r2"], 4),
                "R² (OOB)": round(res["r2_oob"], 4),
                "MAE (mL)": round(res["mae"], 4),
            })
        tbl = pd.DataFrame(rows)
        all_tables[comp_key] = tbl

        print(f"\n    {comp_label} (R²={res['r2']:.4f}, "
              f"R²_OOB={res['r2_oob']:.4f}, MAE={res['mae']:.4f} mL):")
        for _, row in tbl.iterrows():
            print(f"    {row['Feature']:<25} {row['Mean |SHAP|']:<10.4f} "
                  f"{row['95% CI']:<12} {row['% Influence']:>5.1f}%")

    # Save Excel
    excel_path = os.path.join(PLOT_DIR, "shap_feature_importance.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, tbl in all_tables.items():
            tbl.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"\nAll tables saved: {excel_path}")

    # Comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for comp_key, res in all_results.items():
        print(f"\n  {comp_key}:")
        print(f"    R² in-sample = {res['r2']:.4f}")
        print(f"    R² OOB       = {res['r2_oob']:.4f}")
        print(f"    MAE          = {res['mae']:.4f} mL")
        top_feat = max(res["stats"], key=lambda f: res["stats"][f]["mean"])
        print(f"    Highest-ranked: {FEATURE_DISPLAY[top_feat]} "
              f"({100 * res['stats'][top_feat]['mean'] / sum(res['stats'][f2]['mean'] for f2 in res['stats']):.1f}%)")

    print("\nDone.")