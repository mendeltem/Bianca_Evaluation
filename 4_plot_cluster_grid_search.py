#!/usr/bin/env python3
"""Plot cluster size grid search results with best min_cluster_size annotations.

# =============================================================
# REVIEWER RESPONSE (R1, Comment 2; R5, #17)
# =============================================================
#
# R1 raised the concern that global Dice alone cannot distinguish
# genuine sensitivity gains from spatially distributed false positives,
# and requested lesion-level metrics (lesion-level F1, count of detected
# lesions). R5 (#17) noted that precision values (~0.64) were not
# contextualized relative to segmentation standards.
#
# To address both concerns, we added cluster-based lesion-level metrics.
# The minimum cluster size was determined empirically via a grid search
# across 1 to 100 voxels, evaluated using stratified 5-fold
# cross-validation with 10 random seeds (see Supplemental Figure X and
# Supplemental Table X). Results are reported in the supplemental
# material and referenced in the main text (Section 3.X).
#
# Consistent with the MICCAI 2017 WMH Segmentation Challenge
# (Kuijf et al., 2019), a predicted cluster was considered a true
# positive if it overlapped with any ground truth cluster by at least
# one voxel (any-overlap criterion).
# =============================================================

# =============================================================
# PAPER TEXT: METHODS (Supplemental Material)
# =============================================================
#
# 2.X Minimum cluster size determination
#
# To determine the optimal minimum cluster size for lesion-level
# evaluation, we performed a grid search over cluster sizes ranging
# from 1 to 100 voxels in single-voxel increments. At each candidate
# size, connected components below the specified threshold were removed
# from predicted segmentations prior to computing lesion-level
# precision, recall, and F1 score. Ground truth masks were not filtered,
# as manual annotations represent expert-delineated lesions irrespective
# of cluster size. Consistent with the MICCAI 2017 WMH Segmentation
# Challenge (Kuijf et al., 2019), a predicted cluster was classified as
# a true positive if it overlapped with any ground truth cluster by at
# least one voxel (any-overlap criterion). The search was performed
# across all three thresholding approaches (B_0.85, B_0.90, and B+L),
# with metrics averaged across subjects, 5-fold cross-validation
# splits, and 10 random seeds to ensure stability of the selected
# threshold.
# =============================================================

# =============================================================
# PAPER TEXT: RESULTS (Supplemental Material)
# =============================================================
#
# The optimal minimum cluster size differed across thresholding
# methods: 3 voxels for B_0.85 (F1 = 0.359), 2 voxels for B_0.90
# (F1 = 0.350), and 16 voxels for B+L (F1 = 0.406). The selected
# values were stable across random seeds (B_0.85: range 3-4; B_0.90:
# range 2-3; B+L: range 7-17; Supplemental Figure X). Lesion-level
# F1 peaked at very small cluster sizes for B_0.85 and B_0.90, where
# precision was moderate (0.56-0.58) and recall was limited (0.34-0.35).
# B+L required a larger minimum cluster size (16 voxels) due to a
# higher number of predicted clusters (mean 230 vs. 131-142 for
# B_0.85/B_0.90 at MCS = 1), resulting in the highest precision
# (0.73) among all thresholding approaches.
#
# When analyzed separately by dataset, both the BeLOVE cohort and the
# WMH Segmentation Challenge dataset showed consistent trends, with
# BeLOVE achieving higher F1 scores (0.37-0.44) compared to the
# Challenge dataset (0.31-0.36), consistent with the fact that BIANCA
# was trained predominantly on BeLOVE data.
# =============================================================

# =============================================================
# FIGURE CAPTION (Supplemental)
# =============================================================
#
# Supplemental Figure X. Minimum cluster size optimization for
# lesion-level evaluation. (A) Mean lesion-level F1 score as a function
# of minimum cluster size for three BIANCA thresholding approaches
# (B_0.85, B_0.90, B+L). Stars indicate the optimal cluster size per
# threshold. Filtering was applied only to predicted masks; ground truth
# annotations were retained without size filtering. (B) Lesion-level
# precision, recall, and F1 shown separately per threshold; dashed red
# lines mark the optimum. (C) Lesion-level precision, recall, and F1
# shown separately for the BeLOVE cohort and the WMH Segmentation
# Challenge dataset. (D) Heatmap of mean F1 across cluster sizes 1-30
# and thresholds; black borders highlight the optimal row per threshold.
# =============================================================
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Paths ---
cluster_dir = "Phase_1/Cluster_analysis_CC/"
plot_dir = os.path.join(cluster_dir, "plot")
os.makedirs(plot_dir, exist_ok=True)

# --- Load data (multi-seed file) ---
df = pd.read_excel(os.path.join(cluster_dir, "cluster_size_grid_search_all_seeds.xlsx"))

# --- Dataset column ---
df['dataset'] = df['subject'].apply(
    lambda x: 'BeLOVE' if x.startswith('belove') else 'Challenge'
)

# --- Threshold display labels ---
LABELS = {'85': r'B$_{0.85}$', '90': r'B$_{0.90}$', 'locate': 'B+L'}
COLORS = {'85': '#1f77b4', '90': '#ff7f0e', 'locate': '#2ca02c'}
THRESH_ORDER = ['85', '90', 'locate']

# --- Combined averages ---
avg = df.groupby(['threshold', 'min_cluster_size'])[
    ['lesion_f1', 'lesion_precision', 'lesion_recall']
].mean().reset_index()
avg['threshold'] = avg['threshold'].astype(str)


# =============================================================
# Helper: Print summary table
# =============================================================
def print_summary():
    print(f"\n{'=' * 70}")
    print(f"BEST MCS PER THRESHOLD (averaged over all seeds, folds, subjects)")
    print(f"{'=' * 70}")
    for th in THRESH_ORDER:
        sub = avg[avg['threshold'] == th]
        best = sub.loc[sub['lesion_f1'].idxmax()]
        print(f"  {LABELS[th]:>10s}: MCS={int(best['min_cluster_size']):3d}v  "
              f"F1={best['lesion_f1']:.4f}  P={best['lesion_precision']:.4f}  "
              f"R={best['lesion_recall']:.4f}")

    # Stability across seeds
    seed_avg = df.groupby(['seed', 'threshold', 'min_cluster_size'])[
        ['lesion_f1']].mean().reset_index()
    print(f"\nSTABILITY (best MCS per seed):")
    for th in THRESH_ORDER:
        bests = []
        for s in sorted(df['seed'].unique()):
            sub = seed_avg[(seed_avg['seed'] == s) & (seed_avg['threshold'] == th)]
            best_mcs = int(sub.loc[sub['lesion_f1'].idxmax(), 'min_cluster_size'])
            bests.append(best_mcs)
        print(f"  {LABELS[th]:>10s}: seeds -> {bests}  (range: {min(bests)}-{max(bests)})")

    # Per dataset
    for ds in ['BeLOVE', 'Challenge']:
        dsdf = df[df['dataset'] == ds]
        ds_avg = dsdf.groupby(['threshold', 'min_cluster_size'])[
            ['lesion_f1', 'lesion_precision', 'lesion_recall']].mean().reset_index()
        print(f"\n--- {ds} (n={dsdf['subject'].nunique()}) ---")
        for th in THRESH_ORDER:
            sub = ds_avg[ds_avg['threshold'] == th]
            best = sub.loc[sub['lesion_f1'].idxmax()]
            print(f"  {LABELS[th]:>10s}: MCS={int(best['min_cluster_size']):3d}v  "
                  f"F1={best['lesion_f1']:.4f}  P={best['lesion_precision']:.4f}  "
                  f"R={best['lesion_recall']:.4f}")


# =============================================================
# (A) F1 vs min_cluster_size  combined
# =============================================================
def plot_f1_combined():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for th in THRESH_ORDER:
        sub = avg[avg['threshold'] == th].sort_values('min_cluster_size')
        ax.plot(sub['min_cluster_size'], sub['lesion_f1'],
                label=LABELS[th], color=COLORS[th], linewidth=1.5, alpha=0.9)

        best = sub.loc[sub['lesion_f1'].idxmax()]
        ax.plot(best['min_cluster_size'], best['lesion_f1'], marker='*',
                markersize=18, color=COLORS[th], zorder=5,
                markeredgecolor='black', markeredgewidth=0.7)

        offsets = {'85': (12, 14), '90': (12, -18), 'locate': (12, 8)}
        ax.annotate(
            f"MCS = {int(best['min_cluster_size'])}v (F1 = {best['lesion_f1']:.3f})",
            xy=(best['min_cluster_size'], best['lesion_f1']),
            textcoords="offset points", xytext=offsets[th],
            fontsize=9, fontweight='bold', color=COLORS[th],
            arrowprops=dict(arrowstyle='->', color=COLORS[th], lw=1.2),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=COLORS[th], alpha=0.85))

    ax.set_xlabel('Minimum cluster size (voxels)', fontsize=11)
    ax.set_ylabel('Mean lesion-level F1', fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "A_f1_vs_mcs_combined.png"), dpi=300)
    plt.close()
    print("  Saved: A_f1_vs_mcs_combined.png")


# =============================================================
# (B) Precision / Recall / F1 per threshold  combined
# =============================================================
def plot_prf1_combined():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for i, th in enumerate(THRESH_ORDER):
        sub = avg[avg['threshold'] == th].sort_values('min_cluster_size')
        axes[i].plot(sub['min_cluster_size'], sub['lesion_precision'],
                     linewidth=1.5, label='Precision', color='#e377c2')
        axes[i].plot(sub['min_cluster_size'], sub['lesion_recall'],
                     linewidth=1.5, label='Recall', color='#17becf')
        axes[i].plot(sub['min_cluster_size'], sub['lesion_f1'],
                     linewidth=1.5, label='F1', color=COLORS[th])

        best = sub.loc[sub['lesion_f1'].idxmax()]
        axes[i].axvline(best['min_cluster_size'], color='red',
                        linestyle='--', alpha=0.4, lw=1.2)
        axes[i].annotate(
            f"MCS = {int(best['min_cluster_size'])}v\nF1 = {best['lesion_f1']:.3f}",
            xy=(best['min_cluster_size'], best['lesion_f1']),
            textcoords="offset points", xytext=(30, -25),
            fontsize=9, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='red', alpha=0.85))

        axes[i].set_title(LABELS[th], fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Minimum cluster size (voxels)')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.25)
        axes[i].set_xlim(0, 100)

    axes[0].set_ylabel('Score')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "B_prf1_per_threshold_combined.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: B_prf1_per_threshold_combined.png")


# =============================================================
# (C) P / R / F1 per threshold  BeLOVE vs Challenge (2x3 grid)
# =============================================================
def plot_prf1_by_dataset():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True, sharex=True)

    for row, ds in enumerate(['BeLOVE', 'Challenge']):
        dsdf = df[df['dataset'] == ds]
        ds_avg = dsdf.groupby(['threshold', 'min_cluster_size'])[
            ['lesion_f1', 'lesion_precision', 'lesion_recall']
        ].mean().reset_index()
        ds_avg['threshold'] = ds_avg['threshold'].astype(str)

        for col, th in enumerate(THRESH_ORDER):
            ax = axes[row, col]
            sub = ds_avg[ds_avg['threshold'] == th].sort_values('min_cluster_size')
            ax.plot(sub['min_cluster_size'], sub['lesion_precision'],
                    linewidth=1.3, label='Precision', color='#e377c2')
            ax.plot(sub['min_cluster_size'], sub['lesion_recall'],
                    linewidth=1.3, label='Recall', color='#17becf')
            ax.plot(sub['min_cluster_size'], sub['lesion_f1'],
                    linewidth=1.5, label='F1', color=COLORS[th])

            best = sub.loc[sub['lesion_f1'].idxmax()]
            ax.axvline(best['min_cluster_size'], color='red',
                       linestyle='--', alpha=0.4, lw=1)
            ax.annotate(
                f"MCS = {int(best['min_cluster_size'])}v\n"
                f"F1 = {best['lesion_f1']:.3f}",
                xy=(best['min_cluster_size'], best['lesion_f1']),
                textcoords="offset points", xytext=(25, -20),
                fontsize=8, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='red', alpha=0.85))

            if row == 0:
                ax.set_title(LABELS[th], fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{ds}\nScore', fontsize=11)
            if row == 1:
                ax.set_xlabel('Minimum cluster size (voxels)', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.2)
            ax.set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "C_prf1_belove_vs_challenge.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: C_prf1_belove_vs_challenge.png")


# =============================================================
# (D) Heatmap  zoomed to MCS 1-30
# =============================================================
def plot_heatmap():
    avg_zoom = avg[avg['min_cluster_size'] <= 30].copy()
    avg_zoom['threshold'] = avg_zoom['threshold'].map(LABELS)
    pivot = avg_zoom.pivot(
        index='min_cluster_size', columns='threshold', values='lesion_f1')
    # Reorder columns
    pivot = pivot[[LABELS[t] for t in THRESH_ORDER if LABELS[t] in pivot.columns]]

    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                linewidths=0.3, annot_kws={'size': 8})

    for col_idx, th in enumerate(pivot.columns):
        best_mcs = pivot[th].idxmax()
        row_idx = list(pivot.index).index(best_mcs)
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False,
                                    edgecolor='black', lw=2.5))

    ax.set_ylabel('Minimum cluster size (voxels)')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "D_heatmap_f1_mcs1_30.png"), dpi=300)
    plt.close()
    print("  Saved: D_heatmap_f1_mcs1_30.png")


# =============================================================
# (E) F1 overlay: BeLOVE vs Challenge per threshold
# =============================================================
def plot_f1_by_dataset_overlay():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for i, th in enumerate(THRESH_ORDER):
        ax = axes[i]
        for ds, ls, alpha in [('BeLOVE', '-', 1.0), ('Challenge', '--', 0.8)]:
            dsdf = df[df['dataset'] == ds]
            ds_avg = dsdf.groupby(['threshold', 'min_cluster_size'])[
                ['lesion_f1']].mean().reset_index()
            ds_avg['threshold'] = ds_avg['threshold'].astype(str)
            sub = ds_avg[ds_avg['threshold'] == th].sort_values('min_cluster_size')
            ax.plot(sub['min_cluster_size'], sub['lesion_f1'],
                    linewidth=1.5, linestyle=ls, alpha=alpha,
                    label=ds, color=COLORS[th])
            best = sub.loc[sub['lesion_f1'].idxmax()]
            ax.plot(best['min_cluster_size'], best['lesion_f1'], marker='*',
                    markersize=14, color=COLORS[th], zorder=5,
                    markeredgecolor='black', markeredgewidth=0.5)

        ax.set_title(LABELS[th], fontsize=12)
        ax.set_xlabel('Minimum cluster size (voxels)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 100)

    axes[0].set_ylabel('Mean lesion-level F1')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "E_f1_belove_vs_challenge_overlay.png"),
                dpi=300)
    plt.close()
    print("  Saved: E_f1_belove_vs_challenge_overlay.png")


# =============================================================
# MAIN
# =============================================================
def main():
    print_summary()
    print(f"\nGenerating plots...")
    plot_f1_combined()
    plot_prf1_combined()
    plot_prf1_by_dataset()
    plot_heatmap()
    plot_f1_by_dataset_overlay()
    print(f"\nAll plots saved to: {plot_dir}/")


if __name__ == "__main__":
    main()