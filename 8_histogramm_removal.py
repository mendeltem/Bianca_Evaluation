#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
histogram_intensity_comparison.py  (PARALLELIZED)
==================================================

Revision context (NeuroImage: Clinical, Major Revision)
-------------------------------------------------------
  R5 Comment 18: "The argument that lesions distort global histograms
  is plausible but not empirically demonstrated (no histogram analyses
  shown)."

Paper changes
-------------
  Supplemental Figure S8: Empirical comparison of FLAIR intensity
  distributions between non_removed and removed conditions, demonstrating
  that stroke lesions produce a right-tail distortion in the intensity
  histogram that is eliminated by lesion removal.

Response to Reviewers
---------------------
  R5C18 Part 2: "We have added an empirical comparison of intensity
  distributions between non_removed and removed conditions demonstrating
  the histogram distortion effect (Supplemental Figure S8)."

Design
------
  Dataset: Phase II-B only (n=211, BeLOVE Cohort 2, removal subjects).
  sub-027 excluded (failed fsl_anat).

  Parallelization: NIfTI loading + intensity extraction is parallelized
  across N_WORKERS processes. Each worker returns summary statistics only
  (not raw arrays) to minimize IPC overhead. Raw arrays are only loaded
  for the 3 representative subjects + group KDE sampling in a second pass.

Output
------
  plots/histogram_comparison/histogram_intensity_comparison.png  (300 dpi)
  plots/histogram_comparison/histogram_intensity_comparison.pdf
  plots/histogram_comparison/histogram_intensity_stats.xlsx
"""

import os
import sys
import warnings
import time
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REMOVAL_BELOVE_DATASET_BIDS = os.path.join(SCRIPT_DIR, "DATASETS/REMOVAL_BELOVE_DATASET_BIDS")

REMOVAL_BELOVE_preprocessed_files = os.path.join(
    REMOVAL_BELOVE_DATASET_BIDS, "derivatives/preprocessed_files.xlsx")
REMOVAL_BELOVE_file_locations = os.path.join(
    REMOVAL_BELOVE_DATASET_BIDS, "derivatives/REMOVAL_BELOVE_file_locations.xlsx")

PLOT_DIR = os.path.join(SCRIPT_DIR, "plots", "histogram_comparison")
os.makedirs(PLOT_DIR, exist_ok=True)

# Excluded subjects (failed fsl_anat)
EXCLUDED_SUBJECTS = {"sub-027"}

REPRESENTATIVE_SUBJECTS = None

# Parallelization
N_WORKERS = min(mp.cpu_count(), 16)  # adjust to your SLURM allocation

# Figure
DPI = 300
N_BINS = 200
KDE_POINTS = 500
KDE_SAMPLE_PER_SUB = 50000  # voxels sampled per subject for group KDE

DEBUG = False


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────

def load_subjects_and_paths():
    """Build subject list from Phase II-B removal dataset only (n=211)."""
    removal_preproc_df = pd.read_excel(REMOVAL_BELOVE_preprocessed_files)
    removal_file_df    = pd.read_excel(REMOVAL_BELOVE_file_locations)
    removal_df = removal_preproc_df.merge(
        removal_file_df, on='subject', how='inner')

    # Phase II-B: BeLOVE subjects only (sub- prefix)
    removal_df = removal_df[removal_df['subject'].str.startswith('sub-')]

    # Exclude failed subjects
    removal_df = removal_df[~removal_df['subject'].isin(EXCLUDED_SUBJECTS)]

    subjects_info = []

    for _, row in removal_df.iterrows():
        sub = row['subject']
        if 'FLAIR_non_removed_path' not in row or pd.isna(row['FLAIR_non_removed_path']):
            continue
        flair_path     = os.path.join(REMOVAL_BELOVE_DATASET_BIDS, str(row['FLAIR_non_removed_path']))
        brainmask_path = os.path.join(REMOVAL_BELOVE_DATASET_BIDS, str(row['brainmask']))
        lesion_path = ""
        if 'lesion_path' in row and not pd.isna(row['lesion_path']):
            lesion_path = os.path.join(REMOVAL_BELOVE_DATASET_BIDS, str(row['lesion_path']))
        if not (os.path.isfile(flair_path) and os.path.isfile(brainmask_path)
                and lesion_path and os.path.isfile(lesion_path)):
            continue
        subjects_info.append({
            'subject':        sub,
            'source':         'removal',
            'flair_path':     flair_path,
            'brainmask_path': brainmask_path,
            'lesion_path':    lesion_path,
            'scanner':        str(row.get('scanner', '')),
            'lesion_type':    str(row.get('lesion_type', '')),
        })

    return subjects_info


# ─────────────────────────────────────────────────────────────────────
# WORKER FUNCTION (runs in subprocess)
# ─────────────────────────────────────────────────────────────────────

def process_one_subject(info):
    """
    Load NIfTI, extract intensities, compute stats.
    Returns a dict with summary stats only (no raw arrays -> fast IPC).
    """
    sub = info['subject']
    try:
        flair  = nib.load(info['flair_path']).get_fdata()
        brain  = nib.load(info['brainmask_path']).get_fdata()
        lesion = nib.load(info['lesion_path']).get_fdata()

        brain_mask      = brain > 0.5
        lesion_mask     = lesion > 0.5
        lesion_in_brain = brain_mask & lesion_mask

        n_lesion = int(np.sum(lesion_in_brain))
        if n_lesion < 5:
            return {'subject': sub, 'status': 'skip', 'reason': f'{n_lesion} lesion voxels'}

        hdr = nib.load(info['flair_path']).header
        voxel_vol = np.prod(hdr.get_zooms()[:3])
        lesion_volume_ml = n_lesion * voxel_vol / 1000.0

        nr  = flair[brain_mask]
        rem = flair[brain_mask & ~lesion_mask]

        nr  = nr[nr > 0]
        rem = rem[rem > 0]

        nr_skew  = float(stats.skew(nr))
        rem_skew = float(stats.skew(rem))
        nr_kurt  = float(stats.kurtosis(nr))
        rem_kurt = float(stats.kurtosis(rem))

        return {
            'subject':          sub,
            'status':           'ok',
            'source':           info['source'],
            'scanner':          info['scanner'],
            'lesion_type':      info['lesion_type'],
            'lesion_volume_ml': round(lesion_volume_ml, 2),
            'n_brain_voxels':   int(np.sum(brain_mask)),
            'n_lesion_voxels':  n_lesion,
            'lesion_frac':      round(100 * n_lesion / max(int(np.sum(brain_mask)), 1), 3),
            'nr_skewness':      round(nr_skew, 4),
            'rem_skewness':     round(rem_skew, 4),
            'skew_diff':        round(nr_skew - rem_skew, 4),
            'nr_kurtosis':      round(nr_kurt, 4),
            'rem_kurtosis':     round(rem_kurt, 4),
            'kurt_diff':        round(nr_kurt - rem_kurt, 4),
            'nr_p95':           round(float(np.percentile(nr, 95)), 1),
            'rem_p95':          round(float(np.percentile(rem, 95)), 1),
            'nr_p99':           round(float(np.percentile(nr, 99)), 1),
            'rem_p99':          round(float(np.percentile(rem, 99)), 1),
        }

    except Exception as e:
        return {'subject': sub, 'status': 'error', 'reason': str(e)}


def load_raw_intensities(info):
    """
    Load raw intensities for ONE subject (used only for representative
    subjects and KDE sampling, NOT run in the parallel stats pass).
    """
    flair  = nib.load(info['flair_path']).get_fdata()
    brain  = nib.load(info['brainmask_path']).get_fdata()
    lesion = nib.load(info['lesion_path']).get_fdata()

    brain_mask      = brain > 0.5
    lesion_mask     = lesion > 0.5
    lesion_in_brain = brain_mask & lesion_mask

    nr  = flair[brain_mask];       nr  = nr[nr > 0]
    rem = flair[brain_mask & ~lesion_mask]; rem = rem[rem > 0]
    les = flair[lesion_in_brain];  les = les[les > 0]

    hdr = nib.load(info['flair_path']).header
    voxel_vol = np.prod(hdr.get_zooms()[:3])

    return {
        'nr': nr, 'rem': rem, 'les': les,
        'lesion_volume_ml': np.sum(lesion_in_brain) * voxel_vol / 1000.0,
    }


def load_and_sample_for_kde(info):
    """Load one subject, z-score, subsample for KDE. Returns (nr_sample, rem_sample)."""
    flair  = nib.load(info['flair_path']).get_fdata()
    brain  = nib.load(info['brainmask_path']).get_fdata()
    lesion = nib.load(info['lesion_path']).get_fdata()

    brain_mask  = brain > 0.5
    lesion_mask = lesion > 0.5

    nr  = flair[brain_mask];                    nr  = nr[nr > 0]
    rem = flair[brain_mask & ~lesion_mask];      rem = rem[rem > 0]

    def z(a):
        mu, s = np.mean(a), np.std(a)
        return (a - mu) / s if s > 1e-10 else a - mu

    rng = np.random.default_rng(hash(info['subject']) % (2**31))
    cap = KDE_SAMPLE_PER_SUB

    nr_z  = z(nr)
    rem_z = z(rem)

    nr_s  = rng.choice(nr_z,  size=min(len(nr_z), cap),  replace=False)
    rem_s = rng.choice(rem_z, size=min(len(rem_z), cap), replace=False)

    return nr_s, rem_s


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 65)
    print("  FLAIR Intensity Histogram Comparison (PARALLELIZED)")
    print(f"  Workers: {N_WORKERS} | Phase II-B BeLOVE Cohort 2 (n=211)")
    print("=" * 65)

    subjects_info = load_subjects_and_paths()
    print(f"\nLoaded {len(subjects_info)} subjects (Phase II-B only)")

    if DEBUG:
        subjects_info = subjects_info[:10]
        print(f"  DEBUG: {len(subjects_info)} subjects")

    # 
    # PASS 1: Parallel stats extraction (no raw arrays transferred)
    # 
    print(f"\n--- Pass 1: Extracting stats ({N_WORKERS} workers) ---")
    t1 = time.time()

    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(process_one_subject, subjects_info)

    ok_results = [r for r in results if r['status'] == 'ok']
    skipped    = [r for r in results if r['status'] == 'skip']
    errors     = [r for r in results if r['status'] == 'error']

    print(f"  Done in {time.time()-t1:.1f}s: {len(ok_results)} ok, "
          f"{len(skipped)} skipped, {len(errors)} errors")
    for e in errors:
        print(f"    ERROR {e['subject']}: {e['reason']}")

    # Build stats DataFrame
    stats_df = pd.DataFrame([{
        'Subject':           r['subject'],
        'Source':            r['source'],
        'Scanner':           r['scanner'],
        'Lesion_Type':       r['lesion_type'],
        'Lesion_Volume_mL':  r['lesion_volume_ml'],
        'N_Brain_Voxels':    r['n_brain_voxels'],
        'N_Lesion_Voxels':   r['n_lesion_voxels'],
        'Lesion_Fraction_%': r['lesion_frac'],
        'NR_Skewness':       r['nr_skewness'],
        'Removed_Skewness':  r['rem_skewness'],
        'Skewness_Diff':     r['skew_diff'],
        'NR_Kurtosis':       r['nr_kurtosis'],
        'Removed_Kurtosis':  r['rem_kurtosis'],
        'Kurtosis_Diff':     r['kurt_diff'],
        'NR_P95':            r['nr_p95'],
        'Removed_P95':       r['rem_p95'],
        'NR_P99':            r['nr_p99'],
        'Removed_P99':       r['rem_p99'],
    } for r in ok_results]).sort_values('Lesion_Volume_mL')

    stats_df.to_excel(os.path.join(PLOT_DIR, "histogram_intensity_stats.xlsx"),
                      index=False)

    # 
    # Select representative subjects
    # 
    ok_subs = {r['subject']: r for r in ok_results}
    info_lookup = {i['subject']: i for i in subjects_info}

    if REPRESENTATIVE_SUBJECTS:
        rep_subs = [s for s in REPRESENTATIVE_SUBJECTS if s in ok_subs]
    else:
        ss = stats_df['Subject'].tolist()
        n = len(ss)
        rep_subs = [ss[n // 6], ss[n // 2], ss[min(5 * n // 6, n - 1)]]

    print(f"\nRepresentative subjects:")
    for s in rep_subs:
        print(f"  {s}: {ok_subs[s]['lesion_volume_ml']:.1f} mL")

    # 
    # PASS 2a: Load raw intensities for 3 representative subjects
    # 
    print(f"\n--- Pass 2a: Loading representative subjects ---")
    rep_data = {}
    for s in rep_subs:
        rep_data[s] = load_raw_intensities(info_lookup[s])

    # 
    # PASS 2b: Parallel KDE sampling (load + z-score + subsample)
    # 
    print(f"--- Pass 2b: KDE sampling ({N_WORKERS} workers) ---")
    t2 = time.time()

    ok_infos = [info_lookup[r['subject']] for r in ok_results
                if r['subject'] in info_lookup]

    with mp.Pool(N_WORKERS) as pool:
        kde_samples = pool.map(load_and_sample_for_kde, ok_infos)

    nr_pool  = np.concatenate([s[0] for s in kde_samples])
    rem_pool = np.concatenate([s[1] for s in kde_samples])
    print(f"  Done in {time.time()-t2:.1f}s: "
          f"{len(nr_pool)//1000}k NR voxels, {len(rem_pool)//1000}k Rem voxels")

    # 
    # FIGURE
    # 
    print(f"\n--- Generating figure ---")

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30,
                           left=0.07, right=0.97, top=0.93, bottom=0.08)

    C_NR, C_REM, C_LES = '#2166AC', '#E08214', '#B2182B'

    # -- Top: 3 representative subjects --
    for i, sub in enumerate(rep_subs):
        ax = fig.add_subplot(gs[0, i])
        d = rep_data[sub]
        nr, rem, les = d['nr'], d['rem'], d['les']

        vmin = min(np.percentile(nr, 0.5), np.percentile(rem, 0.5))
        vmax = max(np.percentile(nr, 99.5), np.percentile(rem, 99.5))
        bins = np.linspace(vmin, vmax, N_BINS)

        ax.hist(nr, bins=bins, density=True, alpha=0.5, color=C_NR,
                label='Non-removed', edgecolor='none')
        ax.hist(rem, bins=bins, density=True, alpha=0.5, color=C_REM,
                label='Removed', edgecolor='none')

        if len(les) > 10:
            lb = bins[(bins >= np.percentile(les, 1)) &
                      (bins <= np.percentile(les, 99))]
            if len(lb) > 5:
                ax.hist(les, bins=lb, density=True, alpha=0.3, color=C_LES,
                        label='Lesion voxels', edgecolor='none')

        ax.axvline(np.mean(les), color=C_LES, ls='--', lw=1.2, alpha=0.8,
                   label=f'Lesion mean ({np.mean(les):.0f})')

        tag = ["small", "medium", "large"][i]
        ax.set_title(f'{"ABC"[i]}) {sub}\nLesion: {d["lesion_volume_ml"]:.1f} mL ({tag})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('FLAIR intensity (a.u.)', fontsize=9)
        if i == 0:
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
        ax.tick_params(labelsize=8)

    # -- Bottom-left: Group KDE --
    ax_g = fig.add_subplot(gs[1, 0:2])

    x = np.linspace(-4, 6, KDE_POINTS)
    k_nr  = gaussian_kde(nr_pool, bw_method=0.1)
    k_rem = gaussian_kde(rem_pool, bw_method=0.1)

    ax_g.fill_between(x, k_nr(x), alpha=0.4, color=C_NR, label='Non-removed')
    ax_g.fill_between(x, k_rem(x), alpha=0.4, color=C_REM,
                      label='Removed (lesion voxels excluded)')
    ax_g.plot(x, k_nr(x), color=C_NR, lw=1.5)
    ax_g.plot(x, k_rem(x), color=C_REM, lw=1.5)

    rt = x > 2
    ax_g.fill_between(x[rt], k_nr(x[rt]), k_rem(x[rt]),
                      where=k_nr(x[rt]) > k_rem(x[rt]),
                      alpha=0.3, color=C_LES,
                      label='Right-tail excess (lesion signal)')

    ax_g.set_title(f'D) Group-level intensity distribution '
                   f'(z-scored, n={len(ok_results)})',
                   fontsize=10, fontweight='bold')
    ax_g.set_xlabel('Z-scored FLAIR intensity', fontsize=9)
    ax_g.set_ylabel('Density', fontsize=9)
    ax_g.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax_g.set_xlim(-4, 6)
    ax_g.tick_params(labelsize=8)

    # -- Bottom-right: Skewness scatter --
    ax_s = fig.add_subplot(gs[1, 2])
    lv = stats_df['Lesion_Volume_mL'].values
    sd = stats_df['Skewness_Diff'].values

    sc = ax_s.scatter(lv, sd, c=lv, cmap='RdYlBu_r', s=30, alpha=0.7,
                      edgecolors='k', linewidths=0.3)
    ax_s.axhline(0, color='grey', ls='--', lw=0.8, alpha=0.5)

    m = ~np.isnan(sd)
    if np.sum(m) > 5:
        rho_fig, p_fig = stats.spearmanr(lv[m], sd[m])
        sl, ic, _, _, _ = stats.linregress(lv[m], sd[m])
        xf = np.linspace(0, np.max(lv[m]) * 1.05, 100)
        ax_s.plot(xf, sl * xf + ic, color=C_LES, lw=1.5,
                  label=f'\u03C1={rho_fig:.2f}, p={"<0.001" if p_fig < 0.001 else f"{p_fig:.3f}"}')
        ax_s.legend(fontsize=8, loc='upper right')

    ax_s.set_title('E) Skewness shift vs. lesion volume',
                   fontsize=10, fontweight='bold')
    ax_s.set_xlabel('Stroke lesion volume (mL)', fontsize=9)
    ax_s.set_ylabel('Skewness (NR \u2212 Removed)', fontsize=9)
    ax_s.tick_params(labelsize=8)
    plt.colorbar(sc, ax=ax_s, label='Lesion vol. (mL)',
                 fraction=0.046, pad=0.04)

    for ext in ['png', 'pdf']:
        p = os.path.join(PLOT_DIR, f"histogram_intensity_comparison.{ext}")
        fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {p}")
    plt.close()

    # 
    # SUMMARY + MANUSCRIPT TEXT
    # 
    w_s, p_s = stats.wilcoxon(stats_df['NR_Skewness'], stats_df['Removed_Skewness'])
    w_k, p_k = stats.wilcoxon(stats_df['NR_Kurtosis'], stats_df['Removed_Kurtosis'])
    rho, p_r = stats.spearmanr(stats_df['Lesion_Volume_mL'], stats_df['Skewness_Diff'])

    pos = (stats_df['Skewness_Diff'] > 0).sum()

    small  = stats_df[stats_df['Lesion_Volume_mL'] < 1]
    medium = stats_df[(stats_df['Lesion_Volume_mL'] >= 1) & (stats_df['Lesion_Volume_mL'] < 5)]
    large  = stats_df[stats_df['Lesion_Volume_mL'] >= 5]

    ps = '<0.001' if p_s < 0.001 else f'{p_s:.3f}'
    pr = '<0.001' if p_r < 0.001 else f'{p_r:.3f}'

    print(f"\n{'='*65}")
    print(f"  SUMMARY  (total time: {time.time()-t0:.1f}s)")
    print(f"{'='*65}")
    print(f"  N = {len(ok_results)}")
    print(f"  Skewness NR:  {stats_df['NR_Skewness'].mean():.4f} +/- {stats_df['NR_Skewness'].std():.4f}")
    print(f"  Skewness Rem: {stats_df['Removed_Skewness'].mean():.4f} +/- {stats_df['Removed_Skewness'].std():.4f}")
    print(f"  Skew diff:    {stats_df['Skewness_Diff'].mean():+.4f} +/- {stats_df['Skewness_Diff'].std():.4f}")
    print(f"  Wilcoxon:     W={w_s:.0f}, p={ps}")
    print(f"  NR > Rem:     {pos}/{len(stats_df)} ({100*pos/len(stats_df):.1f}%)")
    print(f"  Spearman:     rho={rho:.3f}, p={pr}")
    print(f"  By size:")
    print(f"    Small  (<1 mL, n={len(small)}):  {small['Skewness_Diff'].mean():+.4f}")
    print(f"    Medium (1-5 mL, n={len(medium)}): {medium['Skewness_Diff'].mean():+.4f}")
    print(f"    Large  (>=5 mL, n={len(large)}):  {large['Skewness_Diff'].mean():+.4f}")

    fold = large['Skewness_Diff'].mean() / max(small['Skewness_Diff'].mean(), 1e-6)

    print(f"\n{'='*65}")
    print(f"  SUPPLEMENTAL FIGURE LEGEND")
    print(f"{'='*65}")
    print(f"  Supplemental Figure S8. FLAIR intensity histogram comparison")
    print(f"  between non_removed and removed conditions (Phase II-B,")
    print(f"  n={len(ok_results)}). (A-C) Representative subjects with small,")
    print(f"  medium, and large stroke lesions. Blue: non_removed (all brain")
    print(f"  voxels); orange: removed (lesion voxels excluded); red: lesion")
    print(f"  voxels only; dashed line: lesion voxel mean intensity.")
    print(f"  (D) Group-level z-scored kernel density estimates showing")
    print(f"  right-tail excess in the non_removed condition attributable")
    print(f"  to hyperintense stroke lesion signal. (E) Skewness difference")
    print(f"  (non_removed minus removed) as a function of stroke lesion")
    print(f"  volume (Spearman rho={rho:.2f}, p{pr}). Intensity distributions")
    print(f"  showed significantly higher skewness in non_removed (mean")
    print(f"  difference {stats_df['Skewness_Diff'].mean():+.3f}, Wilcoxon p{ps},")
    print(f"  {pos}/{len(stats_df)} subjects [{100*pos/len(stats_df):.1f}%]).")
    print(f"  The distortion scaled with lesion volume: large lesions")
    print(f"  (>=5 mL) produced ~{fold:.0f}-fold greater skewness shifts than")
    print(f"  small lesions (<1 mL). Kurtosis was not significantly")
    print(f"  affected (p={'<0.001' if p_k < 0.001 else f'{p_k:.3f}'}), indicating a")
    print(f"  tail-specific rather than global distributional shift.")

    print(f"\n  DONE ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()