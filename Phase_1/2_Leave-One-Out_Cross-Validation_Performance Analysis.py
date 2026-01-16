"""
================================================================================
Leave-One-Out Cross-Validation Performance Analysis
================================================================================

Paper Reference: "Robustness and Error Susceptibility of BIANCA for White 
Matter Hyperintensity Segmentation"

This script generates:
- Supplemental Figure 1: DICE Score comparison across threshold strategies 
  with and without lesion removal

Key Finding (Section 3.1):
"When evaluating the three WMH segmentation algorithms (B_0.9, B_0.85, and B+L), 
training with R images consistently improved performance compared to training 
with NR images, though improvements were modest across all approaches. B_0.85 
showed mean Dice improvement from 0.59 to 0.61 (3.4% increase), B_0.9 improved 
from 0.54 to 0.58 (7.4% increase), while BIANCA+LOCATE remained essentially 
unchanged (0.58 vs 0.59, 1.7% increase)."

Methodology (Section 2.4):
"To assess whether the presence of lesions affects WMH segmentation accuracy, 
we employed leave-one-out cross-validation (LOO) on the full cohort (n=103), 
as originally proposed by Anbeek et al. (2004) and implemented within BIANCA 
(Griffanti et al., 2016)."

Bootstrap CI Reference:
- Efron, B., Tibshirani, R.J., 1993. An Introduction to the Bootstrap. 
  Chapman and Hall, New York.

Author: Uchralt Temuulen
================================================================================
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
from typing import Optional, List, Tuple


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define paths relative to project root
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '.')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'loo_performance')

# Input data file (LOO cross-validation results)
INPUT_FILE = os.path.join(DATA_DIR, 'loo_results.xlsx')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bootstrap parameters
N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# Plotting parameters
THRESHOLD_ORDER = ["locate", "0.90", "0.85"]
HUE_ORDER = ["Trained with removal", "Trained without removal"]

# Global font size for plots
FONT_SIZE = 12


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def format_threshold(x):
    """
    Format threshold values for consistent display.
    
    Converts numeric thresholds (e.g., 90, 85) to decimal format (0.90, 0.85)
    and keeps 'locate' as is.
    
    Parameters
    ----------
    x : str or numeric
        Raw threshold value
        
    Returns
    -------
    str
        Formatted threshold string
    """
    if str(x).lower() == "locate":
        return "locate"
    try:
        pct = float(x) / 100.0
        return f"{pct:.2f}"
    except ValueError:
        return str(x)


def bootstrap_ci(series: pd.Series,
                 n_iter: int = 10000,
                 alpha: float = 0.05,
                 seed: int = 42) -> Tuple[float, float]:
    """
    Calculate percentile bootstrap confidence interval for the mean.
    
    Reference: Efron & Tibshirani (1993) - An Introduction to the Bootstrap
    
    Parameters
    ----------
    series : pd.Series
        Data to bootstrap
    n_iter : int
        Number of bootstrap iterations
    alpha : float
        Significance level (default: 0.05 for 95% CI)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound) of confidence interval
    """
    rng = np.random.default_rng(seed)
    resampled = rng.choice(series.values, size=(n_iter, series.size), replace=True)
    means = resampled.mean(axis=1)
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def calculate_group_statistics(df: pd.DataFrame,
                                group_cols: List[str],
                                value_col: str,
                                n_iter: int = 10000,
                                alpha: float = 0.05,
                                seed: int = 42) -> pd.DataFrame:
    """
    Calculate descriptive statistics with bootstrap CI for grouped data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    group_cols : list
        Columns to group by
    value_col : str
        Column containing values to analyze
    n_iter : int
        Number of bootstrap iterations
    alpha : float
        Significance level for CI
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Statistics including mean, SD, n, CI bounds
    """
    stats_list = []
    
    for group_name, group_data in df.groupby(group_cols, observed=False):
        series = group_data[value_col]
        ci_low, ci_high = bootstrap_ci(series, n_iter=n_iter, alpha=alpha, seed=seed)
        
        # Handle single vs multiple group columns
        if isinstance(group_name, tuple):
            row_dict = dict(zip(group_cols, group_name))
        else:
            row_dict = {group_cols[0]: group_name}
        
        row_dict.update({
            'mean': series.mean(),
            'sd': series.std(ddof=1),
            'median': series.median(),
            'q1': series.quantile(0.25),
            'q3': series.quantile(0.75),
            'n': len(series),
            'ci_low': ci_low,
            'ci_high': ci_high
        })
        stats_list.append(row_dict)
    
    return pd.DataFrame(stats_list)


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_metric_violinplot(df: pd.DataFrame,
                           metric_col: str,
                           y_label: str,
                           filename: str,
                           title: Optional[str] = None,
                           palette: Optional[List[str]] = None,
                           show_bootstrap_ci: bool = True,
                           n_iter: int = 10000):
    """
    Create violin plot with mean, SD, and bootstrap CI overlay.
    
    Paper Reference: Supplemental Figure 1
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: threshold, TRAIN_SET, and metric_col
    metric_col : str
        Column name for the metric to plot (e.g., 'Dice Score')
    y_label : str
        Y-axis label
    filename : str
        Output file path
    title : str, optional
        Plot title
    palette : list, optional
        Colors for [with_removal, without_removal]
    show_bootstrap_ci : bool
        Whether to show bootstrap confidence intervals
    n_iter : int
        Number of bootstrap iterations
    """
    # Set up colors
    default_blue = mpl.rcParams['axes.prop_cycle'].by_key()['color'][0]
    if palette is None:
        palette = ["orange", default_blue]
    
    # Calculate statistics with bootstrap CI
    stats = calculate_group_statistics(
        df, 
        ["threshold", "TRAIN_SET"], 
        metric_col,
        n_iter=n_iter
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create violin plot
    sns.violinplot(
        data=df,
        x="threshold",
        y=metric_col,
        hue="TRAIN_SET",
        order=THRESHOLD_ORDER,
        hue_order=HUE_ORDER,
        palette=palette,
        split=True,
        inner=None,
        density_norm="width",
        cut=0,
        linewidth=1,
        ax=ax
    )
    
    # Overlay statistics on each violin
    for _, row in stats.iterrows():
        thr_idx = THRESHOLD_ORDER.index(row["threshold"])
        hue_idx = HUE_ORDER.index(row["TRAIN_SET"])
        x = thr_idx + (-0.2 if hue_idx == 0 else 0.2)
        
        # Mean marker (black dot)
        ax.plot(x, row["mean"], marker='o', color='black', markersize=5, zorder=10)
        
        # SD whiskers (solid lines)
        ax.hlines([row["mean"] - row["sd"], row["mean"] + row["sd"]],
                  x - 0.04, x + 0.04, color='black', linewidth=1.4, zorder=9)
        
        # Bootstrap CI (dotted lines)
        if show_bootstrap_ci:
            ax.hlines([row["ci_low"], row["ci_high"]],
                      x - 0.04, x + 0.04,
                      color='black', linestyle=':', linewidth=1.4, zorder=9)
    
    # Axis formatting
    ax.set_xlabel("Threshold strategy", fontsize=FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE)
    ax.set_title(
        title if title else f"{y_label} by threshold (Leave-one-out)",
        fontsize=FONT_SIZE + 2, pad=20
    )
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    
    # Clean up spines and add grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    ax.xaxis.grid(False)
    
    # Create custom legend
    custom_handles = [
        Patch(facecolor=palette[0], label=HUE_ORDER[0]),
        Patch(facecolor=palette[1], label=HUE_ORDER[1]),
        Line2D([0], [0], marker='o', color='black', linestyle='None', 
               label='Mean', markersize=6),
        Line2D([0], [0], color='black', linewidth=1.4, label='± SD'),
    ]
    
    if show_bootstrap_ci:
        custom_handles.append(
            Line2D([0], [0], color='black', linestyle=':', linewidth=1.4, 
                   label='95% CI')
        )
    
    ax.legend(
        handles=custom_handles,
        title="Legend",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        frameon=True,
        fontsize=FONT_SIZE - 2
    )
    
    # Save figure
    fig.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Plot saved to: {filename}")
    
    return stats


def generate_statistics_summary(stats: pd.DataFrame, metric_name: str) -> str:
    """
    Generate text summary of statistics for paper methods section.
    
    Parameters
    ----------
    stats : pd.DataFrame
        Statistics dataframe from calculate_group_statistics
    metric_name : str
        Name of the metric being summarized
        
    Returns
    -------
    str
        Formatted summary text
    """
    lines = [
        f"\n{metric_name} Statistics Summary",
        "=" * 60,
        f"{'Threshold':<10} {'Training':<25} {'N':<5} {'Mean±SD':<15} {'95% CI':<20}",
        "-" * 60
    ]
    
    for _, row in stats.iterrows():
        lines.append(
            f"{row['threshold']:<10} "
            f"{row['TRAIN_SET']:<25} "
            f"{int(row['n']):<5} "
            f"{row['mean']:.3f}±{row['sd']:.3f}    "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )
    
    return "\n".join(lines)


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def main():
    """
    Main analysis pipeline for LOO cross-validation performance.
    
    Paper Reference:
    - Supplemental Figure 1: DICE Score comparison across threshold strategies
    - Section 3.1: Algorithm selection and lesion removal effects
    - Section 2.4: Leave-one-out cross-validation methodology
    """
    print("=" * 80)
    print("LEAVE-ONE-OUT CROSS-VALIDATION PERFORMANCE ANALYSIS")
    print("Paper: Robustness and Error Susceptibility of BIANCA")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  Confidence level: {CONFIDENCE_LEVEL}")
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n[1] Loading LOO results...")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            f"Please update the DATA_DIR path or ensure the file exists."
        )
    
    df = pd.read_excel(INPUT_FILE, index_col=0)
    
    # Format threshold values
    df["threshold"] = df["threshold"].apply(format_threshold)
    
    # Filter to allowed thresholds
    df = df[df["threshold"].isin(THRESHOLD_ORDER)].copy()
    
    # Rename training set labels for consistency
    df["TRAIN_SET"] = df["TRAIN_SET"].replace({
        "BIANCA trained with removal": "Trained with removal",
        "BIANCA trained without removal": "Trained without removal",
        "with_removal": "Trained with removal",
        "without_removal": "Trained without removal"
    })
    
    # Rename columns for plotting
    df = df.rename(columns={
        'dice_score': 'Dice Score',
        'sensitivity': 'Sensitivity',
        'precision': 'Precision'
    })
    
    print(f"  Total samples: {len(df)}")
    print(f"  Unique subjects: {df['subject_id'].nunique()}")
    print(f"  Thresholds: {df['threshold'].unique().tolist()}")
    
    # -------------------------------------------------------------------------
    # Define color palettes
    # -------------------------------------------------------------------------
    default_blue = mpl.rcParams['axes.prop_cycle'].by_key()['color'][0]
    palette_dice = ["orange", default_blue]
    palette_sens = ["#66c2a5", "#fc8d62"]
    palette_prec = ["#8da0cb", "#fc8d62"]
    
    # -------------------------------------------------------------------------
    # Generate Supplemental Figure 1: Dice Score comparison
    # -------------------------------------------------------------------------
    print("\n[2] Generating Supplemental Figure 1: Dice Score comparison...")
    
    dice_stats = plot_metric_violinplot(
        df,
        metric_col="Dice Score",
        y_label="Dice Score",
        filename=os.path.join(OUTPUT_DIR, "supplemental_figure1_dice_loo.png"),
        title="DICE Score comparison across threshold strategies\n(Leave-one-out cross-validation, n=103)",
        palette=palette_dice,
        show_bootstrap_ci=True,
        n_iter=N_BOOTSTRAP
    )
    
    print(generate_statistics_summary(dice_stats, "Dice Score"))
    
    # -------------------------------------------------------------------------
    # Additional metrics: Sensitivity and Precision
    # -------------------------------------------------------------------------
    print("\n[3] Generating additional metric plots...")
    
    # Sensitivity
    sens_stats = plot_metric_violinplot(
        df,
        metric_col="Sensitivity",
        y_label="Sensitivity",
        filename=os.path.join(OUTPUT_DIR, "sensitivity_loo.png"),
        title="Sensitivity by threshold (Leave-one-out)",
        palette=palette_sens,
        show_bootstrap_ci=True,
        n_iter=N_BOOTSTRAP
    )
    
    print(generate_statistics_summary(sens_stats, "Sensitivity"))
    
    # Precision
    prec_stats = plot_metric_violinplot(
        df,
        metric_col="Precision",
        y_label="Precision",
        filename=os.path.join(OUTPUT_DIR, "precision_loo.png"),
        title="Precision by threshold (Leave-one-out)",
        palette=palette_prec,
        show_bootstrap_ci=True,
        n_iter=N_BOOTSTRAP
    )
    
    print(generate_statistics_summary(prec_stats, "Precision"))
    
    # -------------------------------------------------------------------------
    # Save statistics to Excel
    # -------------------------------------------------------------------------
    print("\n[4] Saving statistics to Excel...")
    
    # Combine all statistics
    dice_stats['metric'] = 'Dice Score'
    sens_stats['metric'] = 'Sensitivity'
    prec_stats['metric'] = 'Precision'
    
    all_stats = pd.concat([dice_stats, sens_stats, prec_stats], ignore_index=True)
    
    stats_path = os.path.join(OUTPUT_DIR, "loo_statistics_with_bootstrap_ci.xlsx")
    all_stats.to_excel(stats_path, index=False)
    print(f"  Statistics saved to: {stats_path}")
    
    # -------------------------------------------------------------------------
    # Key findings summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("KEY FINDINGS (Section 3.1)")
    print("=" * 80)
    
    # Calculate improvement percentages for paper
    for metric in ['Dice Score', 'Sensitivity', 'Precision']:
        metric_stats = all_stats[all_stats['metric'] == metric]
        
        for thresh in THRESHOLD_ORDER:
            thresh_stats = metric_stats[metric_stats['threshold'] == thresh]
            
            if len(thresh_stats) == 2:
                with_removal = thresh_stats[thresh_stats['TRAIN_SET'] == 'Trained with removal']['mean'].values[0]
                without_removal = thresh_stats[thresh_stats['TRAIN_SET'] == 'Trained without removal']['mean'].values[0]
                
                if without_removal > 0:
                    improvement = ((with_removal - without_removal) / without_removal) * 100
                    print(f"  {metric} @ {thresh}: {without_removal:.3f} → {with_removal:.3f} ({improvement:+.1f}%)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return all_stats


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    all_stats = main()