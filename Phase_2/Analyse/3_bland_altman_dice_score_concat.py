"""
================================================================================
Bland-Altman Analysis: Agreement Between R and NR Segmentation Approaches
================================================================================

Paper Reference: "Robustness and Error Susceptibility of BIANCA for White 
Matter Hyperintensity Segmentation"

This script generates:
- Figure 4: Bland-Altman Analysis - Total White Matter Hyperintensity
- Section 3.2.2: Overall agreement between R and NR approaches

Key Finding (Section 3.2.2):
"Bland-Altman analysis demonstrated systematic directional differences between 
R and NR segmentation approaches. For the primary metric (Dice coefficient), 
the 95% limits of agreement were narrow (−0.02 to +0.006) with near-zero 
median bias, indicating that lesion removal consistently produced marginally 
improved overlap with expert-delineated ground truth masks."

Statistical Methods (Section 2.8):
"Due to non-normal distribution of differences, non-parametric limits of 
agreement were calculated using the 2.5th and 97.5th percentiles with the 
median difference as the measure of central tendency (Giavarina, 2015)."

Author: Uchralt Temuulen
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, spearmanr

# Optional: Import custom plotting setup if available
try:
    from setup import setup_plot_styling
    setup_plot_styling()
except ImportError:
    print("Note: Custom plot styling not available, using defaults")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define paths relative to project root (modify these for your environment)
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '.')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'bland_altman_analysis')

# Input data file
INPUT_FILE = os.path.join(DATA_DIR, 'all_results_post_processed.xlsx')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Figure parameters
DPI = 1000
FIG_WIDTH = 20
FIG_HEIGHT = 5
FONTSIZE = 20

# Analysis parameters
# Y-axis range for performance metrics (Dice, Sensitivity, Precision)
# Based on paper Figure 4: narrow limits of agreement
PERFORMANCE_Y_RANGE = (-0.12, 0.06)


# ==============================================================================
# BLAND-ALTMAN ANALYSIS FUNCTIONS
# ==============================================================================

def calculate_nonparametric_limits(x, y):
    """
    Calculate non-parametric Bland-Altman limits of agreement.
    
    Paper Reference (Section 2.8):
    "Due to non-normal distribution of differences, non-parametric limits of 
    agreement were calculated using the 2.5th and 97.5th percentiles with the 
    median difference as the measure of central tendency (Giavarina, 2015)."
    
    Parameters
    ----------
    x : pd.Series
        First measurement series (Non-Removed condition)
    y : pd.Series
        Second measurement series (Removed condition)
        
    Returns
    -------
    tuple
        (lower_limit, upper_limit, median_difference)
    """
    differences = x - y
    differences = differences.dropna()
    
    if len(differences) == 0:
        return np.nan, np.nan, np.nan
    
    # Handle constant differences (edge case)
    if differences.nunique() == 1:
        constant_value = differences.iloc[0]
        return constant_value, constant_value, constant_value
    
    # Non-parametric 95% limits of agreement
    lower_limit = np.percentile(differences, 2.5)
    upper_limit = np.percentile(differences, 97.5)
    median_diff = np.median(differences)
    
    return lower_limit, upper_limit, median_diff


def create_bland_altman_subplot(ax, x, y, lesion_type, metric_name, display_name,
                                 unit="", force_y_range=None):
    """
    Create a single Bland-Altman subplot.
    
    Paper Reference (Figure 4):
    "Bland-Altman plots assess agreement between R and NR conditions across 
    Dice coefficient, sensitivity, and precision. Minimal systematic bias was 
    observed (95% LoA: −0.020 to 0.006 for Dice; −0.043 to 0.001 for sensitivity; 
    −0.001 to 0.033 for precision), with most points clustering near zero 
    difference. Color coding indicates lesion types."
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Subplot axes object
    x : pd.Series
        Non-Removed condition values
    y : pd.Series
        Removed condition values
    lesion_type : pd.Series
        Lesion type labels for color coding
    metric_name : str
        Internal metric name for results
    display_name : str
        Display name for axis labels
    unit : str
        Unit string for labels (e.g., "mL", "")
    force_y_range : tuple or None
        Optional fixed y-axis limits (min, max)
        
    Returns
    -------
    dict
        Statistics dictionary with limits of agreement
    """
    # Calculate Bland-Altman statistics
    mean_values = (x + y) / 2
    differences = x - y  # NR - R (positive = NR larger)
    lower_limit, upper_limit, median_diff = calculate_nonparametric_limits(x, y)
    
    # Identify outliers (outside 95% LoA)
    mask_outside = (differences > upper_limit) | (differences < lower_limit)
    
    # Color mapping by lesion type
    unique_lesions = np.unique(lesion_type.dropna())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_lesions)))
    color_map = {val: color for val, color in zip(unique_lesions, colors)}
    
    # Plot points by lesion type
    for lesion in unique_lesions:
        mask_lesion = lesion_type == lesion
        
        # Normal points (within LoA)
        ax.scatter(
            mean_values[mask_lesion & ~mask_outside],
            differences[mask_lesion & ~mask_outside],
            color=color_map[lesion],
            alpha=0.7,
            s=40,
            marker='o',
            label=f'{lesion}'
        )
        
        # Outlier points (outside LoA)
        ax.scatter(
            mean_values[mask_lesion & mask_outside],
            differences[mask_lesion & mask_outside],
            color=color_map[lesion],
            marker='x',
            s=80,
            alpha=0.9,
            linewidth=2
        )
    
    # Format limit values for display
    decimals = 2 if unit == "mL" else 3
    ll_text = f"{lower_limit:.{decimals}f}"
    ul_text = f"{upper_limit:.{decimals}f}"
    md_text = f"{median_diff:.{decimals}f}"
    
    # Add reference lines
    ax.axhline(y=lower_limit, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=upper_limit, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=median_diff, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add text annotations for limit values
    xlim = ax.get_xlim()
    text_x = xlim[0] + (xlim[1] - xlim[0]) * 0.02
    
    ax.text(text_x, lower_limit, f' {ll_text}', va='center', fontsize=8, 
            color='green', fontweight='bold')
    ax.text(text_x, upper_limit, f' {ul_text}', va='center', fontsize=8, 
            color='red', fontweight='bold')
    ax.text(text_x, median_diff, f' {md_text}', va='center', fontsize=8, 
            color='black', fontweight='bold')
    
    # Set y-axis limits
    if force_y_range is not None:
        ax.set_ylim(force_y_range)
    else:
        # Dynamic scaling with buffer
        y_max = max(upper_limit, differences.max())
        y_min = min(lower_limit, differences.min())
        range_span = y_max - y_min
        ax.set_ylim(y_min - (range_span * 0.1), y_max + (range_span * 0.1))
    
    # Axis labels and title
    unit_str = f" ({unit})" if unit else ""
    ax.set_xlabel(f'Mean {display_name}{unit_str}', fontsize=FONTSIZE)
    
    if unit == "mL":
        ax.set_ylabel('Difference (mL)', fontsize=FONTSIZE)
    else:
        ax.set_ylabel('Difference', fontsize=FONTSIZE)
    
    ax.set_title(f'{display_name}\n95% LoA: {ll_text} to {ul_text}', 
                 fontsize=FONTSIZE + 1, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    return {
        'metric': metric_name,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'median_diff': median_diff,
        'n_outliers': mask_outside.sum(),
        'n_total': len(differences.dropna())
    }


def create_combined_bland_altman_figure(df, output_dir):
    """
    Create combined Bland-Altman figure for performance metrics.
    
    Paper Reference (Figure 4):
    "Bland-Altman plots assess agreement between R and NR conditions across 
    Dice coefficient, sensitivity, and precision."
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with performance metric columns
    output_dir : str
        Directory to save output figure
        
    Returns
    -------
    pd.DataFrame
        Summary statistics for all metrics
    """
    print(f"\n{'=' * 80}")
    print("CREATING BLAND-ALTMAN ANALYSIS FIGURE")
    print(f"{'=' * 80}")
    
    # Define metrics to analyze
    # Format: (metric_name, nr_column, r_column, display_name, unit, y_range)
    metrics_config = [
        (
            "Dice",
            "WMH_dice_score_non_removed",
            "WMH_dice_score_removed",
            "Dice Score",
            "",
            PERFORMANCE_Y_RANGE
        ),
        (
            "Sensitivity",
            "WMH_sensitivity_non_removed",
            "WMH_sensitivity_removed",
            "Sensitivity",
            "",
            PERFORMANCE_Y_RANGE
        ),
        (
            "Precision",
            "WMH_precision_non_removed",
            "WMH_precision_removed",
            "Precision",
            "",
            PERFORMANCE_Y_RANGE
        )
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.suptitle(
        'Bland-Altman Analysis: Non-Removed − Removed',
        fontsize=FONTSIZE,
        fontweight='bold',
        y=1.05
    )
    
    stats_results = []
    
    # Generate each subplot
    for idx, (name, col_nr, col_r, display, unit, y_range) in enumerate(metrics_config):
        ax = axes[idx]
        
        # Filter valid data
        valid_mask = df[col_nr].notna() & df[col_r].notna()
        valid_data = df[valid_mask]
        
        print(f"  Plotting {name}: n={len(valid_data)} subjects")
        
        # Create subplot
        result = create_bland_altman_subplot(
            ax=ax,
            x=valid_data[col_nr],
            y=valid_data[col_r],
            lesion_type=valid_data["lesion_type"],
            metric_name=name,
            display_name=display,
            unit=unit,
            force_y_range=y_range
        )
        stats_results.append(result)
        
        # Only show y-label on first subplot
        if idx != 0:
            ax.set_ylabel("")
    
    # Create unified legend
    unique_lesions = np.unique(df["lesion_type"].dropna())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_lesions)))
    color_map = {val: color for val, color in zip(unique_lesions, colors)}
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[val],
                   markersize=8, label=val)
        for val in unique_lesions
    ]
    legend_elements.extend([
        plt.Line2D([0], [0], color='green', linestyle='--', label='Lower LoA'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Upper LoA'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Median'),
        plt.Line2D([0], [0], marker='x', color='gray', linestyle='None', label='Outliers')
    ])
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.13),
        ncol=6,
        fontsize=FONTSIZE
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save figure
    save_path = os.path.join(output_dir, "bland_altman_performance_metrics.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"\n  Figure saved to: {save_path}")
    plt.close()
    
    return pd.DataFrame(stats_results)


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def main():
    """
    Main analysis pipeline for Bland-Altman agreement analysis.
    
    Paper Reference:
    - Figure 4: Bland-Altman Analysis
    - Section 3.2.2: Overall agreement between R and NR approaches
    - Phase II-A (n=86): Detailed robustness assessment
    """
    print("=" * 80)
    print("BLAND-ALTMAN AGREEMENT ANALYSIS")
    print("Paper: Robustness and Error Susceptibility of BIANCA")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n[1] Loading data...")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            f"Please update the DATA_DIR path or ensure the file exists."
        )
    
    df = pd.read_excel(INPUT_FILE)
    print(f"  Total subjects loaded: {len(df)}")
    
    # -------------------------------------------------------------------------
    # Filter for masked subjects (Phase II-A analysis)
    # -------------------------------------------------------------------------
    print("\n[2] Filtering data...")
    
    if "subject_with_mask" in df.columns:
        df_analysis = df[df["subject_with_mask"] == 1].copy()
        print(f"  Using masked subjects: n={len(df_analysis)}")
    else:
        df_analysis = df.copy()
        print(f"  Using all subjects: n={len(df_analysis)}")
    
    # Standardize lesion type naming
    if 'lesion_type' in df_analysis.columns:
        df_analysis['lesion_type'] = df_analysis['lesion_type'].replace('ICB', 'ICH')
        print(f"\n  Lesion type distribution:")
        print(df_analysis['lesion_type'].value_counts().to_string())
    
    # -------------------------------------------------------------------------
    # Generate Bland-Altman analysis
    # -------------------------------------------------------------------------
    print("\n[3] Generating Bland-Altman analysis...")
    
    stats_df = create_combined_bland_altman_figure(df_analysis, OUTPUT_DIR)
    
    # -------------------------------------------------------------------------
    # Display summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    
    # Save statistics
    stats_path = os.path.join(OUTPUT_DIR, "bland_altman_statistics.xlsx")
    stats_df.to_excel(stats_path, index=False)
    print(f"\n  Statistics saved to: {stats_path}")
    
    # -------------------------------------------------------------------------
    # Key findings
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("KEY FINDINGS (Section 3.2.2)")
    print("=" * 80)
    
    dice_stats = stats_df[stats_df['metric'] == 'Dice'].iloc[0]
    print(f"\nDice Coefficient:")
    print(f"  95% LoA: {dice_stats['lower_limit']:.3f} to {dice_stats['upper_limit']:.3f}")
    print(f"  Median bias: {dice_stats['median_diff']:.3f}")
    print(f"  Outliers: {dice_stats['n_outliers']}/{dice_stats['n_total']}")
    
    print(f"\nPaper conclusion:")
    print("  'Lesion removal consistently produced marginally improved overlap")
    print("  with expert-delineated ground truth masks.'")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return stats_df


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    stats_df = main()