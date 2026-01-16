"""
================================================================================
Trimmed Correlation Analysis: Lesion Volume vs WMH Segmentation Differences
================================================================================

Paper Reference: "Robustness and Error Susceptibility of BIANCA for White 
Matter Hyperintensity Segmentation"

This script generates:
- Figure 5: Correlation between lesion volume and WMH segmentation differences
- Supplemental Table 1: Sensitivity analysis of correlation robustness

Key Finding (Section 3.2.3):
"Lesion volume correlated with differences between R and NR preprocessing 
across supratentorial pathologies. Ischemic infarcts demonstrated the strongest 
correlation (Spearman ρ = 0.84, p < 0.001, 95% CI [0.67, 0.93], n = 26). 
Lacunes showed similar patterns (Spearman ρ = 0.68, p < 0.001, 95% CI 
[0.45, 0.83], n = 35)."

Statistical Methodology (Section 2.8):
"To minimize outlier influence on correlation estimates, we computed trimmed 
Spearman correlations excluding cases above the 90th percentile of lesion 
volume (Wilcox, 2012; Pernet et al., 2012). Bootstrap 95% confidence intervals 
(1000 iterations) were calculated on the trimmed sample (Efron & Tibshirani, 1993)."

References:
- Efron, B., Tibshirani, R.J., 1993. An Introduction to the Bootstrap. 
  Chapman and Hall, New York.
- Pernet, C.R., Wilcox, R., Rousselet, G.A., 2012. Robust Correlation Analyses. 
  Front. Psychol. 3, 606.
- Wilcox, R.R., 2012. Introduction to Robust Estimation and Hypothesis Testing, 
  3rd ed. Academic Press, Amsterdam.

Author: Uchralt Temuulen
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define paths relative to project root (modify these for your environment)
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '.')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'correlation_analysis')

# Input data file
INPUT_FILE = os.path.join(DATA_DIR, 'all_results_post_processed.xlsx')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis parameters
# Paper Section 2.8: "excluding cases above the 90th percentile of lesion volume"
TRIM_PERCENTILE = 90
N_BOOTSTRAP = 1000
RANDOM_STATE = 42

# Outlier detection parameters (for visualization)
IQR_MULTIPLIER = 1.5
OUTLIER_METRIC = 'y'  # Detect outliers based on y-axis (volume difference)

# Lesion types to analyze (order for plots)
LESION_ORDER = ['lacune', 'infarct', 'infra', 'mixed']

# Display names for lesion types
LESION_DISPLAY_NAMES = {
    'infarct': 'Ischemic Infarcts',
    'lacune': 'Lacunes',
    'mixed': 'Mixed (Infarct + Lacune)',
    'infra': 'Infratentorial Infarcts'
}

# Column names
X_COLUMN = 'infarct_volume_ml'  # Stroke lesion volume
Y_COLUMN = 'WMH_volume_diff_abs'  # WMH volume difference (|NR - R|)
DICE_CRITERION = 'whole_dice_removal_better'  # Color coding criterion

# Figure parameters
DPI = 1000
FIG_WIDTH = 14
FIG_HEIGHT = 11


# ==============================================================================
# STATISTICAL FUNCTIONS
# ==============================================================================

def compute_trimmed_spearman_correlation(x, y, trim_percentile=90, 
                                          n_bootstrap=1000, random_state=42):
    """
    Compute Spearman correlation on trimmed data with bootstrap confidence intervals.
    
    Paper Reference (Section 2.8):
    "To minimize outlier influence on correlation estimates, we computed trimmed 
    Spearman correlations excluding cases above the 90th percentile of lesion 
    volume (Wilcox, 2012; Pernet et al., 2012). Bootstrap 95% confidence intervals 
    (1000 iterations) were calculated on the trimmed sample (Efron & Tibshirani, 1993)."
    
    Parameters
    ----------
    x : array-like
        Independent variable (lesion volume)
    y : array-like
        Dependent variable (WMH volume difference)
    trim_percentile : int
        Percentile threshold for trimming (default: 90)
    n_bootstrap : int
        Number of bootstrap iterations (default: 1000)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing correlation statistics
    """
    np.random.seed(random_state)
    x = np.array(x)
    y = np.array(y)
    
    # Remove missing values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    n_full = len(x_clean)
    
    # Check minimum sample size
    if n_full < 4:
        return {
            'n_full': n_full,
            'n_trimmed': 0,
            'n_excluded': 0,
            'threshold': np.nan,
            'rho': np.nan,
            'p': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }
    
    # Apply trimming: keep only cases <= 90th percentile of X (lesion volume)
    threshold = np.percentile(x_clean, trim_percentile)
    trim_mask = x_clean <= threshold
    
    x_trimmed = x_clean[trim_mask]
    y_trimmed = y_clean[trim_mask]
    
    n_trimmed = len(x_trimmed)
    n_excluded = n_full - n_trimmed
    
    if n_trimmed < 4:
        return {
            'n_full': n_full,
            'n_trimmed': n_trimmed,
            'n_excluded': n_excluded,
            'threshold': threshold,
            'rho': np.nan,
            'p': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }
    
    # Compute Spearman correlation on trimmed data
    rho_trimmed, p_trimmed = spearmanr(x_trimmed, y_trimmed)
    
    # Bootstrap confidence intervals on trimmed data
    bootstrap_rhos = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_trimmed, size=n_trimmed, replace=True)
        rho_boot, _ = spearmanr(x_trimmed[indices], y_trimmed[indices])
        bootstrap_rhos.append(rho_boot)
    
    bootstrap_rhos = np.array(bootstrap_rhos)
    ci_lower = np.percentile(bootstrap_rhos, 2.5)
    ci_upper = np.percentile(bootstrap_rhos, 97.5)
    
    return {
        'n_full': n_full,
        'n_trimmed': n_trimmed,
        'n_excluded': n_excluded,
        'threshold': threshold,
        'rho': rho_trimmed,
        'p': p_trimmed,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def identify_iqr_outliers(x, y, metric='y', iqr_multiplier=1.5):
    """
    Identify outliers using IQR method (for visualization purposes).
    
    Parameters
    ----------
    x : array-like
        X-axis values
    y : array-like
        Y-axis values
    metric : str
        Which variable to check for outliers ('x' or 'y')
    iqr_multiplier : float
        IQR multiplier for outlier threshold (default: 1.5)
        
    Returns
    -------
    dict
        Dictionary with outlier indices and count
    """
    x = np.array(x)
    y = np.array(y)
    
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    data_to_check = y_clean if metric.lower() == 'y' else x_clean
    
    n = len(data_to_check)
    if n < 5:
        return {'outlier_indices': [], 'n_outliers': 0, 'threshold': np.inf}
    
    q1 = np.percentile(data_to_check, 25)
    q3 = np.percentile(data_to_check, 75)
    iqr = q3 - q1
    threshold = q3 + (iqr_multiplier * iqr)
    
    is_outlier = data_to_check > threshold
    outlier_indices = np.where(is_outlier)[0].tolist()
    
    return {
        'outlier_indices': outlier_indices,
        'n_outliers': len(outlier_indices),
        'threshold': threshold
    }


def analyze_single_lesion_type(df_group, x_col, y_col, trim_percentile, 
                                iqr_multiplier, outlier_metric):
    """
    Calculate correlation statistics for a single lesion type.
    
    Parameters
    ----------
    df_group : pd.DataFrame
        Data subset for one lesion type
    x_col : str
        Column name for X variable
    y_col : str
        Column name for Y variable
    trim_percentile : int
        Percentile for trimming
    iqr_multiplier : float
        IQR multiplier for outlier detection
    outlier_metric : str
        Variable for outlier detection ('x' or 'y')
        
    Returns
    -------
    dict
        Complete statistics for the lesion type
    """
    x = df_group[x_col].values
    y = df_group[y_col].values
    
    # Compute trimmed correlation with bootstrap CI
    corr_stats = compute_trimmed_spearman_correlation(
        x, y, 
        trim_percentile=trim_percentile,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE
    )
    
    # Identify outliers for visualization
    outliers = identify_iqr_outliers(x, y, metric=outlier_metric, 
                                      iqr_multiplier=iqr_multiplier)
    
    # Map outlier indices back to dataframe indices
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    valid_indices = df_group.index[valid_mask].tolist()
    outlier_df_indices = [valid_indices[i] for i in outliers['outlier_indices']]
    
    return {
        'n_full': corr_stats['n_full'],
        'n_trimmed': corr_stats['n_trimmed'],
        'n_excluded': corr_stats['n_excluded'],
        'trim_threshold': corr_stats['threshold'],
        'rho': corr_stats['rho'],
        'p': corr_stats['p'],
        'ci_lower': corr_stats['ci_lower'],
        'ci_upper': corr_stats['ci_upper'],
        'n_outliers_iqr': outliers['n_outliers'],
        'outlier_df_indices': outlier_df_indices
    }


# ==============================================================================
# ANALYSIS PIPELINE
# ==============================================================================

def perform_correlation_analysis(df, group_col, group_order, x_col, y_col,
                                  trim_percentile, iqr_multiplier, outlier_metric):
    """
    Perform trimmed correlation analysis for all lesion types.
    
    Paper Reference (Section 3.2.3):
    "Lesion volume correlated with differences between R and NR preprocessing 
    across supratentorial pathologies."
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    group_col : str
        Column for grouping (lesion type)
    group_order : list
        Order of groups for analysis
    x_col : str
        X variable column name
    y_col : str
        Y variable column name
    trim_percentile : int
        Percentile for trimming
    iqr_multiplier : float
        IQR multiplier for outliers
    outlier_metric : str
        Variable for outlier detection
        
    Returns
    -------
    dict
        Analysis results for all groups
    """
    print("Performing trimmed correlation analysis...")
    analysis_results = {}
    
    for group in group_order:
        df_group = df[df[group_col] == group]
        
        if len(df_group) < 4:
            print(f"  {group}: n={len(df_group)} (insufficient data)")
            analysis_results[group] = {'n_full': len(df_group), 'insufficient_data': True}
            continue
        
        group_stats = analyze_single_lesion_type(
            df_group, x_col, y_col,
            trim_percentile, iqr_multiplier, outlier_metric
        )
        group_stats['insufficient_data'] = False
        analysis_results[group] = group_stats
        
        print(f"  {group}: n={group_stats['n_trimmed']} (trimmed), "
              f"ρ={group_stats['rho']:.2f}, p={group_stats['p']:.4f}")
    
    return analysis_results


# ==============================================================================
# TABLE GENERATION
# ==============================================================================

def generate_supplemental_table(analysis_results, group_display_names=None):
    """
    Generate Supplemental Table 1: Sensitivity analysis of correlation robustness.
    
    Paper Reference (Supplemental Table 1):
    "Trimmed Spearman correlations between stroke lesion volume and WMH volume 
    difference (R − NR), excluding cases above the 90th percentile of lesion volume."
    
    Parameters
    ----------
    analysis_results : dict
        Analysis results from perform_correlation_analysis
    group_display_names : dict
        Mapping from internal names to display names
        
    Returns
    -------
    pd.DataFrame
        Formatted table for publication
    """
    rows = []
    
    for group, stats in analysis_results.items():
        if stats.get('insufficient_data', False):
            continue
        
        display_name = group_display_names.get(group, group) if group_display_names else group
        
        # Format p-value
        p_val = stats['p']
        if np.isnan(p_val):
            p_str = "-"
        elif p_val < 0.001:
            p_str = '<0.001'
        elif p_val < 0.01:
            p_str = f'{p_val:.3f}'
        else:
            p_str = f'{p_val:.2f}'
        
        # Format confidence interval
        if np.isnan(stats['ci_lower']):
            ci_str = "-"
        else:
            ci_str = f"[{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]"
        
        # Format correlation coefficient
        rho_str = f"{stats['rho']:.2f}" if not np.isnan(stats['rho']) else "-"
        
        row = {
            'Lesion Type': display_name,
            'n': int(stats['n_trimmed']),
            'Spearman correlation': rho_str,
            'p-value': p_str,
            '95% CI': ci_str,
            'IQR Outliers': stats['n_outliers_iqr']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_detailed_table(analysis_results):
    """
    Generate detailed statistics table for internal use.
    
    Parameters
    ----------
    analysis_results : dict
        Analysis results from perform_correlation_analysis
        
    Returns
    -------
    pd.DataFrame
        Detailed statistics table
    """
    rows = []
    
    for group, stats in analysis_results.items():
        if stats.get('insufficient_data', False):
            continue
        
        row = {
            'Group': group,
            'n_full': stats['n_full'],
            'n_trimmed': stats['n_trimmed'],
            'n_excluded': stats['n_excluded'],
            'trim_threshold_ml': round(stats['trim_threshold'], 2) if not np.isnan(stats['trim_threshold']) else None,
            'rho': round(stats['rho'], 3) if not np.isnan(stats['rho']) else None,
            'p_value': stats['p'],
            'CI_lower': round(stats['ci_lower'], 3) if not np.isnan(stats['ci_lower']) else None,
            'CI_upper': round(stats['ci_upper'], 3) if not np.isnan(stats['ci_upper']) else None,
            'n_outliers_iqr': stats['n_outliers_iqr']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def format_p_value_string(p):
    """Format p-value for plot annotation."""
    if np.isnan(p):
        return "p = N/A"
    elif p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def create_correlation_figure(df, analysis_results, x_col, y_col, dice_criterion,
                               lesion_order, output_path=None, title="",
                               mark_outliers=True, outlier_metric='y'):
    """
    Create Figure 5: Correlation between lesion volume and WMH segmentation differences.
    
    Paper Reference (Figure 5):
    "Scatter plots demonstrate size-dependent scaling of preprocessing effects 
    across lesion etiologies. Trimmed Spearman correlations (excluding top 10% 
    of lesion volume differences) were significant for ischemic infarcts 
    (ρ = 0.84, 95% CI [0.67, 0.93], p < 0.001) and lacunes (ρ = 0.68, 95% CI 
    [0.45, 0.83], p < 0.001)."
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    analysis_results : dict
        Results from perform_correlation_analysis
    x_col : str
        X variable column name
    y_col : str
        Y variable column name
    dice_criterion : str
        Column for color coding points
    lesion_order : list
        Order of lesion types for subplots
    output_path : str
        Path to save figure
    title : str
        Figure title
    mark_outliers : bool
        Whether to mark IQR outliers
    outlier_metric : str
        Variable used for outlier detection
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Filter to available lesion types
    available_lesions = [lt for lt in lesion_order if lt in df['lesion_type'].unique()]
    
    n_plots = len(available_lesions)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, FIG_HEIGHT))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Color scheme for dice comparison
    color_map = {1: '#2ecc71', -1: '#e74c3c', 0: '#95a5a6'}
    label_map = {1: 'Removal Better', -1: 'Removal Worse', 0: 'No Change'}
    
    for idx, lesion_type in enumerate(available_lesions):
        ax = axes[idx]
        df_lesion = df[df['lesion_type'] == lesion_type].copy()
        
        if lesion_type not in analysis_results:
            ax.set_visible(False)
            continue
        
        group_stats = analysis_results[lesion_type]
        
        if group_stats.get('insufficient_data', False):
            ax.text(0.5, 0.5, f'n={len(df_lesion)}\n(insufficient data)',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(LESION_DISPLAY_NAMES.get(lesion_type, lesion_type))
            continue
        
        outlier_indices = group_stats['outlier_df_indices']
        
        # Plot data points by dice criterion
        for dice_val, color in color_map.items():
            mask_dice = df_lesion[dice_criterion] == dice_val
            mask_not_outlier = ~df_lesion.index.isin(outlier_indices)
            mask = mask_dice & mask_not_outlier
            
            if mask.sum() > 0:
                ax.scatter(
                    df_lesion.loc[mask, x_col],
                    df_lesion.loc[mask, y_col],
                    c=color, alpha=0.7, s=50, edgecolors='white', 
                    linewidth=0.5, zorder=2
                )
        
        # Mark outliers with circles
        if mark_outliers and len(outlier_indices) > 0:
            df_outliers = df_lesion.loc[outlier_indices]
            ax.scatter(
                df_outliers[x_col], df_outliers[y_col],
                facecolors='none', edgecolors='black', s=200, 
                linewidths=2.5, zorder=4
            )
            # Color fill for outliers
            for dice_val, color in color_map.items():
                mask = df_outliers[dice_criterion] == dice_val
                if mask.sum() > 0:
                    ax.scatter(
                        df_outliers.loc[mask, x_col], 
                        df_outliers.loc[mask, y_col],
                        c=color, alpha=0.9, s=50, edgecolors='white', 
                        linewidth=0.5, zorder=3
                    )
        
        # Add regression line and confidence band
        x_data = df_lesion[x_col].dropna()
        y_data = df_lesion[y_col].dropna()
        common_idx = x_data.index.intersection(y_data.index)
        x_plot = df_lesion.loc[common_idx, x_col].values
        y_plot = df_lesion.loc[common_idx, y_col].values
        
        if len(x_plot) > 2:
            slope, intercept, r_val, p_val, std_err = stats.linregress(x_plot, y_plot)
            
            x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color='#ff3333', linestyle='--', 
                    linewidth=2, zorder=1)
            
            # 95% confidence interval for regression
            n = len(x_plot)
            dof = n - 2
            t_score = stats.t.ppf(0.975, dof)
            y_model = slope * x_plot + intercept
            residuals = y_plot - y_model
            sy = np.sqrt(np.sum(residuals**2) / dof)
            mean_x = np.mean(x_plot)
            sum_sq_diff_x = np.sum((x_plot - mean_x)**2)
            ci_interval = t_score * sy * np.sqrt(1/n + (x_line - mean_x)**2 / sum_sq_diff_x)
            
            ax.fill_between(x_line, y_line - ci_interval, y_line + ci_interval,
                            color='#ff3333', alpha=0.15, zorder=0, edgecolor='none')
        
        # Statistics annotation box
        p_str = format_p_value_string(group_stats['p'])
        
        stats_text = f"n = {group_stats['n_trimmed']} (trimmed)\n"
        stats_text += f"Spearman ρ = {group_stats['rho']:.2f}\n"
        stats_text += f"{p_str}\n"
        stats_text += f"95% CI: [{group_stats['ci_lower']:.2f}, {group_stats['ci_upper']:.2f}]"
        
        if mark_outliers:
            stats_text += f"\n\nIQR Outliers: {group_stats['n_outliers_iqr']}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         alpha=0.9, edgecolor='gray'))
        
        # Axis labels
        ax.set_xlabel('Stroke Lesion Volume (mL)', fontsize=11)
        ax.set_ylabel('WMH Volume Difference (mL)\n(Non-Removed − Removed)', fontsize=11)
        ax.set_title(LESION_DISPLAY_NAMES.get(lesion_type, lesion_type), 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Create legend
    legend_handles = []
    for dice_val, color in color_map.items():
        legend_handles.append(
            mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                         markersize=8, label=label_map[dice_val])
        )
    
    if mark_outliers:
        legend_handles.append(
            mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                         markerfacecolor='none', markeredgewidth=2, 
                         markersize=10, label='IQR Outlier')
        )
    
    legend_handles.append(
        mlines.Line2D([], [], color='#ff3333', linestyle='--', 
                     linewidth=2, label='Trend line')
    )
    legend_handles.append(
        mpatches.Patch(color='#ff3333', alpha=0.15, label='95% confidence interval')
    )
    
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.02), frameon=True, fontsize=10)
    
    plt.suptitle(
        title if title else 'Trimmed Correlation: Stroke Lesion Volume vs WMH Segmentation Difference',
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"  Figure saved to: {output_path}")
    
    plt.close()
    return fig


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def main():
    """
    Main analysis pipeline for trimmed correlation analysis.
    
    Paper Reference:
    - Figure 5: Correlation between lesion volume and WMH segmentation differences
    - Supplemental Table 1: Sensitivity analysis of correlation robustness
    - Section 3.2.3: Size Effects and Segmentation Performance
    - Phase II-A (n=86): Detailed robustness assessment
    """
    print("=" * 80)
    print("TRIMMED CORRELATION ANALYSIS")
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
    
    df = pd.read_excel(INPUT_FILE, index_col=False)
    print(f"  Total subjects loaded: {len(df)}")
    
    # -------------------------------------------------------------------------
    # Filter for masked subjects
    # -------------------------------------------------------------------------
    print("\n[2] Filtering data...")
    
    if 'subject_with_mask' in df.columns:
        df_analysis = df[df['subject_with_mask'] == 1].copy()
        print(f"  Using masked subjects: n={len(df_analysis)}")
    else:
        df_analysis = df.copy()
        print(f"  Using all subjects: n={len(df_analysis)}")
    
    print(f"\n  Lesion type distribution:")
    print(df_analysis['lesion_type'].value_counts().to_string())
    
    # -------------------------------------------------------------------------
    # Perform correlation analysis
    # -------------------------------------------------------------------------
    print(f"\n[3] Computing trimmed correlations (≤{TRIM_PERCENTILE}th percentile)...")
    
    analysis_results = perform_correlation_analysis(
        df_analysis,
        group_col='lesion_type',
        group_order=LESION_ORDER,
        x_col=X_COLUMN,
        y_col=Y_COLUMN,
        trim_percentile=TRIM_PERCENTILE,
        iqr_multiplier=IQR_MULTIPLIER,
        outlier_metric=OUTLIER_METRIC
    )
    
    # -------------------------------------------------------------------------
    # Generate tables
    # -------------------------------------------------------------------------
    print("\n[4] Generating tables...")
    
    # Supplemental Table 1
    group_names = {
        'infarct': 'Ischemic infarcts',
        'lacune': 'Lacunes',
        'mixed': 'Mixed',
        'infra': 'Infratentorial'
    }
    
    supplemental_table = generate_supplemental_table(analysis_results, 
                                                      group_display_names=group_names)
    detailed_table = generate_detailed_table(analysis_results)
    
    # Save tables to Excel
    excel_path = os.path.join(OUTPUT_DIR, 'correlation_analysis_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        supplemental_table.to_excel(writer, sheet_name='Supplemental_Table_1', index=False)
        detailed_table.to_excel(writer, sheet_name='Detailed_Statistics', index=False)
    print(f"  Tables saved to: {excel_path}")
    
    # Display table
    print("\n" + "=" * 80)
    print("SUPPLEMENTAL TABLE 1: Sensitivity analysis of correlation robustness")
    print("=" * 80)
    print(supplemental_table.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # Generate figure
    # -------------------------------------------------------------------------
    print("\n[5] Generating Figure 5...")
    
    figure_path = os.path.join(OUTPUT_DIR, 'figure5_correlation_analysis.png')
    
    fig = create_correlation_figure(
        df_analysis,
        analysis_results,
        x_col=X_COLUMN,
        y_col=Y_COLUMN,
        dice_criterion=DICE_CRITERION,
        lesion_order=LESION_ORDER,
        output_path=figure_path,
        title=f"Trimmed Correlation Analysis (≤{TRIM_PERCENTILE}th percentile)",
        mark_outliers=True,
        outlier_metric=OUTLIER_METRIC
    )
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("KEY FINDINGS (Section 3.2.3)")
    print("=" * 80)
    
    for lesion in ['infarct', 'lacune']:
        if lesion in analysis_results and not analysis_results[lesion].get('insufficient_data'):
            stats = analysis_results[lesion]
            print(f"\n{LESION_DISPLAY_NAMES[lesion]}:")
            print(f"  Spearman ρ = {stats['rho']:.2f}")
            print(f"  p-value: {'<0.001' if stats['p'] < 0.001 else f'{stats[\"p\"]:.3f}'}")
            print(f"  95% CI: [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]")
            print(f"  n = {stats['n_trimmed']} (trimmed from {stats['n_full']})")
    
    print(f"\nPaper conclusion:")
    print("  'The impact of lesion removal was not random; it demonstrated robust")
    print("  size-dependent scaling (Spearman ρ 0.680.84, p < 0.001), where larger")
    print("  lesions produced proportionally larger segmentation discrepancies.'")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return analysis_results, supplemental_table


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    analysis_results, supplemental_table = main()