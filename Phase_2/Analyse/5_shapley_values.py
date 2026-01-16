"""
================================================================================
SHAP Feature Importance Analysis: Factors Influencing WMH Segmentation Differences
================================================================================

Paper Reference: "Robustness and Error Susceptibility of BIANCA for White 
Matter Hyperintensity Segmentation"

This script generates:
- Figure 6: SHAP feature importance analysis
- Supplemental Figure 2: SHAP feature importance (Phase II-B expanded cohort)

Key Finding (Section 3.2.4):
"Lesion volume emerged as the dominant predictor (SHAP value = 0.232, 81.4% 
relative importance) substantially outweighing all other factors. WMH lesion 
burden assessed by ARWMC score ranked second in importance (0.025, 8.6%), 
followed by patient age (0.015, 5.3%) and brain volume (0.007, 2.5%)."

Critical Discovery (Section 3.3.6):
"Scanner type showed minimal influence in Phase II-A analysis (SHAP value = 0.002, 
0.8%). However, scanner importance increased 31-fold (0.002 → 0.062) from 
Phase II-A to Phase II-B as Philips scanner representation increased from 
9.3% to 17.1%."

Statistical Methods (Section 2.8):
"Feature importance analysis (lesion volume, scanner type, age, ARWMC score) 
regarding segmentation differences was quantified using SHAP (SHapley Additive 
exPlanations) values with TreeExplainer (Lundberg & Lee, 2017). A Random Forest 
model (500 trees, max depth=10) was optimized via grid search with 5-fold 
cross-validation to capture non-linear relationships between variables 
(Breiman, 2001)."

References:
- Lundberg, S.M., Lee, S.-I., 2017. A unified approach to interpreting model 
  predictions. In: Advances in Neural Information Processing Systems 30, 
  pp. 4765-4774.
- Breiman, L., 2001. Random forests. Mach. Learn. 45, 5-32.

Author: Uchralt Temuulen
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define paths relative to project root (modify these for your environment)
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '.')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'shap_analysis')

# Input data file
INPUT_FILE = os.path.join(DATA_DIR, 'all_results_post_processed.xlsx')

# Create output directories
OUTPUT_DIR_MASKED = os.path.join(OUTPUT_DIR, 'masked_subjects')
OUTPUT_DIR_FULL = os.path.join(OUTPUT_DIR, 'full_cohort')
os.makedirs(OUTPUT_DIR_MASKED, exist_ok=True)
os.makedirs(OUTPUT_DIR_FULL, exist_ok=True)

# Target variable
TARGET_COLUMN = 'WMH_vol_diff'
TARGET_DISPLAY_NAME = "WMH Volume Difference"

# Feature columns for SHAP analysis
# Paper Section 2.8: "Feature importance analysis (lesion volume, scanner type, 
# age, ARWMC score)"
FEATURE_COLUMNS = [
    'ARWMC',           # Age-Related White Matter Changes score (Wahlund et al., 2001)
    'sex',             # Patient sex
    'age',             # Patient age
    'infarct_volume',  # Stroke lesion volume
    'brain_volume',    # Total brain volume
    'scanner',         # Scanner type (encoded)
    'lesion_type'      # Lesion type (encoded)
]

# Category mappings for encoding
# Paper Section 2.2: Scanner distribution
SCANNER_MAPPING = {
    'Prisma_fit': 1,  # Siemens
    'Tim Trio': 1,    # Siemens
    'Philips': 2      # Philips
}

LESION_TYPE_MAPPING = {
    'infra': 1,
    'lacune': 2,
    'infarct': 3,
    'mixed': 4,
    'ICH': 5
}

SEX_MAPPING = {
    'Women': 0,
    'Men': 1
}

# Random Forest parameters
# Paper Section 2.8: "Random Forest model (500 trees, max depth=10)"
RF_N_ESTIMATORS = 100  # Using 100 for computational efficiency
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42

# Figure parameters
DPI = 1000
FIG_WIDTH = 10
FIG_HEIGHT = 6


# ==============================================================================
# SHAP STATISTICS FUNCTIONS
# ==============================================================================

def calculate_shap_statistics(shap_values, feature_names):
    """
    Calculate mean, standard deviation, and confidence intervals for SHAP values.
    
    Paper Reference (Section 2.8):
    "Feature importance was quantified using SHAP values with TreeExplainer"
    
    Parameters
    ----------
    shap_values : np.array
        SHAP values array (n_samples, n_features)
    feature_names : list
        List of feature names
        
    Returns
    -------
    dict
        Dictionary with statistics for each feature
    """
    abs_shap = np.abs(shap_values)
    
    stats_dict = {}
    for i, feature in enumerate(feature_names):
        shap_feature = abs_shap[:, i]
        
        mean_val = shap_feature.mean()
        std_val = shap_feature.std()
        sem_val = sem(shap_feature)
        ci_95 = 1.96 * sem_val  # 95% confidence interval
        
        stats_dict[feature] = {
            'mean': mean_val,
            'std': std_val,
            'sem': sem_val,
            'ci_95': ci_95,
            'values': shap_feature
        }
    
    return stats_dict


def calculate_percentage_importance(stats_dict):
    """
    Calculate percentage importance from SHAP statistics.
    
    Paper Reference (Section 3.2.4):
    "Lesion volume emerged as the dominant predictor (SHAP value = 0.232, 
    81.4% relative importance)"
    
    Parameters
    ----------
    stats_dict : dict
        Statistics dictionary from calculate_shap_statistics
        
    Returns
    -------
    dict
        Dictionary mapping features to percentage importance
    """
    means = {f: stats_dict[f]['mean'] for f in stats_dict}
    total = sum(means.values())
    
    return {f: (v / total) * 100 for f, v in means.items()}


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def create_importance_barplot_with_ci(shap_values, feature_names, output_dir,
                                       target_name="Target", figsize=(10, 6)):
    """
    Create bar plot showing feature importance with 95% confidence intervals.
    
    This generates Figure 6 from the paper:
    "SHAP feature importance analysis showing lesion volume as the dominant 
    factor influencing WMH volume differences"
    
    Parameters
    ----------
    shap_values : np.array
        SHAP values array
    feature_names : list
        Feature names
    output_dir : str
        Directory to save figure
    target_name : str
        Display name for target variable
    figsize : tuple
        Figure size
        
    Returns
    -------
    dict
        Statistics dictionary for all features
    """
    # Calculate statistics
    stats_dict = calculate_shap_statistics(shap_values, feature_names)
    pct_importance = calculate_percentage_importance(stats_dict)
    
    # Extract values for plotting
    means = np.array([stats_dict[f]['mean'] for f in feature_names])
    ci_95s = np.array([stats_dict[f]['ci_95'] for f in feature_names])
    percentages = np.array([pct_importance[f] for f in feature_names])
    
    # Sort by importance (descending)
    sorted_idx = np.argsort(means)[::-1]
    
    features_sorted = [feature_names[i] for i in sorted_idx]
    means_sorted = means[sorted_idx]
    ci_sorted = ci_95s[sorted_idx]
    pct_sorted = percentages[sorted_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(features_sorted)))
    y_pos = np.arange(len(features_sorted))
    
    # Bar plot with error bars
    bars = ax.barh(y_pos, means_sorted, xerr=ci_sorted, color=colors,
                    capsize=5, error_kw={'linewidth': 2, 'elinewidth': 2})
    
    # Add value labels
    for i, (mean_val, ci_val, pct_val) in enumerate(zip(means_sorted, ci_sorted, pct_sorted)):
        label_text = f'{mean_val:.3f} ({pct_val:.1f}%)'
        ax.text(mean_val + ci_val + 0.01, i, label_text, va='center',
                fontsize=9, fontweight='bold')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'SHAP Feature Importance - {target_name}\n(with 95% confidence intervals)',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Set x-axis limit with padding
    max_val = (means_sorted + ci_sorted).max()
    ax.set_xlim(0, max_val * 1.25)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'shap_feature_importance_with_ci.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    return stats_dict


def create_percentage_barplot(shap_values, feature_names, output_dir,
                               target_name="Target", figsize=(10, 6)):
    """
    Create bar plot showing feature importance as percentage of total influence.
    
    Parameters
    ----------
    shap_values : np.array
        SHAP values array
    feature_names : list
        Feature names
    output_dir : str
        Output directory
    target_name : str
        Target variable display name
    figsize : tuple
        Figure size
        
    Returns
    -------
    dict
        Percentage importance for each feature
    """
    stats_dict = calculate_shap_statistics(shap_values, feature_names)
    pct_importance = calculate_percentage_importance(stats_dict)
    
    # Sort by percentage
    sorted_features = sorted(pct_importance.keys(),
                             key=lambda x: pct_importance[x], reverse=True)
    sorted_pct = [pct_importance[f] for f in sorted_features]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(sorted_features)))
    bars = ax.barh(sorted_features, sorted_pct, color=colors)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, sorted_pct)):
        ax.text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Percentage Influence (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance for {target_name}\n(% of Total SHAP Value Magnitude)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(sorted_pct) * 1.15)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'shap_feature_importance_percentage.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    return pct_importance


def create_shap_summary_plot(X, shap_values, feature_names, output_dir):
    """
    Create SHAP summary plot (beeswarm/violin style).
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    shap_values : np.array
        SHAP values
    feature_names : list
        Feature names
    output_dir : str
        Output directory
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'shap_summary_beeswarm.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_pie_chart(shap_values, feature_names, output_dir,
                     target_name="Target", figsize=(10, 8)):
    """
    Create pie chart showing feature influence distribution.
    
    Parameters
    ----------
    shap_values : np.array
        SHAP values
    feature_names : list
        Feature names
    output_dir : str
        Output directory
    target_name : str
        Target display name
    figsize : tuple
        Figure size
    """
    stats_dict = calculate_shap_statistics(shap_values, feature_names)
    pct_importance = calculate_percentage_importance(stats_dict)
    
    # Sort and prepare data
    sorted_items = sorted(pct_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Keep top 6, combine rest as "Other"
    threshold = 6
    if len(sorted_items) > threshold:
        top_items = sorted_items[:threshold]
        other_pct = sum(item[1] for item in sorted_items[threshold:])
        features_to_plot = [item[0] for item in top_items] + ['Other']
        percentages_to_plot = [item[1] for item in top_items] + [other_pct]
    else:
        features_to_plot = [item[0] for item in sorted_items]
        percentages_to_plot = [item[1] for item in sorted_items]
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    colors = plt.cm.Set3(np.linspace(0, 1, len(features_to_plot)))
    
    wedges, texts, autotexts = ax.pie(
        percentages_to_plot, labels=features_to_plot,
        autopct='%1.1f%%', colors=colors, startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    
    ax.set_title(f'Feature Influence Distribution for {target_name}',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'shap_feature_influence_pie.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_waterfall_plot(explainer, X, shap_values, feature_names, output_dir,
                          target_name="Target", sample_idx=0):
    """
    Create waterfall plot for a single prediction example.
    
    Parameters
    ----------
    explainer : shap.Explainer
        SHAP explainer object
    X : pd.DataFrame
        Feature matrix
    shap_values : np.array
        SHAP values
    feature_names : list
        Feature names
    output_dir : str
        Output directory
    target_name : str
        Target display name
    sample_idx : int
        Index of sample to visualize
    """
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[0]
    
    sv = shap_values
    if isinstance(sv, list):
        sv = sv[0]
    
    # Create SHAP explanation object
    shap_explanation = shap.Explanation(
        values=sv[sample_idx],
        base_values=expected_value,
        data=X.iloc[sample_idx],
        feature_names=feature_names
    )
    
    fig = plt.figure(figsize=(10, 6), dpi=DPI)
    shap.waterfall_plot(shap_explanation, show=False)
    
    plt.title(f'Individual Prediction: {target_name} (Sample {sample_idx})',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'shap_waterfall_example.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ==============================================================================
# DATA PREPARATION FUNCTIONS
# ==============================================================================

def load_and_prepare_data():
    """
    Load data and apply category encodings.
    
    Returns
    -------
    pd.DataFrame
        Prepared dataframe with encoded categories
    """
    print("\n[1] Loading data...")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            f"Please update the DATA_DIR path or ensure the file exists."
        )
    
    df = pd.read_excel(INPUT_FILE, index_col=False)
    print(f"  Loaded {len(df)} subjects")
    
    # Apply category encodings
    print("\n[2] Applying category encodings...")
    df['scanner_int'] = df['scanner'].map(SCANNER_MAPPING)
    df['lesion_type_int'] = df['lesion_type'].map(LESION_TYPE_MAPPING)
    df['sex_int'] = df['sex'].map(SEX_MAPPING)
    df['ROI_Volume'] = df['infarct_volume_ml']
    
    # Check data quality
    print(f"  Subjects with mask: {(df['subject_with_mask'] == 1).sum()}")
    if 'DATASET' in df.columns:
        print(f"  Full cohort subjects: {(df['DATASET'] == 'NULLING').sum()}")
    
    return df


def prepare_feature_matrix(df, dataset_type='MASKED'):
    """
    Prepare feature matrix for a specific dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe
    dataset_type : str
        'MASKED' for Phase II-A (n=86) or 'FULL' for Phase II-B (n=211)
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataset with renamed columns
    """
    # Filter dataset
    if dataset_type == 'MASKED':
        df_subset = df[df['subject_with_mask'] == 1].copy()
        print(f"\n  Phase II-A (Masked): {len(df_subset)} subjects")
    else:
        if 'DATASET' in df.columns:
            df_subset = df[df['DATASET'] == 'NULLING'].copy()
        else:
            df_subset = df.copy()
        print(f"\n  Phase II-B (Full): {len(df_subset)} subjects")
    
    # Rename columns to match FEATURE_COLUMNS
    df_subset['ARWMC'] = df_subset['Wahlund']
    df_subset['sex'] = df_subset['sex_int']
    df_subset['infarct_volume'] = df_subset['ROI_Volume']
    df_subset['brain_volume'] = df_subset['brain_volume_ml']
    df_subset['scanner'] = df_subset['scanner_int']
    df_subset['lesion_type'] = df_subset['lesion_type_int']
    
    # Remove rows with missing values
    df_clean = df_subset.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    print(f"  After removing NaN: {len(df_clean)} subjects")
    
    return df_clean


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def train_model_and_compute_shap(df_clean, dataset_name='Dataset'):
    """
    Train Random Forest model and compute SHAP values.
    
    Paper Reference (Section 2.8):
    "A Random Forest model (500 trees, max depth=10) was optimized via grid 
    search with 5-fold cross-validation to capture non-linear relationships 
    between variables (Breiman, 2001)."
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned dataset
    dataset_name : str
        Name for logging
        
    Returns
    -------
    tuple
        (model, shap_values, X_scaled, y, explainer, X_unscaled)
    """
    print(f"\n[3] Training Random Forest model ({dataset_name})...")
    
    # Prepare features and target
    X = df_clean[FEATURE_COLUMNS].copy()
    y = df_clean[TARGET_COLUMN].copy()
    
    print(f"  Features: {X.shape}")
    print(f"  Target: {y.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=FEATURE_COLUMNS,
        index=X.index
    )
    
    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    r2_score = model.score(X_scaled, y)
    print(f"  Model R² score: {r2_score:.4f}")
    
    # Compute SHAP values using TreeExplainer
    # Paper: "quantified using SHAP values with TreeExplainer (Lundberg & Lee, 2017)"
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    return model, shap_values, X_scaled, y, explainer, X


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def run_shap_analysis(df_clean, output_dir, dataset_name):
    """
    Run complete SHAP analysis pipeline for a dataset.
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Prepared dataset
    output_dir : str
        Output directory for figures
    dataset_name : str
        Dataset name for display
        
    Returns
    -------
    dict
        Analysis results including model, SHAP values, and statistics
    """
    print(f"\n{'=' * 70}")
    print(f"SHAP ANALYSIS: {dataset_name}")
    print(f"{'=' * 70}")
    
    # Train model and compute SHAP
    model, shap_values, X_scaled, y, explainer, X_unscaled = train_model_and_compute_shap(
        df_clean, dataset_name
    )
    
    # Generate visualizations
    print(f"\n[4] Generating visualizations...")
    
    # Main importance plot with CI (Figure 6)
    stats_dict = create_importance_barplot_with_ci(
        shap_values, FEATURE_COLUMNS, output_dir,
        target_name=TARGET_DISPLAY_NAME
    )
    
    # Percentage bar plot
    pct_importance = create_percentage_barplot(
        shap_values, FEATURE_COLUMNS, output_dir,
        target_name=TARGET_DISPLAY_NAME
    )
    
    # Summary beeswarm plot
    create_shap_summary_plot(X_scaled, shap_values, FEATURE_COLUMNS, output_dir)
    
    # Pie chart
    create_pie_chart(shap_values, FEATURE_COLUMNS, output_dir,
                     target_name=TARGET_DISPLAY_NAME)
    
    # Waterfall example
    create_waterfall_plot(explainer, X_scaled, shap_values, FEATURE_COLUMNS,
                          output_dir, target_name=TARGET_DISPLAY_NAME)
    
    # Print results table
    print(f"\n  Feature Importance Summary ({dataset_name}):")
    print(f"  {'-' * 65}")
    print(f"  {'Feature':<20} {'Mean |SHAP|':<15} {'95% CI':<15} {'%':<10}")
    print(f"  {'-' * 65}")
    
    sorted_features = sorted(stats_dict.keys(),
                             key=lambda x: stats_dict[x]['mean'], reverse=True)
    for feature in sorted_features:
        stats = stats_dict[feature]
        pct = pct_importance.get(feature, 0)
        print(f"  {feature:<20} {stats['mean']:<15.4f} ±{stats['ci_95']:<13.4f} {pct:>6.1f}%")
    
    return {
        'model': model,
        'shap_values': shap_values,
        'stats_dict': stats_dict,
        'pct_importance': pct_importance,
        'X': X_scaled,
        'y': y,
        'explainer': explainer
    }


def main():
    """
    Main analysis pipeline for SHAP feature importance analysis.
    
    Paper Reference:
    - Figure 6: SHAP feature importance analysis (Phase II-A, n=86)
    - Supplemental Figure 2: SHAP analysis (Phase II-B, n=211)
    - Section 3.2.4: Feature importance analysis
    - Section 3.3.6: Scanner emerges as dominant determinant
    """
    print("=" * 70)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("Paper: Robustness and Error Susceptibility of BIANCA")
    print("=" * 70)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # =========================================================================
    # Phase II-A Analysis (Masked subjects, n=86)
    # Paper: "Phase II-A - Detailed robustness assessment (n=86)"
    # =========================================================================
    df_masked = prepare_feature_matrix(df, dataset_type='MASKED')
    results_masked = run_shap_analysis(
        df_masked, OUTPUT_DIR_MASKED, 'Phase II-A (n=86)'
    )
    
    # =========================================================================
    # Phase II-B Analysis (Full cohort, n=211)
    # Paper: "Phase II-B - large-scale real-world validation (n=211)"
    # =========================================================================
    df_full = prepare_feature_matrix(df, dataset_type='FULL')
    results_full = run_shap_analysis(
        df_full, OUTPUT_DIR_FULL, 'Phase II-B (n=211)'
    )
    
    # =========================================================================
    # Summary: Compare scanner importance between phases
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY FINDING: Scanner Importance Comparison")
    print("=" * 70)
    
    scanner_masked = results_masked['stats_dict'].get('scanner', {}).get('mean', 0)
    scanner_full = results_full['stats_dict'].get('scanner', {}).get('mean', 0)
    
    if scanner_masked > 0:
        fold_change = scanner_full / scanner_masked
        print(f"\n  Phase II-A scanner SHAP: {scanner_masked:.4f}")
        print(f"  Phase II-B scanner SHAP: {scanner_full:.4f}")
        print(f"  Fold change: {fold_change:.1f}x")
        print(f"\n  Paper conclusion (Section 3.3.6):")
        print("  'Scanner importance increased 31-fold (0.002 → 0.062) from")
        print("  Phase II-A to Phase II-B as Philips representation increased")
        print("  from 9.3% to 17.1%.'")
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"\n  Phase II-A (Masked): {OUTPUT_DIR_MASKED}")
    print(f"  Phase II-B (Full):   {OUTPUT_DIR_FULL}")
    print(f"\n  Generated files per dataset:")
    print("    - shap_feature_importance_with_ci.png (Bar plot with 95% CI)")
    print("    - shap_feature_importance_percentage.png (Percentage bar plot)")
    print("    - shap_summary_beeswarm.png (SHAP summary plot)")
    print("    - shap_feature_influence_pie.png (Pie chart)")
    print("    - shap_waterfall_example.png (Individual prediction)")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return {
        'masked': results_masked,
        'full': results_full
    }


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    results = main()