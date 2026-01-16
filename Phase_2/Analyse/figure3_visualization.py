"""
================================================================================
Figure 3: Representative WMH Segmentation Examples
================================================================================

Paper Reference: "Robustness and Error Susceptibility of BIANCA for White 
Matter Hyperintensity Segmentation"

This script generates Figure 3 from the paper:

"Representative segmentation examples (Figure 3) illustrate the variable 
impact of lesion removal across different cases. While some cases showed 
minimal differences between R and NR approaches regardless of preprocessing 
strategy, others demonstrated substantial false positive reduction with 
lesion removal, particularly in cases with complex lesion presentations 
adjacent to WMH regions."

Figure Layout:
- 4 columns: Infarct | NR | R | Difference
- 2 rows: Representative cases (minimal vs substantial R/NR differences)

Color Coding:
- Yellow: Infarct/lesion location
- Dark green: Periventricular WMH
- Light green: Deep WMH  
- Blue: R > NR (additional WMH detected with removal)
- Red: NR > R (WMH only detected without removal)

Output:
- figure3_representative_segmentation.png (high-resolution for publication)
- figure3_representative_segmentation.pdf (vector format)

Author: Uchralt Temuulen
================================================================================
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths (configure for your environment)
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '.')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Figure parameters
FIGURE_DPI = 300
FIGURE_WIDTH = 10
FIGURE_HEIGHT_PER_ROW = 3.2

# Color definitions (matching paper Figure 3)
COLORS = {
    'infarct': [1.0, 1.0, 0.0],           # Yellow
    'peri_wmh': [0.0, 0.55, 0.0],         # Dark green
    'deep_wmh': [0.0, 1.0, 0.0],          # Light green (bright)
    'r_more': [0.0, 0.6, 1.0],            # Blue (R detected more)
    'nr_more': [1.0, 0.0, 0.0],           # Red (NR detected more)
}

# Overlay parameters
ALPHA_INFARCT = 0.85
ALPHA_WMH = 0.85
ALPHA_DIFF = 0.9
DIM_FACTOR = 0.6  # Background dimming for contrast


# ==============================================================================
# IMAGE LOADING UTILITIES
# ==============================================================================

def load_slice(path: str, slice_idx: int) -> Optional[np.ndarray]:
    """
    Load a single axial slice from a NIfTI image.
    
    Parameters
    ----------
    path : str
        Path to NIfTI file
    slice_idx : int
        Axial slice index (z-coordinate)
        
    Returns
    -------
    np.ndarray or None
        2D array of the slice, rotated for display
    """
    if path is None or not os.path.isfile(path):
        return None
    
    try:
        img = nib.load(path)
        data = img.get_fdata()
        # Rotate 90 degrees for standard neuroimaging orientation
        return np.rot90(data[:, :, slice_idx])
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def get_top_slices(mask_path: str, top_n: int = 10) -> List[int]:
    """
    Find slices with the highest number of non-zero voxels.
    
    Useful for automatically selecting representative slices that show
    the most lesion/WMH content.
    
    Parameters
    ----------
    mask_path : str
        Path to binary mask NIfTI file
    top_n : int
        Number of top slices to return
        
    Returns
    -------
    list
        Slice indices sorted by descending non-zero voxel counts
    """
    try:
        img = nib.load(mask_path)
        data = img.get_fdata()
        
        slice_counts = []
        for z in range(data.shape[2]):
            count = np.count_nonzero(data[:, :, z])
            slice_counts.append((z, count))
        
        # Sort by count (descending)
        sorted_slices = sorted(slice_counts, key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_slices[:top_n]]
    except Exception as e:
        print(f"Error finding top slices: {e}")
        return []


def calculate_bbox(
    image: np.ndarray, 
    padding: int = 5, 
    threshold_percent: float = 0.02
) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box for cropping to brain region.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array
    padding : int
        Padding around detected brain region
    threshold_percent : float
        Threshold as percentage of max intensity
        
    Returns
    -------
    tuple
        (ymin, ymax, xmin, xmax) coordinates
    """
    threshold = np.max(image) * threshold_percent
    mask = image > threshold
    
    if not np.any(mask):
        return 0, image.shape[0], 0, image.shape[1]
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Add padding
    ymin = max(0, ymin - padding)
    ymax = min(image.shape[0], ymax + padding)
    xmin = max(0, xmin - padding)
    xmax = min(image.shape[1], xmax + padding)
    
    return ymin, ymax, xmin, xmax


# ==============================================================================
# OVERLAY FUNCTIONS
# ==============================================================================

def apply_overlay(
    rgb_image: np.ndarray, 
    mask: np.ndarray, 
    color: List[float], 
    alpha: float
) -> np.ndarray:
    """
    Apply colored overlay on RGB image where mask is positive.
    
    Parameters
    ----------
    rgb_image : np.ndarray
        RGB image array (H x W x 3)
    mask : np.ndarray
        Binary mask (H x W), True where overlay should be applied
    color : list
        RGB color values [R, G, B], each 0-1
    alpha : float
        Overlay opacity (0-1)
        
    Returns
    -------
    np.ndarray
        Modified RGB image
    """
    if mask is None:
        return rgb_image
    
    mask_bool = mask > 0
    for c in range(3):
        rgb_image[mask_bool, c] = (
            alpha * color[c] + (1 - alpha) * rgb_image[mask_bool, c]
        )
    
    return rgb_image


# ==============================================================================
# FIGURE GENERATION
# ==============================================================================

def create_figure3(
    row_configs: List[Dict],
    output_path: str,
    dpi: int = FIGURE_DPI
):
    """
    Create Figure 3: Representative WMH segmentation examples.
    
    Paper Reference:
    "Representative segmentation examples (Figure 3) illustrate the variable 
    impact of lesion removal across different cases."
    
    Parameters
    ----------
    row_configs : list
        List of dictionaries, each containing:
        - 'images': List of (flair_path, [overlay_paths...]) tuples
        - 'slice_idx': Axial slice index to display
        - 'left_label': Label for left side (optional)
        - 'right_label': Label for right side (optional)
    output_path : str
        Path for output image
    dpi : int
        Output resolution
        
    Figure Layout:
    Each row shows one subject with 4 columns:
    - Column 1: FLAIR with infarct highlighted (yellow)
    - Column 2: NR segmentation (periventricular=dark green, deep=light green)
    - Column 3: R segmentation (same color coding)
    - Column 4: Difference map (blue=R>NR, red=NR>R)
    """
    n_rows = len(row_configs)
    n_cols = 4
    
    # Figure dimensions
    fig_width = FIGURE_WIDTH
    fig_height = FIGURE_HEIGHT_PER_ROW * n_rows + 1.0
    
    # Create figure with black background
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='black')
    
    # GridSpec layout
    gs = fig.add_gridspec(
        n_rows + 1,  # +1 for header row
        n_cols + 2,   # +2 for left/right labels
        width_ratios=[0.001] + [1] * n_cols + [0.001],
        height_ratios=[0.2] + [1] * n_rows,
        hspace=0.0,
        wspace=0.0
    )
    
    # Column titles (white text on black background)
    col_titles = ["Infarct", "NR", "R", "Difference"]
    for j, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.text(
            0.5, 0.3, title,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='white'
        )
        ax.axis('off')
    
    # Process each row (each subject)
    for i, config in enumerate(row_configs):
        row_idx = i + 1  # Skip header row
        slice_idx = config['slice_idx']
        images = config['images']
        
        # Left label
        ax_left = fig.add_subplot(gs[row_idx, 0])
        left_text = config.get('left_label', '')
        ax_left.text(
            0.5, 0.5, left_text,
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            rotation=90, color='white'
        )
        ax_left.axis('off')
        
        # Load FLAIR (base image for all columns)
        flair_path = images[0][0]
        flair = load_slice(flair_path, slice_idx)
        if flair is None:
            print(f"Warning: Could not load FLAIR for row {i}")
            continue
        
        # Calculate cropping box (tight around brain)
        ymin, ymax, xmin, xmax = calculate_bbox(flair, padding=1, threshold_percent=0.02)
        
        # Normalize and dim FLAIR for better overlay visibility
        if np.max(flair) > 0:
            flair_norm = (flair / np.max(flair)) * DIM_FACTOR
        else:
            flair_norm = flair
        
        # === COLUMN 1: Infarct (Yellow) ===
        ax1 = fig.add_subplot(gs[row_idx, 1])
        rgb1 = np.stack([flair_norm] * 3, axis=-1).copy()
        
        infarct_paths = images[0][1]  # [infarct_path]
        if infarct_paths and len(infarct_paths) > 0:
            infarct = load_slice(infarct_paths[0], slice_idx)
            if infarct is not None:
                rgb1 = apply_overlay(rgb1, infarct > 0, COLORS['infarct'], ALPHA_INFARCT)
        
        ax1.imshow(np.clip(rgb1, 0, 1))
        ax1.axis('off')
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymax, ymin)  # Inverted for proper orientation
        
        # === COLUMN 2: NR (Non-Removed) ===
        ax2 = fig.add_subplot(gs[row_idx, 2])
        rgb2 = np.stack([flair_norm] * 3, axis=-1).copy()
        
        nr_paths = images[1][1]  # [peri_nr, deep_nr]
        if nr_paths and len(nr_paths) >= 2:
            # Deep WMH first (light green)
            deep_nr = load_slice(nr_paths[1], slice_idx)
            if deep_nr is not None:
                rgb2 = apply_overlay(rgb2, deep_nr > 0, COLORS['deep_wmh'], ALPHA_WMH)
            
            # Periventricular WMH (dark green) - on top
            peri_nr = load_slice(nr_paths[0], slice_idx)
            if peri_nr is not None:
                rgb2 = apply_overlay(rgb2, peri_nr > 0, COLORS['peri_wmh'], ALPHA_WMH)
        
        ax2.imshow(np.clip(rgb2, 0, 1))
        ax2.axis('off')
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymax, ymin)
        
        # === COLUMN 3: R (Removed) ===
        ax3 = fig.add_subplot(gs[row_idx, 3])
        rgb3 = np.stack([flair_norm] * 3, axis=-1).copy()
        
        r_paths = images[2][1]  # [peri_r, deep_r]
        if r_paths and len(r_paths) >= 2:
            # Deep WMH first
            deep_r = load_slice(r_paths[1], slice_idx)
            if deep_r is not None:
                rgb3 = apply_overlay(rgb3, deep_r > 0, COLORS['deep_wmh'], ALPHA_WMH)
            
            # Periventricular WMH on top
            peri_r = load_slice(r_paths[0], slice_idx)
            if peri_r is not None:
                rgb3 = apply_overlay(rgb3, peri_r > 0, COLORS['peri_wmh'], ALPHA_WMH)
        
        ax3.imshow(np.clip(rgb3, 0, 1))
        ax3.axis('off')
        ax3.set_xlim(xmin, xmax)
        ax3.set_ylim(ymax, ymin)
        
        # === COLUMN 4: Difference (R - NR) ===
        ax4 = fig.add_subplot(gs[row_idx, 4])
        rgb4 = np.stack([flair_norm] * 3, axis=-1).copy()
        
        diff_paths = images[3][1]  # [diff_path]
        if diff_paths and len(diff_paths) > 0:
            diff = load_slice(diff_paths[0], slice_idx)
            if diff is not None:
                # diff = NR - R (from fslmaths)
                # Negative: R > NR (R detected more) -> Blue
                # Positive: NR > R (NR detected more) -> Red
                threshold = 0.05
                
                r_more = diff < -threshold   # R detected more (blue)
                nr_more = diff > threshold   # NR detected more (red)
                
                rgb4 = apply_overlay(rgb4, r_more, COLORS['r_more'], ALPHA_DIFF)
                rgb4 = apply_overlay(rgb4, nr_more, COLORS['nr_more'], ALPHA_DIFF)
        
        ax4.imshow(np.clip(rgb4, 0, 1))
        ax4.axis('off')
        ax4.set_xlim(xmin, xmax)
        ax4.set_ylim(ymax, ymin)
        
        # Right label
        ax_right = fig.add_subplot(gs[row_idx, 5])
        right_text = config.get('right_label', '')
        ax_right.text(
            0.1, 0.5, right_text,
            ha='left', va='center',
            fontsize=10, linespacing=1.3,
            color='white'
        )
        ax_right.axis('off')
    
    # Legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor='yellow', edgecolor='black', label='Infarct (yellow)'),
        mpatches.Patch(facecolor='#006400', edgecolor='black', label='Periventricular WMH (dark green)'),
        mpatches.Patch(facecolor='#00ff00', edgecolor='black', label='Deep WMH (green)'),
        mpatches.Patch(facecolor='#0099ff', edgecolor='black', label='R > NR (blue)'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='NR > R (red)'),
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=5,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        fancybox=True,
        shadow=True
    )
    
    # Layout adjustment
    plt.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    
    # Save figure
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.01, dpi=dpi, facecolor='black')
    
    # Also save PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.01, facecolor='black')
    
    plt.close()
    print(f"✓ Figure 3 saved: {output_path}")
    print(f"✓ PDF version: {pdf_path}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Generate Figure 3 with example data.
    
    In practice, replace the example paths with actual subject data.
    """
    print("=" * 80)
    print("FIGURE 3: REPRESENTATIVE WMH SEGMENTATION EXAMPLES")
    print("Paper: Robustness and Error Susceptibility of BIANCA")
    print("=" * 80)
    
    # Example configuration (replace with actual subject data)
    # This structure should be populated from your data pipeline
    example_row_configs = [
        {
            # Row 1: Subject with minimal R/NR differences (top row in Figure 3)
            'images': [
                # (FLAIR_path, [infarct_path])
                ('path/to/sub-000_FLAIR.nii.gz', ['path/to/sub-000_infarct.nii.gz']),
                # (FLAIR_path, [peri_NR_path, deep_NR_path])
                ('path/to/sub-000_FLAIR.nii.gz', ['path/to/sub-000_peri_NR.nii.gz', 'path/to/sub-000_deep_NR.nii.gz']),
                # (FLAIR_path, [peri_R_path, deep_R_path])
                ('path/to/sub-000_FLAIR.nii.gz', ['path/to/sub-000_peri_R.nii.gz', 'path/to/sub-000_deep_R.nii.gz']),
                # (FLAIR_path, [diff_path])
                ('path/to/sub-000_FLAIR.nii.gz', ['path/to/sub-000_diff.nii.gz']),
            ],
            'slice_idx': 19,  # Optimal slice showing lesion
            'left_label': '',  # Optional row label
            'right_label': ''  # Optional description
        },
        {
            # Row 2: Subject with substantial R/NR differences (bottom row in Figure 3)
            'images': [
                ('path/to/sub-163_FLAIR.nii.gz', ['path/to/sub-163_infarct.nii.gz']),
                ('path/to/sub-163_FLAIR.nii.gz', ['path/to/sub-163_peri_NR.nii.gz', 'path/to/sub-163_deep_NR.nii.gz']),
                ('path/to/sub-163_FLAIR.nii.gz', ['path/to/sub-163_peri_R.nii.gz', 'path/to/sub-163_deep_R.nii.gz']),
                ('path/to/sub-163_FLAIR.nii.gz', ['path/to/sub-163_diff.nii.gz']),
            ],
            'slice_idx': 20,
            'left_label': '',
            'right_label': ''
        }
    ]
    
    output_path = os.path.join(RESULTS_DIR, 'figure3_representative_segmentation.png')
    
    print(f"\nOutput: {output_path}")
    print("\nNote: Replace example paths with actual subject data before running.")
    print("The row_configs structure should match your data organization.")
    
    # Uncomment below when actual data is available:
    # create_figure3(row_configs=example_row_configs, output_path=output_path)
    
    print("\nTo generate the figure, call:")
    print("  create_figure3(row_configs=your_data, output_path=output_path)")


if __name__ == "__main__":
    main()