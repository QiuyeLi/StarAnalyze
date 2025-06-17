# ============================================================
# Script: Visualize Particles by Angle Psi (computed PsiPrior)
# Version: v5.6
#
# Usage:
#   python visualize_polarity.py --inRefine path/to/run_data.star --o v5.6_test
#
# Description:
#   - Estimates PsiPrior for each filament using two endpoints of the helical tube
#   - PsiPrior is flipped to match the AnglePsi rotation convention (additive inverse)
#   - Computes per-particle Psi = |AnglePsi - PsiPrior|
#   - Confidence = ||Psi - 90|| (90° being most uncertain about flipping)
#   - Plots particles colored by confidence per micrograph
#   - Saves PNGs, Psi histogram, and Psi vs PsiPrior plot to output folder
# ============================================================

import mrcfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import starfile
from skimage import exposure
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tqdm import tqdm
import multiprocessing as mp
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Visualize particles colored by ||Psi - 90||.')
parser.add_argument('--inRefine', required=True, help='Input STAR file from Refine3D (e.g. run_data.star)')
parser.add_argument('--o', required=True, help='Output directory')
args = parser.parse_args()

# Version and output directory
version = 'v5.6'
output_dir = args.o
os.makedirs(output_dir, exist_ok=True)

# Load STAR file
refine_star = starfile.read(args.inRefine)['particles']

# Convert numeric fields
numeric_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnHelicalTubeID',
                'rlnHelicalTrackLengthAngst', 'rlnAnglePsi']
refine_star[numeric_cols] = refine_star[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Corrected PsiPrior assignment: compute per (micrograph, tube) and broadcast
psi_debug_info = []  # for dx, dy, psi_deg plots

def assign_psi_prior(group):
    group_sorted = group.sort_values('rlnHelicalTrackLengthAngst')
    start = group_sorted.iloc[0]
    end = group_sorted.iloc[-1]
    dx = end['rlnCoordinateX'] - start['rlnCoordinateX']
    dy = end['rlnCoordinateY'] - start['rlnCoordinateY']
    psi_rad = np.arctan2(dy, dx)
    psi_deg = -np.degrees(psi_rad)  # Invert angle to match AnglePsi rotation
    group['PsiPrior'] = psi_deg
    psi_debug_info.append((group.name, dx, dy, psi_deg))
    return group

merged_df = refine_star.groupby(['rlnMicrographName', 'rlnHelicalTubeID'], group_keys=False).apply(assign_psi_prior)

# Compute Psi = |AnglePsi - PsiPrior| and Confidence = ||Psi - 90||
merged_df['Psi'] = np.abs(merged_df['rlnAnglePsi'] - merged_df['PsiPrior']) % 360
merged_df['Psi'] = merged_df['Psi'].apply(lambda x: x if x <= 180 else 360 - x)
merged_df['Confidence'] = np.abs(merged_df['Psi'] - 90)

# Unique micrographs
micrographs = merged_df['rlnMicrographName'].unique()

# Function to process a single micrograph
def process_micrograph(micrograph_name):
    df_micro = merged_df[merged_df['rlnMicrographName'] == micrograph_name].copy()
    try:
        with mrcfile.open(micrograph_name, permissive=True) as mrc:
            img = mrc.data.astype(np.float32)
    except Exception as e:
        return f"❌ Skipping {micrograph_name}: {e}"

    # Normalize and enhance image
    img -= np.nanmin(img)
    img /= np.nanmax(img) if np.nanmax(img) != 0 else 1.0
    img = np.nan_to_num(img)
    img_eq = exposure.equalize_adapthist(img, clip_limit=0.02)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_eq, cmap='gray')
    ax.set_title(f'Particles in {os.path.basename(micrograph_name)}')
    ax.axis('off')

    # Group by helical tube
    grouped = df_micro.groupby('rlnHelicalTubeID')
    for _, group in grouped:
        group = group.sort_values('rlnHelicalTrackLengthAngst')
        coords_x = group['rlnCoordinateX'].values
        coords_y = group['rlnCoordinateY'].values
        confidence = group['Confidence'].values
        norm_conf = confidence / 90  # Normalize to 0–1
        colors = [plt.cm.plasma(val) for val in norm_conf]

        for x, y, color in zip(coords_x, coords_y, colors):
            ax.plot(x, y, 'o', color=color, markersize=10)

        if len(coords_x) >= 2:
            x0, y0 = coords_x[0], coords_y[0]
            x1, y1 = coords_x[-1], coords_y[-1]
            mean_psi = group['Psi'].mean()
            if mean_psi > 90:
                x0, y0, x1, y1 = x1, y1, x0, y0

            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    facecolor='#ff3333', edgecolor='#ff3333',
                    arrowstyle='->,head_length=2,head_width=1', lw=2
                )
            )

    # Add colorbar
    norm = Normalize(vmin=0, vmax=90)
    sm = ScalarMappable(norm=norm, cmap='plasma')
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8,
                        label='||Psi - 90|| (confidence score)')

    output_filename = os.path.splitext(os.path.basename(micrograph_name))[0] + f'_{version}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return f"✅ Saved: {output_path}"

# Run in parallel with progress bar
if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_micrograph, micrographs), total=len(micrographs)):
            print(result)

    # Save Psi histogram
    plt.figure(figsize=(8, 5))
    plt.hist(merged_df['Psi'], bins=90, color='steelblue', edgecolor='black')
    plt.title('Histogram of calculated Psi angles')
    plt.xlabel('Psi (|AnglePsi - PsiPrior|)')
    plt.ylabel('Count')
    plt.grid(True)
    hist_path = os.path.join(output_dir, f'psi_histogram_{version}.png')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"✅ Histogram saved: {hist_path}")

    # Save Psi vs PsiPrior plot
    plt.figure(figsize=(8, 6))
    plt.scatter(merged_df['PsiPrior'], merged_df['rlnAnglePsi'], alpha=0.4, s=10, color='darkorange', edgecolors='k')
    plt.xlabel('PsiPrior (deg)')
    plt.ylabel('rlnAnglePsi (deg)')
    plt.title('rlnAnglePsi vs PsiPrior')
    plt.grid(True)
    psi_vs_prior_path = os.path.join(output_dir, f'psi_vs_psiprior_{version}.png')
    plt.tight_layout()
    plt.savefig(psi_vs_prior_path, dpi=300)
    plt.close()
    print(f"✅ Psi vs PsiPrior plot saved: {psi_vs_prior_path}")

    # Save dx, dy, psi_deg per tube
    psi_debug_df = pd.DataFrame(psi_debug_info, columns=['TubeID', 'dx', 'dy', 'PsiPrior'])
    psi_debug_df['TubeID'] = psi_debug_df['TubeID'].apply(lambda x: f"{x[0]}_{x[1]}" if isinstance(x, tuple) else str(x))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(psi_debug_df['TubeID'], psi_debug_df['dx'], 'o-', label='dx')
    axes[1].plot(psi_debug_df['TubeID'], psi_debug_df['dy'], 'o-', label='dy', color='green')
    axes[2].plot(psi_debug_df['TubeID'], psi_debug_df['PsiPrior'], 'o-', label='PsiPrior', color='purple')
    axes[0].set_ylabel('dx')
    axes[1].set_ylabel('dy')
    axes[2].set_ylabel('PsiPrior (deg)')
    axes[2].set_xlabel('HelicalTubeID')
    for ax in axes:
        ax.grid(True)
        ax.legend()
    plt.suptitle('dx, dy, PsiPrior per HelicalTubeID')
    plt.tight_layout()
    dxdy_path = os.path.join(output_dir, f'dx_dy_psi_per_tube_{version}.png')
    plt.savefig(dxdy_path, dpi=300)
    plt.close()
    print(f"✅ dx/dy/PsiPrior plot saved: {dxdy_path}")
