# ============================================================
# Script: Visualize Particles by Angle Psi (||ψ| - 90| scoring)
# Version: v5.1
#
# Usage:
#   python visualize_polarity.py --i path/to/run_data.star --o v5.1_test
#
# Description:
#   - Processes all unique micrographs listed in rlnMicrographName
#   - Plots particles colored by ||ψ| - 90|, indicating alignment confidence
#   - Saves a PNG for each micrograph in the specified output directory
#   - Parallelized across all available CPU cores with a progress bar
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
parser = argparse.ArgumentParser(description='Visualize particles colored by ||ψ| - 90|.')
parser.add_argument('--i', required=True, help='Input STAR file (e.g. run_data.star)')
parser.add_argument('--o', required=True, help='Output directory')
args = parser.parse_args()

# Version and output directory
version = 'v5.1'
output_dir = args.o
os.makedirs(output_dir, exist_ok=True)

# STAR file path
star_path = args.i
star_data = starfile.read(star_path)

# Ensure required block and columns
df = star_data['particles']
required_cols = [
    'rlnCoordinateX', 'rlnCoordinateY', 'rlnHelicalTubeID',
    'rlnHelicalTrackLengthAngst', 'rlnMicrographName', 'rlnAnglePsi'
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in particles data")

# Convert to numeric
numeric_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnHelicalTubeID', 'rlnHelicalTrackLengthAngst', 'rlnAnglePsi']
df.loc[:, numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

micrographs = df['rlnMicrographName'].unique()

# Function to process a single micrograph
def process_micrograph(micrograph_name):
    df_micro = df[df['rlnMicrographName'] == micrograph_name].copy()
    try:
        with mrcfile.open(micrograph_name, permissive=True) as mrc:
            img = mrc.data.astype(np.float32)
    except Exception as e:
        return f"❌ Skipping {micrograph_name}: {e}"

    img -= np.nanmin(img)
    img /= np.nanmax(img) if np.nanmax(img) != 0 else 1.0
    img = np.nan_to_num(img)
    img_eq = exposure.equalize_adapthist(img, clip_limit=0.02)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_eq, cmap='gray')
    ax.set_title(f'Particles in {os.path.basename(micrograph_name)}')
    ax.axis('off')

    grouped = df_micro.groupby('rlnHelicalTubeID')
    for _, group in grouped:
        group = group.sort_values('rlnHelicalTrackLengthAngst')
        coords_x = group['rlnCoordinateX'].values
        coords_y = group['rlnCoordinateY'].values
        angle_psi = group['rlnAnglePsi'].values

        abs_psi = np.abs(angle_psi)
        confidence = np.abs(abs_psi - 90)
        norm_conf = confidence / 90
        colors = [plt.cm.plasma(val) for val in norm_conf]

        for x, y, color in zip(coords_x, coords_y, colors):
            ax.plot(x, y, 'o', color=color, markersize=10)

        if len(coords_x) >= 2:
            x0, y0 = coords_x[0], coords_y[0]
            x1, y1 = coords_x[-1], coords_y[-1]
            psi_end = np.abs(angle_psi[-1])
            if psi_end > 90:
                x0, y0, x1, y1 = x1, y1, x0, y0

            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    facecolor='#ff3333', edgecolor='#ff3333',
                    arrowstyle='->,head_length=2,head_width=1', lw=2
                )
            )

    norm = Normalize(vmin=0, vmax=90)
    sm = ScalarMappable(norm=norm, cmap='plasma')
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8,
                        label='||ψ| - 90| (confidence score)')

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
