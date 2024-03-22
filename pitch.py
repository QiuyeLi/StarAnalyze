import matplotlib
matplotlib.use('Agg')  # Using a headless backend
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def process_star_file(input_file, output_name):
    # Read the CSV file with specified column numbers
    df = pd.read_csv(input_file, skiprows=56, delim_whitespace=True, usecols=[0, 1, 2, 8])

    # Group by '_rlnMicrographName' and '_rlnHelicalTubeID'
    grouped = df.groupby([df.columns[3], df.columns[2]])

    # Plotting
    num_groups = len(grouped)
    max_subplots = 10  # Maximum number of subplots
    num_plots = min(num_groups, max_subplots)  # Number of subplots to create

    # Calculate figure size based on desired pixel dimensions and DPI
    dpi = 100  # Adjust DPI as needed
    figsize = (5760 / dpi, 4092 / dpi)  # Convert pixels to inches

    fig, axs = plt.subplots(num_plots, figsize=figsize, sharex=True, dpi=dpi)

    if num_plots > 1:
        axs = axs.ravel()  # Flatten the array of subplots

    for idx, ((name, tube), group) in enumerate(grouped):
        if idx >= max_subplots:
            break  # Exit loop if reached max_subplots

        ax = axs[idx] if num_plots > 1 else axs
        ax.scatter(group.iloc[:, 0], group.iloc[:, 1], label=f'{name}_{tube}')
        ax.set_title(f'{name}_{tube}')
        ax.legend()

        # Set equal aspect ratio for the plot
        ax.set_aspect('equal', adjustable='box')

        # Set X and Y axis limits
        ax.set_xlim(0, 5760)
        ax.set_ylim(0, 4092)

    plt.tight_layout()
    plt.savefig(f'{output_name}.png', dpi=dpi)  # Save with specified DPI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help='Input star file')
    parser.add_argument('--o', help='Output name')
    args = parser.parse_args()

    process_star_file(args.i, args.o)
