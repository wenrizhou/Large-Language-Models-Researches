"""
Plot Pass@k Curves
Reads Pass@k metrics for different temperatures from a CSV file and plots the curves.
"""

import csv
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt
from pathlib import Path

# Use a cleaner, pure white style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# Set font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def plot_pass_at_k(csv_file='output/paask&maj.csv', output_file='output/pass_at_k_curve.png', dataset_name='Math500', model_name='Qwen2.5-Math-1.5B'):
    """
    Plots the Pass@k curves.
    
    Args:
        csv_file: Path to the CSV file.
        output_file: Path to the output image file.
        dataset_name: Name of the dataset for the title.
        model_name: Name of the model for the title.
    """
    # Read CSV file
    data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'temperature': float(row['temperature']),
                    'pass@1': float(row['pass@1']),
                    'pass@2': float(row['pass@2']),
                    'pass@4': float(row['pass@4']),
                    'pass@8': float(row['pass@8']),
                    'pass@16': float(row['pass@16']),
                })
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
        return

    # k values
    k_values = [1, 2, 4, 8, 16]
    k_columns = [f'pass@{k}' for k in k_values]
    
    # Create the figure with higher quality
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    
    # Modern color palette (Tableau 10 or similar)
    colors = ['#4E79A7', '#F28E2B', '#E15759']  # Blue, Orange, Red
    markers = ['o', 's', 'D']  # Circle, Square, Diamond
    
    for idx, row in enumerate(data):
        temp = row['temperature']
        pass_values = [row[col] for col in k_columns]
        
        # Plot lines with markers
        ax.plot(
            k_values, 
            pass_values, 
            marker=markers[idx % len(markers)],
            linewidth=2.5,
            markersize=10,
            markeredgecolor='white',
            markeredgewidth=1.5,
            label=f'Temperature = {temp}',
            color=colors[idx % len(colors)],
            alpha=0.9,
            zorder=3
        )
        
        # Add value labels with logic to prevent overlapping
        # Staggered labels to prevent overlap
        if idx == 0:     # Blue (Temp 0.6)
            y_offset, va = 15, 'bottom'
        elif idx == 1:   # Yellow/Orange (Temp 1.0)
            y_offset, va = -12, 'top'
        else:            # Red (Temp 1.2)
            y_offset, va = -22, 'top'
        
        for k, val in zip(k_values, pass_values):
            ax.annotate(
                f'{val:.1f}%',
                xy=(k, val),
                xytext=(0, y_offset),
                textcoords='offset points',
                fontsize=9,
                ha='center',
                va=va,
                fontweight='bold',
                color=colors[idx % len(colors)],
                alpha=1.0
            )

    # Set chart properties
    ax.set_xlabel('Number of Samples (k)', fontsize=13, labelpad=12, fontweight='500')
    ax.set_ylabel('Pass@k (%)', fontsize=13, labelpad=12, fontweight='500')
    ax.set_title(f'Pass@k Performance at {dataset_name} using {model_name}', fontsize=16, pad=20, fontweight='bold')
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    # Legend styling
    legend = ax.legend(
        title='Hyperparameters', 
        loc='lower right', 
        fontsize=11, 
        frameon=True, 
        facecolor='white', 
        edgecolor='#cccccc'
    )
    legend.get_title().set_fontweight('bold')
    
    # Set x-axis to log scale (base 2)
    ax.set_xscale('log', base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    
    # Set y-axis range and formatting
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    
    # Remove top and right spines for a modern look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the plot
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Chart successfully saved to: {output_file}")
    
    # Show the chart if possible
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    # --- Configuration ---
    DATASET = "AIME25"
    MODEL = "Qwen2.5-Math-1.5B"
    CSV_PATH = 'outputs/passk&maj_qwen_instruct.csv'
    IMG_PATH = 'output/instruct_AIME25.png'

    # Run plotting
    plot_pass_at_k(
        csv_file=CSV_PATH, 
        output_file=IMG_PATH,
        dataset_name=DATASET,
        model_name=MODEL
    )

