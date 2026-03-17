import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
from scipy.stats import gaussian_kde

def get_pop_sums(dataset_name, subset=None, split=None):
    ds = load_dataset(dataset_name, subset, split=split)
    return np.array([float(x['pop_sum']) for x in ds if x['pop_sum'] is not None])

# Load datasets
duet_rare = get_pop_sums("SwetieePawsss/DUET", split="city_forget_rare_10")
duet_popular = get_pop_sums("SwetieePawsss/DUET", split="city_forget_popular_10")
duet_merged = np.concatenate([duet_rare, duet_popular])
rwku = get_pop_sums("SwetieePawsss/exp_r", "forget_level2", split="test")

# Plotting setup
plt.figure(figsize=(12, 8))
all_vals = np.concatenate([duet_merged, rwku])
min_val = max(1, np.min(all_vals))
max_val = np.max(all_vals)
x_grid = np.logspace(np.log10(min_val), np.log10(max_val), 500)

def plot_kde_with_range(data, label, color, y_offset):
    # Log-transform for KDE
    log_data = np.log10(data[data > 0])
    kde = gaussian_kde(log_data)
    log_x_grid = np.log10(x_grid)
    kde_values = kde(log_x_grid)
    kde_values = kde_values / np.max(kde_values) # Normalized
    
    # Plot curve (only this goes to legend)
    plt.plot(x_grid, kde_values, color=color, lw=2.5, label=label)
    plt.fill_between(x_grid, kde_values, alpha=0.2, color=color)
    
    # Range visualization
    p5, p25, p50, p75, p95 = np.percentile(data, [5, 25, 50, 75, 95])
    y_bar = -0.15 - y_offset
    d_min, d_max = np.min(data), np.max(data)
    
    # Range bars
    plt.hlines(y_bar, p5, p95, colors=color, linestyles='-', lw=2, alpha=0.6)
    plt.hlines(y_bar, p25, p75, colors=color, linestyles='-', lw=8, alpha=0.8)
    plt.plot(p50, y_bar, 'o', color='white', markeredgecolor=color, markersize=7)
    
    # Range text centered below the bar
    plt.text(p50, y_bar - 0.05, f"Range: [Min: {d_min:.0f}, Max: {d_max:.0f}]",
             ha='center', va='top', color=color, fontsize=17, fontweight='bold')

plot_kde_with_range(duet_merged, "DUET (Merged)", "royalblue", 0.0)
plot_kde_with_range(rwku, "RWKU (forget_level2)", "darkorange", 0.18)

plt.xscale('log')
plt.xlabel('Popularity Score (pop_sum)', fontsize=16)
plt.ylabel('Relative Density', fontsize=16)
plt.title('Popularity Distribution and Diversity: DUET vs RWKU', fontsize=18, fontweight='bold', pad=20)
plt.legend(loc='upper left', frameon=True, fontsize=15)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.ylim(-0.5, 1.1)
plt.yticks([0, 0.5, 1.0])
plt.tick_params(axis='both', labelsize=14)

plt.tight_layout()
output_file = 'pop_sum_diversity_comparison.png'
plt.savefig(output_file, dpi=300)
print(f"Plot saved to {output_file}")
