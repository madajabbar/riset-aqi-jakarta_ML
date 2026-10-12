#!/usr/bin/env python3
"""Generate visualisasi hasil EMD-Transformer-BiLSTM"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Setup
results_dir = Path('our_model/results')
results_dir.mkdir(parents=True, exist_ok=True)

# Data dari sweep
data = {
    'horizon': [5, 7, 12, 24],
    'rmse_mean': [0.5557, 2.7736, 3.1407, 9.2941],
    'rmse_std': [0.0430, 1.7598, 0.4787, 3.2711],
    'mae_mean': [0.3289, 1.4823, 1.8816, 6.1875],
    'mape_mean': [0.2067, 0.8933, 1.1633, 4.0867],
    'improvement': [20.02, 5.41, 4.56, 1.98],
    'persistence_rmse': [11.0805, 12.0493, 14.1141, 17.1268]
}

df = pd.DataFrame(data)

# Per-seed data
seed_data = {
    5: [0.6035, 0.5432, 0.5203],
    7: [1.7332, 4.8055, 1.7822],
    12: [3.6709, 3.0112, 2.7401],
    24: [7.9317, 6.9243, 13.0262]
}

# Color palette
colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

# ============================================================================
# PLOT 1: RMSE per Horizon dengan Error Bar + Improvement Factor
# ============================================================================
fig, ax1 = plt.subplots(figsize=(12, 7))

x = np.arange(len(df['horizon']))
bars = ax1.bar(x, df['rmse_mean'], yerr=df['rmse_std'], 
               capsize=10, 
               color=colors, alpha=0.8, 
               edgecolor='black', linewidth=1.5,
               label='Model RMSE (mean ± std)',
               error_kw=dict(capthick=2, elinewidth=2))

ax1.set_xlabel('Prediction Horizon (hours)', fontsize=13, fontweight='bold')
ax1.set_ylabel('RMSE', fontsize=13, fontweight='bold', color='#2c3e50')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{h}h' for h in df['horizon']], fontsize=11)
ax1.set_title('Model Performance vs Prediction Horizon\nEMD-Transformer-BiLSTM for Jakarta AQI', 
              fontsize=15, fontweight='bold', pad=20)
ax1.tick_params(axis='y', labelcolor='#2c3e50')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 14)

# Annotasi RMSE values
for i, (bar, mean, std) in enumerate(zip(bars, df['rmse_mean'], df['rmse_std'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.3,
            f'{mean:.2f}±{std:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Secondary axis untuk improvement factor
ax2 = ax1.twinx()
ax2.plot(x, df['improvement'], 'ro-', linewidth=3, markersize=12, 
         markerfacecolor='white', markeredgewidth=3,
         label='Improvement vs Persistence', zorder=10)
ax2.set_ylabel('Improvement Factor (×)', fontsize=13, fontweight='bold', color='#c0392b')
ax2.tick_params(axis='y', labelcolor='#c0392b')
ax2.set_ylim(0, 25)

# Annotasi improvement
for i, imp in enumerate(df['improvement']):
    ax2.annotate(f'{imp:.1f}×', 
                (x[i], imp),
                textcoords="offset points", xytext=(0,15),
                ha='center', fontsize=11, fontweight='bold', color='#c0392b')

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig(results_dir / 'analysis_horizon_performance.png', dpi=300, bbox_inches='tight')
print('✓ Saved: analysis_horizon_performance.png')
plt.close()

# ============================================================================
# PLOT 2: Boxplot Distribusi RMSE per Horizon (Seed Stability)
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 7))

horizons = list(seed_data.keys())
rmse_values = [seed_data[h] for h in horizons]

bp = ax.boxplot(rmse_values, tick_labels=[f'{h}h' for h in horizons],
                patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=2),
                capprops=dict(linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=10, alpha=0.7))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

ax.set_xlabel('Prediction Horizon (hours)', fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=13, fontweight='bold')
ax.set_title('RMSE Distribution Across Seeds\n(Boxplot shows stability per horizon)', 
            fontsize=15, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Annotasi outliers dan statistik
for i, (h, vals) in enumerate(zip(horizons, rmse_values)):
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    cv = std_val / mean_val * 100  # Coefficient of variation
    
    # Label CV
    ax.text(i+1, max(vals) + 1.5, 
           f'CV: {cv:.1f}%\n(std: {std_val:.2f})',
           ha='center', fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Mark outliers
    for j, v in enumerate(vals):
        if v > mean_val + 1.5 * std_val:
            ax.annotate(f'Outlier\n(Seed {j})', 
                       (i+1, v),
                       textcoords="offset points", xytext=(15,0),
                       ha='left', fontsize=9, color='red', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

plt.tight_layout()
plt.savefig(results_dir / 'analysis_seed_stability.png', dpi=300, bbox_inches='tight')
print('✓ Saved: analysis_seed_stability.png')
plt.close()

# ============================================================================
# PLOT 3: Comparison Model vs Persistence
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 7))

x = np.arange(len(df['horizon']))
width = 0.35

bars1 = ax.bar(x - width/2, df['rmse_mean'], width, 
               label='EMD-Transformer-BiLSTM', 
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, df['persistence_rmse'], width,
               label='Persistence Baseline',
               color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Prediction Horizon (hours)', fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=13, fontweight='bold')
ax.set_title('Model vs Baseline Comparison\n(Lower is better)', 
            fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f'{h}h' for h in df['horizon']], fontsize=11)
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Annotasi values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

# Highlight improvement dengan arrow
for i in range(len(df['horizon'])):
    ax.annotate('', xy=(i - width/2, df['rmse_mean'][i]), 
               xytext=(i + width/2, df['persistence_rmse'][i]),
               arrowprops=dict(arrowstyle='->', color='green', lw=2, ls='--'))

plt.tight_layout()
plt.savefig(results_dir / 'analysis_model_vs_baseline.png', dpi=300, bbox_inches='tight')
print('✓ Saved: analysis_model_vs_baseline.png')
plt.close()

# ============================================================================
# PLOT 4: Metric Radar Chart (Horizon 5 jam - Best Performance)
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))

categories = ['RMSE\n(inverted)', 'MAE\n(inverted)', 'MAPE\n(inverted)', 
             'Stability', 'Improvement\nvs Baseline']

# Normalize values ke 0-1 (inverted untuk metrics dimana lower = better)
rmse_norm = 1 - (df['rmse_mean'][0] / 10)  # 0.56 → 0.94
mae_norm = 1 - (df['mae_mean'][0] / 10)    # 0.33 → 0.97
mape_norm = 1 - (df['mape_mean'][0] / 10)  # 0.21 → 0.98
stability_norm = 1 - (df['rmse_std'][0] / df['rmse_mean'][0])  # low std = high stability
improvement_norm = df['improvement'][0] / 25  # 20/25 = 0.8

values = [rmse_norm, mae_norm, mape_norm, stability_norm, improvement_norm]
values += values[:1]  # close the polygon

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax.plot(angles, values, 'o-', linewidth=2.5, color='#2ecc71', markersize=10)
ax.fill(angles, values, alpha=0.25, color='#2ecc71')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.set_title('Horizon 5h Performance Profile\n(Normalized 0-1, higher is better)', 
            fontsize=14, fontweight='bold', pad=20)

# Add value labels
for angle, value, category in zip(angles[:-1], values[:-1], categories):
    ax.text(angle, value + 0.05, f'{value:.2f}',
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(results_dir / 'analysis_horizon5_radar.png', dpi=300, bbox_inches='tight')
print('✓ Saved: analysis_horizon5_radar.png')
plt.close()

print('\n' + '='*60)
print('✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY')
print('='*60)
print(f'\nOutput directory: {results_dir.absolute()}')
print('\nFiles created:')
print('  1. analysis_horizon_performance.png - RMSE + improvement factor')
print('  2. analysis_seed_stability.png - Boxplot stability analysis')
print('  3. analysis_model_vs_baseline.png - Comparison with persistence')
print('  4. analysis_horizon5_radar.png - Best horizon performance profile')
