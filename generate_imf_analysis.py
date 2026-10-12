#!/usr/bin/env python3
"""Generate IMF decomposition deep-dive analysis plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json, re

results_dir = Path('our_model/results')
results_dir.mkdir(parents=True, exist_ok=True)

# 1. Parse raw data from notebook
nb = json.load(open('riset-aqi-jakarta.ipynb'))
runs = []
current_run = None

for idx in [8, 10]:
    cell = nb['cells'][idx]
    text = ''.join([line for out in cell.get('outputs', []) for line in out.get('text', [])])
    for line in text.split('\n'):
        m_run = re.search(r'===== horizon (\d+) jam \(--horizon (\d+)\) seed (\d+) =====', line)
        if m_run:
            current_run = {
                'horizon': int(m_run.group(1)),
                'ow': int(m_run.group(2)),
                'seed': int(m_run.group(3)),
                'imfs': {}
            }
            runs.append(current_run)
        m_imf = re.search(r'IMF(\d+): val transformer ([\d\.]+) \| linear ([\d\.]+) -> (\w+)', line)
        if m_imf and current_run:
            imf_idx = int(m_imf.group(1))
            tf_val = float(m_imf.group(2))
            lin_val = float(m_imf.group(3))
            choice = m_imf.group(4)
            current_run['imfs'][imf_idx] = {'tf': tf_val, 'lin': lin_val, 'choice': choice}

rows = []
for r in runs:
    for i in range(1, 11):
        info = r['imfs'].get(i, {})
        rows.append({
            'horizon': r['horizon'],
            'seed': r['seed'],
            'imf': i,
            'choice': info.get('choice', 'none'),
            'val_tf': info.get('tf', None),
            'val_lin': info.get('lin', None),
            'chosen_loss': info.get('tf') if info.get('choice') == 'tf' else info.get('lin')
        })
df = pd.DataFrame(rows)

# ============================================================================
# PLOT 1: IMF Architecture Selection Transition Matrix across Horizons
# ============================================================================
horizons = [5, 7, 12, 24]
imfs = list(range(1, 11))

# Fraction of seeds choosing Transformer per (horizon, imf)
tf_matrix = np.zeros((len(imfs), len(horizons)))
for j, h in enumerate(horizons):
    for i, imf in enumerate(imfs):
        subset = df[(df['horizon'] == h) & (df['imf'] == imf)]
        tf_count = (subset['choice'] == 'tf').sum()
        tf_matrix[i, j] = tf_count / len(subset) if len(subset) > 0 else 0

fig, ax = plt.subplots(figsize=(8, 9))
cax = ax.matshow(tf_matrix, cmap='Blues', vmin=0, vmax=1)

ax.set_xticks(range(len(horizons)))
ax.set_xticklabels([f'{h}h Horizon' for h in horizons], fontsize=11, fontweight='bold')
ax.set_yticks(range(len(imfs)))
ax.set_yticklabels([f'IMF {imf} (High $\\rightarrow$ Low freq)' if imf in [1, 10] else f'IMF {imf}' for imf in imfs], fontsize=10)

ax.set_title('Architecture Auto-Selection Dynamics\n(Fraction of Seeds choosing Transformer vs Linear)', fontsize=13, fontweight='bold', pad=25)

# Text inside cells
for i in range(len(imfs)):
    for j in range(len(horizons)):
        ratio = tf_matrix[i, j]
        label = "Transformer" if ratio == 1.0 else ("Linear" if ratio == 0.0 else f"{int(ratio*3)}/3 TF")
        color = "white" if ratio > 0.5 else "black"
        ax.text(j, i, label, ha='center', va='center', fontsize=9, fontweight='bold', color=color)

fig.colorbar(cax, ax=ax, fraction=0.04, pad=0.04, label='Probability of Selecting Transformer')
plt.tight_layout()
p1 = results_dir / 'analysis_imf_selection_heatmap.png'
plt.savefig(p1, dpi=300, bbox_inches='tight')
plt.close()
print(f'✓ Saved: {p1}')

# ============================================================================
# PLOT 2: Validation Loss Escalation by IMF across Horizons
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))

loss_pivot = df.pivot_table(index='imf', columns='horizon', values='chosen_loss', aggfunc='mean')

colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c']
markers = ['o', 's', '^', 'D']

for j, h in enumerate(horizons):
    ax.plot(imfs, loss_pivot[h], marker=markers[j], color=colors[j], linewidth=2.2, markersize=8, label=f'Horizon {h}h')

ax.set_xlabel('IMF Component (1 = High Freq Noise, 10 = Low Freq Trend)', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Loss (MSE)', fontsize=12, fontweight='bold')
ax.set_title('Error Accumulation per IMF Across Horizons\n(Middle IMFs 3–5 degrade severely at H=24h)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(imfs)
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.legend(fontsize=11, loc='upper right')

# Highlight mid frequencies
ax.axvspan(3, 5, color='red', alpha=0.08, label='Error Bottleneck Zone (IMF 3-5)')

plt.tight_layout()
p2 = results_dir / 'analysis_imf_loss_progression.png'
plt.savefig(p2, dpi=300, bbox_inches='tight')
plt.close()
print(f'✓ Saved: {p2}')

print('\n✓ IMF Deep Analysis Plots Generated.')
