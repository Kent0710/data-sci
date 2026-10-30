import matplotlib.pyplot as plt
import numpy as np

# Set clean black and white canvas style matching academic figures
plt.style.use('default')
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Generate Sigmoid curve
z = np.linspace(-6, 6, 300)
p = 1 / (1 + np.exp(-z))

# Plot Sigmoid curve
ax.plot(z, p, color='black', linewidth=2.4, label=r'Sigmoid Function: $P(Y=1|z) = \frac{1}{1 + e^{-z}}$')

# Decision threshold lines (at P = 0.5 and z = 0)
ax.axhline(0.5, color='black', linestyle='--', linewidth=1.4)
ax.axvline(0, color='black', linestyle=':', linewidth=1.2)

# Shading for decision regions
# Left region: Not Depressed (0)
ax.fill_between(z[z <= 0], 0, 1, color='gray', alpha=0.10)
# Right region: Depressed (1)
ax.fill_between(z[z >= 0], 0, 1, color='gray', alpha=0.25)

# Center threshold point
ax.plot(0, 0.5, marker='o', markersize=7, markerfacecolor='black', markeredgecolor='black')
ax.text(0.25, 0.47, 'Decision Threshold\n(0, 0.50)', fontsize=9.5, fontweight='bold', color='black')

# Region 1: Not Depressed Label
ax.text(-3.3, 0.78, 'CLASS 0: NOT DEPRESSED\nPredicted Label = 0', fontsize=10, fontweight='bold', ha='center', color='black')
ax.text(-3.3, 0.68, r'$P(Y=1|z) < 0.50$' + '\n' + r'($z < 0$)', fontsize=9, ha='center', color='black', style='italic')

# Region 2: Depressed Label
ax.text(3.3, 0.28, 'CLASS 1: DEPRESSED\nPredicted Label = 1', fontsize=10, fontweight='bold', ha='center', color='black')
ax.text(3.3, 0.18, r'$P(Y=1|z) \geq 0.50$' + '\n' + r'($z \geq 0$)', fontsize=9, ha='center', color='black', style='italic')

# Axis styling and labels
ax.set_xlabel('Weighted Linear Score (z)', fontsize=11, fontweight='bold', color='black')
ax.set_ylabel('Predicted Probability P(Depression = 1)', fontsize=11, fontweight='bold', color='black')
ax.set_title('Logistic Regression Decision Threshold (Threshold = 0.50)', fontsize=12, fontweight='bold', pad=12, color='black')

ax.set_xlim(-6, 6)
ax.set_ylim(-0.02, 1.05)
ax.set_yticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])

for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.2)
ax.tick_params(colors='black')
ax.grid(True, linestyle=':', color='gray', alpha=0.6)

ax.legend(frameon=True, edgecolor='black', facecolor='white', fontsize=9.5, loc='upper left')

plt.tight_layout()
output_path = '/Users/kent/Projects/data-sci/student-depression/figures/12_decision_threshold.png'
plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
print(f'Successfully generated decision threshold diagram: {output_path}')
