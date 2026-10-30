import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set clean black and white canvas style matching academic figures
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
fig.patch.set_facecolor('white')

# -------------------------------------------------------------
# Panel 1: Geometric Constraint (L2 Ball & Loss Contours)
# -------------------------------------------------------------
ax1.set_facecolor('white')

# Loss function ellipses (Binary Cross-Entropy Contours)
center = (2.2, 1.8)
for r in [0.7, 1.2, 1.7, 2.3]:
    ellipse = patches.Ellipse(center, r*1.8, r*1.2, angle=35, edgecolor='gray',
                              facecolor='none', linestyle='--', linewidth=1.2)
    ax1.add_patch(ellipse)

# L2 Constraint Circle (Ridge Penalty: w1^2 + w2^2 <= C)
circle = patches.Circle((0, 0), 1.5, edgecolor='black', facecolor='gray',
                        alpha=0.15, linewidth=2.0, linestyle='-')
ax1.add_patch(circle)
# Border for circle
circle_border = patches.Circle((0, 0), 1.5, edgecolor='black', facecolor='none',
                               linewidth=2.0, linestyle='-')
ax1.add_patch(circle_border)

# Optimal points
ax1.plot(center[0], center[1], 'o', color='gray', markersize=6)
ax1.text(center[0] + 0.1, center[1] + 0.15, 'Unconstrained\nMinimum (Overfitting)',
         fontsize=8.5, color='gray', style='italic')

# Point of tangency (L2 Regularized Solution)
tangent_point = (1.1, 1.0)
ax1.plot(tangent_point[0], tangent_point[1], 's', color='black', markersize=7)
ax1.text(tangent_point[0] - 0.2, tangent_point[1] + 0.3,
         'L2 Regularized Solution\n(Weight Shrunk to Boundary)',
         fontsize=9, fontweight='bold', color='black')

# Coordinate axes
ax1.axhline(0, color='black', linewidth=1.0)
ax1.axvline(0, color='black', linewidth=1.0)

ax1.set_xlim(-2.2, 3.8)
ax1.set_ylim(-2.2, 3.2)
ax1.set_xlabel('Feature Weight 1 ($w_1$)', fontsize=10.5, fontweight='bold', color='black')
ax1.set_ylabel('Feature Weight 2 ($w_2$)', fontsize=10.5, fontweight='bold', color='black')
ax1.set_title('A. Geometric L2 Penalty\n(Restricts Total Weight Magnitude)', fontsize=11, fontweight='bold', pad=10)

ax1.text(0, -0.7, r'L2 Penalty Ball: $\sum w_j^2 \leq C$', fontsize=9.5, ha='center',
         fontweight='bold', color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.8))

for spine in ax1.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.2)
ax1.tick_params(colors='black')
ax1.grid(True, linestyle=':', color='gray', alpha=0.5)

# -------------------------------------------------------------
# Panel 2: Weight Shrinkage across 78 Transformed Features
# -------------------------------------------------------------
ax2.set_facecolor('white')

np.random.seed(101)
n_features = 20
feature_indices = np.arange(1, n_features + 1)

# Large unregularized weights
unreg_weights = np.random.normal(0, 2.5, n_features)
# L2 shrinks all weights proportionally towards zero
reg_weights = unreg_weights / (1.0 + 1.8)

marker1, stem1, base1 = ax2.stem(feature_indices - 0.15, unreg_weights, linefmt='--', markerfmt='o', basefmt='k-', label='Standard Logistic Loss (Noisy / Inflated)')
plt.setp(stem1, color='gray', linewidth=1.1)
plt.setp(marker1, color='gray', markerfacecolor='white', markeredgecolor='gray', markersize=5)

marker2, stem2, base2 = ax2.stem(feature_indices + 0.15, reg_weights, linefmt='-', markerfmt='s', basefmt='k-', label='With L2 Penalty (Shrunk & Stable)')
plt.setp(stem2, color='black', linewidth=1.4)
plt.setp(marker2, color='black', markerfacecolor='black', markeredgecolor='black', markersize=5)

ax2.axhline(0, color='black', linewidth=1.0)
ax2.set_xlim(0, n_features + 1)
ax2.set_ylim(-6.5, 6.5)
ax2.set_xlabel('Sample Features (1 to 20 of 78 Features)', fontsize=10.5, fontweight='bold', color='black')
ax2.set_ylabel('Learned Weight Magnitude', fontsize=10.5, fontweight='bold', color='black')
ax2.set_title('B. Weight Shrinkage Effect\n(Prevents Dominance of Spurious Indicators)', fontsize=11, fontweight='bold', pad=10)

ax2.legend(frameon=True, edgecolor='black', facecolor='white', fontsize=8.5, loc='upper right')

for spine in ax2.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.2)
ax2.tick_params(colors='black')
ax2.grid(True, linestyle=':', color='gray', alpha=0.5)

plt.tight_layout()
output_path = '/Users/kent/Projects/data-sci/student-depression/figures/14_l2_regularization.png'
plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
print(f'Successfully generated L2 regularization figure: {output_path}')
