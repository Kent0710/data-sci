import matplotlib.pyplot as plt
import numpy as np

# Set clean black and white canvas style matching academic figures
plt.style.use('default')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
fig.patch.set_facecolor('white')

# -------------------------------------------------------------
# Panel 1: Transparent & Interpretable Weights (Fixed Text Spacing)
# -------------------------------------------------------------
ax1.set_facecolor('white')
features = ['Suicidal Thoughts', 'Academic Pressure', 'Financial Stress', 'Study Satisfaction', 'Dietary Habits', 'Age']
weights = [1.85, 1.25, 0.95, -0.65, -0.85, -1.10]
y_pos = np.arange(len(features))

bars = ax1.barh(y_pos, weights, color='white', edgecolor='black', linewidth=1.5, height=0.55)

# Pattern hatching for black and white distinction
for i, bar in enumerate(bars):
    if weights[i] > 0:
        bar.set_hatch('///')
    else:
        bar.set_hatch('...')

ax1.axvline(0, color='black', linewidth=1.2)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(features, fontsize=9.5, fontweight='bold', color='black')
ax1.set_xlabel('Model Coefficient (Weight)', fontsize=10, fontweight='bold', color='black')
ax1.set_title('1. Transparent Weights\n(Actionable Risk Factors)', fontsize=11, fontweight='bold', pad=10, color='black')
ax1.set_xlim(-1.7, 2.3)

# Clear annotations placed safely away from bar labels
ax1.text(1.2, 0.8, 'Risk Factors (+)\nIncreases Odds', color='black', fontsize=8.5, ha='center', style='italic')
ax1.text(-1.0, 3.8, 'Protective Factors (-)\nDecreases Odds', color='black', fontsize=8.5, ha='center', style='italic')

for spine in ax1.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.2)
ax1.tick_params(colors='black')
ax1.grid(axis='x', linestyle=':', color='gray', alpha=0.6)

# -------------------------------------------------------------
# Panel 2: Calibrated Probabilities & Triage
# -------------------------------------------------------------
ax2.set_facecolor('white')
z = np.linspace(-6, 6, 200)
sigmoid = 1 / (1 + np.exp(-z))

ax2.plot(z, sigmoid, color='black', linewidth=2.0)
ax2.axhline(0.5, color='black', linestyle='--', linewidth=1.2, label='Decision Boundary (0.5)')
ax2.axhline(0.75, color='gray', linestyle=':', linewidth=1.0)
ax2.axhline(0.25, color='gray', linestyle=':', linewidth=1.0)

# Triage brackets / labels
ax2.text(3.5, 0.83, 'High Priority Triage\n(P > 0.75)', fontsize=8.5, fontweight='bold', ha='center', color='black')
ax2.text(3.5, 0.53, 'Moderate Concern\n(0.50 - 0.75)', fontsize=8.0, ha='center', color='black')
ax2.text(3.5, 0.12, 'Low Risk Monitor\n(P < 0.25)', fontsize=8.0, ha='center', color='black')

ax2.set_title('2. Calibrated Probabilities\n(Student Triage & Staging)', fontsize=11, fontweight='bold', pad=10, color='black')
ax2.set_xlabel('Weighted Linear Score (z)', fontsize=10, fontweight='bold', color='black')
ax2.set_ylabel('Probability P(Depression = 1)', fontsize=10, fontweight='bold', color='black')
ax2.set_ylim(-0.02, 1.05)
ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

for spine in ax2.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.2)
ax2.tick_params(colors='black')
ax2.grid(True, linestyle=':', color='gray', alpha=0.6)

# -------------------------------------------------------------
# Panel 3: Built-in L2 Regularization (78 features)
# -------------------------------------------------------------
ax3.set_facecolor('white')
np.random.seed(42)
feat_idx = np.arange(1, 16)
unreg_w = np.random.normal(0, 1.8, 15)
reg_w = unreg_w / 2.2

markerline1, stemlines1, baseline1 = ax3.stem(feat_idx - 0.15, unreg_w, linefmt='--', markerfmt='o', basefmt='k-', label='Without L2 (Noisy / Overfitting)')
plt.setp(stemlines1, color='gray', linewidth=1.0)
plt.setp(markerline1, color='gray', markerfacecolor='white', markeredgecolor='gray', markersize=5)

markerline2, stemlines2, baseline2 = ax3.stem(feat_idx + 0.15, reg_w, linefmt='-', markerfmt='s', basefmt='k-', label='With L2 Penalty (Stable Weights)')
plt.setp(stemlines2, color='black', linewidth=1.4)
plt.setp(markerline2, color='black', markerfacecolor='black', markeredgecolor='black', markersize=5)

ax3.set_title('3. L2 Regularization\n(Controls 78 Feature Weights)', fontsize=11, fontweight='bold', pad=10, color='black')
ax3.set_xlabel('Sample Feature Index (1 to 15 of 78)', fontsize=10, fontweight='bold', color='black')
ax3.set_ylabel('Weight Value', fontsize=10, fontweight='bold', color='black')
ax3.legend(frameon=True, edgecolor='black', facecolor='white', fontsize=8.0, loc='upper right')

for spine in ax3.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.2)
ax3.tick_params(colors='black')
ax3.grid(True, linestyle=':', color='gray', alpha=0.6)

plt.tight_layout()
output_path = '/Users/kent/Projects/data-sci/student-depression/figures/10_logistic_regression_advantages.png'
plt.savefig(output_path, dpi=300, facecolor='white')
print(f'Successfully updated figure: {output_path}')
