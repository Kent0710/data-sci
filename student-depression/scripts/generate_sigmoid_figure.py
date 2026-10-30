import matplotlib.pyplot as plt
import numpy as np

# Set clean black and white canvas style matching academic figures
plt.style.use('default')
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Generate input score range
z = np.linspace(-6, 6, 300)
phi_z = 1 / (1 + np.exp(-z))

# Plot sigmoid curve with crisp black line
ax.plot(z, phi_z, color='black', linewidth=2.4)

# Grid lines matching academic textbook figures
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.0)
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.0)
ax.axhline(0.0, color='gray', linestyle=':', linewidth=1.0)
ax.axvline(0.0, color='black', linewidth=1.2)

# Sigmoid formula annotation on upper-left quadrant
formula_text = r"$\phi(z) = \frac{1}{1 + e^{-z}}$"
ax.text(-4.8, 0.82, formula_text, fontsize=16, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.0))

# Inflection point at (0, 0.5)
ax.plot(0, 0.5, marker='o', markersize=6, markerfacecolor='black', markeredgecolor='black')
ax.text(0.3, 0.48, r"$(0, 0.5)$", fontsize=10, fontweight='bold', color='black')

# Axis limits and labels
ax.set_xlim(-6, 6)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
ax.set_yticks([0.0, 0.5, 1.0])

ax.set_xlabel('z', fontsize=12, fontweight='bold', color='black')
ax.set_ylabel(r'$\phi(z)$', fontsize=12, fontweight='bold', color='black')
ax.set_title('Standard Sigmoid (Logistic) Activation Curve', fontsize=13, fontweight='bold', pad=12, color='black')

# Spine styling
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.2)
ax.tick_params(colors='black', labelsize=10.5)

plt.tight_layout()
output_path = '/Users/kent/Projects/data-sci/student-depression/figures/13_sigmoid_curve.png'
plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
print(f'Successfully generated sigmoid curve figure: {output_path}')
