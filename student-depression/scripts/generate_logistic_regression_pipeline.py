import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set clean black and white canvas style
fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis('off')

# Box styling helper
box_props = dict(boxstyle='square,pad=0.6', facecolor='white', edgecolor='black', linewidth=1.5)
round_box_props = dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='black', linewidth=1.5)
arrow_props = dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.5)

# -------------------------------------------------------------
# 1. Input Features Block (Left)
# -------------------------------------------------------------
ax.text(1.3, 4.4, "Input Features (x)", fontsize=11, fontweight='bold', ha='center', color='black')

features = [
    r"$x_1$: Academic Pressure",
    r"$x_2$: Financial Stress",
    r"$x_3$: Suicidal Thoughts",
    r"$\vdots$",
    r"$x_n$: Lifestyle & Demographics"
]
y_positions = [3.7, 3.0, 2.3, 1.6, 0.9]

for feat, y_pos in zip(features, y_positions):
    if feat == r"$\vdots$":
        ax.text(1.3, y_pos, feat, fontsize=14, fontweight='bold', ha='center', va='center')
    else:
        ax.text(1.3, y_pos, feat, fontsize=9.5, ha='center', va='center', bbox=box_props)
        # Arrow to summation
        ax.annotate('', xy=(3.8, 2.3), xytext=(2.55, y_pos), arrowprops=arrow_props)

# Weights labels along arrows
ax.text(3.1, 3.4, r"$w_1$", fontsize=9.5, style='italic', color='black')
ax.text(3.1, 2.85, r"$w_2$", fontsize=9.5, style='italic', color='black')
ax.text(3.1, 2.45, r"$w_3$", fontsize=9.5, style='italic', color='black')
ax.text(3.1, 1.5, r"$w_n$", fontsize=9.5, style='italic', color='black')

# -------------------------------------------------------------
# 2. Linear Combination Node (Center-Left)
# -------------------------------------------------------------
ax.text(4.7, 4.4, "1. Linear Combination", fontsize=11, fontweight='bold', ha='center', color='black')

# Summation Circle / Box
sum_box = patches.FancyBboxPatch((3.8, 1.2), 1.8, 2.2, boxstyle="round,pad=0.2", facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(sum_box)

ax.text(4.7, 2.8, r"$\sum + \ b$", fontsize=15, fontweight='bold', ha='center', va='center')
ax.text(4.7, 2.1, r"$z = \mathbf{w}^T \mathbf{x} + b$", fontsize=11, fontweight='bold', ha='center', va='center')
ax.text(4.7, 1.5, "Unbounded Score\n" + r"$(-\infty < z < +\infty)$", fontsize=8.5, ha='center', va='center', color='black', style='italic')

# Arrow to Sigmoid
ax.annotate('', xy=(6.6, 2.3), xytext=(5.6, 2.3), arrowprops=arrow_props)
ax.text(6.1, 2.55, r"$z$", fontsize=11, fontweight='bold', ha='center', color='black')

# -------------------------------------------------------------
# 3. Sigmoid Activation Function (Center-Right)
# -------------------------------------------------------------
ax.text(8.0, 4.4, "2. Sigmoid Activation", fontsize=11, fontweight='bold', ha='center', color='black')

sig_box = patches.FancyBboxPatch((6.6, 0.9), 2.8, 2.8, boxstyle="round,pad=0.2", facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(sig_box)

ax.text(8.0, 3.1, r"$\sigma(z) = \frac{1}{1 + e^{-z}}$", fontsize=12.5, fontweight='bold', ha='center', va='center')

# Mini S-Curve inside box
z_mini = np.linspace(-3, 3, 50)
sig_mini = 1 / (1 + np.exp(-z_mini))
# Scale and position to (x: 7.1 to 8.9, y: 1.3 to 2.2)
x_plot = 7.1 + (z_mini + 3) / 6 * 1.8
y_plot = 1.3 + sig_mini * 0.9

ax.plot(x_plot, y_plot, color='black', linewidth=1.8)
# Axis lines for mini plot
ax.plot([7.1, 8.9], [1.3, 1.3], color='black', linewidth=0.8) # x axis
ax.plot([8.0, 8.0], [1.3, 2.2], color='black', linewidth=0.8, linestyle=':') # center line
ax.text(8.0, 1.05, "Squashes to range (0, 1)", fontsize=8.5, ha='center', va='center', style='italic')

# Arrow to Probability Output
ax.annotate('', xy=(10.4, 2.3), xytext=(9.4, 2.3), arrowprops=arrow_props)
ax.text(9.9, 2.55, r"$\sigma(z)$", fontsize=10, fontweight='bold', ha='center', color='black')

# -------------------------------------------------------------
# 4. Predicted Probability & Decision (Right)
# -------------------------------------------------------------
ax.text(11.6, 4.4, "3. Class Probability", fontsize=11, fontweight='bold', ha='center', color='black')

out_box = patches.FancyBboxPatch((10.4, 1.1), 2.4, 2.4, boxstyle="square,pad=0.2", facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(out_box)

ax.text(11.6, 2.9, r"$\hat{y} = P(\text{Depression}=1)$", fontsize=10.5, fontweight='bold', ha='center', va='center')
ax.text(11.6, 2.3, r"Probability: $[0.0, 1.0]$", fontsize=9.5, ha='center', va='center')

# Decision threshold rule inside output box
ax.plot([10.6, 12.6], [1.9, 1.9], color='black', linewidth=0.8, linestyle='--')
ax.text(11.6, 1.55, r"$\hat{y} \geq 0.5 \to$ Depressed (1)" + "\n" + r"$\hat{y} < 0.5 \to$ Not Depressed (0)", fontsize=8.0, ha='center', va='center', style='italic')

# Border around entire canvas for clean framed appearance
canvas_border = patches.Rectangle((0.1, 0.1), 12.8, 4.8, linewidth=1.2, edgecolor='black', facecolor='none')
ax.add_patch(canvas_border)

plt.tight_layout()
output_path = '/Users/kent/Projects/data-sci/student-depression/figures/11_logistic_regression_pipeline.png'
plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
print(f"Successfully generated pipeline diagram: {output_path}")
