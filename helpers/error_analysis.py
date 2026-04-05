import matplotlib.pyplot as plt

# Labels
labels = [
    "Intent Reasoning Error",
    "Binary Intent Error",
    "Motion Misunderstanding",
    "Visual Perception Error",
    "Scale Estimation",
    "Correct"
]

# Data
ours = [5.99, 1.81, 9.41, 5.41, 2.61, 74.77]
baseline = [9.18, 3.36, 13.04, 6.60, 5.54, 62.48]

# Colors (lighter "Correct")
colors = [
    "#A58ACF",  # Intent reasoning
    "#C9C75A",  # Binary intent
    "#D291BC",  # Motion
    "#6A8FBF",  # Visual
    "#76B061",  # Scale
    "#BFDDE6"   # Correct (lighter blue)
]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

# Function to format percentage text
def autopct_format(pct):
    return f"{pct:.0f}%" if pct > 0 else ""

# --- Video-LLaVA ---
wedges1, _, autotexts1 = axes[0].pie(
    baseline,
    autopct=autopct_format,
    colors=colors,
    startangle=140
)

axes[0].set_title("(a) Video-LLaVA", y=-0.1, fontweight='bold', fontsize=16)

# --- Ours ---
wedges2, _, autotexts2 = axes[1].pie(
    ours,
    autopct=autopct_format,
    colors=colors,
    startangle=140
)

axes[1].set_title("(b)  Ours", y=-0.1, fontweight='bold', fontsize=16)

# Set white percentage text
for autotexts in [autotexts1, autotexts2]:
    for text in autotexts:
        text.set_color("white")
        text.set_fontsize(14)

# Equal aspect ratio
for ax in axes:
    ax.axis('equal')

# Single legend on top
fig.legend(
    wedges1,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.05),
    fontsize=14
)

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=600, bbox_inches='tight')
plt.show()