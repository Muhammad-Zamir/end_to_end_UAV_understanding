import matplotlib.pyplot as plt

# ---- Data ----
labels_outer = [
    "Caption",
    "Yes/No",
    "Motion",
    "Environment",
    "Swarm",
    "Single"
]

sizes_outer = [17, 9, 23, 13, 23, 15]

labels_inner = ["Caption", "Question Answer"]
sizes_inner = [17, 83]

# ---- Color Strategy ----
# Caption → Blue family
caption_inner_color = "#E65100"
caption_outer_color = "#FFB74D"

# Question Answer → Orange family (gradient shades)
qa_inner_color = "#2F5597"
qa_outer_colors = [
    "#1E88E5",  # YesNo (slightly stronger highlight)
    "#42A5F5",
    "#64B5F6",
    "#90CAF9",
    "#BBDEFB"
]

# Combine outer colors
colors_outer = [caption_outer_color] + qa_outer_colors
colors_inner = [caption_inner_color, qa_inner_color]

# ---- Plot ----
fig, ax = plt.subplots(figsize=(7, 7))

# Outer ring
ax.pie(
    sizes_outer,
    labels=[f"{l}\n{s}%" for l, s in zip(labels_outer, sizes_outer)],
    radius=1,
    colors=colors_outer,
    labeldistance=0.85,
    wedgeprops=dict(width=0.4, edgecolor='white'),
    textprops=dict(fontsize=12)
)

# Inner ring
ax.pie(
    sizes_inner,
    labels=[f"{l}\n{s}%" for l, s in zip(labels_inner, sizes_inner)],
    radius=0.6,
    colors=colors_inner,
    labeldistance=0.6,
    wedgeprops=dict(width=0.35, edgecolor='white'),
    textprops=dict(fontsize=12, weight='bold', color='white')
)

# Center hole (clean donut look)
centre_circle = plt.Circle((0, 0), 0.25, fc='white')
fig.gca().add_artist(centre_circle)

# Final styling
ax.set(aspect="equal")
plt.tight_layout()
plt.savefig('pai_chart2.png', dpi=600, bbox_inches='tight')
plt.show()