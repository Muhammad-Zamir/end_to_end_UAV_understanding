import numpy as np
import matplotlib.pyplot as plt

categories = [
    "Caption",
    "Yes/No",
    "Motion",
    "Environment",
    "Swarm",
    "Single"
]

ours = np.array([71.39, 91.33, 55.04, 74.17, 74.77, 83.96]) / 100
video_llava = np.array([57.34, 82.5, 37.23, 60.88, 63.47, 72.57]) / 100

N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

angles += angles[:1]
ours = np.concatenate((ours, [ours[0]]))
video_llava = np.concatenate((video_llava, [video_llava[0]]))

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)

# Y-axis
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0.1, 0.9, 0.1))
ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0.1, 0.9, 0.1)], fontsize=10)

# ---- KEY FIXES ----
ax.set_axisbelow(False)  # bring grid ABOVE fills
ax.grid(color='gray', linestyle='-', linewidth=0.8, alpha=0.7)

# Plot
ax.plot(angles, ours, color="blue", linewidth=2, label="Ours", zorder=3)
ax.fill(angles, ours, color="blue", alpha=0.25, zorder=2)

ax.plot(angles, video_llava, color="red", linewidth=2, label="Video-LLaVA", zorder=3)
ax.fill(angles, video_llava, color="red", alpha=0.25, zorder=2)

# Legend
plt.legend(loc="upper center", fontsize=12, bbox_to_anchor=(0.5, 1.15), ncol=2)

plt.tight_layout()
plt.savefig('radar_chart.png', dpi=600, bbox_inches='tight')
plt.show()