import matplotlib.pyplot as plt

# Full model BLEU-4 score
full_bleu4 = 18.03

# Ablation BLEU-4 scores
ablation = {
    "Freq-Aware": 15.6,
    "Motion Blur": 16.8,
    "Swarm Temporal": 13.98,
    "Causal-Aware": 12.54
}

# Compute performance drop
components = list(ablation.keys())
drops = [full_bleu4 - v for v in ablation.values()]

# Plot
plt.figure(figsize=(8, 5))
plt.bar(components, drops)

# Labels
plt.ylabel("BLEU-4 Performance Drop", fontsize=12)
# plt.xlabel("Removed Component")
plt.title("Component Contribution Analysis (TD-UAV)", fontsize=14, fontweight='bold')

# Rotate x-axis labels for readability
plt.xticks(rotation=20)

# Show values on bars
for i, v in enumerate(drops):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('component_contribution.png', dpi=600, bbox_inches='tight')
plt.show()