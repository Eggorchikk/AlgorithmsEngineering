import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load results
with open("results/raw_results.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Ensure nice styling
sns.set_theme(style="whitegrid", font_scale=1.2)

# --------------------------
# BAR PLOTS PER GRAPH
# --------------------------
metrics = ["runtime", "pushes", "relabels"]

for graph in ["small", "medium", "large"]:
    sub = df[df["graph"] == graph]

    for metric in metrics:
        plt.figure(figsize=(7,5))
        sns.barplot(data=sub, x="policy", y=metric, palette="viridis")
        plt.title(f"{metric.capitalize()} on {graph.capitalize()} Graph")
        plt.tight_layout()
        plt.savefig(f"results/{graph}_{metric}.png")
        plt.close()

# --------------------------
# LINE PLOT ACROSS GRAPHS
# --------------------------
for metric in metrics:
    plt.figure(figsize=(7,5))
    sns.lineplot(data=df, x="graph", y=metric, hue="policy", marker="o")
    plt.title(f"{metric.capitalize()} Scaling Across Graphs")
    plt.tight_layout()
    plt.savefig(f"results/scaling_{metric}.png")
    plt.close()

print("Plots generated in results/")