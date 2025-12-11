import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Load results
with open("results/raw_results.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

sns.set_theme(style="whitegrid", font_scale=1.2)

# BAR PLOTS PER GRAPH
metrics = ["runtime", "pushes", "relabels"]

for graph in ["small", "medium", "large"]:
    sub = df[df["graph"] == graph]

    for metric in metrics:
        plt.figure(figsize=(7,5))
        ax = sns.barplot(data=sub, x="policy", y=metric, palette="pastel")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10)

        plt.title(f"{metric.capitalize()} on {graph.capitalize()} Graph")
        plt.tight_layout()
        plt.savefig(f"results/{graph}_{metric}.png")
        plt.close()

# LINE PLOT ACROSS GRAPHS
for metric in metrics:
    plt.figure(figsize=(7,5))
    sns.lineplot(data=df, x="graph", y=metric, hue="policy", marker="o")
    plt.title(f"{metric.capitalize()} Scaling Across Graphs")
    plt.tight_layout()
    plt.savefig(f"results/scaling_{metric}.png")
    plt.close()


# RADAR PLOT FUNCTION
def radar_plot(subdf, graph_name):
    categories = ["runtime", "pushes", "relabels"]
    N = len(categories)

    # Normalize values per metric so all fit 0..1
    sub = subdf.copy()
    for c in categories:
        max_val = sub[c].max()
        if max_val > 0:
            sub[c] = sub[c] / max_val
        else:
            sub[c] = 0

    # Angles around the circle
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    for policy in sub["policy"].unique():
        row = sub[sub["policy"] == policy][categories].values.flatten().tolist()
        row += row[:1]

        ax.plot(angles, row, linewidth=2, label=policy)
        ax.fill(angles, row, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title(f"Radar plot for {graph_name.capitalize()} Graph")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f"results/{graph_name}_radar.png")
    plt.close()


# GENERATE RADAR PLOTS
for graph in ["small", "medium", "large"]:
    sub = df[df["graph"] == graph]
    radar_plot(sub, graph)

print("All plots (bar, line, radar) generated in results/")

