import json
import random
from collections import deque
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class PushRelabel:
    def __init__(self, n, policy="fifo"):
        self.n = n
        self.policy = policy

        # Graph
        self.adj = [[] for _ in range(n)]
        self.cap = [[0] * n for _ in range(n)]

        # Stats
        self.pushes = 0
        self.relabels = 0

    def add_edge(self, u, v, w):
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.cap[u][v] += w   # Handle multiedges by accumulation

    # MAIN MAX-FLOW ROUTINE
    def max_flow(self, s, t):
        n = self.n
        height = [0] * n
        excess = [0] * n

        # Initialize heights
        height[s] = n

        # Preflow: Saturate edges from source
        for v in self.adj[s]:
            flow = self.cap[s][v]
            if flow > 0:
                self.cap[s][v] -= flow
                self.cap[v][s] += flow
                excess[v] += flow
                excess[s] -= flow

        # Initialize active nodes (exclude s, t)
        active = [u for u in range(n) if u != s and u != t and excess[u] > 0]

        # Convert active into structure depending on policy
        if self.policy == "fifo":
            active_queue = deque(active)
        elif self.policy == "highest":
            active_queue = active[:]
        elif self.policy == "lowest":
            active_queue = active[:]
        elif self.policy == "random":
            active_queue = active[:]

        def push(u, v):
            send = min(excess[u], self.cap[u][v])
            self.cap[u][v] -= send
            self.cap[v][u] += send
            excess[u] -= send
            excess[v] += send
            self.pushes += 1
            
            if v != s and v != t and excess[v] == send:
                add_active(v)

        def relabel(u):
            self.relabels += 1
            min_h = float('inf')
            for v in self.adj[u]:
                if self.cap[u][v] > 0:
                    min_h = min(min_h, height[v])
            height[u] = min_h + 1

        def discharge(u):
            # Try pushing because we have excess
            while excess[u] > 0:
                pushed = False
                for v in self.adj[u]:
                    if self.cap[u][v] > 0 and height[u] == height[v] + 1:
                        push(u, v)
                        pushed = True
                        if excess[u] == 0:
                            break
                if not pushed:
                    relabel(u)

        # Helper to select next node
        def get_next_active():
            if self.policy == "fifo":
                if active_queue:
                    return active_queue.popleft()
                return None

            elif self.policy == "highest":
                if not active_queue:
                    return None
                return max(active_queue, key=lambda node: height[node])

            elif self.policy == "lowest":
                if not active_queue:
                    return None
                return min(active_queue, key=lambda node: height[node])

            elif self.policy == "random":
                if not active_queue:
                    return None
                return random.choice(active_queue)

        def add_active(u):
            if u != s and u != t and excess[u] > 0:
                if self.policy == "fifo":
                    active_queue.append(u)
                else:
                    active_queue.append(u)

        # Main Loop
        while True:
            u = get_next_active()
            if u is None:
                break

            # Remove from active set
            if self.policy != "fifo":
                if u in active_queue:
                    active_queue.remove(u)

            old_height = height[u]
            discharge(u)

            # If height increased, some policies reposition node
            old_height = height[u]
            discharge(u)
            if excess[u] > 0:
                if height[u] > old_height:
                    add_active(u)


        return excess[t]
    
def run_all_policies(graph_builder):
    policies = ["fifo", "highest", "lowest", "random"]

    for policy in policies:
        g = graph_builder(policy)
        maxflow = g.max_flow(0, g.n - 1)
        print(f"\nPolicy: {policy}")
        print("Max Flow:", maxflow)
        print("Pushes:", g.pushes)
        print("Relabels:", g.relabels)

    
def build_small_graph(policy):
    import random
    random.seed(1)

    n = 50
    m = 2   # edges per new node
    g = PushRelabel(n, policy)

    # Start with clique of (m+1) nodes
    for u in range(m + 1):
        for v in range(u + 1, m + 1):
            cap = random.randint(5, 30)
            g.add_edge(u, v, cap)

    degree_list = []
    for i in range(m + 1):
        degree_list.extend([i] * m)

    # Preferential attachment
    for new in range(m + 1, n):
        targets = random.sample(degree_list, m)
        for t in targets:
            cap = random.randint(5, 30)
            g.add_edge(new, t, cap)
        degree_list.extend(targets)
        degree_list.extend([new] * m)

    return g
def build_medium_graph(policy):
    import random
    random.seed(2)

    n = 200
    num_clusters = 4
    intra_density = 0.25
    inter_density = 0.005

    g = PushRelabel(n, policy)
    cluster_size = n // num_clusters
    clusters = [list(range(i*cluster_size, (i+1)*cluster_size))
                for i in range(num_clusters)]

    # Dense intra-cluster edges
    for cluster in clusters:
        for u in cluster:
            for v in cluster:
                if u != v and random.random() < intra_density:
                    g.add_edge(u, v, random.randint(5, 30))

    # Sparse inter-cluster edges (bottlenecks)
    for i in range(num_clusters):
        for j in range(i+1, num_clusters):
            for u in clusters[i]:
                for v in clusters[j]:
                    if random.random() < inter_density:
                        g.add_edge(u, v, random.randint(10, 40))

    return g
def build_large_graph(policy):
    import random
    random.seed(3)

    rows, cols = 40, 40
    n = rows * cols
    g = PushRelabel(n, policy)

    def node(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            u = node(r, c)

            # Down edge
            if r + 1 < rows:
                g.add_edge(u, node(r+1, c), random.randint(1, 20))

            # Right edge
            if c + 1 < cols:
                g.add_edge(u, node(r, c+1), random.randint(1, 20))

    return g



# # Parameters
runs_per_graph = 10
graphs_list = [("small", build_small_graph), 
               ("medium", build_medium_graph), 
               ("large", build_large_graph)]
policies = ["fifo", "highest", "lowest", "random"]
metrics = ["runtime", "pushes", "relabels", "maxflow"]

# ----------------------------
# Collect all runs
# ----------------------------
all_runs = []

for graph_name, builder in graphs_list:
    for policy in policies:
        for run_id in range(runs_per_graph):
            g = builder(policy)
            start = time.time()
            mf = g.max_flow(0, g.n-1)
            end = time.time()
            all_runs.append({
                "graph": graph_name,
                "policy": policy,
                "run": run_id + 1,
                "runtime": end - start,
                "pushes": g.pushes,
                "relabels": g.relabels,
                "maxflow": mf
            })

# Save all runs
with open("results/all_runs.json", "w") as f:
    json.dump(all_runs, f, indent=4)

# ----------------------------
# Compute averages for table
# ----------------------------
df_runs = pd.DataFrame(all_runs)
df_avg = df_runs.groupby(["graph","policy"]).mean().reset_index()

# Save averages
with open("results/avg_results.json", "w") as f:
    df_avg.to_json(f, orient="records", indent=4)

# ----------------------------
# Generate table figure
# ----------------------------
def format_value(val):
    if abs(val) < 0.01 and abs(val) != 0.00:  # choose thresholds
        return f"{val:.2e}"
    else:
        return f"{val:.2f}"

fig, axes = plt.subplots(len(graphs_list), 1, figsize=(10, 4*len(graphs_list)))
if len(graphs_list) == 1:
    axes = [axes]

for ax, (graph_name, _) in zip(axes, graphs_list):
    ax.axis('off')
    ax.set_title(f"Average Metrics - Graph: {graph_name}", fontsize=14, fontweight='bold')
    sub_df = df_avg[df_avg['graph'] == graph_name].set_index('policy')[metrics]
    table_data = [[policy] + [format_value(sub_df.loc[policy, m]) for m in metrics] 
              for policy in sub_df.index]
    col_labels = ["Policy"] + metrics
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

plt.tight_layout()
plt.savefig("results/avg_metrics_table.png", dpi=300)


# Generate scatter plots
metrics = ["runtime", "pushes", "relabels", "maxflow"]
graphs = df_runs['graph'].unique()

sns.set_theme(style="whitegrid", font_scale=1.2)

for metric in metrics:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)  # 3 columns for 3 graph sizes
    fig.suptitle(f"{metric.capitalize()} Across Policies and Graph Sizes", fontsize=16, fontweight='bold')
    
    for ax, graph in zip(axes, graphs):
        df_sub = df_runs[df_runs['graph'] == graph]
        
        sns.scatterplot(
            data=df_sub,
            x="policy",
            y=metric,
            hue="policy",
            s=100,
            alpha=0.7,
            ax=ax
        )
        
        ax.set_title(f"Graph: {graph}")
        ax.set_xlabel("Policy")
        ax.set_ylabel(metric.capitalize())
        
        # Use log scale for metrics with huge differences
        if metric in ["runtime", "pushes", "relabels"]:
            ax.set_yscale("log")
        
        ax.legend().remove()  # Remove individual legends for cleaner figure

    # Add a single legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Policy", bbox_to_anchor=(0.5, 0.05), loc="lower center", ncol=len(labels))
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.show()