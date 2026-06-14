import pickle
import matplotlib.pyplot as plt
import math
import re

def parse_objectives(ind):
    try:
        l = float(ind.objectives[0])
        p = float(ind.objectives[1])
        r = float(ind.objectives[2])
        return l, p, r
    except Exception:
        obj_str = str(getattr(ind, 'objectives', ''))
        l_m = re.search(r'instruction_length=([0-9.]+)', obj_str)
        p_m = re.search(r'perplexity=([0-9.]+)', obj_str)
        r_m = re.search(r'reciprocal_metric=([0-9.]+)', obj_str)
        if l_m and p_m and r_m:
            return float(l_m.group(1)), float(p_m.group(1)), float(r_m.group(1))
    return None, None, None

# Load data from all 3 datasets (matching paper: Laptop14, SST2, SNLI)
datasets = {
    "Laptop14": "nsga2_result_v4.pkl",
    "SST2":     "nsga2_result_tc.pkl",
    "SNLI":     "nsga2_result_nli.pkl",
}

dataset_fronts = {}
for name, pkl_path in datasets.items():
    try:
        with open(pkl_path, "rb") as f:
            evo_data = pickle.load(f)
        dataset_fronts[name] = evo_data.get("fronts", [])
        print(f"Loaded {pkl_path}: {len(dataset_fronts[name])} fronts")
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        dataset_fronts[name] = []

# Figure 3: 3x3 grid
# Rows = 3 pairs of objectives (as in paper: Perplexity vs Performance, Length vs Performance, Length vs Perplexity)
# Columns = 3 datasets (Laptop14, SST2, SNLI)
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
plt.style.use('default')

# Colors for Pareto front ranks: front 1 (best) = red, front 2 = green, front 3 = blue
colors  = ['red', 'green', 'blue']
markers = ['o', 'd', 's']
sizes   = [20, 15, 15]

dataset_names = ["Laptop14", "SST2", "SNLI"]

for col, ds_name in enumerate(dataset_names):
    fronts_data = dataset_fronts.get(ds_name, [])
    num_fronts = min(3, len(fronts_data))

    for i in range(num_fronts):
        l_list, p_list, r_list = [], [], []
        for ind in fronts_data[i]:
            l, p, r = parse_objectives(ind)
            if (l is not None and not math.isinf(l)
                    and not math.isinf(p) and not math.isinf(r)):
                l_list.append(l)
                p_list.append(p)
                r_list.append(r)

        if not l_list:
            continue

        label = f'Front {i+1}' if col == 0 else None
        # Row 0: Perplexity (y) vs Reciprocal-metric (x)
        axes[0, col].scatter(r_list, p_list, c=colors[i], marker=markers[i],
                             s=sizes[i], label=label)
        # Row 1: Length (y) vs Reciprocal-metric (x)
        axes[1, col].scatter(r_list, l_list, c=colors[i], marker=markers[i],
                             s=sizes[i])
        # Row 2: Length (y) vs Perplexity (x)
        axes[2, col].scatter(p_list, l_list, c=colors[i], marker=markers[i],
                             s=sizes[i])

    # Column titles = dataset names
    axes[0, col].set_title(ds_name, fontsize=11, fontweight='bold')

# Row y-labels
for row, ylabel in enumerate(["Perplexity", "Length", "Length"]):
    axes[row, 0].set_ylabel(ylabel, fontsize=9)

# Row x-labels
for col in range(3):
    axes[0, col].set_xlabel("Performance", fontsize=8)
    axes[1, col].set_xlabel("Performance", fontsize=8)
    axes[2, col].set_xlabel("Perplexity",  fontsize=8)

for ax in axes.flat:
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

axes[0, 0].legend(fontsize=8, loc='upper right')

plt.suptitle("Figure 3: 2D-Pareto Fronts on Three Datasets", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("paper_fig3_pareto.png", dpi=300, bbox_inches='tight')
print("Saved: paper_fig3_pareto.png")
