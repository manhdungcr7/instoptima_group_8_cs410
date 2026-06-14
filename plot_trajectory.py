import pickle
import re
import math
import numpy as np
import matplotlib.pyplot as plt

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

# Load final pkl files which contain __history_fronts__ (one entry per generation)
datasets = {
    "Laptop14": "nsga2_result_v4.pkl",
    "SST2":     "nsga2_result_tc.pkl",
    "SNLI":     "nsga2_result_nli.pkl",
}

# Figure 2: 3 rows (objectives) x 3 columns (datasets)
# Lower values = better performance (as stated in paper)
fig, axes = plt.subplots(3, 3, figsize=(14, 9))
plt.style.use('default')

obj_labels = ["Performance", "Length", "Perplexity"]
dataset_names = ["Laptop14", "SST2", "SNLI"]

# Paper shows: main generations in black, 10 additional generations in red.
# We split our 10 generations into first half (black) and second half (red)
# to reproduce the visual style. If more generations were run, set EXTRA_START
# to the actual boundary (e.g., 11 for generations 11-20 shown in red).
EXTRA_START = 6  # generations 1-5 in black, 6-10 in red (adjust if you ran more)

for col, (ds_name, pkl_path) in enumerate(datasets.items()):
    try:
        with open(pkl_path, "rb") as f:
            evo_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        continue

    history_fronts = evo_data.get("__history_fronts__", [])
    if not history_fronts:
        print(f"No __history_fronts__ in {pkl_path}, skipping {ds_name}")
        continue

    # Collect objective values per generation from all individuals in all fronts
    r_by_gen, l_by_gen, p_by_gen = [], [], []
    for gen_fronts in history_fronts:
        r_vals, l_vals, p_vals = [], [], []
        for front in gen_fronts:
            for ind in front:
                l, p, r = parse_objectives(ind)
                if (l is not None and not math.isinf(l)
                        and not math.isinf(p) and not math.isinf(r)):
                    l_vals.append(l)
                    p_vals.append(p)
                    r_vals.append(r)
        r_by_gen.append(r_vals)
        l_by_gen.append(l_vals)
        p_by_gen.append(p_vals)

    n_gens = len(history_fronts)
    gens = list(range(1, n_gens + 1))

    for row, obj_data in enumerate([r_by_gen, l_by_gen, p_by_gen]):
        means = [np.mean(d) if d else float('nan') for d in obj_data]
        stds  = [np.std(d)  if d else 0.0          for d in obj_data]

        ax = axes[row, col]

        # Split into main (black) and additional (red) generations
        main_gens  = [g for g in gens if g < EXTRA_START]
        extra_gens = [g for g in gens if g >= EXTRA_START]

        main_means  = [means[g-1] for g in main_gens]
        main_stds   = [stds[g-1]  for g in main_gens]
        extra_means = [means[g-1] for g in extra_gens]
        extra_stds  = [stds[g-1]  for g in extra_gens]

        cap = 3
        ms  = 4
        lw  = 1.5

        if main_gens:
            ax.errorbar(main_gens, main_means, yerr=main_stds,
                        fmt='-s', color='black', ecolor='gray',
                        capsize=cap, elinewidth=1, markersize=ms, linewidth=lw)
        if extra_gens:
            # Connect last main point to first extra point for continuity
            if main_gens:
                ax.plot([main_gens[-1], extra_gens[0]],
                        [main_means[-1], extra_means[0]],
                        color='red', linewidth=lw, linestyle='-')
            ax.errorbar(extra_gens, extra_means, yerr=extra_stds,
                        fmt='-s', color='red', ecolor='salmon',
                        capsize=cap, elinewidth=1, markersize=ms, linewidth=lw)

        ax.grid(True, linestyle='-', linewidth=0.5)
        even_ticks = [g for g in gens if g % 2 == 0]
        ax.set_xticks(even_ticks if even_ticks else gens)

        if col == 0:
            ax.set_ylabel(obj_labels[row], fontsize=9)
        if row == 0:
            ax.set_title(ds_name, fontsize=11, fontweight='bold')
        if row == 2:
            ax.set_xlabel("Generation", fontsize=9)

plt.suptitle("Figure 2: Trajectory Plots of Objective Values Across Datasets",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("paper_fig2_trajectory.png", dpi=300, bbox_inches='tight')
print("Saved: paper_fig2_trajectory.png")
