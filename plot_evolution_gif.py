"""
Tạo GIF animation cho quá trình tiến hóa Pareto front qua các generation.
Hiển thị 2D scatter (Performance vs Perplexity) cho từng dataset.
"""
import pickle, re, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, '.')

# ── helper ────────────────────────────────────────────────────────────────────

def parse_obj(ind):
    try:
        return float(ind.objectives[0]), float(ind.objectives[1]), float(ind.objectives[2])
    except Exception:
        s = str(getattr(ind, 'objectives', ''))
        lm = re.search(r'instruction_length=([0-9.]+)', s)
        pm = re.search(r'perplexity=([0-9.]+)', s)
        rm = re.search(r'reciprocal_metric=([0-9.]+)', s)
        if lm and pm and rm:
            return float(lm.group(1)), float(pm.group(1)), float(rm.group(1))
    return None, None, None

# ── load data ─────────────────────────────────────────────────────────────────

DATASETS = {
    'Laptop14': 'nsga2_result_v4.pkl',
    'SST2':     'nsga2_result_tc.pkl',
    'SNLI':     'nsga2_result_nli.pkl',
}

NUM_METRICS = 4

all_history = {}
for ds, pkl in DATASETS.items():
    evo = pickle.load(open(pkl, 'rb'))
    hf = evo.get('__history_fronts__', [])
    all_history[ds] = hf
    print(f"{ds}: {len(hf)} generations")

n_gens = min(len(v) for v in all_history.values())
ds_names = list(DATASETS.keys())

# ── compute axis limits (global, so axes don't jump) ─────────────────────────

def collect_all(ds):
    perfs, ppls, lens = [], [], []
    for gen_fronts in all_history[ds]:
        for front in gen_fronts:
            for ind in front:
                l, p, r = parse_obj(ind)
                if l and p and r and not math.isinf(r):
                    perfs.append(1.0/r/NUM_METRICS)
                    ppls.append(p)
                    lens.append(l)
    return perfs, ppls, lens

axis_lims = {}
for ds in ds_names:
    perfs, ppls, lens = collect_all(ds)
    axis_lims[ds] = {
        'perf': (min(perfs)*0.998, max(perfs)*1.002),
        'ppl':  (min(ppls)*0.998,  max(ppls)*1.002),
        'len':  (min(lens)*0.98,   max(lens)*1.02),
    }

# ── build frames ──────────────────────────────────────────────────────────────
# Layout: 2 rows × 3 cols
#   Row 0: Performance vs Perplexity (all 3 datasets)
#   Row 1: Performance vs Length     (all 3 datasets)

FRONT_COLORS  = ['#E74C3C', '#27AE60', '#2980B9', '#8E44AD', '#F39C12']
FRONT_MARKERS = ['o', 'D', 's', '^', 'v']
FRONT_SIZES   = [25, 18, 16, 14, 12]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

gen_title = fig.suptitle('', fontsize=13, fontweight='bold')

# ── Draw fixed grid once, then only update scatter artists ───────────────────

def setup_axes():
    for row in range(2):
        for col, ds in enumerate(ds_names):
            ax = axes[row, col]
            lims = axis_lims[ds]
            ax.set_xlim(*lims['perf'])
            if row == 0:
                ax.set_ylim(*lims['ppl'])
                ax.set_ylabel('Perplexity', fontsize=8)
            else:
                ax.set_ylim(*lims['len'])
                ax.set_ylabel('Length', fontsize=8)
            ax.set_xlabel('Performance', fontsize=8)
            if row == 0:
                ax.set_title(ds, fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.tick_params(labelsize=7)
            # fixed tick values so axis never rescales
            ax.set_autoscale_on(False)

setup_axes()

# Pre-create scatter collections (one per front per cell) and update data each frame
MAX_FRONTS = 5
scatter_map = {}   # (row, col, fi) -> PathCollection

for row in range(2):
    for col, ds in enumerate(ds_names):
        ax = axes[row, col]
        for fi in range(MAX_FRONTS):
            label = f'Front {fi+1}' if (col == 0 and row == 0) else None
            sc = ax.scatter([], [],
                            c=FRONT_COLORS[fi % len(FRONT_COLORS)],
                            marker=FRONT_MARKERS[fi % len(FRONT_MARKERS)],
                            s=FRONT_SIZES[fi % len(FRONT_SIZES)],
                            label=label, alpha=0.85, zorder=3)
            scatter_map[(row, col, fi)] = sc

axes[0, 0].legend(fontsize=7, loc='upper left')

# progress bar line (bottom of figure)
prog_ax = fig.add_axes([0.1, 0.01, 0.8, 0.012])
prog_ax.set_xlim(0, n_gens); prog_ax.set_ylim(0, 1)
prog_ax.axis('off')
prog_bg  = prog_ax.barh(0, n_gens, 1, color='#e0e0e0', align='edge')[0]
prog_bar = prog_ax.barh(0, 0,      1, color='#D65F5F', align='edge')[0]

def update(frame_gen):
    gen = frame_gen % n_gens
    gen_title.set_text(
        f'Pareto Front Evolution  —  Generation {gen + 1} / {n_gens}  '
        f'{"▶" * (gen + 1)}{"·" * (n_gens - gen - 1)}'
    )
    prog_bar.set_width(gen + 1)

    for row in range(2):
        for col, ds in enumerate(ds_names):
            gen_fronts = all_history[ds][gen]
            for fi in range(MAX_FRONTS):
                sc = scatter_map[(row, col, fi)]
                if fi < len(gen_fronts):
                    xs, ys = [], []
                    for ind in gen_fronts[fi]:
                        l, p, r = parse_obj(ind)
                        if l and p and r and not math.isinf(r):
                            avg = 1.0 / r / NUM_METRICS
                            xs.append(avg)
                            ys.append(p if row == 0 else l)
                    if xs:
                        sc.set_offsets(np.c_[xs, ys])
                    else:
                        sc.set_offsets(np.empty((0, 2)))
                else:
                    sc.set_offsets(np.empty((0, 2)))

    return list(scatter_map.values()) + [prog_bar]

ani = animation.FuncAnimation(
    fig, update, frames=n_gens,
    interval=800, blit=True, repeat=True
)

ani.save('paper_evolution.gif', writer='pillow', fps=1.4, dpi=120)
print("Saved: paper_evolution.gif")
plt.close()
