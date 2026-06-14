"""
Tái tạo bảng so sánh kết quả chính (Table trong paper):
  No-Instruction vs Random-Instruction vs InstOptima
  trên 3 datasets: Laptop14, SST2, SNLI
"""
import pickle, re, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '.')

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_obj(ind):
    """Extract (length, perplexity, reciprocal_metric) from an Individual."""
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

NUM_METRICS = 4   # accuracy, precision, recall, f1

def recip_to_avg(r):
    """Convert reciprocal_metric back to avg_metric (sum/4)."""
    return (1.0 / r) / NUM_METRICS if r and not math.isinf(r) else 0.0

# ── load baseline data ────────────────────────────────────────────────────────

noi_raw = pickle.load(open('noinstruct_all.pkl', 'rb'))   # {ds: dict}
ran_raw = pickle.load(open('raninstruct_all.pkl', 'rb'))   # {ds: [dict, ...]}

DATASETS = ['Laptop14', 'SST2', 'SNLI']
NSGA2_FILES = {
    'Laptop14': 'nsga2_result_v4.pkl',
    'SST2':     'nsga2_result_tc.pkl',
    'SNLI':     'nsga2_result_nli.pkl',
}

# ── collect numbers ───────────────────────────────────────────────────────────

rows = {}   # ds -> {method -> {metric_key -> value}}

for ds in DATASETS:
    rows[ds] = {}

    # ── No-Instruction ──
    noi = noi_raw[ds]
    noi_m = noi.get('metrics', {})
    rows[ds]['NoInstruct'] = {
        'accuracy':  noi_m.get('accuracy', 0),
        'precision': noi_m.get('precision', 0),
        'recall':    noi_m.get('recall', 0),
        'f1':        noi_m.get('f1', 0),
        'avg':       noi.get('avg_metric', 0),
        'length':    noi['objectives'][0],
        'perplexity':noi['objectives'][1],
    }

    # ── Random-Instruction (mean ± std of 5 runs) ──
    ran_items = ran_raw[ds]
    ran_acc  = [x['metrics']['accuracy']  for x in ran_items]
    ran_pre  = [x['metrics']['precision'] for x in ran_items]
    ran_rec  = [x['metrics']['recall']    for x in ran_items]
    ran_f1   = [x['metrics']['f1']        for x in ran_items]
    ran_avg  = [x['avg_metric']           for x in ran_items]
    ran_len  = [x['objectives'][0]        for x in ran_items]
    ran_ppl  = [x['objectives'][1]        for x in ran_items]
    rows[ds]['RanInstruct'] = {
        'accuracy':  (np.mean(ran_acc),  np.std(ran_acc)),
        'precision': (np.mean(ran_pre),  np.std(ran_pre)),
        'recall':    (np.mean(ran_rec),  np.std(ran_rec)),
        'f1':        (np.mean(ran_f1),   np.std(ran_f1)),
        'avg':       (np.mean(ran_avg),  np.std(ran_avg)),
        'length':    (np.mean(ran_len),  np.std(ran_len)),
        'perplexity':(np.mean(ran_ppl),  np.std(ran_ppl)),
    }

    # ── InstOptima (statistics across all front-0 individuals) ──
    evo = pickle.load(open(NSGA2_FILES[ds], 'rb'))
    front0 = evo['fronts'][0]

    all_accs, all_pres, all_recs, all_f1s = [], [], [], []
    all_avgs, all_lens, all_ppls = [], [], []

    for ind in front0:
        try:
            m = ind.objectives.metric
            acc  = float(m['accuracy'])
            pre  = float(m['precision'])
            rec  = float(m['recall'])
            f1   = float(m['f1'])
            avg  = (acc + pre + rec + f1) / 4
            leng = float(ind.objectives.prompt_length)
            ppl  = float(ind.objectives.perplexity)
        except Exception:
            l, p, r = parse_obj(ind)
            if r is None or math.isinf(r):
                continue
            avg  = recip_to_avg(r)
            leng = l
            ppl  = p
            acc = pre = rec = f1 = None

        all_avgs.append(avg)
        all_lens.append(leng)
        all_ppls.append(ppl)
        if acc is not None:
            all_accs.append(acc)
            all_pres.append(pre)
            all_recs.append(rec)
            all_f1s.append(f1)

    def mean_std(lst):
        return (np.mean(lst), np.std(lst)) if lst else (0.0, 0.0)

    rows[ds]['InstOptima'] = {
        'accuracy':  mean_std(all_accs),
        'precision': mean_std(all_pres),
        'recall':    mean_std(all_recs),
        'f1':        mean_std(all_f1s),
        'avg':       mean_std(all_avgs),
        'length':    mean_std(all_lens),
        'perplexity':mean_std(all_ppls),
    }

# ── print table to console ────────────────────────────────────────────────────

def fmt(v):
    if isinstance(v, tuple):
        return f"{v[0]:.4f} ±{v[1]:.4f}"
    return f"{v:.4f}"

print("\n" + "="*90)
print("COMPARISON TABLE: NoInstruct vs RanInstruct vs InstOptima")
print("="*90)
print(f"{'Dataset':<12} {'Method':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg':>10}")
print("-"*90)
for ds in DATASETS:
    for method in ['NoInstruct', 'RanInstruct']:
        r = rows[ds][method]
        print(f"{ds:<12} {method:<14} {fmt(r['accuracy']):>15} {fmt(r['precision']):>15} "
              f"{fmt(r['recall']):>15} {fmt(r['f1']):>15} {fmt(r['avg']):>15}")
    imo = rows[ds]['InstOptima']
    print(f"{ds:<12} {'InstOptima':<14} {fmt(imo['accuracy']):>15} {fmt(imo['precision']):>15} "
          f"{fmt(imo['recall']):>15} {fmt(imo['f1']):>15} {fmt(imo['avg']):>15}")
    print()

# ── Figure A: Bar chart — 2 rows ─────────────────────────────────────────────
# Row 1: Avg Metric (all 3 methods, NoInstruct included)
# Row 2: Length & Perplexity (RanInstruct vs InstOptima only — NoInstruct
#         has length=0 and perplexity=2.7 from empty prompt which is a
#         different scale and makes the chart unreadable)

METHODS_ALL  = ['NoInstruct', 'RanInstruct', 'InstOptima']
METHODS_INST = ['RanInstruct', 'InstOptima']
COLORS_ALL   = ['#4878CF', '#6ACC65', '#D65F5F']
COLORS_INST  = ['#6ACC65', '#D65F5F']

x = np.arange(len(DATASETS))
w = 0.25

fig = plt.figure(figsize=(15, 9))
fig.suptitle("Performance Comparison: No-Instruction vs Random-Instruction vs InstOptima",
             fontsize=13, fontweight='bold')

# ── Row 1: Avg Metric (all 3 methods) ────────────────────────────────────────
ax_avg = fig.add_subplot(2, 1, 1)
for j, (method, color) in enumerate(zip(METHODS_ALL, COLORS_ALL)):
    vals = [rows[ds][method].get('avg', (0,0)) for ds in DATASETS]
    means = [v[0] if isinstance(v, tuple) else float(v) for v in vals]
    errs  = [v[1] if isinstance(v, tuple) else 0.0      for v in vals]
    bars = ax_avg.bar(x + (j - 1)*w, means, w, label=method, color=color,
                      yerr=errs, capsize=4, error_kw={'elinewidth': 1.2})
    for bar, m in zip(bars, means):
        ax_avg.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{m:.4f}', ha='center', va='bottom', fontsize=7.5)

# zoom y-axis to show differences clearly
all_avgs = [rows[ds][m].get('avg', (0,0)) for ds in DATASETS for m in METHODS_ALL]
all_means = [v[0] if isinstance(v, tuple) else float(v) for v in all_avgs]
margin = (max(all_means) - min(all_means)) * 2
ax_avg.set_ylim(min(all_means) - margin, max(all_means) + margin * 2)
ax_avg.set_title("Avg Metric ↑ (all 3 methods)", fontsize=10, fontweight='bold')
ax_avg.set_xticks(x); ax_avg.set_xticklabels(DATASETS, fontsize=10)
ax_avg.legend(fontsize=9); ax_avg.grid(axis='y', linestyle='--', alpha=0.5)
ax_avg.tick_params(labelsize=9)

# ── Row 2: Length & Perplexity (RanInstruct vs InstOptima only) ──────────────
ax_len = fig.add_subplot(2, 2, 3)
ax_ppl = fig.add_subplot(2, 2, 4)

for ax, mkey, title in [(ax_len, 'length', 'Length ↓  (instruction chars)'),
                         (ax_ppl, 'perplexity', 'Perplexity ↓  (lower = more natural)')]:
    for j, (method, color) in enumerate(zip(METHODS_INST, COLORS_INST)):
        vals = [rows[ds][method].get(mkey, (0,0)) for ds in DATASETS]
        means = [v[0] if isinstance(v, tuple) else float(v) for v in vals]
        errs  = [v[1] if isinstance(v, tuple) else 0.0      for v in vals]
        bars = ax.bar(x + (j - 0.5)*w, means, w, label=method, color=color,
                      yerr=errs, capsize=4, error_kw={'elinewidth': 1.2})
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.005,
                    f'{m:.0f}' if mkey == 'length' else f'{m:.4f}',
                    ha='center', va='bottom', fontsize=7.5)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(DATASETS, fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=9)
    # tight y-range
    all_v = [rows[ds][m].get(mkey, (0,0)) for ds in DATASETS for m in METHODS_INST]
    all_m = [v[0] if isinstance(v, tuple) else float(v) for v in all_v]
    margin = (max(all_m) - min(all_m)) * 0.5
    ax.set_ylim(max(0, min(all_m) - margin), max(all_m) + margin * 2)

    note = ("Note: NoInstruct excluded (length=0, no prompt)\n"
            if mkey == 'length' else
            "Note: NoInstruct excluded (perplexity≈2.7, empty prompt)\n")
    ax.text(0.5, -0.18, note, transform=ax.transAxes,
            ha='center', fontsize=7, color='gray', style='italic')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('paper_table1_bar_chart.png', dpi=200, bbox_inches='tight')
print("Saved: paper_table1_bar_chart.png")

# ── Figure B: Rendered HTML-style table as PNG ────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.axis('off')

col_labels = ['Dataset', 'Method', 'Accuracy', 'Precision', 'Recall', 'F1', 'Avg Metric',
               'Length', 'Perplexity']
table_data = []
row_colors = []
for ds in DATASETS:
    for method, bg in [('NoInstruct','#f0f4ff'), ('RanInstruct','#f0fff0'), ('InstOptima','#fff0f0')]:
        r = rows[ds][method]
        if False:
            pass
        else:
            def fmtc(v):
                if isinstance(v, tuple): return f"{v[0]:.4f}±{v[1]:.4f}"
                return f"{v:.4f}"
            row = [ds, method,
                   fmtc(r['accuracy']), fmtc(r['precision']),
                   fmtc(r['recall']),   fmtc(r['f1']),
                   fmtc(r['avg']),
                   fmtc(r['length']),   fmtc(r['perplexity'])]
        table_data.append(row)
        row_colors.append([bg]*len(col_labels))

tbl = ax2.table(cellText=table_data, colLabels=col_labels,
                cellColours=row_colors,
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.8)

# Bold header
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#333333')
        cell.set_text_props(color='white', fontweight='bold')

plt.title("Table 1: Comparison of Methods Across Datasets", fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('paper_table1_rendered.png', dpi=200, bbox_inches='tight')
print("Saved: paper_table1_rendered.png")
