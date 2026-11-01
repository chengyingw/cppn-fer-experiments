"""Replay curves from the sequence test, on both machines."""
import json
import os
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = [('skull', '骷髅'), ('butterfly', '蝴蝶'), ('apple', '苹果')]
DOSES = [0.0, 0.05, 0.2, 1.0]
MACHINES = [('local', 'RTX 4060', '#2a78d6', 'o'), ('a100', 'A100', '#eb6834', 's')]
SURFACE, MUTED, GRID = '#fcfcfb', '#52514e', '#e4e3df'
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': GRID,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def medians(rows, task, key):
    return [statistics.median(r[key] for r in rows if r['task'] == task and r['replay_fraction'] == d and r[key] is not None)
            for d in DOSES]


data = {m: [json.loads(l) for l in open(os.path.join(HERE, m, 'cppn_continual_db.jsonl'), encoding='utf-8')] for m, *_ in MACHINES}
fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.8), sharex=True)
for c, (task, label) in enumerate(TASKS):
    for r, key in enumerate(['forget', 'background']):
        ax = axes[r, c]
        for m, name, color, marker in MACHINES:
            ax.plot(range(4), medians(data[m], task, key), color=color, marker=marker, lw=2, ms=7, label=name)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_xticks(range(4), ['0%', '5%', '20%', '100%'])
    axes[0, c].set_title(label, loc='left', fontsize=11)
    axes[0, c].set_ylim(0, 0.26)
    axes[1, c].set_yscale('log')
    axes[1, c].set_ylim(0.001, 2)
    axes[1, c].set_xlabel('回放比例')
axes[0, 0].set_ylabel('旧改动的遗忘量（中位数）')
axes[1, 0].set_ylabel('从未改动区域的变动（中位数）')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, frameon=False, fontsize=9, loc='lower center')
fig.suptitle('少量回放几乎压不住旧改动的遗忘，却能大幅压住背景漂移（背景是事后才看的指标）', x=0.01, ha='left', fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig(os.path.join(HERE, 'report_replay_curve.png'), dpi=160)
