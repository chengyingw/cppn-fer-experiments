"""One summary figure of the reportable results, one point per network. Run: python summary_figure.py"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
INK, MUTED, GRID, GREY = '#1f1e1c', '#52514e', '#e4e3df', '#a3a19b'
TASKS = [('skull', '骷髅', '#2a78d6'), ('butterfly', '蝴蝶', '#eb6834')]
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
U, F, FS = 'replay0.05', 'G0_r0.05_focus0.5', 'G0_r0.05_focus0.5_snapshot'


def rows(folder):
    with open(os.path.join(ROOT, folder, 'cppn_continual_db.jsonl'), encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def per_net(table, group, task, key):
    """Median of a metric for each network, keyed by its seed."""
    seeds = sorted({r['seed'] for r in table if r['group'] == group and r['task'] == task})
    return {s: st.median(r[key] for r in table if r['group'] == group and r['task'] == task and r['seed'] == s and r.get(key) is not None)
            for s in seeds}


def paired(ax, table, key, ylabel):
    for t, (task, label, color) in enumerate(TASKS):
        a, b = per_net(table, U, task, key), per_net(table, F, task, key)
        x0, x1 = t * 2.6, t * 2.6 + 1
        for s in a:
            ax.plot([x0, x1], [a[s], b[s]], color=color, lw=1, alpha=0.6)
            ax.scatter([x0, x1], [a[s], b[s]], s=16, color=color, zorder=3)
        ax.text((x0 + x1) / 2, -0.17, f'{label}（n = {len(a)}）', transform=ax.get_xaxis_transform(), ha='center', fontsize=9)
    ax.set_xticks([0, 1, 2.6, 3.6], ['均匀', '聚焦'] * 2)
    ax.set_xlim(-0.5, 4.1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=2)


fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
q2, q3, q4 = rows('kbloop_q2'), rows('kbloop_q3'), rows('kbloop_q4')

paired(axes[0, 0], q2, 'forget', '旧改动的遗忘量')
paired(axes[0, 1], q2, 'background', '未改动区域的漂移')

ax = axes[1, 0]
checks = [('整盘\nA100', q2, F, 'forget'), ('整盘\nRTX 4060', q4, F, 'forget'), ('未复习的像素\nRTX 4060', q4, F, 'forget_held'),
          ('不存答案\nRTX 4060', q4, FS, 'forget')]
for i, (name, table, group, key) in enumerate(checks):
    for t, (task, label, color) in enumerate(TASKS):
        a, b = per_net(table, U, task, key), per_net(table, group, task, key)
        ys = [b[s] / a[s] for s in a]
        ax.scatter([i + (t - 0.5) * 0.24] * len(ys), ys, s=16, color=color, zorder=3, label=label if i == 0 else None)
ax.axhline(1, color=MUTED, lw=0.8)
ax.axhline(0.6, color=MUTED, lw=0.8, ls='--')
ax.text(3.45, 1.02, '无差别', ha='right', va='bottom', fontsize=8, color=MUTED)
ax.text(3.45, 0.62, '预测成立线 0.6', ha='right', va='bottom', fontsize=8, color=MUTED)
ax.set_xticks(range(len(checks)), [c[0] for c in checks], fontsize=8)
ax.set_ylim(0, 1.15)
ax.set_ylabel('遗忘量之比（聚焦 / 均匀）')
ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 0.84))
ax.tick_params(length=2)

ax = axes[1, 1]
for t, (task, label, color) in enumerate(TASKS):
    dense = per_net(q3, 'replay0', task, 'forget')
    pb = per_net(q3, 'PBd_r0', task, 'forget')
    ax.scatter([t - 0.12] * len(dense), list(dense.values()), s=18, facecolors='none', edgecolors=color, zorder=3)
    ax.scatter([t + 0.12], list(pb.values()), s=34, marker='D', color=color, zorder=3)
ax.scatter([], [], s=18, facecolors='none', edgecolors=MUTED, label='稠密网络（每个点一个网络）')
ax.scatter([], [], s=34, marker='D', color=MUTED, label='Picbreeder（每张图一个）')
ax.set_xticks(range(2), [f'{x[1]}（稠密 n = {len(per_net(q3, "replay0", x[0], "forget"))}）' for x in TASKS])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0, 0.4)
ax.set_ylabel('旧改动的遗忘量（无复习）')
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.tick_params(length=2)

for ax, letter in zip(axes.flat, 'abcd'):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')
fig.tight_layout(h_pad=2.6, w_pad=2.0)
fig.savefig(os.path.join(HERE, 'summary_figure.png'), dpi=300)
fig.savefig(os.path.join(HERE, 'summary_figure.pdf'))
for name, table in (('q2', q2), ('q3', q3), ('q4', q4)):
    print(name, {t: {g: len(per_net(table, g, t, 'forget')) for g in (U, F, FS, 'replay0', 'PBd_r0')} for t, _, _ in TASKS})
