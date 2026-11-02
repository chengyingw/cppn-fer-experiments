"""The report's storyline in four panels, one point per network. Run: python storyline_figure.py"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
INK, MUTED = '#1f1e1c', '#52514e'
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


def paired(ax, table, key):
    for t, (task, label, color) in enumerate(TASKS):
        a, b = per_net(table, U, task, key), per_net(table, F, task, key)
        x0, x1 = t * 2.6, t * 2.6 + 1
        for s in a:
            ax.plot([x0, x1], [a[s], b[s]], color=color, lw=1, alpha=0.6)
            ax.scatter([x0, x1], [a[s], b[s]], s=14, color=color, zorder=3)
        ax.text((x0 + x1) / 2, -0.2, f'{label}（n = {len(a)}）', transform=ax.get_xaxis_transform(), ha='center')
    ax.set_xticks([0, 1, 2.6, 3.6], ['均匀', '集中'] * 2)
    ax.set_xlim(-0.5, 4.1)
    ax.set_ylim(0, 0.25)


def header(ax, step, title, result):
    ax.set_title(f'{step}  {title}', loc='left', fontsize=10.5, fontweight='bold', pad=22)
    ax.text(0, 1.04, result, transform=ax.transAxes, fontsize=9, color=MUTED)


q2, q3, q4 = rows('kbloop_q2'), rows('kbloop_q3'), rows('kbloop_q4')
fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), gridspec_kw={'wspace': 0.42})

ax = axes[0]
ratios = []
for t, (task, label, color) in enumerate(TASKS):
    dense, pb = per_net(q3, 'replay0', task, 'forget'), per_net(q3, 'PBd_r0', task, 'forget')
    ratios.append(st.median(r['forget'] for r in q3 if r['group'] == 'PBd_r0' and r['task'] == task and r['forget'] is not None) /
                  st.median(r['forget'] for r in q3 if r['group'] == 'replay0' and r['task'] == task and r['forget'] is not None))
    ax.scatter([t - 0.13] * len(dense), list(dense.values()), s=16, facecolors='none', edgecolors=color, zorder=3)
    ax.scatter([t + 0.13], list(pb.values()), s=30, marker='D', color=color, zorder=3)
ax.scatter([], [], s=16, facecolors='none', edgecolors=MUTED, label='普通网络')
ax.scatter([], [], s=30, marker='D', color=MUTED, label='进化网络（Picbreeder）')
ax.set_xticks(range(2), [f"{x[1]}\n（普通 n = {len(per_net(q3, 'replay0', x[0], 'forget'))}）" for x in TASKS])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0, 0.4)
ax.set_ylabel('旧改动的遗忘量')
ax.legend(frameon=False, fontsize=8, loc='upper left')
header(ax, '1', '网络本身', f'进化网络遗忘更多（{min(ratios):.2f}～{max(ratios):.2f} 倍）')

ax = axes[1]
paired(ax, q2, 'forget')
ax.set_ylabel('旧改动的遗忘量')
fall = [st.median(r['forget'] for r in q2 if r['group'] == F and r['task'] == t and r['forget'] is not None) /
        st.median(r['forget'] for r in q2 if r['group'] == U and r['task'] == t and r['forget'] is not None) for t, _, _ in TASKS]
header(ax, '2', '复习放在哪里', f'集中复习，遗忘降到 {min(fall):.2f}～{max(fall):.2f}')

ax = axes[2]
worst = 0.0
checks = [('没复习到\n的点', q4, F, 'forget_held'), ('换一台\n机器', q4, F, 'forget'), ('不存\n正确答案', q4, FS, 'forget')]
for i, (name, table, group, key) in enumerate(checks):
    for t, (task, label, color) in enumerate(TASKS):
        a, b = per_net(table, U, task, key), per_net(table, group, task, key)
        ys = [b[s] / a[s] for s in a]
        worst = max(worst, *ys)
        ax.scatter([i + (t - 0.5) * 0.24] * len(ys), ys, s=14, color=color, zorder=3)
for task, label, color in TASKS:
    ax.scatter([], [], s=14, color=color, label=label)
ax.legend(frameon=False, fontsize=8, loc='center right', bbox_to_anchor=(1.0, 0.68))
ax.axhline(1, color=MUTED, lw=0.8)
ax.text(2.45, 1.02, '没有效果', ha='right', va='bottom', fontsize=8, color=MUTED)
ax.set_xticks(range(len(checks)), [c[0] for c in checks], fontsize=8.5)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, 1.15)
ax.set_ylabel('遗忘量之比（集中 / 均匀）')
header(ax, '3', '不是假象', f'每个网络的比值都不超过 {worst:.2f}')

ax = axes[3]
paired(ax, q2, 'background')
ax.set_ylabel('未改动区域的漂移')
rise = [st.median(r['background'] for r in q2 if r['group'] == F and r['task'] == t) / st.median(r['background'] for r in q2 if r['group'] == U and r['task'] == t)
        for t, _, _ in TASKS]
header(ax, '4', '代价', f'漂移增加到 {min(rise):.1f}～{max(rise):.1f} 倍')

for ax in axes:
    ax.tick_params(length=2)
fig.text(0.01, -0.2,'稠密 CPPN 依次学 4 个局部改动，复习预算 5%。每个点是一个网络，连线连接同一个网络；n 为网络数，进化网络每张图只有一个。'
         '第 1 格无复习（RTX 4060），第 2、4 格为 A100，第 3 格为 RTX 4060。\n'
         '目前只有 3 张简单图像，apple 受输出层截断影响未画入。', fontsize=8.5, color=MUTED)
fig.savefig(os.path.join(HERE, 'storyline_figure.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'storyline_figure.pdf'), bbox_inches='tight')
print('ratios', [round(r, 2) for r in ratios], 'fall', [round(f, 2) for f in fall], 'rise', [round(r, 2) for r in rise], 'worst check', round(worst, 3))
