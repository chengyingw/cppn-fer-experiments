"""The report figure: what forgetting looks like, then how much each choice forgets. Run: python intuitive_figure.py"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
INK, MUTED, LIGHT, BLUE, AMBER = '#1f1e1c', '#52514e', '#c9c7c1', '#2a78d6', '#eda100'
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 9.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED, 'pdf.fonttype': 42,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
DISCS = [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)]
RES = 64

with open(os.path.join(ROOT, 'kbloop_q3', 'cppn_continual_db.jsonl'), encoding='utf-8') as f:
    q3 = [json.loads(line) for line in f]


def per_net(group, task, key):
    seeds = sorted({r['seed'] for r in q3 if r['group'] == group and r['task'] == task})
    return [st.median(r[key] for r in q3 if r['group'] == group and r['task'] == task and r['seed'] == s and r.get(key) is not None) for s in seeds]


example = np.load(os.path.join(HERE, 'example_images.npz'))
fig = plt.figure(figsize=(12, 4.6))

panels = [('应有的样子', example['skull_target'], None),
          ('不复习', example['skull_final'][0], example['skull_forget'][0]),
          ('均匀复习 5%', example['skull_final'][1], example['skull_forget'][1]),
          ('集中复习 5%', example['skull_final'][2], example['skull_forget'][2])]
side = 0.106
for k, (title, image, forget) in enumerate(panels):
    ax = fig.add_axes([0.02 + k * 0.118, 0.36, side, side * 12 / 4.6])
    ax.imshow(np.clip(image, 0, 1), interpolation='nearest')
    color = BLUE if k == 3 else AMBER
    for i, (cx, cy) in enumerate(DISCS):
        col, row = (cx + 1) / 2 * (RES - 1), (cy + 1) / 2 * (RES - 1)
        ax.add_patch(Circle((col, row), 0.2 / 2 * (RES - 1) + 0.8, fill=False, ec=color, lw=1.2))
        ax.text(col + 8.5, row - 6.5, str(i + 1), color='white', fontsize=7.5, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.15', fc=color, ec='none'))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, fontweight='bold' if k == 3 else 'normal', color=BLUE if k == 3 else INK)
    if forget is not None:
        ax.text(0.5, -0.06, f'遗忘量 {np.mean(forget):.3f}', transform=ax.transAxes, ha='center', va='top', fontsize=9.5,
                color=BLUE if k == 3 else INK, fontweight='bold' if k == 3 else 'normal')
    else:
        ax.text(0.5, -0.06, '圆圈：按 1→4 依次\n学的颜色反转', transform=ax.transAxes, ha='center', va='top', fontsize=8.5, color=MUTED)
fig.text(0.02, 0.95, 'a  一个网络学完 4 个改动后画出的图', fontsize=11, fontweight='bold', va='top')

bars = [('进化\n不复习', 'PBd_r0', '#7a7873'), ('普通\n不复习', 'replay0', LIGHT),
        ('普通\n均匀 5%', 'replay0.05', LIGHT), ('普通\n集中 5%', 'G0_r0.05_focus0.5', BLUE)]
for c, (task, label) in enumerate([('skull', '骷髅'), ('butterfly', '蝴蝶')]):
    ax = fig.add_axes([0.575 + c * 0.215, 0.2, 0.19, 0.6])
    for i, (name, group, color) in enumerate(bars):
        values = per_net(group, task, 'forget')
        height = st.median(values)
        ax.bar(i, height, 0.68, color=color, zorder=1)
        ax.scatter([i] * len(values), values, s=11, color=INK, zorder=3, alpha=0.75)
        ax.text(i, max(max(values), height) + 0.012, f'{height:.2f}', ha='center', fontsize=9, color=BLUE if i == 3 else INK,
                fontweight='bold' if i == 3 else 'normal')
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=8.5)
    ax.set_ylim(0, 0.36)
    ax.tick_params(length=2)
    n_dense = len(per_net('replay0', task, 'forget'))
    ax.set_title(f'{label}（普通 {n_dense} 个，进化 1 个）', fontsize=9.5)
    if c == 0:
        ax.set_ylabel('旧改动的遗忘量（越低越好）')
    else:
        ax.tick_params(labelleft=False)
        ax.spines['left'].set_visible(False)
rise = [st.median(per_net('G0_r0.05_focus0.5', t, 'background')) / st.median(per_net('replay0.05', t, 'background')) for t in ('skull', 'butterfly')]
fig.text(0.535, 0.95, 'b  遗忘量：每个点是一个网络，柱高是中位数', fontsize=11, fontweight='bold', va='top')
fig.text(0.535, 0.03, f'集中复习的代价：从没改动过的区域被带偏得更多，是均匀复习的 {min(rise):.1f}～{max(rise):.1f} 倍。\n'
         '结论在只看没复习到的点、换一台机器、不存正确答案时都成立。', fontsize=9, color=MUTED, va='top')
fig.text(0.02, 0.03, '图 a 是骷髅上一个网络、一种学习顺序的重跑，数值与结果表中这一行一致。\n'
         '实验在 kb-loop 中运行：预测先提交，结果自动判定。只有 3 张简单图像，apple 未画入。', fontsize=9, color=MUTED, va='top')
fig.savefig(os.path.join(HERE, 'intuitive_figure.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'intuitive_figure.pdf'), bbox_inches='tight')
print('rise', [round(r, 2) for r in rise], 'example forget', [round(float(np.mean(f)), 3) for f in example['skull_forget']])
