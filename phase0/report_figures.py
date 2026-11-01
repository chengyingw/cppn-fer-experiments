"""Report figures drawn from kb/experiments_db.jsonl and kb/continual_db.jsonl."""
import json
import os
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from train_sgd import HERE

KB = os.path.join(HERE, '..', 'kb')
OUT = os.path.join(HERE, 'out')
TASKS = ['skull', 'butterfly', 'apple']
TASK_LABEL = {'skull': '骷髅', 'butterfly': '蝴蝶', 'apple': '苹果'}
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COLOR = {'PB': '#2a78d6', 'PBd': '#2a78d6', 'PBa': '#eda100', 'G0': '#eb6834', 'G0e': '#1baf7a',
         'G1a': '#1baf7a', 'G1b': '#2a78d6', 'G4': '#eda100'}
MARK = {'G0': 'o', 'G1a': '^', 'G1b': 's', 'G4': 'D', 'PBd': 'o', 'PBa': 's', 'G0e': '^'}
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.edgecolor': GRID, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def load(name):
    with open(os.path.join(KB, name), encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def fig_wiring(rows):
    pb = {r['task']: r for r in rows if r['group'] == 'PB'}
    vals = [100 * pb[t]['forced_frac'] for t in TASKS]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    y = np.arange(3)[::-1]
    ax.barh(y, vals, height=0.42, color=COLOR['PB'])
    for yi, v in zip(y, vals):
        ax.text(v + 1.5, yi, f'{v:.0f}%', va='center', color=INK)
    ax.set_yticks(y, [TASK_LABEL[t] for t in TASKS])
    ax.set_xlim(0, 100)
    ax.set_xlabel('不同特征图中，对称性由接线强制的比例（与权重无关）')
    ax.tick_params(length=0)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title('Picbreeder 的对称性有相当一部分来自接线', loc='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'report_wiring_forced.png'), dpi=160)
    plt.close(fig)


def fig_fit_level(rows):
    tasks = ['butterfly', 'apple']
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, t in zip(axes, tasks):
        sel = [r for r in rows if r['task'] == t and r.get('fit')]
        pb = next(r for r in sel if r['group'] == 'PB')['M3a']
        ax.axhline(pb, color=COLOR['PB'], lw=2)
        ax.text(2.2e-5, pb + 0.012, f'Picbreeder {pb:.2f}', color=INK, fontsize=9)
        for g, label in [('G0', '稠密，训练到底'), ('G0e', '稠密，提前停止')]:
            pts = [r for r in sel if r['group'] == g]
            ax.scatter([r['mse'] for r in pts], [r['M3a'] for r in pts], s=46, marker=MARK[g], color=COLOR[g],
                       edgecolor=SURFACE, linewidth=1, label=label, zorder=3)
        ax.set_xscale('log')
        ax.set_xlim(1.8e-5, 0.02)
        ax.set_ylim(0, 0.52)
        ax.set_xlabel('拟合 MSE（对数坐标）')
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_title(TASK_LABEL[t], loc='left', fontsize=11)
    axes[0].set_ylabel('扰动不对称占比 M3a')
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')
    fig.suptitle('拟合水平的影响因任务而异：苹果的差距在粗拟合时消失，蝴蝶的还在', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, 'report_fit_level.png'), dpi=160)
    plt.close(fig)


def fig_sparse_fit(rows):
    groups = [('G0', '稠密'), ('G1a', '原接线，\n只训 NEAT 连接'), ('G1b', '原接线，\n全部现存边'), ('G4', '剪枝')]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True)
    for ax, t in zip(axes, TASKS):
        for i, (g, _) in enumerate(groups):
            mse = sorted(r['mse'] for r in rows if r['task'] == t and r['group'] == g)
            if not mse:
                ax.text(i, 0.25, '未跑', ha='center', fontsize=9, color=MUTED)
                continue
            offs = np.linspace(-0.14, 0.14, len(mse)) if len(mse) > 1 else np.zeros(1)
            ax.scatter(i + offs, mse, s=34, marker=MARK[g], color=COLOR[g], edgecolor=SURFACE, linewidth=0.8, zorder=3)
            ax.text(i, 0.25, f'{sum(m < 0.01 for m in mse)}/{len(mse)}', ha='center', fontsize=9, color=INK)
        ax.axhline(0.01, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_yscale('log')
        ax.set_ylim(1.5e-5, 0.5)
        ax.set_xticks(range(4), [l for _, l in groups], fontsize=8)
        ax.tick_params(length=0)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(TASK_LABEL[t], loc='left', fontsize=11)
    axes[0].text(3.45, 0.0118, '门槛 0.01', ha='right', fontsize=8, color=MUTED)
    axes[0].set_ylabel('每个 seed 的最终 MSE（对数坐标）')
    fig.suptitle('稀疏网络大多达不到门槛（每列顶部为达标数）', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, 'report_sparse_fit.png'), dpi=160)
    plt.close(fig)


def fig_continual(rows):
    classes = [('local_sym', '局部对称'), ('local_asym', '局部不对称'), ('warp_sym', '对称变形'), ('warp_asym', '不对称变形')]
    groups = [('PBa', 'Picbreeder，只动现存边'), ('PBd', 'Picbreeder，全部权重可学'), ('G0', '稠密，训练到底'),
              ('G0e', '稠密，提前停止')]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharey=True)
    width = 0.19
    for ax, t in zip(axes, TASKS):
        for g_i, (g, label) in enumerate(groups):
            vals = []
            for cls, _ in classes:
                sel = [r for r in rows if r['task'] == t and r['group'] == g and r['edit_class'] == cls and r['lr_scale'] == 1.0]
                vals.append(100 * sum(r['adapted'] for r in sel) / len(sel) if sel else np.nan)
            x = np.arange(4) + (g_i - 1.5) * (width + 0.015)
            ax.bar(x, np.nan_to_num(vals), width=width, color=COLOR[g], label=label)
        ax.set_xticks(range(4), [c for _, c in classes], fontsize=8)
        ax.set_ylim(0, 105)
        ax.tick_params(length=0)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(TASK_LABEL[t] + ('（无提前停止组）' if t == 'skull' else ''), loc='left', fontsize=11)
    axes[0].set_ylabel('2000 步内达到适应标准的比例（%）')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, fontsize=9, loc='lower center')
    fig.suptitle('Picbreeder 起点很少能适应；训练到底的稠密网络在苹果的局部改动上同样很少达标（提前停止组门槛更宽松，不可直接比较）',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(os.path.join(OUT, 'report_continual.png'), dpi=160)
    plt.close(fig)


def fig_tradeoff(rows):
    """Local edits at 2000 steps: error left inside the edit against change outside it. Lower left is better on both."""
    groups = [('PBa', 'Picbreeder，只动现存边'), ('PBd', 'Picbreeder，全部权重可学'), ('G0', '稠密，训练到底')]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), sharex=True, sharey=True)
    for r, lr in enumerate([1.0, 0.1]):
        for c, t in enumerate(TASKS):
            ax = axes[r, c]
            for g, label in groups:
                pts = [x for x in rows if x['task'] == t and x['group'] == g and x['lr_scale'] == lr
                       and x['edit_class'].startswith('local')]
                ax.scatter([x['R2000'] for x in pts], [x['F2000'] for x in pts], s=38, marker=MARK[g], color=COLOR[g],
                           edgecolor=SURFACE, linewidth=0.8, label=label, zorder=3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            ax.set_title(f'{TASK_LABEL[t]}，学习率 ×{lr:g}', loc='left', fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel('改动区域内的剩余误差 R（2000 步）')
    for ax in axes[:, 0]:
        ax.set_ylabel('区域外的变动 F（2000 步）')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, fontsize=9, loc='lower center')
    fig.suptitle('局部改动：左下角最好。Picbreeder 起点在区域外的变动更大；苹果上偶有区域内误差更低的情况（事后描述，每图一个网络）',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(OUT, 'report_tradeoff.png'), dpi=160)
    plt.close(fig)


def main():
    exp, con = load('experiments_db.jsonl'), load('continual_db.jsonl')
    fig_wiring(exp)
    fig_fit_level(exp)
    fig_sparse_fit(exp)
    fig_continual(con)
    fig_tradeoff(load('continual_db_v2.jsonl'))


if __name__ == '__main__':
    main()
