"""Draws the report's result figures from the tables and logs in this repository. Run from anywhere: python make_figures.py"""
import json
import math
import os
import re
import shutil
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
SURFACE, INK, MUTED, GRID, GREY = '#fcfcfb', '#1f1e1c', '#52514e', '#e4e3df', '#a3a19b'
BLUE, ORANGE, GREEN, AMBER = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
TASKS = [('skull', '骷髅', BLUE), ('butterfly', '蝴蝶', ORANGE), ('apple', '苹果', GREEN)]
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.edgecolor': GRID, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlecolor': INK, 'text.color': INK})


def rows(folder):
    with open(os.path.join(ROOT, folder, 'cppn_continual_db.jsonl'), encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def median(table, group, task, key):
    return st.median(r[key] for r in table if r['group'] == group and r['task'] == task and r.get(key) is not None)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=180)
    plt.close(fig)


def clean(ax, axis='x'):
    ax.grid(axis=axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def relation():
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    def box(x, y, w, h, title, body, color=BLUE, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=SURFACE, ec=color, lw=1.6, ls='--' if dashed else '-'))
        ax.text(x + w / 2, y + h - 2.6, title, ha='center', va='center', fontsize=11, fontweight='bold', color=INK)
        ax.text(x + w / 2, y + (h - 4.2) / 2, body, ha='center', va='center', fontsize=9, color=MUTED, linespacing=1.45)

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13, lw=1.4, color=GREY if dashed else MUTED, ls='--' if dashed else '-', shrinkA=0, shrinkB=0))

    ax.text(1, 57.5, 'CPPN 是被研究的对象，kb-loop 是做实验和判定的框架', fontsize=14, fontweight='bold')
    box(2, 31, 27, 19, 'CPPN（被研究的对象）', 'Picbreeder 基因组\n解析为网络：PBd、PBa\nSGD 训练的稠密网络：G0\n输入坐标，输出图像', ORANGE)
    box(36, 31, 27, 19, '实验入口（kb-loop）', 'kb_lab.run_job\n新增的 cppn_continual 家族\n校验实验配置、管理缓存\n调用 GPU 训练脚本', BLUE)
    box(71, 31, 27, 19, '结果表', '每个改动一行\n遗忘量、区外漂移\n学习后的误差\ncppn_continual_db.jsonl', GREEN)
    box(71, 5, 27, 18, '判定（kb-loop）', 'kb_checks 重放判定带\n成立、部分成立、证伪\nverdict.json', BLUE)
    box(36, 5, 27, 18, '预测（kb-loop 格式）', 'seed_kb.json\nK9001～K9515\n运行前提交到 git', BLUE)
    box(2, 5, 27, 18, 'LLM 循环（没有运行）', 'kb_science_loop.py\n本应由 agent 提假说\n现在由人和 Claude 完成', GREY, dashed=True)
    arrow((29.6, 40.5), (35.4, 40.5))
    arrow((63.6, 40.5), (70.4, 40.5))
    arrow((84.5, 30.4), (84.5, 23.6))
    arrow((63.6, 14), (70.4, 14))
    arrow((29.6, 14), (35.4, 14), dashed=True)
    arrow((49.5, 23.6), (49.5, 30.4))
    save(fig, 'fig00_relation.png')


def apple_trap():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 1.1]})
    observed, gray = [0.190, 0.189, 0.080, 0.080], [0.189, 0.188, 0.079, 0.079]  # apple_plateau.py and apple_trap.py
    for i, (o, g) in enumerate(zip(observed, gray)):
        a.bar(i - 0.19, o, 0.36, color=ORANGE, label='实测平台' if i == 0 else None)
        a.bar(i + 0.19, g, 0.36, color=GREY, label='灰色输出的理论残差' if i == 0 else None)
    a.set_xticks(range(4), [f'圆盘 {i}' for i in range(4)])
    a.set_ylabel('改动区域的剩余误差')
    a.set_title('卡住的误差和理论值只差 0.001', loc='left', fontsize=11)
    a.legend(frameon=False, fontsize=9)
    clean(a, 'y')
    names = {'invert': '颜色反转', 'invert_st': '反转，保留梯度', 'hue60': '色相转 60°', 'hue120': '色相转 120°', 'bright': '只反转亮度'}
    values = {}
    with open(os.path.join(ROOT, 'kbloop_q1', 'apple_trap.log'), encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0] in names and parts[1] == '1':
                values[parts[0]] = float(parts[2])
    order = list(names)
    b.barh(range(5), [values[k] for k in order], color=[ORANGE] + [GREY] * 4, height=0.6)
    b.set_yticks(range(5), [names[k] for k in order])
    b.invert_yaxis()
    for i, k in enumerate(order):
        b.text(values[k] + 0.003, i, f'{values[k]:.3f}', va='center', fontsize=9)
    b.set_xlim(0, 0.16)
    b.set_xlabel('500 步后的剩余误差（100% 回放）')
    b.set_title('只有颜色反转且截断梯度时才卡住', loc='left', fontsize=11)
    clean(b)
    fig.tight_layout()
    save(fig, 'fig03_apple_trap.png')


def dots(ax_f, ax_b, table, methods, highlight, xmax, log=True):
    for i, (group, _) in enumerate(methods):
        if i == highlight:
            ax_f.axhspan(i - 0.42, i + 0.42, color=BLUE, alpha=0.07, lw=0)
            ax_b.axhspan(i - 0.42, i + 0.42, color=BLUE, alpha=0.07, lw=0)
        for task, label, color in TASKS[:2]:
            ax_f.scatter(median(table, group, task, 'forget'), i, s=55, color=color, zorder=3, label=label if i == 0 else None)
            ax_b.scatter(median(table, group, task, 'background'), i, s=55, color=color, zorder=3)
    ax_f.set_yticks(range(len(methods)), [m for _, m in methods])
    ax_f.invert_yaxis()  # the axes share y, so once is enough
    for ax in (ax_f, ax_b):
        clean(ax)
    ax_b.tick_params(labelleft=False)
    ax_f.set_xlim(0, xmax)
    ax_f.set_xlabel('旧改动的遗忘量（中位数）')
    ax_b.set_xlabel('从未改动区域的漂移（中位数）')
    if log:
        ax_b.set_xscale('log')
    handles, labels = ax_f.get_legend_handles_labels()
    ax_f.figure.legend(handles, labels, frameon=False, fontsize=9, ncol=2, loc='upper right')


def replay_methods():
    table = rows('kbloop_q2')
    methods = [('replay0', '无回放'), ('G0_r0_l20.1', 'L2 约束 0.1，无回放'), ('replay0.05', '均匀 5%'), ('G0_r0.05_snapshot', '旧输出当标签 5%'),
               ('G0_r0.05_focus0.5', '聚焦到旧改动 5%'), ('G0_r1_snapshot', '旧输出当标签 100%'), ('replay1', '存储标签 100%')]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    dots(a, b, table, methods, 4, 0.25)
    fig.suptitle('同样 5% 的预算，聚焦到旧改动后，遗忘降到约五分之一（A100）', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, 'fig06_replay_methods.png')


def heldout():
    data = {}
    with open(os.path.join(ROOT, 'kbloop_q2', 'focus_heldout.log'), encoding='utf-8') as f:
        for line in f:
            p = line.split()
            if len(p) == 7 and p[1] in ('uniform', 'focus'):
                data[p[0], p[1]] = dict(all=float(p[4]), held=float(p[6]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (task, label, _) in zip(axes, TASKS[:2]):
        for i, (key, name, color) in enumerate([('all', '整个圆盘', GREY), ('held', '从未被标注的像素', BLUE)]):
            values = [data[task, 'uniform'][key], data[task, 'focus'][key]]
            ax.bar([j + (i - 0.5) * 0.38 for j in range(2)], values, 0.36, color=color, label=name)
            for j, v in enumerate(values):
                ax.text(j + (i - 0.5) * 0.38, v + 0.004, f'{v:.3f}', ha='center', fontsize=8.5)
        ax.set_xticks(range(2), ['均匀 5%', '聚焦 5%'])
        ax.set_title(f"{label}：未标注像素的比值 {data[task, 'focus']['held'] / data[task, 'uniform']['held']:.2f}", loc='left', fontsize=11)
        clean(ax, 'y')
    axes[0].set_ylabel('遗忘量（中位数）')
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, 'fig07_heldout.png')


def picbreeder():
    table = rows('kbloop_q3')
    panels = [('保留梯度', 'G0_r0_st', 'PBd_r0_st', 'PBa_r0_st'), ('原始输出', 'replay0', 'PBd_r0', 'PBa_r0')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (title, g0, pbd, pba) in zip(axes, panels):
        for t, (task, label, _) in enumerate(TASKS):
            base = median(table, g0, task, 'forget')
            ax.plot([base / 2, base / 2], [t - 0.3, t + 0.3], color=MUTED, lw=2, label='稠密的一半（预测的成立线）' if t == 0 else None)
            for k, (group, name, color) in enumerate([(g0, '稠密', GREY), (pbd, 'Picbreeder，全部权重', BLUE), (pba, 'Picbreeder，仅现存边', AMBER)]):
                ax.scatter(median(table, group, task, 'forget'), t - 0.2 + 0.2 * k, s=55, color=color, zorder=3, label=name if t == 0 else None)
            ax.text(0.68, t - 0.33, f"全部权重 / 稠密 = {median(table, pbd, task, 'forget') / base:.2f}", ha='right', fontsize=8.5, color=MUTED)
        ax.set_yticks(range(3), [x[1] for x in TASKS])
        ax.set_ylim(2.6, -0.6)
        ax.set_xlim(0, 0.7)
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_xlabel('旧改动的遗忘量（中位数，无回放）')
        clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc='lower center')
    fig.suptitle('连续改动时，Picbreeder 没有遗忘得更少', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, 'fig08_picbreeder_sequence.png')


def focus_snapshot():
    table = rows('kbloop_q4')
    methods = [('replay0.05', '均匀，存储标签'), ('G0_r0.05_snapshot', '均匀，旧输出'), ('G0_r0.05_focus0.5', '聚焦，存储标签'), ('G0_r0.05_focus0.5_snapshot', '聚焦，旧输出')]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    dots(a, b, table, methods, 3, 0.25, log=False)
    b.set_xlim(0, 0.3)
    fig.suptitle('不存标签也够用：聚焦加旧输出的遗忘最低，代价是背景漂移（5% 预算，RTX 4060）', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, 'fig09_focus_snapshot.png')


def knob():
    table = rows('kbloop_q5')
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for t, (task, _, _) in enumerate(TASKS):
        for k, (group, name, color) in enumerate([('PBd_r0_knobpb', 'Picbreeder，全部权重', BLUE), ('PBa_r0_knobpb', 'Picbreeder，仅现存边', AMBER)]):
            y = t - 0.12 + 0.24 * k
            a.scatter(median(table, group, task, 'R_after') / median(table, 'replay0_knobpb', task, 'R_after'), y, s=55, color=color, zorder=3, label=name if t == 0 else None)
            b.scatter(median(table, group, task, 'forget') / median(table, 'replay0_knobpb', task, 'forget'), y, s=55, color=color, zorder=3)
    for ax in (a, b):
        ax.axvline(1, color=MUTED, lw=1.2)
        ax.axvline(0.5, color=MUTED, lw=1.2, ls='--')
        ax.set_yticks(range(3), [x[1] for x in TASKS])
        ax.set_ylim(2.6, -0.6)
        clean(ax)
    a.set_xscale('log')
    a.set_xticks([0.5, 1, 10, 25], ['0.5', '1', '10', '25'])
    a.set_xlabel('改动后的误差，与稠密网络的比值（对数）')
    b.set_xlim(0, 2)
    b.set_xlabel('遗忘量，与稠密网络的比值')
    b.text(0.48, -0.42, '预测线 0.5', ha='right', fontsize=8.5, color=MUTED)
    b.text(1.02, -0.42, '相等', fontsize=8.5, color=MUTED)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2, loc='lower center')
    fig.suptitle('在自己的旋钮上，Picbreeder 也没有学得更好或忘得更少（无回放）', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save(fig, 'fig11_knob_results.png')


relation()
apple_trap()
replay_methods()
heldout()
picbreeder()
focus_snapshot()
knob()
for source, name in [('phase0/out/report_wiring_forced.png', 'fig01_wiring_forced.png'), ('phase0/out/report_fit_level.png', 'fig02_fit_level.png'),
                     ('phase0/out/report_tradeoff.png', 'fig04_single_edit_tradeoff.png'), ('kbloop_q1/report_replay_curve.png', 'fig05_replay_curve.png'),
                     ('kbloop_q5/knob_examples.png', 'fig10_knob_examples.png')]:
    shutil.copyfile(os.path.join(ROOT, source), os.path.join(HERE, name))
print(sorted(f for f in os.listdir(HERE) if f.endswith('.png')))
