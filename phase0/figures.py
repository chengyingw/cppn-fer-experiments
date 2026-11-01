"""Phase 0 figures: replication checks (paper Figs. 4/5/25) and metric summaries."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from calibrate import NAMES, DIRECTION, load_sgd
from cppn_torch import coord_inputs
from metrics import (feature_maps, internal_metrics, perturbation_metrics, random_direction_specs,
                     single_weight_specs)
from train_sgd import pb_network

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
RUNS = os.path.join(HERE, 'runs')
DEV = 'cuda'
PB_C, SGD_C = '#1f77b4', '#d62728'


def nets(name):
    net, ws, _ = pb_network(name, DEV)
    snet, sws, fit = load_sgd(os.path.join(RUNS, f'sgd_{name}.pt'), DEV)
    return (net, ws), (snet, sws), fit


def fig_outputs():
    fig, ax = plt.subplots(3, 4, figsize=(10, 7.6))
    for r, name in enumerate(NAMES):
        (net, ws), (snet, sws), fit = nets(name)
        X = coord_inputs(256, DEV)
        a = net.render(ws, X, 256)[0].cpu()
        b = snet.render(sws[0], X, 256)[0].cpu()
        X128 = coord_inputs(128, DEV)
        d = (net.render(ws, X128, 128)[0] - snet.render(sws[0], X128, 128)[0]).abs().max(-1)[0].cpu()
        for c, (im, t) in enumerate([(a, f'Picbreeder {name}'), (b, 'SGD first fitted seed (256 px)'),
                                     (d, f'|diff| at 128 px, max {d.max():.2f}')]):
            ax[r, c].imshow(im, cmap='magma' if c == 2 else None, vmin=0, vmax=0.2 if c == 2 else None)
            ax[r, c].set_title(t, fontsize=9)
        ck = torch.load(os.path.join(RUNS, f'sgd_{name}.pt'), weights_only=False)
        L = ck['losses']
        for s in range(L.shape[1] - 1):
            ax[r, 3].semilogy(L[:, 0], L[:, s + 1], color=SGD_C, alpha=0.4 + 0.2 * s)
        ax[r, 3].set_title('SGD training MSE (3 seeds)', fontsize=9)
        ax[r, 3].set_xlabel('iteration', fontsize=8)
    for a in ax[:, :3].ravel():
        a.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig1_outputs.png'), dpi=110)
    plt.close()


def fig_feature_maps(name, res=48):
    (net, ws), (snet, sws), _ = nets(name)
    for tag, n, w in [('pb', net, ws), ('sgd', snet, sws[0])]:
        layers = feature_maps(n, w, res)
        L, W = len(layers), layers[0].shape[0]
        fig, ax = plt.subplots(L, W, figsize=(W * 0.42, L * 0.42))
        for i, f in enumerate(layers):
            for j in range(W):
                a = ax[L - 1 - i, j]
                a.imshow(f[j].reshape(res, res).cpu(), cmap='bwr_r', vmin=-1, vmax=1)
                a.set_xticks([]); a.set_yticks([])
                for s in a.spines.values():
                    s.set_linewidth(0.2)
        fig.suptitle(f'{name} {tag}: hidden feature maps (bottom = layer 1)', fontsize=10)
        plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95)
        plt.savefig(os.path.join(OUT, f'fig2_fmaps_{name}_{tag}.png'), dpi=90)
        plt.close()


def fig_sweeps(name, n_dirs=120, show=6):
    """Random-direction sweeps (paper Fig. 25): the `show` highest-variance sweeps of each network."""
    (net, ws), (snet, sws), _ = nets(name)
    amps = (-1.0, -0.5, 0.0, 0.5, 1.0)
    fig, ax = plt.subplots(show, 2 * len(amps), figsize=(2 * len(amps) * 1.0, show * 1.05))
    for c0, (tag, n, w) in enumerate([('Picbreeder', net, ws), ('SGD first fitted seed', snet, sws[0])]):
        rng = np.random.default_rng(1)
        p = perturbation_metrics(n, w, random_direction_specs(n, n_dirs, rng, DEV, amps), res=64)
        imgs = p['imgs'].reshape(n_dirs, len(amps), 64, 64, 3)
        order = imgs.var(1).mean((1, 2, 3)).argsort(descending=True)[:show]
        for r, k in enumerate(order):
            for j in range(len(amps)):
                a = ax[r, c0 * len(amps) + j]
                a.imshow(imgs[k, j].cpu()); a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(f'{tag}\nt={amps[j]:+.1f}' if j == 2 else f't={amps[j]:+.1f}', fontsize=8)
    fig.suptitle(f'{name}: random-direction weight sweeps (top-variance)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f'fig3_sweeps_{name}.png'), dpi=100)
    plt.close()


def fig_metrics():
    with open(os.path.join(OUT, 'calibration.json')) as f:
        R = json.load(f)
    ms = list(DIRECTION)
    fig, ax = plt.subplots(1, len(ms), figsize=(2.3 * len(ms), 3.0))
    for a, m in zip(ax, ms):
        for i, name in enumerate(NAMES):
            rows = R[name]
            a.scatter([i], [rows['pb'][m]], color=PB_C, marker='o', s=40, zorder=3)
            sg = [rows[k][m] for k in rows if k != 'pb']
            a.scatter([i + 0.15] * len(sg), sg, color=SGD_C, marker='x', s=30)
        a.set_xticks(range(len(NAMES))); a.set_xticklabels(NAMES, fontsize=8, rotation=30)
        a.set_title(f"{m}\n(UFR {'higher' if DIRECTION[m] else 'lower'})", fontsize=8)
        if m == 'M2_distinct':
            a.set_yscale('log')
    ax[0].scatter([], [], color=PB_C, label='Picbreeder'); ax[0].scatter([], [], color=SGD_C, marker='x', label='SGD seeds')
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig4_metrics.png'), dpi=110)
    plt.close()


def fig_profiles_and_scatter():
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
    for c, name in enumerate(NAMES):
        (net, ws), (snet, sws), _ = nets(name)
        for tag, n, wl, col in [('Picbreeder', net, [ws], PB_C), ('SGD', snet, sws, SGD_C)]:
            for k, w in enumerate(wl):
                prof = internal_metrics(n, w)['layer_mirror']
                ax[0, c].plot(np.arange(1, len(prof) + 1), prof, color=col, alpha=0.9 if k == 0 else 0.4,
                              label=tag if k == 0 else None)
            rng = np.random.default_rng(0)
            p = perturbation_metrics(n, wl[0], single_weight_specs(n, 400, rng, DEV), res=64)
            ax[1, c].scatter(p['mag'], p['asym'], s=4, alpha=0.4, color=col, label=tag)
        ax[0, c].set_title(f'{name}: mean |mirror corr| of hidden neurons', fontsize=9)
        ax[0, c].set_xlabel('layer'); ax[0, c].set_ylim(0, 1.02); ax[0, c].legend(fontsize=8)
        ax[1, c].set_xscale('log'); ax[1, c].set_ylim(0, 1)
        ax[1, c].axvspan(0.02, 0.10, color='grey', alpha=0.12)
        ax[1, c].axhline(0.5, color='k', lw=0.5, ls=':')
        ax[1, c].set_title(f'{name}: single-weight perturbations', fontsize=9)
        ax[1, c].set_xlabel('image change (mean |dRGB|)'); ax[1, c].set_ylabel('asymmetric share of change')
        ax[1, c].legend(fontsize=8, markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig5_profiles_scatter.png'), dpi=110)
    plt.close()


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_outputs()
    for name in NAMES:
        fig_feature_maps(name)
        fig_sweeps(name)
    fig_metrics()
    fig_profiles_and_scatter()
    print('figures written to', OUT)
