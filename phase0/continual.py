"""Continual-learning test: networks that already draw an image learn a small change to it.

Twelve edits, three per class:
  local_sym / local_asym  invert colours inside two mirrored discs / one disc of the same total area
  warp_sym / warp_asym    scale the image horizontally about its centre / shift it horizontally
Groups: PBd (Picbreeder start, every weight may learn, so the search space equals G0's), PBa (Picbreeder, existing
edges only), G0 (fully fitted dense SGD), G0e (dense SGD stopped at MSE <= 0.0065).

Recorded per network and edit, every 10 steps:
  A1     steps until MSE on the new target <= the network's MSE on the old target + 10% of the edit's size E0
  F      mean squared change of the network's own output outside the edited region, divided by E0. Local edits only.

Known limits: A1 is right-censored at the step budget. A1's threshold and F both depend on fit level: a coarse network
gets a looser threshold from its larger old error, and F includes its continued fitting of the old image, so G0e is not
directly comparable with the fully fitted groups.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from calibrate import MAX_FIT_MSE, NAMES
from cppn_torch import DenseCPPN, coord_inputs, hsv2rgb
from train_control import LR
from train_sgd import HERE, fit, pb_network

RUNS = os.path.join(HERE, 'runs')
DB = os.path.join(HERE, '..', 'kb', 'continual_db.jsonl')
DEV = 'cuda'
RES = 64
ITERS = 2000
LR_SCALES = (1.0, 0.1)  # 1.0 is the registered condition; 0.1 is descriptive, for the step-size bias
RADIUS = 0.25    # single-disc radius; each of the two mirrored discs has the same total area
MARGIN = 0.05    # band around the discs left out of the unchanged region
ROWS = (-0.35, 0.0, 0.35)


def warped_inputs(X, sx=1.0, tx=0.0):
    x = X[:, 0] / sx - tx
    d = torch.sqrt(x ** 2 + X[:, 1] ** 2) * 1.4
    return torch.stack([x, X[:, 1], d, X[:, 3]], -1)


def disc_distance(X, centers, radius):
    """Per pixel: distance from the nearest disc edge, negative inside (res, res)."""
    dist = torch.stack([torch.sqrt((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2) - radius for cx, cy in centers])
    return dist.min(0)[0].reshape(RES, RES)


def make_edits(net, ws, X):
    """Returns the old target and a list of dict(name, cls, target, keep); keep marks the unchanged region."""
    old = net.render(ws, X, RES)[0]
    edits = []
    for k, y in enumerate(ROWS):
        for cls, centers, radius in [('local_sym', [(-0.35, y), (0.35, y)], RADIUS / 2 ** 0.5),
                                     ('local_asym', [(0.35, y)], RADIUS)]:
            dist = disc_distance(X, centers, radius)
            inside = (dist < 0)[..., None]
            edits.append(dict(name=f'{cls}{k}', cls=cls, target=torch.where(inside, 1 - old, old), keep=dist > MARGIN))
    for k, (s, t) in enumerate([(1.1, 0.1), (1.15, 0.15), (1.2, 0.2)]):
        for cls, inp in [('warp_sym', warped_inputs(X, sx=s)), ('warp_asym', warped_inputs(X, tx=t))]:
            edits.append(dict(name=f'{cls}{k}', cls=cls, target=net.render(ws, inp, RES)[0], keep=None))
    return old, edits


def networks(name, net_pb, ws_pb):
    """Yields (group, net, weights as list of (K, in, out), seeds, batch tags, train_mask, value_mask)."""
    mask = [(W != 0).float() for W in ws_pb]
    yield 'PBd', net_pb, ws_pb, [None], [None], None, None
    yield 'PBa', net_pb, ws_pb, [None], [None], mask, mask
    for group, stem in [('G0', f'sgd_{name}.pt'), ('G0e', f'control_G0e_{name}.pt')]:
        keep, seeds, tags, net = [], [], [], None
        for batch, folder in [('b0', RUNS), ('b6', os.path.join(RUNS, 'b6'))]:
            path = os.path.join(folder, stem)
            if not os.path.exists(path):
                continue
            ck = torch.load(path, weights_only=False)
            net = DenseCPPN(ck['groups'], ck['n_hidden'])
            X = coord_inputs(128, DEV)
            target = net_pb.render(ws_pb, X, 128)[0]
            for k, seed in enumerate(ck['seeds']):
                ws = [W[k:k + 1].to(DEV) for W in ck['weights']]
                with torch.no_grad():
                    if ((net.render(ws, X, 128)[0] - target) ** 2).mean().item() < MAX_FIT_MSE:
                        keep.append(ws)
                        seeds.append(seed)
                        tags.append(batch)
        if keep:
            yield group, net, [torch.cat([ws[l] for ws in keep]) for l in range(len(net.shapes))], seeds, tags, None, None


@torch.no_grad()
def render(net, params, X):
    hsv = net.forward(params, X)
    return hsv2rgb(hsv[..., 0], hsv[..., 1], hsv[..., 2]).reshape(-1, RES, RES, 3)


def run_group(task, group, net, ws, seeds, tags, train_mask, value_mask, old, edits, X, iters, lr_scale):
    K, E = len(seeds), len(edits)
    params = [W.repeat_interleave(E, 0).clone().requires_grad_() for W in ws]
    targets = torch.stack([e['target'] for e in edits]).repeat(K, 1, 1, 1)
    start = render(net, params, X)
    E0 = ((old - targets) ** 2).mean((1, 2, 3))
    base = ((start - old) ** 2).mean((1, 2, 3))
    keep = [e['keep'] for e in edits] * K
    history = []

    def record(it):
        if it % 10 == 0 or it == iters - 1:
            with torch.no_grad():
                out = render(net, params, X)
                new = ((out - targets) ** 2).mean((1, 2, 3))
                moved = (out - start).pow(2).mean(-1)
                F = [moved[i][keep[i]].mean().item() / E0[i].item() if keep[i] is not None else float('nan')
                     for i in range(K * E)]
            history.append((it, new.cpu().numpy(), np.array(F)))

    record(0)
    fit(net, params, X, RES, targets, iters, LR[task] * lr_scale, train_mask=train_mask, value_mask=value_mask,
        hook=record, tag=f'{group} {task} lr x{lr_scale}')
    steps = np.array([h[0] for h in history])
    mse = np.stack([h[1] for h in history], 1)
    moved = np.stack([h[2] for h in history], 1)
    rows = []
    for k, seed in enumerate(seeds):
        for j, e in enumerate(edits):
            i = k * E + j
            hit = np.flatnonzero(mse[i] <= base[i].item() + 0.1 * E0[i].item())
            at = int(hit[0]) if len(hit) else None
            rows.append(dict(kind='continual', exp_id=f'{task}_{group}_s{seed}_{e["name"]}_lr{lr_scale}', task=task,
                             group=group, seed=seed, batch=tags[k], lr_scale=lr_scale, edit=e['name'],
                             edit_class=e['cls'], E0=round(E0[i].item(), 6),
                             base_mse=round(base[i].item(), 6), adapted=at is not None,
                             A1=int(steps[at]) if at is not None else None,
                             A1c=int(steps[at]) if at is not None else iters,
                             final_mse=round(float(mse[i, -1]), 6),
                             F=round(float(moved[i, at]), 6) if at is not None and e['keep'] is not None else None))
    return rows, dict(steps=steps, mse=mse, moved=moved)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--names', nargs='+', default=NAMES)
    p.add_argument('--iters', type=int, default=ITERS)
    p.add_argument('--db', default=DB)
    p.add_argument('--curves', default=os.path.join(RUNS, 'continual'))
    args = p.parse_args()
    os.makedirs(args.curves, exist_ok=True)
    X = coord_inputs(RES, DEV)
    rows = []
    if os.path.exists(args.db):  # resume: a finished task, group and learning rate is not run again
        with open(args.db, encoding='utf-8') as f:
            rows = [json.loads(l) for l in f if l.strip()]
    done = {(r['task'], r['group'], r['lr_scale']) for r in rows}
    for task in args.names:
        net_pb, ws_pb, _ = pb_network(task, DEV)
        with torch.no_grad():
            old, edits = make_edits(net_pb, ws_pb, X)
        for group, net, ws, seeds, tags, tm, vm in networks(task, net_pb, ws_pb):
            for lr_scale in LR_SCALES:
                if (task, group, lr_scale) in done:
                    continue
                r, curves = run_group(task, group, net, ws, seeds, tags, tm, vm, old, edits, X, args.iters, lr_scale)
                rows += r
                np.savez_compressed(os.path.join(args.curves, f'{task}_{group}_lr{lr_scale}.npz'), **curves)
                with open(args.db, 'a', encoding='utf-8') as f:
                    for x in r:
                        f.write(json.dumps(x, ensure_ascii=False) + '\n')
                print(task, group, 'lr x', lr_scale, 'networks', len(seeds), 'adapted',
                      sum(x['adapted'] for x in r), '/', len(r), flush=True)


if __name__ == '__main__':
    main()
