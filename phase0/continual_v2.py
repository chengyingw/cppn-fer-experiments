"""Continual-learning remeasurement: same edits and optimizer as continual.py, with metrics that do not depend on the
network under test. Fully fitted groups only: PBd, PBa and G0; two learning rates.

At fixed steps (CHECKPOINTS), per network and edit:
  R<step>  remaining error toward the new target, relative to the edit's own size. Local edits: inside the edited
           region only, against the error of the old image there. Warps: whole image, against E0.
  F<step>  mean squared change of the network's own output outside the edited region, relative to E0. Local edits only.
  A1abs    first step with R <= ADAPTED, a threshold shared by all networks; None if not reached (right-censored).
A network that does not learn scores a low F, so F is only ever read together with R.
Results go to kb/continual_db_v2.jsonl; continual.py's table, which older claims rest on, is left alone.
"""
import argparse
import json
import os

import numpy as np
import torch

from calibrate import NAMES
from continual import RES, make_edits, networks, render
from cppn_torch import coord_inputs
from train_control import LR
from train_sgd import HERE, fit, pb_network

RUNS = os.path.join(HERE, 'runs')
DB = os.path.join(HERE, '..', 'kb', 'continual_db_v2.jsonl')
DEV = 'cuda'
ITERS = 2000
CHECKPOINTS = (100, 500, 2000)
ADAPTED = 0.1
LR_SCALES = (1.0, 0.1)
GROUPS = ('PBd', 'PBa', 'G0')


def run_group(task, group, net, ws, seeds, tags, train_mask, value_mask, old, edits, X, iters, lr_scale):
    K, E = len(seeds), len(edits)
    n = K * E
    params = [W.repeat_interleave(E, 0).clone().requires_grad_() for W in ws]
    targets = torch.stack([e['target'] for e in edits]).repeat(K, 1, 1, 1)
    start = render(net, params, X)
    masks = []  # per run: (region where R is measured, unchanged region or None)
    for _ in range(K):
        for e in edits:
            changed = (e['target'] != old).any(-1) if e['keep'] is not None else torch.ones_like(old[..., 0]).bool()
            masks.append((changed, e['keep']))
    size = torch.stack([((old - t) ** 2).mean(-1)[m[0]].mean() for t, m in zip(targets, masks)])
    E0 = ((old - targets) ** 2).mean((1, 2, 3))
    history = []

    def record(it):
        if it % 10 == 0 or it == iters - 1:
            with torch.no_grad():
                out = render(net, params, X)
                err = ((out - targets) ** 2).mean(-1)
                moved = ((out - start) ** 2).mean(-1)
                R = torch.stack([err[i][masks[i][0]].mean() / size[i] for i in range(n)])
                F = [moved[i][masks[i][1]].mean().item() / E0[i].item() if masks[i][1] is not None else float('nan')
                     for i in range(n)]
            history.append((it, R.cpu().numpy(), np.array(F)))

    record(0)
    fit(net, params, X, RES, targets, iters, LR[task] * lr_scale, train_mask=train_mask, value_mask=value_mask,
        hook=record, tag=f'{group} {task} lr x{lr_scale}')
    steps = np.array([h[0] for h in history])
    R = np.stack([h[1] for h in history], 1)
    F = np.stack([h[2] for h in history], 1)
    rows = []
    for k, seed in enumerate(seeds):
        for j, e in enumerate(edits):
            i = k * E + j
            hit = np.flatnonzero(R[i] <= ADAPTED)
            row = dict(kind='continual_v2', exp_id=f'{task}_{group}_s{seed}_{e["name"]}_lr{lr_scale}', task=task,
                       group=group, seed=seed, batch=tags[k], lr_scale=lr_scale, edit=e['name'], edit_class=e['cls'],
                       E0=round(E0[i].item(), 6), adapted=len(hit) > 0, A1abs=int(steps[hit[0]]) if len(hit) else None)
            for c in CHECKPOINTS:
                t = int(np.argmin(np.abs(steps - min(c, iters - 1))))
                row[f'R{c}'] = round(float(R[i, t]), 5)
                row[f'F{c}'] = round(float(F[i, t]), 5) if e['keep'] is not None else None
            rows.append(row)
    return rows, dict(steps=steps, R=R, F=F)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--names', nargs='+', default=NAMES)
    p.add_argument('--iters', type=int, default=ITERS)
    p.add_argument('--db', default=DB)
    p.add_argument('--curves', default=os.path.join(RUNS, 'continual_v2'))
    args = p.parse_args()
    os.makedirs(args.curves, exist_ok=True)
    X = coord_inputs(RES, DEV)
    done = set()
    if os.path.exists(args.db):  # resume after a failure
        with open(args.db, encoding='utf-8') as f:
            done = {(r['task'], r['group'], r['lr_scale']) for r in map(json.loads, f) if r}
    for task in args.names:
        net_pb, ws_pb, _ = pb_network(task, DEV)
        with torch.no_grad():
            old, edits = make_edits(net_pb, ws_pb, X)
        for group, net, ws, seeds, tags, tm, vm in networks(task, net_pb, ws_pb):
            if group not in GROUPS:
                continue
            for lr_scale in LR_SCALES:
                if (task, group, lr_scale) in done:
                    continue
                rows, curves = run_group(task, group, net, ws, seeds, tags, tm, vm, old, edits, X, args.iters, lr_scale)
                np.savez_compressed(os.path.join(args.curves, f'{task}_{group}_lr{lr_scale}.npz'), **curves)
                with open(args.db, 'a', encoding='utf-8') as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                print(task, group, 'lr x', lr_scale, 'networks', len(seeds), 'adapted',
                      sum(r['adapted'] for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
