"""Measure every network with one code path and rewrite kb/experiments_db.jsonl: one row per network and seed.

Perturbations are restricted to existing edges in all groups, so nothing adds an edge to the wiring.
Seeds that missed the fit threshold stay in the table with fit=false and no metrics.
"""
import glob
import json
import os

import numpy as np
import torch

from calibrate import MAX_FIT_MSE, NAMES
from cppn_torch import DenseCPPN, coord_inputs
from metrics import (internal_metrics, matched_median, perturbation_metrics, random_direction_specs,
                     single_weight_specs, structural_metrics)
from train_sgd import HERE, pb_network

DB = os.path.join(HERE, '..', 'kb', 'experiments_db.jsonl')
RUNS = os.path.join(HERE, 'runs')
BATCHES = [('b0', RUNS), ('b6', os.path.join(RUNS, 'b6'))]  # b6: the fixed six extra butterfly seeds
DEV = 'cuda'
RES = 128


def measure(net, ws, seed=0, n_single=400, n_dir=200):
    rng = np.random.default_rng(seed)
    i, s = internal_metrics(net, ws), structural_metrics(net, ws)
    p3 = perturbation_metrics(net, ws, single_weight_specs(net, n_single, rng, DEV, ws=ws))
    p4 = perturbation_metrics(net, ws, random_direction_specs(net, n_dir, rng, DEV, ws=ws))
    clean = lambda v: None if v is None or np.isnan(v) else round(float(v), 5)
    return dict(M1=clean(i['M1_mirror_frac']), M2=i['M2_distinct'], forced_frac=clean(s['forced_frac']),
                M1x=clean(s['M1x']), n_free=s['n_free'], M3a=clean(matched_median(p3, 'asym')),
                M4a=clean(matched_median(p4, 'asym')),
                nonzero=int(sum((W != 0).sum() for W in ws)))


def main():
    rows = []
    for name in NAMES:
        net, ws_pb, _ = pb_network(name, DEV)
        X = coord_inputs(RES, DEV)
        target = net.render(ws_pb, X, RES)[0]
        base = dict(set_id=f'img{RES}_{name}', task=name, res=RES)
        rows.append(dict(base, exp_id=f'{name}_PB', group='PB', seed=None, mse=0.0, fit=True, **measure(net, ws_pb)))
        print(rows[-1], flush=True)
        for batch, folder in BATCHES:
            paths = [('G0', os.path.join(folder, f'sgd_{name}.pt'))]
            paths += [(os.path.basename(f).split('_')[1], f)
                      for f in sorted(glob.glob(os.path.join(folder, f'control_*_{name}.pt')))]
            for group, path in paths:
                if not os.path.exists(path):
                    continue
                ck = torch.load(path, weights_only=False)
                snet = DenseCPPN(ck['groups'], ck['n_hidden'])
                for k, seed in enumerate(ck['seeds']):
                    ws = [W[k:k + 1].to(DEV) for W in ck['weights']]
                    with torch.no_grad():  # measured here rather than read from the (noisy, lagging) training log
                        mse = ((snet.render(ws, X, RES)[0] - target) ** 2).mean().item()
                    row = dict(base, exp_id=f'{name}_{group}_s{seed}', group=group, seed=seed, batch=batch,
                               mse=round(mse, 6), fit=mse < MAX_FIT_MSE)
                    if row['fit']:
                        row.update(measure(snet, ws))
                    rows.append(row)
                    print(row, flush=True)
    with open(DB, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
