"""Phase 0: do the candidate metrics separate Picbreeder (UFR) from SGD-imitation (FER) CPPNs?"""
import argparse
import json
import os

import numpy as np
import torch

from cppn_torch import DenseCPPN, coord_inputs
from metrics import (internal_metrics, perturbation_metrics, single_weight_specs,
                     random_direction_specs, base_symmetry, matched_median)
from train_sgd import pb_network

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['skull', 'butterfly', 'apple']
MAX_FIT_MSE = 0.01
# metric -> True if a Picbreeder-like (UFR) network should score HIGHER
DIRECTION = {'M1_mirror_frac': True, 'M2_distinct': False,
             'M3_asym_single': False, 'M3m_asym_single_matched': False,
             'M4_asym_randdir': False, 'M4m_asym_randdir_matched': False,
             'M5_hf_single': False, 'M5_hf_randdir': False}


def load_sgd(path, device):
    """Our own checkpoint written by train_sgd.py. Returns (net, [weights for each fitted seed], fit loss per
    fitted seed). Seeds above MAX_FIT_MSE did not reproduce the target, so the same-output premise fails for them."""
    ck = torch.load(path, weights_only=False)
    net = DenseCPPN(ck['groups'], ck['n_hidden'])
    fit = ck['losses'][-1, 1:]
    keep = [s for s in range(len(ck['seeds'])) if fit[s] < MAX_FIT_MSE]
    if len(keep) < len(fit):
        print(f'{os.path.basename(path)}: excluded unfitted seeds', [(ck['seeds'][s], float(fit[s]))
              for s in range(len(fit)) if s not in keep], flush=True)
    ws = [[W[s:s + 1].to(device) for W in ck['weights']] for s in keep]
    return net, ws, fit[keep]


def evaluate(net, ws, seed, n_single, n_dir):
    out = internal_metrics(net, ws)
    rng = np.random.default_rng(seed)
    p1 = perturbation_metrics(net, ws, single_weight_specs(net, n_single, rng, ws[0].device))
    p2 = perturbation_metrics(net, ws, random_direction_specs(net, n_dir, rng, ws[0].device))
    out.update(M3_asym_single=float(np.median(p1['asym'])), M4_asym_randdir=float(np.median(p2['asym'])),
               M3m_asym_single_matched=matched_median(p1, 'asym'),
               M4m_asym_randdir_matched=matched_median(p2, 'asym'),
               M5_hf_single=float(np.median(p1['hf'])), M5_hf_randdir=float(np.median(p2['hf'])),
               valid_single=p1['valid_frac'], valid_randdir=p2['valid_frac'],
               change_single=p1['median_change'], change_randdir=p2['median_change'],
               base_asym=base_symmetry(net, ws))
    return out, p1, p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--names', nargs='+', default=NAMES)
    ap.add_argument('--n_single', type=int, default=400)
    ap.add_argument('--n_dir', type=int, default=200)
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    args = ap.parse_args()
    dev = 'cuda'
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for name in args.names:
        net, ws_pb, _ = pb_network(name, dev)
        rows = {'pb': evaluate(net, ws_pb, 0, args.n_single, args.n_dir)[0]}
        snet, sws, fit = load_sgd(os.path.join(args.runs, f'sgd_{name}.pt'), dev)
        for s, ws in enumerate(sws):
            rows[f'sgd{s}'] = evaluate(snet, ws, 0, args.n_single, args.n_dir)[0]
            rows[f'sgd{s}']['fit_mse'] = float(fit[s])
        results[name] = rows

        print(f'\n=== {name}  (base image asym share: pb {rows["pb"]["base_asym"]:.3f})')
        print(f'{"metric":18s} {"pb":>8s} ' + ' '.join(f'{k:>8s}' for k in rows if k != 'pb') + '   separates?')
        for m, higher in DIRECTION.items():
            pb, sg = rows['pb'][m], [rows[k][m] for k in rows if k != 'pb']
            sep = (pb > max(sg)) if higher else (pb < min(sg))
            print(f'{m:18s} {pb:8.3f} ' + ' '.join(f'{v:8.3f}' for v in sg) + f'   {"YES" if sep else "no"}')
        for m in ['n_nonconst', 'valid_single', 'valid_randdir', 'change_single', 'change_randdir', 'fit_mse']:
            vals = [rows[k].get(m, np.nan) for k in rows]
            print(f'  {m:16s} ' + ' '.join(f'{v:8.4g}' for v in vals))

    def clean(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    with open(os.path.join(args.out, 'calibration.json'), 'w') as f:
        json.dump({n: {k: clean(v) for k, v in r.items()} for n, r in results.items()}, f, indent=1)


if __name__ == '__main__':
    main()
