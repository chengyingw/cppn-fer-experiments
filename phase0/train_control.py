"""Sparse SGD controls for the Picbreeder-vs-dense gap, same target, resolution and budget as train_sgd.py.

  G1a  Picbreeder wiring, only the NEAT links learn; cache and output copies stay fixed at 1 (raw architecture)
  G1b  Picbreeder wiring, every existing edge is re-initialized and learns
  G4   Dense SGD network pruned over five rounds to Picbreeder's total edge count, spread evenly over the
       layers and chosen by path strength, then fine-tuned; the same sparsity, but wiring chosen from the
       SGD network rather than inherited from Picbreeder
"""
import argparse
import os
import time

import numpy as np
import torch

from cppn_torch import DenseCPPN, coord_inputs
from train_sgd import HERE, fit, pb_network

LR = {'skull': 3e-3, 'butterfly': 3e-3, 'apple': 3e-4}
PRUNE_ITERS = (5000, 10000, 15000, 20000)  # with the initial prune, five rounds; the rest is fine-tuning


def sparse_init(mask, seeds, device):
    """LeCun normal per neuron, using the fan-in the wiring actually gives it (mask: list of (1, in, out))."""
    ws = []
    for m in mask:
        fan_in = m.sum(1, keepdim=True).clamp(min=1)
        z = torch.stack([torch.nn.init.trunc_normal_(torch.empty(m.shape[1:]), a=-2, b=2,
                                                     generator=torch.Generator().manual_seed(s + 1000 * len(ws)))
                         for s in seeds]).to(device)
        ws.append(z * (1.0 / fan_in) ** 0.5 / 0.87962566103423978 * m)
    return ws


@torch.no_grad()
def prune_to(params, masks, counts):
    """Keep, per seed and layer, the `count` edges carrying the most path strength (input flow x |w| x output
    flow). Layers are chosen from the output backward: each output channel keeps its strongest edge, and every
    neuron still used downstream keeps its strongest input before the rest are filled, so no output is cut off."""
    for s in range(params[0].shape[0]):
        mags = [p_[s].abs() * m[s] for p_, m in zip(params, masks)]
        forward = [torch.ones(mags[0].shape[0], device=mags[0].device)]
        for w in mags:
            f = w.T @ forward[-1]
            forward.append(f / f.max().clamp(min=1e-12))
        b = torch.ones(mags[-1].shape[1], device=mags[0].device)
        for l in reversed(range(len(mags))):
            w, k = mags[l], int(counts[l])
            score = forward[l][:, None] * w * b[None, :] + 1e-20 * w  # dead edges rank last
            keep = torch.zeros_like(w)
            if l == len(mags) - 1:
                keep.scatter_(0, score.topk(k // w.shape[1], dim=0).indices, 1)
            else:
                cols = (b > 0).nonzero().flatten()
                best, row = score[:, cols].max(0)
                pick = best.topk(min(k, len(cols))).indices
                keep[row[pick], cols[pick]] = 1
                left = k - int(keep.sum())
                if left > 0:
                    keep.view(-1)[(score * (1 - keep)).flatten().topk(left).indices] = 1
            masks[l][s] = keep
            b = (w * keep) @ b
            b = b / b.max().clamp(min=1e-12)
    for p_, m in zip(params, masks):
        p_.mul_(m)


def even_counts(total, n_hidden, n_out=3):
    """`total` edges spread evenly over the hidden layers, with one edge per output channel."""
    base, extra = divmod(total - n_out, n_hidden)
    return [base + (i < extra) for i in range(n_hidden)] + [n_out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--group', required=True, choices=['G1a', 'G1b', 'G4'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--iters', type=int, default=50_000)
    p.add_argument('--res', type=int, default=128)
    p.add_argument('--out', default=os.path.join(HERE, 'runs'))
    args = p.parse_args()

    dev = 'cuda'
    net, ws_pb, L = pb_network(args.name, dev)
    X = coord_inputs(args.res, dev)
    with torch.no_grad():
        target = net.render(ws_pb, X, args.res)[0]
    mask = [(W != 0).float() for W in ws_pb]
    tag = f'{args.group} {args.name}'
    hook, seeds = None, args.seeds

    if args.group == 'G4':
        ck = torch.load(os.path.join(args.out, f'sgd_{args.name}.pt'), weights_only=False)
        keep = [i for i, v in enumerate(ck['losses'][-1, 1:]) if v < 0.01]
        seeds = [ck['seeds'][i] for i in keep]
        params = [W[keep].to(dev).clone().requires_grad_() for W in ck['weights']]
        masks = [torch.ones_like(w) for w in params]
        final = even_counts(int(sum(m.sum().item() for m in mask)), net.n_hidden)
        rounds = len(PRUNE_ITERS) + 1
        sizes = [a * b for a, b in net.shapes]
        schedule = [[max(k, round(n * (k / n) ** ((r + 1) / rounds))) for n, k in zip(sizes, final)]
                    for r in range(rounds)]
        prune_to(params, masks, schedule[0])
        due = dict(zip(PRUNE_ITERS, schedule[1:]))

        def hook(it):
            if it in due:
                prune_to(params, masks, due[it])
        train_mask = value_mask = masks
    else:
        init = sparse_init(mask, seeds, dev)
        if args.group == 'G1a':
            learns = []
            for l, m in enumerate(mask):
                t = torch.zeros_like(m)
                if l < len(L['slots']):
                    for col, (_, novel) in L['slots'][l].items():
                        if novel:
                            t[..., col] = m[..., col]
                learns.append(t)
            init = [torch.where(t > 0, w, W) for w, t, W in zip(init, learns, ws_pb)]
            train_mask = learns
        else:
            train_mask = mask
        params = [w.clone().requires_grad_() for w in init]
        value_mask = mask

    t0 = time.time()
    losses = fit(net, params, X, args.res, target, args.iters, LR[args.name], train_mask=train_mask,
                 value_mask=value_mask, hook=hook, tag=tag)
    if args.group == 'G4':
        got = [int((w[0] != 0).sum()) for w in params]
        assert got == [int(k) for k in final], (got, final)
    torch.save(dict(groups=net.groups, n_hidden=net.n_hidden, seeds=seeds, iters=args.iters, lr=LR[args.name],
                    group=args.group, weights=[w.detach().cpu() for w in params], losses=losses),
               os.path.join(args.out, f'control_{args.group}_{args.name}.pt'))
    print('saved', tag, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
