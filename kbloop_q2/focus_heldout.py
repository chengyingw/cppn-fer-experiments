"""Does focused replay protect a disc, or only the pixels it labels? Predictions committed before running, in focus_heldout.md.

Reruns replay0.05 and G0_r0.05_focus0.5 with continual_seq.py's own functions, and splits each old disc's forgetting by whether
a pixel was ever a replay pixel after that disc was learned. `all` must match the table, which checks this reimplementation.
Usage: focus_heldout.py <cppn_gpu dir> <assets dir>
"""
import statistics as st
import sys

import torch

sys.path.insert(0, sys.argv[1])
from continual_seq import DEV, DISCS, LR, ORDERS, RADIUS, RES, coord_inputs, fit, fitted_dense, picbreeder, replay_mask

ASSETS = sys.argv[2]


def run(task, focus):
    net_pb, ws_pb = picbreeder(ASSETS, task)
    net, ws, seeds = fitted_dense(ASSETS, task, net_pb, ws_pb, 'G0')
    X = coord_inputs(RES, DEV)
    xx, yy = X[:, 0].reshape(RES, RES), X[:, 1].reshape(RES, RES)
    discs = [torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < RADIUS for cx, cy in DISCS]
    with torch.no_grad():
        old = net_pb.render(ws_pb, X, RES)[0]
    new = 1 - old
    size = [((old - new) ** 2).mean(-1)[d].mean().item() for d in discs]
    K, O = len(seeds), len(ORDERS)
    params = [W.repeat_interleave(O, 0).clone().requires_grad_() for W in ws]
    images, masks = [], []
    for phase in range(4):
        T, Mn, Mr = [], [], []
        for _ in range(K):
            for o, order in enumerate(ORDERS):
                t = old.clone()
                for i in order[:phase + 1]:
                    t[discs[i]] = new[discs[i]]
                T.append(t)
                Mn.append(discs[order[phase]])
                learned = torch.zeros_like(old[..., 0]).bool()
                for i in order[:phase]:
                    learned |= discs[i]
                Mr.append(replay_mask(~discs[order[phase]], 0.05, 1009 * (o * 10 + phase) + sum(map(ord, task)), focus, learned))
        T, Mn, Mr = torch.stack(T), torch.stack(Mn).float(), torch.stack(Mr).float()

        def loss_fn(img):
            err = ((img - T) ** 2).mean(-1)
            return (err * Mn).sum((1, 2)) / Mn.sum((1, 2)) + (err * Mr).sum((1, 2)) / Mr.sum((1, 2)).clamp(min=1)

        fit(net, params, X, loss_fn, 500, LR[task])
        with torch.no_grad():
            images.append(((net.render(params, X, RES) - new) ** 2).mean(-1))
        masks.append(Mr.bool())
    rows = []
    for k in range(K):
        for o, order in enumerate(ORDERS):
            j = k * O + o
            for pos, disc in enumerate(order[:3]):
                labelled = torch.stack([masks[p][j] for p in range(pos + 1, 4)]).any(0)
                parts = {'all': discs[disc], 'labelled': discs[disc] & labelled, 'held': discs[disc] & ~labelled}
                row = {'n_held': parts['held'].sum().item(), 'n_disc': discs[disc].sum().item()}
                for name, m in parts.items():
                    if m.any():
                        row[name] = (images[3][j][m].mean() - images[pos][j][m].mean()).item() / size[disc]
                rows.append(row)
    return rows


print('task       group    n  held_share  all     labelled  held')
for task in ('skull', 'butterfly', 'apple'):
    held = {}
    for name, focus in (('uniform', 0.0), ('focus', 0.5)):
        rows = run(task, focus)
        med = lambda key: st.median(r[key] for r in rows if key in r)
        share = st.median(r['n_held'] / r['n_disc'] for r in rows)
        held[name] = med('held')
        print(f"{task:10s} {name:7s} {len(rows):3d}  {share:.2f}        {med('all'):.4f}  {med('labelled'):.4f}    {med('held'):.4f}", flush=True)
    print(f"{task:10s} held ratio focus / uniform = {held['focus'] / held['uniform']:.3f}", flush=True)
