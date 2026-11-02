"""Example images for the report. One fitted dense network per image learns the four discs in the first order, with no,
uniform or focused replay, using the kb-loop runner's own functions and replay seeds. Its forgetting should match the
table's row for that network and order. Usage: example_images.py <cppn_gpu dir> <assets dir> <out .npz>
"""
import sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from continual_seq import DEV, DISCS, LR, RADIUS, RES, coord_inputs, fit, fitted_dense, picbreeder, replay_mask

SETTINGS = [('none', 0.0, 0.0), ('uniform', 0.05, 0.0), ('focus', 0.05, 0.5)]
ORDER = [0, 1, 2, 3]
saved = {}
for task in ('skull', 'butterfly'):
    net_pb, ws_pb = picbreeder(sys.argv[2], task)
    net, ws, seeds = fitted_dense(sys.argv[2], task, net_pb, ws_pb, 'G0')
    X = coord_inputs(RES, DEV)
    xx, yy = X[:, 0].reshape(RES, RES), X[:, 1].reshape(RES, RES)
    discs = [torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < RADIUS for cx, cy in DISCS]
    with torch.no_grad():
        old = net_pb.render(ws_pb, X, RES)[0]
    new = 1 - old
    size = [((old - new) ** 2).mean(-1)[d].mean().item() for d in discs]
    params = [W[:1].repeat(len(SETTINGS), 1, 1).clone().requires_grad_() for W in ws]
    after = []
    for phase, current in enumerate(ORDER):
        target = old.clone()
        for i in ORDER[:phase + 1]:
            target[discs[i]] = new[discs[i]]
        learned = torch.zeros_like(old[..., 0]).bool()
        for i in ORDER[:phase]:
            learned |= discs[i]
        T = torch.stack([target] * len(SETTINGS))
        Mn = torch.stack([discs[current]] * len(SETTINGS)).float()
        Mr = torch.stack([replay_mask(~discs[current], f, 1009 * phase + sum(map(ord, task)), focus, learned) for _, f, focus in SETTINGS]).float()
        has_replay = (Mr.sum((1, 2)) > 0).float()

        def loss_fn(img, T=T, Mn=Mn, Mr=Mr, has_replay=has_replay):
            err = ((img - T) ** 2).mean(-1)
            return (err * Mn).sum((1, 2)) / Mn.sum((1, 2)) + has_replay * (err * Mr).sum((1, 2)) / Mr.sum((1, 2)).clamp(min=1)

        fit(net, params, X, loss_fn, 500, LR[task])
        with torch.no_grad():
            after.append(net.render(params, X, RES))
    forget = [[(((after[3][k] - new) ** 2).mean(-1)[discs[d]].mean().item() - ((after[p][k] - new) ** 2).mean(-1)[discs[d]].mean().item()) / size[d]
               for p, d in enumerate(ORDER[:3])] for k in range(len(SETTINGS))]
    print(task, 'seed', seeds[0], {name: [round(x, 4) for x in f] for (name, _, _), f in zip(SETTINGS, forget)}, flush=True)
    saved[f'{task}_target'] = target.cpu().numpy()
    saved[f'{task}_final'] = after[3].cpu().numpy()
    saved[f'{task}_forget'] = np.array(forget)
    saved[f'{task}_seed'] = np.array(seeds[0])
np.savez_compressed(sys.argv[3], **saved)
