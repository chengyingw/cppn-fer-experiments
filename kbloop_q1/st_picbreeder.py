"""Does keeping gradients through the output clipping change who learns? Exploratory; predictions committed before running.

Same image mapping in both modes. `hard` is the original clipping; `st` keeps the gradient where saturation or brightness is
clipped (straight-through). One colour-inversion disc, 500 steps, full replay, each task's training learning rate.
Groups: PBd (Picbreeder start, all weights free), PBa (Picbreeder start, existing edges only), G0 (fitted dense).
Two discs per network; R is error left in the disc, F is change outside it relative to the edit's size.

Predictions, median R at step 500, ratio st / hard:
  P1 G0 on apple: at most 0.2. The plateau is gone for dense networks, as apple_trap.py showed.
  P2 PBd on apple: at most 0.5. Picbreeder also suffers the trap on the colour image.
  P3 PBd on skull and butterfly: at least 0.7. Their targets have little chroma, so clipping is not why Picbreeder learns badly.
     If this ratio falls below 0.3, the earlier Picbreeder result was largely an output-layer effect and must be re-read.
  P4 PBd stays at least twice G0's R under st on skull and butterfly, as in C-019.
"""
import os
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase0'))
from cppn_torch import DenseCPPN, coord_inputs, hsv2rgb
from train_sgd import pb_network

DEV, RES, STEPS = 'cuda', 64, 500
LR = {'skull': 3e-3, 'butterfly': 3e-3, 'apple': 3e-4}
DISCS = [(-0.4, -0.4), (-0.4, 0.4)]
X = coord_inputs(RES, DEV)
xx, yy = X[:, 0].reshape(RES, RES), X[:, 1].reshape(RES, RES)


def image(net, params, straight):
    hsv = net.forward(params, X)[0].reshape(RES, RES, 3)
    s, v = hsv[..., 1], hsv[..., 2].abs()
    if straight:
        s = s + (s.clamp(0, 1) - s).detach()
        v = v + (v.clamp(0, 1) - v).detach()
    return hsv2rgb(hsv[..., 0], s, v), hsv


def run(task, group, disc_xy, straight, net, ws, mask, old):
    params = [W.clone().requires_grad_() for W in ws]
    disc = torch.sqrt((xx - disc_xy[0]) ** 2 + (yy - disc_xy[1]) ** 2) < 0.2
    target = old.clone()
    target[disc] = 1 - old[disc]
    size = ((old - target) ** 2).mean(-1)[disc].mean().item()
    with torch.no_grad():
        start = image(net, params, False)[0]
    opt = torch.optim.Adam(params, lr=LR[task])
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        err = ((image(net, params, straight)[0] - target) ** 2).mean(-1)
        (err[disc].mean() + err[~disc].mean()).backward()
        with torch.no_grad():
            if mask is not None:
                for p, m in zip(params, mask):
                    p.grad *= m
            norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in params)) + 1e-12
            for p in params:
                p.grad /= norm
        opt.step()
        if mask is not None:
            with torch.no_grad():
                for p, m in zip(params, mask):
                    p.mul_(m)
    with torch.no_grad():
        img, hsv = image(net, params, False)
        R = ((img - target) ** 2).mean(-1)[disc].mean().item() / size
        F = ((img - start) ** 2).mean(-1)[~disc].mean().item() / size
        s, v = hsv[..., 1][disc], hsv[..., 2][disc]
    return dict(R=R, F=F, s_clip=((s < 0) | (s > 1)).float().mean().item(), v_clip=(v.abs() > 1).float().mean().item())


print('task       group  mode  n   R500    F500    s_clip  v_clip')
for task in ('skull', 'butterfly', 'apple'):
    net_pb, ws_pb, _ = pb_network(task, DEV)
    with torch.no_grad():
        old = net_pb.render(ws_pb, X, RES)[0]
    ck = torch.load(os.path.join(HERE, '..', 'phase0', 'runs', f'sgd_{task}.pt'), weights_only=False)
    dense = DenseCPPN(ck['groups'], ck['n_hidden'])
    fit = [k for k, v in enumerate(ck['losses'][-1, 1:]) if v < 0.01]
    dense_ws = [[W[k:k + 1].to(DEV) for W in ck['weights']] for k in fit]
    mask = [(W != 0).float() for W in ws_pb]
    groups = [('PBd', net_pb, [ws_pb], None), ('PBa', net_pb, [ws_pb], mask), ('G0', dense, dense_ws, None)]
    for group, net, nets, m in groups:
        for straight in (False, True):
            rows = [run(task, group, d, straight, net, ws, m, old) for ws in nets for d in DISCS]
            med = lambda k: st.median(r[k] for r in rows)
            print(f"{task:10s} {group:5s} {'st' if straight else 'hard':4s} {len(rows):2d}  {med('R'):.4f}  {med('F'):.4f}  "
                  f"{med('s_clip'):.2f}  {med('v_clip'):.2f}", flush=True)
