"""Why do apple's local edits stop at a fixed error once most of the image is supervised? Exploratory, not pre-registered.

A fitted dense network learns one disc for 500 steps, with and without full replay. Error left in the disc is then split by
whether the raw output is clipped by the HSV mapping there, which zeroes its gradient: saturation outside [0, 1], or
|brightness| above 1. Also tests the target itself: can the mapping reach it, given unclipped outputs?
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase0'))
from cppn_torch import DenseCPPN, coord_inputs, hsv2rgb
from train_sgd import pb_network

DEV = 'cuda'
RES = 64
DISCS = [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)]
RADIUS = 0.2
LR = {'skull': 3e-3, 'butterfly': 3e-3, 'apple': 3e-4}


def split(hsv):
    s, v = hsv[..., 1], hsv[..., 2]
    return (s < 0) | (s > 1), v.abs() > 1


def render(hsv):
    return hsv2rgb(hsv[..., 0], hsv[..., 1], hsv[..., 2]).reshape(-1, RES, RES, 3)


def run(task, seed_index, disc_index, replay, steps=500):
    net_pb, ws_pb, _ = pb_network(task, DEV)
    X = coord_inputs(RES, DEV)
    with torch.no_grad():
        old = net_pb.render(ws_pb, X, RES)[0]
    ck = torch.load(os.path.join(HERE, '..', 'phase0', 'runs', f'sgd_{task}.pt'), weights_only=False)
    net = DenseCPPN(ck['groups'], ck['n_hidden'])
    params = [W[seed_index:seed_index + 1].to(DEV).clone().requires_grad_() for W in ck['weights']]
    xx, yy = X[:, 0].reshape(RES, RES), X[:, 1].reshape(RES, RES)
    cx, cy = DISCS[disc_index]
    disc = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < RADIUS
    target = old.clone()
    target[disc] = 1 - old[disc]
    size = ((old - target) ** 2).mean(-1)[disc].mean().item()
    opt = torch.optim.Adam(params, lr=LR[task])
    trace = []
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        img = net.render(params, X, RES)[0]
        err = ((img - target) ** 2).mean(-1)
        loss = err[disc].mean() + (err[~disc].mean() if replay else 0)
        loss.backward()
        with torch.no_grad():
            norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in params)) + 1e-12
            for p in params:
                p.grad /= norm
        opt.step()
        if it in (0, 99, 499):
            trace.append(round(err[disc].mean().item() / size, 3))
    with torch.no_grad():
        hsv = net.forward(params, X)[0].reshape(RES, RES, 3)
        img = net.render(params, X, RES)[0]
        err = ((img - target) ** 2).mean(-1)
        s_clip, v_clip = split(hsv)
        d = disc
        total = err[d].sum().item()
        share = lambda m: round(err[d & m].sum().item() / max(total, 1e-12), 3)
        frac = lambda m: round(m[d].float().mean().item(), 3)
        # Largest per-pixel errors, in the disc, and what the target asks for there
        gray = (target.max(-1)[0] - target.min(-1)[0])[d].mean().item()
        print(f'{task} seed{seed_index} disc{disc_index} replay={int(replay)} R(1,100,500)={trace} size={size:.3f} '
              f'target_chroma={gray:.3f} | s_clipped {frac(s_clip)} carries {share(s_clip)} of error; '
              f'v_clipped {frac(v_clip)} carries {share(v_clip)}; any {frac(s_clip | v_clip)} carries {share(s_clip | v_clip)}')
    return hsv, err, disc


for task, cases in [('apple', [(0, 0), (1, 0), (0, 2), (0, 1)]), ('butterfly', [(1, 0)])]:
    for seed_index, disc_index in cases:
        for replay in (False, True):
            run(task, seed_index, disc_index, replay)
