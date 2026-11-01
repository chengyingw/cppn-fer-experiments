"""Calibration on a target outside the probe: one filled disc at the centre, so it says nothing about compounding.

Usage: calibrate.py <cppn_gpu dir> <assets dir>
"""
import sys
import time

import torch

sys.path.insert(0, sys.argv[1])
from cppn_torch import DenseCPPN, coord_inputs

DEV = "cuda"
ck = torch.load(sys.argv[2] + "/nets/b0/sgd_skull.pt", weights_only=False, map_location="cpu")
print("groups", ck["groups"], "hidden", ck["n_hidden"])
for shrink, hidden in ((4, 3), (2, 3)):
    net = DenseCPPN([(a, max(1, c // shrink)) for a, c in ck["groups"]], hidden)
    X = coord_inputs(32, DEV)
    print("width", net.width, "params", net.n_params)
    disc = (torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) < 0.4).float()
    for name, inp in (("orig", X), ("shuf", X[torch.randperm(len(X), generator=torch.Generator().manual_seed(1)).to(DEV)])):
        for lr in (0.003, 0.01, 0.03):
            params = [W.requires_grad_() for W in net.lecun_init(12, torch.Generator().manual_seed(0), DEV)]
            opt = torch.optim.Adam(params, lr=lr)
            trace, t0 = {}, time.time()
            for s in range(201):
                opt.zero_grad(set_to_none=True)
                err = (net.forward(params, inp)[..., 2] - disc) ** 2
                R = (err * disc).sum(1) / disc.sum()
                bg = (err * (1 - disc)).sum(1) / (1 - disc).sum()
                if s in (25, 50, 100, 200):
                    trace[s] = round(R.median().item(), 3), round(bg.median().item(), 3)
                (R + bg).sum().backward()
                with torch.no_grad():
                    norm = torch.sqrt(sum(p.grad.pow(2).sum(dim=(1, 2)) for p in params)) + 1e-12
                    for p in params:
                        p.grad /= norm[:, None, None]
                opt.step()
            print(name, "lr", lr, "(R, background) at step", trace, "ms per step", round((time.time() - t0) * 5, 1), flush=True)
