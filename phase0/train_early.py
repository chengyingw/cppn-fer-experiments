"""G0e: the dense SGD setup of train_sgd.py, each seed saved the first time it reaches a target MSE."""
import argparse
import os

import numpy as np
import torch

from cppn_torch import DenseCPPN, coord_inputs
from train_control import LR
from train_sgd import HERE, pb_network


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--target', type=float, default=0.0065)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--max_iters', type=int, default=20_000)
    p.add_argument('--res', type=int, default=128)
    p.add_argument('--out', default=os.path.join(HERE, 'runs'))
    args = p.parse_args()

    dev = 'cuda'
    net_pb, ws_pb, _ = pb_network(args.name, dev)
    X = coord_inputs(args.res, dev)
    with torch.no_grad():
        target = net_pb.render(ws_pb, X, args.res)[0]
    net = DenseCPPN(net_pb.groups, net_pb.n_hidden)
    inits = [net.lecun_init(1, torch.Generator().manual_seed(s)) for s in args.seeds]
    params = [torch.cat([w[i] for w in inits]).to(dev).requires_grad_() for i in range(len(net.shapes))]
    opt = torch.optim.Adam(params, lr=LR[args.name])
    saved, stopped = [W.detach().clone() for W in params], {}
    log = []
    for it in range(args.max_iters):
        opt.zero_grad(set_to_none=True)
        per_seed = ((net.render(params, X, args.res) - target) ** 2).mean(dim=(1, 2, 3))
        per_seed.sum().backward()
        with torch.no_grad():
            for s, v in enumerate(per_seed.tolist()):  # these weights produced v, before the update below
                if s not in stopped and v <= args.target:
                    stopped[s] = it
                    for W, p_ in zip(saved, params):
                        W[s] = p_[s]
            norm = torch.sqrt(sum(p_.grad.pow(2).sum(dim=(1, 2)) for p_ in params)) + 1e-12
            for p_ in params:
                p_.grad /= norm[:, None, None]
        if len(stopped) == len(args.seeds):
            break
        opt.step()
        if it % 500 == 0:
            log.append([it] + per_seed.tolist())
    print(args.name, 'stopped at', {args.seeds[s]: i for s, i in stopped.items()}, 'of', len(args.seeds), 'seeds', flush=True)
    done = sorted(stopped)  # a seed that never reached the target is not an early-stopped network
    torch.save(dict(groups=net.groups, n_hidden=net.n_hidden, seeds=[args.seeds[s] for s in done], iters=args.max_iters,
                    lr=LR[args.name], group='G0e', stopped={args.seeds[s]: stopped[s] for s in done},
                    weights=[W[done].cpu() for W in saved], losses=np.array(log)),
               os.path.join(args.out, f'control_G0e_{args.name}.pt'))


if __name__ == '__main__':
    main()
