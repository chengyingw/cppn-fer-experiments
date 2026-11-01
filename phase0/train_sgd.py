"""Train SGD CPPNs to imitate a Picbreeder CPPN's output (FER paper, Appendix C), several seeds at once.

Same layerized architecture (same per-neuron activations), fresh LeCun init, full-batch MSE on the
256x256 RGB image, Adam with the gradient normalized to unit norm each step (as in the paper's code).
"""
import argparse
import os
import time

import numpy as np
import torch

from cppn_torch import load_genome, layerize, DenseCPPN, coord_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..', 'refs', 'fer_repo')


def pb_network(name, device='cpu'):
    """The Picbreeder CPPN as a layerized dense network: (net, weights list of (1, in, out))."""
    L = layerize(load_genome(os.path.join(REPO, 'picbreeder_genomes', f'{name}.zip')))
    net = DenseCPPN(L['groups'], L['n_hidden'])
    ws = [torch.tensor(W, dtype=torch.float32, device=device)[None] for W in L['weights']]
    return net, ws, L


def fit(net, params, X, res, target, iters, lr, train_mask=None, value_mask=None, hook=None, tag=''):
    """Full-batch Adam with per-seed unit-norm gradients on `params` (list of (S, in, out) leaf tensors).
    train_mask zeroes the gradient of edges that must not learn; value_mask keeps removed edges at zero;
    hook(it) may change both masks in place between steps. Returns the loss log (iter + per-seed)."""
    opt = torch.optim.Adam(params, lr=lr, capturable=True)

    def train_step():
        img = net.render(params, X, res)
        per_seed = ((img - target) ** 2).mean(dim=(1, 2, 3))
        per_seed.sum().backward()
        with torch.no_grad():  # unit-norm gradient per seed
            if train_mask is not None:
                for p_, m in zip(params, train_mask):
                    p_.grad *= m
            norm = torch.sqrt(sum(p_.grad.pow(2).sum(dim=(1, 2)) for p_ in params)) + 1e-12
            for p_ in params:
                p_.grad /= norm[:, None, None]
        opt.step()
        if value_mask is not None:
            with torch.no_grad():
                for p_, m in zip(params, value_mask):
                    p_.mul_(m)
        return per_seed.detach()

    # The step is ~500 tiny kernels; launch overhead dominates on Windows, so capture it as one CUDA graph.
    n_warmup = 3
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(n_warmup):
            opt.zero_grad(set_to_none=True)
            train_step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        static_loss = train_step()

    log_every = 500
    losses, t0 = [], time.time()
    for it in range(n_warmup, iters):
        if hook is not None:
            hook(it)
        graph.replay()
        if it % log_every == 0 or it == iters - 1:
            losses.append([it] + static_loss.cpu().tolist())
            if it % (log_every * 20) == 0 or it == iters - 1:
                print(f'{tag} it {it:6d}  loss {np.round(losses[-1][1:], 6)}  {time.time() - t0:6.0f}s', flush=True)
    return np.array(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--iters', type=int, default=100_000)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--res', type=int, default=256)
    p.add_argument('--relu', action='store_true', help='all-ReLU MLP of the same shape (paper App. E)')
    p.add_argument('--out', default=os.path.join(HERE, 'runs'))
    args = p.parse_args()

    dev = 'cuda'
    net_pb, ws_pb, _ = pb_network(args.name, dev)
    X = coord_inputs(args.res, dev)
    with torch.no_grad():
        target = net_pb.render(ws_pb, X, args.res)[0]

    groups = [('relu', net_pb.width)] if args.relu else net_pb.groups
    net = DenseCPPN(groups, net_pb.n_hidden)
    S = len(args.seeds)
    inits = [net.lecun_init(1, torch.Generator().manual_seed(s)) for s in args.seeds]
    params = [torch.cat([w[i] for w in inits]).to(dev).requires_grad_() for i in range(len(net.shapes))]
    t0 = time.time()
    losses = fit(net, params, X, args.res, target, args.iters, args.lr, tag=args.name)

    tag = f"{args.name}{'_relu' if args.relu else ''}"
    os.makedirs(args.out, exist_ok=True)
    torch.save(dict(groups=groups, n_hidden=net.n_hidden, seeds=args.seeds, iters=args.iters, lr=args.lr,
                    weights=[w.detach().cpu() for w in params], losses=losses),
               os.path.join(args.out, f'sgd_{tag}.pt'))
    print('saved', tag, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
