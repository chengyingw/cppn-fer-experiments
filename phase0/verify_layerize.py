"""Check genome parsing + layerization against the FER repo's published images and architectures."""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from cppn_torch import load_genome, neat_forward, layerize, arch_string, DenseCPPN, coord_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..', 'refs', 'fer_repo')
EXPECTED = {  # from the repo's fer.ipynb train_sgd.py commands
    'skull': '12;cache:15,gaussian:4,identity:2,sin:1',
    'butterfly': '16;cache:17,gaussian:4,sigmoid:1,sin:1',
    'apple': '33;cache:28,gaussian:3,sigmoid:5,sin:2',
}

for name in EXPECTED:
    g = load_genome(os.path.join(REPO, 'picbreeder_genomes', f'{name}.zip'))
    ref = plt.imread(os.path.join(REPO, 'data', f'picbreeder_{name}', 'img.png'))[..., :3]
    img_graph, _ = neat_forward(g, 256)

    L = layerize(g)
    net = DenseCPPN(L['groups'], L['n_hidden'])
    ws = [torch.tensor(W, dtype=torch.float32)[None] for W in L['weights']]
    img_dense = net.render(ws, coord_inputs(256), 256)[0]

    err_ref = np.abs(img_graph.numpy() - ref).max()
    err_dense = (img_graph - img_dense).abs().max().item()
    nz = sum(int((np.abs(W) > 0).sum()) for W in L['weights'])
    print(f"{name:9s} arch={arch_string(L):42s} matches_repo={arch_string(L) == EXPECTED[name]} "
          f"params={net.n_params} nonzero={nz} | graph-vs-repo-png max|err|={err_ref:.4f} "
          f"dense-vs-graph max|err|={err_dense:.2e}")
