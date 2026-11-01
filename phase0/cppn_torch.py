"""Picbreeder genomes -> NEAT graph -> layerized dense CPPN, all in PyTorch.

Re-implements the conventions of the FER paper's code (github.com/akarshkumar0101/fer)
without loading its pickles or running its code:
  * inputs x (column axis), y (row axis), d = 1.4*sqrt(x^2+y^2), b = 1; no biases
  * sigmoid(z) = 2*sigma(z)-1, gaussian(z) = 2*exp(-z^2)-1
  * rgb = hsv2rgb((h+1) mod 1, clip(s,0,1), clip(|v|,0,1))
  * layerization: longest-path layers, skip connections carried by identity ("cache") neurons,
    each layer laid out as activation groups sorted by name.
"""
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch

ACTS = {
    'cache': lambda z: z,
    'identity': lambda z: z,
    'sin': torch.sin,
    'cos': torch.cos,
    'tanh': torch.tanh,
    'sigmoid': lambda z: torch.sigmoid(z) * 2 - 1,
    'gaussian': lambda z: torch.exp(-z ** 2) * 2 - 1,
    'relu': torch.relu,
}


# ----------------------------------------------------------------------------- genome parsing
def load_genome(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert len(names) == 1
        root = ET.fromstring(z.read(names[0]).decode('utf-8'))
    genome = root if root.tag == 'genome' else root.find('genome')
    key = lambda e: f"{e.get('branch')}_{e.get('id')}"

    nodes, links = [], []
    for n in genome.find('nodes').findall('node'):
        act = n.find('activation').text.strip()
        nodes.append(dict(id=key(n.find('marking')), label=n.get('label', ''),
                          act=act[:-3] if act.endswith('(x)') else act))
    for l in genome.find('links').findall('link'):
        links.append(dict(src=key(l.find('source')), dst=key(l.find('target')),
                          w=float(l.find('weight').text)))

    labels = {n['label']: n['id'] for n in nodes if n['label']}
    if 'ink' in labels:  # grayscale genome: ink -> brightness, constant-zero hue/saturation
        labels['brightness'] = labels.pop('ink')
        for name in ['hue', 'saturation']:
            nid = f'extra_{name}'
            nodes.append(dict(id=nid, label=name, act='identity'))
            links.append(dict(src=labels['brightness'], dst=nid, w=0.0))
            labels[name] = nid
    special = dict(x=labels['x'], y=labels['y'], d=labels['d'], b=labels['bias'],
                   h=labels['hue'], s=labels['saturation'], v=labels['brightness'])
    return dict(nodes=nodes, links=links, special=special)


def coord_inputs(res, device='cpu'):
    """(res*res, 4) inputs [x, y, d, b]; x varies along columns, y along rows (top = -1)."""
    xs = torch.linspace(-1, 1, res, device=device)
    Y, X = torch.meshgrid(xs, xs, indexing='ij')
    D = torch.sqrt(X ** 2 + Y ** 2) * 1.4
    return torch.stack([X, Y, D, torch.ones_like(X)], -1).reshape(-1, 4)


def hsv2rgb(h, s, v):
    h = torch.remainder(h + 1, 1.0) * 360.0
    s, v = s.clamp(0, 1), v.abs().clamp(0, 1)
    c = v * s
    x = c * (1 - torch.abs(torch.remainder(h / 60, 2) - 1))
    m = v - c
    z = torch.zeros_like(h)
    sectors = [(c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c), (c, z, x)]
    r = g = b = 0
    for k, (r_, g_, b_) in enumerate(sectors):
        mask = ((h >= 60 * k) & (h < 60 * (k + 1))).to(h.dtype)
        r, g, b = r + r_ * mask, g + g_ * mask, b + b_ * mask
    return torch.stack([r + m, g + m, b + m], -1).clamp(0, 1)


# ----------------------------------------------------------------------------- NEAT graph evaluation
def neat_forward(g, res=256):
    """Reference evaluation directly on the NEAT graph (recursive, like the paper's code)."""
    act = {n['id']: n['act'] for n in g['nodes']}
    ins = defaultdict(list)
    for l in g['links']:
        ins[l['dst']].append((l['src'], l['w']))
    sp = g['special']
    inp = coord_inputs(res).double()
    val = {sp['x']: inp[:, 0], sp['y']: inp[:, 1], sp['d']: inp[:, 2], sp['b']: inp[:, 3]}

    def get(n):
        if n not in val:
            z = torch.zeros(res * res, dtype=torch.float64)
            for s, w in ins[n]:
                z = z + w * get(s)
            val[n] = ACTS[act[n]](z)
        return val[n]

    h, s, v = get(sp['h']), get(sp['s']), get(sp['v'])
    return hsv2rgb(h, s, v).reshape(res, res, 3).float(), val


# ----------------------------------------------------------------------------- layerization
def layerize(g):
    """Port the NEAT graph to a dense MLP computing exactly the same function.

    Returns dict(groups=[(act, count), ...], n_hidden, width, weights=[W_1..W_L, W_out] (numpy),
    slots=[per hidden layer: {slot: (node_id, novel)}]).
    """
    act = {n['id']: n['act'] for n in g['nodes']}
    sp = g['special']
    inputs = [sp['x'], sp['y'], sp['d'], sp['b']]
    outputs = [sp['h'], sp['s'], sp['v']]
    ins, outs = defaultdict(list), defaultdict(list)
    for l in g['links']:
        ins[l['dst']].append((l['src'], l['w']))
        outs[l['src']].append(l['dst'])

    reach, stack = set(inputs), list(outputs)
    while stack:
        n = stack.pop()
        if n in reach:
            continue
        reach.add(n)
        stack.extend(s for s, _ in ins[n])

    layer = {n: 0 for n in inputs}

    def get_layer(n):
        if n not in layer:
            layer[n] = 1 + max((get_layer(s) for s, _ in ins[n]), default=0)
        return layer[n]

    for n in reach:
        get_layer(n)
    n_layers = max(layer.values()) + 1  # including the input layer 0

    contents = [[] for _ in range(n_layers)]
    for n in sorted(reach, key=lambda n: (layer[n], n)):
        start = layer[n]
        if n in outputs:
            end = n_layers
        else:
            targets = [layer[t] for t in outs[n] if t in reach]
            end = max(targets) if targets else start + 1
        for i in range(start, end):
            contents[i].append((n, i == start and n not in inputs))
    contents[0] = [(n, False) for n in inputs]

    counts = defaultdict(int)
    for lay in contents:
        c = defaultdict(int)
        for n, novel in lay:
            c[act[n] if novel else 'cache'] += 1
        for k, v in c.items():
            counts[k] = max(counts[k], v)
    groups = order_groups(counts.items())
    width = sum(v for _, v in groups)
    offset = dict(zip([k for k, _ in groups], np.cumsum([0] + [v for _, v in groups[:-1]])))

    def positions(lay):
        pos, used = {}, defaultdict(int)
        for n, novel in lay:
            a = act[n] if novel else 'cache'
            pos[n] = offset[a] + used[a]
            used[a] += 1
        return pos

    pos = [None] + [positions(lay) for lay in contents[1:]]
    pos[0] = {n: i for i, n in enumerate(inputs)}

    weights = []
    for i in range(1, n_layers):
        W = np.zeros((4 if i == 1 else width, width))
        for n, novel in contents[i]:
            if novel:
                for s, w in ins[n]:
                    W[pos[i - 1][s], pos[i][n]] += w
            else:
                W[pos[i - 1][n], pos[i][n]] = 1.0
        weights.append(W)
    W_out = np.zeros((width, 3))
    for j, n in enumerate(outputs):
        W_out[pos[-1][n], j] = 1.0
    weights.append(W_out)

    slots = [{pos[i][n]: (n, novel) for n, novel in contents[i]} for i in range(1, n_layers)]
    return dict(groups=groups, n_hidden=n_layers - 1, width=width, weights=weights, slots=slots)


LINEAR = ('cache', 'identity')


def order_groups(groups):
    """Linear groups first, then nonlinear ones by name; lets activate() touch only the nonlinear tail.
    (A permutation of hidden units, so the computed function is unchanged.)"""
    return sorted(groups, key=lambda g: (g[0] not in LINEAR, g[0]))


def arch_string(L):
    """Same format as the FER repo (groups sorted by name)."""
    return f"{L['n_hidden']};" + ','.join(f'{a}:{c}' for a, c in sorted(L['groups']))


# ----------------------------------------------------------------------------- dense CPPN (batched over parameter sets)
class DenseCPPN:
    """Dense CPPN with per-neuron activation groups. Weights are lists of (B, in, out) tensors,
    so B parameter sets (seeds, perturbations) run in one batched forward pass."""

    def __init__(self, groups, n_hidden):
        self.groups, self.n_hidden = order_groups(groups), n_hidden
        self.width = sum(c for _, c in self.groups)
        self.n_linear = sum(c for a, c in self.groups if a in LINEAR)
        self.shapes = [(4, self.width)] + [(self.width, self.width)] * (n_hidden - 1) + [(self.width, 3)]
        self.sizes = [a * b for a, b in self.shapes]
        self.n_params = sum(self.sizes)

    def activate(self, z):
        if self.n_linear == self.width:
            return z
        parts, i = [z[..., :self.n_linear]], self.n_linear
        for a, c in self.groups:
            if a not in LINEAR:
                parts.append(ACTS[a](z[..., i:i + c]))
                i += c
        return torch.cat(parts, -1)

    def forward(self, weights, inputs, return_features=False, chunks=64):
        """weights: list of (B, in, out); inputs: (N, 4). Returns hsv (B, N, 3) [, features (B, N, w)].

        Pixels are split into `chunks` groups so the backward weight-gradient (a reduction over all
        pixels into a tiny w x w matrix) runs as many parallel matmuls instead of one starved one."""
        B, N = weights[0].shape[0], inputs.shape[0]
        C = chunks if N % chunks == 0 else 1
        h = inputs.reshape(1, C, N // C, -1).expand(B, -1, -1, -1)
        feats = [h]
        for W in weights[:-1]:
            h = self.activate(torch.matmul(h, W[:, None]))
            if return_features:
                feats.append(h)
        out = torch.matmul(h, weights[-1][:, None]).reshape(B, N, 3)
        if return_features:
            return out, [f.reshape(B, N, -1) for f in feats]
        return out

    def render(self, weights, inputs, res):
        hsv = self.forward(weights, inputs)
        return hsv2rgb(hsv[..., 0], hsv[..., 1], hsv[..., 2]).reshape(-1, res, res, 3)

    # flat-vector helpers (one parameter set per row)
    def unflatten(self, flat):
        flat = flat if flat.dim() == 2 else flat.unsqueeze(0)
        out, i = [], 0
        for (a, b), n in zip(self.shapes, self.sizes):
            out.append(flat[:, i:i + n].reshape(-1, a, b))
            i += n
        return out

    def flatten(self, weights):
        return torch.cat([W.reshape(W.shape[0], -1) for W in weights], 1)

    def lecun_init(self, n_sets, generator, device='cpu'):
        """Truncated-normal LeCun init (matches flax's default nn.Dense kernel init)."""
        ws = []
        for a, b in self.shapes:
            std = (1.0 / a) ** 0.5 / 0.87962566103423978
            w = torch.empty(n_sets, a, b)
            torch.nn.init.trunc_normal_(w, std=std, a=-2 * std, b=2 * std, generator=generator)
            ws.append(w.to(device))
        return ws
