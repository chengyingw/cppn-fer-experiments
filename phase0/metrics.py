"""Candidate quantitative FER/UFR metrics for a CPPN (one parameter set).

Predicted direction for a more unified/factored (Picbreeder-like) network is given per metric.
Symmetry is left-right (flip along x = image columns), the dominant regularity of skull/butterfly.
"""
import numpy as np
import torch

from cppn_torch import coord_inputs

CONST_STD = 1e-3      # feature maps flatter than this are ignored (unused slots)
MIRROR_CORR = 0.9     # |corr(f, flip f)| above this counts as mirror-structured
DUP_CORR = 0.99       # |corr| above this counts as a duplicate feature map
VALID_CHANGE = 0.01   # perturbations changing the image less than this (mean |dRGB|) are ignored


def _flip(maps, res):
    """maps (..., res*res) -> left-right mirrored."""
    return maps.reshape(*maps.shape[:-1], res, res).flip(-1).reshape(maps.shape)


def _zscore(maps):
    m = maps - maps.mean(-1, keepdim=True)
    return m / (m.norm(dim=-1, keepdim=True) + 1e-12)


@torch.no_grad()
def feature_maps(net, ws, res=128):
    """All hidden-layer feature maps: list over layers of (width, res*res) tensors."""
    _, feats = net.forward(ws, coord_inputs(res, ws[0].device), return_features=True)
    return [f[0].T for f in feats[1:]]


@torch.no_grad()
def internal_metrics(net, ws, res=128):
    """M1: fraction of distinct feature maps that are mirror-(anti)symmetric  [UFR: higher]
       M2: number of distinct feature maps (automates the paper's 'novel' count) [UFR: lower]
       plus the per-layer mean |mirror corr| profile (non-constant neurons)."""
    layers = feature_maps(net, ws, res)
    per_layer, maps = [], []
    for f in layers:
        keep = f.std(-1) > CONST_STD
        f = f[keep]
        if len(f) == 0:
            per_layer.append(np.nan)
            continue
        c = (_zscore(f) * _zscore(_flip(f, res))).sum(-1)
        per_layer.append(c.abs().mean().item())
        maps.append(f)
    maps = torch.cat(maps)
    z = _zscore(maps)
    C = (z @ z.T).abs()
    kept = []
    for i in range(len(maps)):  # greedy de-duplication in layer order
        if not kept or C[i, kept].max() < DUP_CORR:
            kept.append(i)
    distinct = maps[kept]
    mc = (_zscore(distinct) * _zscore(_flip(distinct, res))).sum(-1).abs()
    return dict(M1_mirror_frac=(mc > MIRROR_CORR).float().mean().item(),
                M2_distinct=len(kept),
                n_nonconst=len(maps),
                layer_mirror=np.array(per_layer))


ODD_ACTS = ('cache', 'identity', 'sin', 'tanh', 'sigmoid')  # f(-z) = -f(z)
EVEN_ACTS = ('gaussian', 'cos')                              # f(-z) = f(z)


def mask_parity(net, masks):
    """Left-right parity each hidden neuron has for ANY weights on this wiring. List over hidden layers of
    arrays: 1 even (mirror-symmetric), -1 odd (mirror-antisymmetric), 0 free (the wiring implies nothing).
    Inputs: x is odd; y, d and b are even."""
    acts = [a for a, c in net.groups for _ in range(c)]
    state, out = np.array([-1, 1, 1, 1]), []
    for M in masks[:-1]:
        M = np.asarray(M) != 0
        cur = np.ones(M.shape[1], dtype=int)
        for j in range(M.shape[1]):
            s = state[M[:, j]]
            if (s == 0).any() or ((s == 1).any() and (s == -1).any()):
                cur[j] = 0
            elif (s == -1).any():
                cur[j] = 1 if acts[j] in EVEN_ACTS else -1 if acts[j] in ODD_ACTS else 0
        out.append(cur)
        state = cur
    return out


@torch.no_grad()
def structural_metrics(net, ws, res=128):
    """Separates symmetry the wiring forces from symmetry the weights achieve, over distinct non-constant
    hidden maps (copies of the x, y, d inputs excluded):
       forced_frac: share whose parity the wiring fixes for any weights
       M1x: among the rest, share that are mirror-(anti)symmetric anyway [UFR: higher]"""
    device = ws[0].device
    parity = mask_parity(net, [W[0].cpu().numpy() for W in ws])
    maps, pars = [coord_inputs(res, device)[:, :3].T], [np.ones(3, dtype=int)]
    for f, p in zip(feature_maps(net, ws, res), parity):
        keep = f.std(-1) > CONST_STD
        maps.append(f[keep])
        pars.append(p[keep.cpu().numpy()])
    maps, pars = torch.cat(maps), np.concatenate(pars)
    z = _zscore(maps)
    C = (z @ z.T).abs()
    kept = []
    for i in range(len(maps)):
        if not kept or C[i, kept].max() < DUP_CORR:
            kept.append(i)
    kept = [i for i in kept if i >= 3]
    d = maps[kept]
    mc = (_zscore(d) * _zscore(_flip(d, res))).sum(-1).abs().cpu().numpy()
    free = pars[kept] == 0
    return dict(forced_frac=float(1 - free.mean()), n_distinct=len(kept), n_free=int(free.sum()),
                M1x=float((mc[free] > MIRROR_CORR).mean()) if free.any() else float('nan'))


AMPS = (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0)


def _render_perturbed(net, ws, specs, X, res, batch=256):
    """specs: list of (layer, row, direction over the row's columns (tensor), amplitude).
    Deltas are materialized one batch at a time. Returns (P, res, res, 3)."""
    out = []
    for i in range(0, len(specs), batch):
        chunk = specs[i:i + batch]
        deltas = [torch.zeros(len(chunk), a, b, device=X.device) for a, b in net.shapes]
        for k, (l, r, v, a) in enumerate(chunk):
            deltas[l][k, r] += a * v
        out.append(net.render([W + d for W, d in zip(ws, deltas)], X, res))
    return torch.cat(out)


def single_weight_specs(net, n, rng, device, amps=AMPS, ws=None):
    """Single-weight perturbations (paper Fig. 6): n weights sampled uniformly over all parameters, or
    only over the existing edges of `ws` (nudging a zero weight would add an edge to the wiring)."""
    specs = []
    bounds = np.cumsum([0] + net.sizes)
    pool = net.n_params if ws is None else np.flatnonzero(torch.cat([W[0].flatten() for W in ws]).cpu().numpy())
    size = n if ws is None else min(n, len(pool))
    for flat in rng.choice(pool, size=size, replace=False):
        l = int(np.searchsorted(bounds, flat, side='right') - 1)
        r, c = divmod(int(flat - bounds[l]), net.shapes[l][1])
        v = torch.zeros(net.shapes[l][1], device=device)
        v[c] = 1.0
        specs += [(l, r, v, a) for a in amps]
    return specs


def random_direction_specs(net, n, rng, device, amps=AMPS, ws=None):
    """Basis-agnostic perturbations (paper App. F): random layer, random input row, random unit
    direction over that row's outgoing weights (restricted to existing edges if `ws` is given)."""
    specs = []
    for _ in range(n):
        l = int(rng.integers(len(net.shapes)))
        if ws is None:
            r, m = int(rng.integers(net.shapes[l][0])), 1.0
        else:
            rows = (ws[l][0] != 0).any(1).nonzero().flatten()
            r = int(rows[int(rng.integers(len(rows)))])
            m = (ws[l][0, r] != 0).float()
        v = torch.as_tensor(rng.standard_normal(net.shapes[l][1]), dtype=torch.float32, device=device) * m
        specs += [(l, r, v / v.norm(), a) for a in amps]
    return specs


def _hf_fraction(d, res, cutoff):
    """Share of spectral power of the change d (P, res, res) above radial frequency `cutoff`."""
    F = torch.fft.fft2(d - d.mean((-1, -2), keepdim=True)).abs() ** 2
    k = torch.fft.fftfreq(res, d=1.0 / res, device=d.device)
    rad = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    return (F * (rad > cutoff)).sum((-1, -2)) / (F.sum((-1, -2)) + 1e-12)


@torch.no_grad()
def perturbation_metrics(net, ws, specs, res=64, hf_cutoff=8):
    """For each perturbation: dI = I' - I.
       asym = ||antisymmetric part of dI||^2 / ||dI||^2  (0 = symmetry kept, ~0.5 = random) [UFR: lower]
       hf   = share of dI's spectral power above hf_cutoff cycles/image                      [UFR: lower]
       mag  = mean |dI|, so scores can be compared at matched change size.
       Only perturbations that visibly change the image (mag > VALID_CHANGE) are scored."""
    X = coord_inputs(res, ws[0].device)
    base = net.render(ws, X, res)[0]
    imgs = _render_perturbed(net, ws, specs, X, res)
    d = imgs - base
    mag = d.abs().mean((1, 2, 3))
    valid = mag > VALID_CHANGE
    dv = d[valid]
    anti = 0.5 * (dv - dv.flip(2))
    asym = anti.pow(2).sum((1, 2, 3)) / dv.pow(2).sum((1, 2, 3))
    hf = _hf_fraction(dv.mean(-1), res, hf_cutoff)
    return dict(asym=asym.cpu().numpy(), hf=hf.cpu().numpy(), mag=mag[valid].cpu().numpy(),
                valid_frac=valid.float().mean().item(),
                median_change=mag[valid].median().item() if valid.any() else np.nan, imgs=imgs, base=base)


def matched_median(p, key, lo=0.02, hi=0.10):
    """Median of p[key] over perturbations whose image change lies in [lo, hi] (removes the
    'Picbreeder weights are larger, so a fixed dw is a smaller relative kick' confound)."""
    sel = (p['mag'] >= lo) & (p['mag'] <= hi)
    return float(np.median(p[key][sel])) if sel.sum() >= 10 else np.nan


@torch.no_grad()
def base_symmetry(net, ws, res=128):
    """Asymmetric share of the base image itself (same definition as `asym`, about the image mean)."""
    img = net.render(ws, coord_inputs(res, ws[0].device), res)[0]
    c = img - img.mean((0, 1))
    return (0.5 * (c - c.flip(1))).pow(2).sum().item() / c.pow(2).sum().item()
