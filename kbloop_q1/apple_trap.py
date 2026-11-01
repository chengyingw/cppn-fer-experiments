"""Is apple's gray plateau caused by clipping, or by inverting colours? Exploratory; predictions committed before running.

Inverting an RGB colour turns its hue by 180 degrees. For that target, plain squared error has a stationary point in hue and
falls as saturation falls, so gray is a local optimum whatever the clipping does. Once saturation is at or below 0, chroma is
0 and neither saturation nor hue has a useful gradient.

Variants, on one disc for 500 steps, starting from fitted dense networks, on apple:
  invert         the original target, 1 - rgb
  hue60, hue120  turn hue by 60 or 120 degrees, keeping saturation and brightness
  bright         keep hue and saturation, invert brightness only
  invert_st      the original target, with a straight-through saturation clip: same image, gradient kept when clipped

Predictions, median R at step 500 with full replay:
  P1 invert stays at the gray value, within 0.02.
  P2 hue60 ends below 0.05: no antipodal hue, no trap.
  P3 bright ends below 0.05: chroma is not asked to reverse.
  P4 hue120 ends below 0.10, higher than hue60: closer to antipodal.
  P5 invert_st does not escape: R at least 0.8 of invert. Clipping is then not the root cause, because the hue target is still
     antipodal. If invert_st falls below 0.05, the clipping account stands and this one is wrong.
  P6 Each variant's saturation stays in range more often than invert, except invert_st.
"""
import os
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase0'))
from cppn_torch import DenseCPPN, coord_inputs, hsv2rgb
from train_sgd import pb_network

DEV, RES, STEPS, LR = 'cuda', 64, 500, 3e-4
DISCS = [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)]
net_pb, ws_pb, _ = pb_network('apple', DEV)
X = coord_inputs(RES, DEV)
xx, yy = X[:, 0].reshape(RES, RES), X[:, 1].reshape(RES, RES)
with torch.no_grad():
    hsv0 = net_pb.forward(ws_pb, X)[0].reshape(RES, RES, 3)
    old = hsv2rgb(hsv0[..., 0], hsv0[..., 1], hsv0[..., 2])
h0, s0, v0 = hsv0[..., 0], hsv0[..., 1], hsv0[..., 2]
TARGETS = {'invert': 1 - old, 'invert_st': 1 - old, 'hue60': hsv2rgb(h0 + 1 / 6, s0, v0), 'hue120': hsv2rgb(h0 + 1 / 3, s0, v0),
           'bright': hsv2rgb(h0, s0, 1 - v0.abs().clamp(0, 1))}
ck = torch.load(os.path.join(HERE, '..', 'phase0', 'runs', 'sgd_apple.pt'), weights_only=False)
net = DenseCPPN(ck['groups'], ck['n_hidden'])


def render(params, st_clip):
    hsv = net.forward(params, X)[0].reshape(RES, RES, 3)
    s = hsv[..., 1]
    if st_clip:
        s = s + (s.clamp(0, 1) - s).detach()
    return hsv2rgb(hsv[..., 0], s, hsv[..., 2]), hsv


def run(variant, seed, disc_index, replay):
    params = [W[seed:seed + 1].to(DEV).clone().requires_grad_() for W in ck['weights']]
    cx, cy = DISCS[disc_index]
    disc = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < 0.2
    target = old.clone()
    target[disc] = TARGETS[variant][disc]
    size = ((old - target) ** 2).mean(-1)[disc].mean().item()
    t = target[disc]
    gray = ((t - t.mean(-1, keepdim=True)) ** 2).mean().item() / size
    opt = torch.optim.Adam(params, lr=LR)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        img, _ = render(params, variant == 'invert_st')
        err = ((img - target) ** 2).mean(-1)
        (err[disc].mean() + (err[~disc].mean() if replay else 0)).backward()
        with torch.no_grad():
            norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in params)) + 1e-12
            for p in params:
                p.grad /= norm
        opt.step()
    with torch.no_grad():
        img, hsv = render(params, variant == 'invert_st')
        R = ((img - target) ** 2).mean(-1)[disc].mean().item() / size
        s = hsv[..., 1][disc]
        chroma = (img.max(-1)[0] - img.min(-1)[0])[disc].mean().item()
    return dict(R=R, gray=gray, s_neg=(s < 0).float().mean().item(), s_high=(s > 1).float().mean().item(), chroma=chroma, size=size)


print('variant     replay  R500_med  gray_R_med  s<0   s>1   chroma_out  size')
for replay in (True, False):
    for variant in TARGETS:
        rows = [run(variant, seed, disc, replay) for seed in (0, 1, 2) for disc in (0, 2)]
        m = lambda k: st.median(r[k] for r in rows)
        print(f"{variant:11s} {int(replay)}  {m('R'):.3f}  {m('gray'):.3f}  {m('s_neg'):.2f}  {m('s_high'):.2f}  {m('chroma'):.3f}  {m('size'):.3f}", flush=True)
