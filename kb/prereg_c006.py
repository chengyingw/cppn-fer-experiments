"""Writes C-006: is G4's Picbreeder-like score on apple a consequence of its poor fit?"""
import json
from pathlib import Path

from prereg_sparse_controls import SCOPE, position

TASKS = ['butterfly', 'apple']
claim = dict(
    claim_id='C-006',
    statement='Fit level alone does not make a network Picbreeder-like: a dense network stopped at the fit level of the '
              'pruned networks keeps symmetry metrics near those of the fully trained dense network.',
    scope=SCOPE + ' G0e is the dense architecture and seeds of G0, stopped the first time its full-image MSE is at most '
                  '0.0065, about the median MSE of fitted G4 runs on butterfly (0.0067) and apple (0.0066). Written after '
                  'seeing the G4 results, as a check on how to read C-005; skull has no fitted G4 run to match.',
    group='G0e',
    flip_boundary='Early training may favour low-frequency symmetric features on near-symmetric targets, which would '
                  'lower r most on apple.',
    prediction='For M1, M1x, M3a and M4a on butterfly and apple, r = (G0e - PB) / (G0 - PB) is at least 0.7. If r <= 0.3, '
               'fit level can explain that gap, and the C-005 falsification on apple does not show a sparsity effect.',
    checks=[position(f'c006_{m.lower()}_{t}', m, t, 'G0e', True) for m in ['M1', 'M1x', 'M3a', 'M4a'] for t in TASKS],
    parents=[])

Path(__file__).with_suffix('.json').write_text(json.dumps([claim], ensure_ascii=False, indent=1), 'utf-8')
