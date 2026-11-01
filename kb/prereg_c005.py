"""Writes C-005, which replaces C-004 before any G4 run: same prediction, corrected pruning design."""
import json
from pathlib import Path

from prereg_sparse_controls import SCOPE, TASKS, position

claim = dict(
    claim_id='C-005',
    statement='Sparsity alone does not close the gap: pruning a dense SGD network to the same total edge count leaves '
              'its symmetry metrics near those of the dense network.',
    scope=SCOPE + ' G4 keeps the Picbreeder total edge count, spread evenly over hidden layers with one edge per output '
                  'channel, chosen over five rounds by path strength from the dense SGD network, then fine-tuned. M1x '
                  'excludes skull, whose Picbreeder network has only 3 free maps.',
    group='G4',
    flip_boundary='Pruning may keep near-symmetric edges if the near-symmetric target rewards them.',
    prediction='For M1, M3a and M4a on all tasks, and M1x on butterfly and apple, r = (G4 - PB) / (G0 - PB) is at least '
               '0.7. r <= 0.3 on any falsifies: sparsity would then explain that gap. Between them is MIXED.',
    checks=[position(f'c005_{m.lower()}_{t}', m, t, 'G4', True) for m in ['M1', 'M3a', 'M4a'] for t in TASKS]
           + [position(f'c005_m1x_{t}', 'M1x', t, 'G4', True) for t in ['butterfly', 'apple']],
    parents=[])

Path(__file__).with_suffix('.json').write_text(json.dumps([claim], ensure_ascii=False, indent=1), 'utf-8')
