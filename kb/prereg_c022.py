"""Writes C-022, a post-hoc description of disturbance outside local edits seen in the C-019 to C-021 remeasurement."""
import json
from pathlib import Path

from prereg_v2 import SCOPE, ratio

checks = [ratio(f'c022_{t}_{c}_lr{lr}', 'F2000', t, c, lr) for t in ['skull', 'butterfly', 'apple']
          for c in ['local_sym', 'local_asym'] for lr in [1.0, 0.1]]
claim = dict(
    claim_id='C-022',
    statement='After 2000 steps on a local edit, a Picbreeder start has changed its output outside the edit at least twice '
              'as much as a fully fitted dense network, while also leaving more or similar error inside it.',
    scope=SCOPE + ' F is the change from the network\'s own starting output, relative to E0, read at a fixed step.',
    group='PBd',
    flip_boundary='Unknown. Only these edits, two learning rates and one Picbreeder network per task were measured.',
    prediction='Post-hoc description of results already seen, not a prediction. It is kept as a regression lock and is no '
               'independent evidence. Ratio >= 2 passes, ratio <= 1 fails, between them is MIXED.',
    checks=checks,
    parents=[])

Path(__file__).with_suffix('.json').write_text(json.dumps([claim], ensure_ascii=False, indent=1), 'utf-8')
