"""Writes the pre-registered claims for the sparse SGD controls to prereg_sparse_controls.json."""
import json
from pathlib import Path

TASKS = ['skull', 'butterfly', 'apple']
SCOPE = ('128x128 image regression on the three Picbreeder CPPNs, 50k iterations, Adam with unit-norm gradients, '
         'fit = MSE < 0.01. Perturbations act on existing edges only; matched window of mean |dRGB| in [0.02, 0.10].')
WIRING = (' Wiring facts known before registration: the Picbreeder masks force the parity of 82%/59%/38% of distinct '
          'maps (skull/butterfly/apple), for any weights.')


def position(cid, metric, task, group, high_is_pass):
    bands = ({'pass': dict(op='>=', value=0.7), 'fail': dict(op='<=', value=0.3)} if high_is_pass
             else {'pass': dict(op='<=', value=0.3), 'fail': dict(op='>=', value=0.7)})
    return dict(check_id=cid, type='position', metric=metric, where=dict(task=task), low_group='PB', high_group='G0',
                test_group=group, min_n=2, **bands)


claims = [
    dict(claim_id='C-001',
         statement='SGD does not reach the skull when only the NEAT links of the raw architecture can learn.',
         scope=SCOPE, group='G1a',
         prediction='At most 1 of 3 G1a seeds on skull reaches MSE < 0.01, replicating FER paper Fig. 12.',
         flip_boundary='Unknown; the fit threshold is looser than dense SGD (about 5e-4), so a pass here is a weak fit.',
         checks=[dict(check_id='c001_skull_fit', type='count', where=dict(task='skull', group='G1a', fit=True),
                      where_total=dict(task='skull', group='G1a'), min_total=3, op='<=', value=1)],
         parents=[]),
    dict(claim_id='C-002',
         statement='Weights that SGD finds on Picbreeder wiring do not reproduce the symmetry Picbreeder weights achieve '
                   'on neurons whose parity the wiring leaves free.',
         scope=SCOPE + WIRING + ' Skull is excluded: its Picbreeder network has only 3 free maps.', group='G1b',
         prediction='For M1x, r = (G1b - PB) / (G0 - PB) is at least 0.7 on butterfly and apple. r <= 0.3 falsifies: the '
                    'wiring would then be sufficient. Between them is MIXED.',
         flip_boundary='Low redundancy on a sparse network may push SGD toward reuse, which would lower r.',
         checks=[position(f'c002_m1x_{t}', 'M1x', t, 'G1b', True) for t in ['butterfly', 'apple']],
         parents=[]),
    dict(claim_id='C-003',
         statement='Most of the Picbreeder-vs-dense gap in perturbation asymmetry comes from the wiring, not from how the '
                   'weights were found.',
         scope=SCOPE + WIRING, group='G1b',
         prediction='For M3a and M4a, r = (G1b - PB) / (G0 - PB) is at most 0.3 on all three tasks. r >= 0.7 falsifies. '
                    'Between them is MIXED.',
         flip_boundary='Weaker where the mask forces less parity, so apple is the most likely to break this.',
         checks=[position(f'c003_{m.lower()}_{t}', m, t, 'G1b', False) for m in ['M3a', 'M4a'] for t in TASKS],
         parents=[]),
    dict(claim_id='C-004',
         statement='Sparsity alone does not close the gap: pruning a dense SGD network to the same edge count per layer '
                   'leaves its symmetry metrics near those of the dense network.',
         scope=SCOPE + ' G4 keeps the Picbreeder edge count in each layer, but SGD chooses the edges by magnitude. '
                       'M1x excludes skull, whose Picbreeder network has only 3 free maps.', group='G4',
         prediction='For M1, M3a and M4a on all tasks, and M1x on butterfly and apple, r = (G4 - PB) / (G0 - PB) is at '
                    'least 0.7. r <= 0.3 on any falsifies: sparsity would then explain that gap. Between them is MIXED.',
         flip_boundary='Pruning may keep near-symmetric edges if the near-symmetric target rewards them.',
         checks=[position(f'c004_{m.lower()}_{t}', m, t, 'G4', True) for m in ['M1', 'M3a', 'M4a'] for t in TASKS]
                + [position(f'c004_m1x_{t}', 'M1x', t, 'G4', True) for t in ['butterfly', 'apple']],
         parents=[]),
]

Path(__file__).with_suffix('.json').write_text(json.dumps(claims, ensure_ascii=False, indent=1), 'utf-8')
print('wrote', len(claims), 'claims,', sum(len(c['checks']) for c in claims), 'checks')
