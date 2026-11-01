"""Writes C-007 to C-010. The butterfly batch is fixed in advance: seeds 3-8, one run of each group, no more added."""
import json
from pathlib import Path

from prereg_sparse_controls import SCOPE, position

BATCH = (' Batch b6: butterfly seeds 3-8 in each group, fixed in advance; no seeds will be added after seeing results. '
         'Only b6 runs count as the test group. A G4 seed whose dense parent misses the fit threshold is not pruned.')
REF = (' Reference is G0e, a dense network stopped at MSE <= 0.0065, not the fully trained G0, so that r compares networks '
       'at a similar fit level.')
METRICS = ['M1', 'M1x', 'M3a', 'M4a']


def matched(cid, metric, group, high_is_pass):
    return dict(position(cid, metric, 'butterfly', group, high_is_pass), high_group='G0e', min_n=3,
                test_where=dict(batch='b6'))


claims = [
    dict(claim_id='C-007',
         statement='Whether the gap survives at a coarse fit depends on the task: on apple dense networks at MSE 0.0065 '
                   'already score like Picbreeder on M3a and M4a; on butterfly they stay near the fully trained network.',
         scope=SCOPE + ' G0e against PB and G0. Apple M1 and M1x are not covered: they were in the middle band.',
         group='G0e',
         flip_boundary='Unknown; only two tasks and one fit level were measured.',
         prediction='Post-hoc description of results already seen in C-006, not a prediction. It is kept as a regression '
                    'lock, and new G0e seeds can still break it.',
         checks=[position(f'c007_{m.lower()}_apple', m, 'apple', 'G0e', False) for m in ['M3a', 'M4a']]
                + [position(f'c007_{m.lower()}_butterfly', m, 'butterfly', 'G0e', True) for m in ['M3a', 'M4a']],
         parents=[]),
    dict(claim_id='C-008',
         statement='Fixing the copy edges at 1 and training only NEAT links makes butterfly fit reliably.',
         scope=SCOPE + BATCH, group='G1a',
         flip_boundary='Skull already failed this setup 3 of 3, so success here would be specific to butterfly.',
         prediction='Hypothesis proposed from outside this analysis: at least 4 of 6 G1a seeds reach MSE < 0.01. The '
                    'analyst expects this to fail, given C-001.',
         checks=[dict(check_id='c008_butterfly_fit', type='count', where=dict(task='butterfly', group='G1a', fit=True),
                      where_total=dict(task='butterfly', group='G1a'), min_total=6, op='>=', value=4)],
         parents=[]),
    dict(claim_id='C-009',
         statement='On butterfly at a similar fit level, Picbreeder wiring with SGD-found weights is closer to Picbreeder '
                   'than to a dense network, on symmetry metrics.',
         scope=SCOPE + BATCH + REF, group='G1b',
         flip_boundary='The wiring forces the parity of 59% of maps, so M1 and M3a are partly fixed by construction; M1x '
                       'is the part the wiring leaves free.',
         prediction='r = (G1b - PB) / (G0e - PB) is at most 0.3 for M1, M1x, M3a and M4a. r >= 0.7 falsifies. Between '
                    'them is MIXED. This reverses the C-002 prediction, based on one fitted butterfly seed.',
         checks=[matched(f'c009_{m.lower()}', m, 'G1b', False) for m in METRICS],
         parents=[]),
    dict(claim_id='C-010',
         statement='On butterfly at a similar fit level, sparsity alone does not make a network Picbreeder-like: pruned '
                   'networks stay closer to a dense network.',
         scope=SCOPE + BATCH + REF + ' G4 as in C-005, pruned from b6 dense networks.', group='G4',
         flip_boundary='Results may differ on tasks where coarse fits are already symmetric, as apple is.',
         prediction='r = (G4 - PB) / (G0e - PB) is at least 0.7 for M1, M1x, M3a and M4a. r <= 0.3 falsifies. Between '
                    'them is MIXED.',
         checks=[matched(f'c010_{m.lower()}', m, 'G4', True) for m in METRICS],
         parents=[]),
]

Path(__file__).with_suffix('.json').write_text(json.dumps(claims, ensure_ascii=False, indent=1), 'utf-8')
