"""Writes C-019 to C-021 for phase0/continual_v2.py, and logs notes on earlier claims so they cannot silently vanish."""
import json
from pathlib import Path

TASKS = ['skull', 'butterfly', 'apple']
CLASSES = ['local_sym', 'local_asym', 'warp_sym', 'warp_asym']
SCOPE = ('phase0/continual_v2.py at 2000 steps, 64x64, same edits and optimizer as C-011 to C-018, learning rate stated per '
         'check. R is remaining error toward the new target, relative to the edit\'s size; its normalization does not '
         'depend on the network. Only fully fitted networks: PBd, PBa, G0. Written after seeing the earlier continual-learning '
         'results, so these are remeasurements with informed predictions, not blind tests. Picbreeder is one network per '
         'task and its edits are not independent samples, so results describe these networks and do not generalize to '
         'Picbreeder networks. Medians are descriptive. R at a fixed step needs no censoring.')


def ratio(cid, metric, task, cls, lr, a='PBd', b='G0', ok=2.0, bad=1.0, higher_passes=True):
    bands = ({'pass': dict(op='>=', value=ok), 'fail': dict(op='<=', value=bad)} if higher_passes
             else {'pass': dict(op='<=', value=ok), 'fail': dict(op='>=', value=bad)})
    return dict(check_id=cid, type='ratio', table='v2', metric=metric, where=dict(task=task, edit_class=cls, lr_scale=lr),
                group_a=a, group_b=b, min_n=3, **bands)


claims = [
    dict(claim_id='C-019',
         statement='At the training learning rate, a Picbreeder start leaves at least twice the remaining edit error of a '
                   'fully fitted dense network after 2000 steps, on skull and butterfly.',
         scope=SCOPE, group='PBd',
         flip_boundary='Learning rate may reverse this, which C-020 tests. Apple is left out: earlier results showed both '
                       'groups near 0.22 of E0 there on local edits.',
         prediction='R2000 ratio PBd / G0 >= 2 on every edit class. Ratio <= 1 falsifies. Between them is MIXED.',
         checks=[ratio(f'c019_{t}_{c}', 'R2000', t, c, 1.0) for t in ['skull', 'butterfly'] for c in CLASSES],
         parents=[]),
    dict(claim_id='C-020',
         statement='Lowering the learning rate tenfold does not remove the gap in remaining edit error on skull and '
                   'butterfly, so the gap is not only a learning-rate mismatch.',
         scope=SCOPE, group='PBd',
         flip_boundary='Earlier adaptation counts improved for PBd warps at this rate, so warps are the most likely to '
                       'break this. Other rates and larger budgets are not covered.',
         prediction='At lr_scale 0.1, R2000 ratio PBd / G0 >= 2 on every edit class. Ratio <= 1 falsifies. Between them is '
                    'MIXED. The analyst expects MIXED, driven by symmetric warps.',
         checks=[ratio(f'c020_{t}_{c}', 'R2000', t, c, 0.1) for t in ['skull', 'butterfly'] for c in CLASSES],
         parents=[]),
    dict(claim_id='C-021',
         statement='Local edits on apple are hard for both starts: neither leaves less than half the other\'s remaining '
                   'edit error after 2000 steps.',
         scope=SCOPE, group='PBd',
         flip_boundary='This is limited by the step budget and says nothing of the error reachable with more steps.',
         prediction='At lr_scale 1, R2000 ratio PBd / G0 is at most 1.5 for both local classes, and at least 1 / 1.5 in the '
                    'reverse direction. A ratio >= 2 either way falsifies. Between them is MIXED.',
         checks=[ratio(f'c021_{c}_pb_over_dense', 'R2000', 'apple', c, 1.0, ok=1.5, bad=2.0, higher_passes=False)
                 for c in CLASSES[:2]]
                + [ratio(f'c021_{c}_dense_over_pb', 'R2000', 'apple', c, 1.0, a='G0', b='PBd', ok=1.5, bad=2.0, higher_passes=False)
                   for c in CLASSES[:2]],
         parents=[]),
]

Path(__file__).with_suffix('.json').write_text(json.dumps(claims, ensure_ascii=False, indent=1), 'utf-8')
