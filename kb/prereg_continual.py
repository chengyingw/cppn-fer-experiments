"""Writes C-011 to C-018: does a Picbreeder start help a network learn a small change without disturbing the rest?

Route decision, fixed in advance: if none of the eight is VERIFIED or MIXED, "CPPN representations help continual
learning" loses its main support, and that route is cut back to what the data still supports.
"""
import json
from pathlib import Path

TASKS = ['skull', 'butterfly', 'apple']
SCOPE = ('phase0/continual.py: 2000 steps of Adam with unit-norm gradients at each task\'s training learning rate '
         '(lr_scale 1), 64x64. G0 is fully fitted dense SGD, including seeds from batch b6. Three edits per class per '
         'task. Rows share networks, so medians are descriptive summaries, not independent samples; Picbreeder is one '
         'network per task. lr_scale 0.1 rows are descriptive and count toward no check.')
GROUPS = {'PBd': ' PBd starts from Picbreeder with every weight free, so its search space equals G0\'s. A 30-step smoke '
                 'test, seen before registration, showed PBd MSE rising at first, because Adam moves all zero weights at '
                 'once. No full run had been seen.',
          'PBa': ' PBa starts from Picbreeder and trains existing edges only, so it cannot add wiring. It was added after '
                 'the PBd smoke test and has far fewer parameters than G0, so it is not the same search space.'}
CONFOUNDS = (' Known biases: Picbreeder weights are larger, so one learning rate is a smaller relative step on existing '
             'edges, which can lower interference and slow adaptation; warp targets come from the Picbreeder generator '
             'itself.')


def ratio(cid, metric, task, cls, group, ok, bad, min_n):
    return dict(check_id=cid, type='ratio', metric=metric, where=dict(task=task, edit_class=cls, lr_scale=1.0),
                group_a=group, group_b='G0', min_n=min_n, **{'pass': dict(op='<=', value=ok), 'fail': dict(op='>=', value=bad)})


def local_claim(cid, cls, label, group, expect):
    key = cid.lower().replace('-', '')
    return dict(
        claim_id=cid,
        statement=f'Starting from Picbreeder ({group}), a network learns a local {label} edit with much less disturbance '
                  'outside the edit than a fully fitted dense network, without being much slower.',
        scope=SCOPE + GROUPS[group] + CONFOUNDS, group=group,
        flip_boundary='A slower learner moves less per step, so the speed checks guard against reading slowness as '
                      'low interference.',
        prediction=f'On each task, median F of {group} is at most 0.5 times G0\'s, and median A1c at most 1.5 times G0\'s. '
                   'F ratio >= 1 or A1c ratio >= 3 falsifies. Between them is MIXED. F needs at least 3 adapted runs. '
                   'Expectation: ' + expect,
        checks=[ratio(f'{key}_f_{t}', 'F', t, cls, group, 0.5, 1.0, 3) for t in TASKS]
               + [ratio(f'{key}_a1_{t}', 'A1c', t, cls, group, 1.5, 3.0, 3) for t in TASKS],
        parents=[])


def warp_claim(cid, cls, label, group, expect):
    key = cid.lower().replace('-', '')
    return dict(
        claim_id=cid,
        statement=f'Starting from Picbreeder ({group}), a network adapts to a {label} warp of its image at least twice as '
                  'fast as a fully fitted dense network.',
        scope=SCOPE + GROUPS[group] + CONFOUNDS, group=group,
        flip_boundary='A warp may be few weights away for Picbreeder because it generated the target.',
        prediction=f'On each task, median A1c of {group} is at most 0.5 times G0\'s. Ratio >= 1 falsifies. Between them is '
                   'MIXED. Expectation: ' + expect,
        checks=[ratio(f'{key}_a1_{t}', 'A1c', t, cls, group, 0.5, 1.0, 3) for t in TASKS],
        parents=[])


claims = [
    local_claim('C-011', 'local_sym', 'symmetric', 'PBd', 'the smoke test makes the analyst doubt this for PBd.'),
    local_claim('C-012', 'local_asym', 'symmetry-breaking', 'PBd', 'the analyst expects falsified.'),
    warp_claim('C-013', 'warp_sym', 'symmetric horizontal scale', 'PBd', 'uncertain, given the smoke test.'),
    warp_claim('C-014', 'warp_asym', 'horizontal shift', 'PBd', 'the analyst expects falsified.'),
    local_claim('C-015', 'local_sym', 'symmetric', 'PBa', 'the FER paper implies this should hold; it is also the edit '
                'type most favourable to a mirror-symmetric representation.'),
    local_claim('C-016', 'local_asym', 'symmetry-breaking', 'PBa', 'the analyst expects MIXED or falsified: the wiring '
                'forces much of the symmetry and cannot change here, so one-sided edits may be out of reach.'),
    warp_claim('C-017', 'warp_sym', 'symmetric horizontal scale', 'PBa', 'the analyst expects this to hold, helped by the '
               'generator bias.'),
    warp_claim('C-018', 'warp_asym', 'horizontal shift', 'PBa', 'uncertain: a shift breaks the mirror axis, yet the target '
               'still comes from the Picbreeder generator.'),
]

Path(__file__).with_suffix('.json').write_text(json.dumps(claims, ensure_ascii=False, indent=1), 'utf-8')
