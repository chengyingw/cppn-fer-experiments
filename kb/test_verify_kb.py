"""Regression tests for the claim verifier. Run: python test_verify_kb.py"""
import json
import tempfile
import unittest
from pathlib import Path

from verify_kb import (BAND, FAIL, NODATA, PASS, Store, add_claim, add_note, integrity_errors, merge_claims, run_check,
                       verdict, withdraw_claim)


def row(group, value, task='t', fit=True):
    return dict(task=task, group=group, fit=fit, m=value)


def position(cid='c1', task='t'):
    return dict(check_id=cid, type='position', metric='m', where=dict(task=task), low_group='L', high_group='H',
                test_group='T', **{'pass': dict(op='>=', value=0.7), 'fail': dict(op='<=', value=0.3)})


def claim(cid, checks, parents=()):
    return dict(claim_id=cid, statement='s', prediction='p', checks=checks, parents=list(parents))


class Checks(unittest.TestCase):
    def setUp(self):
        self.refs = [row('L', 0.0), row('H', 1.0)]

    def test_bands(self):
        for v, want in [(0.8, PASS), (0.2, FAIL), (0.5, BAND)]:
            self.assertEqual(run_check(position(), self.refs + [row('T', v), row('T', v)])[0], want)

    def test_too_few_fitted_runs_is_no_data_not_failure(self):
        rows = self.refs + [row('T', 0.0), row('T', None, fit=False), row('T', 0.0, fit=False)]
        self.assertEqual(run_check(position(), rows)[0], NODATA)

    def test_test_where_restricts_only_the_test_group(self):
        rows = self.refs + [dict(row('T', 0.0), batch='old'), dict(row('T', 0.9), batch='new'),
                            dict(row('T', 0.9), batch='new')]
        check = dict(position(), test_where=dict(batch='new'))
        self.assertEqual(run_check(check, rows)[0], PASS)
        self.assertEqual(run_check(dict(check, min_n=3), rows)[0], NODATA)

    def test_ratio_bands_and_missing_data(self):
        check = dict(check_id='q', type='ratio', metric='m', where=dict(task='t'), group_a='A', group_b='B',
                     **{'pass': dict(op='<=', value=0.5), 'fail': dict(op='>=', value=1.0)})
        rows = lambda a: [row('A', a), row('A', a), row('B', 1.0), row('B', 1.0)]
        for v, want in [(0.4, PASS), (0.7, BAND), (1.2, FAIL)]:
            self.assertEqual(run_check(check, rows(v))[0], want)
        self.assertEqual(run_check(check, [row('A', 0.1), row('B', 1.0), row('B', 1.0)])[0], NODATA)
        self.assertEqual(run_check(check, [row('A', 0.1), row('A', 0.1), row('B', 0.0), row('B', 0.0)])[0], NODATA)

    def test_both_groups_censored_is_no_data(self):
        check = dict(check_id='q', type='ratio', metric='m', where=dict(task='t'), group_a='A', group_b='B',
                     censored_at=2000, **{'pass': dict(op='<=', value=1.5), 'fail': dict(op='>=', value=3.0)})
        both = [row('A', 2000), row('A', 2000), row('B', 2000), row('B', 2000)]
        self.assertEqual(run_check(check, both)[0], NODATA)
        one = [row('A', 2000), row('A', 2000), row('B', 50), row('B', 50)]
        self.assertEqual(run_check(check, one)[0], FAIL)

    def test_checks_read_only_their_own_table(self):
        v2 = [dict(r, _table='v2') for r in self.refs + [row('T', 0.9), row('T', 0.9)]]
        self.assertEqual(run_check(position(), v2)[0], NODATA)
        self.assertEqual(run_check(dict(position(), table='v2'), v2)[0], PASS)

    def test_verdict_precedence(self):
        self.assertEqual(verdict([(PASS, ''), (NODATA, ''), (FAIL, '')]), 'FALSIFIED')
        self.assertEqual(verdict([(PASS, ''), (NODATA, ''), (BAND, '')]), 'INCONCLUSIVE')
        self.assertEqual(verdict([(PASS, ''), (BAND, '')]), 'MIXED')
        self.assertEqual(verdict([(PASS, '')]), 'VERIFIED')


class Lineage(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = Store(self.dir.name)
        rows = [row('L', 0.0, 'a'), row('H', 1.0, 'a'), row('T', 0.9, 'a'), row('T', 0.9, 'a'),
                row('L', 0.0, 'b'), row('H', 1.0, 'b'), row('T', 0.1, 'b'), row('T', 0.1, 'b')]
        self.store.db_path.write_text('\n'.join(json.dumps(r) for r in rows), 'utf-8')
        add_claim(self.store, claim('A', [position('ca', 'a')]))
        add_claim(self.store, claim('B', [position('cb', 'b')]))

    def tearDown(self):
        self.dir.cleanup()

    def test_rows_come_from_both_tables(self):
        self.store.db_path.with_name('continual_db.jsonl').write_text(json.dumps(row('X', 1.0)), 'utf-8')
        self.assertEqual(len(self.store.rows()), 9)

    def test_merge_inherits_parent_checks_and_rejects_regression(self):
        with self.assertRaises(ValueError):
            merge_claims(self.store, ['A', 'B'], claim('M', []))
        self.assertNotIn('M', self.store.claims)
        self.assertEqual(self.store.claims['A']['status'], 'PRE_REGISTERED')

    def test_merge_of_passing_parents_keeps_lineage_and_is_rechecked(self):
        add_claim(self.store, claim('C', [position('cc', 'a')]))
        m = merge_claims(self.store, ['A', 'C'], claim('M', []))
        self.assertEqual({c['check_id'] for c in m['checks']}, {'ca', 'cc'})
        self.assertEqual(self.store.claims['A']['status'], 'MERGED')
        self.assertEqual(integrity_errors(self.store), [])

    def test_transitive_ancestor_checks_cannot_be_dropped(self):
        add_claim(self.store, claim('C', [position('cc', 'a')]))
        merge_claims(self.store, ['A', 'C'], claim('M', []))
        with self.assertRaises(ValueError):
            add_claim(self.store, claim('N', [], parents=['M']))

    def test_editing_a_frozen_claim_is_detected(self):
        self.store.claims['A']['prediction'] = 'rewritten after seeing data'
        self.assertTrue(any('changed after logging' in e for e in integrity_errors(self.store)))

    def test_withdrawal_is_refused_once_the_group_has_data(self):
        self.store.claims['A']['group'] = 'T'
        with self.assertRaises(ValueError):
            withdraw_claim(self.store, 'A', 'dislike the result')
        self.store.claims['B']['group'] = 'unrun'
        withdraw_claim(self.store, 'B', 'design error found before data')
        self.assertEqual(self.store.claims['B']['status'], 'WITHDRAWN')

    def test_removing_a_logged_note_is_detected(self):
        add_note(self.store, 'A', 'censored at the budget')
        self.assertEqual(integrity_errors(self.store), [])
        self.store.claims['A'].pop('note')
        self.assertTrue(any('note' in e for e in integrity_errors(self.store)))

    def test_status_change_is_not_tampering(self):
        self.store.claims['A']['status'] = 'FALSIFIED'
        self.assertEqual(integrity_errors(self.store), [])

    def test_editing_the_log_is_detected(self):
        lines = self.store.log_path.read_text('utf-8').splitlines()
        rec = json.loads(lines[0])
        rec['time'] = '2000-01-01T00:00:00+00:00'
        lines[0] = json.dumps(rec)
        self.store.log_path.write_text('\n'.join(lines) + '\n', 'utf-8')
        self.assertTrue(any('log broken' in e for e in integrity_errors(self.store)))


if __name__ == '__main__':
    unittest.main()
