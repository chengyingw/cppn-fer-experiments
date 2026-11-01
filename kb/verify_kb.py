#!/usr/bin/env python3
"""Executable claims: every claim carries checks that are replayed on experiments_db.jsonl.

  python verify_kb.py                 replay all checks and integrity rules; exit 1 on any violation
  python verify_kb.py --settle        same, then write each active claim's status
  python verify_kb.py --add FILE      freeze the claims in a JSON list and log them, before any data is computed

  python verify_kb.py --withdraw ID REASON   retire a claim, only while its group has no runs in the database

Status: PRE_REGISTERED -> VERIFIED | FALSIFIED | MIXED | INCONCLUSIVE | WITHDRAWN; merged parents become MERGED.
Too little data is INCONCLUSIVE, never FALSIFIED. Falsified claims stay in the file. The statement, prediction
and checks of a logged claim are frozen by hash, and commits.jsonl is hash-chained. The chain detects later
edits; it does not prove when a claim was written, because this is a local self-attested log.
"""
import argparse
import hashlib
import json
import operator
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt}
FROZEN = ('claim_id', 'statement', 'scope', 'flip_boundary', 'group', 'prediction', 'checks', 'parents')
PASS, FAIL, BAND, NODATA = 'PASS', 'FAIL', 'BAND', 'NODATA'


class Store:
    def __init__(self, root=HERE):
        self.claims_path = Path(root) / 'kb_claims.json'
        self.log_path = Path(root) / 'commits.jsonl'
        self.db_path = Path(root) / 'experiments_db.jsonl'
        self.claims = json.loads(self.claims_path.read_text('utf-8')) if self.claims_path.exists() else {}

    def rows(self):
        out = []
        for path in (self.db_path, self.db_path.with_name('continual_db.jsonl')):
            if path.exists():
                out += [json.loads(l) for l in path.read_text('utf-8').splitlines() if l.strip()]
        v2 = self.db_path.with_name('continual_db_v2.jsonl')
        if v2.exists():  # reused group names must not reach checks written against the older tables
            out += [dict(json.loads(l), _table='v2') for l in v2.read_text('utf-8').splitlines() if l.strip()]
        return out

    def log(self):
        if not self.log_path.exists():
            return []
        return [json.loads(l) for l in self.log_path.read_text('utf-8').splitlines() if l.strip()]

    def save(self):
        self.claims_path.write_text(json.dumps(self.claims, ensure_ascii=False, indent=1), 'utf-8')

    def commit(self, op, claim, **extra):
        log = self.log()
        rec = dict(seq=len(log), time=datetime.now(timezone.utc).isoformat(timespec='seconds'), op=op,
                   claim_id=claim['claim_id'], claim_sha256=claim_hash(claim),
                   prev=sha(json.dumps(log[-1], sort_keys=True)) if log else None, **extra)
        with self.log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def claim_hash(claim):
    return sha(json.dumps({k: claim.get(k) for k in FROZEN}, sort_keys=True, ensure_ascii=False))


def select(rows, where):
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def group_median(rows, where, metric, min_n=1):
    vals = [r[metric] for r in select(rows, where) if r.get(metric) is not None]
    return statistics.median(vals) if len(vals) >= min_n else None


def run_check(check, rows):
    """Returns (PASS | FAIL | BAND | NODATA, message)."""
    cid = check['check_id']
    rows = [r for r in rows if r.get('_table') == check.get('table')]
    if check['type'] == 'count':
        if len(select(rows, check['where_total'])) < check['min_total']:
            return NODATA, f'{cid}: fewer than {check["min_total"]} runs'
        n = len(select(rows, check['where']))
        ok = OPS[check['op']](n, check['value'])
        return (PASS if ok else FAIL), f'{cid}: count={n}, need {check["op"]} {check["value"]}'
    if check['type'] == 'position':
        # r = 0 means the test group sits at the low group's value, r = 1 at the high group's
        w, m = dict(check['where'], fit=True), check['metric']
        low = group_median(rows, dict(w, group=check['low_group']), m)
        high = group_median(rows, dict(w, group=check['high_group']), m)
        test = group_median(rows, dict(w, group=check['test_group'], **check.get('test_where', {})), m,
                            check.get('min_n', 2))
        if low is None or high is None or test is None:
            return NODATA, f'{cid}: not enough fitted runs for {check["test_group"]} (low={low}, high={high}, test={test})'
        if abs(high - low) < 1e-9:
            return NODATA, f'{cid}: reference groups coincide'
        r = (test - low) / (high - low)
        msg = f'{cid}: r={r:.3f} ({check["low_group"]}={low:.3f}, {check["test_group"]}={test:.3f}, {check["high_group"]}={high:.3f})'
        if OPS[check['pass']['op']](r, check['pass']['value']):
            return PASS, msg
        if OPS[check['fail']['op']](r, check['fail']['value']):
            return FAIL, msg
        return BAND, msg
    if check['type'] == 'ratio':
        # median of group_a / median of group_b over the rows selected by `where`
        w, m = check['where'], check['metric']
        a = group_median(rows, dict(w, group=check['group_a']), m, check.get('min_n', 2))
        b = group_median(rows, dict(w, group=check['group_b']), m, check.get('min_n', 2))
        if a is None or b is None or b <= 0:
            return NODATA, f'{cid}: not enough data ({check["group_a"]}={a}, {check["group_b"]}={b})'
        if 'censored_at' in check and a >= check['censored_at'] and b >= check['censored_at']:
            return NODATA, f'{cid}: both groups censored at {check["censored_at"]}'
        q = a / b
        msg = f'{cid}: ratio={q:.3f} ({check["group_a"]}={a:.4g}, {check["group_b"]}={b:.4g})'
        if OPS[check['pass']['op']](q, check['pass']['value']):
            return PASS, msg
        if OPS[check['fail']['op']](q, check['fail']['value']):
            return FAIL, msg
        return BAND, msg
    raise ValueError(f'{cid}: unknown check type {check["type"]}')


def verdict(results):
    states = [s for s, _ in results]
    if FAIL in states:
        return 'FALSIFIED'
    if NODATA in states:
        return 'INCONCLUSIVE'
    return 'MIXED' if BAND in states else 'VERIFIED'


def ancestor_checks(store, claim):
    out = {}
    for pid in claim.get('parents', []):
        parent = store.claims[pid]
        out.update(ancestor_checks(store, parent))
        out.update({c['check_id']: c for c in parent['checks']})
    return out


def add_claim(store, claim):
    if claim['claim_id'] in store.claims:
        raise ValueError(f'{claim["claim_id"]} already exists')
    for c in ancestor_checks(store, claim):
        if c not in {x['check_id'] for x in claim['checks']}:
            raise ValueError(f'{claim["claim_id"]} dropped inherited check {c}')
    store.claims[claim['claim_id']] = dict(claim, status='PRE_REGISTERED')
    store.save()
    store.commit('add', claim)


def merge_claims(store, parent_ids, new_claim):
    """Inherit every ancestor check, replay all of them, and write only if every one passes."""
    claim = dict(new_claim, parents=list(parent_ids), checks=list(new_claim.get('checks', [])))
    have = {c['check_id'] for c in claim['checks']}
    claim['checks'] += [c for k, c in ancestor_checks(store, claim).items() if k not in have]
    rows = store.rows()
    results = [run_check(c, rows) for c in claim['checks']]
    bad = [m for s, m in results if s != PASS]
    if bad:
        raise ValueError('merge rejected: ' + '; '.join(bad))
    claim['status'] = 'VERIFIED'
    store.claims[claim['claim_id']] = claim
    for pid in parent_ids:
        store.claims[pid].update(status='MERGED', merged_into=claim['claim_id'])
    store.save()
    store.commit('merge', claim)
    return claim


def withdraw_claim(store, cid, reason):
    """Allowed only before the database holds runs of the claim's group, so it cannot hide a bad result."""
    claim = store.claims[cid]
    if select(store.rows(), dict(group=claim.get('group'))):
        raise ValueError(f'{cid}: data for {claim.get("group")} already exists; settle it instead')
    claim.update(status='WITHDRAWN', withdrawn_reason=reason)
    store.save()
    store.commit('withdraw', claim)


def add_note(store, cid, note):
    """Notes do not change a frozen claim, but each one is logged so that editing or removing it is detected."""
    claim = store.claims[cid]
    claim['note'] = note
    store.save()
    store.commit('note', claim, note_sha256=sha(note))


def integrity_errors(store):
    errors, log, prev = [], store.log(), None
    logged, notes = {}, {}
    for rec in log:
        if rec['prev'] != prev:
            errors.append(f'commit log broken at seq {rec["seq"]}')
        prev = sha(json.dumps(rec, sort_keys=True))
        logged[rec['claim_id']] = rec['claim_sha256']
        if 'note_sha256' in rec:
            notes[rec['claim_id']] = rec['note_sha256']
    for cid, claim in store.claims.items():
        if cid not in logged:
            errors.append(f'{cid} has no commit')
        elif logged[cid] != claim_hash(claim):
            errors.append(f'{cid}: statement, prediction or checks changed after logging')
        if cid in notes and sha(claim.get('note', '')) != notes[cid]:
            errors.append(f'{cid}: logged note was edited or removed')
        for c in ancestor_checks(store, claim):
            if c not in {x['check_id'] for x in claim['checks']}:
                errors.append(f'{cid} dropped inherited check {c}')
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--settle', action='store_true')
    ap.add_argument('--add')
    ap.add_argument('--withdraw', nargs=2, metavar=('CLAIM_ID', 'REASON'))
    args = ap.parse_args()
    store = Store()
    if args.withdraw:
        withdraw_claim(store, *args.withdraw)
        print('withdrew', args.withdraw[0])
    if args.add:
        for claim in json.loads(Path(args.add).read_text('utf-8')):
            add_claim(store, claim)
            print('registered', claim['claim_id'])
    errors = integrity_errors(store)
    rows = store.rows()
    for cid, claim in store.claims.items():
        if claim['status'] == 'WITHDRAWN':
            print(f'\n{cid} [WITHDRAWN]: {claim["withdrawn_reason"]}')
            continue
        results = [run_check(c, rows) for c in claim['checks']]
        status = verdict(results)
        print(f'\n{cid} [{claim["status"]}] -> {status}: {claim["statement"]}')
        for s, m in results:
            print(f'  {s:6s} {m}')
        if claim['status'] == 'VERIFIED' and status != 'VERIFIED':
            errors.append(f'{cid}: was VERIFIED, now {status}')
        if args.settle and claim['status'] != 'MERGED':
            claim['status'] = status
    if args.settle:
        store.save()
    for e in errors:
        print('ERROR', e)
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
