"""Read-only audit of saved TV sources, counts and exports (standard library).

Run from any directory: python analysis/audit_data.py
Writes only output/data_audit/. Does not clean or overwrite research data.
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import re
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data_collection'))
from collect_tv import PATTERNS, srt_to_text

OUT = ROOT / 'output/data_audit'
PEOPLE = ('stalin', 'lenin')
# Deliberately a review flag, not a classifier: proximity alone is insufficient.
OBJECT = re.compile(r'\b(?:улиц\w*|проспект\w*|площад\w*|переул\w*|бульвар\w*|'
                    r'шоссе|набережн\w*|станци\w*|метро|имени|им\.|орден\w*|'
                    r'памятник\w*|мавзоле\w*|ледокол\w*|библиотек\w*)', re.I)
BROAD = re.compile(r'(?<!\w)(?:сталин|ленин)[а-яё]*(?!\w)', re.I)


def jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def write_csv(name, rows, fields):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-sources', action='store_true', help='Re-read all 114k originals; potentially slow on Windows')
    parser.add_argument('--sample-size', type=int, default=1000)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error('--sample-size must be positive')
    OUT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect((ROOT / 'data_collection/data/tv.sqlite').as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    stats = dict(audited_at=dt.datetime.now(dt.timezone.utc).isoformat(), errors=[],
                 scope='Saved TV archive; checks do not establish complete television airtime or ASR accuracy')
    counts, forms, omitted, suspicious = Counter(), Counter(), Counter(), Counter()
    fingerprints = defaultdict(list)
    review = []
    stored = {}
    strata = defaultdict(list)
    for r in db.execute("SELECT id,channel,day,counts,source_format FROM broadcast WHERE status='200'"):
        for p in PEOPLE:
            stored[r['id'],p] = json.loads(r['counts'])[p]
        strata[r['channel'],r['day'][:4],r['source_format'] or 'txt'].append(r['id'])
    rng = random.Random(20261017)
    # Round-robin allocation covers every channel/year/source-format stratum.
    pools = [rng.sample(v, len(v)) for k,v in sorted(strata.items())]
    selected = set()
    while len(selected) < min(args.sample_size, sum(map(len,pools))):
        for pool in pools:
            if pool and len(selected) < args.sample_size:
                selected.add(pool.pop())
    stats['source_check_mode'] = 'all' if args.all_sources else 'stratified_sample'
    stats['source_sample_seed'] = 20261017
    stats['source_population_size'] = len(stored)//2
    formats = Counter()
    for i, row in enumerate(db.execute('SELECT * FROM broadcast ORDER BY id'), 1):
        sid = row['id']
        if row['status'] != '200':
            continue
        if not args.all_sources and sid not in selected:
            continue
        fmt = row['source_format'] or 'txt'
        path = ROOT / 'data_collection/data' / ('tv_subtitles' if fmt == 'srt' else 'tv_transcripts') / sid[:4] / (sid + ('.srt.gz' if fmt == 'srt' else '.txt.gz'))
        try:
            raw = gzip.decompress(path.read_bytes())
            body = srt_to_text(raw).encode('utf-8') if fmt == 'srt' else raw
            if hashlib.sha256(raw).hexdigest() != (row['source_sha256'] or row['sha256']):
                stats['errors'].append([sid, 'source checksum'])
            if hashlib.sha256(body).hexdigest() != row['sha256']:
                stats['errors'].append([sid, 'text checksum'])
            text = body.decode('utf-8-sig')
        except (OSError, ValueError, UnicodeError) as e:
            stats['errors'].append([sid, str(e)])
            continue
        formats[fmt] += 1
        fingerprints[row['channel'], row['sha256']].append(sid)
        expected = json.loads(row['counts'])
        actual_contexts = []
        for person, pattern in PATTERNS.items():
            matches = list(pattern.finditer(text))
            if len(matches) != expected[person]:
                stats['errors'].append([sid, person + ' count mismatch'])
            for m in matches:
                actual_contexts.append(dict(person=person, offset=m.start(), matched=m.group(), context=text[max(0, m.start()-100):m.end()+100]))
                if person not in PEOPLE:
                    continue
                counts[person] += 1
                forms[person, m.group().lower()] += 1
                left, right = text[max(0, m.start()-65):m.start()], text[m.end():m.end()+65]
                if OBJECT.search(left + ' ' + right):
                    suspicious[person] += 1
                    review.append(dict(broadcast_id=sid, person=person, offset=m.start(), matched=m.group(),
                                       review_reason='object_word_nearby_not_confirmed_false_positive',
                                       context=text[max(0, m.start()-220):m.end()+220], label='', comment=''))
            if person in PEOPLE:
                stored[sid, person] = expected[person]
        if actual_contexts != json.loads(row['contexts'] or '[]'):
            stats['errors'].append([sid, 'stored contexts mismatch'])
        for m in BROAD.finditer(text):
            if not any(PATTERNS[p].fullmatch(m.group()) for p in PEOPLE):
                omitted[m.group().lower()] += 1
        if '\ufffd' in text:
            counts['texts_with_replacement_character'] += 1
        if not text.strip():
            counts['empty_successful_texts'] += 1
        if i % 10000 == 0:
            print(f'Checked {i:,} broadcasts', flush=True)
    stats.update(verified_files=sum(formats.values()), formats=dict(formats), source_checked_counts=dict(counts),
                 nearby_object_flags=dict(suspicious),
                 duplicate_text_groups=sum(len(v)>1 for v in fingerprints.values()),
                 extra_identical_text_records=sum(len(v)-1 for v in fingerprints.values()))
    write_csv('matched_forms.csv', [dict(person=p, form=f, count=n) for (p,f),n in sorted(forms.items())], ['person','form','count'])
    write_csv('unmatched_surname_prefixes.csv', [dict(token=t,count=n) for t,n in omitted.most_common()], ['token','count'])
    write_csv('object_contexts_for_review.csv', review, ['broadcast_id','person','offset','matched','review_reason','context','label','comment'])
    write_csv('identical_text_groups.csv', [dict(channel=k[0], sha256=k[1], broadcasts=';'.join(v), n=len(v)) for k,v in fingerprints.items() if len(v)>1], ['channel','sha256','broadcasts','n'])
    exports = ROOT / 'data_collection/data/bigquery_upload'
    exported = list(jsonl(exports / 'tv_broadcasts.jsonl.gz'))
    if len({r['broadcast_id'] for r in exported}) != len(exported):
        stats['errors'].append(['export', 'duplicate broadcast IDs'])
    group = defaultdict(Counter)
    for r in exported:
        if r['transcript_status'] != '200':
            continue
        g = group[r['channel'],r['day']]
        g['available_broadcasts'] += 1
        for p in PEOPLE:
            n = r[p+'_candidates']
            if n != stored.get((r['broadcast_id'],p)):
                stats['errors'].append([r['broadcast_id'], 'export count mismatch'])
            g[p+'_mentions'] += n
            g[p+'_broadcasts'] += n > 0
    with gzip.open(ROOT/'data/tv_daily_targets.csv.gz','rt',encoding='utf-8') as f:
        targets = list(csv.DictReader(f))
    coverage = list(jsonl(exports/'tv_daily_coverage.jsonl.gz'))
    cov = {(r['channel'],r['day']):r for r in coverage}
    missing = []
    by_channel = defaultdict(Counter)
    for r in targets:
        key = r['channel'],r['day']
        g, c = group[key], cov[key]
        observed = c['inventory_status']=='200' and c['coverage_status']=='observed_listed_broadcasts'
        if int(r['observed']) != observed or int(r['available_broadcasts']) != g['available_broadcasts']:
            stats['errors'].append([list(key), 'coverage/denominator mismatch'])
        by_channel[r['channel']]['observed_days' if observed else 'missing_days'] += 1
        for p in PEOPLE:
            for suffix in ('mentions','broadcasts'):
                field = p+'_'+suffix
                if (r[field] != '' if not observed else int(r[field]) != g[field]):
                    stats['errors'].append([list(key), field+' mismatch'])
            if not 0 <= g[p+'_broadcasts'] <= min(g['available_broadcasts'],g[p+'_mentions']):
                stats['errors'].append([list(key), p+' hierarchy violation'])
            by_channel[r['channel']][p+'_exported_candidates'] += g[p+'_mentions']
        if not observed:
            missing.append({k:c.get(k) for k in ('channel','day','inventory_status','listed_broadcasts','available_transcripts','missing_transcripts','coverage_status')})
    stats.update(exported_broadcasts=len(exported), target_rows=len(targets), missing_target_days=len(missing), channels=dict(by_channel))
    expected_keys = {(c,(dt.date(2023,1,1)+dt.timedelta(days=i)).isoformat()) for c in ('1TV','NTV','RUSSIA1','RUSSIA24') for i in range(1371)}
    actual_keys = [(r['channel'],r['day']) for r in targets]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys or set(cov) != expected_keys:
        stats['errors'].append(['calendar', 'duplicate or missing channel/day keys'])
    write_csv('missing_channel_days.csv', missing, ['channel','day','inventory_status','listed_broadcasts','available_transcripts','missing_transcripts','coverage_status'])
    (OUT/'audit_summary.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    db.close()
    return bool(stats['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
