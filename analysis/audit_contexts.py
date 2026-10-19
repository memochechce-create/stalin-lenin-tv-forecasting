"""Read-only, reproducible screening; flags are not validated false positives."""
import collections
import csv
import gzip
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/data_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect((ROOT / 'data_collection/data/tv.sqlite').as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    targets = {(r['channel'], r['day']): r for r in csv.DictReader(
        gzip.open(ROOT / 'data/tv_daily_targets.csv.gz', 'rt', encoding='utf-8'))}
    aggregates = collections.defaultdict(collections.Counter)
    matches = []
    counts = {}
    for row in db.execute("SELECT * FROM broadcast WHERE status='200'"):
        day = json.loads(row['metadata'])['start_localtime'][:10]
        key = row['channel'], day
        if key not in targets:
            continue
        n = json.loads(row['counts'])
        counts[row['id']] = n
        aggregates[key]['available_broadcasts'] += 1
        for person in ('stalin', 'lenin'):
            aggregates[key][person + '_mentions'] += n[person]
            aggregates[key][person + '_broadcasts'] += int(n[person] > 0)
        for context in json.loads(row['contexts']):
            if context['person'] not in ('stalin', 'lenin'):
                continue
            prefix = context['context'][:min(100, context['offset'])]
            category = None
            # Anchored immediately before this match, never just proximity.
            if re.search(r'(?:улиц\w*|проспект\w*|площад\w*|станци\w*)[\s.,«"\-]+(?:имени\s+)?$', prefix, re.I):
                category = 'address_or_station_phrase'
            elif re.search(r'ледокол\w*[\s.,«"\-]+(?:имени\s+)?$', prefix, re.I):
                category = 'ship_name_phrase'
            elif re.search(r'(?:памятник\w*|орден\w*|мавзоле\w*)[\s.,«"\-]+(?:имени\s+)?$', prefix, re.I):
                category = 'ambiguous_commemoration_or_possession'
            if category:
                matches.append(dict(broadcast_id=row['id'], channel=row['channel'], day=day,
                    observed=int(targets[key]['observed']), category=category, **context))
    mismatches = []
    for key, target in targets.items():
        for column, value in target.items():
            if column in ('available_broadcasts', 'stalin_mentions', 'lenin_mentions', 'stalin_broadcasts', 'lenin_broadcasts') and value != '':
                if int(value) != aggregates[key][column]:
                    mismatches.append([*key, column, value, aggregates[key][column]])
    summaries = []
    for category in sorted({r['category'] for r in matches}):
        for person in ('stalin', 'lenin'):
            for channel in ('1TV', 'NTV', 'RUSSIA1', 'RUSSIA24'):
                selected = [r for r in matches if r['observed'] and r['category'] == category
                    and r['person'] == person and r['channel'] == channel]
                frequency = collections.Counter(r['broadcast_id'] for r in selected)
                summaries.append(dict(category=category, person=person, channel=channel,
                    tokens=len(selected), broadcasts=len(frequency),
                    broadcasts_all_matches_flagged=sum(v == counts[k][person] for k, v in frequency.items())))
    report = dict(note='Rule-based screening, not human-labelled false positives. Categories require a research definition.',
        rows=len(targets), observed=sum(r['observed'] == '1' for r in targets.values()),
        aggregation_mismatches=mismatches, categories=summaries)
    (OUT / 'context_screen_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'context_screen_matches.json').write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding='utf-8')
    db.close()
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
