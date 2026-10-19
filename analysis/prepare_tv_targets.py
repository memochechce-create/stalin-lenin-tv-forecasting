"""Build corrected daily targets; --legacy reproduces the original raw export panel."""
import argparse
import subprocess
import sys
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'data_collection/data/bigquery_upload'


def read_jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def legacy_main():
    totals = defaultdict(lambda: dict(available_broadcasts=0, stalin_mentions=0,
        lenin_mentions=0, stalin_broadcasts=0, lenin_broadcasts=0))
    seen = set()
    for record in read_jsonl(EXPORT / 'tv_broadcasts.jsonl.gz'):
        assert record['broadcast_id'] not in seen
        seen.add(record['broadcast_id'])
        if str(record['transcript_status']) != '200':
            continue
        group = totals[record['channel'], record['day']]
        group['available_broadcasts'] += 1
        for person in ('stalin', 'lenin'):
            count = int(record[person + '_candidates'] or 0)
            group[person + '_mentions'] += count
            group[person + '_broadcasts'] += int(count > 0)

    rows = []
    for record in read_jsonl(EXPORT / 'tv_daily_coverage.jsonl.gz'):
        group = totals[record['channel'], record['day']]
        observed = (str(record['inventory_status']) == '200'
                    and record['coverage_status'] == 'observed_listed_broadcasts')
        row = dict(channel=record['channel'], day=record['day'], observed=int(observed),
            inventory_status=record['inventory_status'], coverage_status=record['coverage_status'],
            available_broadcasts=group['available_broadcasts'])
        for person in ('stalin', 'lenin'):
            expected = record[person + '_candidates']
            if expected is not None:
                assert group[person + '_mentions'] == expected
            else:
                assert not observed and group['available_broadcasts'] == 0
            assert 0 <= group[person + '_broadcasts'] <= group['available_broadcasts']
            for target in ('mentions', 'broadcasts'):
                row[person + '_' + target] = group[person + '_' + target] if observed else ''
        rows.append(row)

    rows.sort(key=lambda row: (row['channel'], row['day']))
    assert len(rows) == 5484
    target = ROOT / 'data/tv_daily_targets.csv.gz'
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Created daily targets:', len(rows), 'channel-days; missing targets left blank.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', action='store_true', help='Reproduce unfiltered candidates with the historical coverage definition')
    args = parser.parse_args()
    if args.legacy:
        legacy_main()
    else:
        subprocess.run([sys.executable, str(ROOT/'analysis/prepare_clean_data.py')], cwd=ROOT, check=True)
