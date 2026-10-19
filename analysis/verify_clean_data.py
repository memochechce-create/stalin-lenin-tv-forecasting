"""Verify produced labels, canonical denominators and daily targets independently."""
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/cleaned'


def read_jsonl(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def main():
    errors=[]
    records={r['id']:r for r in read_jsonl(DATA/'broadcast_coverage.jsonl.gz')}
    labels=list(read_jsonl(DATA/'mention_labels.jsonl.gz'))
    by_mention={m['mention_id']:m for m in labels}
    if len(by_mention)!=len(labels): errors.append('duplicate mention IDs')
    totals=defaultdict(Counter); marked=defaultdict(set); raw_days=defaultdict(list)
    for r in records.values(): raw_days[r['channel'],r['day']].append(r)
    for m in labels:
        record=records[m['broadcast_id']]
        if m['canonical_broadcast_id']!=record['canonical_id']: errors.append('canonical mismatch '+m['mention_id'])
        pos=m['context_offset']
        if m['context'][pos:pos+len(m['matched'])]!=m['matched']: errors.append('context offset mismatch '+m['mention_id'])
        if m['duplicate_of']:
            parent=by_mention.get(m['duplicate_of'])
            if parent is None or parent['duplicate_of'] or parent['person']!=m['person'] or parent['channel']!=m['channel']:
                errors.append('invalid duplicate chain '+m['mention_id'])
            continue
        key=m['channel'],m['day']; person=m['person']
        if m['label']=='person_likely':
            totals[key][person+'_mentions']+=1
            marked[key,person,'broadcasts'].add(m['canonical_broadcast_id'])
        if m['label'] in ('person_likely','uncertain_asr','uncertain_reference'):
            totals[key][person+'_possible_mentions']+=1
            marked[key,person,'possible_broadcasts'].add(m['canonical_broadcast_id'])
    with gzip.open(DATA/'tv_daily_targets.csv.gz','rt',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    keys=[(r['channel'],r['day']) for r in rows]
    if len(set(keys))!=len(rows): errors.append('duplicate day keys')
    for row in rows:
        key=row['channel'],row['day']; raw=raw_days[key]
        canonical=[r for r in raw if r['is_canonical']]
        n=sum(r['available'] for r in canonical)
        if int(row['available_broadcasts'])!=n: errors.append('denominator '+str(key))
        if row['observed']=='1':
            if row['previous_utc_inventory_status']!='200' or row['current_utc_inventory_status']!='200': errors.append('inventory completeness '+str(key))
            if not canonical or any(not r['available'] for r in canonical): errors.append('source completeness '+str(key))
        for person in ('stalin','lenin'):
            for suffix in ('mentions','possible_mentions','broadcasts','possible_broadcasts'):
                field=person+'_'+suffix
                value=len(marked[key,person,suffix]) if suffix.endswith('broadcasts') else totals[key][field]
                if row['observed']=='1':
                    if row[field]=='' or int(row[field])!=value: errors.append(field+' aggregate '+str(key))
                elif row[field]!='': errors.append(field+' missing must be blank '+str(key))
            if row['observed']=='1':
                c,b,possible,pb=(int(row[person+'_'+s]) for s in ('mentions','broadcasts','possible_mentions','possible_broadcasts'))
                if not (0<=b<=min(c,n) and b<=pb<=min(possible,n) and c<=possible): errors.append('joint count bounds '+str(key))
    report=dict(status='pass' if not errors else 'fail', errors=errors[:100], error_count=len(errors),
                rows=len(rows),labels=len(labels),broadcast_records=len(records),
                interpretation='Structural/data consistency only; does not establish semantic classifier accuracy',
                outputs_sha256={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in
                    ('mention_labels.jsonl.gz','broadcast_coverage.jsonl.gz','tv_daily_targets.csv.gz')})
    output=ROOT/'output/data_cleaning/validation.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(bool(errors))


if __name__=='__main__': main()
