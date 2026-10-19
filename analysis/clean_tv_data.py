"""Build versioned, provisional person-context targets without overwriting raw data."""
import argparse
import csv
import datetime as dt
import difflib
import gzip
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .mention_rules import VERSION, classify, normalize_tokens
    from .tv_coverage import load_coverage
except ImportError:
    from mention_rules import VERSION, classify, normalize_tokens
    from tv_coverage import load_coverage

ROOT = Path(__file__).resolve().parents[1]
PEOPLE = ('stalin','lenin')
UNCERTAIN = {'uncertain_reference','uncertain_asr'}


def read_jsonl(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path,rows):
    with path.open('wb') as f, gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as z:
        for r in rows:
            z.write((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8'))


def write_csv(path,rows):
    with path.open('wb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as z, io.TextIOWrapper(z,encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def duplicate_pairs(previous,current,source_previous,source_current):
    """Require a long exact token block and, when available, compatible cue times.

    Never deduplicates repeated shows at nonoverlapping broadcast times.
    TXT-only matches require >=20 exact consecutive words; SRT pairs >=12.
    """
    if not previous or not current:
        return []
    overlap_start=max(dt.datetime.fromisoformat(previous[0]['archive_start']),dt.datetime.fromisoformat(current[0]['archive_start']))
    overlap_stop=min(dt.datetime.fromisoformat(previous[0]['archive_stop']),dt.datetime.fromisoformat(current[0]['archive_stop']))
    if overlap_start >= overlap_stop:
        return []
    a,b=source_previous,source_current
    at=normalize_tokens(a['tail']); bt=normalize_tokens(b['head'])
    av,bv=[t[0] for t in at],[t[0] for t in bt]
    aoff=a['chars']-len(a['tail'])
    a_by_offset={t[1]+aoff:i for i,t in enumerate(at)}
    b_by_offset={t[1]:i for i,t in enumerate(bt)}
    am={a_by_offset[m['offset']]:m for m in previous if m['offset'] in a_by_offset}
    bm={b_by_offset[m['offset']]:m for m in current if m['offset'] in b_by_offset}
    if not am or not bm:
        return []
    matches=[]
    for block in difflib.SequenceMatcher(None,av,bv,autojunk=False).get_matching_blocks():
        for bi,m in bm.items():
            if not block.b <= bi < block.b+block.size:
                continue
            ai=block.a+bi-block.b
            old=am.get(ai)
            if old is None or old['person']!=m['person']:
                continue
            timed=old['start_seconds'] is not None and m['start_seconds'] is not None
            if block.size < (12 if timed else 20):
                continue
            # Require context on both sides, not just an isolated name at an edge.
            if min(ai-block.a,block.a+block.size-ai-1) < 3:
                continue
            if timed:
                ta=dt.datetime.fromisoformat(old['archive_start'])+dt.timedelta(seconds=old['start_seconds'])
                tb=dt.datetime.fromisoformat(m['archive_start'])+dt.timedelta(seconds=m['start_seconds'])
                if abs((ta-tb).total_seconds())>15:
                    continue
                if not (overlap_start-dt.timedelta(seconds=10) <= ta <= overlap_stop+dt.timedelta(seconds=10)
                        and overlap_start-dt.timedelta(seconds=10) <= tb <= overlap_stop+dt.timedelta(seconds=10)):
                    continue
            matches.append((m,old,'overlap_exact_tokens_and_cue_time' if timed else 'overlap_exact_20_tokens'))
    return matches


def calendar(day):
    d=dt.date.fromisoformat(day)
    dates={'nov7':(11,7),'may8':(5,8),'may9':(5,9),'lenin_birth':(4,22),
           'lenin_death':(1,21),'stalin_death':(3,5),'stalin_official_birth':(12,21)}
    out={}
    for name,(month,date) in dates.items():
        distance=min(abs((d-dt.date(y,month,date)).days) for y in (d.year-1,d.year,d.year+1))
        out['is_'+name]=int(distance==0)
        out['within_2d_'+name]=int(distance<=2)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',default='2023-01-01')
    p.add_argument('--end',default='2026-10-02')
    args=p.parse_args()
    dest=ROOT/'data/cleaned'; dest.mkdir(parents=True,exist_ok=True)
    audit=ROOT/'output/data_cleaning'; audit.mkdir(parents=True,exist_ok=True)
    supplemental=dest/'supplemental.sqlite'
    coverage=load_coverage(ROOT/'data_collection/data/tv.sqlite',args.start,args.end,
                           supplemental=supplemental if supplemental.exists() else None)
    if coverage['diagnostics']['invalid_timestamp_records']:
        raise RuntimeError('Invalid source timestamps must be resolved before aggregation')
    broadcasts={r['id']:r for r in coverage['broadcasts']}
    all_sources={}; source_errors=[]; source_count=0
    for r in read_jsonl(dest/'source_mentions.jsonl.gz'):
        source_count+=1
        if r.get('error'): source_errors.append([r['broadcast_id'],r['error']])
        if r['broadcast_id'] in broadcasts and r.get('candidates'):
            all_sources[r['broadcast_id']]=r
    available={r['id'] for r in broadcasts.values() if r['available']}
    # source IDs are checked independently without retaining empty source text.
    extracted_ids={r['broadcast_id'] for r in read_jsonl(dest/'source_mentions.jsonl.gz') if not r.get('error')}
    if available-extracted_ids or source_errors:
        raise RuntimeError('Missing/unverified original sources: '+str(sorted(available-extracted_ids)[:10])+str(source_errors[:5]))
    candidates=[]; by_id=defaultdict(list)
    for sid,source in sorted(all_sources.items()):
        record=broadcasts[sid]
        for raw in source['candidates']:
            m=dict(raw,broadcast_id=sid,canonical_broadcast_id=record['canonical_id'],channel=record['channel'],
                   day=record['day'],archive_start=record['start_time'],archive_stop=record['stop_time'],mention_id=f"{sid}:{raw['person']}:{raw['offset']}",
                   rule_version=VERSION,duplicate_of='',dedup_evidence='')
            m['label'],m['reason']=classify(m)
            if not record['is_canonical'] and m['label']=='person_likely':
                m['label'],m['reason']='uncertain_reference','alternate_same_start_record'
            candidates.append(m); by_id[sid].append(m)
    pairs=0
    by_channel=defaultdict(list)
    for sid in by_id:
        by_channel[broadcasts[sid]['channel']].append(broadcasts[sid])
    for records in by_channel.values():
        active=[]
        for r in sorted(records,key=lambda r:(r['start_time'],not r['is_canonical'],r['id'])):
            active=[old for old in active if old['stop_time']>r['start_time']]
            for old in active:
                pairs+=1
                for m,previous,evidence in duplicate_pairs(by_id[old['id']],by_id[r['id']],all_sources[old['id']],all_sources[r['id']]):
                    if not m['duplicate_of']:
                        m['duplicate_of']=previous['duplicate_of'] or previous['mention_id']
                        m['dedup_evidence']=evidence
            active.append(r)
    overrides_path=ROOT/'data/mention_overrides.json'
    overrides=json.loads(overrides_path.read_text(encoding='utf-8')) if overrides_path.exists() else {}
    valid_labels={'person_likely','named_object','other_person',*UNCERTAIN}
    known={m['mention_id'] for m in candidates}
    unknown=set(overrides)-known
    if unknown and args.start=='2023-01-01' and args.end=='2026-10-02':
        raise ValueError('Stale annotation IDs: '+str(sorted(unknown)[:5]))
    for m in candidates:
        if m['mention_id'] in overrides:
            override=overrides[m['mention_id']]
            if override['label'] not in valid_labels or not override.get('reason'):
                raise ValueError('Invalid annotation '+m['mention_id'])
            m.update(label=override['label'],reason=override['reason'],review_method=override.get('method','context_review'))
    totals=defaultdict(Counter); sets=defaultdict(set)
    for m in candidates:
        key=m['channel'],m['day']; p=m['person']; g=totals[key]
        g[p+'_raw_candidates']+=1
        if m['duplicate_of']:
            g[p+'_overlap_duplicates']+=1
            continue
        g[p+'_'+m['label']]+=1
        if m['label']=='person_likely':
            g[p+'_mentions']+=1
            sets[key,p,'likely'].add(m['canonical_broadcast_id'])
        if m['label']=='person_likely' or m['label'] in UNCERTAIN:
            g[p+'_possible_mentions']+=1
            sets[key,p,'possible'].add(m['canonical_broadcast_id'])
    rows=[]
    for c in coverage['daily']:
        key=c['channel'],c['day']; g=totals[key]
        row={k:('|'.join(v) if isinstance(v,list) else int(v) if isinstance(v,bool) else v) for k,v in c.items()}
        row['available_broadcasts']=c['canonical_available_broadcasts']
        row['target_definition']='provisional_person_context_archive_start_day'
        for person in PEOPLE:
            for field in ('raw_candidates','overlap_duplicates','person_likely','named_object','other_person','uncertain_reference','uncertain_asr'):
                row[person+'_'+field+'_observed_part']=g[person+'_'+field]
            for suffix,value in [('mentions',g[person+'_mentions']),('broadcasts',len(sets[key,person,'likely'])),
                                 ('possible_mentions',g[person+'_possible_mentions']),('possible_broadcasts',len(sets[key,person,'possible']))]:
                row[person+'_'+suffix]=value if c['observed'] else ''
        row.update(calendar(c['day']))
        rows.append(row)
    candidates.sort(key=lambda r:(r['broadcast_id'],r['offset']))
    write_jsonl(dest/'mention_labels.jsonl.gz',candidates)
    write_jsonl(dest/'broadcast_coverage.jsonl.gz',coverage['broadcasts'])
    write_csv(dest/'tv_daily_targets.csv.gz',rows)
    # Prepared FOR review: balanced across nonempty person/label/channel strata.
    # Sampling is not annotation and does not establish classifier accuracy.
    review=[]; strata=defaultdict(list)
    for m in candidates:
        strata[m['person'],m['label'],m['channel']].append(m)
    rng=random.Random(20261017)
    for key,group in sorted(strata.items()):
        review.extend(rng.sample(group,min(6,len(group))))
    (audit/'review_sample.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(rule_version=VERSION,sources_verified=source_count,candidates=len(candidates),
                labels=dict(Counter(m['label'] for m in candidates)),
                nonduplicate_labels=dict(Counter(m['label'] for m in candidates if not m['duplicate_of'])),
                overlap_duplicates=sum(bool(m['duplicate_of']) for m in candidates),overlapping_candidate_pairs_checked=pairs,
                context_overrides_applied=sum(m['mention_id'] in overrides for m in candidates),
                days=len(rows),observed_days=sum(bool(r['observed']) for r in rows),
                coverage=coverage['diagnostics'],
                semantics='Rule-based provisional likely-person estimate and broader possible-person sensitivity; neither is a validated bound on truth.',
                units='canonical archive records grouped by MSK broadcast start; surname occurrences, not all aliases',
                source_cache_sha256=hashlib.sha256((dest/'source_mentions.jsonl.gz').read_bytes()).hexdigest())
    # Executable provenance includes all rules/algorithms affecting these outputs.
    report['code_sha256']={name:hashlib.sha256((ROOT/'analysis'/name).read_bytes()).hexdigest()
                          for name in ('clean_tv_data.py','mention_rules.py','tv_coverage.py','source_database.py')}
    (audit/'cleaning_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
