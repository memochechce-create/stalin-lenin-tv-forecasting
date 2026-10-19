"""Run the pinned contextual model over all candidate spans, with resumable cache."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from entity_model import EntityModel, input_key, MODEL_ID, model_revision, SCHEMA_VERSION

ROOT=Path(__file__).resolve().parents[1]


def candidates():
    """Read original extraction, not previous heuristic semantic decisions."""
    with gzip.open(ROOT/'data/cleaned/source_mentions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            source=json.loads(line)
            if source.get('error'): raise ValueError('Unverified source '+source['broadcast_id'])
            for m in source['candidates']:
                yield dict(m,broadcast_id=source['broadcast_id'],
                           mention_id=f"{source['broadcast_id']}:{m['person']}:{m['offset']}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    data=ROOT/'data/cleaned'
    db=sqlite3.connect(data/'entity_predictions.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS prediction (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    rows=list(candidates())
    unique={input_key(m):m for m in rows}
    existing={key for key, in db.execute('SELECT key FROM prediction')}
    missing=[(k,m) for k,m in unique.items() if k not in existing]
    print(f'{len(rows)} candidate spans, {len(unique)} distinct inputs, {len(missing)} new model decisions',flush=True)
    started=time.monotonic()
    if missing:
        model=EntityModel(threads=args.threads)
        for offset in range(0,len(missing),args.batch_size):
            batch=missing[offset:offset+args.batch_size]
            predictions=model.predict([m for _,m in batch],batch_size=args.batch_size)
            for (key,m),prediction in zip(batch,predictions):
                if prediction['input_key']!=key: raise RuntimeError('Model-cache key mismatch')
                db.execute('INSERT OR REPLACE INTO prediction VALUES (?,?)',(key,json.dumps(prediction,ensure_ascii=False)))
            db.commit()
            done=min(offset+len(batch),len(missing))
            if done%100<args.batch_size or done==len(missing):
                print(f'Model decisions {done}/{len(missing)}; elapsed {time.monotonic()-started:.0f}s',flush=True)
    lookup={key:json.loads(payload) for key,payload in db.execute('SELECT key,payload FROM prediction') if key in unique}
    target=data/'entity_predictions.jsonl.gz'; partial=target.with_suffix('.partial')
    counts=Counter()
    with partial.open('wb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as out:
        for m in rows:
            decision=lookup[input_key(m)]
            result=dict(mention_id=m['mention_id'],model=MODEL_ID,revision=model_revision(),schema=SCHEMA_VERSION,**decision)
            out.write((json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8'))
            counts[result['category']]+=1
    partial.replace(target)
    report=dict(model=MODEL_ID,revision=model_revision(),schema=SCHEMA_VERSION,
                candidate_spans=len(rows),unique_inputs=len(unique),new_decisions=len(missing),
                raw_model_categories=dict(counts),elapsed_seconds=round(time.monotonic()-started,2),
                predictions_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                note='Raw uncalibrated model scores. Model errors and domain shift remain possible.')
    path=ROOT/'output/data_cleaning/neural/inference_summary.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    db.close()


if __name__=='__main__': main()
