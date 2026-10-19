"""Diagnostic benchmark against explicit primary-agent context judgments.

The reference sample is challenge-enriched and is not independent human gold.
The test split is not used to select prompts or confidence thresholds.
"""
import argparse
import json
import time
from pathlib import Path
from entity_model import EntityModel, MODEL_ID, MODEL_REVISION, SCHEMA_VERSION

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--no-quantize',action='store_true')
    args=p.parse_args()
    rows=json.loads((ROOT/'data/entity_evaluation.json').read_text(encoding='utf-8'))
    began=time.monotonic(); print('Loading local multilingual transformer',flush=True)
    model=EntityModel(quantize=not args.no_quantize)
    print(f'Loaded in {time.monotonic()-began:.1f}s',flush=True)
    started=time.monotonic()
    predictions=model.predict(rows,batch_size=args.batch_size)
    output=[dict(r,prediction=p) for r,p in zip(rows,predictions)]
    report=dict(model=MODEL_ID,revision=MODEL_REVISION,schema=SCHEMA_VERSION,
                quantized=not args.no_quantize,rows=len(rows),inference_seconds=round(time.monotonic()-started,2),
                reference='Primary-agent judgments on challenge-enriched real ASR snippets; not independent human gold.')
    for split in ('development','test'):
        subset=[r for r in output if r['split']==split]
        n=len(subset); correct=sum(r['reference_category']==r['prediction']['category'] for r in subset)
        report[split]=dict(n=n,accuracy=round(correct/n,4),correct=correct)
    dest=ROOT/'output/data_cleaning/neural'; dest.mkdir(parents=True,exist_ok=True)
    (dest/'evaluation_predictions.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'evaluation_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    for r in output:
        print(r['evaluation_id'],r['reference_category'],r['prediction']['category'],round(r['prediction']['score'],3),flush=True)


if __name__=='__main__': main()
