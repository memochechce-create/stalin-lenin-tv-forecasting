"""Development-only selection of a classifier on contextual target-span embeddings."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from train_entity_bert import inputs,category,macro_f1,CLASSES
from span_bert import SpanEncoder

ROOT=Path(__file__).resolve().parents[1]


def main():
    train=json.loads((ROOT/'data/entity_training.json').read_text(encoding='utf-8'))
    train=[r for r in train if category(r['reference_category']) is not None]
    train=list({inputs(r):r for r in train}.values())
    dev=[r for r in json.loads((ROOT/'data/entity_evaluation.json').read_text(encoding='utf-8')) if r['split']=='development' and category(r['reference_category']) is not None]
    began=time.monotonic();encoder=SpanEncoder()
    features=encoder.encode(train+dev);x,dx=features[:len(train)],features[len(train):]
    y=[category(r['reference_category']) for r in train];dy=[category(r['reference_category']) for r in dev]
    scaler=StandardScaler();x=scaler.fit_transform(x);dx=scaler.transform(dx)
    best=-1;history=[];destination=ROOT/'analysis/models/tv-span-bert';destination.mkdir(exist_ok=True)
    for c in (.0001,.001,.01,.1,1):
        model=LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=20261017)
        model.fit(x,y);pred=model.predict(dx);f1=macro_f1(dy,pred)
        row=dict(C=c,development_macro_f1=f1,accuracy=float(np.mean(pred==dy)),
                 binary_accuracy=float(np.mean((pred==0)==(np.asarray(dy)==0))))
        history.append(row);print(json.dumps(row),flush=True)
        if f1>best:
            best=f1
            np.savez(destination/'classifier.npz',coef=model.coef_,intercept=model.intercept_,mean=scaler.mean_,scale=scaler.scale_)
            best_c=c;best_pred=pred.tolist();best_probs=model.predict_proba(dx).tolist()
    report=dict(encoder='cointegrated/rubert-base-cased-nli-threeway',encoder_revision='920cbb52ef830e94461bf141ec2119979b6049e2',
        training_examples=len(train),development_examples=len(dev),classes=CLASSES,C=best_c,history=history,elapsed_seconds=round(time.monotonic()-began,2),
        representation='CLS, target subword mean, four left/right contextual token means; frozen encoder, learned classifier',
        annotation_source='Primary LLM directly labelled the snippets, not independent human gold',
        training_data_sha256=hashlib.sha256((ROOT/'data/entity_training.json').read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256((destination/'classifier.npz').read_bytes()).hexdigest(),
        development_predictions=[dict(evaluation_id=r['evaluation_id'],reference=category(r['reference_category']),prediction=p,probabilities=q) for r,p,q in zip(dev,best_pred,best_probs)])
    (destination/'training_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'output/data_cleaning/neural/span_training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
