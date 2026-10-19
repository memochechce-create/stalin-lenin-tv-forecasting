"""Fine-tune Russian BERT on directly LLM-annotated TV contexts; test split stays sealed."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

ROOT=Path(__file__).resolve().parents[1]
CLASSES=['historical_person','non_target_reference','commemoration']


def category(reference):
    if reference=='unclear':return None
    if reference=='historical_person':return 0
    if reference=='commemoration':return 2
    return 1


def inputs(m,window=160):
    p=m['context_offset'];e=p+len(m['matched'])
    if m['context'][p:e]!=m['matched']:raise ValueError('Bad target offset')
    text=m['context'][max(0,p-window):p]+' [MENTION] '+m['matched']+' [/MENTION] '+m['context'][e:e+window]
    return ' '.join(text.split()),{'lenin':'Владимир Ильич Ленин','stalin':'Иосиф Виссарионович Сталин'}[m['person']]


def macro_f1(gold,pred):
    f=[]
    for k in range(3):
        tp=sum(a==b==k for a,b in zip(gold,pred));fp=sum(a!=k and b==k for a,b in zip(gold,pred));fn=sum(a==k and b!=k for a,b in zip(gold,pred))
        f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return sum(f)/3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--backbone',choices=('tiny','base'),default='tiny')
    args=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(4);torch.manual_seed(20261017);random.seed(20261017)
    train=json.loads((ROOT/'data/entity_training.json').read_text(encoding='utf-8'))
    dev=[r for r in json.loads((ROOT/'data/entity_evaluation.json').read_text(encoding='utf-8')) if r['split']=='development']
    train=[r for r in train if category(r['reference_category']) is not None]
    dev=[r for r in dev if category(r['reference_category']) is not None]
    # Remove exact repeat contexts rather than allowing repeated TV stories to dominate.
    train=list({inputs(r):r for r in train}.values())
    initial=ROOT/'analysis/models'/('rubert-tiny-nli' if args.backbone=='tiny' else 'rubert-base-nli')
    tokenizer=AutoTokenizer.from_pretrained(initial,local_files_only=True)
    tokenizer.add_special_tokens({'additional_special_tokens':['[MENTION]','[/MENTION]']})
    model=AutoModelForSequenceClassification.from_pretrained(initial,local_files_only=True,num_labels=3,
        id2label=dict(enumerate(CLASSES)),label2id={v:k for k,v in enumerate(CLASSES)},ignore_mismatched_sizes=True)
    model.resize_token_embeddings(len(tokenizer),mean_resizing=False)
    if args.backbone=='base':
        model.classifier=torch.nn.Linear(model.config.hidden_size,3)
        model._init_weights(model.classifier)
        for parameter in model.bert.embeddings.parameters():parameter.requires_grad=False
        for block in model.bert.encoder.layer[:-4]:
            for parameter in block.parameters():parameter.requires_grad=False
    def tokenize(rows):
        pairs=[inputs(r) for r in rows]
        encoded=tokenizer([r[0] for r in pairs],[r[1] for r in pairs],padding=True,truncation='only_first',max_length=256,return_tensors='pt')
        return encoded,torch.tensor([category(r['reference_category']) for r in rows])
    x,y=tokenize(train);dx,dy=tokenize(dev)
    freq=Counter(y.tolist());weights=torch.tensor([math.sqrt(len(train)/(3*freq[k])) for k in range(3)])
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    groups=[dict(params=[v for k,v in model.named_parameters() if v.requires_grad and not k.startswith('classifier')],lr=3e-5 if args.backbone=='base' else 5e-5),
            dict(params=list(model.classifier.parameters()),lr=5e-4 if args.backbone=='base' else 5e-5)]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    destination=ROOT/'analysis/models'/('tv-entity-bert' if args.backbone=='tiny' else 'tv-entity-bert-base');destination.mkdir(parents=True,exist_ok=True)
    history=[];best=-1;best_epoch=0;started=time.monotonic()
    for epoch in range(args.epochs):
        model.train();order=torch.randperm(len(y));loss_sum=0
        batch_size=8 if args.backbone=='base' else 16
        for start in range(0,len(order),batch_size):
            ix=order[start:start+batch_size];optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(**{k:v[ix] for k,v in x.items()}).logits,y[ix])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            loss_sum+=loss.item()*len(ix)
        model.eval()
        with torch.inference_mode():
            probs=model(**dx).logits.softmax(-1);pred=probs.argmax(-1).tolist()
        f1=macro_f1(dy.tolist(),pred)
        row=dict(epoch=epoch+1,loss=loss_sum/len(y),development_macro_f1=f1,
                 development_accuracy=sum(a==b for a,b in zip(dy.tolist(),pred))/len(pred))
        history.append(row);print(json.dumps(row),flush=True)
        if f1>best:
            best=f1;best_epoch=epoch+1
            model.save_pretrained(destination,safe_serialization=True);tokenizer.save_pretrained(destination)
        if epoch+1-best_epoch>=4:break
    report=dict(initial_model='cointegrated/rubert-tiny-bilingual-nli' if args.backbone=='tiny' else 'cointegrated/rubert-base-cased-nli-threeway',initial_revision='3907306f5e5d65e281a524c536883b766c89ded2' if args.backbone=='tiny' else '920cbb52ef830e94461bf141ec2119979b6049e2',
                training_examples=len(train),development_examples=len(dev),classes=CLASSES,best_epoch=best_epoch,
                annotation_source='Primary LLM agent directly read and labelled real ASR contexts. No independent human-gold claim.',
                seed=20261017,context_window=160,history=history,elapsed_seconds=round(time.monotonic()-started,2),
                training_data_sha256=hashlib.sha256((ROOT/'data/entity_training.json').read_bytes()).hexdigest(),
                evaluation_data_sha256=hashlib.sha256((ROOT/'data/entity_evaluation.json').read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256((destination/'model.safetensors').read_bytes()).hexdigest())
    (destination/'training_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    out=ROOT/'output/data_cleaning/neural'
    (out/('training_report.json' if args.backbone=='tiny' else 'base_training_report.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='history'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
