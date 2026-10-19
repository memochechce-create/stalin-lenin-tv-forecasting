"""Frozen Russian BERT span representations plus a supervised small classifier."""
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class SpanEncoder:
    def __init__(self,threads=4):
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
        import torch
        from transformers import AutoTokenizer,AutoModel
        torch.set_num_threads(threads);self.torch=torch
        path=ROOT/'analysis/models/rubert-base-nli'
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModel.from_pretrained(path,local_files_only=True).eval()

    def encode(self,rows,batch_size=12):
        features=[]
        with self.torch.inference_mode():
            for start in range(0,len(rows),batch_size):
                batch=rows[start:start+batch_size];texts=[];names=[];positions=[]
                for m in batch:
                    p=m['context_offset'];e=p+len(m['matched']);a=max(0,p-160)
                    if m['context'][p:e]!=m['matched']:raise ValueError('Bad span')
                    texts.append(m['context'][a:e+160]);positions.append((p-a,e-a))
                    names.append({'lenin':'Владимир Ильич Ленин','stalin':'Иосиф Виссарионович Сталин'}[m['person']])
                tokens=self.tokenizer(texts,names,padding=True,truncation='only_first',max_length=256,
                    return_offsets_mapping=True,return_tensors='pt')
                offsets=tokens.pop('offset_mapping')
                hidden=self.model(**tokens).last_hidden_state
                for i,(p,e) in enumerate(positions):
                    seq=tokens.sequence_ids(i)
                    indices=[j for j,(a,b) in enumerate(offsets[i].tolist()) if seq[j]==0 and a<e and b>p]
                    if not indices:raise ValueError('Target was truncated')
                    left=[j for j in range(max(1,indices[0]-4),indices[0]) if seq[j]==0] or indices
                    right=[j for j in range(indices[-1]+1,min(indices[-1]+5,len(seq))) if seq[j]==0] or indices
                    feature=self.torch.cat([hidden[i,0],hidden[i,indices].mean(0),hidden[i,left].mean(0),hidden[i,right].mean(0)])
                    features.append(feature.cpu().numpy())
        return np.asarray(features)
