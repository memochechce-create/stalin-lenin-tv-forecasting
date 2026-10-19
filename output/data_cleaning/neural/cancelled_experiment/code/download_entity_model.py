"""Download only the pinned public checkpoint assets, with verified ranged reads."""
import concurrent.futures
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL='fastino/GLiNER2.5-multi-Decide'
REVISION='e173bca1f0c4217d7a55b3b0c705d5ece8a0218d'
DEST=ROOT/'analysis/models/gliner-multi-decide'
FILES=['config.json','encoder_config/config.json','model.safetensors','tokenizer.json','tokenizer_config.json']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=MODEL)
    p.add_argument('--revision',default=REVISION)
    p.add_argument('--directory',default=str(DEST))
    args=p.parse_args()
    dest=Path(args.directory).resolve()
    if not dest.is_relative_to(ROOT): raise ValueError('Model directory must be inside project')
    dest.mkdir(parents=True,exist_ok=True)
    api=f'https://huggingface.co/api/models/{args.model}/revision/{args.revision}?blobs=true'
    with urllib.request.urlopen(api,timeout=40) as r:
        meta=json.load(r)
    info={r['rfilename']:r for r in meta['siblings']}
    manifest={'model':args.model,'revision':args.revision,'files':{}}
    wanted=[f for f in ('config.json','encoder_config/config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','spm.model') if f in info]
    def digest_file(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        return h.hexdigest()
    for filename in wanted:
        target=dest/filename; target.parent.mkdir(parents=True,exist_ok=True)
        expected=info[filename].get('lfs',{}).get('sha256')
        size=info[filename].get('size')
        if target.exists() and target.stat().st_size==size:
            digest=digest_file(target)
            if not expected or digest==expected:
                manifest['files'][filename]={'bytes':size,'sha256':digest}
                continue
        url=f'https://huggingface.co/{args.model}/resolve/{args.revision}/{filename}'
        part=target.with_suffix(target.suffix+'.part')
        if filename!='model.safetensors':
            with urllib.request.urlopen(url,timeout=60) as r: body=r.read()
            part.write_bytes(body)
        else:
            chunk=8*1024*1024
            tasks=[(i,min(i+chunk,size)-1) for i in range(0,size,chunk)]
            def fetch(bounds):
                a,b=bounds
                for attempt in range(3):
                    try:
                        req=urllib.request.Request(url+f'?download=true&segment={a}',headers={'Range':f'bytes={a}-{b}'})
                        with urllib.request.urlopen(req,timeout=60) as r:
                            content_range=r.headers.get('Content-Range')
                            body=r.read()
                        if content_range!=f'bytes {a}-{b}/{size}' or len(body)!=b-a+1:
                            raise ValueError('unexpected range response')
                        return a,body
                    except Exception:
                        if attempt==2: raise
                        time.sleep(attempt+1)
            with part.open('wb') as f, concurrent.futures.ThreadPoolExecutor(6) as pool:
                done=0
                for first in range(0,len(tasks),12):
                    for offset,body in pool.map(fetch,tasks[first:first+12]):
                        f.seek(offset); f.write(body); done+=1
                    print(f'Model weights {done}/{len(tasks)} chunks',flush=True)
        if part.stat().st_size!=size: raise ValueError('download size mismatch '+filename)
        digest=digest_file(part)
        if expected and digest!=expected: raise ValueError('model SHA256 mismatch')
        part.replace(target)
        manifest['files'][filename]={'bytes':size,'sha256':digest}
    (dest/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
