"""Re-extract candidate mentions from immutable originals, with source checksums.

No network, no third-party packages. Thread workers read different original files.
The deterministic gzip JSONL is reusable by the cleaning and review stages.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
try:
    from .source_database import open_source_database
except ImportError:
    from source_database import open_source_database

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r'(?<!\w)(сталин(?:а|у|ым|ом|е)?|ленин(?:а|у|ым|ом|е)?)(?!\w)', re.I)
TIME = re.compile(r'^(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)')


def parse_source(raw, fmt):
    text = raw.decode('utf-8-sig')
    if fmt != 'srt':
        return text, []
    cues, intervals, pos = [], [], 0
    for block in re.split(r'\r?\n\s*\r?\n', text.strip()):
        if not block.strip():
            continue
        lines = block.splitlines()
        timing = next(((i, TIME.match(line.strip())) for i,line in enumerate(lines) if TIME.match(line.strip())), None)
        if timing is None:
            raise ValueError('SRT block lacks valid timing')
        i, m = timing
        nums = m.groups()
        start = int(nums[0])*3600 + int(nums[1])*60 + int(nums[2]) + float('0.'+nums[3])
        stop = int(nums[4])*3600 + int(nums[5])*60 + int(nums[6]) + float('0.'+nums[7])
        speech = ' '.join(line.strip() for line in lines[i+1:] if line.strip())
        if speech:
            if stop < start:
                # Upstream timing defect: retain the speech and checksum contract,
                # but never fabricate a timestamp for deduplication.
                start, stop = None, None
            cues.append(speech)
            intervals.append((pos, pos+len(speech), start, stop))
            pos += len(speech)+1
    return '\n'.join(cues), intervals


def extract(row):
    sid, fmt = row['id'], row['source_format'] or 'txt'
    path = ROOT/'data_collection/data'/('tv_subtitles' if fmt=='srt' else 'tv_transcripts')/sid[:4]/(sid+('.srt.gz' if fmt=='srt' else '.txt.gz'))
    repair_path = ROOT/'data/cleaned/supplemental_sources'/('tv_subtitles' if fmt=='srt' else 'tv_transcripts')/sid[:4]/path.name
    if repair_path.exists():
        path = repair_path
    try:
        raw = gzip.decompress(path.read_bytes())
        text, cues = parse_source(raw, fmt)
        source_sha = hashlib.sha256(raw).hexdigest()
        text_sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
        # Existing TXT hash includes the UTF-8 BOM if any; preserve that contract.
        legacy_sha = hashlib.sha256(text.encode('utf-8') if fmt=='srt' else raw).hexdigest()
        if source_sha != (row['source_sha256'] or row['sha256']) or legacy_sha != row['sha256']:
            raise ValueError('stored checksum mismatch')
        candidates = []
        lowered = text.lower()
        if 'ленин' in lowered or 'сталин' in lowered:
            cue_i = 0
            for m in PATTERN.finditer(text):
                while cue_i < len(cues) and cues[cue_i][1] <= m.start():
                    cue_i += 1
                cue = cues[cue_i] if cue_i < len(cues) and cues[cue_i][0] <= m.start() < cues[cue_i][1] else None
                candidates.append(dict(person='stalin' if m.group().lower().startswith('сталин') else 'lenin',
                    offset=m.start(), matched=text[m.start():m.end()],
                    context=text[max(0,m.start()-300):m.end()+300],
                    context_offset=min(300,m.start()), start_seconds=cue[2] if cue else None,
                    stop_seconds=cue[3] if cue else None,
                    nonstandard_asr_form=m.group().lower().endswith('ом')))
        return dict(broadcast_id=sid, format=fmt, source_sha256=source_sha,
                    text_sha256=text_sha, chars=len(text), head=text[:2500], tail=text[-2500:],
                    invalid_timing_cues=sum(c[2] is None for c in cues), candidates=candidates, error=None)
    except (OSError, UnicodeError, ValueError) as e:
        return dict(broadcast_id=sid, error=str(e))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    supplemental = ROOT/'data/cleaned/supplemental.sqlite'
    with open_source_database(ROOT/'data_collection/data/tv.sqlite', supplemental) as db:
        rows = [dict(r) for r in db.execute("SELECT id,source_format,sha256,source_sha256 FROM broadcast WHERE status='200' ORDER BY id")]
    if args.limit:
        rows = rows[:args.limit]
    out = ROOT/'data/cleaned'
    out.mkdir(parents=True, exist_ok=True)
    target = out/('source_mentions.sample.jsonl.gz' if args.limit else 'source_mentions.jsonl.gz')
    partial = target.with_suffix('.partial')
    started = time.monotonic()
    report = dict(files=0, candidates=0, errors=[], sample=bool(args.limit), formats={}, nonstandard_forms=0,
                  files_with_invalid_timing=0, invalid_timing_cues=0,
                  extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with partial.open('wb') as raw_out, gzip.GzipFile(filename='',fileobj=raw_out,mode='wb',mtime=0,compresslevel=3) as gz:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            # Bounded batches avoid retaining 114k pending futures and their results.
            for offset in range(0,len(rows),1200):
                for r in pool.map(extract, rows[offset:offset+1200]):
                    gz.write((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8'))
                    report['files'] += 1
                    if r['error']:
                        report['errors'].append([r['broadcast_id'],r['error']])
                    else:
                        report['formats'][r['format']] = report['formats'].get(r['format'],0)+1
                        report['candidates'] += len(r['candidates'])
                        report['nonstandard_forms'] += sum(m['nonstandard_asr_form'] for m in r['candidates'])
                        report['files_with_invalid_timing'] += r['invalid_timing_cues'] > 0
                        report['invalid_timing_cues'] += r['invalid_timing_cues']
                print(f"Verified {report['files']}/{len(rows)} sources; {time.monotonic()-started:.0f}s",flush=True)
    if report['errors']:
        raise RuntimeError(json.dumps(report['errors'][:20],ensure_ascii=False))
    partial.replace(target)
    report['elapsed_seconds'] = round(time.monotonic()-started,2)
    report['output'] = str(target.relative_to(ROOT))
    audit = ROOT/'output/data_cleaning'
    audit.mkdir(parents=True,exist_ok=True)
    (audit/('extraction_sample.json' if args.limit else 'extraction.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
