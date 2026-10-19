"""Bounded repair of recorded missing TV sources and the initial MSK boundary.

Writes only data/cleaned/supplemental* and output/data_cleaning/downloads.json.
Uses public HTTPS with standard certificate validation and at most 8 workers.
"""
import concurrent.futures as cf
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'data_collection'))
from collect_tv import BASE, CHANNELS, PATTERNS, fetch, srt_to_text

BASE_DB = ROOT/'data_collection/data/tv.sqlite'
OVERLAY = ROOT/'data/cleaned/supplemental.sqlite'
SOURCES = ROOT/'data/cleaned/supplemental_sources'
REPORT = ROOT/'output/data_cleaning/downloads.json'
START = dt.datetime(2023, 1, 1)
END = dt.datetime(2026, 10, 3)


def save_raw(folder, name, raw):
    path = SOURCES/folder/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, mtime=0))


def get_inventory(task):
    channel, day = task
    url = BASE+f'{channel}.{day}.inventory.json'
    path = SOURCES/'tv_inventories'/f'{channel}.{day}.json.gz'
    status, raw = (200, gzip.decompress(path.read_bytes())) if path.exists() else fetch(url)
    if status == 200:
        shows = json.loads(raw).get('shows', [])
        save_raw('tv_inventories', path.name, raw)
    else:
        shows = []
    return channel, day, str(status), url, shows


def in_scope(show):
    # Collection timestamps are UTC; the analysis calendar is UTC+03:00.
    start = dt.datetime.fromisoformat(show['start_time']) + dt.timedelta(hours=3)
    return START <= start < END


def get_transcript(row):
    sid = row['id']
    status, raw, fmt, source_url = None, b'', 'txt', BASE+sid+'.transcript.txt'
    for fmt, folder in [('txt', 'tv_transcripts'), ('srt', 'tv_subtitles')]:
        source_url = BASE+sid+'.transcript.'+fmt
        path = SOURCES/folder/sid[:4]/(sid+'.'+fmt+'.gz')
        status, raw = (200, gzip.decompress(path.read_bytes())) if path.exists() else fetch(source_url)
        if status == 200:
            save_raw(folder, f'{sid[:4]}/{sid}.{fmt}.gz', raw)
            break
        if status != 404:
            break
    row = dict(row)
    row.update(status=str(status), url=source_url, retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(), source_format=fmt)
    if status == 200:
        body = srt_to_text(raw).encode('utf-8') if fmt == 'srt' else raw
        text = body.decode('utf-8-sig', errors='replace')
        matches = {person:list(pattern.finditer(text)) for person, pattern in PATTERNS.items()}
        contexts = [{'person':person, 'offset':m.start(), 'matched':m.group(),
                     'context':text[max(0,m.start()-100):m.end()+100]}
                    for person, found in matches.items() for m in found]
        row.update(word_count=len(re.findall(r'\w+',text)), counts=json.dumps({p:len(ms) for p,ms in matches.items()}),
                   contexts=json.dumps(contexts,ensure_ascii=False), sha256=hashlib.sha256(body).hexdigest(),
                   source_sha256=hashlib.sha256(raw).hexdigest())
    return row


def main():
    base = sqlite3.connect(BASE_DB.as_uri()+'?mode=ro', uri=True)
    base.row_factory = sqlite3.Row
    OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    overlay = sqlite3.connect(OVERLAY)
    for name in ('inventory', 'broadcast'):
        overlay.execute(base.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()[0].replace('CREATE TABLE ', 'CREATE TABLE IF NOT EXISTS ', 1))
    columns = [r[1] for r in base.execute('PRAGMA table_info(broadcast)')]
    missing_inventory = [(r['channel'],r['day']) for r in base.execute("SELECT * FROM inventory WHERE status='404'")]
    tasks = sorted(set(missing_inventory + [(channel, '20221231') for channel in CHANNELS]))
    retry_rows = {r['id']:dict(r) for r in base.execute("SELECT * FROM broadcast WHERE status='404'")}
    successful = {r[0] for r in base.execute("SELECT id FROM broadcast WHERE status='200'")}
    candidates = dict(retry_rows)
    report = {'scope':{'inventory404':len(missing_inventory), 'initial_boundary_inventories':4,
                       'transcript404':len(retry_rows), 'msk_start':'2023-01-01', 'msk_end':'2026-10-02', 'workers':8},
              'inventory_results':[], 'transcript_results':[]}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for channel, day, status, url, shows in pool.map(get_inventory, tasks):
            overlay.execute('INSERT OR REPLACE INTO inventory VALUES(?,?,?,?,?)',
                            (channel,day,status,url,dt.datetime.now(dt.timezone.utc).isoformat()))
            selected = [show for show in shows if in_scope(show) and show['id'] not in successful]
            for show in selected:
                if show['id'] not in candidates:
                    row = dict.fromkeys(columns)
                    row.update(id=show['id'],channel=channel,day=day,metadata=json.dumps(show,ensure_ascii=False))
                    candidates[show['id']] = row
            report['inventory_results'].append(dict(channel=channel, day=day, status=status, shows=len(shows), selected=len(selected)))
    overlay.commit()
    print(json.dumps({'stage':'inventories_complete','tasks':len(tasks),'transcript_tasks':len(candidates)}),flush=True)
    statement = 'INSERT OR REPLACE INTO broadcast ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')'
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for i,row in enumerate(pool.map(get_transcript, candidates.values()),1):
            overlay.execute(statement, [row.get(col) for col in columns])
            report['transcript_results'].append(dict(id=row['id'],status=row['status'],source_format=row['source_format'],
                                                      kind='retry404' if row['id'] in retry_rows else 'new'))
            if i % 100 == 0:
                overlay.commit()
                print(json.dumps({'stage':'transcripts','done':i,'total':len(candidates)}),flush=True)
    overlay.commit()
    report['totals'] = {'inventory_attempts':len(tasks), 'inventory_recovered':sum(r['status']=='200' and r['day']!='20221231' for r in report['inventory_results']),
                        'boundary_inventory_success':sum(r['status']=='200' and r['day']=='20221231' for r in report['inventory_results']),
                        'transcript_attempts':len(candidates), 'transcript_success':sum(r['status']=='200' for r in report['transcript_results']),
                        'transcript_retry_recovered':sum(r['status']=='200' and r['kind']=='retry404' for r in report['transcript_results']),
                        'new_transcript_success':sum(r['status']=='200' and r['kind']=='new' for r in report['transcript_results'])}
    report['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['totals']),flush=True)
    overlay.close()
    base.close()


if __name__ == '__main__':
    main()
