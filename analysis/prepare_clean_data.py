"""One-command offline build. Use --retry-downloads for bounded public-source repair."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run(name,*args):
    subprocess.run([sys.executable,str(ROOT/'analysis'/name),*args],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retry-downloads',action='store_true')
    p.add_argument('--refresh-sources',action='store_true')
    args=p.parse_args()
    if args.retry_downloads:
        run('repair_tv_downloads.py')
    cache=ROOT/'data/cleaned/source_mentions.jsonl.gz'
    report=ROOT/'output/data_cleaning/extraction.json'
    refresh=args.refresh_sources or args.retry_downloads or not cache.exists() or not report.exists()
    if not refresh:
        info=json.loads(report.read_text(encoding='utf-8'))
        if info.get('errors') or info.get('sample'):
            refresh=True
        # New repair source files require a new complete checksum/candidate pass.
        overlay=ROOT/'data/cleaned/supplemental.sqlite'
        if overlay.exists() and overlay.stat().st_mtime > cache.stat().st_mtime:
            refresh=True
    if refresh:
        run('extract_tv_mentions.py')
    run('clean_tv_data.py')
    run('verify_clean_data.py')
    print('Prepared data/cleaned/tv_daily_targets.csv.gz. Context labels remain provisional; see output/data_cleaning/README.md.')


if __name__=='__main__': main()
