"""Create a compact, offline-reproducible submission (no corpus or fit cache)."""
from pathlib import Path
import json
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/joint_forecast'


def package():
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    files = {ROOT/'analysis'/name for name in manifest['code_sha256']}
    files.update({Path(__file__).resolve(), ROOT/'requirements-forecast.txt',
                  ROOT/'tests/test_joint_forecast.py', ROOT/'data/cleaned/tv_daily_targets.csv.gz'})
    files.update(p for p in OUT.rglob('*') if p.is_file()
                 and not any(part in ('cache', 'preview') for part in p.relative_to(OUT).parts)
                 and p.suffix != '.zip')
    for folder in ('output/proposal', 'output/data_cleaning'):
        base = ROOT/folder
        files.update(p for p in base.glob('*') if p.is_file() and p.suffix in ('.md', '.json', '.pdf', '.tex'))
    files.add(ROOT/'analysis/mention_rules.py')
    for p in files:
        if not p.is_file() or not p.resolve().is_relative_to(ROOT):
            raise ValueError(f'Invalid bundle member: {p}')
    readme = '''# Final forecasting project

Open output/joint_forecast/forecast_report.html for the visual results.
Open output/joint_forecast/final_project.ipynb for the explained course report.

Unzip this entire folder before running. Python 3.12/3.13, from this root:

    python -m pip install -r requirements-forecast.txt
    python analysis/verify_joint_forecast.py
    python analysis/run_joint_forecast.py --jobs 3
    python analysis/execute_forecast_notebook.py
    python -m unittest discover -s tests -p test_joint_forecast.py -v

The supplied cleaned CSV is sufficient to reproduce every forecasting computation
offline. A full rebuild is CPU-intensive; --rebuild forces fresh posterior fitting.
Data cutoff: 2026-10-02. Forecast: 2026-10-03 through 2026-11-02.
The full original transcript corpus and rebuildable posterior caches are excluded.
Data preparation notes and the active contextual rules are supplied for provenance;
this compact bundle starts at the verified daily-data stage. The full repository
contains the original collection/cleaning scripts and locally retained corpus.
The legacy verify_forecast.py, if present, is retained only because its source hash
was captured in provenance; the entry point here is verify_joint_forecast.py.
'''
    destination = OUT/'submission_bundle.zip'
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr('README.md', readme)
        for p in sorted(files):
            z.write(p, p.relative_to(ROOT).as_posix())
    with ZipFile(destination) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('Corrupt ZIP member: '+bad)
    print(f'Submission bundle verified: {destination} ({destination.stat().st_size/1e6:.1f} MB)')
    return destination


if __name__ == '__main__':
    package()
