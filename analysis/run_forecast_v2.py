"""Version 2 (post-hoc) backtest paths: v1 candidates plus P1/P2 statics.

python analysis/run_forecast_v2.py --phase validation [--jobs 6]
python analysis/run_forecast_v2.py --phase test [--jobs 6]
python analysis/run_forecast_v2.py --phase final

Same origins, seeds and 2048 paths as v1, so M1-M4 reproduce v1 exactly.
Decisions use validation only (docs/memory/project_plan.md, section 3b).
Paths and scores are cached in output/forecast_v2/cache (not tracked).
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time

import numpy as np
import pandas as pd

import joint_count_models as m
import forecast_v2 as v
import run_joint_forecast as rj

OUT = m.ROOT/'output/forecast_v2'
CACHE = OUT/'cache'
KEYS = [p+'_'+t for p in m.PEOPLE for t in ('broadcasts', 'mentions')]
VARIANTS = {'M3H': (True, False), 'M3R': (False, True), 'M3HR': (True, True)}
Q = .015  # frozen in v1 before the outer test


def v2_fits(train, seed, name, source_sha):
    hier, level = VARIANTS[name]
    code = hashlib.sha256(Path(v.__file__).read_bytes()).hexdigest()
    tag = hashlib.sha256(f'{code}|{source_sha}|{train.date.max()}|{seed}|{name}'.encode()).hexdigest()[:20]
    path = CACHE/f'fit_{tag}.npz'; meta = path.with_suffix('.json')
    if path.exists() and meta.exists():
        info = json.loads(meta.read_text(encoding='utf-8'))
        with np.load(path, allow_pickle=False) as z:
            return {c: dict(info[c], theta=z[c], map=z[c+'_map']) for c in m.COMPONENTS}
    fits = v.fit_static(train, seed, hier, level)
    np.savez_compressed(path, **{c: f['theta'] for c, f in fits.items()}, **{c+'_map': f['map'] for c, f in fits.items()})
    rj.dump_json(meta, {c: {k: x for k, x in f.items() if k not in ('theta', 'map')} for c, f in fits.items()})
    return fits


def save_paths(path, paths):
    for key in KEYS:
        if paths[key].max() > np.iinfo(np.int16).max:
            raise ValueError('Path count exceeds int16')
    np.savez_compressed(path, **{key: paths[key].astype(np.int16) for key in KEYS})


def load_paths(phase, origin, model):
    with np.load(CACHE/'paths'/f'{phase}_{pd.Timestamp(origin):%Y%m%d}_{model}.npz') as z:
        return {key: z[key].astype(np.int64) for key in KEYS}


def series_scales(train, days):
    """Daily scale max(1, training mean) as v1; monthly max(1, days*mean)."""
    rows = []
    for c in m.CHANNELS:
        h = train[(train.channel == c) & train.observed]
        for key in KEYS:
            mean = float(h[key].mean())
            rows.append(dict(channel=c, target=key, daily_scale=max(1., mean), monthly_scale=max(1., days*mean)))
    return pd.DataFrame(rows)


def build_origin(payload):
    origin_string, phase, source_sha, models = payload
    origin = pd.Timestamp(origin_string); d, _ = m.load_data()
    train = d[d.date <= origin].copy(); dates = pd.date_range(origin+pd.Timedelta(days=1), origin+pd.offsets.MonthEnd(1))
    actual = d[d.date.isin(dates)].copy(); seed = rj.BASE_SEED+int(origin.strftime('%Y%m')); t0 = time.monotonic()
    tag = origin.strftime('%Y%m%d'); done = []; diags = {}
    fits3 = None
    for name in models:
        target = CACHE/'paths'/f'{phase}_{tag}_{name}.npz'
        if target.exists():
            continue
        if name == 'M1':
            paths = m.conjugate_predict(train, dates, rj.BACKTEST_DRAWS, seed+10)
        elif name == 'M2':
            paths = m.static_predict(rj.fit_cache(train, False, seed, source_sha), dates, rj.BACKTEST_DRAWS, seed+20)
        elif name in ('M3', 'M4'):
            fits3 = fits3 or rj.fit_cache(train, True, seed+1000, source_sha)
            if name == 'M3':
                paths = m.static_predict(fits3, dates, rj.BACKTEST_DRAWS, seed+30)
            else:
                paths, _ = m.dynamic_predict(train, fits3, dates, Q, rj.BACKTEST_DRAWS, seed+40)
        else:
            fits = v2_fits(train, seed+1000, name, source_sha)
            diags[name] = {c: f['diagnostics'] for c, f in fits.items()}
            # Common random numbers with M3 for a paired comparison.
            paths = v.static_predict(fits, dates, rj.BACKTEST_DRAWS, seed+30)
        save_paths(target, paths)
        daily, monthly = m.score_forecast(paths, actual, train, name, origin, phase, seed)
        pd.DataFrame(daily).to_csv(CACHE/'scores'/f'{phase}_{tag}_{name}_daily.csv.gz', index=False)
        pd.DataFrame(monthly).to_csv(CACHE/'scores'/f'{phase}_{tag}_{name}_monthly.csv.gz', index=False)
        done.append(name)
    series_scales(train, len(dates)).assign(origin=origin_string).to_csv(CACHE/'scores'/f'{phase}_{tag}_scales.csv', index=False)
    if diags:
        rj.dump_json(CACHE/f'{phase}_{tag}_v2_diagnostics.json', diags)
    print(f'{phase} {origin.date()}: {done or "cached"} {time.monotonic()-t0:.0f}s', flush=True)
    return origin_string


def final(source_sha, models):
    """Frozen-cutoff 3.10-2.11 paths for v2 statics (v1 paths are reused)."""
    d, _ = m.load_data(); dates = pd.date_range('2026-10-03', '2026-11-02'); seed = rj.BASE_SEED+99999
    for name in models:
        target = OUT/f'final_paths_{name}.npz'
        fits = v2_fits(d, seed+1000, name, source_sha)
        paths = v.static_predict(fits, dates, rj.FINAL_DRAWS, seed+30)
        np.savez_compressed(target, dates=np.array(dates.strftime('%Y-%m-%d'), dtype='U10'),
                            **{key: paths[key].astype(np.int32) for key in KEYS})
        rj.dump_json(OUT/f'final_{name}_diagnostics.json', {c: f['diagnostics'] for c, f in fits.items()})
        summary = {c: {k: f['diagnostics'][k] for k in ('max_rhat', 'min_bulk_ess', 'divergences')} for c, f in fits.items()}
        print(name, json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('validation', 'test', 'final'), required=True)
    parser.add_argument('--models', default='M1,M2,M3,M4,M3H,M3R,M3HR')
    parser.add_argument('--jobs', type=int, default=6)
    args = parser.parse_args()
    for p in (CACHE/'paths', CACHE/'scores', rj.OUT/'cache'):
        p.mkdir(parents=True, exist_ok=True)
    d, source_sha = m.load_data(); models = args.models.split(',')
    if args.phase == 'final':
        return final(source_sha, [x for x in models if x in VARIANTS])
    ends = rj.VALID_ENDS if args.phase == 'validation' else rj.TEST_ENDS
    payloads = [(str(e.date()), args.phase, source_sha, models) for e in ends]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for future in as_completed([pool.submit(build_origin, p) for p in payloads]):
            future.result()


if __name__ == '__main__':
    main()
