"""Reproduce the complete corrected-data forecasting project from the terminal.

python analysis/run_joint_forecast.py [--jobs 3] [--rebuild]
Numerical caches are content-addressed by model source, input SHA and settings.
No network or credentials are needed. --rebuild ignores existing fit caches.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

import joint_count_models as m

OUT = m.ROOT/'output/joint_forecast'
FIT_VERSION = 'coherent-v1'
VALID_ENDS = pd.date_range('2024-06-30', '2024-11-30', freq='ME')
TEST_ENDS = pd.date_range('2024-12-31', '2026-08-31', freq='ME')
Q_GRID = (.015, .04, .08)
BACKTEST_DRAWS = 2048
FINAL_DRAWS = 30000
BASE_SEED = 20261017


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fit_cache(train, calendar, seed, source_sha, rebuild=False, suffix='', prior_scale=1.):
    code_sha = hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
    tag = hashlib.sha256(f'{FIT_VERSION}|{code_sha}|{source_sha}|{train.date.max()}|{calendar}|{seed}|{suffix}|{prior_scale}'.encode()).hexdigest()[:20]
    path = OUT/'cache'/f'fit_{tag}.npz'; meta = path.with_suffix('.json')
    if path.exists() and meta.exists() and not rebuild:
        info = json.loads(meta.read_text(encoding='utf-8')); fits = {}
        with np.load(path, allow_pickle=False) as z:
            for name in m.COMPONENTS:
                fits[name] = dict(info[name], theta=z[name], map=z[name+'_map'])
        return fits
    fits = m.fit_static(train, calendar, seed, prior_scale)
    np.savez_compressed(path, **{k: v['theta'] for k, v in fits.items()}, **{k+'_map': v['map'] for k, v in fits.items()})
    dump_json(meta, {k: {name: val for name, val in v.items() if name not in ('theta', 'map')} for k, v in fits.items()})
    return fits


def build_origin(payload):
    origin_string, phase, source_sha, rebuild, q_final = payload
    origin = pd.Timestamp(origin_string); d, _ = m.load_data()
    train = d[d.date <= origin].copy(); dates = pd.date_range(origin+pd.Timedelta(days=1), origin+pd.offsets.MonthEnd(1))
    actual = d[d.date.isin(dates)].copy(); seed = BASE_SEED+int(origin.strftime('%Y%m'))
    t0 = time.monotonic()
    fits2 = fit_cache(train, False, seed, source_sha, rebuild)
    fits3 = fit_cache(train, True, seed+1000, source_sha, rebuild)
    forecasts = {'M1': m.conjugate_predict(train, dates, BACKTEST_DRAWS, seed+10),
                 'M2': m.static_predict(fits2, dates, BACKTEST_DRAWS, seed+20),
                 'M3': m.static_predict(fits3, dates, BACKTEST_DRAWS, seed+30)}
    rows = []; totals = []; q_rows = []; filter_diags = {}
    for name, paths in forecasts.items():
        r, z = m.score_forecast(paths, actual, train, name, origin, phase, seed)
        rows.extend(r); totals.extend(z)
    for q in (Q_GRID if phase == 'validation' else [q_final]):
        paths, diag = m.dynamic_predict(train, fits3, dates, q, BACKTEST_DRAWS, seed+40)
        filter_diags[str(q)] = diag
        name = f'M4_q{q:g}' if phase == 'validation' else 'M4'
        r, z = m.score_forecast(paths, actual, train, name, origin, phase, seed)
        rows.extend(r); totals.extend(z)
        if phase == 'validation':
            q_rows.append(dict(origin=origin_string, q=q, scaled_crps=float(np.mean([v['scaled_crps'] for v in r]))))
        # Keep validation mixtures paired with their M4 candidate; freeze q
        # first, then select a model using only validation scores.
        pool = m.linear_pool(forecasts | {'M4': paths}, seed+50)
        name = f'POOL_q{q:g}' if phase == 'validation' else 'POOL'
        r, z = m.score_forecast(pool, actual, train, name, origin, phase, seed)
        rows.extend(r); totals.extend(z)
    tag = origin.strftime('%Y%m%d')
    pd.DataFrame(rows).to_csv(OUT/'cache'/f'{phase}_{tag}_daily.csv.gz', index=False)
    pd.DataFrame(totals).to_csv(OUT/'cache'/f'{phase}_{tag}_monthly.csv.gz', index=False)
    diag = {'origin': origin_string, 'phase': phase, 'train_end': str(train.date.max().date()),
            'train_observed_channel_days': int(train.observed.sum()),
            'M2': {k: v['diagnostics'] for k, v in fits2.items()},
            'M3': {k: v['diagnostics'] for k, v in fits3.items()},
            'filters': filter_diags, 'elapsed_seconds': time.monotonic()-t0}
    dump_json(OUT/'cache'/f'{phase}_{tag}_diagnostics.json', diag)
    print(f'{phase}: {origin.date()} -> {dates[-1].date()}, {time.monotonic()-t0:.1f}s', flush=True)
    return rows, totals, q_rows, diag


def run_origins(ends, phase, source_sha, rebuild, jobs, q=None):
    payloads = [(str(d.date()), phase, source_sha, rebuild, q) for d in ends]
    if jobs == 1:
        return [build_origin(p) for p in payloads]
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        tasks = {pool.submit(build_origin, p): p[0] for p in payloads}
        for future in as_completed(tasks):
            results.append(future.result())
    return sorted(results, key=lambda x: x[3]['origin'])


def flatten(results, i):
    return pd.DataFrame([row for result in results for row in result[i]])


def evaluation_tables(daily, monthly):
    def metrics(g):
        return pd.Series(dict(n=len(g), origins=g.origin.nunique(), rmse=np.sqrt(g.se.mean()),
                              crps=g.crps.mean(), scaled_crps=g.scaled_crps.mean(),
                              scaled_rmse=np.sqrt(g.scaled_se.mean()), coverage80=g.coverage80.mean(),
                              coverage95=g.coverage95.mean(), width80=g.width80.mean(), width95=g.width95.mean()))
    by_series = daily.groupby(['model', 'channel', 'target']).apply(metrics, include_groups=False).reset_index()
    overall = daily.groupby('model').apply(metrics, include_groups=False).reset_index()
    horizons = daily.assign(horizon_band=pd.cut(daily.horizon, [0, 7, 14, 21, 31], labels=['1–7', '8–14', '15–21', '22–31']))
    horizon_scores = horizons.groupby(['model', 'horizon_band'], observed=True).apply(metrics, include_groups=False).reset_index()
    return overall, by_series, horizon_scores


def dm_tests(daily):
    """Paired origin-level losses, Bartlett HAC lag1; 21 calendar blocks.

    These are approximate large-sample tests, exploratory with few blocks.
    Holm correct the four baseline comparisons for each prespecified loss.
    """
    rows = []
    for loss in ('scaled_crps', 'scaled_se'):
        table = daily.groupby(['origin', 'model'])[loss].mean().unstack()
        for name in ('M2', 'M3', 'M4', 'POOL'):
            diff = (table['M1']-table[name]).to_numpy(); T = len(diff); mean = diff.mean(); u = diff-mean
            variance = np.dot(u, u)/T + np.dot(u[1:], u[:-1])/T
            if variance <= 0:
                raise RuntimeError('Nonpositive DM long-run variance')
            stat = mean/np.sqrt(variance/T)
            rows.append(dict(loss=loss, baseline='M1', model=name, origins=T, hac_lag=1,
                             improvement=float(mean), dm_statistic=float(stat), p_value=float(2*norm.sf(abs(stat)))))
    result = pd.DataFrame(rows); result['p_holm'] = np.nan
    for _, group in result.groupby('loss'):
        order = group.sort_values('p_value').index; raw = result.loc[order, 'p_value'].to_numpy()
        result.loc[order, 'p_holm'] = np.minimum(1, np.maximum.accumulate(raw*np.arange(len(raw), 0, -1)))
    return result


def forecast_rows(all_paths, dates, scenario='main'):
    daily = []; monthly = []
    for model, paths in all_paths.items():
        for target, samples in paths.items():
            summary = m.summarize_paths(samples); totals = m.summarize_paths(samples.sum(axis=1))
            for c, channel in enumerate(m.CHANNELS):
                for t, day in enumerate(dates):
                    daily.append(dict(scenario=scenario, model=model, day=str(day.date()), channel=channel, target=target,
                                      **{k: float(v[t, c]) for k, v in summary.items()}))
                monthly.append(dict(scenario=scenario, model=model, channel=channel, target=target,
                                    **{k: float(v[c]) for k, v in totals.items()}))
    return daily, monthly


def save_paths(path, forecasts, dates):
    np.savez_compressed(path, dates=np.array(dates.strftime('%Y-%m-%d'), dtype='U10'),
                        **{model+'__'+key: value.astype(np.int32) for model, paths in forecasts.items() for key, value in paths.items()})


def final_forecasts(d, source_sha, selected, q, rebuild=False, possible=False):
    dates = pd.date_range('2026-10-03', '2026-11-02'); seed = BASE_SEED+99999
    suffix = 'possible' if possible else 'main'
    # Same random streams allow cleaner paired sensitivity comparisons.
    f2 = fit_cache(d, False, seed, source_sha, rebuild, suffix)
    f3 = fit_cache(d, True, seed+1000, source_sha, rebuild, suffix)
    paths = {'M1': m.conjugate_predict(d, dates, FINAL_DRAWS, seed+10),
             'M2': m.static_predict(f2, dates, FINAL_DRAWS, seed+20),
             'M3': m.static_predict(f3, dates, FINAL_DRAWS, seed+30)}
    paths['M4'], diag = m.dynamic_predict(d, f3, dates, q, FINAL_DRAWS, seed+40, groups=64, particles=96, trace=not possible)
    paths['POOL'] = m.linear_pool(paths, seed+50)
    # Keep complete selected paths for sensitivity, all models for main.
    saved = {selected: paths[selected]} if possible else paths
    save_paths(OUT/('forecast_paths_possible.npz' if possible else 'forecast_paths.npz'), saved, dates)
    rows, totals = forecast_rows(saved, dates, suffix)
    dump_json(OUT/f'final_{suffix}_diagnostics.json', {'M2': {k: v['diagnostics'] for k, v in f2.items()},
                                                     'M3': {k: v['diagnostics'] for k, v in f3.items()}, 'filters': diag})
    if not possible:
        pd.DataFrame([row for v in diag for row in v['history']]).to_csv(OUT/'filtered_states.csv.gz', index=False)
        # Finite-particle sensitivity changes both static groups and particles.
        check, _ = m.dynamic_predict(d, f3, dates, q, 10000, seed+640, groups=96, particles=128)
        comparison = []
        for key in check:
            a = paths['M4'][key].sum(axis=1); b = check[key].sum(axis=1)
            for c, channel in enumerate(m.CHANNELS):
                se = np.sqrt(a[:, c].var()/len(a)+b[:, c].var()/len(b))
                comparison.append(dict(channel=channel, target=key, original_mean=float(a[:, c].mean()),
                                       more_particles_mean=float(b[:, c].mean()),
                                       relative_difference=float((b[:, c].mean()-a[:, c].mean())/max(1, a[:, c].mean())),
                                       simulation_se_conditional=float(se)))
        pd.DataFrame(comparison).to_csv(OUT/'particle_sensitivity.csv', index=False)
    return rows, totals


def run_experiment(jobs=3, rebuild=False):
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'cache').mkdir(exist_ok=True)
    d, source_sha = m.load_data(); started = time.monotonic()
    if str(d.date.max().date()) != '2026-10-02':
        raise ValueError('This proposal fixes the data cutoff to 2026-10-02')
    print('Development: six historical calendar months; q and model frozen before the outer test.', flush=True)
    valid_results = run_origins(VALID_ENDS, 'validation', source_sha, rebuild, jobs)
    valid = flatten(valid_results, 0); q_scores = flatten(valid_results, 2)
    chosen_q = float(q_scores.groupby('q').scaled_crps.mean().idxmin())
    use = valid.model.isin(['M1', 'M2', 'M3', f'M4_q{chosen_q:g}', f'POOL_q{chosen_q:g}'])
    valid = valid[use].copy(); valid['model'] = valid.model.replace({f'M4_q{chosen_q:g}': 'M4', f'POOL_q{chosen_q:g}': 'POOL'})
    valid_scores = valid.groupby('model').scaled_crps.mean().sort_values(); selected = str(valid_scores.index[0])
    dump_json(OUT/'selection.json', dict(q=chosen_q, selected_model=selected, criterion='mean daily CRPS / max(training series mean,1)',
                                       validation_months=6, validation_scores=valid_scores.to_dict(),
                                       pool_weights={k: .25 for k in ('M1', 'M2', 'M3', 'M4')}))
    q_scores.to_csv(OUT/'q_validation.csv', index=False); valid.to_csv(OUT/'validation_daily.csv.gz', index=False)
    print(f'Frozen selection: q={chosen_q:g}; {selected}. Outer test: 21 new months.', flush=True)
    test_results = run_origins(TEST_ENDS, 'test', source_sha, rebuild, jobs, chosen_q)
    test = flatten(test_results, 0); monthly = flatten(test_results, 1)
    test.to_csv(OUT/'backtest_daily.csv.gz', index=False); monthly.to_csv(OUT/'backtest_monthly.csv', index=False)
    overall, by_series, horizons = evaluation_tables(test, monthly)
    overall.to_csv(OUT/'evaluation.csv', index=False); by_series.to_csv(OUT/'evaluation_by_series.csv', index=False)
    horizons.to_csv(OUT/'evaluation_by_horizon.csv', index=False); dm_tests(test).to_csv(OUT/'dm_tests.csv', index=False)
    dump_json(OUT/'backtest_diagnostics.json', [v[3] for v in valid_results+test_results])
    print('Final frozen-cutoff forecast and context-label sensitivity.', flush=True)
    rows, totals = final_forecasts(d, source_sha, selected, chosen_q, rebuild)
    possible, _ = m.load_data(possible=True)
    extra_rows, extra_totals = final_forecasts(possible, source_sha, selected, chosen_q, rebuild, True)
    pd.DataFrame(rows+extra_rows).to_csv(OUT/'forecast_daily.csv', index=False)
    pd.DataFrame(totals+extra_totals).to_csv(OUT/'forecast_month.csv', index=False)
    d.to_csv(OUT/'analysis_data.csv.gz', index=False)
    import scipy, matplotlib, arviz
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    scipy=scipy.__version__, matplotlib=matplotlib.__version__, arviz=arviz.__version__)
    manifest = dict(source_sha256=source_sha, code_sha256=m.provenance(), seed=BASE_SEED,
                    data_cutoff='2026-10-02', forecast_start='2026-10-03', forecast_end='2026-11-02',
                    retrospective_frozen_cutoff=True, validation='2024-07 through 2024-12',
                    test='2025-01 through 2026-09', test_origins=21, q=chosen_q, selected_model=selected,
                    backtest_draws=BACKTEST_DRAWS, final_draws=FINAL_DRAWS, versions=versions,
                    elapsed_seconds=time.monotonic()-started,
                    inference='exact static posterior MH; M4 conditional/cut state filters, not full joint Bayes',
                    target='provisional rule-selected surname contexts per MSK archive-start day',
                    outputs_sha256={name: hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                                    for name in ('forecast_daily.csv', 'forecast_month.csv',
                                                 'forecast_paths.npz', 'forecast_paths_possible.npz')})
    dump_json(OUT/'manifest.json', manifest)
    print(overall.to_string(index=False), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    run_experiment(max(1, args.jobs), args.rebuild)
    from verify_joint_forecast import verify
    verify()
    from forecast_artifacts import build_artifacts
    build_artifacts()
