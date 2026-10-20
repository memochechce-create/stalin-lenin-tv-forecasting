"""Version 2 evaluation, acceptance gates, stacking and event recalibration.

python analysis/evaluate_forecast_v2.py --step p1|p2|p3|p4|p5|posthoc|final

Rules are pre-registered in output/forecast_v2/README.md. Every
decision uses validation only; the test is scored afterwards and reported.
Step posthoc records the authors' choice made after seeing the test (P2 model
M3HR plus P5 recalibration of its events); it is reported as post-hoc and
leaves the frozen p1..p5 decisions untouched.
D  = mean daily CRPS / max(1, training series mean) (v1 criterion).
Mo = mean monthly-total CRPS / max(1, days * training series mean).
For empirical integer distributions CRPS(F, y) = sum_z (F(z) - 1{y <= z})^2,
so the CRPS of a mixture sum_k w_k F_k is the quadratic form w'Gw (exact).
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import joint_count_models as m
import run_forecast_v2 as rv
import run_joint_forecast as rj

OUT = rv.OUT
PRIOR_PERIODS = 30  # L6 p. 33: prior strength of equal weights
EVENT_EPS = .005     # L6 p. 65: move 0 and 1 slightly inside before probits


def ends(phase):
    return rj.VALID_ENDS if phase == 'validation' else rj.TEST_ENDS


def scores(phase, models):
    daily, monthly = [], []; data, _ = m.load_data()
    for e in ends(phase):
        tag = f'{phase}_{e:%Y%m%d}'; days = (e+pd.offsets.MonthEnd(1)-e).days
        scale = rv.series_scales(data[data.date <= e], days)
        for model in models:
            daily.append(pd.read_csv(rv.CACHE/'scores'/f'{tag}_{model}_daily.csv.gz'))
            try:  # a month with a missing day has no complete totals
                z = pd.read_csv(rv.CACHE/'scores'/f'{tag}_{model}_monthly.csv.gz')
            except pd.errors.EmptyDataError:
                continue
            if len(z):
                monthly.append(z.merge(scale[['channel', 'target', 'monthly_scale']], on=['channel', 'target']))
    daily = pd.concat(daily); monthly = pd.concat(monthly)
    monthly['scaled_crps'] = monthly.crps/monthly.monthly_scale
    monthly['coverage80'] = (monthly.lo80 <= monthly.actual) & (monthly.actual <= monthly.hi80)
    monthly['coverage95'] = (monthly.lo95 <= monthly.actual) & (monthly.actual <= monthly.hi95)
    return daily, monthly


def metrics(daily, monthly):
    a = daily.groupby('model').agg(D=('scaled_crps', 'mean'), daily_cov80=('coverage80', 'mean'),
                                   daily_cov95=('coverage95', 'mean'), n_daily=('scaled_crps', 'size'))
    b = monthly.groupby('model').agg(Mo=('scaled_crps', 'mean'), monthly_cov80=('coverage80', 'mean'),
                                     monthly_cov95=('coverage95', 'mean'), n_monthly=('scaled_crps', 'size'))
    return a.join(b)


def passes(candidate, best):
    """Pre-registered rule: neither metric worse, at least one strictly better."""
    return bool(candidate['D'] <= best['D'] and candidate['Mo'] <= best['Mo']
                and (candidate['D'] < best['D'] or candidate['Mo'] < best['Mo']))


def gram(samples, y):
    """Per-observation K x K CRPS forms; samples: list of (S_k, n) arrays."""
    z = np.arange(0, int(max(max(s.max() for s in samples), y.max()))+1)
    diff = []
    for s in samples:
        s = np.sort(s, axis=0)
        F = np.stack([np.searchsorted(s[:, j], z, side='right') for j in range(s.shape[1])])/len(s)
        diff.append(F-(y[:, None] <= z[None, :]))
    diff = np.stack(diff)
    return np.einsum('knz,lnz->nkl', diff, diff)


def origin_forms(phase, origin, models, data):
    """Sum of scaled daily and monthly CRPS forms for one origin."""
    tag = pd.Timestamp(origin); dates = pd.date_range(tag+pd.Timedelta(days=1), tag+pd.offsets.MonthEnd(1))
    train = data[data.date <= tag]; actual = data[data.date.isin(dates)]
    scale = rv.series_scales(train, len(dates)).set_index(['channel', 'target'])
    paths = {k: rv.load_paths(phase, origin, k) for k in models}
    K = len(models); Gd = np.zeros((K, K)); Gm = np.zeros((K, K)); nd = nm = 0
    for c, channel in enumerate(m.CHANNELS):
        target = actual[actual.channel == channel].sort_values('date')
        ok = target.observed.to_numpy(bool)
        for key in rv.KEYS:
            y = target[key].to_numpy(float); sc = scale.loc[(channel, key)]
            if ok.any():
                Gd += gram([paths[k][key][:, ok, c] for k in models], y[ok]).sum(0)/sc.daily_scale; nd += ok.sum()
            if ok.all():
                Gm += gram([paths[k][key][:, :, c].sum(1, keepdims=True) for k in models],
                           np.array([y.sum()])).sum(0)/sc.monthly_scale; nm += 1
    return Gd, nd, Gm, nm


def forms(phase, models):
    path = rv.CACHE/f'forms_{phase}_{"-".join(models)}.npz'
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    data, _ = m.load_data(); rows = [origin_forms(phase, str(e.date()), models, data) for e in ends(phase)]
    result = {name: np.array([r[i] for r in rows]) for i, name in enumerate(('Gd', 'nd', 'Gm', 'nm'))}
    np.savez(path, **result)
    return result


def mixture_scores(f, w, use=None):
    use = np.ones(len(f['nd']), bool) if use is None else use
    return dict(D=float(sum(w@g@w for g in f['Gd'][use])/f['nd'][use].sum()),
                Mo=float(sum(w@g@w for g in f['Gm'][use])/f['nm'][use].sum()))


def stack_weights(f, use):
    """Mixture weights minimizing held-in daily CRPS, shrunk to equal (L6)."""
    G = f['Gd'][use].sum(0)/f['nd'][use].sum(); K = len(G)
    fit = minimize(lambda w: w@G@w, np.full(K, 1/K), jac=lambda w: 2*G@w, method='SLSQP',
                   bounds=[(0, 1)]*K, constraints=({'type': 'eq', 'fun': lambda w: w.sum()-1},))
    raw = np.clip(fit.x, 0, None); raw /= raw.sum(); n = int(use.sum())
    return raw, n/(n+PRIOR_PERIODS)*raw+PRIOR_PERIODS/(n+PRIOR_PERIODS)/K


def loo_stacking(f):
    """Leave one validation origin out: weights from the other origins."""
    T = len(f['nd']); w_out = []
    num_d = num_m = 0.
    for o in range(T):
        use = np.arange(T) != o; w = stack_weights(f, use)[1]; w_out.append(w)
        num_d += w@f['Gd'][o]@w; num_m += w@f['Gm'][o]@w
    return dict(D=float(num_d/f['nd'].sum()), Mo=float(num_m/f['nm'].sum())), np.array(w_out)


def test_ready(models):
    return all((rv.CACHE/'paths'/f'test_{e:%Y%m%d}_{k}.npz').exists() for e in rj.TEST_ENDS for k in models)


def dump(name, value):
    rj.dump_json(OUT/name, value)


def load_decisions():
    path = OUT/'decisions.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def save_decision(step, value):
    d = load_decisions(); d[step] = value; dump('decisions.json', d)


def step_static(step, candidate, baseline):
    daily, monthly = scores('validation', [baseline, candidate]); table = metrics(daily, monthly)
    ok = passes(table.loc[candidate], table.loc[baseline])
    decision = dict(candidate=candidate, baseline=baseline, accepted=ok,
                    validation=table.loc[[baseline, candidate]].round(6).to_dict('index'))
    if test_ready([baseline, candidate]):
        daily, monthly = scores('test', [baseline, candidate])
        decision['test_reported_after_decision'] = metrics(daily, monthly).loc[[baseline, candidate]].round(6).to_dict('index')
    save_decision(step, decision); return decision


def best_static():
    d = load_decisions(); best = 'M3'
    for step in ('p1', 'p2'):
        if step in d and d[step]['accepted']:
            best = d[step]['candidate']
    return best


def step_stacking(step, models, against=None):
    f = forms('validation', models)
    singles = {k: mixture_scores(f, np.eye(len(models))[i]) for i, k in enumerate(models)}
    best = min(singles, key=lambda k: singles[k]['D'])
    loo, w_loo = loo_stacking(f)
    raw, shrunk = stack_weights(f, np.ones(len(f['nd']), bool))
    reference = against or singles[best]
    if against is None:
        ok = passes(loo, singles[best])
    else:  # P4: include Chronos only if LOO D improves and LOO Mo is not worse than the reference
        ok = bool(loo['D'] < against['D'] and loo['Mo'] <= against['Mo'])
    decision = dict(models=models, accepted=ok, best_single=best, validation_single=singles,
                    validation_loo_stack=loo, loo_weights=w_loo.round(4).tolist(), reference=reference,
                    raw_weights=dict(zip(models, raw.round(4).tolist())),
                    weights=dict(zip(models, shrunk.round(4).tolist())))
    if test_ready(models):
        t = forms('test', models)
        decision['test_reported_after_decision'] = dict(
            stack=mixture_scores(t, shrunk), **{k: mixture_scores(t, np.eye(len(models))[i]) for i, k in enumerate(models)})
    save_decision(step, decision); return decision


def final_models():
    """Weights of the accepted v2 rule (stacking if accepted, else best static)."""
    d = load_decisions()
    for step in ('p4', 'p3'):
        if step in d and d[step]['accepted']:
            return d[step]['weights']
    return {best_static(): 1.}


def monthly_totals(train, key, channel=None):
    """Totals of complete, fully observed calendar months in training."""
    g = train if channel is None else train[train.channel == channel]
    width = 1 if channel else len(m.CHANNELS)
    t = g.assign(month=g.date.dt.to_period('M')).groupby('month').agg(
        total=(key, 'sum'), observed=('observed', 'sum'), rows=('observed', 'size'))
    days = t.index.days_in_month.to_numpy()
    return t[(t.observed == t.rows) & (t.rows == days*width)].total


def event_rows(phase, origin, weights, data, paths=None, dates=None):
    """P(month total > x) for 16 series and 4 national totals, 2 thresholds."""
    tag = pd.Timestamp(origin)
    if dates is None:
        dates = pd.date_range(tag+pd.Timedelta(days=1), tag+pd.offsets.MonthEnd(1))
        paths = {k: rv.load_paths(phase, origin, k) for k in weights}
    train = data[data.date <= tag]; actual = data[data.date.isin(dates)]
    window = train[train.date > tag-pd.DateOffset(months=12)]; rows = []
    for key in rv.KEYS:
        for c, channel in enumerate([*m.CHANNELS, 'ALL']):
            past = monthly_totals(window, key, None if channel == 'ALL' else channel)
            if len(past) < 6:
                continue
            a = actual if channel == 'ALL' else actual[actual.channel == channel]
            complete = bool(a.observed.all()) and len(a) == len(dates)*(4 if channel == 'ALL' else 1)
            y = float(a[key].sum()) if complete else np.nan
            for level in (.5, .75):
                x = float(np.quantile(past, level)); prob = 0.
                for model, w in weights.items():
                    s = paths[model][key]
                    total = s.sum(axis=(1, 2)) if channel == 'ALL' else s[:, :, c].sum(axis=1)
                    prob += w*float((total > x).mean())
                rows.append(dict(origin=str(tag.date()), channel=channel, target=key, level=level, threshold=x,
                                 p=prob, outcome=np.nan if np.isnan(y) else float(y > x), actual=y))
    return rows


def events(phase, weights):
    data, _ = m.load_data()
    rows = [r for e in ends(phase) for r in event_rows(phase, str(e.date()), weights, data)]
    return pd.DataFrame(rows).dropna(subset=['outcome'])


def probit_fit(p, y):
    """Probit recalibration Phi(a + b * probit(p)) by maximum likelihood (L6 pp. 50, 65)."""
    z = norm.ppf(np.clip(p, EVENT_EPS, 1-EVENT_EPS))
    def nll(ab):
        q = np.clip(norm.cdf(ab[0]+ab[1]*z), 1e-12, 1-1e-12)
        return -np.sum(y*np.log(q)+(1-y)*np.log(1-q))
    return minimize(nll, np.array([0., 1.]), method='Nelder-Mead').x


def probit_apply(ab, p):
    return norm.cdf(ab[0]+ab[1]*norm.ppf(np.clip(p, EVENT_EPS, 1-EVENT_EPS)))


def reliability(p, y):
    bins = pd.cut(p, [0, .1, .3, .5, .7, .9, 1.], include_lowest=True)
    return pd.DataFrame({'p': p, 'y': y, 'bin': bins}).groupby('bin', observed=True).agg(
        n=('y', 'size'), mean_p=('p', 'mean'), frequency=('y', 'mean')).reset_index().astype({'bin': str})


def recalibration(weights):
    """P5 for given weights: LOO over validation origins, then validation fit applied to test."""
    v = events('validation', weights); loo = np.empty(len(v))
    for o in v.origin.unique():
        use = v.origin != o; ab = probit_fit(v.p[use].to_numpy(), v.outcome[use].to_numpy())
        loo[~use.to_numpy()] = probit_apply(ab, v.p[~use].to_numpy())
    ab = probit_fit(v.p.to_numpy(), v.outcome.to_numpy())
    return v.assign(p_loo=loo), ab


def step_posthoc(model='M3HR'):
    weights = {model: 1.}; v, ab = recalibration(weights)
    t = events('test', weights); t['p_recalibrated'] = probit_apply(ab, t.p.to_numpy())
    both = pd.concat([v, t])
    ab_all = probit_fit(both.p.to_numpy(), both.outcome.to_numpy())  # refit on all origins for the final forecast
    brier = lambda p, y: float(np.mean((p-y)**2))
    v.to_csv(OUT/f'events_validation_{model}.csv', index=False); t.to_csv(OUT/f'events_test_{model}.csv', index=False)
    daily, monthly = scores('test', ['M3', 'M3H', model])
    decision = dict(
        posthoc=True, chosen_after_test=True, weights=weights,
        note='Authors chose M3HR after seeing the test; P2 was rejected by the rule (validation D 0.000218 worse than M3H).',
        test_metrics=metrics(daily, monthly).round(6).to_dict('index'),
        recalibrate_events=bool(brier(v.p_loo, v.outcome) < brier(v.p, v.outcome)),
        brier_validation_raw=brier(v.p, v.outcome), brier_validation_loo_recalibrated=brier(v.p_loo, v.outcome),
        brier_test_raw=brier(t.p, t.outcome), brier_test_recalibrated=brier(t.p_recalibrated, t.outcome),
        brier_test_climatology=brier(t.level.map({.5: .5, .75: .25}), t.outcome),
        probit_validation=[float(ab[0]), float(ab[1])], probit_final=[float(ab_all[0]), float(ab_all[1])],
        events_final_fit=int(len(both)),
        reliability_test_recalibrated=reliability(t.p_recalibrated.to_numpy(), t.outcome.to_numpy()).to_dict('records'))
    save_decision('posthoc', decision); return decision


def step_events():
    weights = final_models(); v, ab = recalibration(weights)
    raw = float(np.mean((v.p-v.outcome)**2)); cal = float(np.mean((v.p_loo-v.outcome)**2))
    v.to_csv(OUT/'events_validation.csv', index=False)
    decision = dict(weights=weights, events_validation=int(len(v)), base_rate_validation=float(v.outcome.mean()),
                    brier_validation_raw=raw, brier_validation_loo_recalibrated=cal, accepted=cal < raw,
                    probit_intercept=float(ab[0]), probit_slope=float(ab[1]),
                    reliability_validation_raw=reliability(v.p.to_numpy(), v.outcome.to_numpy()).to_dict('records'))
    if not test_ready(list(weights)):
        save_decision('p5', decision); return decision
    t = events('test', weights); t['p_recalibrated'] = probit_apply(ab, t.p.to_numpy())
    t.to_csv(OUT/'events_test.csv', index=False)
    decision.update(test_reported_after_decision=dict(
                        events=int(len(t)), base_rate=float(t.outcome.mean()),
                        brier_raw=float(np.mean((t.p-t.outcome)**2)),
                        brier_recalibrated=float(np.mean((t.p_recalibrated-t.outcome)**2)),
                        brier_climatology=float(np.mean((t.level.map({.5: .5, .75: .25})-t.outcome)**2))),
                    reliability_test_raw=reliability(t.p.to_numpy(), t.outcome.to_numpy()).to_dict('records'),
                    reliability_test_recalibrated=reliability(t.p_recalibrated.to_numpy(), t.outcome.to_numpy()).to_dict('records'))
    save_decision('p5', decision); return decision


def final_paths():
    out = {}
    with np.load(m.ROOT/'output/joint_forecast/forecast_paths.npz') as z:
        for model in ('M1', 'M2', 'M3', 'M4'):
            out[model] = {key: z[f'{model}__{key}'].astype(np.int64) for key in rv.KEYS}
        dates = pd.DatetimeIndex(z['dates'])
    for path in OUT.glob('final_paths_*.npz'):
        with np.load(path) as z:
            out[path.stem.removeprefix('final_paths_')] = {key: z[key].astype(np.int64) for key in rv.KEYS}
    return out, dates


def mixture_quantiles(totals, weights, levels=(.025, .1, .5, .9, .975)):
    z = np.arange(0, max(int(t.max()) for t in totals.values())+2)
    F = sum(w*np.searchsorted(np.sort(totals[k]), z, side='right')/len(totals[k]) for k, w in weights.items())
    mean = sum(w*float(totals[k].mean()) for k, w in weights.items())
    return mean, [float(z[np.searchsorted(F, q)]) for q in levels]


def step_final():
    paths, dates = final_paths(); d = load_decisions(); rows = []
    versions = [('v1_M3', {'M3': 1.}), ('v2_rule', final_models())]
    if 'posthoc' in d:
        versions.append(('v2_final', d['posthoc']['weights']))
    for version, w in versions:
        for key in rv.KEYS:
            for c, channel in enumerate([*m.CHANNELS, 'ALL']):
                totals = {k: (paths[k][key].sum(axis=(1, 2)) if channel == 'ALL' else paths[k][key][:, :, c].sum(axis=1)) for k in w}
                mean, q = mixture_quantiles(totals, w)
                rows.append(dict(version=version, channel=channel, target=key, mean=mean, lo95=q[0], lo80=q[1],
                                 median=q[2], hi80=q[3], hi95=q[4]))
    table = pd.DataFrame(rows); table.to_csv(OUT/'forecast_month_v2.csv', index=False)
    data, _ = m.load_data(); version, weights = versions[-1]
    ev = pd.DataFrame(event_rows('final', '2026-10-02', weights, data, paths, dates)).assign(version=version)
    if version == 'v2_final' and d['posthoc']['recalibrate_events']:
        ev['p_recalibrated'] = probit_apply(d['posthoc']['probit_final'], ev.p.to_numpy())
    elif d.get('p5', {}).get('accepted'):
        ev['p_recalibrated'] = probit_apply([d['p5']['probit_intercept'], d['p5']['probit_slope']], ev.p.to_numpy())
    ev.drop(columns=['outcome', 'actual']).to_csv(OUT/'forecast_events_v2.csv', index=False)
    return table[table.channel == 'ALL'].round(1).to_dict('records')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=('p1', 'p2', 'p3', 'p4', 'p5', 'posthoc', 'final'), required=True)
    args = parser.parse_args()
    if args.step == 'p1':
        result = step_static('p1', 'M3H', 'M3')
    elif args.step == 'p2':
        base = best_static(); result = step_static('p2', 'M3HR' if base == 'M3H' else 'M3R', base)
    elif args.step == 'p3':
        result = step_stacking('p3', [best_static(), 'M4'])
    elif args.step == 'p4':
        # Compare with the best accepted rule: P3 stacking if accepted, else the best static.
        p3 = load_decisions()['p3']
        reference = p3['validation_loo_stack'] if p3['accepted'] else p3['validation_single'][best_static()]
        result = step_stacking('p4', [best_static(), 'M4', 'CHRONOS2'], reference)
    elif args.step == 'p5':
        result = step_events()
    elif args.step == 'posthoc':
        result = step_posthoc()
    else:
        result = step_final()
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
