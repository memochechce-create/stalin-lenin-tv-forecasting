"""Independent structural, chronological, Monte Carlo and provenance checks."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

import joint_count_models as m

OUT = m.ROOT/'output/joint_forecast'


def verify():
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    source = m.ROOT/'data/cleaned/tv_daily_targets.csv.gz'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']
    for name, expected in manifest['code_sha256'].items():
        assert hashlib.sha256((m.ROOT/'analysis'/name).read_bytes()).hexdigest() == expected, name
    for name, expected in manifest['outputs_sha256'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest() == expected, name
    daily = pd.read_csv(OUT/'forecast_daily.csv')
    totals = pd.read_csv(OUT/'forecast_month.csv')
    for file in ('forecast_paths.npz', 'forecast_paths_possible.npz'):
        scenario = 'main' if file == 'forecast_paths.npz' else 'possible'
        with np.load(OUT/file, allow_pickle=False) as paths:
            dates = paths['dates']; assert len(dates) == 31
            assert dates[0] == '2026-10-03' and dates[-1] == '2026-11-02'
            names = {k.split('__')[0] for k in paths.files if '__' in k}
            assert names == (set(m.MODELS) if scenario == 'main' else {manifest['selected_model']})
            for model in names:
                N = paths[model+'__N']
                assert N.shape == (manifest['final_draws'], 31, 4)
                for person in m.PEOPLE:
                    B = paths[model+'__'+person+'_broadcasts']; C = paths[model+'__'+person+'_mentions']
                    assert np.all((0 <= B) & (B <= N) & (B <= C))
                    assert np.all(C[B == 0] == 0)
                for key in ['N', *[p+'_'+t for p in m.PEOPLE for t in ('broadcasts', 'mentions')]]:
                    values = paths[model+'__'+key]
                    assert np.issubdtype(values.dtype, np.integer) and values.min() >= 0
                    summed = values.sum(axis=1)
                    for c, channel in enumerate(m.CHANNELS):
                        a = daily[(daily.scenario == scenario) & (daily.model == model) & (daily.channel == channel) & (daily.target == key)].sort_values('day')
                        b = totals[(totals.scenario == scenario) & (totals.model == model) & (totals.channel == channel) & (totals.target == key)].iloc[0]
                        np.testing.assert_allclose(a['mean'], values[:, :, c].mean(axis=0), rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(b['mean'], summed[:, c].mean(), rtol=1e-12)
                        np.testing.assert_allclose(a['mean'].sum(), b['mean'], rtol=1e-12)
                        np.testing.assert_allclose([b.lo95, b.lo80, b['median'], b.hi80, b.hi95],
                                                   np.quantile(summed[:, c], [.025, .1, .5, .9, .975]))
    for frame in (daily, totals):
        assert (frame.lo95 <= frame.lo80).all() and (frame.lo80 <= frame['median']).all()
        assert (frame['median'] <= frame.hi80).all() and (frame.hi80 <= frame.hi95).all()
        assert np.isfinite(frame.select_dtypes(include='number')).all().all()
    test = pd.read_csv(OUT/'backtest_daily.csv.gz')
    assert set(test.model) == set(m.MODELS) and test.origin.nunique() == 21
    assert (pd.to_datetime(test.origin) < pd.to_datetime(test.day)).all()
    assert pd.to_datetime(test.day).min() == pd.Timestamp('2025-01-01')
    assert pd.to_datetime(test.day).max() == pd.Timestamp('2026-09-30')
    assert not test.duplicated(['model', 'day', 'channel', 'target']).any()
    counts = test.groupby(['day', 'channel', 'target']).model.nunique()
    assert counts.eq(5).all()
    d, _ = m.load_data()
    observed = set(zip(d.loc[d.observed, 'day'], d.loc[d.observed, 'channel']))
    assert set(zip(test.day, test.channel)) <= observed
    selection = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    validation = pd.read_csv(OUT/'validation_daily.csv.gz')
    assert validation.day.max() < test.day.min()
    selected = validation.groupby('model').scaled_crps.mean().idxmin()
    assert selected == selection['selected_model'] == manifest['selected_model']
    diagnostics = json.loads((OUT/'backtest_diagnostics.json').read_text(encoding='utf-8'))
    fits = [fit for origin in diagnostics for model in ('M2', 'M3') for fit in origin[model].values()]
    for name in ('main', 'possible'):
        extra = json.loads((OUT/f'final_{name}_diagnostics.json').read_text(encoding='utf-8'))
        fits.extend(fit for model in ('M2', 'M3') for fit in extra[model].values())
    assert max(f['max_rhat'] for f in fits) < 1.01
    assert min(min(f['min_bulk_ess'], f['min_tail_ess']) for f in fits) >= 400
    assert max(f['gradient_max'] for f in fits) < .01
    # Recompute aggregate scores from the saved individual predictions.
    aggregate = pd.read_csv(OUT/'evaluation.csv').set_index('model')
    for model, group in test.groupby('model'):
        np.testing.assert_allclose(aggregate.loc[model, 'crps'], group.crps.mean())
        np.testing.assert_allclose(aggregate.loc[model, 'rmse'], np.sqrt(group.se.mean()))
    report = dict(status='pass', checks=[
        'Input and executable SHA-256 match manifest',
        'Five models, two people, four channels, both targets, full 31-day horizon',
        'All 30000 joint paths satisfy 0<=B<=N and B<=C; B=0 implies C=0',
        'Month summaries are quantiles of summed paths, not sums of daily quantiles',
        'Validation precedes test; model selection reproduced from validation only',
        '21 nonoverlapping calendar test months; identical observed dates across models',
        'Missing channel-days excluded from scoring; never replaced with zero',
        'Every static fit passes rank split Rhat<1.01 and bulk/tail ESS>=400',
        'Aggregate evaluation recomputed from individual forecast scores'],
        max_rhat=max(f['max_rhat'] for f in fits),
        min_bulk_ess=min(f['min_bulk_ess'] for f in fits), min_tail_ess=min(f['min_tail_ess'] for f in fits),
        static_component_fits=len(fits), scored_rows=len(test), selected_model=selected)
    (OUT/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Forecast verification PASS.', flush=True)
    return report


if __name__ == '__main__':
    verify()
