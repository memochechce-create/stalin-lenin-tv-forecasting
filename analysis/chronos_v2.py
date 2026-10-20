"""P4: Chronos-2 zero-shot competitor and the seasonal naive baseline (L7, p. 40).

python analysis/chronos_v2.py --phase validation|test|final

Chronos-2 predicts quantiles directly. Its 21 training quantiles are
interpolated to the levels 0.01..0.99 and rounded to counts; these 99 values
act as an equally weighted sample, so the v2 scoring code applies unchanged.
Summing them across days at equal levels gives a comonotone monthly total,
an assumption (Chronos gives no joint paths). B <= C is not enforced.
Each channel is one multivariate input (the four targets); weekdays and the
v1 holiday indicators are known future covariates. Missing days are NaN.
Seasonal naive: each day repeats the last observed value 7 days earlier.
Lookahead caveat (L7): the pretraining corpus may overlap the test period.
"""
import argparse

import numpy as np
import pandas as pd

import joint_count_models as m
import run_forecast_v2 as rv
import run_joint_forecast as rj

MODEL = 'amazon/chronos-2'
LEVELS = np.round(np.arange(1, 100)/100, 2)


def calendar(dates):
    dates = pd.DatetimeIndex(dates)
    cov = {'weekday': dates.dayofweek.to_numpy(float)}
    for name, (month, day) in m.HOLIDAYS.items():
        delta = np.array([min(abs((t-pd.Timestamp(y, month, day)).days) for y in (t.year-1, t.year, t.year+1)) for t in dates])
        cov[f'exact_{name}'] = (delta == 0).astype(float); cov[f'near_{name}'] = ((delta > 0) & (delta <= 2)).astype(float)
    return cov


def chronos_paths(pipeline, train, dates):
    history = pd.date_range(train.date.min(), train.date.max())
    past, future = calendar(history), calendar(dates); inputs = []
    for c in m.CHANNELS:
        g = train[train.channel == c].set_index('date').reindex(history)
        y = g[rv.KEYS].to_numpy(float).T
        y[:, ~g.observed.fillna(False).to_numpy(bool)] = np.nan
        inputs.append({'target': y, 'past_covariates': past, 'future_covariates': future})
    q, _ = pipeline.predict_quantiles(inputs, prediction_length=len(dates), quantile_levels=list(pipeline.quantiles))
    base = np.array(pipeline.quantiles); paths = {}
    for k, key in enumerate(rv.KEYS):
        out = np.empty((len(LEVELS), len(dates), 4))
        for c in range(4):
            values = np.maximum.accumulate(q[c][k].numpy(), axis=-1)  # monotone in level
            for t in range(len(dates)):
                out[:, t, c] = np.interp(LEVELS, base, values[t])
        paths[key] = np.clip(np.rint(out), 0, None).astype(np.int64)
    return paths


def seasonal_naive(train, dates):
    paths = {key: np.zeros((1, len(dates), 4), dtype=np.int64) for key in rv.KEYS}
    for c, channel in enumerate(m.CHANNELS):
        g = train[(train.channel == channel) & train.observed].set_index('date')
        for t, day in enumerate(dates):
            lag = day-pd.Timedelta(days=7)
            while lag > train.date.max() or lag not in g.index:
                lag -= pd.Timedelta(days=7)
            for key in rv.KEYS:
                paths[key][0, t, c] = int(g.loc[lag, key])
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('validation', 'test', 'final'), required=True)
    args = parser.parse_args()
    import torch
    from chronos import Chronos2Pipeline
    torch.manual_seed(rj.BASE_SEED)
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map='cpu')
    d, _ = m.load_data()
    if args.phase == 'final':
        dates = pd.date_range('2026-10-03', '2026-11-02'); paths = chronos_paths(pipeline, d, dates)
        np.savez_compressed(rv.OUT/'final_paths_CHRONOS2.npz', dates=np.array(dates.strftime('%Y-%m-%d'), dtype='U10'),
                            **{key: v.astype(np.int32) for key, v in paths.items()})
        return
    for p in (rv.CACHE/'paths', rv.CACHE/'scores'):
        p.mkdir(parents=True, exist_ok=True)
    for end in (rj.VALID_ENDS if args.phase == 'validation' else rj.TEST_ENDS):
        origin = pd.Timestamp(end); train = d[d.date <= origin]
        dates = pd.date_range(origin+pd.Timedelta(days=1), origin+pd.offsets.MonthEnd(1))
        actual = d[d.date.isin(dates)]; tag = f'{args.phase}_{origin:%Y%m%d}'
        for name, paths in (('CHRONOS2', chronos_paths(pipeline, train, dates)), ('SNAIVE', seasonal_naive(train, dates))):
            rv.save_paths(rv.CACHE/'paths'/f'{tag}_{name}.npz', paths)
            daily, monthly = m.score_forecast(paths, actual, train, name, origin, args.phase, 1)
            pd.DataFrame(daily).to_csv(rv.CACHE/'scores'/f'{tag}_{name}_daily.csv.gz', index=False)
            pd.DataFrame(monthly).to_csv(rv.CACHE/'scores'/f'{tag}_{name}_monthly.csv.gz', index=False)
        rv.series_scales(train, len(dates)).assign(origin=str(origin.date())).to_csv(rv.CACHE/'scores'/f'{tag}_scales.csv', index=False)
        print(args.phase, origin.date(), flush=True)


if __name__ == '__main__':
    main()
