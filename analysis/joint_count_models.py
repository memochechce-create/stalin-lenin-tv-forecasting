"""Coherent Bayesian archive-count forecasts: N -> B|N -> C=B+E|B.

No future denominators, counts or fitted preprocessing enter a forecast. Calendar
effects have exchangeable, fixed-scale Normal priors (prior-specified pooling).
MCMC targets the exact static posterior; a Hessian only constructs proposals.
M4 is explicitly a cut posterior: static M3 draws + conditional state filters.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, gammaln, logsumexp, logit
from scipy.stats import multivariate_t
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ('1TV', 'NTV', 'RUSSIA1', 'RUSSIA24')
PEOPLE = ('lenin', 'stalin')
COMPONENTS = ('N', 'B_lenin', 'E_lenin', 'B_stalin', 'E_stalin')
MODELS = ('M1', 'M2', 'M3', 'M4', 'POOL')
HOLIDAYS = {'nov7': (11, 7), 'may8': (5, 8), 'may9': (5, 9),
            'lenin_birth': (4, 22), 'lenin_death': (1, 21),
            'stalin_death': (3, 5), 'stalin_official_birth': (12, 21)}
threadpool_limits(limits=1)


def load_data(path=None, possible=False):
    path = Path(path or ROOT/'data/cleaned/tv_daily_targets.csv.gz')
    d = pd.read_csv(path)
    d['date'] = pd.to_datetime(d['day'])
    d['observed'] = d['observed'].astype(bool)
    if possible:
        for person in PEOPLE:
            for target in ('mentions', 'broadcasts'):
                d[f'{person}_{target}'] = d[f'{person}_possible_{target}']
    d = d.sort_values(['date', 'channel']).reset_index(drop=True)
    if d.duplicated(['date', 'channel']).any():
        raise ValueError('Duplicate date/channel')
    for c in CHANNELS:
        g = d[d.channel == c]
        if len(g) != len(pd.date_range(d.date.min(), d.date.max())):
            raise ValueError('Calendar gaps must be explicit rows')
    return d, hashlib.sha256(path.read_bytes()).hexdigest()


def design(dates, channels, calendar, center=0.):
    dates = pd.DatetimeIndex(dates)
    ix = np.array([CHANNELS.index(c) for c in channels])
    X = [np.ones(len(ix)), *helmert(4, full=False).T[ix].T]
    names = ['population', 'channel_c1', 'channel_c2', 'channel_c3']
    sd = [1.5, .8, .8, .8]
    if calendar:
        X.append(np.asarray((dates-pd.Timestamp('2023-01-01')).days)/365.25-center)
        names.append('trend_per_year'); sd.append(.35)
        for j, v in enumerate(helmert(12, full=False).T[dates.month-1].T):
            X.append(v); names.append(f'month_c{j+1}'); sd.append(.35)
        for j, v in enumerate(helmert(7, full=False).T[dates.dayofweek].T):
            X.append(v); names.append(f'weekday_c{j+1}'); sd.append(.25)
        # Separate exact-date and surrounding-day effects. Nov 7 and May 8 are
        # explicit, even when absent from the final 31-day forecast window.
        for name, (month, day) in HOLIDAYS.items():
            delta = np.array([min(abs((t-pd.Timestamp(y, month, day)).days)
                                 for y in (t.year-1, t.year, t.year+1)) for t in dates])
            X.extend([delta == 0, (delta > 0) & (delta <= 2)])
            names.extend([f'exact_{name}', f'near_{name}']); sd.extend([.7, .5])
    return np.column_stack(X).astype(float), names, np.array(sd)


def nb_logpmf(y, rate, k, exposure=1.):
    """NB sum of exposure independent per-record excess counts.

    mean=exposure*rate, size=exposure*k. Exposure must be positive.
    """
    size = exposure*k
    return (gammaln(y+size)-gammaln(size)-gammaln(y+1)
            +size*(np.log(k)-np.log(k+rate))
            +y*(np.log(rate)-np.log(k+rate)))


def bb_logpmf(y, n, p, phi):
    a, b = p*phi, (1-p)*phi
    return (gammaln(n+1)-gammaln(y+1)-gammaln(n-y+1)
            +betaln(y+a, n-y+b)-betaln(a, b))


@dataclass
class Problem:
    X: np.ndarray
    y: np.ndarray
    exposure: np.ndarray
    prior: np.ndarray
    sd: np.ndarray
    kind: str
    center: float
    names: list

    def value_grad(self, theta, gradient=True):
        if np.max(np.abs(theta)) > 25:
            return (1e100, np.zeros_like(theta)) if gradient else 1e100
        eta = self.X@theta[:-1]
        if np.max(np.abs(eta)) > 25:
            return (1e100, np.zeros_like(theta)) if gradient else 1e100
        y, w = self.y, self.exposure
        dispersion = np.exp(theta[-1])
        if self.kind == 'B':
            p = expit(eta); a, b = p*dispersion, (1-p)*dispersion
            ll = bb_logpmf(y, w, p, dispersion)
            if gradient:
                da = digamma(y+a)-digamma(w+dispersion)-digamma(a)+digamma(dispersion)
                db = digamma(w-y+b)-digamma(w+dispersion)-digamma(b)+digamma(dispersion)
                ge = dispersion*p*(1-p)*(da-db)
                gd = np.sum(a*da+b*db)
        else:
            rate = np.exp(eta); k = dispersion; size = w*k
            ll = nb_logpmf(y, rate, k, w)
            if gradient:
                ge = k*(y-w*rate)/(k+rate)
                gd = np.sum(size*(digamma(y+size)-digamma(size)+theta[-1]
                                  -np.log(k+rate)+1-(size+y)/(w*(k+rate))))
        diff = (theta-self.prior)/self.sd
        value = float(-ll.sum()+.5*(diff@diff))
        if not gradient:
            return value
        grad = np.r_[self.X.T@ge, gd]-(theta-self.prior)/self.sd**2
        return value, -grad


def make_problem(train, component, calendar, prior_scale=1.):
    d = train[train.observed].copy()
    kind = component[0]
    if kind == 'N':
        y = d.available_broadcasts.to_numpy(float); w = np.ones(len(d)); level = np.log(20); disp = np.log(5)
    else:
        person = component.split('_')[1]
        B = d[f'{person}_broadcasts'].to_numpy(float)
        if kind == 'B':
            y = B; w = d.available_broadcasts.to_numpy(float); level = logit(.05); disp = np.log(20)
            keep = w > 0
        else:
            y = d[f'{person}_mentions'].to_numpy(float)-B; w = B; level = 0.; disp = 0.
            keep = w > 0
        d = d[keep]; y = y[keep]; w = w[keep]
    center = float(np.mean((d.date-pd.Timestamp('2023-01-01')).dt.days)/365.25)
    X, names, sd = design(d.date, d.channel, calendar, center)
    sd[4:] *= prior_scale
    prior = np.r_[level, np.zeros(X.shape[1]-1), disp]
    return Problem(X, y, w, prior, np.r_[sd, 1.5], kind, center, names+['log_dispersion'])


def mcmc_diagnostics(chains):
    # ArviZ implements rank-normalized folded split Rhat and bulk/tail ESS.
    import arviz as az
    data = az.from_dict(posterior={'theta': chains})
    return {'max_rhat': float(az.rhat(data).theta.max()),
            'min_bulk_ess': float(az.ess(data, method='bulk').theta.min()),
            'min_tail_ess': float(az.ess(data, method='tail').theta.min())}


def fit_component(train, component, calendar, seed, draws=1000, warm=250, prior_scale=1.):
    problem = make_problem(train, component, calendar, prior_scale)
    opt = minimize(problem.value_grad, problem.prior.copy(), jac=True, method='BFGS',
                   options={'gtol': 2e-5, 'maxiter': 500})
    x = opt.x; dim = len(x); eps = 1e-4
    # Near-Poisson N can have large dispersion and BFGS reports precision loss:
    # objective differences cancel while the analytic score remains usable.
    # Newton refinement solves that score rather than relaxing the criterion.
    for refinement in range(5):
        H = np.column_stack([(problem.value_grad(x+np.eye(dim)[j]*eps)[1]
                              -problem.value_grad(x-np.eye(dim)[j]*eps)[1])/(2*eps) for j in range(dim)])
        score = problem.value_grad(x)[1]
        if np.max(np.abs(score)) < 1e-4:
            break
        step = np.linalg.solve((H+H.T)/2, score)
        improved = False
        for fraction in (1., .5, .25, .125):
            candidate = x-fraction*step
            if np.max(abs(problem.value_grad(candidate)[1])) < np.max(abs(score)):
                x = candidate; improved = True; break
        if not improved:
            break
    eigen, U = np.linalg.eigh((H+H.T)/2)
    if eigen.min() <= 0 or not np.isfinite(opt.fun):
        raise RuntimeError(f'Invalid posterior curvature: {component}')
    cov = (U/eigen)@U.T
    # Heavy-tailed independence proposals; exact likelihood/prior acceptance.
    proposal = multivariate_t(loc=x, shape=cov, df=12)
    rng = np.random.default_rng(seed)
    chains = np.empty((4, draws, dim)); accepts = []
    for c in range(4):
        current = proposal.rvs(random_state=rng)
        target = -problem.value_grad(current, False)-float(proposal.logpdf(current))
        candidates = proposal.rvs(size=draws+warm, random_state=rng)
        logq = proposal.logpdf(candidates); logu = np.log(rng.random(draws+warm)); acc = 0
        for i, candidate in enumerate(candidates):
            new = -problem.value_grad(candidate, False)-logq[i]
            if logu[i] < new-target:
                current = candidate; target = new
                acc += i >= warm
            if i >= warm:
                chains[c, i-warm] = current
        accepts.append(acc/draws)
    diag = mcmc_diagnostics(chains)
    diag.update(component=component, calendar=calendar, n=int(len(problem.y)),
                acceptance=accepts, draws_per_chain=draws, warmup=warm,
                gradient_max=float(abs(problem.value_grad(x)[1]).max()),
                optimizer_success=bool(opt.success), hessian_min_eigen=float(eigen.min()))
    if diag['gradient_max'] > .01:
        raise RuntimeError(f'Unresolved posterior optimization: {diag}')
    if diag['max_rhat'] >= 1.01 or min(diag['min_bulk_ess'], diag['min_tail_ess']) < 400:
        if draws < 8000:
            return fit_component(train, component, calendar, seed, draws*2, warm*2, prior_scale)
        raise RuntimeError(f'MCMC did not converge: {diag}')
    return {'theta': chains.reshape(-1, dim), 'center': problem.center,
            'calendar': calendar, 'component': component, 'diagnostics': diag,
            'map': x, 'names': problem.names}


def fit_static(train, calendar, seed, prior_scale=1.):
    return {name: fit_component(train, name, calendar, seed+23*i, prior_scale=prior_scale)
            for i, name in enumerate(COMPONENTS)}


def grid(dates):
    return pd.DataFrame([(d, c) for d in pd.DatetimeIndex(dates) for c in CHANNELS], columns=['date', 'channel'])


def parameter_paths(fit, dates, n, rng):
    future = grid(dates)
    theta = fit['theta'][rng.integers(len(fit['theta']), size=n)]
    X = design(future.date, future.channel, fit['calendar'], fit['center'])[0]
    eta = (theta[:, :-1]@X.T).reshape(n, len(dates), 4)
    return eta, np.exp(theta[:, -1])[:, None, None]


def simulate_components(parameters, rng):
    eta, k = parameters['N']; rate = np.exp(eta)
    N = rng.negative_binomial(k, k/(k+rate))
    result = {'N': N}
    for person in PEOPLE:
        eta, phi = parameters['B_'+person]; p = expit(eta)
        prob = rng.beta(p*phi, (1-p)*phi)
        B = rng.binomial(N, prob)
        eta, k = parameters['E_'+person]; rate = np.exp(eta)
        # B independent positive counts 1+NB(k, mean=rate), summed exactly.
        E = np.zeros_like(B); size = np.broadcast_to(B*k, B.shape)
        success = np.broadcast_to(k/(k+rate), B.shape); ok = B > 0
        E[ok] = rng.negative_binomial(size[ok], success[ok])
        result[person+'_broadcasts'] = B
        result[person+'_mentions'] = B+E
    return result


def static_predict(fits, dates, n, seed):
    rng = np.random.default_rng(seed)
    return simulate_components({name: parameter_paths(fit, dates, n, rng) for name, fit in fits.items()}, rng)


def conjugate_predict(train, dates, n, seed):
    rng = np.random.default_rng(seed); days = len(dates)
    result = {key: np.zeros((n, days, 4), dtype=np.int64)
              for key in ['N', *[p+'_'+t for p in PEOPLE for t in ('broadcasts', 'mentions')]]}
    for c, channel in enumerate(CHANNELS):
        d = train[(train.channel == channel) & train.observed]
        # Gamma(shape, rate): weak prior mean 20 records/day, 0.1 pseudo-day.
        lam = rng.gamma(2+d.available_broadcasts.sum(), 1/(.1+len(d)), n)
        N = rng.poisson(lam[:, None], (n, days)); result['N'][:, :, c] = N
        for person in PEOPLE:
            B0 = d[person+'_broadcasts'].sum(); E0 = (d[person+'_mentions']-d[person+'_broadcasts']).sum()
            # Beta(1,19) mean5%; Gamma(1,1) mean1 excess mention/positive record.
            p = rng.beta(1+B0, 19+d.available_broadcasts.sum()-B0, n)
            rate = rng.gamma(1+E0, 1/(1+B0), n)
            B = rng.binomial(N, p[:, None]); C = B+rng.poisson(B*rate[:, None])
            result[person+'_broadcasts'][:, :, c] = B; result[person+'_mentions'][:, :, c] = C
    return result


def systematic_indices(weights, rng):
    n = weights.shape[0]
    cdf = np.cumsum(weights); cdf[-1] = 1.
    return np.searchsorted(cdf, (rng.random()+np.arange(n))/n)


def filter_component(train, fit, q, seed, groups=24, particles=48, trace=False):
    """Conditional filters for each static draw/channel; equal static weights.

    The last 365 calendar days form an explicitly initialized state window.
    Missing observations only propagate; no zero substitution, including B=0
    when the excess-count likelihood contributes no information about its rate.
    """
    rng = np.random.default_rng(seed)
    last = train.date.max(); dates = pd.date_range(max(train.date.min(), last-pd.Timedelta(days=364)), last)
    frame = grid(dates).merge(train, on=['date', 'channel'], how='left')
    theta = fit['theta'][rng.integers(len(fit['theta']), size=groups)]
    X = design(frame.date, frame.channel, True, fit['center'])[0]
    base = (theta[:, :-1]@X.T).reshape(groups, len(dates), 4)
    disp = np.exp(theta[:, -1])[:, None, None]
    state = rng.normal(0, .25, (groups, particles, 4))
    lw = np.full_like(state, -np.log(particles)); ess_min = particles; count = 0; history = []
    observed = frame.observed.fillna(False).to_numpy(bool).reshape(-1, 4)
    kind = fit['component'][0]
    if kind == 'N':
        y = frame.available_broadcasts.to_numpy(float).reshape(-1, 4); exposure = np.ones_like(y)
    else:
        person = fit['component'].split('_')[1]
        B = frame[person+'_broadcasts'].to_numpy(float).reshape(-1, 4)
        if kind == 'B':
            y = B; exposure = frame.available_broadcasts.to_numpy(float).reshape(-1, 4)
        else:
            y = frame[person+'_mentions'].to_numpy(float).reshape(-1, 4)-B; exposure = B
    for t, date in enumerate(dates):
        state += rng.normal(0, q, state.shape)
        for c in range(4):
            if not observed[t, c] or exposure[t, c] <= 0:
                continue
            eta = base[:, t, c, None]+state[:, :, c]
            if kind == 'B':
                ll = bb_logpmf(y[t, c], exposure[t, c], expit(eta), disp[:, :, 0])
            else:
                ll = nb_logpmf(y[t, c], np.exp(eta), disp[:, :, 0], exposure[t, c])
            lw[:, :, c] += ll; lw[:, :, c] -= logsumexp(lw[:, :, c], axis=1)[:, None]
        weights = np.exp(lw)
        if trace:
            history.extend({'day': str(date.date()), 'channel': CHANNELS[c], 'component': fit['component'],
                            'state_mean': float(np.sum(weights[:, :, c]*state[:, :, c])/groups)} for c in range(4))
        ess = 1/np.sum(weights**2, axis=1); ess_min = min(ess_min, float(ess.min()))
        for g, c in zip(*np.where(ess < particles/2)):
            state[g, :, c] = state[g, systematic_indices(weights[g, :, c], rng), c]
            lw[g, :, c] = -np.log(particles); count += 1
    # Independent conditional resampling within group/channel; static weights
    # deliberately remain equal, as prescribed by the cut approximation.
    for g in range(groups):
        for c in range(4):
            state[g, :, c] = state[g, systematic_indices(np.exp(lw[g, :, c]), rng), c]
    return theta, state, {'component': fit['component'], 'q': q, 'groups': groups,
                          'particles_per_group': particles, 'min_ess': ess_min,
                          'resamples': count, 'state_window_days': len(dates), 'history': history}


def dynamic_predict(train, fits, dates, q, n, seed, groups=24, particles=48, trace=False):
    rng = np.random.default_rng(seed); parameters = {}; diagnostics = []
    future = grid(dates)
    for i, (name, fit) in enumerate(fits.items()):
        theta, state, diag = filter_component(train, fit, q, seed+13*i, groups, particles, trace)
        g = rng.integers(groups, size=n); j = rng.integers(particles, size=(n, 4))
        level = state[g[:, None], j, np.arange(4)[None, :]].copy()
        X = design(future.date, future.channel, True, fit['center'])[0]
        base = (theta[g, :-1]@X.T).reshape(n, len(dates), 4)
        eta = np.empty_like(base)
        # Account for elapsed days even if callers request nonconsecutive dates.
        previous = train.date.max()
        for t, date in enumerate(pd.DatetimeIndex(dates)):
            level += rng.normal(0, q*np.sqrt((date-previous).days), level.shape)
            eta[:, t, :] = base[:, t, :]+level; previous = date
        parameters[name] = eta, np.exp(theta[g, -1])[:, None, None]
        diagnostics.append(diag)
    return simulate_components(parameters, rng), diagnostics


def linear_pool(forecasts, seed):
    """Select one whole coherent trajectory, never average counts/quantiles."""
    rng = np.random.default_rng(seed); names = ('M1', 'M2', 'M3', 'M4')
    n = forecasts['M1']['N'].shape[0]; choice = rng.integers(4, size=n)
    result = {key: np.empty_like(value) for key, value in forecasts['M1'].items()}
    for i, name in enumerate(names):
        use = choice == i
        for key in result:
            result[key][use] = forecasts[name][key][use]
    return result


def crps_sample(samples, y):
    """Exact empirical-distribution CRPS, using the sorted-sample identity."""
    s = np.sort(samples, axis=0); n = len(s)
    coeff = (2*np.arange(1, n+1)-n-1).reshape((n,)+(1,)*(s.ndim-1))
    return np.mean(abs(s-y), axis=0)-np.sum(coeff*s, axis=0)/n**2


def summarize_paths(samples):
    q = np.quantile(samples, [.025, .1, .5, .9, .975], axis=0)
    return {'mean': samples.mean(axis=0), 'lo95': q[0], 'lo80': q[1],
            'median': q[2], 'hi80': q[3], 'hi95': q[4]}


def score_forecast(paths, actual, train, model, origin, phase, seed=1):
    rows = []; monthly = []; rng = np.random.default_rng(seed)
    for c, channel in enumerate(CHANNELS):
        target = actual[actual.channel == channel].sort_values('date')
        history = train[(train.channel == channel) & train.observed]
        for key in [p+'_'+t for p in PEOPLE for t in ('broadcasts', 'mentions')]:
            s = paths[key][:, :, c]; y = target[key].to_numpy(float)
            ok = target.observed.to_numpy(bool) & np.isfinite(y)
            summary = summarize_paths(s); crps = crps_sample(s[:, ok], y[ok])
            scale = max(1., float(history[key].mean()))
            less = (s[:, ok] < y[ok]).mean(axis=0); equal = (s[:, ok] == y[ok]).mean(axis=0)
            pit = less+rng.random(ok.sum())*equal
            for j, t in enumerate(np.flatnonzero(ok)):
                rows.append(dict(phase=phase, origin=str(pd.Timestamp(origin).date()), day=str(target.date.iloc[t].date()),
                                 horizon=t+1, channel=channel, target=key, model=model, actual=float(y[t]),
                                 mean=float(summary['mean'][t]), se=float((summary['mean'][t]-y[t])**2),
                                 scaled_se=float(((summary['mean'][t]-y[t])/scale)**2),
                                 crps=float(crps[j]), scaled_crps=float(crps[j]/scale), pit=float(pit[j]),
                                 **{name: float(value[t]) for name, value in summary.items() if name != 'mean'},
                                 coverage80=bool(summary['lo80'][t] <= y[t] <= summary['hi80'][t]),
                                 coverage95=bool(summary['lo95'][t] <= y[t] <= summary['hi95'][t]),
                                 width80=float(summary['hi80'][t]-summary['lo80'][t]),
                                 width95=float(summary['hi95'][t]-summary['lo95'][t])))
            # Full calendar total only when ALL days are observed for this channel.
            if ok.all():
                total = s.sum(axis=1); observed_total = y.sum(); su = summarize_paths(total)
                monthly.append(dict(phase=phase, origin=str(pd.Timestamp(origin).date()), channel=channel, target=key,
                                    model=model, actual=float(observed_total), crps=float(crps_sample(total, observed_total)),
                                    se=float((total.mean()-observed_total)**2),
                                    **{k: float(v) for k, v in su.items()}))
    return rows, monthly


def provenance():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'analysis').glob('*forecast*.py')} | {
                'joint_count_models.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
