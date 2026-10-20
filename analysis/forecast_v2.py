"""Version 2 (post-hoc) static components: estimated pooling scales.

P1: monthly contrasts c_j ~ N(0, tau^2), tau^2 ~ InvGamma(2, 0.35^2), so the
    data choose how far month effects shrink to the common level (L4, p. 30).
P2: a level u_ym ~ N(0, s^2) shared by channels and days of each calendar month,
    s^2 ~ InvGamma(2, 0.35^2); a new month draws a new u (L4, p. 31).
Inference is exact-posterior HMC preconditioned by the Laplace covariance (L3). Version 1 code and outputs are not modified.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import joint_count_models as m

IG_SHAPE, IG_SCALE = 2., .35**2


@dataclass
class HierProblem:
    """Likelihood of v1 plus Normal coefficients with sampled log-variances."""
    base: m.Problem
    groups: tuple  # (coefficient positions, InvGamma shape, InvGamma scale)

    @property
    def dim(self):
        return len(self.base.prior)+len(self.groups)

    def value_grad(self, theta, gradient=True):
        G = len(self.groups); core, s = theta[:-G], theta[-G:]
        if np.max(np.abs(s)) > 25:
            return (1e100, np.zeros_like(theta)) if gradient else 1e100
        out = self.base.value_grad(core, gradient)
        value = out[0] if gradient else out
        grad = np.r_[out[1], np.zeros(G)] if gradient else None
        for g, (idx, a, b) in enumerate(self.groups):
            beta = core[idx]; inv = np.exp(-s[g]); J = len(idx)
            # -log N(beta|0,e^s) - log InvGamma(e^s|a,b) - log Jacobian.
            value += .5*inv*beta@beta+.5*J*s[g]+a*s[g]+b*inv
            if gradient:
                grad[idx] += beta*inv
                grad[len(core)+g] = -.5*inv*beta@beta+.5*J+a-b*inv
        return (value, grad) if gradient else value


def month_keys(dates):
    return pd.DatetimeIndex(dates).to_period('M').astype(str)


def make_problem(train, component, hier_month, month_re):
    p = m.make_problem(train, component, True)
    d = train[train.observed].copy()
    if component != 'N':
        person = component.split('_')[1]
        w = d.available_broadcasts if component[0] == 'B' else d[f'{person}_broadcasts']
        d = d[w.to_numpy(float) > 0]
    X, names, sd, prior = p.X, list(p.names[:-1]), p.sd.copy(), p.prior.copy()
    groups = []
    if hier_month:
        idx = np.array([i for i, n in enumerate(names) if n.startswith('month_c')])
        sd[idx] = np.inf; groups.append((idx, IG_SHAPE, IG_SCALE))
    keys = []
    if month_re:
        keys = sorted(set(month_keys(d.date)))
        Z = (month_keys(d.date).to_numpy()[:, None] == np.array(keys)[None, :]).astype(float)
        start = X.shape[1]; X = np.column_stack([X, Z]); names += [f'level_{k}' for k in keys]
        idx = np.arange(start, start+len(keys))
        sd = np.r_[sd[:-1], np.full(len(keys), np.inf), sd[-1]]
        prior = np.r_[prior[:-1], np.zeros(len(keys)), prior[-1]]
        groups.append((idx, IG_SHAPE, IG_SCALE))
    base = m.Problem(X, p.y, p.exposure, prior, sd, p.kind, p.center, names+['log_dispersion'])
    labels = (['log_tau2_month'] if hier_month else [])+(['log_sigma2_level'] if month_re else [])
    return HierProblem(base, tuple(groups)), keys, labels


def posterior_mode(problem):
    """MAP and Hessian; the Hessian only preconditions the sampler."""
    start = np.r_[problem.base.prior, np.full(len(problem.groups), np.log(IG_SCALE))]
    opt = minimize(problem.value_grad, start, jac=True, method='BFGS', options={'gtol': 2e-5, 'maxiter': 2000})
    x = opt.x; dim = len(x); eps = 1e-4
    for _ in range(8):
        H = np.column_stack([(problem.value_grad(x+np.eye(dim)[j]*eps)[1]
                              -problem.value_grad(x-np.eye(dim)[j]*eps)[1])/(2*eps) for j in range(dim)])
        score = problem.value_grad(x)[1]
        if np.max(np.abs(score)) < 1e-4:
            break
        step = np.linalg.solve((H+H.T)/2, score); improved = False
        for fraction in (1., .5, .25, .125):
            candidate = x-fraction*step
            if np.max(abs(problem.value_grad(candidate)[1])) < np.max(abs(score)):
                x = candidate; improved = True; break
        if not improved:
            break
    eigen, U = np.linalg.eigh((H+H.T)/2)
    if eigen.min() <= 0:
        raise RuntimeError('Invalid posterior curvature')
    return x, (U/eigen)@U.T, float(eigen.min())


def sample(problem, seed, draws=1000, warm=400, path=1.6):
    """HMC (L3, pp. 51-53) in coordinates whitened by the Laplace covariance.

    Normal momentum, leapfrog path, energy accept/reject; the step size is
    tuned in warmup towards 80% acceptance, then frozen. Path lengths are
    jittered. Energy errors above 1000 are counted as divergences.
    The v1 independence sampler accepted ~2.5% with sampled variances.
    """
    x0, cov, min_eigen = posterior_mode(problem); dim = len(x0)
    L = np.linalg.cholesky(cov); rng = np.random.default_rng(seed)

    def energy(z):
        value, grad = problem.value_grad(x0+L@z)
        return value, L.T@grad

    chains = np.empty((4, draws, dim)); accepts = []; divergences = 0; steps = []
    for c in range(4):
        z = rng.normal(0, .5, dim); U, g = energy(z); log_eps = np.log(.3); kept = []; acc = 0
        for i in range(warm+draws):
            eps = np.exp(log_eps) if i < warm else step
            n = max(1, int(round(path/eps*rng.uniform(.7, 1.3))))
            p = rng.standard_normal(dim); H0 = U+.5*p@p
            zn, pn, Un, gn = z.copy(), p-.5*eps*g, U, g
            for k in range(n):
                zn = zn+eps*pn; Un, gn = energy(zn)
                if Un >= 1e99:
                    break
                if k < n-1:
                    pn = pn-eps*gn
            pn = pn-.5*eps*gn; error = Un+.5*pn@pn-H0
            prob = 0. if not np.isfinite(error) or Un >= 1e99 else min(1., np.exp(-error))
            if np.isfinite(error) and error > 1000 and i >= warm:
                divergences += 1
            if rng.random() < prob:
                z, U, g = zn, Un, gn; acc += i >= warm
            if i < warm:
                log_eps += (prob-.8)/np.sqrt(i+10)
                if i >= warm//2:
                    kept.append(log_eps)
                if i == warm-1:
                    step = float(np.exp(np.mean(kept))); steps.append(step)
            else:
                chains[c, i-warm] = x0+L@z
        accepts.append(acc/draws)
    diag = m.mcmc_diagnostics(chains)
    diag.update(sampler='HMC, Laplace-whitened', n=int(len(problem.base.y)), dim=dim, acceptance=accepts,
                step_size=steps, divergences=divergences, draws_per_chain=draws,
                gradient_max=float(abs(problem.value_grad(x0)[1]).max()), hessian_min_eigen=min_eigen)
    if diag['gradient_max'] > .01:
        raise RuntimeError(f'Unresolved posterior optimization: {diag}')
    if diag['max_rhat'] >= 1.01 or min(diag['min_bulk_ess'], diag['min_tail_ess']) < 400 or divergences:
        # Chains that converge earlier are unchanged by this cap (same seeds).
        if draws < 32000:
            return sample(problem, seed, draws*2, warm*2, path)
        raise RuntimeError(f'MCMC did not converge: {diag}')
    return chains.reshape(-1, dim), x0, diag


def fit_component(train, component, seed, hier_month, month_re):
    problem, keys, labels = make_problem(train, component, hier_month, month_re)
    theta, x, diag = sample(problem, seed)
    names = problem.base.names; n_fixed = len(m.design(train.date[:1], m.CHANNELS[:1], True)[1])
    diag.update(component=component)
    return {'theta': theta, 'map': x, 'names': names+labels, 'center': problem.base.center,
            'n_fixed': n_fixed, 'level_keys': keys, 'hier_month': hier_month, 'month_re': month_re,
            'diagnostics': diag}


def fit_static(train, seed, hier_month, month_re):
    return {name: fit_component(train, name, seed+23*i, hier_month, month_re)
            for i, name in enumerate(m.COMPONENTS)}


def parameter_paths(fit, dates, n, rng):
    future = m.grid(dates); theta = fit['theta'][rng.integers(len(fit['theta']), size=n)]
    k = fit['n_fixed']; X = m.design(future.date, future.channel, True, fit['center'])[0]
    eta = theta[:, :k]@X.T
    if fit['month_re']:
        keys = fit['level_keys']
        s_level = theta[:, -1]; future_keys = month_keys(future.date)
        for key in sorted(set(future_keys)):
            use = future_keys == key
            if key in keys:  # ragged edge: the month is partly observed
                u = theta[:, k+keys.index(key)]
            else:
                u = rng.normal(0, np.sqrt(np.exp(s_level)))
            eta[:, use] += u[:, None]
    disp = theta[:, k+len(fit['level_keys'])]
    return eta.reshape(n, len(dates), 4), np.exp(disp)[:, None, None]


def static_predict(fits, dates, n, seed):
    rng = np.random.default_rng(seed)
    return m.simulate_components({name: parameter_paths(fit, dates, n, rng) for name, fit in fits.items()}, rng)
