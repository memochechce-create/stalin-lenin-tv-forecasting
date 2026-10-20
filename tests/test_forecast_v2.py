"""Numerical identities for the version 2 components."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis'))
import joint_count_models as m
import forecast_v2 as v
import evaluate_forecast_v2 as e


class ForecastV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = []; rng = np.random.default_rng(3)
        for i, day in enumerate(pd.date_range('2023-01-01', periods=120)):
            for c, channel in enumerate(m.CHANNELS):
                b = int(rng.integers(0, 4))
                rows.append(dict(date=day, channel=channel, observed=i % 17 != 3, available_broadcasts=10+c,
                                 lenin_broadcasts=b, lenin_mentions=b+int(rng.integers(0, 3)*(b > 0)),
                                 stalin_broadcasts=b, stalin_mentions=2*b))
        cls.d = pd.DataFrame(rows)

    def test_hierarchical_gradients(self):
        for name in m.COMPONENTS:
            for hier, level in ((True, False), (False, True), (True, True)):
                p, _, _ = v.make_problem(self.d, name, hier, level)
                x = np.r_[p.base.prior, np.full(len(p.groups), -1.)]+np.linspace(-.03, .03, p.dim); eps = 1e-5
                numeric = np.array([(p.value_grad(x+u*eps, False)-p.value_grad(x-u*eps, False))/(2*eps) for u in np.eye(p.dim)])
                np.testing.assert_allclose(p.value_grad(x)[1], numeric, rtol=1e-4, atol=1e-4)

    def test_inverse_gamma_hyperprior(self):
        # With no coefficients the log-variance density is IG(a, b) on e^s times e^s.
        p, _, _ = v.make_problem(self.d, 'N', True, False); idx, a, b = p.groups[0]
        x = np.r_[p.base.prior, 0.]; x[idx] = 0.
        s = np.array([-2., -1., 0.]); values = []
        for value in s:
            x[-1] = value; values.append(p.value_grad(x, False))
        expected = .5*len(idx)*s+a*s+b*np.exp(-s)
        np.testing.assert_allclose(np.diff(values), np.diff(expected), atol=1e-9)

    def test_gram_matches_sample_crps(self):
        rng = np.random.default_rng(1)
        a = rng.poisson(3, (500, 7)); b = rng.poisson(6, (300, 7)); y = rng.poisson(4, 7).astype(float)
        G = e.gram([a, b], y)
        np.testing.assert_allclose(G[:, 0, 0], m.crps_sample(a, y), atol=1e-12)
        np.testing.assert_allclose(G[:, 1, 1], m.crps_sample(b, y), atol=1e-12)
        mix = np.r_[a, a, a, b, b, b, b, b]  # weights 1500/3000 and 1500/3000
        w = np.array([.5, .5])
        np.testing.assert_allclose(np.einsum('k,nkl,l->n', w, G, w), m.crps_sample(mix, y), atol=1e-12)

    def test_shrunk_weights_are_on_simplex(self):
        rng = np.random.default_rng(2); X = rng.normal(size=(6, 4, 3))
        f = {'Gd': np.einsum('tik,tjk->tij', X, X), 'nd': np.full(6, 10), 'Gm': np.einsum('tik,tjk->tij', X, X), 'nm': np.full(6, 2)}
        raw, shrunk = e.stack_weights(f, np.ones(6, bool))
        self.assertAlmostEqual(raw.sum(), 1); self.assertTrue((raw >= 0).all())
        np.testing.assert_allclose(shrunk, 6/36*raw+30/36/4)  # K = 4 candidates

    def test_new_month_level_is_shared_and_observed_month_reused(self):
        k = len(m.design(self.d.date[:1], m.CHANNELS[:1], True)[1])
        theta = np.zeros((50, k+2+1+2)); theta[:, k+1] = 1.5  # level of 2023-03, the last observed month
        theta[:, -1] = np.log(.04)
        fit = {'theta': theta, 'names': ['x']*(k+3), 'center': 0., 'n_fixed': k, 'level_keys': ['2023-02', '2023-03'],
               'month_re': True}
        eta, _ = v.parameter_paths(fit, pd.date_range('2023-03-30', '2023-04-02'), 50, np.random.default_rng(0))
        np.testing.assert_allclose(eta[:, :2], 1.5+0*eta[:, :2])  # ragged edge: posterior level of March
        april = eta[:, 2:]
        self.assertTrue(np.allclose(april, april[:, :1, :1]))  # one draw shared by days and channels
        self.assertGreater(april[:, 0, 0].std(), .05)


if __name__ == '__main__':
    unittest.main()
