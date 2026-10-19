"""Independent numerical identities and deployment constraints for the forecast."""
import sys
import unittest
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import betabinom, nbinom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis'))
import joint_count_models as m


class JointForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = []
        for i, day in enumerate(pd.date_range('2023-01-01', periods=28)):
            for c, channel in enumerate(m.CHANNELS):
                rows.append(dict(date=day, channel=channel, observed=i != 3,
                                 available_broadcasts=10+c,
                                 lenin_broadcasts=(i+c) % 3, lenin_mentions=2*((i+c) % 3),
                                 stalin_broadcasts=(i+c) % 4, stalin_mentions=3*((i+c) % 4)))
        cls.d = pd.DataFrame(rows)

    def test_likelihoods_against_scipy(self):
        y = np.arange(15); rate = 1.8; k = .6
        for exposure in (1, 3, 9):
            np.testing.assert_allclose(m.nb_logpmf(y, rate, k, exposure),
                                       nbinom.logpmf(y, exposure*k, k/(k+rate)), atol=1e-12)
        np.testing.assert_allclose(m.bb_logpmf(y, 20, .08, 12),
                                   betabinom.logpmf(y, 20, .96, 11.04), atol=1e-12)

    def test_all_analytic_gradients(self):
        for name in m.COMPONENTS:
            for calendar in (False, True):
                p = m.make_problem(self.d, name, calendar)
                x = p.prior+np.linspace(-.03, .03, len(p.prior)); eps = 1e-5
                numeric = np.array([(p.value_grad(x+v*eps, False)-p.value_grad(x-v*eps, False))/(2*eps)
                                    for v in np.eye(len(x))])
                np.testing.assert_allclose(p.value_grad(x)[1], numeric, atol=2e-5, rtol=2e-5)

    def test_missing_rows_cannot_change_static_fit(self):
        d = self.d.copy(); d.loc[~d.observed, ['available_broadcasts', 'lenin_mentions', 'lenin_broadcasts']] = 9999
        for name in ('N', 'B_lenin', 'E_lenin'):
            p = m.make_problem(d, name, True); original = m.make_problem(self.d, name, True)
            self.assertEqual(p.value_grad(p.prior, False), original.value_grad(original.prior, False))

    def test_sample_crps_against_pairwise_definition(self):
        s = np.array([[0, 3], [1, 6], [4, 7], [9, 12]]); y = np.array([3, 8])
        truth = np.mean(abs(s-y), axis=0)-.5*np.mean(abs(s[:, None, :]-s[None, :, :]), axis=(0, 1))
        np.testing.assert_allclose(m.crps_sample(s, y), truth)

    def test_joint_generator_support_and_pool(self):
        rng = np.random.default_rng(17); shape = (1000, 7, 4)
        params = {name: (np.full(shape, -2. if name[0] == 'B' else 1.), np.full((1000, 1, 1), 2.))
                  for name in m.COMPONENTS}
        paths = m.simulate_components(params, rng)
        for person in m.PEOPLE:
            B = paths[person+'_broadcasts']; C = paths[person+'_mentions']
            self.assertTrue(np.all(B <= paths['N']))
            self.assertTrue(np.all(B <= C))
            self.assertTrue(np.all(C[B == 0] == 0))
        pool = m.linear_pool({name: paths for name in ('M1', 'M2', 'M3', 'M4')}, 5)
        for key in paths:
            np.testing.assert_array_equal(pool[key], paths[key])

    def test_exposure_additivity(self):
        # NB per positive broadcast sums to NB with size B*k. Check independent
        # distribution moments as well as a Monte Carlo sum of broadcast counts.
        rng = np.random.default_rng(8); B = 4; k = .7; rate = 2.
        summed = rng.negative_binomial(k, k/(k+rate), (200000, B)).sum(axis=1)
        self.assertAlmostEqual(summed.mean(), B*rate, delta=.05)
        self.assertAlmostEqual(summed.var(), B*(rate+rate**2/k), delta=.4)

    def test_exact_holidays_and_sum_zero_contrasts(self):
        dates = pd.to_datetime(['2025-05-08', '2025-05-09', '2025-11-07'])
        X, names, _ = m.design(dates, ['1TV']*3, True)
        np.testing.assert_array_equal(X[:, names.index('exact_may8')], [1, 0, 0])
        np.testing.assert_array_equal(X[:, names.index('exact_nov7')], [0, 0, 1])
        self.assertEqual(names.count('exact_may9'), 1)

    def test_forecast_file_needs_no_pickle(self):
        from run_joint_forecast import save_paths
        dates = pd.date_range('2026-10-03', periods=31)
        buffer = io.BytesIO()
        save_paths(buffer, {'M1': {'N': np.zeros((2, 31, 4), dtype=int)}}, dates)
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as result:
            self.assertEqual(result['dates'][0], '2026-10-03')
            self.assertEqual(result['dates'][-1], '2026-11-02')
            self.assertEqual(result['M1__N'].shape, (2, 31, 4))


if __name__ == '__main__':
    unittest.main()
