"""Unit tests for the split-conformal machinery (flare/conformal.py).

Run:  python -m pytest tests/test_conformal.py -q
or:   python tests/test_conformal.py   (runs the same checks with asserts)
"""
import math
import numpy as np
import pandas as pd

from flare import conformal as CF


def test_conformal_q_order_statistic():
    # m=19, alpha=0.05 -> j = ceil(20*0.95) = 19 -> the max of 19 sorted scores.
    s = np.arange(1, 20)  # 1..19
    assert CF.conformal_q(s, 0.05) == 19.0
    # m=20, alpha=0.05 -> j = ceil(21*0.95) = ceil(19.95) = 20 -> 20th smallest.
    s = np.arange(1, 21)
    assert CF.conformal_q(s, 0.05) == 20.0
    # m=9, alpha=0.10 -> j = ceil(10*0.9) = 9 -> the max.
    assert CF.conformal_q(np.arange(1, 10), 0.10) == 9.0
    # It must NOT equal numpy's interpolated quantile in general.
    rng = np.random.RandomState(0)
    x = rng.rand(37)
    assert not math.isclose(CF.conformal_q(x, 0.10), float(np.quantile(x, 0.90)), rel_tol=1e-9)


def test_conformal_q_unavailable():
    # j > m -> inf. m=18, alpha=0.05 -> j = ceil(19*0.95)=ceil(18.05)=19 > 18.
    assert CF.conformal_q(np.arange(18), 0.05) == float('inf')
    assert CF.conformal_q([], 0.05) == float('inf')


def test_visit_group():
    assert [CF.visit_group(k) for k in (1, 2, 3, 4, 5, 30)] == ['1', '2', '3', '4+', '4+', '4+']


def test_group_residuals_one_per_subject():
    # subject A: visits 1..5 ; subject B: visits 1..2
    rows = []
    for pid, T in [('A', 5), ('B', 2)]:
        for k in range(1, T + 1):
            rows.append({'pid': pid, 'visit_index': k, 'y_true': float(k), 'y_pred': 0.0})
    cal = pd.DataFrame(rows)
    gr = CF.group_residuals(cal, subsample_seed=1)
    assert len(gr['1']) == 2 and len(gr['2']) == 2 and len(gr['3']) == 1
    assert len(gr['4+']) == 1          # only subject A reaches >=4, ONE residual
    assert gr['4+'][0] in {4.0, 5.0}   # a randomly chosen >=4 residual


def test_subsample_ge4_rows_one_per_subject():
    rows = []
    for pid, T in [('A', 6), ('B', 2), ('C', 4)]:
        for k in range(1, T + 1):
            rows.append({'pid': pid, 'visit_index': k, 'y_true': 0.0, 'y_pred': 0.0})
    df = pd.DataFrame(rows)
    ev = CF.subsample_ge4_rows(df, seed=3)
    # all visits 1-3 kept: A has 3, B has 2, C has 3 -> 8 low rows
    assert (ev['visit_index'] <= 3).sum() == 8
    # exactly one >=4 row per subject that reaches >=4 (A, C) -> 2 rows
    ge4 = ev[ev['visit_index'] >= 4]
    assert len(ge4) == 2 and set(ge4['pid']) == {'A', 'C'}
    assert ge4[ge4.pid == 'A']['visit_index'].iloc[0] in (4, 5, 6)


def test_interval_validity_and_metrics():
    test = pd.DataFrame({'pid': ['A', 'A', 'B'], 'visit_index': [1, 4, 4],
                         'y_true': [0.0, 0.0, 5.0], 'y_pred': [0.0, 0.0, 0.0]})
    calib = {g: {'m': 10, 'emp_qL': -1.0, 'emp_qU': 1.0, 'cparam_c': 0.0} for g in CF.GROUPS}
    L, U = CF.build_intervals(test, 'empirical', -1.0, 1.0, calib)
    assert np.all(U >= L) and np.all(np.isfinite(L)) and np.all(np.isfinite(U))
    im = CF.interval_metrics(test, L, U, 0.10)
    assert im['1']['coverage'] == 1.0            # y=0 in [-1,1]
    assert im['4+']['coverage'] == 0.5           # A covered, B (y=5) not
    assert abs(im['overall']['avg_width'] - 2.0) < 1e-9


def test_shortest_interval():
    # unimodal symmetric (normal): shortest 90% width ~ central 90% width (2*1.645) and never wider
    e = np.sort(np.random.RandomState(0).normal(0, 1, 200000))
    qL, qU = CF.shortest_interval(e, 0.10)
    cL, cU = float(np.quantile(e, 0.05)), float(np.quantile(e, 0.95))
    assert (qU - qL) <= (cU - cL) + 1e-9              # shortest is never wider than central
    assert abs((qU - qL) - 2 * 1.645) < 0.1          # ~ the central 90% width
    # right-skewed sample: shortest window sits low and is NARROWER than the central 90%
    rng = np.random.RandomState(0)
    s = np.sort(np.concatenate([rng.normal(0, 1, 9000), rng.normal(8, 1, 1000)]))
    qLs, qUs = CF.shortest_interval(s, 0.10)
    cL, cU = float(np.quantile(s, 0.05)), float(np.quantile(s, 0.95))
    assert (qUs - qLs) <= (cU - cL) + 1e-9           # shortest <= central width
    assert qUs < cU                                  # avoids the far upper mode


def test_asymmetric_empirical():
    # right-skewed residuals -> asymmetric band: emp_qL != -emp_qU, and coverage >= nominal
    rng = np.random.RandomState(1)
    r = np.concatenate([rng.normal(0, 1, 900), rng.normal(6, 1, 100)])  # skewed right
    calib = CF.calibrate({'1': r}, 0.10, qL=-2.0, qU=2.0)
    lo, hi = calib['1']['emp_qL'], calib['1']['emp_qU']
    assert hi > 0 and lo < 0 and abs(hi + lo) > 0.5      # clearly asymmetric
    # empirical coverage on a fresh draw is >= 1-alpha
    rn = np.concatenate([rng.normal(0, 1, 900), rng.normal(6, 1, 100)])
    cov = np.mean((rn >= lo) & (rn <= hi))
    assert cov >= 0.90 - 0.03


def test_cparam_floor_prevents_inversion():
    # A hugely negative correction would invert; the floor caps it at -(qU-qL)/2.
    gr = {'1': np.zeros(50)}  # residuals all 0 -> S_cpar = max(qL, -qU) = -1 (want shrink)
    calib = CF.calibrate(gr, 0.10, qL=-1.0, qU=1.0)
    c = calib['1']['cparam_c']
    assert c >= -(1.0 - (-1.0)) / 2.0 - 1e-9      # floored at -(qU-qL)/2 = -1
    test = pd.DataFrame({'pid': ['A'], 'visit_index': [1], 'y_true': [0.0], 'y_pred': [0.0]})
    L, U = CF.build_intervals(test, 'cparam', -1.0, 1.0, calib)
    assert U[0] >= L[0]


def _synth_panel(n_subj, sd_fn, seed):
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n_subj):
        for k in range(1, 6):                       # 5 visits -> groups 1,2,3,4+
            yp = float(rng.uniform(-2, 2))
            r = float(rng.normal(0.0, sd_fn(yp)))
            rows.append({'pid': f'{seed}_{s}', 'visit_index': k, 'y_pred': yp, 'y_true': yp + r,
                         'hidden_norm': abs(float(rng.normal()))})
    return pd.DataFrame(rows)


def test_utopia_valid_coverage():
    cal = _synth_panel(400, lambda yp: 1.0, 0)
    test = _synth_panel(400, lambda yp: 1.0, 1)
    tiv = CF.subsample_ge4_rows(test, seed=123)
    y = tiv['y_true'].to_numpy()
    for mode in ('base', 'het_dist_robust_noshape'):
        L, U = CF.utopia_intervals(cal, tiv, 'gaussian', {'sigma': 1.0}, -9, 9,
                                   0.10, split_seed=7, mode=mode)
        assert np.all(U >= L) and np.all(np.isfinite(L)) and np.all(np.isfinite(U))
        assert np.mean((y >= L) & (y <= U)) >= 0.90 - 0.06     # gamma-optimized UTOPIA coverage


def test_utopia_het_adapts():
    cal = _synth_panel(500, lambda yp: 0.3 + abs(yp), 2)       # heteroscedastic sd (in y_pred)
    test = _synth_panel(500, lambda yp: 0.3 + abs(yp), 3)
    tiv = CF.subsample_ge4_rows(test, seed=9)
    Lb, Ub = CF.utopia_intervals(cal, tiv, 'gaussian', {'sigma': 1.5}, -9, 9, 0.10, 5, mode='base')
    Lh, Uh = CF.utopia_intervals(
        cal, tiv, 'gaussian', {'sigma': 1.5}, -9, 9, 0.10, 5,
        mode='het_dist_robust_noshape')
    g = tiv['visit_index'].map(CF.visit_group).to_numpy()
    m = g == '4+'
    assert np.std((Ub - Lb)[m]) < 1e-6                         # base: per-group-constant width
    assert np.std((Uh - Lh)[m]) > 1e-3                         # het: adapts within a group
    y = tiv['y_true'].to_numpy()
    for L, U in [(Lb, Ub), (Lh, Uh)]:
        assert np.mean((y >= L) & (y <= U)) >= 0.90 - 0.06


def test_utopia_split_coverage():
    # Regression test for the split-calibration path (split=True): fit envelope+gamma on
    # cal-A, set scale factors on held-out cal-B. The bug this guards against dropped the
    # out-of-envelope cal-B misses and capped lam at 1, causing infinite/undercovering bands.
    cal = _synth_panel(600, lambda yp: 1.0, 10)
    test = _synth_panel(600, lambda yp: 1.0, 11)
    tiv = CF.subsample_ge4_rows(test, seed=321)
    y = tiv['y_true'].to_numpy()
    for mode in ('base', 'het_dist_robust_noshape'):
        L, U = CF.utopia_intervals(cal, tiv, 'gaussian', {'sigma': 1.0}, -9, 9,
                                   0.10, split_seed=7, mode=mode, split=True)
        assert np.all(U >= L)
        assert np.all(np.isfinite(L)) and np.all(np.isfinite(U))   # no infinite bands
        assert np.mean((y >= L) & (y <= U)) >= 0.90 - 0.05         # ~nominal, not undercovering


def test_pooled_rmse():
    # pooled over two groups equals sqrt of weighted mean of squares
    assert abs(CF.pooled_rmse([(2.0, 100), (4.0, 100)]) -
               math.sqrt((4 * 100 + 16 * 100) / 200)) < 1e-9


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\nall {len(fns)} conformal unit tests passed")
