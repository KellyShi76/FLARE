"""Split-conformal prediction intervals, calibrated per visit group {1,2,3,>=4}.

Three intervals per fitted model (see run_conformal.py):
  (a) parametric  -- SHORTEST connected 1-alpha interval of the fitted error law (plug-in,
                     NOT conformal): the minimum-length window of the Monte-Carlo sample;
  (b) empirical   -- ASYMMETRIC split-conformal band from the signed calibration residuals
                     (one order statistic per tail, alpha/2 each);
  (c) cparam      -- conformalized-parametric (symmetric CQR correction to the (a) band).

Groups 1/2/3 use the exact visit (naturally one residual per subject); the >=4 group uses
ONE randomly chosen >=4 residual per calibration subject, so within every group the
calibration scores are independent across subjects -> exact finite-sample 1-alpha coverage.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .evaluate import sample_error, compute_point_metrics

GROUPS = ['1', '2', '3', '4+']
LEVELS = [0.90, 0.95]
HZ2GROUP = {'Visit 1': '1', 'Visit 2': '2', 'Visit 3': '3', 'Visit >=4': '4+', 'All': 'overall'}
MC_SEED = 20240101
UTOPIA_ENVELOPE_EPS = 1e-12


# ---------------------------------------------------------------- primitives
def visit_group(k):
    """Map a 1-based within-subject visit index to its reporting group."""
    k = int(k)
    return str(k) if k in (1, 2, 3) else '4+'


def conformal_q(scores, alpha):
    """Split-conformal quantile via the EXPLICIT order statistic.

    j = ceil((m+1)(1-alpha)); returns the j-th smallest score (1-indexed).
    Returns +inf (i.e. unavailable) when j > m, which is the honest behaviour --
    NEVER fall back to an interpolated empirical quantile (that voids the guarantee).
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    m = s.size
    if m == 0:
        return np.inf
    j = int(np.ceil((m + 1) * (1.0 - alpha)))
    if j > m:
        return np.inf
    return float(np.sort(s)[j - 1])


# ---------------------------------------------------------------- parametric law
def shortest_interval(sorted_e, alpha):
    """Minimum-length connected interval containing 1-alpha mass of an ASCENDING sample.

    k = ceil((1-alpha)*M); over every window of k consecutive draws pick the narrowest,
    q_L = e[j*], q_U = e[j*+k-1]. This is the shortest connected prediction interval of
    the fitted law (for a multimodal fit an HDR set could be disconnected, but we report
    intervals). Returns (qL, qU)."""
    e = np.asarray(sorted_e, dtype=float)
    M = e.size
    if M == 0:
        return float('-inf'), float('inf')
    k = int(np.ceil((1.0 - alpha) * M))
    k = min(max(k, 1), M)
    widths = e[k - 1:] - e[:M - k + 1]
    j = int(np.argmin(widths))
    return float(e[j]), float(e[j + k - 1])


def parametric_quantiles(dist, dist_params, alpha, cutoff, cutoff_upper,
                         mc_seed=20240101, n=500_000, verify=True):
    """SHORTEST connected 1-alpha interval of the fitted mean-zero error law, from one large
    truncated Monte-Carlo sample. Plug-in (no m+1 conformal correction -- that is applied later
    to the calibration scores). Returns (qL, qU, mc_mean, mc_min, mc_max)."""
    mc = sample_error(dist, dist_params, n=n, seed=mc_seed,
                      cutoff=cutoff, cutoff_upper=cutoff_upper)
    mc_mean, mc_min, mc_max = float(np.mean(mc)), float(np.min(mc)), float(np.max(mc))
    if verify:
        assert abs(mc_mean) < 0.05, f"parametric MC not mean-zero ({dist}): mean={mc_mean:.4f}"
        # Only FLARE's built-in truncated laws are required to stay inside the
        # configured support. A custom law may legitimately be untruncated.
        if dist in ('t', 'tukeyh', 'truncmix', 'truncmix_gauss',
                    'truncmix_t', 'truncmix_tukeyh'):
            assert mc_min >= cutoff - 1e-6 and mc_max <= cutoff_upper + 1e-6, \
                f"parametric MC outside truncation ({dist}): [{mc_min:.3f},{mc_max:.3f}]"
    qL, qU = shortest_interval(np.sort(mc), alpha)
    return qL, qU, mc_mean, mc_min, mc_max


# ---------------------------------------------------------------- calibration
def group_residuals(cal_pred, subsample_seed=0):
    """Per-group calibration residuals, ONE per subject.

    cal_pred: DataFrame with columns pid, visit_index, y_true, y_pred.
    Groups '1','2','3' are the exact visits (one per subject already). '4+' takes one
    randomly chosen >=4 residual per subject. Returns {group: 1-D residual array}.
    """
    d = cal_pred.copy()
    d['r'] = d['y_true'].to_numpy() - d['y_pred'].to_numpy()
    out = {}
    for g in ('1', '2', '3'):
        out[g] = d.loc[d['visit_index'] == int(g), 'r'].to_numpy()
    ge4 = d[d['visit_index'] >= 4]
    rng = np.random.RandomState(subsample_seed)
    picks = [grp['r'].to_numpy()[rng.randint(len(grp))]
             for _, grp in ge4.groupby('pid', sort=True)]
    out['4+'] = np.asarray(picks, dtype=float)
    return out


def subsample_ge4_rows(pred_df, seed=0):
    """Test-side evaluation frame: keep all visit-1/2/3 rows and ONE randomly chosen
    >=4 visit per subject, so the >=4 interval-coverage target matches the exchangeable
    calibration construction (one residual per subject). Visits 1-3 are already one per
    subject. Point/distributional metrics use the FULL frame, not this one."""
    lo = pred_df[pred_df['visit_index'] <= 3]
    ge4 = pred_df[pred_df['visit_index'] >= 4]
    rng = np.random.RandomState(seed)
    keep = [grp.index[rng.randint(len(grp))] for _, grp in ge4.groupby('pid', sort=True)]
    picked = ge4.loc[keep]
    return (pd.concat([lo, picked])
              .sort_values(['pid', 'visit_index']).reset_index(drop=True))


def calibrate(group_resid, alpha, qL, qU):
    """Per-group conformal quantities at one level.

    Returns dict group -> {m, emp_qL, emp_qU, cparam_c}:
      emp_qU = conformal_q(r, alpha/2), emp_qL = -conformal_q(-r, alpha/2)
               (asymmetric empirical split-conformal; alpha/2 per tail -> coverage >= 1-alpha)
      cparam_c = max(conformal_q(max(qL-r, r-qU)), -(qU-qL)/2)   (symmetric CQR correction, floored)
    """
    floor = -(qU - qL) / 2.0
    res = {}
    for g, r in group_resid.items():
        r = np.asarray(r, dtype=float)
        emp_qU = conformal_q(r, alpha / 2.0)
        emp_qL = -conformal_q(-r, alpha / 2.0)
        s_cpar = np.maximum(qL - r, r - qU)
        c = conformal_q(s_cpar, alpha)
        cparam_c = c if not np.isfinite(c) else max(c, floor)
        res[g] = {'m': int(r.size), 'emp_qL': emp_qL, 'emp_qU': emp_qU, 'cparam_c': cparam_c}
    return res


# ---------------------------------------------------------------- intervals
def build_intervals(test_pred, method, qL, qU, calib):
    """Return (L, U) arrays aligned with test_pred rows for one interval method.

    method: 'parametric' | 'empirical' | 'cparam'. calib is the calibrate() dict.
    Unavailable (inf) group quantities yield an infinite interval for that group's rows.
    """
    yp = test_pred['y_pred'].to_numpy()
    groups = test_pred['visit_index'].map(visit_group).to_numpy()
    L = np.empty(len(yp)); U = np.empty(len(yp))
    for g in GROUPS:
        mask = groups == g
        if not mask.any():
            continue
        if method == 'parametric':
            L[mask], U[mask] = yp[mask] + qL, yp[mask] + qU
        elif method == 'empirical':
            L[mask] = yp[mask] + calib[g]['emp_qL']
            U[mask] = yp[mask] + calib[g]['emp_qU']
        elif method == 'cparam':
            c = calib[g]['cparam_c']
            L[mask], U[mask] = yp[mask] + qL - c, yp[mask] + qU + c
        else:
            raise ValueError(method)
    return L, U


def interval_metrics(test_pred, L, U, alpha):
    """Coverage, average width, interval (Winkler) score per group and pooled overall.

    Returns {group -> {n, coverage, avg_width, interval_score}} with an 'overall' key.
    inf-width rows count as covered (score = inf) -- honest for unavailable groups.
    """
    y = test_pred['y_true'].to_numpy()
    groups = test_pred['visit_index'].map(visit_group).to_numpy()
    width = U - L
    covered = (y >= L) & (y <= U)
    score = width + (2.0 / alpha) * np.where(y < L, L - y, 0.0) \
                  + (2.0 / alpha) * np.where(y > U, y - U, 0.0)
    out = {}
    for g in GROUPS + ['overall']:
        mask = np.ones(len(y), bool) if g == 'overall' else (groups == g)
        n = int(mask.sum())
        if n == 0:
            continue
        out[g] = {'n': n,
                  'coverage': float(np.mean(covered[mask])),
                  'avg_width': float(np.mean(width[mask])),
                  'interval_score': float(np.mean(score[mask]))}
    return out


def pooled_rmse(group_rmse_n):
    """Pooled RMSE = sqrt(sum N_g RMSE_g^2 / sum N_g). Input: list of (rmse, n)."""
    num = sum(r * r * n for r, n in group_rmse_n)
    den = sum(n for _, n in group_rmse_n)
    return float(np.sqrt(num / den)) if den else float('nan')


# ---------------------------------------------------------------- UTOPIA aggregation
def _subject_split(pids, frac=0.5, seed=0):
    """Split unique subject ids into (A, B) sets."""
    uniq = np.array(sorted(set(np.asarray(pids).tolist())))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    nA = int(round(frac * len(uniq)))
    return set(uniq[perm[:nA]].tolist()), set(uniq[perm[nA:]].tolist())


def group_frame(cal_pred, seed=0):
    """One row per subject per group (>=4 randomly subsampled), carrying the UTOPIA-het
    covariates (r, y_pred, visit_index, hidden_norm)."""
    d = cal_pred.copy()
    d['r'] = d['y_true'].to_numpy() - d['y_pred'].to_numpy()
    d['group'] = d['visit_index'].map(visit_group)
    if 'hidden_norm' not in d.columns:
        d['hidden_norm'] = 0.0                                # older artifacts: covariate unavailable
    cols = ['group', 'r', 'y_pred', 'visit_index', 'hidden_norm']
    lo = d[d['visit_index'] <= 3][cols]
    ge4 = d[d['visit_index'] >= 4]
    rng = np.random.RandomState(seed)
    keep = [grp.index[rng.randint(len(grp))] for _, grp in ge4.groupby('pid', sort=True)]
    picked = ge4.loc[keep][cols]
    return pd.concat([lo, picked], ignore_index=True)


def _fit_weights(Phi, t):
    """UTOPIA Step-1 LP: min_{w>=0} sum_i w.Phi_i s.t. w.Phi_i >= t_i.

    Phi is nonnegative and contains a strictly positive constant feasibility column.
    """
    Phi = np.atleast_2d(Phi)
    n, k = Phi.shape
    if n == 0:
        return np.zeros(k)
    res = linprog(c=Phi.sum(axis=0), A_ub=-Phi, b_ub=-t,
                  bounds=[(0, None)] * k, method='highs')
    if res.success:
        return np.asarray(res.x, dtype=float)
    # Numerically safe fallback: put all weight on a column that is positive in every row.
    w = np.zeros(k)
    feasible = np.flatnonzero(np.all(Phi > 0.0, axis=0))
    j = int(feasible[np.argmax(np.min(Phi[:, feasible], axis=0))])
    w[j] = float(np.max(np.maximum(t, 0.0) / Phi[:, j]))
    return w


def _fit_weights_quantile(Phi, t, tau):
    """Robust UTOPIA Step-1 envelope: min_{w>=0} sum_i w.Phi_i + C sum_i (t_i - w.Phi_i)_+ with
    C = 1/(1 - tau). This is the quantile-regression (pinball) loss at level tau with nonnegative
    weights, so about a share 1 - tau of the targets may lie above the envelope, which then
    follows the bulk of the residuals rather than the largest one."""
    from scipy import sparse
    Phi = np.atleast_2d(Phi)
    n, k = Phi.shape
    if n == 0:
        return np.zeros(k)
    c = np.r_[Phi.sum(axis=0), np.full(n, 1.0 / (1.0 - tau))]
    A = sparse.hstack([sparse.csr_matrix(-Phi), -sparse.identity(n, format='csr')], format='csr')
    res = linprog(c=c, A_ub=A, b_ub=-np.asarray(t, float), bounds=[(0, None)] * (k + n),
                  method='highs')
    if res.success:
        return np.asarray(res.x[:k], dtype=float)
    return _fit_weights(Phi, t)


def _utopia_tail_multiplier(scores, tail_budget):
    """Algorithm-3 empirical multiplier: inf{lam in [0,1]: mean(score > lam) <= budget}."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    m = scores.size
    if m == 0:
        return 1.0
    allowed = int(np.floor(m * float(tail_budget) + 1e-12))
    if allowed >= m:
        return 0.0
    # At this order statistic at most `allowed` observations are strictly larger.
    lam = np.sort(scores)[m - allowed - 1]
    return float(np.clip(lam, 0.0, 1.0))


def _calibrate_gamma(r, Lenv, Uenv, alpha, n_grid=21, gammas=None):
    """Learn an asymmetric preliminary-band shape on the envelope-fitting sample.

    For every candidate gamma, estimate both shrinkage multipliers, evaluate the
    resulting average width, and return the jointly selected
    (lambda_L, lambda_U, gamma) triple.  This routine is a training heuristic only:
    finite-sample coverage is supplied subsequently by an independent conformal
    quantile computed from *all* cal-B observations.
    """
    if r.size == 0:
        return 1.0, 1.0, 0.5
    s = r / np.where(Lenv < 0, Lenv, -1e-12)  # missed below <=> lam < s
    t = r / np.where(Uenv > 0, Uenv, 1e-12)   # missed above <=> lam < t
    grid = np.linspace(0.0, 1.0, n_grid) if gammas is None else np.atleast_1d(gammas)
    best = None
    for gam in grid:
        lam1 = _utopia_tail_multiplier(s, (1.0 - gam) * alpha)
        lam2 = _utopia_tail_multiplier(t, gam * alpha)
        w = float(np.mean(lam2 * Uenv - lam1 * Lenv))
        if best is None or w < best[0]:
            best = (w, lam1, lam2, float(gam))
    return best[1], best[2], best[3]


def _utopia_scale_scores(r, lower_width, upper_width):
    """Nested-band conformity score for asymmetric positive half-widths.

    The event ``score <= q`` is exactly
    ``-q * lower_width <= r <= q * upper_width``.  In particular, the final
    split-conformal multiplier is not restricted to [0, 1]: observations outside
    the preliminary envelope simply have scores larger than one.
    """
    r = np.asarray(r, dtype=float)
    lw = np.maximum(np.asarray(lower_width, dtype=float), UTOPIA_ENVELOPE_EPS)
    uw = np.maximum(np.asarray(upper_width, dtype=float), UTOPIA_ENVELOPE_EPS)
    return np.maximum(-r / lw, r / uw)


def utopia_intervals(cal_pred, test_iv, dist, dist_params, cutoff, cutoff_upper,
                     alpha, split_seed, mode='base', mc_seed=MC_SEED, split=True,
                     parametric_q=None):
    """Conformalized visit-group implementation of asymmetric UTOPIA aggregation.

    Per error family and visit group:
      Step 1 -- on cal-A, fit nonnegative-weighted lower/upper envelopes over candidate
        endpoint functions and tune their asymmetric preliminary shape;
      Step 2 -- on independent cal-B, score *every* observation relative to that fixed
        shape and use the exact ceil((m+1)(1-alpha)) order statistic as a common final
        multiplier.  The multiplier can exceed one.

    Thus, with split=True, the final nested band has the usual finite-sample
    split-conformal marginal coverage conditional on cal-A (subject to exchangeability).
    Candidates: {parametric, empirical, cparam} offsets. mode='het' (UTOPIA-het) augments the
    endpoint library with per-observation half-widths so the envelope can vary WITHIN a group:
    |y_hat-ybar_g|, 1/visit_index (convex, decreasing in the visit), and the hinges (q25-h)+,
    (q50-h)+, (h-q50)+, (h-q75)+ of h = ||hidden|| at its cal-A quartiles in the group, whose
    nonnegative combinations are the convex piecewise-linear functions of h. Earlier versions,
    kept for the record: 'het_linear' uses {|y_hat-ybar_g|, visit_index, h}, along which a
    nonnegative envelope can only widen; 'het_invk' and 'het_hinge' change one of its features
    ('het_convex' is the same as 'het'); 'het_mirror' adds each 'het_linear' feature mirrored
    about its largest cal-A value, cap - f (test values capped at cap). Two guards for when the
    features carry no information, both decided on cal-A alone so the coverage guarantee holds:
    'het_robust' fits the envelope by quantile regression at level 1 - alpha/2 instead of
    containing every cal-A residual; 'het_select' splits each group's cal-A rows in two, fits base
    UTOPIA and UTOPIA-het on one half, calibrates both on the other, and keeps the one with the
    smaller mean Winkler score (base UTOPIA when a half has fewer than 40 rows); 'het_robust_select' does
    both. Simpler state terms, with |y_hat-ybar_g| and 1/visit_index as in 'het', where m is
    the cal-A median of h in the group: 'het_below' uses (m-h)+, 'het_dist' |h-m|, and 'het_vee'
    (m-h)+ and (h-m)+; each with a '_robust' version using the quantile envelope, and a
    '_robust_noshape' version that also fixes the Step-1 shape multipliers at lambda = 1: the
    Algorithm-3 shrinkage assumes an envelope containing every cal-A residual, so the quantile
    envelope is passed to Step 2 as fitted. split=False is an exploratory compatibility mode that
    reuses the full calibration sample for both steps and therefore does not have this guarantee.
    Returns test (L, U) arrays."""
    if parametric_q is None:
        qLp, qUp, *_ = parametric_quantiles(
            dist, dist_params, alpha, cutoff, cutoff_upper, mc_seed=mc_seed)
    else:
        qLp, qUp = map(float, parametric_q)
    if split:
        A, B = _subject_split(cal_pred['pid'].to_numpy(), 0.5, split_seed)
        frA = group_frame(cal_pred[cal_pred['pid'].isin(A)], seed=split_seed)
        frB = group_frame(cal_pred[cal_pred['pid'].isin(B)], seed=split_seed)
    else:
        frA = frB = group_frame(cal_pred, seed=split_seed)
    gresA = {g: frA.loc[frA.group == g, 'r'].to_numpy() for g in GROUPS}
    calibA = calibrate(gresA, alpha, qLp, qUp)
    ybarA = {g: (float(frA.loc[frA.group == g, 'y_pred'].mean()) if (frA.group == g).any() else 0.0)
             for g in GROUPS}
    simple = {f'het_{shape}{suffix}' for shape in ('below', 'dist', 'vee')
              for suffix in ('', '_robust', '_robust_noshape')}
    robust = mode in ('het_robust', 'het_robust_select') or (mode in simple and '_robust' in mode)
    noshape = mode.endswith('_noshape')   # lambda = 1: the fitted envelope goes to Step 2 as it is
    select = mode in ('het_select', 'het_robust_select')   # base UTOPIA or UTOPIA-het, on cal-A
    current = mode in ('het', 'het_convex', 'het_robust', 'het_select', 'het_robust_select')
    het = current or mode in simple or mode in ('het_linear', 'het_mirror', 'het_invk', 'het_hinge')
    mirror = (mode == 'het_mirror')
    inv_k = current or mode in simple or mode == 'het_invk'   # 1/k: widens the earlier visits
    # how the state norm h enters: linearly, by four quartile hinges, or by a simpler term
    state_shape = ('hinge4' if current or mode == 'het_hinge'
                   else mode.split('_')[1] if mode in simple else 'linear')
    # cal-A quartiles of ||hidden|| in each group: the knots of the hinge features
    knotsA = {g: (np.quantile(frA.loc[frA.group == g, 'hidden_norm'], [0.25, 0.5, 0.75])
                  if (frA.group == g).any() else np.zeros(3)) for g in GROUPS}

    def base_cand(g, tail):      # finite candidate endpoint magnitudes (parametric always finite)
        v = ([qUp, calibA[g]['emp_qU'], qUp + calibA[g]['cparam_c']] if tail == 'U'
             else [-qLp, -calibA[g]['emp_qL'], -(qLp - calibA[g]['cparam_c'])])
        v = np.array(v, float)
        v = np.maximum(v[np.isfinite(v)], 0.0)  # zero-anchor every candidate interval
        return np.r_[v, 1.0]                    # technical symmetric basis guarantees feasibility

    def het_features(g, yp, vidx, hn):
        k = 1.0 / vidx.astype(float) if inv_k else vidx.astype(float)
        q25, q50, q75 = knotsA[g]
        if state_shape == 'hinge4':
            state = [np.maximum(q25 - hn, 0.0), np.maximum(q50 - hn, 0.0),
                     np.maximum(hn - q50, 0.0), np.maximum(hn - q75, 0.0)]
        elif state_shape == 'below':
            state = [np.maximum(q50 - hn, 0.0)]
        elif state_shape == 'dist':
            state = [np.abs(hn - q50)]
        elif state_shape == 'vee':
            state = [np.maximum(q50 - hn, 0.0), np.maximum(hn - q50, 0.0)]
        else:
            state = [hn]
        return np.stack([np.abs(yp - ybarA[g]), k] + state, axis=1)

    # largest cal-A value of each feature in each group, the centre of the mirrored features
    capA = {}
    if mirror:
        for g in GROUPS:
            fr = frA[frA.group == g]
            capA[g] = (het_features(g, fr['y_pred'].to_numpy(), fr['visit_index'].to_numpy(),
                                    fr['hidden_norm'].to_numpy()).max(axis=0)
                       if len(fr) else np.zeros(3))

    def phi(g, tail, yp, vidx, hn, use_het=het):   # (n, k) endpoint-candidate design matrix
        n = len(yp)
        cols = [np.tile(base_cand(g, tail), (n, 1))]
        if use_het:                      # symmetric heteroscedastic half-widths (>=0), per-observation
            f = het_features(g, yp, vidx, hn)
            if mirror:                   # f and cap - f: a slope of either sign on [0, cap]
                f = np.minimum(f, capA[g])
                f = np.concatenate([f, capA[g] - f], axis=1)
            cols.append(f)
        return np.concatenate(cols, axis=1)

    def cols_of(fr):
        return (fr.group.to_numpy(), fr['r'].to_numpy(), fr['y_pred'].to_numpy(),
                fr['visit_index'].to_numpy(), fr['hidden_norm'].to_numpy())

    gA, rA, ypA, vA, hnA = cols_of(frA)
    gB, rB, ypB, vB, hnB = cols_of(frB)
    tg = test_iv['visit_index'].map(visit_group).to_numpy()
    typ = test_iv['y_pred'].to_numpy()
    tv = test_iv['visit_index'].to_numpy()
    thn = (test_iv['hidden_norm'].to_numpy() if 'hidden_norm' in test_iv.columns
           else np.zeros(len(test_iv)))

    # Honest fallback for a group lacking either split: the conformal interval is unavailable.
    Loff = np.full(len(tg), -np.inf); Uoff = np.full(len(tg), np.inf)
    for g in GROUPS:
        mA, mB, mT = gA == g, gB == g, tg == g
        if not mT.any() or mA.sum() == 0 or mB.sum() == 0:
            continue
        def fit(rows, use_het):
            """Step 1 on the cal-A rows in `rows`: envelope weights and preliminary shape."""
            PhiU = phi(g, 'U', ypA[rows], vA[rows], hnA[rows], use_het)
            PhiL = phi(g, 'L', ypA[rows], vA[rows], hnA[rows], use_het)
            if robust and use_het:
                betaU = _fit_weights_quantile(PhiU, rA[rows], 1.0 - alpha / 2)
                betaL = _fit_weights_quantile(PhiL, -rA[rows], 1.0 - alpha / 2)
            else:
                betaU = _fit_weights(PhiU, rA[rows])
                betaL = _fit_weights(PhiL, -rA[rows])
            UA = PhiU @ betaU + UTOPIA_ENVELOPE_EPS
            LA = -(PhiL @ betaL + UTOPIA_ENVELOPE_EPS)
            if noshape and use_het:
                lam1 = lam2 = 1.0
            else:
                lam1, lam2, _ = _calibrate_gamma(rA[rows], LA, UA, alpha)
            return betaU, betaL, lam1, lam2, use_het

        def halfwidths(model, yp, vidx, hn):
            betaU, betaL, lam1, lam2, use_het = model
            low = phi(g, 'L', yp, vidx, hn, use_het) @ betaL + UTOPIA_ENVELOPE_EPS
            up = phi(g, 'U', yp, vidx, hn, use_het) @ betaU + UTOPIA_ENVELOPE_EPS
            return (np.maximum(lam1 * low, UTOPIA_ENVELOPE_EPS),
                    np.maximum(lam2 * up, UTOPIA_ENVELOPE_EPS))

        use_het = het
        if select:
            # inner split of the group's cal-A rows (one row per subject within a group):
            # fit both candidates on one half, calibrate each on the other, keep the one
            # with the smaller mean Winkler score there
            rows = np.random.RandomState(split_seed + 7 + GROUPS.index(g)).permutation(
                np.flatnonzero(mA))
            in1 = np.zeros(len(gA), bool)
            in1[rows[:len(rows) // 2]] = True
            in2 = mA & ~in1
            use_het = False
            if in1.sum() >= 40 and in2.sum() >= 40:
                score = {}
                for cand in (False, True):
                    low, up = halfwidths(fit(in1, cand), ypA[in2], vA[in2], hnA[in2])
                    q2 = conformal_q(_utopia_scale_scores(rA[in2], low, up), alpha)
                    r2, lo2, hi2 = rA[in2], -q2 * low, q2 * up
                    score[cand] = (float(np.mean((hi2 - lo2) + 2 / alpha * np.maximum(lo2 - r2, 0)
                                                 + 2 / alpha * np.maximum(r2 - hi2, 0)))
                                   if np.isfinite(q2) else np.inf)
                use_het = score[True] < score[False]

        # Step 1: fit the entire score function, including its asymmetric shape, on cal-A.
        model = fit(mA, use_het)
        # Step 2: exact final split-conformal calibration on every cal-B observation.
        lowerB, upperB = halfwidths(model, ypB[mB], vB[mB], hnB[mB])
        lowerT, upperT = halfwidths(model, typ[mT], tv[mT], thn[mT])
        q = conformal_q(_utopia_scale_scores(rB[mB], lowerB, upperB), alpha)
        Loff[mT] = -q * lowerT
        Uoff[mT] = q * upperT
    return typ + Loff, typ + Uoff


# ---------------------------------------------------------------- scoring (shared)
def point_metric_map(pm):
    """compute_point_metrics list -> {group: {rmse, mae, medae, kl, jsd, kl_resid, jsd_resid,
    mean_loglik}}."""
    out = {}
    for row in pm:
        g = HZ2GROUP.get(row['horizon'])
        if g is None:
            continue
        out[g] = {k: row.get(k, float('nan'))
                  for k in ('rmse', 'mae', 'medae', 'kl', 'jsd', 'kl_resid', 'jsd_resid',
                            'mean_loglik')}
    return out


def band_offsets(method, g, qL, qU, cg):
    """Applied (offset from y_hat) band endpoints for one (method, group)."""
    if g == 'overall':
        return (float('nan'), float('nan'))
    if method == 'parametric':
        return (qL, qU)
    if method == 'empirical':
        return (cg['emp_qL'], cg['emp_qU'])
    if method == 'cparam':
        c = cg['cparam_c']
        return (qL - c, qU + c)
    return (float('nan'), float('nan'))


def score(test_pred, cal_pred, dist, dist_params, cutoff, cutoff_upper, seed,
          rep=0, split_seed=0, model_name=None, ns=None, mc_seed=MC_SEED,
          levels=None, point=True, utopia_split=True, family=None,
          utopia_modes=(('base', 'utopia'), ('het', 'utopia_het'))):
    """Single source of truth: fitted model's test+calibration predictions -> interval-metric
    rows (three methods x levels x groups) + calibration rows. Used by the drivers AND by
    recompute_conformal.py, so any interval-definition change recomputes offline.

    `dist` is the fitted error law used for the likelihood/parametric intervals; `family`
    (defaults to `dist`) is the label written to the `error_family` column and lets several
    models that share an error law (e.g. RNN / AR(1) / MLP all Gaussian) be kept distinct.
    `utopia_modes` lists the (utopia_intervals mode, interval_method label) pairs to compute."""
    levels = levels or LEVELS
    ns = ns or {}
    fam = family or dist
    pmg = (point_metric_map(compute_point_metrics(
        test_pred, dist=dist, dist_params=dist_params, cutoff=cutoff, cutoff_upper=cutoff_upper))
        if point else {})
    test_iv = subsample_ge4_rows(test_pred, seed=seed + 10_000_000)
    gres = group_residuals(cal_pred, subsample_seed=seed)

    rows, calib_rows = [], []
    for level in levels:
        alpha = round(1 - level, 4)
        qL, qU, *_ = parametric_quantiles(dist, dist_params, alpha, cutoff, cutoff_upper, mc_seed=mc_seed)
        calib = calibrate(gres, alpha, qL, qU)
        for g in GROUPS:
            calib_rows.append({'repeat_id': rep, 'split_seed': split_seed, 'error_family': fam,
                               'nominal_level': level, 'visit_group': g, 'n_cal_at_group': calib[g]['m'],
                               'qL': qL, 'qU': qU, 'emp_qL': calib[g]['emp_qL'],
                               'emp_qU': calib[g]['emp_qU'], 'cparam_c': calib[g]['cparam_c']})
        for method in ('parametric', 'empirical', 'cparam'):
            L, U = build_intervals(test_iv, method, qL, qU, calib)
            im = interval_metrics(test_iv, L, U, alpha)
            for g, mv in im.items():
                off_lo, off_hi = band_offsets(method, g, qL, qU, calib.get(g, {}))
                pmr = pmg.get(g, {})
                rows.append({'repeat_id': rep, 'split_seed': split_seed, 'model_name': model_name,
                             'error_family': fam, 'interval_method': method, 'nominal_level': level,
                             'visit_group': g, **ns,
                             'n_cal_at_group': (calib[g]['m'] if g in GROUPS else np.nan),
                             'n_test_observations': mv['n'], 'parametric_q_lower': qL,
                             'parametric_q_upper': qU, 'band_q_lower': off_lo, 'band_q_upper': off_hi,
                             'coverage': mv['coverage'], 'average_width': mv['avg_width'],
                             'interval_score': mv['interval_score'],
                             'rmse': pmr.get('rmse', np.nan), 'mae': pmr.get('mae', np.nan),
                             'medae': pmr.get('medae', np.nan), 'kl': pmr.get('kl', np.nan),
                             'jsd': pmr.get('jsd', np.nan), 'kl_resid': pmr.get('kl_resid', np.nan),
                             'jsd_resid': pmr.get('jsd_resid', np.nan),
                             'mean_loglik': pmr.get('mean_loglik', np.nan)})

        # UTOPIA optimal aggregation of {parametric, empirical, cparam} (two variants)
        iv_g = test_iv['visit_index'].map(visit_group).to_numpy()
        iv_yhat = test_iv['y_pred'].to_numpy()
        for umode, uname in utopia_modes:
            L, U = utopia_intervals(cal_pred, test_iv, dist, dist_params, cutoff, cutoff_upper,
                                    alpha, split_seed=seed, mode=umode, mc_seed=mc_seed,
                                    split=utopia_split, parametric_q=(qL, qU))
            im = interval_metrics(test_iv, L, U, alpha)
            for g, mv in im.items():
                if g == 'overall':
                    off_lo = off_hi = np.nan
                else:
                    mk = iv_g == g
                    off_lo = float(np.mean(L[mk] - iv_yhat[mk]))
                    off_hi = float(np.mean(U[mk] - iv_yhat[mk]))
                pmr = pmg.get(g, {})
                rows.append({'repeat_id': rep, 'split_seed': split_seed, 'model_name': model_name,
                             'error_family': fam, 'interval_method': uname, 'nominal_level': level,
                             'visit_group': g, **ns,
                             'n_cal_at_group': (calib[g]['m'] if g in GROUPS else np.nan),
                             'n_test_observations': mv['n'], 'parametric_q_lower': qL,
                             'parametric_q_upper': qU, 'band_q_lower': off_lo, 'band_q_upper': off_hi,
                             'coverage': mv['coverage'], 'average_width': mv['avg_width'],
                             'interval_score': mv['interval_score'],
                             'rmse': pmr.get('rmse', np.nan), 'mae': pmr.get('mae', np.nan),
                             'medae': pmr.get('medae', np.nan), 'kl': pmr.get('kl', np.nan),
                             'jsd': pmr.get('jsd', np.nan), 'kl_resid': pmr.get('kl_resid', np.nan),
                             'jsd_resid': pmr.get('jsd_resid', np.nan),
                             'mean_loglik': pmr.get('mean_loglik', np.nan)})
    return rows, calib_rows


# ---------------------------------------------------------------- artifact persistence
def _save_df(df, base):
    try:
        df.to_parquet(base + '.parquet', index=False)
    except Exception:
        df.to_csv(base + '.csv', index=False)


def _load_df(base):
    return (pd.read_parquet(base + '.parquet') if os.path.exists(base + '.parquet')
            else pd.read_csv(base + '.csv'))


def save_artifacts(out_dir, rep, dist, test_pred, cal_pred, dist_params,
                   cutoff, cutoff_upper, split_seed, fstar=None, model_name=None, ns=None,
                   family=None):
    """Persist the minimal per-(rep, family) inputs so intervals recompute without retraining.

    `family` (defaults to `dist`) is the on-disk tag AND the error_family label; it keeps
    models that share an error law (RNN / AR(1) / MLP, all Gaussian) in distinct files."""
    fam = family or dist
    adir = os.path.join(out_dir, 'artifacts'); os.makedirs(adir, exist_ok=True)
    tag = f"rep{rep:03d}_{fam}"
    cols = ['pid', 'visit_index', 'y_true', 'y_pred']
    if 'hidden_norm' in test_pred.columns:                     # UTOPIA-het covariate
        cols = cols + ['hidden_norm']
    _save_df(test_pred[cols], os.path.join(adir, f"{tag}_test"))
    _save_df(cal_pred[cols], os.path.join(adir, f"{tag}_cal"))
    with open(os.path.join(adir, f"{tag}_params.json"), 'w') as f:
        json.dump({'dist': dist, 'family': fam, 'model_name': model_name or f'RNN-{dist}',
                   'dist_params': {k: float(v) for k, v in dist_params.items()},
                   'cutoff': float(cutoff), 'cutoff_upper': float(cutoff_upper),
                   'split_seed': int(split_seed), 'rep': int(rep),
                   'ns': {k: int(v) for k, v in (ns or {}).items()}}, f)
    if fstar is not None:
        _save_df(fstar, os.path.join(adir, f"rep{rep:03d}_fstar"))


def load_artifact(out_dir, rep, family):
    """Return (test_pred, cal_pred, meta) for one (rep, family). `family` is the on-disk
    tag written by save_artifacts (the error law for RNN models, e.g. 'gaussian', or a
    model-specific tag such as 'ar1_gaussian' / 'mlp_gaussian' for the baselines)."""
    adir = os.path.join(out_dir, 'artifacts'); tag = f"rep{rep:03d}_{family}"
    test = _load_df(os.path.join(adir, f"{tag}_test"))
    cal = _load_df(os.path.join(adir, f"{tag}_cal"))
    with open(os.path.join(adir, f"{tag}_params.json")) as f:
        meta = json.load(f)
    return test, cal, meta
