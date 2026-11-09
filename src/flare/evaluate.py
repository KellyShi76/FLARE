"""Prediction, metrics (incl. KL divergence), and prediction intervals.

Plotting lives in `flare.plotting` (imported lazily) so `import flare` never pulls
matplotlib.
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import norm as sp_norm, t as sp_t
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import os

from .distributions import get_distribution, has_distribution


# ============================================================
# Predict
# ============================================================

@torch.no_grad()
def predict_subjects(rnn, subjects, seq_dim, device='cpu'):
    rnn.eval()
    rows = []
    for subj in subjects:
        T = subj['T']
        curr = torch.tensor(subj['current_dyn'], dtype=torch.float32, device=device)
        stat = torch.tensor(subj['static'], dtype=torch.float32, device=device)
        padded = torch.zeros(T, max(T - 1, 1), seq_dim, device=device)
        lens = torch.zeros(T, dtype=torch.long, device=device)
        for t in range(T):
            h = subj['histories'][t]
            hl = h.shape[0] if hasattr(h, 'shape') else len(h)
            if hl > 0:
                padded[t, :hl, :] = torch.tensor(h, dtype=torch.float32, device=device)
            lens[t] = hl
        mu_t, h_t = rnn(padded, curr, stat.unsqueeze(0).expand(T, -1), lens, return_hidden=True)
        mu = mu_t.squeeze(-1).cpu().numpy()
        hnorm = torch.linalg.norm(h_t, dim=1).cpu().numpy()   # ||recurrent state|| per visit (0 at visit 1)
        for t in range(T):
            rows.append({'pid': subj['pid'], 'visit_index': t + 1,
                         'y_true': float(subj['y'][t]), 'y_pred': float(mu[t]),
                         'hidden_norm': float(hnorm[t])})
    return pd.DataFrame(rows)


# ============================================================
# Metrics
# ============================================================

HORIZONS = [
    ("Visit 1", lambda d: d[d['visit_index'] == 1]),
    ("Visit 2", lambda d: d[d['visit_index'] == 2]),
    ("Visit 3", lambda d: d[d['visit_index'] == 3]),
    ("Visit >=4", lambda d: d[d['visit_index'] >= 4]),
    ("All", lambda d: d),
]


def kl_divergence_hist(y_true, y_pred, bins=40, eps=1e-8):
    """KL(p_true || p_pred) of MARGINAL distributions via shared histogram bins.

    Both true and predicted samples are binned on a common grid spanning their
    joint range, then KL is computed in nats with `eps` smoothing for empty bins.
    Returns NaN if there are <10 points or the supports collapse.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_true = y_true[np.isfinite(y_true)]
    y_pred = y_pred[np.isfinite(y_pred)]
    if len(y_true) < 10 or len(y_pred) < 10:
        return float('nan')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    if hi - lo < 1e-12:
        return float('nan')
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(y_true, bins=edges)
    q, _ = np.histogram(y_pred, bins=edges)
    p = p.astype(float) + eps
    q = q.astype(float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence_hist(y_true, y_pred, bins=40, eps=1e-8):
    """Jensen-Shannon divergence between the MARGINAL distributions of y_true
    and y_pred, on the same shared histogram grid used by `kl_divergence_hist`.

    JSD(p || q) = ½·KL(p || m) + ½·KL(q || m),  m = ½(p + q).

    Symmetric and bounded in [0, ln 2] nats (0 = identical, ln 2 ≈ 0.693 =
    disjoint supports). Returns NaN if there are <10 points or the support
    collapses. Reported in nats, consistent with `kl_divergence_hist`.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_true = y_true[np.isfinite(y_true)]
    y_pred = y_pred[np.isfinite(y_pred)]
    if len(y_true) < 10 or len(y_pred) < 10:
        return float('nan')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    if hi - lo < 1e-12:
        return float('nan')
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(y_true, bins=edges)
    q, _ = np.histogram(y_pred, bins=edges)
    p = p.astype(float) + eps
    q = q.astype(float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def residual_law_divergences(resid, dist, dist_params, cutoff=-9.0,
                             cutoff_upper=9.0, width=0.01, lo=-9.0,
                             hi=9.0, sub=20):
    """KL and JSD between empirical residuals and the fitted innovation law."""
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return float('nan'), float('nan')
    edges = np.linspace(lo, hi, int(round((hi - lo) / width)) + 1)
    p, _ = np.histogram(np.clip(r, lo, hi), bins=edges)
    p = p / p.sum()
    step = (edges[1] - edges[0]) / sub
    x = edges[0] + step * (np.arange((len(edges) - 1) * sub) + 0.5)
    density = np.asarray(
        fitted_error_pdf(x, dist, dist_params, cutoff, cutoff_upper)[0], float)
    q = density.reshape(len(edges) - 1, sub).sum(axis=1) * step
    if not np.isfinite(q).all() or q.sum() <= 0:
        return float('nan'), float('nan')
    q = q / q.sum()
    seen = p > 0
    kl = float(np.sum(p[seen] * np.log(p[seen] / np.maximum(q[seen], 1e-300))))
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p[seen] * np.log(p[seen] / m[seen]))
    js += 0.5 * np.sum(q[q > 0] * np.log(q[q > 0] / m[q > 0]))
    return kl, float(js)


def compute_point_metrics(pred_df, kl_bins=40, dist=None, dist_params=None,
                          cutoff=-2.5, cutoff_upper=float("inf")):
    """RMSE, MAE, MedAE, marginal KL & JSD by horizon, plus (optional) test
    log-likelihood of the residuals under the fitted error distribution.

    When `dist` and `dist_params` are supplied, each horizon row also carries:
      - loglik      : Σ_i log f(ε_i)  over residuals inside the fitted support
      - mean_loglik : per-observation log-likelihood (loglik / n_in_supp)
      - n_in_supp   : # residuals with positive fitted density

    `mean_loglik` is the model-selection number (higher = better). It is computed
    on the support of the *fitted* density: for the truncated distributions this
    is [cutoff, cutoff_upper]; for the untruncated gaussian it is all of ℝ, so
    its value is NOT directly comparable to the truncated models (different
    probability space) and is reported for reference only.
    """
    results = []
    for hlabel, hfilt in HORIZONS:
        sub = hfilt(pred_df)
        yt, yp = sub['y_true'].to_numpy(), sub['y_pred'].to_numpy()
        if len(yt) == 0:
            continue
        row = {
            'horizon': hlabel, 'n': len(yt),
            'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
            'mae': float(mean_absolute_error(yt, yp)),
            'medae': float(median_absolute_error(yt, yp)),
            'kl': kl_divergence_hist(yt, yp, bins=kl_bins),
            'jsd': js_divergence_hist(yt, yp, bins=kl_bins),
        }
        if dist is not None and dist_params is not None:
            resid = yt - yp
            row['kl_resid'], row['jsd_resid'] = residual_law_divergences(
                resid, dist, dist_params, cutoff, cutoff_upper)
            pdf, _ = fitted_error_pdf(resid, dist, dist_params, cutoff, cutoff_upper)
            if pdf is None:
                row.update({'loglik': float('nan'), 'mean_loglik': float('nan'),
                            'n_in_supp': 0})
            else:
                # Support = points with positive fitted density. For truncated
                # dists this is exactly [cutoff, cutoff_upper] (pdf is masked to
                # 0 outside); for the untruncated gaussian it is every point.
                pdf = np.asarray(pdf, dtype=float)
                valid = pdf > 0
                if valid.any():
                    log_p = np.log(np.clip(pdf[valid], 1e-300, None))
                    row.update({
                        'loglik':      float(log_p.sum()),
                        'mean_loglik': float(log_p.mean()),
                        'n_in_supp':   int(valid.sum()),
                    })
                else:
                    row.update({'loglik': float('nan'),
                                'mean_loglik': float('nan'),
                                'n_in_supp': 0})
        results.append(row)
    return results


# ============================================================
# MC sampling — used for parametric PI and QQ plots
# ============================================================

def sample_error(dist, dist_params, n=500_000, seed=42, cutoff=-9.0, cutoff_upper=9.0):
    """Sample residuals from the fitted error distribution. For the
    mix-of-truncated 2-component dists, the mixture weight π is the *truncated*
    weight, so the sampler picks component k with weight π_k and then draws a
    single value from that component (rejecting any draw outside the support)."""
    rng = np.random.RandomState(seed)
    dp = dist_params

    if dist == 'gaussian':
        # Untruncated by design (the gaussian baseline lives on ℝ).
        return rng.normal(0.0, dp['sigma'], n)

    if dist == 't':
        out = []
        while len(out) < n:
            eps = dp['sigma'] * rng.standard_t(dp['nu'], n * 2)
            out.extend(eps[(eps >= cutoff) & (eps <= cutoff_upper)].tolist())
        return np.array(out[:n])

    if dist == 'tukeyh':
        out = []
        while len(out) < n:
            z = rng.randn(n * 2)
            eps = dp['sigma'] * z * np.exp(dp['h'] * z**2 / 2)
            out.extend(eps[(eps >= cutoff) & (eps <= cutoff_upper)].tolist())
        return np.array(out[:n])

    # --- 2-component mix-of-truncated samplers -------------------------------
    # f(ε) = π · φ_1(ε)/Z_1 + (1−π) · φ_2(ε)/Z_2  on [a, b].
    # Sample component k with weight π_k, then draw a single value from comp k
    # restricted to [a, b] (rejection — efficient when cutoffs are wide).
    if dist in ('truncmix', 'truncmix_gauss'):
        out = []
        while len(out) < n:
            B = n * 2
            comp = rng.rand(B) < dp['pi']
            eps = np.where(comp,
                           rng.normal(dp['mu1'], dp['sigma1'], B),
                           rng.normal(dp['mu2'], dp['sigma2'], B))
            out.extend(eps[(eps >= cutoff) & (eps <= cutoff_upper)].tolist())
        return np.array(out[:n])

    if dist == 'truncmix_t':
        out = []
        while len(out) < n:
            B = n * 2
            comp = rng.rand(B) < dp['pi']
            eps = np.where(comp,
                           dp['mu1'] + dp['sigma1'] * rng.standard_t(dp['nu1'], B),
                           dp['mu2'] + dp['sigma2'] * rng.standard_t(dp['nu2'], B))
            out.extend(eps[(eps >= cutoff) & (eps <= cutoff_upper)].tolist())
        return np.array(out[:n])

    if dist == 'truncmix_tukeyh':
        out = []
        while len(out) < n:
            B = n * 2
            comp = rng.rand(B) < dp['pi']
            z1 = rng.randn(B); z2 = rng.randn(B)
            eps1 = dp['mu1'] + dp['sigma1'] * z1 * np.exp(dp['h1'] * z1**2 / 2)
            eps2 = dp['mu2'] + dp['sigma2'] * z2 * np.exp(dp['h2'] * z2**2 / 2)
            eps = np.where(comp, eps1, eps2)
            out.extend(eps[(eps >= cutoff) & (eps <= cutoff_upper)].tolist())
        return np.array(out[:n])

    if has_distribution(dist):
        sample = get_distribution(dist).sample(
            params=dict(dp), n=int(n), rng=rng,
            cutoff=float(cutoff), cutoff_upper=float(cutoff_upper))
        sample = np.asarray(sample, dtype=float).reshape(-1)
        if sample.size != int(n) or not np.isfinite(sample).all():
            raise ValueError(
                f"custom sampler for {dist!r} must return exactly {n} finite values"
            )
        return sample

    raise ValueError(f"Unknown dist: {dist}")


# ============================================================
# Prediction Intervals
# ============================================================

def compute_parametric_pi(pred_df, dist, dist_params, model_name,
                          test_subjects=None, seq_dim=None, cutoff=-2.5,
                          cutoff_upper=float("inf"), levels=[0.90, 0.95]):
    """Parametric PI from the fitted error distribution.

    Non-DEC: constant offset from MC error samples.
    DEC: conditional Gaussian PI that narrows with visit history.
    """
    rows = []
    dp = dist_params

    if 'DEC' not in model_name:
        mc = sample_error(dist, dp, cutoff=cutoff, cutoff_upper=cutoff_upper)
        for level in levels:
            lo = np.quantile(mc, (1 - level) / 2)
            hi = np.quantile(mc, 1 - (1 - level) / 2)
            for _, r in pred_df.iterrows():
                rows.append({**r.to_dict(),
                             'pi_lo': r['y_pred'] + lo,
                             'pi_hi': r['y_pred'] + hi,
                             'pi_width': hi - lo,
                             'level': level, 'model': model_name})
    else:
        # DEC branch. Two pieces:
        #   (a) marginal residual SHAPE: quantiles of mc samples from the fitted
        #       (mixture or Gaussian) error model — no standardize-and-rescale.
        #   (b) per-visit refinement from DEC: a conditional mean shift from past
        #       residuals (ct·Cp⁻¹·rp) and a std shrink factor √(1−ct·Cp⁻¹·ct).
        #
        # At t=0 (no history) the PI reduces to ŷ ± mc_quantile, matching the
        # non-DEC behaviour. The previous code standardised mc by √σ²_joint and
        # rescaled by √(σ²_joint·shrink), which algebraically equals mc_quantile·
        # √shrink — fine when correct, but coupled with the mixture-key collision
        # in sample_error it produced widths ~50× too small.
        phi = dp['phi']
        gam = dp['gamma']
        mc = sample_error(dist, dp, cutoff=cutoff, cutoff_upper=cutoff_upper)
        q_dict = {l: (float(np.quantile(mc, (1 - l) / 2)),
                       float(np.quantile(mc, 1 - (1 - l) / 2))) for l in levels}

        for level in levels:
            q_lo_marg, q_hi_marg = q_dict[level]
            for subj in test_subjects:
                preds = pred_df[pred_df['pid'] == subj['pid']].sort_values('visit_index')
                if len(preds) == 0:
                    continue
                mu_all = preds['y_pred'].to_numpy()
                y_all = preds['y_true'].to_numpy()
                T = subj['T']
                for t in range(T):
                    if t == 0:
                        cm = float(mu_all[t])
                        shrink = 1.0
                    else:
                        tp = subj['times'][:t]
                        tc = subj['times'][t]
                        diff = np.abs(tp[:, None] - tp[None, :])
                        Cp = phi ** (diff ** gam)
                        np.fill_diagonal(Cp, 1.0)
                        ct = phi ** (np.abs(tc - tp) ** gam)
                        rp = y_all[:t] - mu_all[:t]
                        try:
                            cm = float(mu_all[t] + ct @ np.linalg.solve(Cp, rp))
                            var_factor = max(1.0 - float(ct @ np.linalg.solve(Cp, ct)), 0.01)
                            shrink = float(np.sqrt(var_factor))
                        except Exception:
                            cm = float(mu_all[t])
                            shrink = 1.0
                    pi_lo = cm + shrink * q_lo_marg
                    pi_hi = cm + shrink * q_hi_marg
                    rows.append({'pid': subj['pid'], 'visit_index': t + 1,
                                 'y_true': float(y_all[t]), 'y_pred': cm,
                                 'pi_lo': pi_lo, 'pi_hi': pi_hi,
                                 'pi_width': pi_hi - pi_lo,
                                 'level': level, 'model': model_name})

    return pd.DataFrame(rows)


def compute_empirical_pi(pred_df, train_resid, model_name, levels=[0.90, 0.95]):
    """Empirical PI from training residuals."""
    rows = []
    for level in levels:
        lo = np.quantile(train_resid, (1 - level) / 2)
        hi = np.quantile(train_resid, 1 - (1 - level) / 2)
        for _, r in pred_df.iterrows():
            rows.append({**r.to_dict(),
                         'pi_lo': r['y_pred'] + lo,
                         'pi_hi': r['y_pred'] + hi,
                         'pi_width': hi - lo,
                         'level': level, 'model': model_name})
    return pd.DataFrame(rows)


def eval_pi(pi_df):
    """Coverage, width, Winkler by horizon × level."""
    results = []
    for hl, hv in [("Visit 1", 1), ("Visit 2", 2), ("Visit 3", 3),
                    ("Visit >=4", -1), ("All", 0)]:
        for level in pi_df['level'].unique():
            sub = pi_df[pi_df['level'] == level]
            if hl == "All":
                pass
            elif hv == -1:
                sub = sub[sub['visit_index'] >= 4]
            else:
                sub = sub[sub['visit_index'] == hv]
            if len(sub) == 0:
                continue
            cov = ((sub['y_true'] >= sub['pi_lo']) & (sub['y_true'] <= sub['pi_hi'])).mean()
            w = sub['pi_width'].mean()
            a = 1 - level
            pen = np.where(sub['y_true'] < sub['pi_lo'],
                           (2 / a) * (sub['pi_lo'] - sub['y_true']),
                  np.where(sub['y_true'] > sub['pi_hi'],
                           (2 / a) * (sub['y_true'] - sub['pi_hi']), 0.0))
            wink = (sub['pi_width'].to_numpy() + pen).mean()
            results.append({'horizon': hl, 'level': float(level), 'n': len(sub),
                            'coverage': float(cov), 'width': float(w), 'winkler': float(wink)})
    return results


# ============================================================
# Fitted PDF and diagnostics plot
# ============================================================

def fitted_error_pdf(x, dist, dist_params, cutoff=-2.5, cutoff_upper=float("inf")):
    """Compute fitted error PDF at points x.

    Returns (total, components) where components is a list of per-component
    PDFs (empty for single-component dists). For 2-comp mixtures the list has
    two entries; for 3-comp mixtures three.
    """
    dp = dist_params

    if dist == 'gaussian':
        # Left untruncated on purpose: the gaussian is trained as an untruncated
        # N(0, σ²), so its fitted density (and hence mean_loglik) reflects the
        # model as trained. Note this puts it on a different probability space
        # than the truncated models — not directly comparable.
        return sp_norm.pdf(x, 0, dp['sigma']), []

    if dist == 't':
        # Trained truncated → score truncated for a consistent likelihood.
        sigma = dp['sigma']
        pdf = sp_t.pdf(x / sigma, df=dp['nu']) / sigma
        Z = float(sp_t.cdf(cutoff_upper / sigma, df=dp['nu'])
                  - sp_t.cdf(cutoff / sigma, df=dp['nu']))
        mask = ((x >= cutoff) & (x <= cutoff_upper)).astype(float)
        return pdf / max(Z, 1e-12) * mask, []

    if dist == 'tukeyh':
        from scipy.special import lambertw
        h = dp['h']; sigma = dp['sigma']
        u = x / sigma
        if h < 1e-12:
            z = u
        else:
            arg = np.maximum(h * u**2, 0.0)
            w = np.real(lambertw(arg))
            z = u * np.exp(-w / 2)
        log_phi = -0.5 * np.log(2 * np.pi) - 0.5 * z**2
        log_jac = np.log(sigma) + h * z**2 / 2 + np.log1p(h * z**2)
        pdf = np.exp(log_phi - log_jac)
        # Truncation normalizer via the monotone Lambert-W inverse (h ≥ 0).
        def _th_surv(c):
            uu = float(c / sigma)
            if h < 1e-12:
                zz = uu
            else:
                arg2 = max(h * uu**2, 0.0)
                zz = uu * np.exp(-np.real(lambertw(arg2)) / 2)
            return float(sp_norm.sf(zz))
        Z = _th_surv(cutoff) - (_th_surv(cutoff_upper) if cutoff_upper < float('inf') else 0.0)
        mask = ((x >= cutoff) & (x <= cutoff_upper)).astype(float)
        return pdf / max(Z, 1e-12) * mask, []

    # -----------------------------------------------------------------------
    # 2-component MIX-OF-TRUNCATED: f(x) = π·φ_1(x)/Z_1 + (1−π)·φ_2(x)/Z_2  on [a, b].
    # Each component is independently truncated and re-normalized by its own Z_k.
    # This matches the paper formulation. With cutoffs at ±9 and σ's < 1, both
    # Z_k ≈ 1, but the form is correct in general and equivalent to the older
    # mix-then-truncate parameterization up to the bijection π ↔ π'.
    # -----------------------------------------------------------------------
    if dist in ('truncmix', 'truncmix_gauss'):
        s1, s2  = dp['sigma1'], dp['sigma2']
        m1, m2  = dp['mu1'],    dp['mu2']
        pi_v    = dp['pi']
        f1 = sp_norm.pdf(x, m1, s1)
        f2 = sp_norm.pdf(x, m2, s2)
        Z1 = float(sp_norm.cdf(cutoff_upper, m1, s1) - sp_norm.cdf(cutoff, m1, s1))
        Z2 = float(sp_norm.cdf(cutoff_upper, m2, s2) - sp_norm.cdf(cutoff, m2, s2))
        Z1, Z2 = max(Z1, 1e-12), max(Z2, 1e-12)
        mask = ((x >= cutoff) & (x <= cutoff_upper)).astype(float)
        comp1 = pi_v       * f1 / Z1 * mask
        comp2 = (1 - pi_v) * f2 / Z2 * mask
        return comp1 + comp2, [comp1, comp2]

    if dist == 'truncmix_t':
        s1, s2  = dp['sigma1'], dp['sigma2']
        m1, m2  = dp['mu1'],    dp['mu2']
        pi_v    = dp['pi']
        n1, n2  = dp['nu1'],    dp['nu2']
        f1 = sp_t.pdf((x - m1) / s1, df=n1) / s1
        f2 = sp_t.pdf((x - m2) / s2, df=n2) / s2
        Z1 = float(sp_t.cdf((cutoff_upper - m1) / s1, df=n1)
                   - sp_t.cdf((cutoff       - m1) / s1, df=n1))
        Z2 = float(sp_t.cdf((cutoff_upper - m2) / s2, df=n2)
                   - sp_t.cdf((cutoff       - m2) / s2, df=n2))
        Z1, Z2 = max(Z1, 1e-12), max(Z2, 1e-12)
        mask = ((x >= cutoff) & (x <= cutoff_upper)).astype(float)
        comp1 = pi_v       * f1 / Z1 * mask
        comp2 = (1 - pi_v) * f2 / Z2 * mask
        return comp1 + comp2, [comp1, comp2]

    if dist == 'truncmix_tukeyh':
        from scipy.special import lambertw
        s1, s2  = dp['sigma1'], dp['sigma2']
        m1, m2  = dp['mu1'],    dp['mu2']
        pi_v    = dp['pi']
        h1, h2  = dp['h1'],     dp['h2']
        def th_pdf(y, xi, om, h):
            u = (y - xi) / om
            if h < 1e-12:
                z = u
            else:
                arg = np.maximum(h * u**2, 0.0)
                z = u * np.exp(-np.real(lambertw(arg)) / 2)
            log_phi = -0.5 * np.log(2*np.pi) - 0.5 * z**2
            log_jac = np.log(om) + h * z**2 / 2 + np.log1p(h * z**2)
            return np.exp(log_phi - log_jac)
        def th_surv(c, xi, om, h):
            u = float((c - xi) / om)
            if h < 1e-12:
                z = u
            else:
                arg = max(h * u**2, 0.0)
                z = u * np.exp(-np.real(lambertw(arg)) / 2)
            return float(sp_norm.sf(z))
        f1 = th_pdf(x, m1, s1, h1)
        f2 = th_pdf(x, m2, s2, h2)
        Z1 = th_surv(cutoff, m1, s1, h1) - th_surv(cutoff_upper, m1, s1, h1)
        Z2 = th_surv(cutoff, m2, s2, h2) - th_surv(cutoff_upper, m2, s2, h2)
        Z1, Z2 = max(Z1, 1e-12), max(Z2, 1e-12)
        mask = ((x >= cutoff) & (x <= cutoff_upper)).astype(float)
        comp1 = pi_v       * f1 / Z1 * mask
        comp2 = (1 - pi_v) * f2 / Z2 * mask
        return comp1 + comp2, [comp1, comp2]

    if has_distribution(dist):
        ans = get_distribution(dist).pdf(
            x=np.asarray(x, dtype=float), params=dict(dp),
            cutoff=float(cutoff), cutoff_upper=float(cutoff_upper))
        total, components = ans if isinstance(ans, tuple) else (ans, [])
        total = np.asarray(total, dtype=float)
        if total.shape != np.asarray(x).shape:
            raise ValueError(
                f"custom PDF for {dist!r} returned shape {total.shape}; "
                f"expected {np.asarray(x).shape}"
            )
        if np.any(total < 0) or not np.isfinite(total).all():
            raise ValueError(f"custom PDF for {dist!r} must be finite and nonnegative")
        return total, [np.asarray(c, dtype=float) for c in components]

    return None, []


def print_results(model_name, point_metrics, pi_results):
    """Pretty-print metrics + PI for one model."""
    print(f"\n  {model_name}")
    has_ll = bool(point_metrics) and ('mean_loglik' in point_metrics[0])
    if has_ll:
        print(f"  {'Horizon':<12} {'N':>6} {'RMSE':>8} {'MAE':>8} {'MedAE':>8} "
              f"{'KL':>8} {'JSD':>8} {'nIn':>6} {'meanLL':>9} {'LL':>11}")
        print("  " + "-" * 90)
    else:
        print(f"  {'Horizon':<12} {'N':>6} {'RMSE':>8} {'MAE':>8} {'MedAE':>8} "
              f"{'KL':>8} {'JSD':>8}")
        print("  " + "-" * 70)
    for r in point_metrics:
        kl_str  = f"{r['kl']:>8.4f}"  if not np.isnan(r['kl'])  else f"{'nan':>8}"
        jsd_str = f"{r.get('jsd', float('nan')):>8.4f}" if not np.isnan(r.get('jsd', float('nan'))) else f"{'nan':>8}"
        base = (f"  {r['horizon']:<12} {r['n']:>6} {r['rmse']:>8.4f} {r['mae']:>8.4f} "
                f"{r['medae']:>8.4f} {kl_str} {jsd_str}")
        if has_ll:
            mll = r.get('mean_loglik', float('nan'))
            ll  = r.get('loglik', float('nan'))
            nin = r.get('n_in_supp', r['n'])
            mll_str = f"{mll:>9.4f}"  if not np.isnan(mll) else f"{'nan':>9}"
            ll_str  = f"{ll:>11.2f}"  if not np.isnan(ll)  else f"{'nan':>11}"
            print(base + f" {nin:>6d} {mll_str} {ll_str}")
        else:
            print(base)

    if pi_results:
        print(f"\n  {'Horizon':<12} {'Level':>6} {'N':>7} {'Cover':>8} {'Width':>8} {'Winkler':>9}")
        print("  " + "-" * 56)
        for r in pi_results:
            print(f"  {r['horizon']:<12} {r['level']:>5.0%} {r['n']:>7} "
                  f"{r['coverage']:>8.3f} {r['width']:>8.4f} {r['winkler']:>9.4f}")
