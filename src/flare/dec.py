"""
DEC (Decaying Exponential Correlation) for temporal error correlation.

C_i[k,k'] = φ^{|t_k - t_k'|^γ}

Stage-2 fitting: freeze RNN, learn (φ, γ, σ²) jointly with distribution params
using the joint Gaussian LL plus a per-observation correction for non-Gaussian
mixtures. The non-Gaussian correction is an approximation — it combines the
Gaussian joint LL on residuals with a per-obs ratio (log f_mix - log f_gauss).
"""

import numpy as np
import torch
import torch.nn as nn

from .losses import (lambertw_torch, tau_h_inv_torch,
                      log_pdf_gauss, log_pdf_t, log_pdf_tukeyh, log_pdf_tukeyhh,
                      log_surv_gauss, log_surv_t, log_surv_tukeyh, log_surv_tukeyhh)


def build_dec_matrix(times, phi, gamma):
    """C[k,k'] = φ^{|t_k-t_k'|^γ}, diag = 1."""
    diff = torch.abs(times.unsqueeze(1) - times.unsqueeze(0))
    C = torch.exp(torch.log(phi + 1e-10) * (diff + 1e-10).pow(gamma))
    T = times.shape[0]
    eye = torch.eye(T, dtype=C.dtype, device=C.device)
    return C * (1 - eye) + eye


def gaussian_joint_ll(y, mu, sigma2, C):
    T = y.shape[0]
    R = sigma2 * C
    L = torch.linalg.cholesky(R)
    r = y - mu
    log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))
    solve = torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
    return -0.5 * (T * np.log(2 * np.pi) + log_det + r @ solve)


def _mixture_correction(y, mu1, dist, params_dict, joint_sigma, cutoff,
                        cutoff_upper=float('inf')):
    """Per-observation correction: log(mixture_pdf/Z) − log N(y; μ, joint_sigma²).

    The subtracted Gaussian baseline MUST use the SAME variance σ²_joint that
    multiplies C in gaussian_joint_ll above, not the mixture's σ_comp1. Then:

        LL_total = LL_joint_gauss(σ²_joint·C) + Σ_t [log_mix - log_gauss(σ²_joint)]
                 = (Σ_t log_gauss(σ²_joint) + corr_part(C, σ²_joint))
                   + Σ_t [log_mix - log_gauss(σ²_joint)]
                 = corr_part(C, σ²_joint) + Σ_t log_mix(y_t)

    The per-obs Gaussian baselines cancel exactly, leaving the within-subject
    correlation contribution plus the marginal mixture log-likelihood. With the
    buggy version (subtracting log_gauss(σ_comp1) instead) the cancellation
    fails, the gradient drags σ_comp1 to fit the residual bulk by itself, and
    the fitted mixture collapses to π≈1, σ_comp1≈0.

    `cutoff_upper` adds a hard upper bound: Z becomes P(L ≤ ε ≤ U) per
    component, and observations with ε > U get the same -100 penalty as ε < L.
    """
    from .losses import _log_diff_surv

    c_lo = torch.tensor(cutoff, dtype=y.dtype, device=y.device)
    zero = torch.tensor(0.0, dtype=y.dtype, device=y.device)
    has_upper = cutoff_upper < float('inf')
    c_hi = (torch.tensor(cutoff_upper, dtype=y.dtype, device=y.device)
            if has_upper else None)

    # Each branch must produce log_f (per-obs log mixture pdf) and log_Z
    # (scalar log truncation normalizer). Comp 1 always anchors at mu1; comps
    # 2 & 3 carry their own offsets.

    if dist == 't':
        sigma = torch.exp(0.5 * params_dict['log_sigma2'])
        nu = 2.0 + nn.functional.softplus(params_dict['unc_nu'])
        log_f = log_pdf_t(y, mu1, sigma, nu)
        log_Z_lo = log_surv_t(c_lo, zero, sigma, nu)
        log_Z = (_log_diff_surv(log_Z_lo, log_surv_t(c_hi, zero, sigma, nu))
                 if has_upper else log_Z_lo)

    elif dist == 'tukeyh':
        sigma = torch.exp(0.5 * params_dict['log_sigma2'])
        h = nn.functional.softplus(params_dict['raw_h'])
        log_f = log_pdf_tukeyh(y, mu1, sigma, h)
        log_Z_lo = log_surv_tukeyh(c_lo, zero, sigma, h)
        log_Z = (_log_diff_surv(log_Z_lo, log_surv_tukeyh(c_hi, zero, sigma, h))
                 if has_upper else log_Z_lo)

    elif dist in ('truncmix_gauss', 'truncmix_t', 'truncmix_tukeyh'):
        # Mix-of-truncated 2-component with mean-zero constraint:
        #   π·μ_1 + (1−π)·μ_2 = 0  ⇒  μ_2 = − exp(logit_pi) · μ_1
        # In DEC the joint Gaussian baseline still uses mu_rnn (the per-obs
        # prediction); the mixture density lives on residuals with components
        # at mu_rnn + m_k. The constraint anchors the mixture's mean at 0,
        # preventing μ_1/μ_2 from silently absorbing a constant bias in mu_rnn.
        s1 = torch.exp(params_dict['log_sigma1'])
        s2 = torch.exp(params_dict['log_sigma2'])  # via _mixture_view: comp-2 std-dev
        logit_pi = params_dict['logit_pi']
        pi = torch.sigmoid(logit_pi)
        m1 = params_dict['mu1']
        m2 = -torch.exp(logit_pi) * m1

        if dist == 'truncmix_gauss':
            log_p1 = log_pdf_gauss(y, mu1 + m1, s1)
            log_p2 = log_pdf_gauss(y, mu1 + m2, s2)
            log_Z1_lo = log_surv_gauss(c_lo, m1, s1)
            log_Z2_lo = log_surv_gauss(c_lo, m2, s2)
            if has_upper:
                log_Z1 = _log_diff_surv(log_Z1_lo, log_surv_gauss(c_hi, m1, s1))
                log_Z2 = _log_diff_surv(log_Z2_lo, log_surv_gauss(c_hi, m2, s2))
            else:
                log_Z1, log_Z2 = log_Z1_lo, log_Z2_lo
        elif dist == 'truncmix_t':
            nu1 = 2.0 + nn.functional.softplus(params_dict['unc_nu1'])
            nu2 = 2.0 + nn.functional.softplus(params_dict['unc_nu2'])
            log_p1 = log_pdf_t(y, mu1 + m1, s1, nu1)
            log_p2 = log_pdf_t(y, mu1 + m2, s2, nu2)
            log_Z1_lo = log_surv_t(c_lo, m1, s1, nu1)
            log_Z2_lo = log_surv_t(c_lo, m2, s2, nu2)
            if has_upper:
                log_Z1 = _log_diff_surv(log_Z1_lo, log_surv_t(c_hi, m1, s1, nu1))
                log_Z2 = _log_diff_surv(log_Z2_lo, log_surv_t(c_hi, m2, s2, nu2))
            else:
                log_Z1, log_Z2 = log_Z1_lo, log_Z2_lo
        else:  # truncmix_tukeyh
            h1 = nn.functional.softplus(params_dict['raw_h1'])
            h2 = nn.functional.softplus(params_dict['raw_h2'])
            log_p1 = log_pdf_tukeyh(y, mu1 + m1, s1, h1)
            log_p2 = log_pdf_tukeyh(y, mu1 + m2, s2, h2)
            log_Z1_lo = log_surv_tukeyh(c_lo, m1, s1, h1)
            log_Z2_lo = log_surv_tukeyh(c_lo, m2, s2, h2)
            if has_upper:
                log_Z1 = _log_diff_surv(log_Z1_lo, log_surv_tukeyh(c_hi, m1, s1, h1))
                log_Z2 = _log_diff_surv(log_Z2_lo, log_surv_tukeyh(c_hi, m2, s2, h2))
            else:
                log_Z1, log_Z2 = log_Z1_lo, log_Z2_lo

        # Mix-of-truncated: each comp independently truncated and re-normalized;
        # the overall truncation normalizer is 1, so log_Z = 0.
        log_p1c = log_p1 - log_Z1
        log_p2c = log_p2 - log_Z2
        log_f = torch.logaddexp(torch.log(pi) + log_p1c,
                                torch.log1p(-pi) + log_p2c)
        log_Z = torch.zeros((), dtype=y.dtype, device=y.device)
    else:
        raise ValueError(f"Mixture correction undefined for dist={dist}")

    eps = y - mu1
    valid = ((eps >= cutoff) & (eps <= cutoff_upper)).float()
    log_lik_obs = (log_f - log_Z) * valid + (-100.0) * (1 - valid)
    log_gauss_obs = log_pdf_gauss(y, mu1, joint_sigma)
    return (log_lik_obs - log_gauss_obs).sum()


def make_dec_params(dist, dist_params_dict, device):
    """Create learnable DEC parameters, initialized from Stage-1 values."""
    dp = dist_params_dict
    # σ² init from Stage 1 (use sigma1² for mixtures, sigma² for gaussian)
    if 'sigma' in dp:
        s2_init = dp['sigma']**2
    elif 'sigma1' in dp:
        s2_init = dp['sigma1']**2
    else:
        s2_init = 0.1

    p = {
        'log_sigma2': nn.Parameter(torch.tensor(np.log(s2_init), device=device)),
        'phi':        nn.Parameter(torch.tensor(0.0, device=device)),     # σ(0)=0.5
        'gamma':      nn.Parameter(torch.tensor(np.log(np.exp(1.0) - 1), device=device)),  # softplus≈1
    }

    if dist == 'gaussian':
        return p

    # ---- Single non-Gaussian distributions (t, tukeyh) -----------------------
    # σ_t / σ_h is stored under log_sigma2_2 (which _mixture_view renames to
    # log_sigma2 for the correction body) to avoid collision with DEC's joint
    # variance under log_sigma2.
    if dist == 't':
        s_init = float(dp.get('sigma', np.sqrt(s2_init)))
        p['log_sigma2_2'] = nn.Parameter(torch.tensor(2 * np.log(s_init), device=device))
        p['unc_nu'] = nn.Parameter(torch.tensor(
            float(np.log(np.exp(max(dp['nu'] - 2.0, 0.5)) - 1)), device=device))
        return p
    if dist == 'tukeyh':
        s_init = float(dp.get('sigma', np.sqrt(s2_init)))
        p['log_sigma2_2'] = nn.Parameter(torch.tensor(2 * np.log(s_init), device=device))
        p['raw_h'] = nn.Parameter(torch.tensor(
            float(np.log(np.exp(max(dp['h'], 1e-3)) - 1)), device=device))
        return p

    # ---- 2-component mix-of-truncated mixtures with mean-zero constraint ---
    p['log_sigma1']   = nn.Parameter(torch.tensor(np.log(dp['sigma1']), device=device))
    p['log_sigma2_2'] = nn.Parameter(torch.tensor(np.log(dp['sigma2']), device=device))
    p['logit_pi'] = nn.Parameter(torch.tensor(
        float(torch.logit(torch.tensor(min(max(dp['pi'], 1e-3), 1 - 1e-3)))),
        device=device))
    p['mu1'] = nn.Parameter(torch.tensor(float(dp.get('mu1', 0.0)), device=device))

    if dist == 'truncmix_t':
        p['unc_nu1'] = nn.Parameter(torch.tensor(
            float(np.log(np.exp(max(dp['nu1'] - 2.0, 0.5)) - 1)), device=device))
        p['unc_nu2'] = nn.Parameter(torch.tensor(
            float(np.log(np.exp(max(dp['nu2'] - 2.0, 0.5)) - 1)), device=device))
    elif dist == 'truncmix_tukeyh':
        p['raw_h1'] = nn.Parameter(torch.tensor(
            float(np.log(np.exp(max(dp['h1'], 1e-3)) - 1)), device=device))
        p['raw_h2'] = nn.Parameter(torch.tensor(
            float(np.log(np.exp(max(dp['h2'], 1e-3)) - 1)), device=device))
    return p


def _mixture_view(params_dict):
    """Return a params view that names mixture σ's consistently for the correction.

    The DEC param dict stores `log_sigma2` for the joint variance and the
    mixture component std-devs under disambiguated keys to avoid the collision
    with DEC's joint variance:

      2-component mixtures: log_sigma1, log_sigma2_2 (→ log_sigma2)

    The view drops DEC's `log_sigma2` (joint variance, passed separately as
    joint_sigma) and renames the disambiguated keys back to the natural names
    that the mixture loss bodies expect.
    """
    view = {k: v for k, v in params_dict.items() if k != 'log_sigma2'}
    if 'log_sigma2_2' in view:
        view['log_sigma2'] = view['log_sigma2_2']
        del view['log_sigma2_2']
    return view


def compute_subject_ll(rnn, subj, params_dict, dist, seq_dim, cutoff, device,
                       cutoff_upper=float('inf')):
    """Joint LL for one subject under DEC.

    If `_dec_mu / _dec_y / _dec_times` were precomputed on `subj` (by
    train_dec before its epoch loop), use them and skip the per-call RNN
    forward pass. Falls back to the on-the-fly forward for callers that
    haven't populated the cache.
    """
    if '_dec_mu' in subj:
        mu    = subj['_dec_mu']
        y     = subj['_dec_y']
        times = subj['_dec_times']
    else:
        T = subj['T']
        curr = torch.tensor(subj['current_dyn'], dtype=torch.float32, device=device)
        stat = torch.tensor(subj['static'],      dtype=torch.float32, device=device)
        y     = torch.tensor(subj['y'],     dtype=torch.float32, device=device)
        times = torch.tensor(subj['times'], dtype=torch.float32, device=device)
        padded = torch.zeros(T, max(T - 1, 1), seq_dim, device=device)
        lens   = torch.zeros(T, dtype=torch.long, device=device)
        for t in range(T):
            h = subj['histories'][t]
            hl = h.shape[0] if hasattr(h, 'shape') else len(h)
            if hl > 0:
                padded[t, :hl, :] = torch.tensor(h, dtype=torch.float32, device=device)
            lens[t] = hl
        mu = rnn(padded, curr, stat.unsqueeze(0).expand(T, -1), lens).squeeze(-1)

    phi    = torch.sigmoid(params_dict['phi'])
    gamma  = nn.functional.softplus(params_dict['gamma'])
    sigma2 = torch.exp(params_dict['log_sigma2'])
    C      = build_dec_matrix(times, phi, gamma)

    base_ll = gaussian_joint_ll(y, mu, sigma2, C)
    if dist == 'gaussian':
        return base_ll
    joint_sigma = torch.sqrt(sigma2)
    return base_ll + _mixture_correction(y, mu, dist, _mixture_view(params_dict),
                                          joint_sigma, cutoff, cutoff_upper)


def train_dec(rnn, subjects_train, subjects_val, optimizer, dist, params_dict,
              seq_dim, cutoff=-9.0, cutoff_upper=9.0, device='cpu',
              num_epochs=300, patience=30, batch_accum=64, verbose=True):
    """Stage-2: train DEC params with frozen RNN, maximizing joint LL."""
    best_ll, best_params, pat = -np.inf, None, 0
    rnn.eval()                                                 # ← once, at the top

    # Precompute mu / y / times for each subject (RNN is frozen, mu is constant)
    cached_subjs = []                                          # ← capture for cleanup
    with torch.no_grad():
        for subj in subjects_train + subjects_val:
            if subj['T'] < 2:
                continue
            T = subj['T']
            padded = torch.zeros(T, max(T - 1, 1), seq_dim, device=device)
            lens   = torch.zeros(T, dtype=torch.long, device=device)
            for t in range(T):
                h = subj['histories'][t]
                hl = h.shape[0] if hasattr(h, 'shape') else len(h)
                if hl > 0:
                    padded[t, :hl, :] = torch.tensor(h, dtype=torch.float32, device=device)
                lens[t] = hl
            curr = torch.tensor(subj['current_dyn'], dtype=torch.float32, device=device)
            stat = torch.tensor(subj['static'],      dtype=torch.float32, device=device)
            subj['_dec_mu']    = rnn(padded, curr, stat.unsqueeze(0).expand(T, -1), lens).squeeze(-1).detach()
            subj['_dec_y']     = torch.tensor(subj['y'],     dtype=torch.float32, device=device)
            subj['_dec_times'] = torch.tensor(subj['times'], dtype=torch.float32, device=device)
            cached_subjs.append(subj)

    try:
        for ep in range(num_epochs):
            accum_ll = torch.tensor(0.0, device=device)
            accum_n = 0
            for si in np.random.permutation(len(subjects_train)):
                subj = subjects_train[si]
                if subj['T'] < 2:
                    continue
                ll = compute_subject_ll(rnn, subj, params_dict, dist, seq_dim,
                                        cutoff, device, cutoff_upper=cutoff_upper)
                accum_ll = accum_ll + ll
                accum_n += 1
                if accum_n >= batch_accum:
                    (-accum_ll / accum_n).backward()
                    nn.utils.clip_grad_norm_(list(params_dict.values()), 5.0)
                    optimizer.step(); optimizer.zero_grad()
                    accum_ll = torch.tensor(0.0, device=device); accum_n = 0
            if accum_n > 0:
                (-accum_ll / accum_n).backward()
                nn.utils.clip_grad_norm_(list(params_dict.values()), 5.0)
                optimizer.step(); optimizer.zero_grad()

            val_ll = 0.0
            with torch.no_grad():
                for subj in subjects_val:
                    if subj['T'] < 2:
                        continue
                    val_ll += compute_subject_ll(rnn, subj, params_dict, dist,
                                                  seq_dim, cutoff, device,
                                                  cutoff_upper=cutoff_upper).item()
            if val_ll > best_ll:
                best_ll = val_ll
                best_params = {k: v.data.clone() for k, v in params_dict.items()}
                pat = 0
            else:
                pat += 1
                if pat >= patience:
                    break
            if verbose and (ep + 1) % 5 == 0:
                phi_v = torch.sigmoid(params_dict['phi']).item()
                gam_v = nn.functional.softplus(params_dict['gamma']).item()
                s2_v  = torch.exp(params_dict['log_sigma2']).item()
                print(f"    DEC Ep {ep+1:3d} | val_LL={val_ll:.1f} | σ²={s2_v:.4f} φ={phi_v:.4f} γ={gam_v:.4f}")
    finally:
        for s in cached_subjs:
            for k in ('_dec_mu', '_dec_y', '_dec_times'):
                s.pop(k, None)

    return best_params, best_ll


def extract_dec_params(dist, params_dict):
    """Human-readable DEC parameter values.

    Key conventions for the merged dist_params dict (run_dec does
    `{**stage1, **dec_fitted}`):
      - For ALL dists: 'joint_sigma2' is DEC's joint marginal variance σ²_joint
        (the variance that multiplies C in the multivariate Gaussian).
      - For 'gaussian': also exposed as 'sigma'/'sigma2' since the joint
        variance IS the marginal variance.
      - For mixtures: 'sigma1' is comp-1 std-dev and 'sigma2' is comp-2
        std-dev (SAME convention as stage-1 losses.extract_params, so the
        merge updates the mixture cleanly and sample_error / fitted_error_pdf
        work without special-casing DEC).
    """
    joint_var = torch.exp(params_dict['log_sigma2']).item()
    result = {
        'joint_sigma2': joint_var,
        'joint_sigma':  float(np.sqrt(joint_var)),
        'phi':          torch.sigmoid(params_dict['phi']).item(),
        'gamma':        nn.functional.softplus(params_dict['gamma']).item(),
    }
    if dist == 'gaussian':
        # Joint variance == marginal variance for the Gaussian case.
        result['sigma']  = result['joint_sigma']
        result['sigma2'] = result['joint_sigma2']
        return result

    # ---- Single non-Gaussian distributions ----------------------------------
    if dist == 't':
        s2 = torch.exp(params_dict['log_sigma2_2']).item()
        result['sigma']  = float(np.sqrt(s2))
        result['sigma2'] = s2
        result['nu']     = (2.0 + nn.functional.softplus(params_dict['unc_nu'])).item()
        return result
    if dist == 'tukeyh':
        s2 = torch.exp(params_dict['log_sigma2_2']).item()
        result['sigma']  = float(np.sqrt(s2))
        result['sigma2'] = s2
        result['h']      = nn.functional.softplus(params_dict['raw_h']).item()
        return result

    # ---- 2-component mix-of-truncated mixtures with mean-zero constraint ---
    result['sigma1'] = torch.exp(params_dict['log_sigma1']).item()
    result['sigma2'] = torch.exp(params_dict['log_sigma2_2']).item()  # comp-2 std-dev
    result['pi']     = torch.sigmoid(params_dict['logit_pi']).item()
    result['mu1']    = params_dict['mu1'].item()
    result['mu2']    = -(result['pi'] / max(1.0 - result['pi'], 1e-12)) * result['mu1']
    if dist == 'truncmix_t':
        result['nu1'] = (2.0 + nn.functional.softplus(params_dict['unc_nu1'])).item()
        result['nu2'] = (2.0 + nn.functional.softplus(params_dict['unc_nu2'])).item()
    elif dist == 'truncmix_tukeyh':
        result['h1'] = nn.functional.softplus(params_dict['raw_h1']).item()
        result['h2'] = nn.functional.softplus(params_dict['raw_h2']).item()
    return result
