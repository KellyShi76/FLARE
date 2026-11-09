"""
Loss functions for the four supported distributions.

  gaussian         : ε ~ N(0, σ²)
  truncmix_gauss   : ε ~ [π·N(0, σ₁²) + (1-π)·N(μ₂, σ₂²)] · 1(ε≥c) / Z
  truncmix_t       : ε ~ [π·t_{ν₁}(0, σ₁²) + (1-π)·t_{ν₂}(μ₂, σ₂²)] · 1(ε≥c) / Z
  truncmix_tukeyh  : ε ~ [π·TukeyH(0, σ₁², h₁) + (1-π)·TukeyH(μ₂, σ₂², h₂)] · 1(ε≥c) / Z

For all three mixtures the model output f(X) anchors component 1's location
at 0 on the residual scale; component 2 is offset by μ₂ (a learnable scalar).
The truncation point c (default −2.5) is fixed by config. No skew terms.
"""

import numpy as np
import torch
import torch.nn as nn

from .distributions import get_distribution, has_distribution


# ============================================================================
# Lambert W (principal branch, x >= 0) via Newton iteration — differentiable.
# Used by the Tukey-h inverse.
# ============================================================================

def lambertw_torch(x, n_iter=8):
    """Differentiable principal Lambert W for nonnegative finite inputs.

    Tukey-h observations can give ``h*u**2`` far above 700.  Clipping that
    argument changes the inverse and understates tail density, so the actual
    finite input is retained and Newton iteration starts from a stable
    large-input approximation.
    """
    finfo = torch.finfo(x.dtype)
    finite_max = finfo.max / 1.0e4
    x_safe = torch.nan_to_num(x, nan=0.0, posinf=finite_max, neginf=0.0)
    x_safe = torch.clamp(x_safe, min=0.0, max=finite_max)
    w_small = x_safe * (1.0 - 0.36787944117144233 * x_safe)
    log_x = torch.log(torch.clamp(x_safe, min=1.0 + finfo.eps))
    log_x_safe = torch.clamp(log_x, min=finfo.eps)
    w_mid = torch.log1p(x_safe)
    w_large = log_x - torch.log(log_x_safe) + torch.log(log_x_safe) / log_x_safe
    w = torch.where(x_safe > 3.0, w_large,
                    torch.where(x_safe > 1.0, w_mid, w_small))
    for _ in range(n_iter):
        ew = torch.exp(w)
        denom = ew * (1.0 + w) + 1e-30
        w = w - (w * ew - x_safe) / denom
    return w


def tau_h_inv_torch(u, h):
    """Inverse Tukey-h: z = u · exp(-W₀(h·u²)/2). Differentiable in h."""
    arg = h * u**2
    w = lambertw_torch(arg)
    return u * torch.exp(-w / 2)


def tau_hh_inv_torch(u, h_l, h_r):
    """Inverse Tukey-hh: same Lambert-W form as Tukey-h, but uses h_l when u<0
    and h_r when u>=0. Since sign(z) = sign(u) for h ≥ 0, picking the side by
    sign of u is well-defined and gradient flows cleanly to each side's h."""
    h = torch.where(u >= 0, h_r, h_l)
    arg = h * u**2
    w = lambertw_torch(arg)
    return u * torch.exp(-w / 2)


# ============================================================================
# Component log-PDFs and survival functions
# ============================================================================

_LOG_2PI = float(np.log(2 * np.pi))


def log_pdf_gauss(y, mu, sigma):
    """log N(y; mu, sigma²)."""
    return torch.distributions.Normal(mu, sigma).log_prob(y)


def log_pdf_t(y, mu, sigma, nu):
    """log Student-t(y; df=nu, loc=mu, scale=sigma) — symmetric."""
    return torch.distributions.StudentT(df=nu, loc=mu, scale=sigma).log_prob(y)


def log_pdf_tukeyh(y, xi, omega, h):
    """log TukeyH(y; xi, omega, h) with Z ~ N(0,1), Y = xi + ω·τ_h(Z). Symmetric in z."""
    u = (y - xi) / omega
    z = tau_h_inv_torch(u, h)
    log_phi = -0.5 * _LOG_2PI - 0.5 * z**2
    log_dy_dz = torch.log(omega) + h * z**2 / 2 + torch.log1p(h * z**2)
    return log_phi - log_dy_dz


def log_pdf_tukeyhh(y, xi, omega, h_l, h_r):
    """log Tukey-hh PDF: Y = xi + ω·τ_{hh}(Z), Z~N(0,1). Asymmetric tail heaviness
    via h_l (left, z<0) and h_r (right, z≥0). The mode and scale are unchanged
    relative to Tukey-h; only the per-side kurtosis differs."""
    u = (y - xi) / omega
    z = tau_hh_inv_torch(u, h_l, h_r)
    h = torch.where(u >= 0, h_r, h_l)
    log_phi = -0.5 * _LOG_2PI - 0.5 * z**2
    log_dy_dz = torch.log(omega) + h * z**2 / 2 + torch.log1p(h * z**2)
    return log_phi - log_dy_dz


# --- Survival P(Y >= c) in log-space ---

def log_surv_gauss(c, mu, sigma):
    normal = torch.distributions.Normal(0.0, 1.0)
    z = (c - mu) / sigma
    return torch.log(normal.cdf(-z).clamp(min=1e-30))


def _log_cdf_t_approx(z, nu):
    """Differentiable t-CDF via Peizer–Pratt: T_ν(z) ≈ Φ((1-1/(4ν))·z / √(1+z²/(2ν)))."""
    y_arg = z * (1.0 - 1.0 / (4.0 * nu)) / torch.sqrt(1.0 + z**2 / (2.0 * nu))
    normal = torch.distributions.Normal(0.0, 1.0)
    return torch.log(normal.cdf(y_arg).clamp(min=1e-30))


def log_surv_t(c, mu, sigma, nu):
    """log P(Y >= c) for Y = mu + sigma·T_ν, using T_ν(-z) = 1 - T_ν(z)."""
    z = (c - mu) / sigma
    return _log_cdf_t_approx(-z, nu)


def log_surv_tukeyh(c, xi, omega, h):
    """log P(Y >= c) for Y = xi + ω·τ_h(Z), Z ~ N(0,1). τ_h is monotone for h≥0."""
    u = (c - xi) / omega
    z = tau_h_inv_torch(u, h)
    normal = torch.distributions.Normal(0.0, 1.0)
    return torch.log(normal.cdf(-z).clamp(min=1e-30))


def log_surv_tukeyhh(c, xi, omega, h_l, h_r):
    """log P(Y >= c) for Y = xi + ω·τ_{hh}(Z), Z~N(0,1). τ_{hh} is monotone
    when both h_l, h_r ≥ 0, so the survival reduces to Φ(-z(c)) where the
    inverse picks h_r/h_l from the sign of (c-xi)/omega."""
    u = (c - xi) / omega
    z = tau_hh_inv_torch(u, h_l, h_r)
    normal = torch.distributions.Normal(0.0, 1.0)
    return torch.log(normal.cdf(-z).clamp(min=1e-30))


def _log_diff_surv(log_surv_lower, log_surv_upper):
    """log P(L ≤ Y ≤ U) = log(surv_lower − surv_upper) via logsubexp.

    log(a − b) = log(a) + log1p(−exp(log(b) − log(a))) for a ≥ b.
    Since L ≤ U ⟹ P(Y ≥ L) ≥ P(Y ≥ U), `log_surv_upper − log_surv_lower ≤ 0`,
    so `exp(·)` is in [0, 1] and `log1p(−·)` is finite. We clamp the inner
    `exp` away from 1 to avoid `log(0)` when the two bounds are numerically
    indistinguishable.
    """
    diff = log_surv_upper - log_surv_lower
    return log_surv_lower + torch.log1p(-torch.exp(diff).clamp(max=1 - 1e-10))


# ============================================================================
# Per-observation losses
# ============================================================================

def loss_mse(model, seq, curr, stat, y, lens, **kwargs):
    return nn.functional.mse_loss(model(seq, curr, stat, lens), y)


def loss_gaussian(model, seq, curr, stat, y, lens, log_sigma2, **kwargs):
    """NLL of y given N(f(X), σ²)."""
    mu = model(seq, curr, stat, lens)
    sigma2 = torch.exp(log_sigma2)
    return 0.5 * (torch.log(sigma2) + (y - mu)**2 / sigma2).mean()


def _common_truncmix_setup(model, seq, curr, stat, y, lens,
                           log_sigma1, log_sigma2, logit_pi, mu1):
    """Return (ε, σ_1, σ_2, π, μ_1, μ_2). The residual ε = y − f(X) is the
    variable on which the truncated mixture lives.

    Only μ_1 is a learnable parameter; μ_2 is derived from the *mean-zero
    mixture constraint*

        π · μ_1 + (1 − π) · μ_2 = 0   ⇒   μ_2 = − (π / (1−π)) · μ_1
                                            = − exp(logit_pi) · μ_1

    The exp(logit_pi) form avoids the 1/(1−π) blow-up as π → 1. This
    constraint pins the mixture's mean to 0, which kills the translation
    symmetry of the joint loss and *forces the RNN to predict an unbiased
    point estimate* (otherwise a constant bias in ŷ could be absorbed
    silently into μ_1 / μ_2, leading to catastrophic point-prediction
    failure)."""
    mu_rnn = model(seq, curr, stat, lens)
    s1 = torch.exp(log_sigma1)
    s2 = torch.exp(log_sigma2)
    pi = torch.sigmoid(logit_pi)
    m1 = mu1
    m2 = -torch.exp(logit_pi) * mu1
    eps = y - mu_rnn
    return eps, s1, s2, pi, m1, m2


def _truncmix_log_lik(eps, log_p1, log_p2, log_Z1, log_Z2, pi,
                      cutoff, cutoff_upper):
    """Mix-of-truncated log-likelihood (paper formulation):

        f(ε) = π · φ_1(ε)/Z_1  +  (1 − π) · φ_2(ε)/Z_2,    ε ∈ [a, b]

    Equivalently in log space:

        log f(ε) = logsumexp([ log π − log Z_1 + log φ_1(ε),
                              log(1−π) − log Z_2 + log φ_2(ε) ])

    Each component is independently truncated to [cutoff, cutoff_upper] and
    re-normalized by its own Z_k = ∫_a^b φ_k(ε) dε. The mixture weight π is
    therefore the *truncated* weight: P(comp 1 | ε ∈ [a, b])."""
    log_pi = torch.log(pi)
    log_one_minus_pi = torch.log1p(-pi)
    log_f = torch.logaddexp(log_pi + log_p1 - log_Z1,
                            log_one_minus_pi + log_p2 - log_Z2)
    valid = ((eps >= cutoff) & (eps <= cutoff_upper)).float()
    return -((log_f * valid + (-100.0) * (1 - valid)).mean())


def loss_truncmix_gauss(model, seq, curr, stat, y, lens,
                        log_sigma1, log_sigma2, logit_pi, mu1,
                        cutoff=-9.0, cutoff_upper=9.0, **kwargs):
    """Mix-of-truncated 2-component Gaussian on residual ε = y − f(X):

        f(ε) = π · N(ε; μ_1, σ_1²)/Z_1  +  (1 − π) · N(ε; μ_2, σ_2²)/Z_2
        Z_k  = Φ((U − μ_k)/σ_k) − Φ((L − μ_k)/σ_k)

    Both μ_1, μ_2 trainable (init 0); σ-asymmetric init breaks symmetry.
    """
    eps, s1, s2, pi, m1, m2 = _common_truncmix_setup(
        model, seq, curr, stat, y, lens,
        log_sigma1, log_sigma2, logit_pi, mu1)

    log_p1 = log_pdf_gauss(eps, m1, s1)
    log_p2 = log_pdf_gauss(eps, m2, s2)

    c_lo = torch.tensor(cutoff,       dtype=y.dtype, device=y.device)
    c_hi = torch.tensor(cutoff_upper, dtype=y.dtype, device=y.device)
    log_Z1 = _log_diff_surv(log_surv_gauss(c_lo, m1, s1),
                            log_surv_gauss(c_hi, m1, s1))
    log_Z2 = _log_diff_surv(log_surv_gauss(c_lo, m2, s2),
                            log_surv_gauss(c_hi, m2, s2))

    return _truncmix_log_lik(eps, log_p1, log_p2, log_Z1, log_Z2, pi,
                             cutoff, cutoff_upper)


def loss_truncmix_t(model, seq, curr, stat, y, lens,
                    log_sigma1, log_sigma2, logit_pi, mu1,
                    unc_nu1, unc_nu2,
                    cutoff=-9.0, cutoff_upper=9.0, **kwargs):
    """Mix-of-truncated 2-component Student-t; same form as gaussian with
    per-component degrees-of-freedom ν_k = 2 + softplus(unc_nu_k)."""
    eps, s1, s2, pi, m1, m2 = _common_truncmix_setup(
        model, seq, curr, stat, y, lens,
        log_sigma1, log_sigma2, logit_pi, mu1)
    nu1 = 2.0 + nn.functional.softplus(unc_nu1)
    nu2 = 2.0 + nn.functional.softplus(unc_nu2)

    log_p1 = log_pdf_t(eps, m1, s1, nu1)
    log_p2 = log_pdf_t(eps, m2, s2, nu2)

    c_lo = torch.tensor(cutoff,       dtype=y.dtype, device=y.device)
    c_hi = torch.tensor(cutoff_upper, dtype=y.dtype, device=y.device)
    log_Z1 = _log_diff_surv(log_surv_t(c_lo, m1, s1, nu1),
                            log_surv_t(c_hi, m1, s1, nu1))
    log_Z2 = _log_diff_surv(log_surv_t(c_lo, m2, s2, nu2),
                            log_surv_t(c_hi, m2, s2, nu2))

    return _truncmix_log_lik(eps, log_p1, log_p2, log_Z1, log_Z2, pi,
                             cutoff, cutoff_upper)


def loss_truncmix_tukeyh(model, seq, curr, stat, y, lens,
                         log_sigma1, log_sigma2, logit_pi, mu1,
                         raw_h1, raw_h2,
                         cutoff=-9.0, cutoff_upper=9.0, **kwargs):
    """Mix-of-truncated 2-component Tukey-h; per-component h_k ≥ 0
    (kurtosis), τ_h(z) = z·exp(h z²/2) is the monotone forward map."""
    eps, s1, s2, pi, m1, m2 = _common_truncmix_setup(
        model, seq, curr, stat, y, lens,
        log_sigma1, log_sigma2, logit_pi, mu1)
    h1 = nn.functional.softplus(raw_h1)
    h2 = nn.functional.softplus(raw_h2)

    log_p1 = log_pdf_tukeyh(eps, m1, s1, h1)
    log_p2 = log_pdf_tukeyh(eps, m2, s2, h2)

    c_lo = torch.tensor(cutoff,       dtype=y.dtype, device=y.device)
    c_hi = torch.tensor(cutoff_upper, dtype=y.dtype, device=y.device)
    log_Z1 = _log_diff_surv(log_surv_tukeyh(c_lo, m1, s1, h1),
                            log_surv_tukeyh(c_hi, m1, s1, h1))
    log_Z2 = _log_diff_surv(log_surv_tukeyh(c_lo, m2, s2, h2),
                            log_surv_tukeyh(c_hi, m2, s2, h2))

    return _truncmix_log_lik(eps, log_p1, log_p2, log_Z1, log_Z2, pi,
                             cutoff, cutoff_upper)



# ----------------------------------------------------------------------------
# Single non-Gaussian distributions (untruncated; mirror loss_gaussian)
# ----------------------------------------------------------------------------

def loss_t(model, seq, curr, stat, y, lens,
           log_sigma2, unc_nu,
           cutoff=-2.5, cutoff_upper=float('inf'), **kwargs):
    """Truncated single Student-t: ε ~ t_ν(0, σ²) on [cutoff, cutoff_upper].

    Convention matches `loss_gaussian`: `log_sigma2 = log(σ²)`, so the t-scale
    is `σ = exp(log_sigma2 / 2)`. With cutoff = -inf, cutoff_upper = +inf this
    recovers the untruncated t. 2 dist params."""
    mu = model(seq, curr, stat, lens)
    sigma = torch.exp(0.5 * log_sigma2)
    nu = 2.0 + nn.functional.softplus(unc_nu)

    log_p = log_pdf_t(y, mu, sigma, nu)

    c_lo = torch.tensor(cutoff, dtype=y.dtype, device=y.device)
    zero = torch.tensor(0.0, dtype=y.dtype, device=y.device)
    log_Z_lo = log_surv_t(c_lo, zero, sigma, nu)
    if cutoff_upper < float('inf'):
        c_hi = torch.tensor(cutoff_upper, dtype=y.dtype, device=y.device)
        log_Z = _log_diff_surv(log_Z_lo, log_surv_t(c_hi, zero, sigma, nu))
    else:
        log_Z = log_Z_lo

    eps = y - mu
    valid = ((eps >= cutoff) & (eps <= cutoff_upper)).float()
    log_lik = (log_p - log_Z) * valid + (-100.0) * (1 - valid)
    return -log_lik.mean()


def loss_tukeyh(model, seq, curr, stat, y, lens,
                log_sigma2, raw_h,
                cutoff=-2.5, cutoff_upper=float('inf'), **kwargs):
    """Truncated single Tukey-h: ε ~ TukeyH(0, σ², h) on [cutoff, cutoff_upper].

    Symmetric heavy tails for h > 0; recovers N(f(X), σ²) at h = 0. Same
    σ-convention as `loss_gaussian`: `log_sigma2 = log(σ²)`. 2 dist params."""
    mu = model(seq, curr, stat, lens)
    sigma = torch.exp(0.5 * log_sigma2)
    h = nn.functional.softplus(raw_h)

    log_p = log_pdf_tukeyh(y, mu, sigma, h)

    c_lo = torch.tensor(cutoff, dtype=y.dtype, device=y.device)
    zero = torch.tensor(0.0, dtype=y.dtype, device=y.device)
    log_Z_lo = log_surv_tukeyh(c_lo, zero, sigma, h)
    if cutoff_upper < float('inf'):
        c_hi = torch.tensor(cutoff_upper, dtype=y.dtype, device=y.device)
        log_Z = _log_diff_surv(log_Z_lo, log_surv_tukeyh(c_hi, zero, sigma, h))
    else:
        log_Z = log_Z_lo

    eps = y - mu
    valid = ((eps >= cutoff) & (eps <= cutoff_upper)).float()
    log_lik = (log_p - log_Z) * valid + (-100.0) * (1 - valid)
    return -log_lik.mean()



# ============================================================================
# Param factory, loss closure, and human-readable extractor
# ============================================================================

# Backwards-compat alias
_DIST_ALIAS = {'truncmix': 'truncmix_gauss'}
BUILTIN_DISTS = ('gaussian', 'truncmix_gauss', 'truncmix_t', 'truncmix_tukeyh')
# Public compatibility alias. Custom laws are intentionally not inserted into
# this immutable tuple; use flare.available_distributions() for the live list.
ALL_DISTS = BUILTIN_DISTS


def _resolve_dist(dist):
    return _DIST_ALIAS.get(dist, dist)


def make_dist_params(dist, device, init_sigma2=0.1,
                     mu1_init=0.0,
                     sigma1_init=0.15, sigma2_init=0.5,
                     distribution_options=None):
    """Create unconstrained learnable distribution parameters.

    For the 2-component truncmix dists, only μ_1 is trainable (init at 0);
    μ_2 is derived inside the loss via the mean-zero mixture constraint
    π·μ_1 + (1−π)·μ_2 = 0 (see `_common_truncmix_setup`). Component scales
    are initialized σ-asymmetrically (σ_1 < σ_2) to break the label-switching
    symmetry — the tight component tends to capture the bulk, the wide one
    the heavier tail. A post-hoc swap (in run_model) enforces σ_1 ≤ σ_2 in
    the reported parameters."""
    dist = _resolve_dist(dist)

    if has_distribution(dist):
        params = get_distribution(dist).make_params(
            device=device, options=dict(distribution_options or {}))
        if not isinstance(params, dict) or not params:
            raise TypeError("custom make_params must return a non-empty dict")
        bad = [k for k, value in params.items()
               if not isinstance(k, str) or not isinstance(value, nn.Parameter)]
        if bad:
            raise TypeError(
                "custom distribution parameters must be dict[str, torch.nn.Parameter]"
            )
        return params

    # --- Single-component dists keep their existing form ---
    if dist == 'gaussian':
        return {'log_sigma2': nn.Parameter(torch.tensor(np.log(init_sigma2), device=device))}
    if dist == 't':
        return {
            'log_sigma2': nn.Parameter(torch.tensor(np.log(init_sigma2), device=device)),
            'unc_nu':     nn.Parameter(torch.tensor(np.log(np.exp(6.0) - 1), device=device)),
        }
    if dist == 'tukeyh':
        return {
            'log_sigma2': nn.Parameter(torch.tensor(np.log(init_sigma2), device=device)),
            'raw_h':      nn.Parameter(torch.tensor(-2.0, device=device)),
        }

    # --- 2-component mean-zero-constrained truncated mixtures ---------------
    base = {
        'log_sigma1': nn.Parameter(torch.tensor(float(np.log(sigma1_init)), device=device)),
        'log_sigma2': nn.Parameter(torch.tensor(float(np.log(sigma2_init)), device=device)),
        'logit_pi':   nn.Parameter(torch.tensor(2.0, device=device)),   # σ(2) ≈ 0.88 → bulk-heavy at init
        'mu1':        nn.Parameter(torch.tensor(float(mu1_init), device=device)),
    }
    if dist == 'truncmix_gauss':
        return base
    if dist == 'truncmix_t':
        base['unc_nu1'] = nn.Parameter(torch.tensor(np.log(np.exp(8.0) - 1), device=device))
        base['unc_nu2'] = nn.Parameter(torch.tensor(np.log(np.exp(4.0) - 1), device=device))
        return base
    if dist == 'truncmix_tukeyh':
        base['raw_h1'] = nn.Parameter(torch.tensor(-2.0, device=device))
        base['raw_h2'] = nn.Parameter(torch.tensor(-2.0, device=device))
        return base
    raise ValueError(f"Unknown dist: {dist}")


def get_loss_fn(dist, dist_params, cutoff=-9.0, cutoff_upper=9.0):
    """Return a per-batch loss closure for one of the surviving distributions:
    gaussian (untruncated), t / tukeyh (truncated singles), or one of the
    2-component mix-of-truncated mixtures."""
    dist = _resolve_dist(dist)
    if has_distribution(dist):
        spec = get_distribution(dist)
        return lambda model, s, c, st, y, l: spec.loss(
            model=model, seq=s, current=c, static=st, y=y, lengths=l,
            params=dist_params, cutoff=cutoff, cutoff_upper=cutoff_upper)
    if dist == 'gaussian':
        return lambda model, s, c, st, y, l: loss_gaussian(
            model, s, c, st, y, l, **dist_params)
    if dist == 't':
        return lambda model, s, c, st, y, l: loss_t(
            model, s, c, st, y, l, **dist_params,
            cutoff=cutoff, cutoff_upper=cutoff_upper)
    if dist == 'tukeyh':
        return lambda model, s, c, st, y, l: loss_tukeyh(
            model, s, c, st, y, l, **dist_params,
            cutoff=cutoff, cutoff_upper=cutoff_upper)
    if dist == 'truncmix_gauss':
        return lambda model, s, c, st, y, l: loss_truncmix_gauss(
            model, s, c, st, y, l, **dist_params,
            cutoff=cutoff, cutoff_upper=cutoff_upper)
    if dist == 'truncmix_t':
        return lambda model, s, c, st, y, l: loss_truncmix_t(
            model, s, c, st, y, l, **dist_params,
            cutoff=cutoff, cutoff_upper=cutoff_upper)
    if dist == 'truncmix_tukeyh':
        return lambda model, s, c, st, y, l: loss_truncmix_tukeyh(
            model, s, c, st, y, l, **dist_params,
            cutoff=cutoff, cutoff_upper=cutoff_upper)
    raise ValueError(f"Unknown dist: {dist}")


def extract_params(dist, dist_params):
    """Extract human-readable parameter values."""
    dist = _resolve_dist(dist)
    if has_distribution(dist):
        fitted = dict(get_distribution(dist).extract_params(dist_params))
        try:
            return {str(k): float(v) for k, v in fitted.items()}
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "custom extract_params must return a mapping of scalar values"
            ) from exc
    if dist == 'gaussian':
        sigma2 = torch.exp(dist_params['log_sigma2']).item()
        return {'sigma': float(np.sqrt(sigma2)), 'sigma2': sigma2}
    if dist == 't':
        sigma2 = torch.exp(dist_params['log_sigma2']).item()
        return {
            'sigma':  float(np.sqrt(sigma2)),
            'sigma2': sigma2,
            'nu':     (2.0 + nn.functional.softplus(dist_params['unc_nu'])).item(),
        }
    if dist == 'tukeyh':
        sigma2 = torch.exp(dist_params['log_sigma2']).item()
        return {
            'sigma':  float(np.sqrt(sigma2)),
            'sigma2': sigma2,
            'h':      nn.functional.softplus(dist_params['raw_h']).item(),
        }
    if dist == 'truncmix_gauss':
        mu1_v = dist_params['mu1'].item()
        pi_v  = torch.sigmoid(dist_params['logit_pi']).item()
        mu2_v = -(pi_v / max(1.0 - pi_v, 1e-12)) * mu1_v
        return {
            'sigma1': torch.exp(dist_params['log_sigma1']).item(),
            'sigma2': torch.exp(dist_params['log_sigma2']).item(),
            'mu1':    mu1_v,
            'mu2':    mu2_v,
            'pi':     pi_v,
        }
    if dist == 'truncmix_t':
        mu1_v = dist_params['mu1'].item()
        pi_v  = torch.sigmoid(dist_params['logit_pi']).item()
        mu2_v = -(pi_v / max(1.0 - pi_v, 1e-12)) * mu1_v
        return {
            'sigma1': torch.exp(dist_params['log_sigma1']).item(),
            'sigma2': torch.exp(dist_params['log_sigma2']).item(),
            'mu1':    mu1_v,
            'mu2':    mu2_v,
            'pi':     pi_v,
            'nu1':    (2.0 + nn.functional.softplus(dist_params['unc_nu1'])).item(),
            'nu2':    (2.0 + nn.functional.softplus(dist_params['unc_nu2'])).item(),
        }
    if dist == 'truncmix_tukeyh':
        mu1_v = dist_params['mu1'].item()
        pi_v  = torch.sigmoid(dist_params['logit_pi']).item()
        mu2_v = -(pi_v / max(1.0 - pi_v, 1e-12)) * mu1_v
        return {
            'sigma1': torch.exp(dist_params['log_sigma1']).item(),
            'sigma2': torch.exp(dist_params['log_sigma2']).item(),
            'mu1':    mu1_v,
            'mu2':    mu2_v,
            'pi':     pi_v,
            'h1':     nn.functional.softplus(dist_params['raw_h1']).item(),
            'h2':     nn.functional.softplus(dist_params['raw_h2']).item(),
        }
    raise ValueError(f"Unknown dist: {dist}")


def canonicalize_truncmix(dist, fitted):
    """Post-hoc label canonicalization: enforce σ_1 ≤ σ_2 by swapping
    component identities if needed. This stabilizes summaries across CV folds
    (no label-switching artefacts in fold-averaged π, μ, σ). Returns a NEW
    dict; the input is left unchanged.

    The swap is identity-preserving for the density:
        (σ_1, μ_1, π)  ↔  (σ_2, μ_2, 1−π),
    plus the corresponding ν or h component params for t / tukeyh.
    """
    dist = _resolve_dist(dist)
    if has_distribution(dist):
        fn = get_distribution(dist).canonicalize
        return dict(fitted) if fn is None else dict(fn(dict(fitted)))
    if dist not in ('truncmix_gauss', 'truncmix_t', 'truncmix_tukeyh'):
        return dict(fitted)
    if fitted['sigma1'] <= fitted['sigma2']:
        return dict(fitted)
    out = dict(fitted)
    out['sigma1'], out['sigma2'] = fitted['sigma2'], fitted['sigma1']
    out['mu1'],    out['mu2']    = fitted['mu2'],    fitted['mu1']
    out['pi']                    = 1.0 - fitted['pi']
    if dist == 'truncmix_t':
        out['nu1'], out['nu2'] = fitted['nu2'], fitted['nu1']
    elif dist == 'truncmix_tukeyh':
        out['h1'],  out['h2']  = fitted['h2'],  fitted['h1']
    return out
