"""
Simulation DGP v2 — oracle signal recovery under bimodal-Gaussian noise.

Goal: generate irregular longitudinal data whose response noise is a mixture of two
Gaussians (0.5 N(-3,1) + 0.5 N(3,1), mean 0). Because the noise is mean-zero every
model targets the same conditional mean f* = sin(a.X + b.D_t + c.S_{t-1}); the model
that correctly models the bimodal noise (truncmix_gauss) recovers f* much better than
the Gaussian (MSE) model. We record f* per observation so the runner can score the
ORACLE metric ||f* - pred_y||^2 directly.

The output is a long DataFrame consumable by :class:`flare.FLARE` and is used
only for public examples and tests; no application data are bundled.

DGP for subject j = 1..n:
    T_j  = Poisson(lambda) + 1                          # >= 1 visit
    d_jt = Uniform(a_int, b_int) days, t = 2..T_j        # irregular gaps
    X    = (X1, X2) ~ Uniform(-1, 1)                     # static, per subject
    S_0  = 0                                             # known common initial state
    for t = 1..T_j:
        D_t : Uniform(-1,1)^2 at t=1, else 0.6 D_{t-1} + Uniform(-0.5,0.5)^2
        f*_t = sin(a.X + b.D_t + c.S_{t-1})              # noise-free signal
        y_t  = f*_t + eps_t,   eps ~ 0.5 N(-3,1)+0.5 N(3,1)
        S_t  = cos(rho S_{t-1} + w_d.D_t + w_y y_t)      # contraction (|.| <= rho < 1)
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


# ============================================================
# Parameters
# ============================================================

@dataclass
class SimParams:
    # panel size / timing
    n_subjects: int = 300
    poisson_lambda: float = 10.0          # T_j = Poisson(lambda) + 1
    interval_a: float = 3.0               # gap ~ Uniform(a, b) in DAYS
    interval_b: float = 12.0
    base_date: str = '2010-01-01'

    # signal coefficients (sin argument spans several radians)
    a: List[float] = field(default_factory=lambda: [2.0, 1.0])   # on X = (X1, X2)
    b: List[float] = field(default_factory=lambda: [1.5, 1.0])   # on D = (D1, D2)
    c: float = 1.0                                                # on scalar S
    signal_amp: float = 1.0                                       # f* = signal_amp * sin(.) (raises SNR)
    initial_state: float = 0.0                                    # S_0, matched by fitted RNN initialization

    # dynamic-covariate transition: D_t = ar_d * D_{t-1} + Uniform(-innov, innov)
    ar_d: float = 0.6
    innov: float = 0.5

    # latent transition: S_t = cos(rho*S_{t-1} + w_d.D_t + w_y*y_t); rho<1 => contraction
    rho: float = 0.6
    w_d: List[float] = field(default_factory=lambda: [0.5, 0.5])
    w_y: float = 0.3

    # noise: symmetric 2-component mixture at +-noise_mu (mean 0), each component
    # either Gaussian ('gaussmix') or Tukey-h ('tukeyhmix'). Tukey-h gives heavy
    # polynomial tails, so only truncmix_tukeyh is correctly
    # specified -- this is the regime where the truncmix models should differ.
    noise_type: str = 'tukeyhmix'         # 'gaussmix' | 'tukeyhmix'
    noise_mu: float = 3.0
    noise_sd: float = 1.0
    noise_h: float = 0.4                  # Tukey-h tail parameter (per component)
    cutoff: float = -9.0                  # model truncation support
    cutoff_upper: float = 9.0
    # optional SKEW and HETEROSCEDASTICITY (appendix DGP; defaults reproduce the symmetric,
    # homoscedastic noise above). 'skewtukeyhmix' uses an asymmetric two-component mixture with
    # weight w1 at m1 and (1-w1) at +mu2, m1 = -(1-w1)*mu2/w1 (mean zero); when w1=0.5, mu2=noise_mu
    # it equals tukeyhmix. hetero_kappa scales each innovation by (1 + hetero_kappa*|S_{t-1}|).
    noise_w1: float = 0.5
    noise_mu2: float = 3.0
    noise_sd2: float = None               # 'skewtukeyhmix': sd of the component at +mu2 (None: noise_sd)
    hetero_kappa: float = 0.0             # sigma_t = 1 + hetero_kappa*|f*_t| (signal-driven scale)


# ============================================================
# Noise sampler
# ============================================================

def truncgauss_mixture_noise(n, rng, mu=3.0, sd=1.0):
    """Draw n samples from 0.5 N(-mu, sd^2) + 0.5 N(+mu, sd^2) (mean 0)."""
    comp = rng.random(n) < 0.5                       # True -> +mu component
    centers = np.where(comp, mu, -mu)
    return centers + sd * rng.standard_normal(n)


def trunc_tukeyhmix_noise(n, rng, mu=3.0, sd=1.0, h=0.4, cutoff=-9.0, cutoff_upper=9.0):
    """Draw n samples from 0.5 TukeyH(-mu,sd,h) + 0.5 TukeyH(+mu,sd,h) (mean 0).

    Each component: eps = center + sd * z * exp(h * z^2 / 2), z ~ N(0,1) (symmetric
    about its center, so the mixture mean is 0). Heavy polynomial tails;
    truncated to [cutoff, cutoff_upper] by rejection (matches the model support).
    """
    out = []
    while len(out) < n:
        B = max(2 * (n - len(out)), 1024)
        center = np.where(rng.random(B) < 0.5, mu, -mu)
        z = rng.standard_normal(B)
        eps = center + sd * z * np.exp(h * z ** 2 / 2)
        eps = eps[(eps >= cutoff) & (eps <= cutoff_upper)]
        out.extend(eps.tolist())
    return np.asarray(out[:n], dtype=float)


def trunc_skewtukeyhmix_noise(n, rng, w1=0.7, mu2=3.0, sd=1.0, h=0.4, cutoff=-9.0, cutoff_upper=9.0,
                              sd2=None):
    """Asymmetric (skewed) two-component truncated Tukey-h mixture, mean zero: weight w1 at
    m1 = -(1-w1)*mu2/w1 and weight (1-w1) at +mu2, each Tukey-h(sd, h). w1=0.5, mu2=noise_mu
    recovers the symmetric tukeyhmix; w1>0.5 gives a long right tail (right skew). sd2 gives the
    component at +mu2 its own scale (None: sd); the mean stays zero, it depends on the weights."""
    m1 = -(1.0 - w1) * mu2 / w1
    sd2 = sd if sd2 is None else sd2
    out = []
    while len(out) < n:
        B = max(2 * (n - len(out)), 1024)
        first = rng.random(B) < w1
        center = np.where(first, m1, mu2)
        z = rng.standard_normal(B)
        eps = center + np.where(first, sd, sd2) * z * np.exp(h * z ** 2 / 2)
        eps = eps[(eps >= cutoff) & (eps <= cutoff_upper)]
        out.extend(eps.tolist())
    return np.asarray(out[:n], dtype=float)


def draw_noise(p, rng, n):
    """Dispatch on p.noise_type (i.i.d. base innovations; heteroscedastic scaling is applied
    per observation in simulate_panel)."""
    if p.noise_type == 'gaussmix':
        return truncgauss_mixture_noise(n, rng, p.noise_mu, p.noise_sd)
    if p.noise_type == 'tukeyhmix':
        return trunc_tukeyhmix_noise(n, rng, p.noise_mu, p.noise_sd, p.noise_h,
                                     p.cutoff, p.cutoff_upper)
    if p.noise_type == 'skewtukeyhmix':
        return trunc_skewtukeyhmix_noise(n, rng, p.noise_w1, p.noise_mu2, p.noise_sd, p.noise_h,
                                         p.cutoff, p.cutoff_upper, p.noise_sd2)
    raise ValueError(p.noise_type)


# ============================================================
# Panel simulator
# ============================================================

def simulate_panel(p: SimParams, seed=0):
    """Generate the long panel. Returns (df, fstar_lookup, diag).

    fstar_lookup : {(STUDY_ID, visit_index): f*}  with visit_index 1-based.
    """
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp(p.base_date)
    a = np.asarray(p.a); b = np.asarray(p.b); w_d = np.asarray(p.w_d)

    rows = []
    fstar_lookup = {}
    all_eps = []
    all_fstar = []
    all_S = []

    for j in range(p.n_subjects):
        T = int(rng.poisson(p.poisson_lambda)) + 1

        X = rng.uniform(-1.0, 1.0, size=2)
        static_term = float(a @ X)
        # The fitted recurrence is initialized at zero.  Using the same known
        # initial state in the DGP prevents the first-visit conditional law from
        # being an unmodelled mixture over a latent random S_0.
        S_prev = float(p.initial_state)

        gaps = np.concatenate([[0.0],
                               rng.uniform(p.interval_a, p.interval_b, size=max(T - 1, 0))])
        visit_day = np.cumsum(gaps)

        D_prev = None
        for t in range(T):
            if t == 0:
                D = rng.uniform(-1.0, 1.0, size=2)
            else:
                D = p.ar_d * D_prev + rng.uniform(-p.innov, p.innov, size=2)

            fstar = p.signal_amp * np.sin(static_term + float(b @ D) + p.c * S_prev)
            scale = 1.0 + p.hetero_kappa * abs(float(fstar))    # signal-driven scale (1 if kappa=0)
            eps = scale * float(draw_noise(p, rng, 1)[0])
            y = float(fstar + eps)

            visit_index = t + 1
            date = base_date + pd.Timedelta(days=float(visit_day[t]))
            rows.append({
                'STUDY_ID': j,
                'MEASURE_DATE': date.strftime('%Y-%m-%d'),
                'Y': y,
                'D1': float(D[0]), 'D2': float(D[1]),
                'X1': float(X[0]), 'X2': float(X[1]),
                'fstar': float(fstar),
            })
            fstar_lookup[(j, visit_index)] = float(fstar)
            all_eps.append(eps); all_fstar.append(float(fstar)); all_S.append(S_prev)

            # contractive latent update (uses noisy y_t, which is observed)
            S_prev = float(np.cos(p.rho * S_prev + float(w_d @ D) + p.w_y * y))
            D_prev = D

    df = pd.DataFrame(rows)
    eps = np.asarray(all_eps); fstar = np.asarray(all_fstar); Sv = np.asarray(all_S)
    diag = {
        'n_obs': len(df),
        'n_subjects': p.n_subjects,
        'visits_mean': float(df.groupby('STUDY_ID').size().mean()),
        'noise_mean': float(eps.mean()),
        'noise_sd': float(eps.std()),
        'signal_sd': float(fstar.std()),
        'snr': float(fstar.std() / eps.std()),
        'latent_min': float(Sv.min()), 'latent_max': float(Sv.max()),
        'initial_state': float(p.initial_state),
    }
    return df, fstar_lookup, diag


def write_sim_csv(df, path):
    # f* is not a model feature; drop it from the CSV the pipeline reads.
    df.drop(columns=['fstar'], errors='ignore').to_csv(path, index=False)
    return path


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    eps = truncgauss_mixture_noise(200000, rng)
    print(f"noise: mean={eps.mean():+.4f} sd={eps.std():.4f} "
          f"frac<0={np.mean(eps < 0):.3f} "
          f"mode_sep: q25={np.quantile(eps, .25):+.2f} q75={np.quantile(eps, .75):+.2f}")
    p = SimParams()
    df, look, diag = simulate_panel(p, seed=0)
    print("panel diagnostics:")
    for k, v in diag.items():
        print(f"  {k:18s} {v}")
    print(df.head())
