"""Register a mean-zero Jones--Faddy skew-t innovation law with FLARE."""

import numpy as np
import torch
from scipy.special import betaln

from flare import DistributionSpec, FLARE, register_distribution


def make_params(device, options):
    scale = float(options.get("initial_scale", 1.0))
    a = float(options.get("initial_a", 4.0))
    b = float(options.get("initial_b", 2.5))
    if scale <= 0 or a <= 1 or b <= 1:
        raise ValueError("require initial_scale>0 and initial_a, initial_b>1")
    inv_softplus = lambda x: np.log(np.expm1(x))
    return {
        "log_scale": torch.nn.Parameter(torch.tensor(
            np.log(scale), dtype=torch.float32, device=device)),
        "raw_a": torch.nn.Parameter(torch.tensor(
            inv_softplus(a - 1), dtype=torch.float32, device=device)),
        "raw_b": torch.nn.Parameter(torch.tensor(
            inv_softplus(b - 1), dtype=torch.float32, device=device)),
    }


def transformed(params):
    scale = torch.exp(params["log_scale"])
    a = 1 + torch.nn.functional.softplus(params["raw_a"])
    b = 1 + torch.nn.functional.softplus(params["raw_b"])
    mean = (torch.sqrt(a + b) * (a - b) / 2 * torch.exp(
        torch.lgamma(a - 0.5) + torch.lgamma(b - 0.5)
        - torch.lgamma(a) - torch.lgamma(b)))
    return scale, a, b, mean


def batch_nll(model, seq, current, static, y, lengths, params,
              cutoff, cutoff_upper):
    scale, a, b, mean = transformed(params)
    z = (y - model(seq, current, static, lengths)) / scale + mean
    ratio = z / torch.sqrt(a + b + z.square())
    log_norm = (-(a + b - 1) * np.log(2) - torch.lgamma(a)
                - torch.lgamma(b) + torch.lgamma(a + b)
                - 0.5 * torch.log(a + b))
    log_pdf = (log_norm + (a + 0.5) * torch.log1p(ratio)
               + (b + 0.5) * torch.log1p(-ratio) - torch.log(scale))
    return -log_pdf.mean()


def extract_params(params):
    scale, a, b, mean = transformed(params)
    return {
        "scale": scale.item(),
        "a": a.item(),
        "b": b.item(),
        "base_mean": mean.item(),
    }


def sample_error(params, n, rng, cutoff, cutoff_upper):
    a, b = params["a"], params["b"]
    u = np.clip(rng.beta(a, b, size=n), 1e-12, 1 - 1e-12)
    z = ((2 * u - 1) * np.sqrt(a + b) /
         (2 * np.sqrt(u * (1 - u))))
    return params["scale"] * (z - params["base_mean"])


def error_pdf(x, params, cutoff, cutoff_upper):
    scale, a, b = params["scale"], params["a"], params["b"]
    z = np.asarray(x, dtype=float) / scale + params["base_mean"]
    ratio = z / np.sqrt(a + b + z ** 2)
    log_norm = (-(a + b - 1) * np.log(2) - betaln(a, b)
                - 0.5 * np.log(a + b))
    log_pdf = (log_norm + (a + 0.5) * np.log1p(ratio)
               + (b + 0.5) * np.log1p(-ratio) - np.log(scale))
    return np.exp(log_pdf)


register_distribution(DistributionSpec(
    name="skew_t",
    make_params=make_params,
    loss=batch_nll,
    extract_params=extract_params,
    sample=sample_error,
    pdf=error_pdf,
))

# `long_panel`, column roles, and subject-level splitting are as in quickstart.py.
# model = FLARE(
#     error_dist="skew_t",
#     distribution_options={"initial_scale": 1, "initial_a": 4, "initial_b": 2.5},
# ).fit(long_panel, ...)
