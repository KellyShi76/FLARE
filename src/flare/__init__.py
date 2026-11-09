"""FLARE — Flexible-Likelihood Autoregressive Recurrent Estimation.

A likelihood-based autoregressive RNN for irregular longitudinal data, with
non-Gaussian truncated-mixture error laws and split-conformal / UTOPIA prediction
intervals.

Headline API:
    from flare import FLARE, aggregate_and_format
    long = aggregate_and_format(raw, subject_col=..., time_col=..., target_col=..., ...)
    est  = FLARE(error_dist="truncmix_tukeyh").fit(long, subject_col=..., ...)
    est.predict(new); est.predict_interval(new, method="conformal", calibration_data=cal)

`import flare` is lightweight (no matplotlib/shap). Plotting lives in `flare.plotting`
(requires the optional `[plot]` extra) and is loaded on demand.
"""

__version__ = "0.2.0"

from .config import Config
from .estimator import FLARE
from .preprocess import aggregate_and_format
from .adapter import make_data_config
from .losses import (ALL_DISTS, BUILTIN_DISTS, make_dist_params, get_loss_fn, extract_params,
                     canonicalize_truncmix)
from .distributions import (DistributionSpec, register_distribution,
                            unregister_distribution, registered_distributions)
from .model import AutoregRNN, AR1Linear, AR1LongRunMean, MLPBaseline, SigmoidRNN
from .data import (prepare_dataset, split_and_build, build_data, make_loader,
                   transform_new_subjects, fit_ar1, predict_ar1)
from .splits import split_ids, build_split4, build_split2
from .train import run_model, run_baseline, run_residual_transfer, run_dec
from .evaluate import (predict_subjects, compute_point_metrics, compute_parametric_pi,
                       compute_empirical_pi, eval_pi, sample_error, fitted_error_pdf,
                       kl_divergence_hist, js_divergence_hist, print_results)
from . import conformal
from . import datasets


def available_distributions():
    """Built-in and currently registered custom distribution names."""
    return BUILTIN_DISTS + registered_distributions()

__all__ = [
    '__version__', 'Config', 'FLARE', 'aggregate_and_format', 'make_data_config',
    'ALL_DISTS', 'BUILTIN_DISTS', 'available_distributions',
    'DistributionSpec', 'register_distribution', 'unregister_distribution',
    'registered_distributions', 'make_dist_params', 'get_loss_fn', 'extract_params',
    'canonicalize_truncmix',
    'AutoregRNN', 'AR1Linear', 'AR1LongRunMean', 'MLPBaseline', 'SigmoidRNN',
    'prepare_dataset', 'split_and_build', 'build_data', 'make_loader',
    'transform_new_subjects', 'fit_ar1', 'predict_ar1',
    'split_ids', 'build_split4', 'build_split2',
    'run_model', 'run_baseline', 'run_residual_transfer', 'run_dec',
    'predict_subjects', 'compute_point_metrics', 'compute_parametric_pi',
    'compute_empirical_pi', 'eval_pi', 'sample_error', 'fitted_error_pdf',
    'kl_divergence_hist', 'js_divergence_hist', 'print_results',
    'conformal', 'datasets', 'plotting',
]


def __getattr__(name):
    # Lazy, optional submodules so `import flare` never pulls matplotlib/shap.
    if name == "plotting":
        import importlib
        return importlib.import_module(".plotting", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
