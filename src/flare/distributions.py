"""Extension interface for user-defined FLARE innovation distributions.

The recurrent mean model is distribution agnostic.  A custom innovation law
supplies the five operations needed by training, reporting, likelihood scoring,
and parametric/conformal interval construction.  Register the specification
before constructing or loading a :class:`flare.FLARE` estimator.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional


@dataclass(frozen=True)
class DistributionSpec:
    """Callbacks defining a custom, mean-zero innovation distribution.

    Parameters
    ----------
    name:
        Unique value accepted by ``FLARE(error_dist=...)``.
    make_params:
        ``make_params(device, options) -> dict[str, torch.nn.Parameter]``.
    loss:
        ``loss(model, seq, current, static, y, lengths, params, cutoff,
        cutoff_upper) -> scalar torch.Tensor``.  It must call ``model`` and
        return the mean negative log-likelihood for the batch.
    extract_params:
        Convert learned tensors into a mapping of plain Python scalars.
    sample:
        ``sample(params, n, rng, cutoff, cutoff_upper) -> ndarray`` using the
        extracted parameters and the supplied ``numpy.random.RandomState``.
    pdf:
        ``pdf(x, params, cutoff, cutoff_upper)`` returning either the total PDF
        array or ``(total_pdf, component_pdf_list)``.
    canonicalize:
        Optional post-fit relabeling function for mixture components.

    Notes
    -----
    The innovation law should be centered at zero; otherwise the recurrent
    readout and the error location are not separately identifiable.  Callback
    functions must be importable at module scope if models will be reloaded in
    another Python process.  FLARE saves the registered name, not callback code.
    """

    name: str
    make_params: Callable
    loss: Callable
    extract_params: Callable
    sample: Callable
    pdf: Callable
    canonicalize: Optional[Callable] = None


_REGISTRY = {}


def register_distribution(spec, *, replace=False):
    """Register and return a :class:`DistributionSpec`.

    Registration is process-local.  Re-register the same specification before
    calling ``FLARE.load`` in a new process.
    """
    if not isinstance(spec, DistributionSpec):
        raise TypeError("spec must be a DistributionSpec")
    name = str(spec.name).strip()
    if not name:
        raise ValueError("distribution name must be non-empty")
    if name in _REGISTRY and not replace:
        raise ValueError(f"custom distribution {name!r} is already registered")
    _REGISTRY[name] = spec
    return spec


def unregister_distribution(name):
    """Remove a custom distribution registration, returning its specification."""
    return _REGISTRY.pop(name)


def get_distribution(name):
    """Return a registered custom distribution or raise a helpful error."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown custom distribution {name!r}; call "
            "flare.register_distribution(...) before constructing or loading the estimator"
        ) from exc


def has_distribution(name):
    """Whether ``name`` is registered as a custom distribution."""
    return name in _REGISTRY


def registered_distributions():
    """Names of currently registered custom distributions."""
    return tuple(sorted(_REGISTRY))

