# FLARE

**Family-specific Likelihood-based Autoregressive Recurrent Estimation** for
continuous longitudinal outcomes. FLARE combines a recurrent conditional-mean
model with a user-selected innovation distribution and subject-level conformal
prediction intervals.

The package name is `flare-arnn`; the Python import is `flare`.

## What the package provides

- A recurrent model for a continuous outcome observed repeatedly within subject.
- Four built-in innovation laws: Gaussian, two-component Gaussian mixture,
  two-component Student-t mixture, and two-component Tukey-h mixture.
- A public plug-in interface for other differentiable, mean-zero innovation laws.
- Parametric, empirical, conformalized-parametric, UTOPIA, and UTOPIA-het
  prediction intervals.
- Subject-level splitting, training-only scaling/encoding, model persistence,
  and synthetic data for examples and tests.

The implementation is schema-agnostic. It contains no study-specific feature
engineering.

## Installation

From a release wheel:

```bash
pip install flare_arnn-0.2.0-py3-none-any.whl
```

From this source directory:

```bash
pip install .
```

Optional plotting and development dependencies are installed with:

```bash
pip install '.[plot]'
pip install '.[dev]'
```

PyTorch chooses platform-specific CPU/CUDA builds. On a GPU system, install the
appropriate PyTorch build first, then install FLARE.

## Required longitudinal-data format

Pass a pandas `DataFrame` in long format. The preferred input has exactly one
row per subject and visit. Column names are arbitrary because their roles are
declared when calling `fit`.

| Role | Required format |
|---|---|
| Subject ID | Any hashable value; identifies independent longitudinal units. |
| Visit time | Numeric or parseable datetime; unique within subject after aggregation. |
| Outcome | Numeric continuous value. |
| Dynamic covariates | Numeric values that may change between visits. |
| Static covariates | Numeric values constant within subject. |
| Categorical covariates | Values constant within subject; one-hot encoded internally. |

Additional requirements:

1. The training, calibration, and test cohorts must be disjoint by subject.
2. Required columns cannot contain missing values.
3. Static and categorical columns must not change within subject.
4. `current_dyn_cols`, when supplied, must be a subset of `dynamic_cols`.
5. Every panel passed to `fit`, `predict`, or interval methods must include the
   outcome column. FLARE produces rolling one-step-ahead fitted values: the
   outcome at a visit is not used to predict that same visit, but it becomes
   part of the history for subsequent visits.

Rows do not need to be sorted; FLARE sorts by subject and time. Numeric time is
used directly. Datetime-like time is converted to days since the subject's
first visit.

If raw data have several measurement-unit rows per subject and visit, aggregate
them before fitting:

```python
from flare import aggregate_and_format

panel = aggregate_and_format(
    raw,
    subject_col="subject_id",
    time_col="visit_date",
    target_col="outcome",
    dynamic_cols=["dynamic_x1", "dynamic_x2"],
    static_cols=["age_at_baseline"],
    categorical_cols=["group"],
    unit_col="measurement_unit",
)
```

The target and dynamic covariates are averaged within visit by default; static
and categorical values use the first value. Both aggregation rules are
configurable.

## Quick example

Assume `panel` satisfies the format above. Split subjects before fitting:

```python
import numpy as np
from flare import FLARE

ids = np.asarray(panel["subject_id"].drop_duplicates())
rng = np.random.default_rng(42)
rng.shuffle(ids)

n = len(ids)
train_ids = set(ids[: int(0.60 * n)])
cal_ids = set(ids[int(0.60 * n): int(0.80 * n)])
test_ids = set(ids[int(0.80 * n):])

train = panel[panel.subject_id.isin(train_ids)]
calibration = panel[panel.subject_id.isin(cal_ids)]
test = panel[panel.subject_id.isin(test_ids)]

model = FLARE(
    error_dist="truncmix_tukeyh",
    rnn_hidden=16,
    num_epochs=200,
    patience=30,
    seed=42,
).fit(
    train,
    subject_col="subject_id",
    time_col="visit_date",
    target_col="outcome",
    dynamic_cols=["dynamic_x1", "dynamic_x2"],
    current_dyn_cols=["dynamic_x1", "dynamic_x2"],
    static_cols=["age_at_baseline"],
    categorical_cols=["group"],
)

point_predictions = model.predict(test)

intervals = model.predict_interval(
    test,
    method="utopia_het",
    level=0.90,
    calibration_data=calibration,
)

summary = model.evaluate_intervals(
    test,
    calibration_data=calibration,
    levels=[0.90, 0.95],
)
```

A completely runnable synthetic version is in
[`examples/quickstart.py`](examples/quickstart.py).

## Built-in innovation distributions

Use one of:

```text
gaussian
truncmix_gauss
truncmix_t
truncmix_tukeyh
```

The three mixture models contain two independently truncated components and a
mean-zero constraint. Centering the innovation law is important: without it, a
constant shift can move between the recurrent mean and the error distribution,
so the two parts are not identifiable.

The truncation support is controlled by `cutoff` and `cutoff_upper`. Very wide
finite values approximate untruncated support while keeping all likelihood
calculations numerically defined.

## Custom innovation distributions

Register a `DistributionSpec` before constructing the estimator. This example
uses one Jones--Faddy skew-t density. Its two shape parameters control the
tails and skewness: `a > b` gives right skew, `a < b` gives left skew, and
`a = b` gives a Student-t distribution. We subtract its analytic mean so that
the FLARE innovation has mean zero.

```python
import numpy as np
import torch
from scipy.special import betaln

from flare import DistributionSpec, FLARE, register_distribution


def make_params(device, options):
    scale0 = float(options.get("initial_scale", 1.0))
    a0 = float(options.get("initial_a", 4.0))
    b0 = float(options.get("initial_b", 2.5))
    if scale0 <= 0 or a0 <= 1 or b0 <= 1:
        raise ValueError("require scale>0 and a,b>1")
    inv_softplus = lambda x: np.log(np.expm1(x))
    return {
        "log_scale": torch.nn.Parameter(torch.tensor(
            np.log(scale0), dtype=torch.float32, device=device)),
        "raw_a": torch.nn.Parameter(torch.tensor(
            inv_softplus(a0 - 1), dtype=torch.float32, device=device)),
        "raw_b": torch.nn.Parameter(torch.tensor(
            inv_softplus(b0 - 1), dtype=torch.float32, device=device)),
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
    scale, a, b, base_mean = transformed(params)
    z = (y - model(seq, current, static, lengths)) / scale + base_mean
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

model = FLARE(
    error_dist="skew_t",
    distribution_options={
        "initial_scale": 1.0,
        "initial_a": 4.0,
        "initial_b": 2.5,
    },
)
```

This example defines an untruncated skew-t law and therefore ignores `cutoff`
and `cutoff_upper` consistently in its loss, sampler, and PDF. A truncated
custom law must include the same normalizing constant in all three operations.

The callbacks have the following contracts:

- `make_params(device, options)` returns a nonempty
  `dict[str, torch.nn.Parameter]`.
- `loss(model, seq, current, static, y, lengths, params, cutoff,
  cutoff_upper)` returns the scalar mean batch NLL.
- `extract_params(params)` returns a mapping of plain numeric scalars.
- `sample(params, n, rng, cutoff, cutoff_upper)` returns exactly `n` finite
  residual draws using the supplied `numpy.random.RandomState`.
- `pdf(x, params, cutoff, cutoff_upper)` returns the fitted density at `x`, or
  `(total_density, component_densities)` for a mixture.

Sampling and PDF callbacks are needed for parametric intervals,
conformalized-parametric candidates, likelihood diagnostics, and UTOPIA. Custom
laws should be centered at zero and should treat truncation consistently in the
loss, sampler, and PDF.

The registry is process-local. Register the custom law again before loading a
saved custom-distribution model in a new process. See the complete skew-t
example in [`examples/custom_distribution.py`](examples/custom_distribution.py).

## Prediction-interval methods

`predict_interval(..., method=...)` accepts:

- `parametric`: fitted innovation quantiles; no finite-sample robustness to
  distributional misspecification.
- `empirical`: quantiles of training residuals.
- `conformal`: groupwise empirical split conformal.
- `cparam`: conformalized parametric intervals.
- `utopia`: aggregation of parametric, empirical, and conformalized-parametric
  candidates.
- `utopia_het`: an adaptive aggregation band whose width can vary within a
  visit group.

The conformal methods use visit groups 1, 2, 3, and 4-or-later. For the last
group, one visit is sampled per calibration subject, preserving a subject-level
exchangeability target.

### Current UTOPIA-het construction

The default `utopia_het` implementation matches the current research code. Its
adaptive envelope uses three generic scale variables:

1. absolute distance between the fitted mean and its calibration-group mean;
2. inverse visit index;
3. absolute distance between the recurrent-state norm and its
   calibration-group median.

On calibration-A, UTOPIA-het fits a robust quantile envelope rather than forcing
the envelope to contain every residual. It does not apply the original
preliminary asymmetric shape-shrinkage step, because that step relies on a
fully containing envelope. Calibration-B then supplies one exact conformal
rank for the complete nested band. With `split=True` (the default), this final
independent rank gives the usual finite-sample split-conformal marginal
coverage within each visit group under subject-level exchangeability.

Base UTOPIA deliberately retains full calibration-A containment. Its original
shape-rescaling step assumes that containment; replacing it by a quantile
envelope without also removing that step changes the score construction and
can produce unstable or anti-conservative bands.

`split=False` reuses calibration observations and is supplied only for
methodological comparison. It does not carry the split-conformal guarantee.

## Saving and loading

```python
model.save("flare_model.pt")
restored = FLARE.load("flare_model.pt", device="cpu")
```

Models contain the fitted encoders, recurrent weights, error parameters, input
column roles, and training-subject identifiers used to guard calibration
splits. They do not contain the original longitudinal records.


## Development and release checks

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```

The wheel and source archive produced under `dist/` are the files intended for
upload. Building the release does not publish it.

