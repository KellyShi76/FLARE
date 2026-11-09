"""Build an internal Config from a long DataFrame + column roles.

This is the estimator's seam onto the generic data pipeline (the generalization of
the simulation-study `make_sim_config`). Because `flare.data.load_and_aggregate` is
schema-agnostic, any long panel flows through `prepare_dataset -> split -> run_model`
once the column roles are mapped here.
"""

from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

from .config import Config


def make_data_config(df, *, subject_col, time_col, target_col,
                     dynamic_cols=None, current_dyn_cols=None,
                     static_cols=None, categorical_cols=None,
                     time_scale=1.0, **training_hparams):
    """Return a Config wired for the caller's column roles + training hyperparameters.

    Static numeric covariates are `agg_first_cols \\ dynamic_cols` inside the pipeline,
    so we set `agg_first_cols = static_cols + dynamic_cols` (dynamic must survive the
    within-visit aggregation) and `agg_mean_cols = [target_col]`. `time_is_numeric` is
    inferred from the dtype of `df[time_col]`.
    """
    dynamic_cols = list(dynamic_cols or [])
    current_dyn_cols = list(current_dyn_cols if current_dyn_cols is not None else dynamic_cols)
    static_cols = list(static_cols or [])
    categorical_cols = list(categorical_cols or [])

    cfg = Config()
    cfg.subject_col = subject_col
    cfg.time_col = time_col
    cfg.target_col = target_col
    cfg.time_scale = time_scale

    col = df[time_col]
    cfg.time_is_numeric = bool(is_numeric_dtype(col) and not is_datetime64_any_dtype(col))

    cfg.dynamic_cols = dynamic_cols
    cfg.current_dyn_cols = [c for c in current_dyn_cols if c in dynamic_cols]
    cfg.categorical_cols = categorical_cols
    cfg.agg_mean_cols = [target_col]
    # dict.fromkeys de-dupes while preserving order; static numerics = agg_first \ dynamic.
    cfg.agg_first_cols = list(dict.fromkeys(static_cols + dynamic_cols))
    cfg.drop_cols = []
    cfg.combine_columns = {}
    cfg.constraints = None

    for k, v in training_hparams.items():
        setattr(cfg, k, v)
    return cfg
