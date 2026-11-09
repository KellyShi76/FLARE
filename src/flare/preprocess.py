"""Generic preprocessing: reshape raw longitudinal data into FLARE's long panel.

`aggregate_and_format` collapses repeated within-visit measurements (e.g. one row
per tooth/site/replicate) to one row per (subject, visit) and returns a clean long
panel ready for `FLARE.fit`. It is schema-agnostic: all column roles are passed in,
and no application-specific derivations are performed.
"""

import pandas as pd


def aggregate_and_format(df, *, subject_col, time_col, target_col,
                         dynamic_cols=None, static_cols=None, categorical_cols=None,
                         unit_col=None, target_agg="mean", static_agg="first",
                         dropna=True, sort=True):
    """Aggregate repeated within-visit rows and return a one-row-per-(subject, visit)
    long panel.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw long data. May have multiple rows per (subject, visit) — e.g. one row per
        measurement unit given by ``unit_col``.
    subject_col, time_col, target_col : str
        Column roles. ``time_col`` may be date-like (strings/datetime) or numeric.
    dynamic_cols : list[str], optional
        Time-varying covariates (aggregated within a visit like the target).
    static_cols : list[str], optional
        Subject-level numeric covariates (aggregated with ``static_agg``).
    categorical_cols : list[str], optional
        Subject-level categorical covariates (aggregated with ``static_agg``).
    unit_col : str, optional
        The within-visit repeated-measure unit. Informational: aggregation is driven
        by grouping on (subject, visit); if the data already has one row per visit,
        the aggregation is a no-op.
    target_agg : str | callable, default "mean"
        Aggregation for the target and continuous dynamic columns.
    static_agg : str | callable | dict, default "first"
        Aggregation for static/categorical columns (a dict maps column -> agg).
    dropna : bool, default True
        Drop rows missing any used column before aggregating.
    sort : bool, default True
        Sort the output by (subject, time).

    Returns
    -------
    pandas.DataFrame
        Columns ``[subject_col, time_col, target_col, *dynamic_cols, *static_cols,
        *categorical_cols]``, one row per (subject, visit).
    """
    dynamic_cols = list(dynamic_cols or [])
    static_cols = list(static_cols or [])
    categorical_cols = list(categorical_cols or [])

    used = [subject_col, time_col, target_col] + dynamic_cols + static_cols + categorical_cols
    missing = [c for c in used if c not in df.columns]
    if missing:
        raise ValueError(f"columns not found in df: {missing}")
    dup = [c for c in used if used.count(c) > 1]
    if dup:
        raise ValueError(f"columns assigned to more than one role: {sorted(set(dup))}")

    keep = list(used)
    if unit_col and unit_col in df.columns and unit_col not in keep:
        keep.append(unit_col)
    out = df[keep].copy()
    if dropna:
        out = out.dropna(subset=used)

    agg_map = {target_col: target_agg}
    for c in dynamic_cols:
        agg_map[c] = target_agg

    def _static_agg_for(c):
        return static_agg.get(c, "first") if isinstance(static_agg, dict) else static_agg
    for c in static_cols + categorical_cols:
        agg_map[c] = _static_agg_for(c)

    grouped = out.groupby([subject_col, time_col], as_index=False).agg(agg_map)

    if sort:
        grouped = grouped.sort_values([subject_col, time_col]).reset_index(drop=True)
    else:
        grouped = grouped.reset_index(drop=True)

    cols = [subject_col, time_col, target_col] + dynamic_cols + static_cols + categorical_cols
    return grouped[cols]
