"""Group-aware subject splitters (reuse the shared builders in flare.data).

- split_ids   : 60/10/15/15 grouped train/val/cal/test id split (conformal workflow)
- build_split4: build subjects for those four id sets (encoders fit on train)
- build_split2: train/val split used by the FLARE estimator for early stopping
"""

from sklearn.model_selection import GroupShuffleSplit

from .data import (resolve_columns, fit_encoders, build_smap, build_subjects, make_loader)


def split_ids(prepared, cfg, seed, pool=None):
    """60/10/15/15 grouped subject split -> (train_ids, val_ids, cal_ids, test_ids)."""
    df = prepared['df']
    if pool is not None:
        df = df[df[cfg.subject_col].isin(pool)]
    g = df[cfg.subject_col].to_numpy()
    gss1 = GroupShuffleSplit(1, test_size=cfg.test_frac, random_state=seed)
    trainval, held = next(gss1.split(df, groups=g))
    val_rel = cfg.val_frac / (1 - cfg.test_frac)
    gss2 = GroupShuffleSplit(1, test_size=val_rel, random_state=seed)
    tr_rel, va_rel = next(gss2.split(df.iloc[trainval], groups=g[trainval]))
    gss3 = GroupShuffleSplit(1, test_size=0.5, random_state=seed)
    cal_rel, te_rel = next(gss3.split(df.iloc[held], groups=g[held]))
    ids = lambda idx: set(df.iloc[idx][cfg.subject_col].unique())
    train_ids = ids(trainval[tr_rel]); val_ids = ids(trainval[va_rel])
    cal_ids = ids(held[cal_rel]); test_ids = ids(held[te_rel])
    subsets = [train_ids, val_ids, cal_ids, test_ids]
    for a in range(4):
        for b in range(a + 1, 4):
            assert subsets[a].isdisjoint(subsets[b]), "subject subsets overlap"
    return train_ids, val_ids, cal_ids, test_ids


def build_split4(prepared, cfg, train_ids, val_ids, cal_ids, test_ids):
    """Build subjects for four explicit id sets; scalers/OHE fit on TRAIN only.
    Returns (tr_s, va_s, cal_s, te_s, tr_loader, va_loader, dims)."""
    hist, curr, cat, snum = resolve_columns(prepared)
    encoders, dims = fit_encoders(prepared, train_ids, hist, curr, cat, snum, cfg.subject_col)
    sa = prepared['subject_arrays']
    ps = prepared['patient_static']

    def build(ids):
        return build_subjects(ids, sa, build_smap(ids, ps, encoders, cfg.subject_col),
                              encoders, cfg.time_scale)

    tr_s = build(train_ids); va_s = build(val_ids)
    cal_s = build(cal_ids); te_s = build(test_ids)
    return (tr_s, va_s, cal_s, te_s,
            make_loader(tr_s, cfg.batch_size, shuffle=True),
            make_loader(va_s, cfg.batch_size, shuffle=False), dims)


def build_split2(prepared, cfg, val_frac=0.15, seed=42):
    """Train/val group split for the estimator. Returns (train_s, val_s, encoders, dims)."""
    df = prepared['df']
    g = df[cfg.subject_col].to_numpy()
    gss = GroupShuffleSplit(1, test_size=val_frac, random_state=seed)
    tr_rel, va_rel = next(gss.split(df, groups=g))
    train_ids = set(df.iloc[tr_rel][cfg.subject_col].unique())
    val_ids = set(df.iloc[va_rel][cfg.subject_col].unique())

    hist, curr, cat, snum = resolve_columns(prepared)
    encoders, dims = fit_encoders(prepared, train_ids, hist, curr, cat, snum, cfg.subject_col)
    sa = prepared['subject_arrays']
    ps = prepared['patient_static']
    train_s = build_subjects(train_ids, sa, build_smap(train_ids, ps, encoders, cfg.subject_col),
                             encoders, cfg.time_scale)
    val_s = build_subjects(val_ids, sa, build_smap(val_ids, ps, encoders, cfg.subject_col),
                           encoders, cfg.time_scale)
    return train_s, val_s, encoders, dims
