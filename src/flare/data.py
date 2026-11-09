"""
Data loading, generic preprocessing, subject building, and DataLoaders.

This is the schema-agnostic data layer for FLARE. It takes a long panel
(one row per subject-visit; use `flare.preprocess.aggregate_and_format` first if
your raw data has repeated within-visit measurements) with caller-specified
column roles and turns it into the per-subject arrays the model consumes.

History uses `dynamic_cols`; the current-visit head input uses
`current_dyn_cols` (a subset of `dynamic_cols`). Static covariates are
standardized (numeric) and one-hot encoded (categorical), fit on TRAIN only.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. Load / normalize a long panel
# ============================================================

def load_and_aggregate(cfg, df=None):
    """Return a clean one-row-per-(subject, visit) panel with a numeric
    `visit_day` column.

    If `df` is None the frame is read from `cfg.data_path`; otherwise the passed
    DataFrame is used in memory (the estimator path). Repeated `(subject, time)`
    rows are collapsed with the aggregations in `cfg.agg_mean_cols` /
    `cfg.agg_first_cols` (a no-op if the panel is already one row per visit).
    Time is handled by `cfg.time_is_numeric`: numeric time columns are used
    directly as `visit_day`; otherwise dates are parsed and `visit_day` is days
    from each subject's baseline. No application-specific derivations are done
    here — do those upstream before calling.
    """
    if df is None:
        df = pd.read_csv(cfg.data_path)
    else:
        df = df.copy()

    df = df.drop(columns=cfg.drop_cols, errors='ignore')

    # Generic dummy-group combining (config-driven; default: none). For each
    # (new_name, [cols...]), set new_name to the column with the row-wise max
    # (the 1-valued dummy for clean 0/1 groups) and drop the originals.
    combine_specs = getattr(cfg, 'combine_columns', None) or {}
    for new_name, src_cols in combine_specs.items():
        present = [c for c in src_cols if c in df.columns]
        if not present:
            continue
        df[new_name] = df[present].idxmax(axis=1).astype(str)
        df = df.drop(columns=present)

    df = df.dropna()

    cat_cols = [c for c in cfg.categorical_cols if c in df.columns]
    agg_dict = {}
    for c in cfg.agg_mean_cols:
        if c in df.columns:
            agg_dict[c] = 'mean'
    for c in cfg.agg_first_cols:
        if c in df.columns:
            agg_dict[c] = 'first'
    for c in cat_cols:
        agg_dict[c] = 'first'

    df = df.groupby([cfg.subject_col, cfg.time_col]).agg(agg_dict).reset_index()

    if getattr(cfg, 'time_is_numeric', False):
        df = df.sort_values([cfg.subject_col, cfg.time_col]).reset_index(drop=True)
        df['visit_day'] = df[cfg.time_col].astype(float)
        df['visit_interval'] = (
            df.groupby(cfg.subject_col)[cfg.time_col].diff().fillna(0).astype(float)
        )
        if cfg.time_col != 'visit_day':
            df = df.drop(columns=[cfg.time_col])
    else:
        df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
        df = df.sort_values([cfg.subject_col, cfg.time_col]).reset_index(drop=True)
        df['baseline_date'] = df.groupby(cfg.subject_col)[cfg.time_col].transform('min')
        df['visit_day'] = (df[cfg.time_col] - df['baseline_date']).dt.days.astype(float)
        df['visit_interval'] = (
            df.groupby(cfg.subject_col)[cfg.time_col].diff().dt.days.fillna(0).astype(float)
        )
        df = df.drop(columns=[cfg.time_col, 'baseline_date'])

    for c in cat_cols:
        df[c] = df[c].astype(str)

    df = df.sort_values([cfg.subject_col, 'visit_day']).reset_index(drop=True)
    return df


# ============================================================
# 2. Apply constraints (subset for transfer learning)
# ============================================================

def apply_constraints(df, cfg):
    if not cfg.constraints:
        return df

    constraints = cfg.constraints
    if isinstance(constraints, dict):
        constraints = [constraints]

    patient_first = (
        df.sort_values([cfg.subject_col, 'visit_day'])
        .groupby(cfg.subject_col, as_index=False).first()
    )

    combined_mask = pd.Series(False, index=patient_first.index)
    for gi, group in enumerate(constraints):
        group_mask = pd.Series(True, index=patient_first.index)
        parts = []
        for col, allowed in group.items():
            if col not in patient_first.columns:
                print(f"  Warning: constraint column '{col}' not in data, skipping")
                continue
            allowed_str = [str(v) for v in allowed]
            col_mask = patient_first[col].astype(str).isin(allowed_str)
            group_mask = group_mask & col_mask
            parts.append(f"{col} in {allowed}")
        n_match = group_mask.sum()
        desc = " AND ".join(parts) if parts else "(all)"
        print(f"  Group {gi+1}: {desc} -> {n_match} subjects")
        combined_mask = combined_mask | group_mask

    keep_ids = patient_first.loc[combined_mask, cfg.subject_col]
    df_filtered = df[df[cfg.subject_col].isin(keep_ids)].copy()
    print(f"  After constraints (OR): {len(keep_ids)} subjects, {len(df_filtered)} obs")
    return df_filtered


# ============================================================
# 3. Prepare (seed-independent) per-subject arrays
# ============================================================

def prepare_dataset(cfg, df=None,
                    override_dynamic_cols=None,
                    override_current_dyn_cols=None,
                    override_categorical_cols=None):
    """Seed-INDEPENDENT data prep — run ONCE per dataset configuration.

    Normalizes the panel and builds per-subject RAW (unscaled) arrays using the
    superset of dynamic/categorical columns requested by `cfg` (or `override_*`).
    Pass `df` to work in memory (no CSV read). Returns a `prepared` dict consumed
    by `split_and_build` / the splitters in `flare.splits`.
    """
    if df is None:
        print(f"  Loading: {cfg.data_path}")
    df = load_and_aggregate(cfg, df=df)
    print(f"  Raw: {df[cfg.subject_col].nunique()} subjects, {len(df)} obs")
    if cfg.constraints:
        print(f"  Applying constraints: {cfg.constraints}")
        df = apply_constraints(df, cfg)

    dyn_cfg = override_dynamic_cols if override_dynamic_cols is not None else cfg.dynamic_cols
    curr_cfg = override_current_dyn_cols if override_current_dyn_cols is not None else cfg.current_dyn_cols
    cat_cfg = override_categorical_cols if override_categorical_cols is not None else cfg.categorical_cols

    hist_dyn_cols = [c for c in dyn_cfg if c in df.columns]
    cat_cols = [c for c in cat_cfg if c in df.columns]
    static_num_cols = [c for c in cfg.agg_first_cols
                       if c not in dyn_cfg and c in df.columns]
    curr_dyn_cols = [c for c in curr_cfg if c in df.columns and c in hist_dyn_cols]

    print(f"  History dyn cols  : {hist_dyn_cols}")
    print(f"  Current dyn cols  : {curr_dyn_cols}")
    print(f"  Categorical static: {cat_cols}")

    df_sorted = df.sort_values([cfg.subject_col, 'visit_day'])
    subject_arrays = {}
    for pid, g in df_sorted.groupby(cfg.subject_col, sort=False):
        subject_arrays[pid] = {
            'y':            g[cfg.target_col].to_numpy(dtype=np.float32),
            'hist_dyn_raw': (g[hist_dyn_cols].to_numpy(dtype=np.float32)
                             if hist_dyn_cols
                             else np.zeros((len(g), 0), dtype=np.float32)),
            'visit_day':    g['visit_day'].to_numpy(dtype=np.float32),
            'T':            len(g),
        }

    patient_static = (
        df_sorted.groupby(cfg.subject_col, as_index=False, sort=False).first()
        [[cfg.subject_col] + cat_cols + static_num_cols].copy()
    )

    return {
        'df':               df,
        'subject_arrays':   subject_arrays,
        'patient_static':   patient_static,
        'hist_dyn_cols':    hist_dyn_cols,
        'curr_dyn_cols':    curr_dyn_cols,
        'cat_cols':         cat_cols,
        'static_num_cols':  static_num_cols,
        'encode_cols':      cat_cols + static_num_cols,
    }


# ============================================================
# 4. Reusable encoder-fitting + subject-building (shared by all splitters)
# ============================================================

def resolve_columns(prepared,
                    override_dynamic_cols=None,
                    override_current_dyn_cols=None,
                    override_categorical_cols=None):
    """Resolve which dynamic/current/categorical/static columns a fit uses,
    honoring optional LOCO-style overrides against the prepared superset."""
    full_hist = prepared['hist_dyn_cols']
    full_cat = prepared['cat_cols']
    full_snum = prepared['static_num_cols']

    hist = ([c for c in override_dynamic_cols if c in full_hist]
            if override_dynamic_cols is not None else list(full_hist))
    curr_src = (override_current_dyn_cols if override_current_dyn_cols is not None
                else prepared['curr_dyn_cols'])
    curr = [c for c in curr_src if c in hist]
    cat = ([c for c in override_categorical_cols if c in full_cat]
           if override_categorical_cols is not None else list(full_cat))
    snum = list(full_snum)
    return hist, curr, cat, snum


def fit_encoders(prepared, train_ids, hist_dyn_cols, curr_dyn_cols,
                 cat_cols, static_num_cols, subject_col):
    """Fit the static OneHot/StandardScaler and the dynamic StandardScaler on
    TRAIN subjects only. Returns `(encoders, dims)` where `encoders` bundles the
    fitted transformers + column indices needed to build subjects (also at
    predict time via `transform_new_subjects`)."""
    full_hist = prepared['hist_dyn_cols']
    dyn_col_idx = (np.array([full_hist.index(c) for c in hist_dyn_cols], dtype=int)
                   if hist_dyn_cols else np.array([], dtype=int))
    patient_static = prepared['patient_static']
    encode_cols = cat_cols + static_num_cols

    transformers = []
    if static_num_cols:
        transformers.append(("num", StandardScaler(), static_num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                  drop="first",
                                                  sparse_output=False), cat_cols))
    if transformers:
        ct_enc = ColumnTransformer(transformers, remainder="drop")
        tsd = patient_static[patient_static[subject_col].isin(train_ids)]
        ct_enc.fit(tsd[encode_cols])
        static_dim = ct_enc.transform(tsd[encode_cols]).shape[1]
        static_feat_names = list(static_num_cols)
        if cat_cols:
            ohe = ct_enc.named_transformers_['cat']
            static_feat_names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    else:
        ct_enc = None
        static_dim = 0
        static_feat_names = []

    if hist_dyn_cols:
        train_arrays = [prepared['subject_arrays'][pid]['hist_dyn_raw'][:, dyn_col_idx]
                        for pid in train_ids if pid in prepared['subject_arrays']]
        dyn_scaler = StandardScaler()
        if train_arrays:
            dyn_scaler.fit(np.vstack(train_arrays))
        else:
            dyn_scaler.fit(np.zeros((1, len(hist_dyn_cols))))
    else:
        dyn_scaler = None

    curr_idx_in_hist = [hist_dyn_cols.index(c) for c in curr_dyn_cols]

    encoders = {
        'ct_enc': ct_enc, 'dyn_scaler': dyn_scaler, 'dyn_col_idx': dyn_col_idx,
        'curr_idx_in_hist': curr_idx_in_hist, 'hist_dyn_cols': hist_dyn_cols,
        'curr_dyn_cols': curr_dyn_cols, 'cat_cols': cat_cols,
        'static_num_cols': static_num_cols, 'encode_cols': encode_cols,
    }
    dims = {
        'seq_dim': 1 + len(hist_dyn_cols), 'dyn_dim': len(curr_dyn_cols),
        'static_dim': static_dim, 'hist_dyn_cols': hist_dyn_cols,
        'curr_dyn_cols': curr_dyn_cols, 'cat_cols': cat_cols,
        'static_num_cols': static_num_cols, 'static_feat_names': static_feat_names,
    }
    return encoders, dims


def build_smap(ids, patient_static, encoders, subject_col):
    """Encode static covariates for `ids` using the fitted ColumnTransformer."""
    ct_enc = encoders['ct_enc']
    encode_cols = encoders['encode_cols']
    sub = patient_static[patient_static[subject_col].isin(ids)]
    if ct_enc is not None:
        X = ct_enc.transform(sub[encode_cols]).astype(np.float32)
    else:
        X = np.zeros((len(sub), 0), dtype=np.float32)
    return dict(zip(sub[subject_col], X))


def build_subjects(ids, subject_arrays, smap, encoders, time_scale, drop_history=False):
    """Slice the cached RAW arrays into the per-subject dict structure the model
    consumes, applying the fitted dynamic scaler (transform only)."""
    hist_dyn_cols = encoders['hist_dyn_cols']
    dyn_col_idx = encoders['dyn_col_idx']
    dyn_scaler = encoders['dyn_scaler']
    curr_idx = encoders['curr_idx_in_hist']
    seq_dim = 1 + len(hist_dyn_cols)

    subjects = []
    for pid in ids:
        arr = subject_arrays.get(pid)
        if arr is None or pid not in smap:
            continue
        T = arr['T']
        y = arr['y']
        if hist_dyn_cols:
            hd_scaled = dyn_scaler.transform(arr['hist_dyn_raw'][:, dyn_col_idx]).astype(np.float32)
        else:
            hd_scaled = np.zeros((T, 0), dtype=np.float32)
        curr_dyn = (hd_scaled[:, curr_idx] if curr_idx
                    else np.zeros((T, 0), dtype=np.float32))
        hist_mat = np.hstack([y.reshape(-1, 1), hd_scaled])
        if drop_history:
            histories = [np.zeros((0, seq_dim), dtype=np.float32) for _ in range(T)]
        else:
            histories = [hist_mat[:t] for t in range(T)]
        subjects.append({
            'pid':         pid,
            'y':           y,
            'histories':   histories,
            'current_dyn': curr_dyn,
            'static':      smap[pid],
            'times':       arr['visit_day'] / time_scale,
            'T':           T,
        })
    return subjects


def transform_new_subjects(prepared_new, encoders, cfg):
    """Build subjects for NEW data (predict time) using ALREADY-fitted encoders
    (transform only, never refit). `prepared_new` is the output of
    `prepare_dataset(cfg, df=new_df)`."""
    ids = list(prepared_new['subject_arrays'].keys())
    smap = build_smap(ids, prepared_new['patient_static'], encoders, cfg.subject_col)
    return build_subjects(ids, prepared_new['subject_arrays'], smap, encoders,
                          cfg.time_scale, drop_history=False)


# ============================================================
# 5. Seed-dependent split + build (3-way train/val/test 6-tuple)
# ============================================================

def split_and_build(prepared, cfg, seed=None, drop_history=False,
                    override_dynamic_cols=None,
                    override_current_dyn_cols=None,
                    override_categorical_cols=None):
    """Seed-DEPENDENT half. Group-aware train/val/test split, fit encoders on
    train, build subjects, return the legacy 6-tuple
    (train_s, val_s, test_s, train_loader, val_loader, dims)."""
    if seed is not None:
        cfg.seed = seed
    df = prepared['df']
    hist, curr, cat, snum = resolve_columns(
        prepared, override_dynamic_cols, override_current_dyn_cols, override_categorical_cols)
    if drop_history:
        print("  ** drop_history=True: histories will be empty")

    g_all = df[cfg.subject_col].to_numpy()
    gss1 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_frac, random_state=cfg.seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=g_all))
    val_relative = cfg.val_frac / (1 - cfg.test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=cfg.seed)
    train_rel, val_rel = next(gss2.split(df.iloc[trainval_idx], groups=g_all[trainval_idx]))
    train_ids = set(df.iloc[trainval_idx[train_rel]][cfg.subject_col].unique())
    val_ids = set(df.iloc[trainval_idx[val_rel]][cfg.subject_col].unique())
    test_ids = set(df.iloc[test_idx][cfg.subject_col].unique())
    print(f"  Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)} subjects")

    encoders, dims = fit_encoders(prepared, train_ids, hist, curr, cat, snum, cfg.subject_col)
    sa = prepared['subject_arrays']
    ps = prepared['patient_static']
    train_s = build_subjects(train_ids, sa, build_smap(train_ids, ps, encoders, cfg.subject_col),
                             encoders, cfg.time_scale, drop_history)
    val_s = build_subjects(val_ids, sa, build_smap(val_ids, ps, encoders, cfg.subject_col),
                           encoders, cfg.time_scale, drop_history)
    test_s = build_subjects(test_ids, sa, build_smap(test_ids, ps, encoders, cfg.subject_col),
                            encoders, cfg.time_scale, drop_history)

    if cfg.subset:
        rng = np.random.RandomState(cfg.seed)
        train_s = [train_s[i] for i in rng.choice(len(train_s),
                                                   min(cfg.subset, len(train_s)), replace=False)]
        val_s = [val_s[i] for i in rng.choice(len(val_s),
                                              min(max(cfg.subset // 3, 1), len(val_s)), replace=False)]
        test_s = [test_s[i] for i in rng.choice(len(test_s),
                                               min(max(cfg.subset // 2, 1), len(test_s)), replace=False)]

    print(f"  Dims: seq={dims['seq_dim']}, dyn={dims['dyn_dim']}, static={dims['static_dim']}")
    print(f"  Obs: train={sum(s['T'] for s in train_s)}, "
          f"val={sum(s['T'] for s in val_s)}, test={sum(s['T'] for s in test_s)}")

    train_loader = make_loader(train_s, cfg.batch_size, shuffle=True)
    val_loader = make_loader(val_s, cfg.batch_size, shuffle=False)
    return train_s, val_s, test_s, train_loader, val_loader, dims


def build_data(cfg, seed=None, drop_history=False,
               override_dynamic_cols=None,
               override_current_dyn_cols=None,
               override_categorical_cols=None):
    """Backward-compatible wrapper: prepare then split once for a single seed."""
    prepared = prepare_dataset(cfg,
                               override_dynamic_cols=override_dynamic_cols,
                               override_current_dyn_cols=override_current_dyn_cols,
                               override_categorical_cols=override_categorical_cols)
    return split_and_build(prepared, cfg, seed=seed, drop_history=drop_history,
                           override_dynamic_cols=override_dynamic_cols,
                           override_current_dyn_cols=override_current_dyn_cols,
                           override_categorical_cols=override_categorical_cols)


# ============================================================
# 6. DataLoader helpers
# ============================================================

def build_obs(subjects):
    return [{'seq_x': s['histories'][t], 'current_dyn': s['current_dyn'][t],
             'static': s['static'], 'y': s['y'][t],
             'seq_len': s['histories'][t].shape[0],
             'pid': s['pid'], 'visit_index': t + 1}
            for s in subjects for t in range(s['T'])]


class ObsDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        return (torch.tensor(e['seq_x'], dtype=torch.float32),
                torch.tensor(e['current_dyn'], dtype=torch.float32),
                torch.tensor(e['static'], dtype=torch.float32),
                torch.tensor(e['y'], dtype=torch.float32),
                e['seq_len'], e['pid'], e['visit_index'])


def collate_fn(batch):
    seqs, currs, stats_b, ys, lens, pids, vis = zip(*batch)
    lens = torch.tensor(lens, dtype=torch.long)
    currs = torch.stack(currs)
    stats_b = torch.stack(stats_b)
    ys = torch.stack(ys).view(-1, 1)
    seq_dim = 1
    for s in seqs:
        if s.shape[0] > 0:
            seq_dim = s.shape[1]
            break
    ml = max(max(lens).item(), 1)
    padded = torch.zeros(len(batch), ml, seq_dim)
    for i, s in enumerate(seqs):
        if s.shape[0] > 0:
            padded[i, :s.shape[0], :] = s
    return padded, currs, stats_b, ys, lens, pids, vis


def make_loader(subjects, batch_size=128, shuffle=True):
    return DataLoader(ObsDataset(build_obs(subjects)), batch_size,
                      shuffle=shuffle, collate_fn=collate_fn)


# ============================================================
# 7. AR(1) baseline
# ============================================================

def fit_ar1(subjects):
    X, Y = [], []
    for s in subjects:
        for t in range(1, s['T']):
            X.append([1.0, s['y'][t - 1]])
            Y.append(s['y'][t])
    reg = LinearRegression(fit_intercept=False).fit(X, Y)
    mean_y = np.mean([s['y'][0] for s in subjects])
    return reg, mean_y


def predict_ar1(reg, mean_y, subjects):
    rows = []
    for s in subjects:
        for t in range(s['T']):
            pred = mean_y if t == 0 else reg.predict([[1.0, s['y'][t - 1]]])[0]
            rows.append({'pid': s['pid'], 'visit_index': t + 1,
                         'y_true': s['y'][t], 'y_pred': float(pred)})
    return pd.DataFrame(rows)
