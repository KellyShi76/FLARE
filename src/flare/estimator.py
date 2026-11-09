"""FLARE — a scikit-learn-style estimator for the likelihood-based autoregressive RNN.

Wraps the functional core (adapter -> data -> splits -> train.run_model -> evaluate /
conformal) behind fit / predict / predict_interval / save / load, with plain-data I/O
(pandas DataFrames, dicts, python scalars) so it is easy to call from R via reticulate.
"""

import numpy as np
import pandas as pd
import torch

from .adapter import make_data_config
from . import data as _data
from .splits import build_split2
from .train import run_model
from .model import AutoregRNN
from .evaluate import predict_subjects, compute_parametric_pi, compute_empirical_pi
from . import conformal as _cf
from .losses import BUILTIN_DISTS
from .distributions import has_distribution, registered_distributions

_CALIB_METHODS = ("conformal", "cparam", "utopia", "utopia_het")


class FLARE:
    """Flexible-Likelihood Autoregressive Recurrent Estimation.

    Parameters
    ----------
    error_dist : str
        One of the four built-ins: ``gaussian``, ``truncmix_gauss``,
        ``truncmix_t``, or ``truncmix_tukeyh``; or the name of a custom law
        registered with :func:`flare.register_distribution`.
    rnn_hidden, rnn_head_layers, dropout : architecture of the recurrent core + head.
    lr, num_epochs, patience, batch_size, weight_decay : optimization.
    cutoff, cutoff_upper : truncation support of the fitted error density.
    mu1_init, sigma1_init, sigma2_init : truncated-mixture initialization.
    time_scale : divides the numeric visit time before it enters the model.
    validation_fraction : held-out group fraction for early stopping.
    level : default nominal coverage for prediction intervals.
    seed : random seed. device : None -> cuda if available else cpu; pass 'cpu' to force.
    distribution_options : dict
        Initialization options passed to a custom distribution's ``make_params`` callback.
    utopia_het_mode : str
        Advanced internal mode. The default is the paper implementation: three
        generic scale features, robust envelope, and no preliminary shape step.
    """

    def __init__(self, error_dist="truncmix_gauss", *, rnn_hidden=16,
                 rnn_head_layers=(128, 64), dropout=0.2, lr=1e-3, num_epochs=300,
                 patience=30, batch_size=128, weight_decay=1e-5,
                 cutoff=-9.0, cutoff_upper=9.0, mu1_init=0.0, sigma1_init=0.15,
                 sigma2_init=0.5, time_scale=1.0, validation_fraction=0.15,
                 level=0.95, seed=42, device=None, verbose=False,
                 distribution_options=None,
                 utopia_het_mode="het_dist_robust_noshape"):
        if error_dist not in BUILTIN_DISTS and not has_distribution(error_dist):
            choices = BUILTIN_DISTS + registered_distributions()
            raise ValueError(
                f"unknown error_dist {error_dist!r}; available distributions are {choices}. "
                "Register custom laws with flare.register_distribution first."
            )
        self.error_dist = error_dist
        self.rnn_hidden = rnn_hidden
        self.rnn_head_layers = list(rnn_head_layers)
        self.dropout = dropout
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.cutoff = cutoff
        self.cutoff_upper = cutoff_upper
        self.mu1_init = mu1_init
        self.sigma1_init = sigma1_init
        self.sigma2_init = sigma2_init
        self.time_scale = time_scale
        self.validation_fraction = validation_fraction
        self.level = level
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.distribution_options = dict(distribution_options or {})
        self.utopia_het_mode = utopia_het_mode
        self._fitted = False
        if not 0.0 < self.level < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly between 0 and 1")

    # ------------------------------------------------------------------ helpers
    def get_params(self):
        """Constructor arguments as a plain dict (for reconstruction / inspection)."""
        return dict(
            error_dist=self.error_dist, rnn_hidden=self.rnn_hidden,
            rnn_head_layers=list(self.rnn_head_layers), dropout=self.dropout,
            lr=self.lr, num_epochs=self.num_epochs, patience=self.patience,
            batch_size=self.batch_size, weight_decay=self.weight_decay,
            cutoff=self.cutoff, cutoff_upper=self.cutoff_upper, mu1_init=self.mu1_init,
            sigma1_init=self.sigma1_init, sigma2_init=self.sigma2_init,
            time_scale=self.time_scale, validation_fraction=self.validation_fraction,
            level=self.level, seed=self.seed, device=self.device, verbose=self.verbose,
            distribution_options=dict(self.distribution_options),
            utopia_het_mode=self.utopia_het_mode,
        )

    def _resolve_device(self):
        if self.device is not None:
            return str(self.device)
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _hparams(self):
        return dict(rnn_hidden=self.rnn_hidden, rnn_head_layers=list(self.rnn_head_layers),
                    dropout=self.dropout, lr=self.lr, num_epochs=self.num_epochs,
                    patience=self.patience, batch_size=self.batch_size,
                    weight_decay=self.weight_decay, cutoff=self.cutoff,
                    cutoff_upper=self.cutoff_upper, mu1_init=self.mu1_init,
                    sigma1_init=self.sigma1_init, sigma2_init=self.sigma2_init,
                    seed=self.seed, level=self.level, error_dist=self.error_dist,
                    distribution_options=dict(self.distribution_options))

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("this FLARE instance is not fitted yet; call fit(...) first")

    def _check_calibration_split(self, prediction_data, calibration_data):
        self._validate_frame(prediction_data, self.col_spec_, context="prediction data")
        self._validate_frame(calibration_data, self.col_spec_, context="calibration data")
        subject = self.col_spec_['subject_col']
        cal_ids = set(calibration_data[subject].unique().tolist())
        test_ids = set(prediction_data[subject].unique().tolist())
        overlap_train = cal_ids & self.fit_subject_ids_
        overlap_test = cal_ids & test_ids
        if overlap_train:
            raise ValueError(
                "calibration_data must contain subjects held out from fit; "
                f"found {len(overlap_train)} overlapping subject(s)"
            )
        if overlap_test:
            raise ValueError(
                "calibration_data and prediction data must be subject-disjoint; "
                f"found {len(overlap_test)} overlapping subject(s)"
            )

    @staticmethod
    def _validate_frame(df, spec, *, context):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{context} must be a pandas.DataFrame")
        if df.empty:
            raise ValueError(f"{context} is empty")
        roles = ([spec['subject_col'], spec['time_col'], spec['target_col']] +
                 spec['dynamic_cols'] + spec['static_cols'] + spec['categorical_cols'])
        duplicated = sorted({c for c in roles if roles.count(c) > 1})
        if duplicated:
            raise ValueError(f"columns assigned to multiple roles: {duplicated}")
        missing = [c for c in roles if c not in df.columns]
        if missing:
            raise ValueError(f"{context} is missing required columns: {missing}")
        if df[roles].isna().any().any():
            bad = df[roles].columns[df[roles].isna().any()].tolist()
            raise ValueError(f"{context} contains missing values in required columns: {bad}")
        numeric = [spec['target_col']] + spec['dynamic_cols'] + spec['static_cols']
        bad_numeric = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
        if bad_numeric:
            raise TypeError(f"these target/covariate columns must be numeric: {bad_numeric}")
        static = spec['static_cols'] + spec['categorical_cols']
        varying = [c for c in static
                   if (df.groupby(spec['subject_col'], sort=False)[c].nunique(dropna=False) > 1).any()]
        if varying:
            raise ValueError(
                f"columns declared static vary within subject: {varying}; "
                "move time-varying columns to dynamic_cols"
            )

    # ------------------------------------------------------------------ fit
    def fit(self, df, *, subject_col, time_col, target_col,
            dynamic_cols=None, current_dyn_cols=None,
            static_cols=None, categorical_cols=None):
        """Fit on a long DataFrame (one row per subject-visit). Returns self."""
        self.col_spec_ = dict(
            subject_col=subject_col, time_col=time_col, target_col=target_col,
            dynamic_cols=list(dynamic_cols or []),
            current_dyn_cols=(list(current_dyn_cols) if current_dyn_cols is not None else None),
            static_cols=list(static_cols or []),
            categorical_cols=list(categorical_cols or []),
        )
        if self.col_spec_['current_dyn_cols'] is not None:
            extra = sorted(set(self.col_spec_['current_dyn_cols']) -
                           set(self.col_spec_['dynamic_cols']))
            if extra:
                raise ValueError(f"current_dyn_cols must be a subset of dynamic_cols: {extra}")
        self._validate_frame(df, self.col_spec_, context="training data")
        if df[subject_col].nunique() < 3:
            raise ValueError("training data must contain at least three subjects")
        self.fit_subject_ids_ = set(df[subject_col].unique().tolist())
        cfg = make_data_config(
            df, subject_col=subject_col, time_col=time_col, target_col=target_col,
            dynamic_cols=self.col_spec_['dynamic_cols'],
            current_dyn_cols=self.col_spec_['current_dyn_cols'],
            static_cols=self.col_spec_['static_cols'],
            categorical_cols=self.col_spec_['categorical_cols'],
            time_scale=self.time_scale, **self._hparams())
        self._cfg = cfg
        device = self._resolve_device()

        prepared = _data.prepare_dataset(cfg, df=df)
        train_s, val_s, encoders, dims = build_split2(
            prepared, cfg, val_frac=self.validation_fraction, seed=self.seed)
        self._encoders = encoders
        self.dims_ = dims

        # run_model wants a 6-tuple; use val_s as a placeholder test set (its outputs
        # are discarded — predictions are recomputed via predict()).
        data6 = (train_s, val_s, val_s,
                 _data.make_loader(train_s, cfg.batch_size, shuffle=True),
                 _data.make_loader(val_s, cfg.batch_size, shuffle=False), dims)
        res = run_model(self.error_dist, self.seed, cfg, data6,
                        verbose=self.verbose, plot=False)
        self._model = res['model']
        self.state_ = res['state']
        self.dist_ = res['dist']
        self.dist_params_ = {k: float(v) for k, v in res['dist_params'].items()}

        tr_pred = predict_subjects(self._model, train_s, dims['seq_dim'], device)
        self.train_resid_ = (tr_pred['y_true'] - tr_pred['y_pred']).to_numpy().astype(float)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ predict
    def _predict_full(self, df):
        self._validate_frame(df, self.col_spec_, context="prediction data")
        device = self._resolve_device()
        self._model.to(device)
        self._model.eval()
        prepared_new = _data.prepare_dataset(self._cfg, df=df)
        subjects = _data.transform_new_subjects(prepared_new, self._encoders, self._cfg)
        return predict_subjects(self._model, subjects, self.dims_['seq_dim'], device)

    def predict(self, df):
        """Point predictions for a long DataFrame. Returns columns
        pid, visit_index, y_pred (+ y_true if the target column is present)."""
        self._check_fitted()
        pred = self._predict_full(df)
        cols = ['pid', 'visit_index', 'y_pred']
        if 'y_true' in pred.columns:
            cols = ['pid', 'visit_index', 'y_true', 'y_pred']
        return pred[cols].reset_index(drop=True)

    # ------------------------------------------------------------------ intervals
    def _pi_out(self, pi_df, method):
        out = pi_df[['pid', 'visit_index', 'y_pred', 'pi_lo', 'pi_hi', 'level']].copy()
        out['method'] = method
        return out.reset_index(drop=True)

    def predict_interval(self, df, *, method="parametric", level=None,
                         calibration_data=None, split=True):
        """Prediction intervals. Returns pid, visit_index, y_pred, pi_lo, pi_hi, level, method.

        method:
          'parametric' / 'empirical'  -> no calibration data needed.
          'conformal'                 -> split-conformal (empirical residual quantiles).
          'cparam'                    -> conformalized-parametric.
          'utopia' / 'utopia_het'     -> UTOPIA optimal-aggregation band.
        The last four REQUIRE `calibration_data` (a labeled long DataFrame held out
        from training).

        split (UTOPIA only, default True): use rigorous split-calibration -- fit the
        aggregation envelope and tail split on one half of the calibration set and set
        the scale factors on the held-out half. split=True is finite-sample valid but
        halves the calibration data, so it can be conservative when that set is small;
        split=False is sharper and empirically valid when the calibration set is large
        (thousands of units) but has no finite-sample coverage guarantee. Ignored by the
        non-UTOPIA methods."""
        self._check_fitted()
        level = self.level if level is None else level
        alpha = round(1.0 - level, 6)
        m = method.lower()
        test_pred = self._predict_full(df)

        if m == "parametric":
            pi = compute_parametric_pi(test_pred, self.dist_, self.dist_params_, "flare",
                                       cutoff=self.cutoff, cutoff_upper=self.cutoff_upper,
                                       levels=[level])
            return self._pi_out(pi, method)
        if m == "empirical":
            pi = compute_empirical_pi(test_pred, self.train_resid_, "flare", levels=[level])
            return self._pi_out(pi, method)
        if m in _CALIB_METHODS:
            if calibration_data is None:
                raise ValueError(
                    f"method {method!r} requires calibration_data "
                    "(a labeled long DataFrame held out from training)")
            self._check_calibration_split(df, calibration_data)
            cal_pred = self._predict_full(calibration_data)
            qL, qU = _cf.parametric_quantiles(
                self.dist_, self.dist_params_, alpha, self.cutoff, self.cutoff_upper)[:2]
            if m in ("conformal", "cparam"):
                gres = _cf.group_residuals(cal_pred, subsample_seed=self.seed)
                calib = _cf.calibrate(gres, alpha, qL, qU)
                key = "empirical" if m == "conformal" else "cparam"
                L, U = _cf.build_intervals(test_pred, key, qL, qU, calib)
            else:
                mode = self.utopia_het_mode if m == "utopia_het" else "base"
                L, U = _cf.utopia_intervals(
                    cal_pred, test_pred, self.dist_, self.dist_params_,
                    self.cutoff, self.cutoff_upper, alpha, split_seed=self.seed,
                    mode=mode, split=split)
            out = test_pred[['pid', 'visit_index', 'y_pred']].copy()
            out['pi_lo'] = np.asarray(L, dtype=float)
            out['pi_hi'] = np.asarray(U, dtype=float)
            out['level'] = level
            out['method'] = method
            return out.reset_index(drop=True)
        raise ValueError(f"unknown method {method!r}")

    def evaluate_intervals(self, df, calibration_data, *, levels=None):
        """Aggregated coverage / width / Winkler table (per method x level x visit group)
        via the conformal scorer. Returns a DataFrame of plain-scalar rows."""
        self._check_fitted()
        self._check_calibration_split(df, calibration_data)
        test_pred = self._predict_full(df)
        cal_pred = self._predict_full(calibration_data)
        rows, _ = _cf.score(test_pred, cal_pred, self.dist_, self.dist_params_,
                            self.cutoff, self.cutoff_upper, self.seed,
                            model_name="flare", levels=levels,
                            utopia_modes=(("base", "utopia"),
                                          (self.utopia_het_mode, "utopia_het")))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Persist the fitted estimator to a single file."""
        self._check_fitted()
        payload = dict(
            params=self.get_params(),
            col_spec_=self.col_spec_,
            dims_=self.dims_,
            dist_=self.dist_,
            dist_params_=self.dist_params_,
            state_=self.state_,
            encoders_=self._encoders,
            train_resid_=self.train_resid_,
            fit_subject_ids_=self.fit_subject_ids_,
            cfg=self._cfg,
        )
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path, device=None):
        """Load a fitted estimator saved by `save`."""
        payload = torch.load(path, map_location='cpu', weights_only=False)
        params = dict(payload['params'])
        if device is not None:
            params['device'] = device
        est = cls(**params)
        est.col_spec_ = payload['col_spec_']
        est.dims_ = payload['dims_']
        est.dist_ = payload['dist_']
        est.dist_params_ = payload['dist_params_']
        est.state_ = payload['state_']
        est._encoders = payload['encoders_']
        est.train_resid_ = payload['train_resid_']
        est.fit_subject_ids_ = set(payload.get('fit_subject_ids_', ()))
        est._cfg = payload['cfg']

        dev = est._resolve_device()
        model = AutoregRNN(est.dims_['seq_dim'], est.dims_['dyn_dim'], est.dims_['static_dim'],
                           hidden=est.rnn_hidden, head_layers=est.rnn_head_layers,
                           dropout=est.dropout)
        model.load_state_dict(est.state_)
        model.to(dev)
        model.eval()
        est._model = model
        est._fitted = True
        return est
