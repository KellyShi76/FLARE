"""End-to-end smoke tests for the FLARE estimator on simulated data (CPU, no private data).

Run:  python -m pytest tests/test_estimator.py -q
or:   python tests/test_estimator.py
"""
import sys
import numpy as np
import pandas as pd
import pytest


def _sim_df(n_subjects=60, lam=4, seed=0):
    from flare.datasets import SimParams, simulate_panel
    df, _fstar, _diag = simulate_panel(
        SimParams(n_subjects=n_subjects, poisson_lambda=lam), seed=seed)
    return df.drop(columns=["fstar"])


def test_import_is_lightweight():
    # Importing flare must not pull matplotlib.
    for m in list(sys.modules):
        if m == "matplotlib" or m.startswith("matplotlib."):
            del sys.modules[m]
    import flare  # noqa: F401
    assert "matplotlib" not in sys.modules


def test_preprocessor_aggregates():
    from flare import aggregate_and_format
    # two units per (subject, visit) -> must collapse to one row/visit
    rows = []
    for s in range(5):
        for v in range(3):
            for unit in range(2):
                rows.append(dict(id=s, date=float(v), site=unit,
                                 y=float(v + unit), x1=0.1 * v, age=40 + s, sex="M"))
    raw = pd.DataFrame(rows)
    long = aggregate_and_format(
        raw, subject_col="id", time_col="date", target_col="y",
        dynamic_cols=["x1"], static_cols=["age"], categorical_cols=["sex"],
        unit_col="site")
    assert len(long) == 5 * 3
    assert list(long.columns) == ["id", "date", "y", "x1", "age", "sex"]
    # y averaged over the two units (v and v+1) -> v + 0.5
    assert np.allclose(long[long.id == 0].sort_values("date")["y"].to_numpy(),
                       [0.5, 1.5, 2.5])


def test_fit_predict_interval_roundtrip(tmp_path):
    from flare import FLARE
    df = _sim_df(n_subjects=180)
    sid = df["STUDY_ID"]
    train = df[sid < 80]
    cal = df[(sid >= 80) & (sid < 130)]
    new = df[sid >= 130]
    est = FLARE(error_dist="truncmix_gauss", num_epochs=5, rnn_hidden=8,
                cutoff=-15, cutoff_upper=15, device="cpu", seed=0)
    est.fit(train, subject_col="STUDY_ID", time_col="MEASURE_DATE", target_col="Y",
            dynamic_cols=["D1", "D2"], static_cols=["X1", "X2"])

    preds = est.predict(new)
    assert {"pid", "visit_index", "y_pred"}.issubset(preds.columns)
    assert len(preds) > 0
    assert np.isfinite(preds["y_pred"].to_numpy()).all()

    for method in ("parametric", "empirical"):
        pi = est.predict_interval(df, method=method, level=0.9)
        assert (pi["pi_hi"] >= pi["pi_lo"]).all()

    # calibration-required methods (utopia defaults to split=True)
    for method in ("conformal", "cparam", "utopia", "utopia_het"):
        pi = est.predict_interval(new, method=method, level=0.9, calibration_data=cal)
        assert (pi["pi_hi"] >= pi["pi_lo"]).all()

    # UTOPIA: both split modes work and yield finite bands at this calibration size
    for sp in (True, False):
        pi = est.predict_interval(new, method="utopia_het", level=0.9,
                                  calibration_data=cal, split=sp)
        assert np.isfinite(pi["pi_lo"].to_numpy()).all()
        assert np.isfinite(pi["pi_hi"].to_numpy()).all()
        assert (pi["pi_hi"] >= pi["pi_lo"]).all()

    # clean error when calibration missing
    with pytest.raises(ValueError):
        est.predict_interval(new, method="conformal", level=0.9)

    # dist params are plain python floats (reticulate-safe)
    assert all(isinstance(v, float) for v in est.dist_params_.values())

    # save / load round-trip
    p = tmp_path / "flare_model.pt"
    est.save(str(p))
    est2 = FLARE.load(str(p), device="cpu")
    preds2 = est2.predict(new)
    assert preds2.shape == preds.shape
    assert np.allclose(preds2["y_pred"].to_numpy(), preds["y_pred"].to_numpy(), atol=1e-5)


def test_numeric_time_branch():
    from flare import FLARE
    df = _sim_df(n_subjects=40, lam=4, seed=1)
    # add a numeric time column (days) and use it as time_col
    df = df.sort_values(["STUDY_ID", "MEASURE_DATE"]).copy()
    df["visit_day"] = df.groupby("STUDY_ID").cumcount().astype(float) * 30.0
    est = FLARE(error_dist="gaussian", num_epochs=4, rnn_hidden=8,
                cutoff=-15, cutoff_upper=15, device="cpu", seed=0)
    est.fit(df, subject_col="STUDY_ID", time_col="visit_day", target_col="Y",
            dynamic_cols=["D1", "D2"], static_cols=["X1", "X2"])
    preds = est.predict(df)
    assert len(preds) > 0


def test_custom_distribution_end_to_end():
    import torch
    from flare import FLARE, DistributionSpec, register_distribution

    def make_params(device, options):
        scale = float(options.get("scale_init", 1.0))
        return {"log_scale": torch.nn.Parameter(
            torch.tensor(np.log(scale), dtype=torch.float32, device=device))}

    def loss(model, seq, current, static, y, lengths, params, cutoff, cutoff_upper):
        mu = model(seq, current, static, lengths)
        scale = torch.exp(params["log_scale"])
        return (torch.log(2.0 * scale) + torch.abs(y - mu) / scale).mean()

    def extract(params):
        return {"scale": torch.exp(params["log_scale"]).item()}

    def sample(params, n, rng, cutoff, cutoff_upper):
        return rng.laplace(0.0, params["scale"], n)

    def pdf(x, params, cutoff, cutoff_upper):
        scale = params["scale"]
        return np.exp(-np.abs(x) / scale) / (2.0 * scale)

    register_distribution(DistributionSpec(
        name="test_laplace", make_params=make_params, loss=loss,
        extract_params=extract, sample=sample, pdf=pdf), replace=True)

    df = _sim_df(n_subjects=45, lam=3, seed=5)
    est = FLARE(error_dist="test_laplace", distribution_options={"scale_init": 2.0},
                num_epochs=3, rnn_hidden=6, device="cpu", seed=1)
    est.fit(df, subject_col="STUDY_ID", time_col="MEASURE_DATE", target_col="Y",
            dynamic_cols=["D1", "D2"], static_cols=["X1", "X2"])
    assert est.dist_ == "test_laplace"
    assert est.dist_params_["scale"] > 0
    pi = est.predict_interval(df, method="parametric", level=0.9)
    assert np.isfinite(pi[["pi_lo", "pi_hi"]].to_numpy()).all()
    assert (pi["pi_hi"] >= pi["pi_lo"]).all()


if __name__ == "__main__":
    test_import_is_lightweight()
    test_preprocessor_aggregates()
    import tempfile, pathlib
    test_fit_predict_interval_roundtrip(pathlib.Path(tempfile.mkdtemp()))
    test_numeric_time_branch()
    test_custom_distribution_end_to_end()
    print("all estimator smoke checks passed")
