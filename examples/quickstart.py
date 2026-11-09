"""Minimal public example using synthetic longitudinal data only."""

from flare import FLARE
from flare.datasets import SimParams, simulate_panel


panel, _, _ = simulate_panel(
    SimParams(n_subjects=180, poisson_lambda=5), seed=7)
panel = panel.drop(columns="fstar")

# Split by subject. Calibration and test subjects must not have appeared in fit.
train = panel[panel.STUDY_ID < 80]
calibration = panel[(panel.STUDY_ID >= 80) & (panel.STUDY_ID < 130)]
test = panel[panel.STUDY_ID >= 130]

model = FLARE(
    error_dist="truncmix_tukeyh",
    num_epochs=20,
    patience=5,
    device="cpu",
    seed=1,
).fit(
    train,
    subject_col="STUDY_ID",
    time_col="MEASURE_DATE",
    target_col="Y",
    dynamic_cols=["D1", "D2"],
    static_cols=["X1", "X2"],
)

predictions = model.predict(test)
intervals = model.predict_interval(
    test,
    method="utopia_het",
    level=0.90,
    calibration_data=calibration,
)

print(predictions.head())
print(intervals.head())

