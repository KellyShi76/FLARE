"""Internal training/data configuration for FLARE.

This dataclass is glue between the generic data layer (`flare.data`) and the
training orchestrators (`flare.train`). End users normally do NOT construct it
directly — the `FLARE` estimator builds it from constructor args + column roles
via `flare.adapter.make_data_config`. It carries NO application-specific defaults.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Config:
    # === Data schema (set by the adapter from the caller's column roles) ===
    data_path: Optional[str] = None
    subject_col: str = 'subject_id'
    time_col: str = 'time'
    target_col: str = 'y'
    time_scale: float = 1.0
    time_is_numeric: bool = True          # True: time_col is numeric -> used as visit_day directly

    dynamic_cols: List[str] = field(default_factory=list)       # time-varying covariates kept in RNN history
    current_dyn_cols: List[str] = field(default_factory=list)   # subset fed to the head at the current visit
    categorical_cols: List[str] = field(default_factory=list)   # static categoricals (one-hot encoded)
    drop_cols: List[str] = field(default_factory=list)          # columns to discard up front
    combine_columns: Dict[str, List[str]] = field(default_factory=dict)  # collapse dummy groups -> one categorical
    agg_mean_cols: List[str] = field(default_factory=list)      # aggregated by mean within (subject, visit)
    agg_first_cols: List[str] = field(default_factory=list)     # by first; static numerics = agg_first \ dynamic
    constraints: Optional[object] = None                        # optional subject subsetting

    # Split fractions (used by split_and_build's 3-way split)
    train_frac: float = 0.6
    val_frac: float = 0.1
    test_frac: float = 0.3

    # === Training ===
    seed: int = 42
    meta_seed: int = 42
    level: float = 0.95
    subset: Optional[int] = None
    num_epochs: int = 300
    patience: int = 30
    lr: float = 1e-3
    batch_size: int = 128
    weight_decay: float = 1e-5
    rnn_hidden: int = 16
    rnn_head_layers: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2

    # Error distribution (see flare.losses.ALL_DISTS)
    error_dist: str = 'gaussian'
    distribution_options: Dict[str, object] = field(default_factory=dict)

    # Truncated-mixture initialization + support
    mu1_init: float = 0.0
    sigma1_init: float = 0.15
    sigma2_init: float = 0.5
    cutoff: float = -9.0
    cutoff_upper: float = 9.0

    use_dec: bool = False
    output_dir: Optional[str] = None
