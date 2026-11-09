"""
Core training API.

    from flare import run_model, run_residual_transfer, run_dec

    r = run_model('truncmix_tukeyh', seed=42, data=data, cfg=cfg)
    r_ft = run_model('gaussian', data=data_tgt, cfg=cfg, pretrained_state=r['state'])
    r_resid = run_residual_transfer(r, data_target=data_tgt, cfg=cfg)
    r_dec = run_dec(r, data=data, cfg=cfg)  # add DEC to any trained model

Supported dist values: 'gaussian', 'truncmix_gauss', 'truncmix_t',
'truncmix_tukeyh', 'ar1'. Legacy 'truncmix' aliases to 'truncmix_gauss'.
"""

import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from .config import Config
from .model import AutoregRNN, AR1Linear, AR1LongRunMean, MLPBaseline
from .losses import make_dist_params, get_loss_fn, extract_params, canonicalize_truncmix
from .dec import make_dec_params, train_dec, extract_dec_params
from .data import build_data, fit_ar1, predict_ar1
from .evaluate import (predict_subjects, compute_point_metrics,
                       compute_parametric_pi, compute_empirical_pi,
                       eval_pi, print_results)
# NOTE: plot_diagnostics is imported lazily inside the `if plot:` blocks below
# (from .plotting) so `import flare` never pulls matplotlib.


def _train_epoch(model, loader, optimizer, loss_fn, extra_params, device):
    model.train(); total, n = 0.0, 0
    for seq, curr, stat, y, lens, _, _ in loader:
        seq, curr, stat, y, lens = [t.to(device) for t in [seq, curr, stat, y, lens]]
        optimizer.zero_grad()
        loss = loss_fn(model, seq, curr, stat, y, lens)
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()) + extra_params, 5.0)
        optimizer.step()
        total += loss.item() * seq.size(0); n += seq.size(0)
    return total / n


@torch.no_grad()
def _eval_epoch(model, loader, loss_fn, device):
    model.eval(); total, n = 0.0, 0
    for seq, curr, stat, y, lens, _, _ in loader:
        seq, curr, stat, y, lens = [t.to(device) for t in [seq, curr, stat, y, lens]]
        total += loss_fn(model, seq, curr, stat, y, lens).item() * seq.size(0); n += seq.size(0)
    return total / n


def _train_loop(model, t_ldr, v_ldr, opt, loss_fn, extra_params=None,
                device='cpu', num_epochs=300, patience=30, verbose=True, tag=""):
    extra_params = [] if extra_params is None else list(extra_params)
    best_val, best_state, best_extra, pat = np.inf, None, None, 0
    for ep in range(num_epochs):
        tl = _train_epoch(model, t_ldr, opt, loss_fn, extra_params, device)
        vl = _eval_epoch(model, v_ldr, loss_fn, device)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_extra = [p.detach().cpu().clone() for p in extra_params]
            pat = 0
        else:
            pat += 1
            if pat >= patience: break
        if verbose and (ep + 1) % 20 == 0:
            print(f"    {tag} Ep {ep+1:3d} | train={tl:.6f} | val={vl:.6f}")
    # The likelihood checkpoint is joint: recurrent weights and separately
    # optimized distribution parameters must come from the same validation epoch.
    if best_state is not None:
        model.load_state_dict(best_state)
        with torch.no_grad():
            for param, saved in zip(extra_params, best_extra):
                param.copy_(saved.to(device=param.device, dtype=param.dtype))
    return best_state, best_val


def run_model(dist='gaussian', seed=42, cfg=None, data=None,
              pretrained_state=None, finetune_last_n=0,
              l2_pretrained=0.0,
              model_name=None, verbose=True, plot=True):
    """
    Train and evaluate one RNN model.

    Args:
        dist:               'gaussian', 'skewt', 'truncmix', or 'ar1'
        seed:               random seed
        cfg:                Config object
        data:               (train_s, val_s, test_s, train_loader, val_loader, dims)
        pretrained_state:   init weights from this (for transfer learning)
        finetune_last_n:    how many MLP blocks to unfreeze (0=all, 1=last Linear,
                            2=last 2 blocks, etc.). RNN is always frozen when >0.
        l2_pretrained:      L2 penalty weight toward pretrained params.
                            loss += l2_pretrained * Σ||θ - θ_pretrained||²
                            Use 0.0 to disable (default). Try 0.01-1.0.
        model_name:         override auto-generated name
        verbose/plot:       output control

    Returns dict:
        model, state, dist_params, predictions,
        point_metrics, param_pi, empirical_pi, param_pi_results, emp_pi_results
    """
    if cfg is None: cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    is_finetune = pretrained_state is not None
    if model_name is None:
        model_name = f"RNN-{dist.capitalize()}"
        if is_finetune:
            if finetune_last_n > 0:
                model_name += f"-FT{finetune_last_n}"
            else:
                model_name += "-FT"
            if l2_pretrained > 0:
                model_name += f"-L2"
        if dist == 'ar1': model_name = 'AR(1)'

    if verbose: print(f"\n  Training: {model_name} (seed={seed})")

    # Build data
    if data is None: data = build_data(cfg, seed)
    train_s, val_s, test_s, train_loader, val_loader, dims = data
    seq_dim, dyn_dim, static_dim = dims['seq_dim'], dims['dyn_dim'], dims['static_dim']

    # --- AR(1) ---
    if dist == 'ar1':
        reg, mean_y = fit_ar1(train_s)
        pred_df = predict_ar1(reg, mean_y, test_s)
        tr_pred = predict_ar1(reg, mean_y, train_s)
        tr_resid = (tr_pred['y_true'] - tr_pred['y_pred']).to_numpy()
        pm = compute_point_metrics(pred_df)
        emp_pi = compute_empirical_pi(pred_df, tr_resid, model_name)
        if verbose: print_results(model_name, pm, eval_pi(emp_pi))
        return dict(model=None, state=None, dist_params=None, dist=dist,
                    predictions=pred_df, point_metrics=pm,
                    param_pi=emp_pi, empirical_pi=emp_pi,
                    param_pi_results=eval_pi(emp_pi), emp_pi_results=eval_pi(emp_pi),
                    model_name=model_name, seed=seed)

    # --- Build model ---
    model = AutoregRNN(seq_dim, dyn_dim, static_dim,
                       hidden=cfg.rnn_hidden, head_layers=cfg.rnn_head_layers,
                       dropout=getattr(cfg, 'dropout', 0.2)).to(device)

    # Store pretrained weights for L2 penalty
    pretrained_dict = None

    if is_finetune:
        model.load_state_dict(pretrained_state)

        if l2_pretrained > 0:
            # Save a copy of pretrained weights for L2 penalty
            pretrained_dict = {k: v.clone().to(device) for k, v in pretrained_state.items()}

        if finetune_last_n > 0:
            # Freeze everything first
            for p in model.parameters():
                p.requires_grad = False

            # Identify Linear layers in the head (each is one "block")
            linear_layers = [m for m in model.head if isinstance(m, nn.Linear)]
            n_linear = len(linear_layers)
            n_unfreeze = min(finetune_last_n, n_linear)

            # Unfreeze last n_unfreeze Linear layers (+ their associated ReLU/Dropout)
            layers_to_unfreeze = linear_layers[-n_unfreeze:]
            for layer in layers_to_unfreeze:
                for p in layer.parameters():
                    p.requires_grad = True

            n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if verbose:
                layer_desc = [f"Linear({l.in_features}→{l.out_features})" for l in layers_to_unfreeze]
                print(f"    Unfreezing last {n_unfreeze} layers: {', '.join(layer_desc)}")
                print(f"    Trainable: {n_trainable} params, Frozen: {n_frozen} params")
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            lr_model = cfg.lr
            tag = f"{dist}-FT{finetune_last_n}"
        else:
            if verbose: print("    Fine-tuning ALL layers from pre-trained weights")
            trainable_params = list(model.parameters())
            lr_model = cfg.lr * 0.1
            tag = f"{dist}-FT"

        if l2_pretrained > 0 and verbose:
            print(f"    L2 penalty toward pretrained: λ={l2_pretrained}")
    else:
        if verbose:
            print(f"    Training from scratch: {dist} (RNN={cfg.rnn_hidden}, MLP={cfg.rnn_head_layers})")
        trainable_params = list(model.parameters())
        lr_model = cfg.lr
        tag = dist

    dist_params = make_dist_params(
        dist, device,
        init_sigma2=0.15,
        mu1_init=getattr(cfg, "mu1_init", 0.0),
        sigma1_init=getattr(cfg, 'sigma1_init', 0.15),
        sigma2_init=getattr(cfg, 'sigma2_init', 0.5),
        distribution_options=getattr(cfg, 'distribution_options', None),
    )

    # Build loss function (with optional L2 penalty)
    base_loss_fn = get_loss_fn(dist, dist_params, cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)
    if pretrained_dict is not None and l2_pretrained > 0:
        def loss_fn(model, seq, curr, stat, y, lens):
            nll = base_loss_fn(model, seq, curr, stat, y, lens)
            l2 = sum(((p - pretrained_dict[k])**2).sum()
                     for k, p in model.named_parameters() if k in pretrained_dict)
            return nll + l2_pretrained * l2
    else:
        loss_fn = base_loss_fn

    extra_p = list(dist_params.values())

    opt = optim.Adam([{'params': trainable_params, 'lr': lr_model},
                       {'params': extra_p, 'lr': cfg.lr}],
                      weight_decay=cfg.weight_decay)

    best_state, _ = _train_loop(model, train_loader, val_loader, opt, loss_fn,
        extra_params=extra_p, device=device, num_epochs=cfg.num_epochs,
        patience=cfg.patience, verbose=verbose, tag=tag)
    model.load_state_dict(best_state); model.to(device)

    fitted = extract_params(dist, dist_params)
    fitted = canonicalize_truncmix(dist, fitted)   # post-hoc σ-canonicalization
    if verbose: print(f"    Fitted: {fitted}")

    # --- Evaluate ---
    pred_df = predict_subjects(model, test_s, seq_dim, device)
    tr_pred = predict_subjects(model, train_s, seq_dim, device)
    tr_resid = (tr_pred['y_true'] - tr_pred['y_pred']).to_numpy()

    pm = compute_point_metrics(pred_df, dist=dist, dist_params=fitted,
                                cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)
    param_pi = compute_parametric_pi(pred_df, dist, fitted, model_name, cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)
    emp_pi = compute_empirical_pi(pred_df, tr_resid, model_name)

    if verbose:
        print_results(model_name, pm, eval_pi(param_pi))

    if plot:
        from .plotting import plot_diagnostics
        os.makedirs(cfg.output_dir, exist_ok=True)
        plot_diagnostics(pred_df, model_name, dist, fitted, cfg.output_dir, cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)

    return dict(
        model=model,
        state={k: v.cpu().clone() for k, v in model.state_dict().items()},
        dist_params=fitted, dist=dist,
        predictions=pred_df, point_metrics=pm,
        param_pi=param_pi, empirical_pi=emp_pi,
        param_pi_results=eval_pi(param_pi), emp_pi_results=eval_pi(emp_pi),
        model_name=model_name, seed=seed,
    )


def run_baseline(kind='mlp', dist='gaussian', seed=42, cfg=None, data=None,
                 model_name=None, verbose=True, plot=False):
    """Train a NON-recurrent baseline on the SAME data as run_model, under the
    given error law, so it flows through the identical loss / predict_subjects /
    conformal pipeline (and yields a real dist_params for the parametric/UTOPIA
    intervals).

    kind:
        'ar1' -- linear AR(1): ŷ_t = a + b·Y_{t-1}, fitted by conditional
                 least squares; visit 1 uses the model-implied long-run mean.
        'mlp' -- feed-forward on [current dynamic, static] with the RNN head
                 widths. Uses only static + current-dynamic covariates.

    Returns the same dict shape as run_model (model, state, dist_params,
    predictions, point_metrics, param_pi, empirical_pi, model_name, seed)."""
    if cfg is None: cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    if model_name is None:
        pretty = {'ar1': 'AR(1)', 'mlp': 'MLP'}.get(kind, kind.upper())
        model_name = f"{pretty}-{dist.capitalize()}"
    if verbose: print(f"\n  Training baseline: {model_name} (seed={seed})")

    if data is None: data = build_data(cfg, seed)
    train_s, val_s, test_s, train_loader, val_loader, dims = data
    seq_dim, dyn_dim, static_dim = dims['seq_dim'], dims['dyn_dim'], dims['static_dim']

    if kind == 'ar1':
        return _fit_ar1_long_run_mean(dist, seed, cfg, data, model_name, device, verbose)
    if kind == 'mlp':
        model = MLPBaseline(seq_dim, dyn_dim, static_dim,
                            hidden=cfg.rnn_hidden, head_layers=cfg.rnn_head_layers,
                            dropout=getattr(cfg, 'dropout', 0.2)).to(device)
    else:
        raise ValueError(f"unknown baseline kind: {kind}")

    dist_params = make_dist_params(
        dist, device, init_sigma2=0.15,
        mu1_init=getattr(cfg, "mu1_init", 0.0),
        sigma1_init=getattr(cfg, 'sigma1_init', 0.15),
        sigma2_init=getattr(cfg, 'sigma2_init', 0.5),
    )
    loss_fn = get_loss_fn(dist, dist_params, cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)
    extra_p = list(dist_params.values())

    opt = optim.Adam([{'params': list(model.parameters()), 'lr': cfg.lr},
                       {'params': extra_p, 'lr': cfg.lr}],
                      weight_decay=cfg.weight_decay)

    best_state, _ = _train_loop(model, train_loader, val_loader, opt, loss_fn,
        extra_params=extra_p, device=device, num_epochs=cfg.num_epochs,
        patience=cfg.patience, verbose=verbose, tag=model_name)
    model.load_state_dict(best_state); model.to(device)

    fitted = extract_params(dist, dist_params)
    fitted = canonicalize_truncmix(dist, fitted)
    if verbose: print(f"    Fitted: {fitted}")

    pred_df = predict_subjects(model, test_s, seq_dim, device)
    tr_pred = predict_subjects(model, train_s, seq_dim, device)
    tr_resid = (tr_pred['y_true'] - tr_pred['y_pred']).to_numpy()

    pm = compute_point_metrics(pred_df, dist=dist, dist_params=fitted,
                               cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)
    param_pi = compute_parametric_pi(pred_df, dist, fitted, model_name,
                                     cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)
    emp_pi = compute_empirical_pi(pred_df, tr_resid, model_name)

    if verbose:
        print_results(model_name, pm, eval_pi(param_pi))

    return dict(
        model=model,
        state={k: v.cpu().clone() for k, v in model.state_dict().items()},
        dist_params=fitted, dist=dist,
        predictions=pred_df, point_metrics=pm,
        param_pi=param_pi, empirical_pi=emp_pi,
        param_pi_results=eval_pi(param_pi), emp_pi_results=eval_pi(emp_pi),
        model_name=model_name, seed=seed,
    )


def _fit_ar1_long_run_mean(dist, seed, cfg, data, model_name, device, verbose):
    """AR(1)-Gaussian baseline aligned with the analysis implementation."""
    if dist != 'gaussian':
        raise ValueError("the AR(1) baseline is defined for a Gaussian error only")
    train_s, _, test_s, _, _, dims = data
    lagged = [s for s in train_s if s['T'] > 1]
    if not lagged:
        raise ValueError("AR(1) fitting requires at least one subject with two visits")
    previous = np.concatenate([s['y'][:-1] for s in lagged]).astype(float)
    current = np.concatenate([s['y'][1:] for s in lagged]).astype(float)
    design = np.column_stack([np.ones_like(previous), previous])
    (a, b), *_ = np.linalg.lstsq(design, current, rcond=None)
    if not abs(float(b)) < 1:
        raise ValueError(f"fitted AR(1) slope {b} has no finite long-run mean")
    model = AR1LongRunMean(a, b).to(device)

    pred_df = predict_subjects(model, test_s, dims['seq_dim'], device)
    tr_pred = predict_subjects(model, train_s, dims['seq_dim'], device)
    tr_resid = (tr_pred['y_true'] - tr_pred['y_pred']).to_numpy()
    sigma2 = float(np.mean(tr_resid ** 2))
    fitted = {'sigma': float(np.sqrt(sigma2)), 'sigma2': sigma2}
    if verbose:
        print(f"    AR(1): a={a:.4f} b={b:.4f} long-run mean={a / (1 - b):.4f} "
              f"sigma={fitted['sigma']:.4f}")

    pm = compute_point_metrics(pred_df, dist=dist, dist_params=fitted,
                               cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)
    param_pi = compute_parametric_pi(
        pred_df, dist, fitted, model_name,
        cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)
    emp_pi = compute_empirical_pi(pred_df, tr_resid, model_name)
    if verbose:
        print_results(model_name, pm, eval_pi(param_pi))
    return dict(
        model=model,
        state={k: v.cpu().clone() for k, v in model.state_dict().items()},
        dist_params=fitted, dist=dist,
        predictions=pred_df, point_metrics=pm,
        param_pi=param_pi, empirical_pi=emp_pi,
        param_pi_results=eval_pi(param_pi), emp_pi_results=eval_pi(emp_pi),
        model_name=model_name, seed=seed,
        ar1_coef={'a': float(a), 'b': float(b),
                  'long_run_mean': float(a / (1 - b))},
    )


def run_residual_transfer(source_result, dist='gaussian', seed=42, cfg=None,
                          data_target=None, residual_hidden=32,
                          residual_head_layers=None,
                          model_name=None, verbose=True, plot=True):
    """
    Residual transfer learning:
      1. Use pre-trained f_S to predict on target train/val/test
      2. Compute residuals r = y - f_S(X)
      3. Build residual histories: [r_{t-1}, Bleeding_{t-1}, AGE_{t-1}, ...]
      4. Train smaller RNN f_R on residuals with distributional loss
      5. Final prediction: ŷ = f_S(X) + f_R(X_resid)

    f_R architecture: RNN(residual_hidden) → MLP(residual_head_layers) → 1
    f_R input: history = [past residuals, past dynamic], current = [current dynamic, static]

    Args:
        source_result:       output from run_model() on source data
        dist:                error distribution for f_R ('gaussian', 'skewt', 'truncmix')
        data_target:         (train_s, val_s, test_s, ..., dims) for target group
        residual_hidden:     RNN hidden size for f_R (default 32)
        residual_head_layers: MLP layers for f_R (default [32, 16])
    """
    if cfg is None: cfg = Config()
    if residual_head_layers is None: residual_head_layers = [32, 16]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    if model_name is None:
        model_name = f"RNN-{dist.capitalize()}-ResidTL"

    if verbose: print(f"\n  Residual Transfer: {model_name} (seed={seed})")

    # Unpack target data
    if data_target is None:
        raise ValueError("data_target must be provided")
    train_s, val_s, test_s, _, _, dims = data_target
    seq_dim, dyn_dim, static_dim = dims['seq_dim'], dims['dyn_dim'], dims['static_dim']

    # --- Step 1: f_S predictions on all target subjects ---
    f_S = source_result['model']
    f_S.eval()
    if verbose: print("    Step 1: f_S predictions on target sets")

    pred_train = predict_subjects(f_S, train_s, seq_dim, device)
    pred_val = predict_subjects(f_S, val_s, seq_dim, device)
    pred_test = predict_subjects(f_S, test_s, seq_dim, device)

    # --- Step 2: build residual subjects with residual histories ---
    if verbose: print("    Step 2: building residual histories [r, dyn_cols]")

    def make_residual_subjects(subjects, pred_df):
        """
        Build subjects where:
          y → residual r = y - f_S(X)
          histories → [r_{t-1}, dyn_{t-1}] instead of [y_{t-1}, dyn_{t-1}]
        """
        resid_subjects = []
        for subj in subjects:
            pid = subj['pid']
            preds = pred_df[pred_df['pid'] == pid].sort_values('visit_index')
            if len(preds) == 0:
                continue
            yhat = preds['y_pred'].to_numpy().astype(np.float32)
            residuals = subj['y'] - yhat
            T = subj['T']

            # Build residual history matrix: [r, dyn_scaled] at each time
            # Original history[t] = [y_0..y_{t-1}, dyn_0..dyn_{t-1}]
            # Residual history[t] = [r_0..r_{t-1}, dyn_0..dyn_{t-1}]
            resid_histories = []
            for t in range(T):
                if t == 0:
                    resid_histories.append(np.zeros((0, seq_dim), dtype=np.float32))
                else:
                    orig_hist = subj['histories'][t]  # (t, seq_dim)
                    resid_hist = orig_hist.copy()
                    resid_hist[:, 0] = residuals[:t]  # replace y with r in column 0
                    resid_histories.append(resid_hist)

            resid_subjects.append({
                'pid': pid,
                'y': residuals,
                'histories': resid_histories,
                'current_dyn': subj['current_dyn'],
                'static': subj['static'],
                'times': subj['times'],
                'T': T,
            })
        return resid_subjects

    resid_train = make_residual_subjects(train_s, pred_train)
    resid_val = make_residual_subjects(val_s, pred_val)
    resid_test = make_residual_subjects(test_s, pred_test)

    if verbose:
        tr_resid = np.concatenate([s['y'] for s in resid_train])
        print(f"    Residual stats: mean={tr_resid.mean():.4f}, "
              f"std={tr_resid.std():.4f}, range=[{tr_resid.min():.3f}, {tr_resid.max():.3f}]")

    # --- Step 3: train f_R on residuals ---
    if verbose:
        print(f"    Step 3: training f_R (RNN={residual_hidden}, "
              f"MLP={residual_head_layers}) with {dist} loss")

    from .data import make_loader
    resid_train_loader = make_loader(resid_train, cfg.batch_size, shuffle=True)
    resid_val_loader = make_loader(resid_val, cfg.batch_size, shuffle=False)

    f_R = AutoregRNN(seq_dim, dyn_dim, static_dim,
                     hidden=residual_hidden,
                     head_layers=residual_head_layers).to(device)

    dist_params = make_dist_params(
        dist, device,
        init_sigma2=0.05,
        mu1_init=getattr(cfg, "mu1_init", 0.0),
        sigma1_init=getattr(cfg, 'sigma1_init', 0.15),
        sigma2_init=getattr(cfg, 'sigma2_init', 0.5),
    )
    loss_fn = get_loss_fn(dist, dist_params, cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)
    extra_p = list(dist_params.values())

    opt = optim.Adam([{'params': f_R.parameters(), 'lr': cfg.lr},
                       {'params': extra_p, 'lr': cfg.lr}],
                      weight_decay=cfg.weight_decay)

    best_state_R, _ = _train_loop(f_R, resid_train_loader, resid_val_loader, opt, loss_fn,
        extra_params=extra_p, device=device, num_epochs=cfg.num_epochs,
        patience=cfg.patience, verbose=verbose, tag=f"{dist}-resid")
    f_R.load_state_dict(best_state_R); f_R.to(device)

    fitted_R = extract_params(dist, dist_params)
    fitted_R = canonicalize_truncmix(dist, fitted_R)
    if verbose: print(f"    f_R dist params: {fitted_R}")

    # --- Step 4: combined prediction ŷ = f_S(X) + f_R(X_resid) ---
    if verbose: print("    Step 4: combined predictions ŷ = f_S(X) + f_R(X_resid)")

    # f_R predicts on residual-history subjects (not original subjects)
    pred_resid_test = predict_subjects(f_R, resid_test, seq_dim, device)

    # Merge: ŷ = ŷ_S + ŷ_R
    combined = pred_test.copy()
    combined = combined.merge(
        pred_resid_test[['pid', 'visit_index', 'y_pred']].rename(columns={'y_pred': 'y_pred_R'}),
        on=['pid', 'visit_index'], how='left',
    )
    combined['y_pred'] = combined['y_pred'] + combined['y_pred_R']
    combined['y_true'] = combined['y_true']  # original y, not residual
    combined = combined.drop(columns=['y_pred_R'])

    # --- Evaluate ---
    pm = compute_point_metrics(combined)

    # Combined training residuals for empirical PI
    pred_resid_train_out = predict_subjects(f_R, resid_train, seq_dim, device)
    combined_train = pred_train.copy()
    combined_train = combined_train.merge(
        pred_resid_train_out[['pid', 'visit_index', 'y_pred']].rename(columns={'y_pred': 'y_pred_R'}),
        on=['pid', 'visit_index'], how='left',
    )
    combined_train['y_pred'] = combined_train['y_pred'] + combined_train['y_pred_R']
    tr_resid_final = (combined_train['y_true'] - combined_train['y_pred']).to_numpy()

    # PI from f_R's fitted distribution (error of f_S + f_R)
    param_pi = compute_parametric_pi(combined, dist, fitted_R, model_name, cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)
    emp_pi = compute_empirical_pi(combined, tr_resid_final, model_name)

    if verbose:
        print_results(model_name, pm, eval_pi(param_pi))

    if plot:
        from .plotting import plot_diagnostics
        os.makedirs(cfg.output_dir, exist_ok=True)
        plot_diagnostics(combined, model_name, dist, fitted_R, cfg.output_dir, cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)

    return dict(
        model_source=f_S,
        model_residual=f_R,
        model=f_R,  # for SHAP compatibility
        state_source=source_result['state'],
        state_residual={k: v.cpu().clone() for k, v in f_R.state_dict().items()},
        state={k: v.cpu().clone() for k, v in f_R.state_dict().items()},
        dist_params=fitted_R, dist=dist,
        predictions=combined, point_metrics=pm,
        param_pi=param_pi, empirical_pi=emp_pi,
        param_pi_results=eval_pi(param_pi), emp_pi_results=eval_pi(emp_pi),
        model_name=model_name, seed=seed,
        pred_source=pred_test, pred_residual=pred_resid_test,
    )


def run_dec(trained_result, data, cfg=None, seed=42,
            model_name=None, verbose=True, plot=True):
    """
    Add DEC temporal correlation to an already-trained model.

    Takes any trained result (from run_model, run_residual_transfer, etc.),
    freezes the RNN, and trains DEC params (φ, γ, σ²) + distribution params
    using joint log-likelihood across visits within each patient.

    Args:
        trained_result: output from run_model() or run_residual_transfer()
                        must contain 'model', 'state', 'dist_params', 'dist'
        data:           (train_s, val_s, test_s, train_loader, val_loader, dims)
        cfg:            Config object
        seed:           random seed
        model_name:     override (default: original name + '-DEC')
        verbose/plot:   output control

    Returns:
        dict with same structure as run_model output, plus DEC params.
        model_name contains '-DEC' suffix for PI conditional computation.

    Usage:
        r = run_model('skewt', data=data, cfg=cfg)
        r_dec = run_dec(r, data=data, cfg=cfg)

        r_ft = run_model('skewt', data=data_target, pretrained_state=r_pre['state'])
        r_ft_dec = run_dec(r_ft, data=data_target, cfg=cfg)

        r_resid = run_residual_transfer(r_pre, data_target=data_target)
        r_resid_dec = run_dec(r_resid, data=data_target, cfg=cfg)
    """
    if cfg is None: cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    dist = trained_result['dist']
    base_name = trained_result['model_name']
    if model_name is None:
        model_name = base_name + '-DEC' if '-DEC' not in base_name else base_name

    if verbose: print(f"\n  DEC: {model_name} (adding DEC to {base_name})")

    # Unpack data
    train_s, val_s, test_s, _, _, dims = data
    seq_dim = dims['seq_dim']

    # Reconstruct and freeze model
    model = trained_result['model']
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Init DEC params from fitted distribution
    fitted_stage1 = trained_result['dist_params']
    dec_params = make_dec_params(dist, fitted_stage1, device)

    if verbose:
        print(f"    Dist: {dist}, init from: {fitted_stage1}")
        print(f"    DEC params: φ, γ, σ² + dist params (all learnable)")

    # Train DEC
    opt = optim.Adam(list(dec_params.values()), lr=cfg.lr)
    best_dec, best_ll = train_dec(
        model, train_s, val_s, opt, dist, dec_params,
        seq_dim=seq_dim, cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper, device=device,
        num_epochs=cfg.num_epochs, patience=cfg.patience, verbose=verbose,
    )
    for k in dec_params:
        dec_params[k].data = best_dec[k]

    dec_fitted = extract_dec_params(dist, dec_params)
    if verbose: print(f"    DEC fitted: {dec_fitted}")

    # Merge: stage1 dist params + DEC params
    fitted_all = {**fitted_stage1, **dec_fitted}

    # Predictions (same f(X), DEC doesn't change point predictions)
    pred_df = predict_subjects(model, test_s, seq_dim, device)
    tr_pred = predict_subjects(model, train_s, seq_dim, device)
    tr_resid = (tr_pred['y_true'] - tr_pred['y_pred']).to_numpy()

    pm = compute_point_metrics(pred_df)

    # Parametric PI: conditional (DEC narrows with visits)
    # model_name must contain 'DEC' for compute_parametric_pi to use conditional PI
    param_pi = compute_parametric_pi(
        pred_df, dist, fitted_all, model_name,
        test_subjects=test_s, seq_dim=seq_dim, cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper,
    )
    emp_pi = compute_empirical_pi(pred_df, tr_resid, model_name)

    if verbose:
        print_results(model_name, pm, eval_pi(param_pi))

    if plot:
        from .plotting import plot_diagnostics
        os.makedirs(cfg.output_dir, exist_ok=True)
        plot_diagnostics(pred_df, model_name, dist, fitted_all, cfg.output_dir,
                         cutoff=cfg.cutoff, cutoff_upper=cfg.cutoff_upper)

    return dict(
        model=model,
        state=trained_result['state'],  # RNN state unchanged
        dist_params=fitted_all, dist=dist,
        dec_params={k: v.data.cpu().clone() for k, v in dec_params.items()},
        predictions=pred_df, point_metrics=pm,
        param_pi=param_pi, empirical_pi=emp_pi,
        param_pi_results=eval_pi(param_pi), emp_pi_results=eval_pi(emp_pi),
        model_name=model_name, seed=seed,
        base_result=trained_result,
    )
