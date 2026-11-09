"""Persistence helpers: save/load run results so reruns can skip retraining.

Layout under output_dir:
    checkpoints/<model_name>.pt   # full result dict + state_dict + dims
    shap/<model_name>.pt          # SHAP values + feature names (optional)
    loco_results.csv              # already written by run_loco

Saved checkpoints are device-agnostic (state_dict tensors on CPU);
load_result places the reconstructed model on the requested device.
"""

import os
import torch

from .model import AutoregRNN


CKPT_SUBDIR = 'checkpoints'
SHAP_SUBDIR = 'shap'
LOCO_CSV = 'loco_results.csv'


def checkpoint_path(model_name, output_dir):
    return os.path.join(output_dir, CKPT_SUBDIR, f'{model_name}.pt')


def checkpoint_exists(model_name, output_dir):
    return os.path.exists(checkpoint_path(model_name, output_dir))


def save_result(result, output_dir, dims=None, cfg=None):
    """Persist a run_model / run_dec result.

    state_dict tensors are moved to CPU before saving. Dims and a cfg snapshot
    are stored so the model can be reconstructed standalone.
    """
    ckpt_dir = os.path.join(output_dir, CKPT_SUBDIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    name = result['model_name']

    state = result.get('state')
    if state is not None:
        state = {k: v.detach().cpu().clone() for k, v in state.items()}

    payload = {
        'model_name':       result['model_name'],
        'dist':             result.get('dist'),
        'state':            state,
        'dist_params':      result.get('dist_params'),
        'dec_params':       result.get('dec_params'),
        'predictions':      result.get('predictions'),
        'point_metrics':    result.get('point_metrics'),
        'param_pi':         result.get('param_pi'),
        'empirical_pi':     result.get('empirical_pi'),
        'param_pi_results': result.get('param_pi_results'),
        'emp_pi_results':   result.get('emp_pi_results'),
        'seed':             result.get('seed'),
    }
    if dims is not None:
        payload['dims'] = {k: v for k, v in dims.items()}
    if cfg is not None:
        payload['cfg_snapshot'] = {
            'rnn_hidden':       cfg.rnn_hidden,
            'rnn_head_layers':  list(cfg.rnn_head_layers),
            'cutoff':           cfg.cutoff,
            'cutoff_upper':     getattr(cfg, 'cutoff_upper', float('inf')),
            'dynamic_cols':     list(cfg.dynamic_cols),
            'current_dyn_cols': list(cfg.current_dyn_cols),
            'categorical_cols': list(cfg.categorical_cols),
            'target_col':       cfg.target_col,
            'time_scale':       cfg.time_scale,
        }

    path = checkpoint_path(name, output_dir)
    torch.save(payload, path)
    return path


def load_result(model_name, output_dir, device='cpu'):
    """Load a saved result and reconstruct the live model.

    Returns None if no checkpoint exists. Returned dict shape matches a
    run_model output; `model` is an AutoregRNN on `device` (or None for
    AR(1)-style results without state).
    """
    path = checkpoint_path(model_name, output_dir)
    if not os.path.exists(path):
        return None

    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload.get('state')
    dims = payload.get('dims')
    cfg_snap = payload.get('cfg_snapshot') or {}

    model = None
    if state is not None and dims is not None:
        rnn_hidden = cfg_snap.get('rnn_hidden', 16)
        rnn_head_layers = cfg_snap.get('rnn_head_layers', [64, 32])
        model = AutoregRNN(
            dims['seq_dim'], dims['dyn_dim'], dims['static_dim'],
            hidden=rnn_hidden, head_layers=rnn_head_layers,
        ).to(device)
        model.load_state_dict(state)
        model.eval()

    return dict(
        model=model,
        state=state,
        dist_params=payload.get('dist_params'),
        dist=payload.get('dist'),
        dec_params=payload.get('dec_params'),
        predictions=payload.get('predictions'),
        point_metrics=payload.get('point_metrics'),
        param_pi=payload.get('param_pi'),
        empirical_pi=payload.get('empirical_pi'),
        param_pi_results=payload.get('param_pi_results'),
        emp_pi_results=payload.get('emp_pi_results'),
        model_name=model_name,
        seed=payload.get('seed'),
        dims=dims,
        cfg_snapshot=cfg_snap,
    )


def list_checkpoints(output_dir):
    """List saved model_name strings in output_dir/checkpoints/."""
    ckpt_dir = os.path.join(output_dir, CKPT_SUBDIR)
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(f[:-3] for f in os.listdir(ckpt_dir) if f.endswith('.pt'))


# ------------------ SHAP cache ------------------

def save_shap(model_name, shap_agg, feat_names, output_dir):
    sdir = os.path.join(output_dir, SHAP_SUBDIR)
    os.makedirs(sdir, exist_ok=True)
    torch.save({'shap_agg': shap_agg, 'feat_names': list(feat_names)},
               os.path.join(sdir, f'{model_name}.pt'))


def load_shap(model_name, output_dir):
    p = os.path.join(output_dir, SHAP_SUBDIR, f'{model_name}.pt')
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location='cpu', weights_only=False)


def shap_exists(model_name, output_dir):
    return os.path.exists(os.path.join(output_dir, SHAP_SUBDIR, f'{model_name}.pt'))


# ------------------ LOCO cache ------------------

def loco_exists(output_dir):
    return os.path.exists(os.path.join(output_dir, LOCO_CSV))


def load_loco(output_dir):
    import pandas as pd
    p = os.path.join(output_dir, LOCO_CSV)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)