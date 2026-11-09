"""Autoregressive RNN model."""

import torch
import torch.nn as nn

class SigmoidRNN(nn.Module):
    """Elman RNN with sigmoid activation: h_t = σ(W_ih x_t + W_hh h_{t-1} + b).
    API-compatible drop-in for nn.RNN(batch_first=True): returns (output, h_n)."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h0=None):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device) if h0 is None else h0.squeeze(0)
        outs = []
        for t in range(T):
            h = torch.sigmoid(self.i2h(x[:, t, :]) + self.h2h(h))
            outs.append(h)
        out = torch.stack(outs, dim=1)
        return out, h.unsqueeze(0)
        
class AutoregRNN(nn.Module):
    def __init__(self, seq_dim, dyn_dim, static_dim, hidden=16,
                 head_layers=None, dropout=0.2):
        """
        Args:
            seq_dim:     LSTM input dim (1 + n_dynamic)
            dyn_dim:     current dynamic features dim
            static_dim:  static features dim
            hidden:      LSTM hidden size
            head_layers: MLP layer sizes, default [256, 128, 64]
            dropout:     dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden
        self.seq_dim = seq_dim
        self.dyn_dim = dyn_dim
        self.static_dim = static_dim
        #self.rnn = nn.RNN(seq_dim, hidden, 1, batch_first=True)
        self.rnn = SigmoidRNN(seq_dim, hidden)

        if head_layers is None:
            head_layers = [32]

        head_in = hidden + dyn_dim + static_dim
        layers = []
        for h_out in head_layers:
            layers.extend([nn.Linear(head_in, h_out), nn.ReLU(), nn.Dropout(dropout)])
            head_in = h_out
        layers.append(nn.Linear(head_in, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, seq_x, curr_dyn, static_x, lengths, return_hidden=False):
        B = seq_x.size(0)
        h = torch.zeros(B, self.hidden_dim, device=seq_x.device)
        mask = lengths > 0
        if mask.any():
            # Run the cell on the padded tensor for non-empty rows.
            # Padded timesteps after `length-1` are computed but never read.
            out, _ = self.rnn(seq_x[mask])                              # (B', T, H)
            sub_len = lengths[mask]
            idx = (sub_len - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
            h[mask] = out.gather(1, idx).squeeze(1)
        y = self.head(torch.cat([h, curr_dyn, static_x], dim=1))
        if return_hidden:               # h = recurrent state fed to the head (0 when no history)
            return y, h
        return y


class AR1Linear(nn.Module):
    """Linear AR(1) mean model: ŷ_t = a + b·Y_{t-1}.

    Uses ONLY the last observed outcome. Column 0 of the (raw-scale) history
    matrix holds the lagged outcome Y; the last valid row is at index
    `lengths-1`. At visit 1 there is no history (`lengths==0`), so the lag is
    absent and ŷ = a (the intercept). Same forward signature as AutoregRNN so
    it plugs into the shared loss / predict_subjects / conformal pipeline; the
    returned hidden state is zero (hidden_norm == 0)."""

    def __init__(self, seq_dim=1, dyn_dim=0, static_dim=0,
                 hidden=1, head_layers=None, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden
        self.a = nn.Parameter(torch.zeros(1))   # intercept (also the visit-1 prediction)
        self.b = nn.Parameter(torch.zeros(1))   # AR(1) slope on Y_{t-1}

    def forward(self, seq_x, curr_dyn, static_x, lengths, return_hidden=False):
        B = seq_x.size(0)
        lag = torch.zeros(B, device=seq_x.device)
        mask = lengths > 0
        if mask.any():
            sub = seq_x[mask]                                   # (B', T, seq_dim)
            idx = (lengths[mask] - 1).clamp(min=0)             # last valid history row
            lag[mask] = sub[torch.arange(sub.size(0), device=sub.device), idx, 0]
        y = (self.a + self.b * lag).view(-1, 1)
        if return_hidden:
            return y, torch.zeros(B, self.hidden_dim, device=seq_x.device)
        return y


class AR1LongRunMean(nn.Module):
    """Linear AR(1) mean whose missing lag is the model-implied long-run mean."""

    def __init__(self, a, b, hidden=1):
        super().__init__()
        if not abs(float(b)) < 1:
            raise ValueError(f"AR(1) slope {b} has no finite long-run mean")
        self.hidden_dim = hidden
        self.register_buffer('a', torch.tensor(float(a)))
        self.register_buffer('b', torch.tensor(float(b)))

    def forward(self, seq_x, curr_dyn, static_x, lengths, return_hidden=False):
        B = seq_x.size(0)
        lag = (self.a / (1 - self.b)).expand(B).clone()
        mask = lengths > 0
        if mask.any():
            sub = seq_x[mask]
            idx = (lengths[mask] - 1).clamp(min=0)
            lag[mask] = sub[torch.arange(sub.size(0), device=sub.device), idx, 0]
        y = (self.a + self.b * lag).view(-1, 1)
        if return_hidden:
            return y, torch.zeros(B, self.hidden_dim, device=seq_x.device)
        return y


class MLPBaseline(nn.Module):
    """Feed-forward baseline on [current_dyn, static] only.

    No recurrent history and no recurrent state: the head sees the current-visit
    dynamic covariates concatenated with the static covariates. Head hidden
    widths match the RNN (cfg.rnn_head_layers). Same forward signature as
    AutoregRNN; `seq_x`/`lengths` are ignored and the returned hidden state is
    zero (hidden_norm == 0)."""

    def __init__(self, seq_dim, dyn_dim, static_dim, hidden=16,
                 head_layers=None, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden
        if head_layers is None:
            head_layers = [32]
        head_in = dyn_dim + static_dim
        layers = []
        for h_out in head_layers:
            layers.extend([nn.Linear(head_in, h_out), nn.ReLU(), nn.Dropout(dropout)])
            head_in = h_out
        layers.append(nn.Linear(head_in, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, seq_x, curr_dyn, static_x, lengths, return_hidden=False):
        B = seq_x.size(0)
        y = self.head(torch.cat([curr_dyn, static_x], dim=1))
        if return_hidden:
            return y, torch.zeros(B, self.hidden_dim, device=seq_x.device)
        return y
