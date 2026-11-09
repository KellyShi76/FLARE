"""Optional diagnostic plotting for FLARE.

matplotlib is imported lazily inside each function so that `import flare` (and the
core estimator / conformal paths) never require matplotlib. Install the extra with
`pip install flare[plot]`.
"""

import os
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

from .evaluate import (sample_error, kl_divergence_hist, fitted_error_pdf, HORIZONS)


def plot_diagnostics(pred_df, model_name, dist, dist_params, output_dir,
                     cutoff=-2.5, cutoff_upper=float("inf"), horizons=None):
    """3-panel diagnostic per horizon (true-vs-pred histogram, residual density
    with the fitted error law, and a Q-Q plot). Writes one PNG per horizon into
    `output_dir`. Requires the `[plot]` extra (matplotlib)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if horizons is None:
        horizons = HORIZONS

    mc = sample_error(dist, dist_params, cutoff=cutoff, cutoff_upper=cutoff_upper)

    for hlabel, hfilt in horizons:
        safe = hlabel.replace(" ", "_").replace(">=", "ge")
        sub = hfilt(pred_df)
        if len(sub) == 0:
            continue
        yt = sub['y_true'].to_numpy()
        yp = sub['y_pred'].to_numpy()
        resid = yt - yp

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.hist(yt, bins=40, alpha=0.5, density=True, label='True', color='steelblue')
        ax.hist(yp, bins=40, alpha=0.5, density=True, label='Pred', color='coral')
        rmse = np.sqrt(mean_squared_error(yt, yp))
        kl = kl_divergence_hist(yt, yp)
        ax.set_title(f"{model_name} | {hlabel}\nRMSE={rmse:.4f}  KL(true||pred)={kl:.4f}", fontsize=10)
        ax.legend(fontsize=7)

        ax = axes[1]
        ax.hist(resid, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        x = np.linspace(resid.min() - 0.5, resid.max() + 0.5, 500)
        pdf_total, pdf_comps = fitted_error_pdf(x, dist, dist_params, cutoff, cutoff_upper)
        if pdf_total is not None:
            ax.plot(x, pdf_total, 'r-', lw=2, label='Fitted')
            comp_colors = ['orange', 'green', 'purple', 'brown']
            for i, pdf_ci in enumerate(pdf_comps):
                ax.plot(x, pdf_ci, '--', color=comp_colors[i % len(comp_colors)],
                        lw=1.2, label=f'Comp {i+1}')
            ax.legend(fontsize=6)
        ax.axvline(cutoff, color='gray', ls=':', lw=1, alpha=0.7)
        if cutoff_upper < float('inf'):
            ax.axvline(cutoff_upper, color='gray', ls=':', lw=1, alpha=0.7)
        ax.set_title(f"Residuals eps=y-yhat\nskew={stats.skew(resid):.3f} kurt={stats.kurtosis(resid):.3f}",
                     fontsize=10)

        ax = axes[2]
        n = len(resid)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theo_q = np.quantile(mc, probs)
        emp_q = np.sort(resid)
        ax.scatter(theo_q, emp_q, s=2, alpha=0.3, color='steelblue')
        lims = [min(theo_q.min(), emp_q.min()), max(theo_q.max(), emp_q.max())]
        ax.plot(lims, lims, 'r-', lw=1.5)
        ax.set_xlabel("Theoretical"); ax.set_ylabel("Empirical")
        ax.set_title("Q-Q vs fitted dist", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_{safe}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
