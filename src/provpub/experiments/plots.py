"""Figures 2-4 of the paper, generated from result CSVs with matplotlib.

Every figure written from a run whose run_metadata.json says run_kind='smoke'
carries the label "synthetic smoke-test output, not a paper result".
Colours: first three slots of the dataviz reference categorical palette
(validated all-pairs), plus distinct markers and line styles so that series
are not identified by colour alone."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..utils.io import SYNTHETIC_LABEL

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "lines.linewidth": 1.6, "legend.frameon": False, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def mean_ci(df, x, y):
    g = df.groupby(x)[y]
    m = g.mean()
    n = g.count()
    h = stats.t.ppf(0.975, (n - 1).clip(lower=1)) * g.std(ddof=1) / np.sqrt(n)  # 95% t-interval over seeds
    return m.index.to_numpy(), m.to_numpy(), h.fillna(0).to_numpy()


def fig_eps(RES, OUT, label=None):
    e = pd.read_csv(os.path.join(RES, "e2_eps_sweep.csv"))
    p = pd.read_csv(os.path.join(RES, "e2_privacy.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharex=True)
    panels = [("mia_auc", "A1 membership AUC", 0.5), ("aia_acc", "A2 size-class accuracy", None),
              ("utility_thr", "Threshold outcomes kept", None)]
    base2 = p[p.config == "RAW"]["aia_base"].mean()
    for ax, (col, title, _chance) in zip(axes, panels):
        for unit, c, mk, ls, lab in (("record", C1, "o", "-", "record level"), ("bucket", C2, "s", "--", "bucket level"),
                                     ("bucket-group", C3, "^", "-.", "bucket, input-set scope")):
            x, m, h = mean_ci(e[e.unit == unit], "eps", col)
            ax.errorbar(x, m, yerr=h, color=c, marker=mk, ms=4, ls=ls, capsize=2, label=lab)
        if col == "mia_auc":
            nos = p[p.config == "FULL-nosize"]["mia_auc"].mean()
            ax.axhline(nos, color=MUTED, ls=":", lw=1.4, label="sizes suppressed")
            ax.axhline(0.5, color=MUTED, lw=0.8)
            ax.text(0.98, 0.51, "chance", color=MUTED, fontsize=8, va="bottom", ha="right",
                    transform=ax.get_yaxis_transform())
        if col == "aia_acc":
            ax.axhline(base2, color=MUTED, lw=0.8)
            ax.text(0.98, base2 - 0.02, "majority", color=MUTED, fontsize=8, va="top", ha="right",
                    transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel(r"$\varepsilon$ per partition")
        ax.set_ylim(0, 1.05 if col != "mia_auc" else 1.02)
        if col == "mia_auc":
            ax.set_ylim(0.45, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _label(fig, label)
    fig.savefig(os.path.join(OUT, "fig_eps_sweep.pdf"))
    plt.close(fig)


def fig_delta(RES, OUT, label=None):
    d = pd.read_csv(os.path.join(RES, "e2_delta_sweep.csv"))
    p = pd.read_csv(os.path.join(RES, "e2_privacy.csv"))
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.3))
    x, m, h = mean_ci(d, "delta", "analyst_acc")
    ax.errorbar(x, m, yerr=h, color=C1, marker="o", ms=4, capsize=2, label="VG+TC+BS")
    vg = p[p.config == "VG"]["analyst_acc"].mean()
    ax.axhline(vg, color=C2, ls="--", lw=1.4, label="VG only (absolute times kept)")
    base = d["analyst_base"].mean()
    ax.axhline(base, color=MUTED, lw=0.8)
    ax.text(0.98, base - 0.02, "majority", color=MUTED, fontsize=8, va="top", ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_title("A3 analyst re-identification accuracy", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"coarsening width $\Delta$ (s)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _label(fig, label)
    fig.savefig(os.path.join(OUT, "fig_delta_sweep.pdf"))
    plt.close(fig)


def fig_scaling(RES, OUT, label=None):
    files = sorted(x for x in os.listdir(RES) if x.startswith("e4_scaling_") and x.endswith(".csv"))
    if not files:
        return
    d = pd.read_csv(os.path.join(RES, files[0]))
    if "error" in d.columns:
        d = d[d["error"].isna()]
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    styles = {("W2", 1): (C1, "o", "-"), ("W2", 2): (C1, "o", "--"), ("W3", 1): (C2, "s", "-"), ("W3", 2): (C2, "s", "--")}
    for (wl, w), g in d.groupby(["workload", "workers"]):
        c, mk, ls = styles.get((wl, w), (C1 if wl == "W2" else C2, "o" if wl == "W2" else "s",
                                         ":" if int(w) <= 4 else "-."))
        g = g.assign(mw=g.get("peak_rss_mb_worker_max", np.nan))
        agg = g.groupby("records").agg(t=("t_sanitise", "median"), tmin=("t_sanitise", "min"), tmax=("t_sanitise", "max"),
                                       m=("peak_rss_mb_sanitise", "max"), mc=("mw", "max")).reset_index()
        lab = f"{wl}, {w} worker{'s' if w > 1 else ''}"
        axes[0].errorbar(agg.records, agg.t, yerr=[agg.t - agg.tmin, agg.tmax - agg.t], color=c, marker=mk, ms=4,
                         ls=ls, capsize=2, label=lab)
        mem = agg.m if w == 1 else agg.mc
        axes[1].plot(agg.records, mem, color=c, marker=mk, ms=4, ls=ls, label=lab)
    xt = sorted(d.groupby("target_records")["records"].first().dropna().tolist())
    xl = [f"{v / 1e6:.2g}M" for v in xt]
    for ax, t in zip(axes, ("Sanitisation wall time (s)", "Peak memory per process (MB)")):
        lo, hi = ax.get_ylim()
        yt = _ticks_125(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(xt)); ax.xaxis.set_major_formatter(FixedFormatter(xl))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(FixedFormatter([f"{v:.3g}" for v in yt]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("PROV records")
        ax.set_title(t, fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)
    fig.tight_layout()
    _label(fig, label)
    fig.savefig(os.path.join(OUT, "fig_scaling.pdf"))
    plt.close(fig)


def _ticks_125(lo: float, hi: float) -> list[float]:
    """1-2-5 ticks inside [lo, hi] for a log axis (at least two)."""
    lo, hi = max(lo, 1e-6), max(hi, 2e-6)
    out = [m * 10.0 ** e for e in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
           for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
    return out if len(out) >= 2 else [lo, hi]


def _label(fig, label) -> None:
    if label:
        fig.subplots_adjust(top=min(fig.subplotpars.top, 0.84))
        fig.text(0.5, 0.995, label, ha="center", va="top", fontsize=8, color="#b42318", fontweight="bold")


def make_all(res_dir: str, out_dir: str, run_kind: str | None = None) -> list[str]:
    """Generate every figure whose input CSVs exist. Smoke runs (or runs of
    unknown kind) are labelled as synthetic smoke-test output."""
    os.makedirs(out_dir, exist_ok=True)
    label = SYNTHETIC_LABEL if run_kind in (None, "smoke") else None
    made = []
    for fn, need, name in ((fig_eps, "e2_eps_sweep.csv", "fig_eps_sweep.pdf"),
                           (fig_delta, "e2_delta_sweep.csv", "fig_delta_sweep.pdf"),
                           (fig_scaling, None, "fig_scaling.pdf")):
        if need and not os.path.exists(os.path.join(res_dir, need)):
            continue
        if need is None and not any(x.startswith("e4_scaling_") for x in os.listdir(res_dir)):
            continue
        fn(res_dir, out_dir, label)
        made.append(os.path.join(out_dir, name))
    return made
