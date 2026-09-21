"""
LCPSN (3-seed mean) against the three reference models, training-matched sep8.

plot_sep8_results.py draws all five families including both ablation variants,
which makes the LCPSN-vs-rest question harder to read than it needs to be. This
script answers that question directly: one LCPSN curve -- uncovON, the default
recipe (lambda_uncov=0.001) -- versus TCN / Trans / DA, every curve a mean over
the same three training seeds. The seed spread is shown only in the right-column
bars (error bar + printed sigma), not as a band on the curves.

The SNR panels start at 0 dB, the floor of the training SNR range; the -10..-2 dB
points exist in the CSV but are extrapolation and are not plotted.

(The uncovON-vs-uncovOFF question has its own figure: plot_uncov_ablation.py.)

Input : results/unified_sep8_seed_avg.csv   (from plot_sep8_results.py)
Output: results/unified_sep8_models_{8deg,none}.png

Layout, one figure per evaluation condition (6 panels):
    row 1 -- vs SNR (T=200)          (a) RMSPE  (b) K accuracy  (c) high-SNR ranking
    row 2 -- vs snapshots (SNR=10dB) (d) RMSPE  (e) K accuracy  (f) high-SNR ranking

The right column summarises the left: mean over SNR >= +6 dB, with the per-seed
standard deviation as the error bar -- the same numbers as the report's §33
table. Those error bars are the point of the figure as much as the ranking is:
LCPSN's seed spread is ~0.02 deg against ~0.5 deg for the three models, so its
lead is not a seed accident.

Colors are categorical slots 1/2/3/7 of the reference palette (blue, orange,
aqua, violet), which pass the CVD, normal-vision and adjacency checks as a set;
LCPSN keeps the blue it has in the ablation figure. Marker shape and direct
labels repeat identity so the figure survives greyscale, and every bar carries
its value as text (the relief the aqua contrast warning requires).

Usage: PYTHONPATH=. python3 plot_model_comparison.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SEED_AVG = os.path.join(RESULTS, "unified_sep8_seed_avg.csv")

LCPSN = "LCPSN-uncovON"
STYLE = {                       # (color, marker, label)
    LCPSN:         ("#2a78d6", "o", "LCPSN"),
    "TCN-MUSIC":   ("#eb6834", "D", "TCN"),
    "Trans-MUSIC": ("#1baf7a", "s", "Trans"),
    "DA-MUSIC":    ("#4a3aa7", "^", "DA"),
}
ORDER = [LCPSN, "TCN-MUSIC", "Trans-MUSIC", "DA-MUSIC"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SNR_FLOOR = 6.0                 # "high SNR" = the saturated region, as in §33
SNR_MIN = 0.0                   # plot floor: training ran from 0 dB up, so the
                                # -10..-2 dB points are extrapolation, not results


def load():
    """(min_sep, sweep, family) -> sorted list of per-x rows."""
    if not os.path.exists(SEED_AVG):
        return None
    acc = defaultdict(list)
    for r in csv.DictReader(open(SEED_AVG)):
        if r["family"] not in STYLE:
            continue
        acc[(r["min_sep"], r["sweep"], r["family"])].append(
            dict(x=float(r["x"]),
                 rmspe=float(r["rmspe_deg_mean"]), rmspe_sd=float(r["rmspe_deg_std"]),
                 kacc=float(r["k_accuracy_mean"]), kacc_sd=float(r["k_accuracy_std"])))
    for k in acc:
        acc[k].sort(key=lambda d: d["x"])
    return acc


def _chrome(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, ls=":", lw=0.8, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")


def _curves(ax, acc, ms, sweep, key, xlabel, ylabel, title, logx=False):
    """All four families as the seed mean, with end-of-line labels.

    The seed spread is not drawn here: overlapping +/-1 sigma bands read as fog
    over four curves. It lives in the right-column bars instead, where it is a
    capped error bar with the number printed next to it.
    """
    ends = []
    for fam in ORDER:
        rows = acc.get((ms, sweep, fam))
        if not rows:
            continue
        if sweep == "snr":
            rows = [r for r in rows if r["x"] >= SNR_MIN]
        if not rows:
            continue
        c, mk, lab = STYLE[fam]
        x = np.array([r["x"] for r in rows])
        mu = np.array([r[key] for r in rows])
        ax.plot(x, mu, "-", color=c, marker=mk, ms=4.5, lw=2, label=lab,
                markeredgecolor="#fcfcfb", markeredgewidth=0.6)
        ends.append([x[-1], mu[-1], lab])
    if logx:
        ax.set_xscale("log")
    _chrome(ax, xlabel, ylabel, title)

    # headroom, so labels nudged apart at the top of the panel still fit
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.10 * (hi - lo))

    # direct labels at the line ends, nudged apart where curves converge
    span = np.ptp(ax.get_ylim())
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        gap = ends[i][1] - ends[i - 1][1]
        if gap < 0.055 * span:
            ends[i][1] = ends[i - 1][1] + 0.055 * span
    for x_end, y_lab, lab in ends:
        ax.annotate(lab, (x_end, y_lab), textcoords="offset points", xytext=(7, 0),
                    va="center", fontsize=8, color=MUTED, annotation_clip=False)


def _summary(ax, acc, ms, key, xlabel, title, fmt="{:.2f}", better="low"):
    """Mean over SNR >= SNR_FLOOR per family, ranked, with the seed sigma as error bar.

    Same aggregation as write_report_part4.high_snr_summary, so the bars and the
    report's §33 table cannot disagree.
    """
    stats = []
    for fam in ORDER:
        rows = [r for r in acc.get((ms, "snr", fam), []) if r["x"] >= SNR_FLOOR]
        if not rows:
            continue
        stats.append((fam,
                      float(np.mean([r[key] for r in rows])),
                      float(np.mean([r[f"{key}_sd"] for r in rows]))))
    if not stats:
        ax.set_visible(False)
        return
    stats.sort(key=lambda s: s[1], reverse=(better == "low"))   # best ends up on top

    y = np.arange(len(stats))
    colors = [STYLE[f][0] for f, _, _ in stats]
    ax.barh(y, [m for _, m, _ in stats], height=0.6, color=colors,
            edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    ax.errorbar([m for _, m, _ in stats], y, xerr=[s for _, _, s in stats], fmt="none",
                ecolor="#52514e", elinewidth=1.4, capsize=4, zorder=4)
    ax.set_yticks(y, [STYLE[f][2] for f, _, _ in stats])
    # K accuracy is bounded at 1, so its axis stops there and the values sit
    # inside the bars; RMSPE is unbounded, so its values sit outside.
    inside = better == "high"
    span = 1.0 if inside else max(m for _, m, _ in stats)
    for yi, (_, m, s) in zip(y, stats):
        txt = f"{fmt.format(m)}  ±{fmt.format(s)}"
        if inside:
            ax.annotate(txt, (m - s - 0.02 * span, yi), va="center", ha="right",
                        fontsize=8.5, color="#ffffff")
        else:
            ax.annotate(txt, (m + s + 0.02 * span, yi), va="center",
                        fontsize=8.5, color=MUTED)
    ax.set_xlim(0, span if inside else span * 1.32)
    _chrome(ax, xlabel, "", title)
    ax.grid(axis="y", visible=False)


def main():
    acc = load()
    if acc is None:
        print(f"missing {SEED_AVG} — run: PYTHONPATH=. python3 plot_sep8_results.py")
        return

    for ms in ("8deg", "none"):
        if not any(k[0] == ms for k in acc):
            continue
        fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))
        fig.patch.set_facecolor("#fcfcfb")
        for a in ax.ravel():
            a.set_facecolor("#fcfcfb")

        _curves(ax[0, 0], acc, ms, "snr", "rmspe", "SNR [dB]", "RMSPE [°]",
                f"(a) RMSPE vs SNR   (T=200, SNR ≥ {SNR_MIN:.0f} dB)")
        _curves(ax[0, 1], acc, ms, "snr", "kacc", "SNR [dB]", "K accuracy",
                f"(b) K accuracy vs SNR   (T=200, SNR ≥ {SNR_MIN:.0f} dB)")
        _summary(ax[0, 2], acc, ms, "rmspe", "RMSPE [°]  (lower is better)",
                 f"(c) high-SNR (≥+{SNR_FLOOR:.0f} dB) mean RMSPE   ± seed σ")

        _curves(ax[1, 0], acc, ms, "snapshot", "rmspe", "# snapshots T", "RMSPE [°]",
                "(d) RMSPE vs snapshots   (SNR=10 dB)", logx=True)
        _curves(ax[1, 1], acc, ms, "snapshot", "kacc", "# snapshots T", "K accuracy",
                "(e) K accuracy vs snapshots   (SNR=10 dB)", logx=True)
        _summary(ax[1, 2], acc, ms, "kacc", "K accuracy  (higher is better)",
                 f"(f) high-SNR (≥+{SNR_FLOOR:.0f} dB) mean K accuracy   ± seed σ",
                 fmt="{:.3f}", better="high")

        ax[0, 0].legend(fontsize=9, frameon=False, labelcolor=MUTED, ncol=2)

        cond = ("unconstrained — off-distribution" if ms == "none"
                else "8° separation — training-matched")
        fig.suptitle("LCPSN (λ_uncov=0.001) vs the three reference models   |   "
                     f"{cond}   |   mean over the same 3 training seeds "
                     "(seed σ shown as the bar error bars)",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(RESULTS, f"unified_sep8_models_{ms}.png")
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("saved:", out)


if __name__ == "__main__":
    main()
