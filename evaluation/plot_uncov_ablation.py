"""
Dedicated figure for the uncertainty-loss ablation: LCPSN uncovON vs uncovOFF.

plot_sep8_results.py draws all five families at once, which answers "how does
LCPSN compare to the three models". This script answers the narrower question
the ablation was run for -- does `--lambda-uncov 0.001` actually help? -- by
plotting only the two LCPSN variants, plus the seed-noise floor the difference
has to clear to mean anything.

Input : results/unified_sep8_seed_avg.csv   (from plot_sep8_results.py)
Output: results/unified_sep8_ablation_uncov_{8deg,none}.png

Layout, one figure per evaluation condition (6 panels):
    row 1 -- vs SNR (T=200)          (a) RMSPE   (b) K accuracy   (c) delta +/- seed sigma
    row 2 -- vs snapshots (SNR=10dB) (d) RMSPE   (e) K accuracy   (f) delta +/- seed sigma

The delta panels carry the actual verdict: the bars are (OFF - ON), so positive
means ON is better, and the grey band is the pooled per-seed standard deviation.
A bar inside the band is indistinguishable from training randomness; a bar
outside it is a real effect.

The SNR panels start at 0 dB, the floor of the training SNR range; the -10..-2 dB
points stay in the CSV but are extrapolation and are not plotted. The curves carry
no +/-1 sigma band -- seed spread is read off the grey floor in (c),(f) instead.
Note this removes the out-of-training-range counting trade-off from view; it
survives as the K-accuracy table in report §36.5.

Colors are the categorical slots 1 and 2 of the reference palette, which pass
the CVD / normal-vision / contrast checks as a pair; marker shape and end-of-
line labels carry identity a second time so the figure survives greyscale.

Usage: PYTHONPATH=. python3 plot_uncov_ablation.py
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

ON, OFF = "LCPSN-uncovON", "LCPSN-uncovOFF"
STYLE = {                       # (color, marker, label)
    ON:  ("#2a78d6", "o", "uncovON  (λ=0.001)"),
    OFF: ("#eb6834", "v", "uncovOFF (λ=0)"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SNR_MIN = 0.0                   # plot floor = the bottom of the training SNR range.
                                # Below it the two variants decode different source
                                # counts anyway, which is why (c) was already clipped.


def load():
    """(min_sep, sweep, family) -> sorted list of per-x rows."""
    if not os.path.exists(SEED_AVG):
        return None
    acc = defaultdict(list)
    for r in csv.DictReader(open(SEED_AVG)):
        if r["family"] not in (ON, OFF):
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
    """Both variants as the seed mean, with end-of-line direct labels.

    No +/-1 sigma band: the seed spread is what (c),(f) are about, and drawing it
    twice -- once as haze here, once as the grey floor there -- only invites the
    reader to compare a band against a band it is not commensurate with.
    """
    ends = []
    for fam in (ON, OFF):
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
        ax.plot(x, mu, "-", color=c, marker=mk, ms=5, lw=2, label=lab,
                markeredgecolor="#fcfcfb", markeredgewidth=0.6)
        ends.append((x[-1], mu[-1], lab.split()[0], c))
    if logx:
        ax.set_xscale("log")
    _chrome(ax, xlabel, ylabel, title)

    # direct labels at the line ends; nudged apart where the curves converge
    span = np.ptp(ax.get_ylim())
    dy = [0.0] * len(ends)
    if len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < 0.06 * span:
        hi = 0 if ends[0][1] >= ends[1][1] else 1
        dy[hi], dy[1 - hi] = 7.0, -7.0
    for (x_end, y_end, text, c), off in zip(ends, dy):
        ax.annotate(text, (x_end, y_end), textcoords="offset points",
                    xytext=(7, off), va="center", fontsize=8, color=MUTED,
                    annotation_clip=False)


def _delta(ax, acc, ms, sweep, key, xlabel, ylabel, title, logx=False, xmin=None):
    """(OFF - ON) against the pooled per-seed sigma band. Positive = ON better.

    The grey band stays -- it is this panel's error display, the counterpart of
    the error bars on a ranking bar, not the per-curve haze dropped from _curves.

    xmin clips the panel to the range where the comparison is meaningful. It now
    coincides with SNR_MIN, but the reason is independent and worth keeping: below
    ~-2 dB the two variants decode a different number of sources (see the K
    accuracy panel), so their RMSPE is not measured on the same angles and the
    difference there is not a like-for-like advantage.
    """
    a = acc.get((ms, sweep, ON))
    b = acc.get((ms, sweep, OFF))
    if not a or not b:
        ax.set_visible(False)
        return
    pairs = [(ra, rb) for ra, rb in zip(a, b) if xmin is None or ra["x"] >= xmin]
    x = np.array([ra["x"] for ra, _ in pairs])
    d = np.array([rb[key] - ra[key] for ra, rb in pairs])
    sig = np.array([(ra[f"{key}_sd"] + rb[f"{key}_sd"]) / 2 for ra, rb in pairs])

    ax.fill_between(x, -sig, sig, color="#898781", alpha=0.22, lw=0,
                    label="±1 seed σ (pooled)")
    ax.axhline(0, color="#c3c2b7", lw=1)
    # stems rather than bars: same reading, and they stay correct on a log x-axis
    beyond = np.abs(d) > sig
    ax.vlines(x, 0, d, color=np.where(beyond, "#2a78d6", "#898781"), lw=2, zorder=3)
    ax.scatter(x, d, s=34, zorder=4, color=np.where(beyond, "#2a78d6", "#898781"),
               edgecolor="#fcfcfb", linewidth=0.6)
    if logx:
        ax.set_xscale("log")
    _chrome(ax, xlabel, ylabel, title)
    ax.set_ylim(min(-2.5 * sig.max(), d.min() * 1.15), d.max() * 1.35)
    n = int(beyond.sum())
    ax.annotate(f"ON better at {n}/{len(d)} points, all outside the seed-noise band"
                if n == len(d) else
                f"{n}/{len(d)} points outside the seed-noise band",
                xy=(0.5, 0.95), xycoords="axes fraction", fontsize=8.5, color=MUTED,
                ha="center", va="top")


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

        # --- row 1: SNR sweep (T=200)
        _curves(ax[0, 0], acc, ms, "snr", "rmspe", "SNR [dB]", "RMSPE [°]",
                f"(a) RMSPE vs SNR   (T=200, SNR ≥ {SNR_MIN:.0f} dB)")
        _curves(ax[0, 1], acc, ms, "snr", "kacc", "SNR [dB]", "K accuracy",
                f"(b) K accuracy vs SNR   (T=200, SNR ≥ {SNR_MIN:.0f} dB)")
        _delta(ax[0, 2], acc, ms, "snr", "rmspe", "SNR [dB]", "RMSPE(OFF) − RMSPE(ON) [°]",
               f"(c) ON's angular advantage   (SNR ≥ {SNR_MIN:.0f} dB)", xmin=SNR_MIN)
        # The shaded "training SNR range" span and the "outside the training range,
        # OFF counts sources better" callout both lived below 0 dB. That region is no
        # longer plotted -- the whole SNR axis is now inside the training range -- so
        # they are gone. The low-SNR counting trade-off is still on the record as the
        # K-accuracy table in report §36.5.

        # --- row 2: snapshot sweep (SNR=10 dB)
        _curves(ax[1, 0], acc, ms, "snapshot", "rmspe", "# snapshots T", "RMSPE [°]",
                "(d) RMSPE vs snapshots   (SNR=10 dB)", logx=True)
        _curves(ax[1, 1], acc, ms, "snapshot", "kacc", "# snapshots T", "K accuracy",
                "(e) K accuracy vs snapshots   (SNR=10 dB)", logx=True)
        _delta(ax[1, 2], acc, ms, "snapshot", "rmspe", "# snapshots T",
               "RMSPE(OFF) − RMSPE(ON) [°]",
               "(f) ON's advantage vs the seed-noise floor", logx=True)

        ax[0, 0].legend(fontsize=9, frameon=False, labelcolor=MUTED)
        ax[0, 2].legend(fontsize=8, frameon=False, labelcolor=MUTED, loc="lower right")

        cond = ("unconstrained — off-distribution" if ms == "none"
                else "8° separation — training-matched")
        fig.suptitle("LCPSN uncertainty-weighted covariance loss: ablation "
                     f"(λ=0.001 vs 0)   |   {cond}   |   mean over 3 training seeds "
                     "(seed σ = the grey band in (c),(f))",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(RESULTS, f"unified_sep8_ablation_uncov_{ms}.png")
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("saved:", out)


if __name__ == "__main__":
    main()
