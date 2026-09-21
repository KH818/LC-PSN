"""
Aggregate the 15-model sep8 evaluation over training seeds and plot it.

Input : results/unified_sep8_{snr,snapshot}_sweep.csv   (from run_eval.py --checkpoint-set sep8)
Output: results/unified_sep8_seed_avg.csv               one row per (family, condition, sweep, x)
        results/unified_sep8_compare_{8deg,none}.png    2x3

Five curves (each a mean over training seeds):
    Trans-MUSIC, TCN-MUSIC, DA-MUSIC, LCPSN-uncovON, LCPSN-uncovOFF

The two LCPSN curves are the --lambda-uncov 0.001 / 0 ablation, so the gap
between them is the effect of the uncertainty-weighted covariance loss.

Layout follows plot_model_comparison.py:
    row 1 -- vs SNR (T=200)          (a) RMSPE  (b) K accuracy  (c) high-SNR ranking
    row 2 -- vs snapshots (SNR=10dB) (d) RMSPE  (e) K accuracy  (f) high-SNR ranking

The seed spread is not drawn as a band on the curves -- five overlapping +/-1 std
bands read as fog. It appears only in the right-column bars, as a capped error bar
with the number printed beside it.

The SNR panels start at 0 dB, the floor of the training SNR range; the -10..-2 dB
points stay in the CSV but are extrapolation and are not plotted.

Note the CSV written here keeps the full SNR range -- the floor is plot-time only,
so plot_model_comparison.py / plot_uncov_ablation.py still see every point.

Usage: PYTHONPATH=. python plot_sep8_results.py
"""
import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

STYLE = {
    "TCN-MUSIC":      ("#d62728", "D"),
    "Trans-MUSIC":    ("#2ca02c", "s"),
    "DA-MUSIC":       ("#1f77b4", "^"),
    "LCPSN-uncovON":  ("#9467bd", "o"),
    "LCPSN-uncovOFF": ("#e377c2", "v"),
}
ORDER = ["LCPSN-uncovON", "LCPSN-uncovOFF", "TCN-MUSIC", "Trans-MUSIC", "DA-MUSIC"]
METRICS = ("rmspe_rad", "rmspe_deg", "k_accuracy",
           "mae_deg", "sr5", "rmspe_deg_kok", "mae_deg_kok")
SNR_MIN = 0.0                   # plot floor: training ran from 0 dB up
SNR_FLOOR = 6.0                 # "high SNR" = the saturated region, as in §33


def load(path, xkey, sweep):
    """(min_sep, family, x) -> {metric: [per-seed values]}"""
    acc = defaultdict(lambda: defaultdict(list))
    if not os.path.exists(path):
        return acc
    for r in csv.DictReader(open(path)):
        key = (r["min_sep"], r.get("family") or r["model"], float(r[xkey]))
        for m in METRICS:
            acc[key][m].append(float(r[m]))
        acc[key]["seeds"].append(r.get("seed", ""))
    return acc


def aggregate(acc, sweep, xname):
    rows = []
    for (ms, fam, x), d in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        row = dict(sweep=sweep, min_sep=ms, family=fam, x=x, xname=xname,
                   num_seeds=len(d["seeds"]),
                   seeds=";".join(sorted(s for s in d["seeds"] if s != "")))
        for m in METRICS:
            v = np.asarray(d[m], dtype=np.float64)
            row[f"{m}_mean"] = float(v.mean())
            row[f"{m}_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        rows.append(row)
    return rows


def _panel(ax, rows, ms, sweep, metric, xlabel, ylabel, title):
    """Seed mean per family. No spread band -- that is the job of _summary."""
    sel = [r for r in rows if r["min_sep"] == ms and r["sweep"] == sweep]
    if sweep == "snr":
        sel = [r for r in sel if r["x"] >= SNR_MIN]
    if not sel:
        ax.set_visible(False)
        return
    fams = {r["family"] for r in sel}
    for fam in [f for f in ORDER if f in fams] + sorted(fams - set(ORDER)):
        fr = sorted([r for r in sel if r["family"] == fam], key=lambda r: r["x"])
        c, mk = STYLE.get(fam, ("#7f7f7f", "x"))
        x = np.array([r["x"] for r in fr])
        mu = np.array([r[f"{metric}_mean"] for r in fr])
        ax.plot(x, mu, "-", color=c, marker=mk, ms=5, lw=1.8, label=fam)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)
    # headroom so the legend does not land on top of the highest curve
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    ax.legend(fontsize=8)


def _summary(ax, rows, ms, metric, xlabel, title, fmt="{:.4f}", better="low"):
    """Mean over SNR >= SNR_FLOOR per family, ranked, seed std as the error bar.

    Same construction as plot_model_comparison._summary, so the two figures'
    right-hand columns are read the same way. Bars carry their value as text
    because the error bars are far smaller than the between-family gaps.
    """
    stats = []
    for fam in ORDER:
        fr = [r for r in rows if r["min_sep"] == ms and r["sweep"] == "snr"
              and r["family"] == fam and r["x"] >= SNR_FLOOR]
        if not fr:
            continue
        stats.append((fam,
                      float(np.mean([r[f"{metric}_mean"] for r in fr])),
                      float(np.mean([r[f"{metric}_std"] for r in fr]))))
    if not stats:
        ax.set_visible(False)
        return
    stats.sort(key=lambda s: s[1], reverse=(better == "low"))   # best ends up on top

    y = np.arange(len(stats))
    ax.barh(y, [m for _, m, _ in stats], height=0.6,
            color=[STYLE.get(f, ("#7f7f7f", "x"))[0] for f, _, _ in stats],
            edgecolor="white", linewidth=1.5, zorder=3)
    ax.errorbar([m for _, m, _ in stats], y, xerr=[s for _, _, s in stats],
                fmt="none", ecolor="#333333", elinewidth=1.4, capsize=4, zorder=4)
    ax.set_yticks(y, [f for f, _, _ in stats], fontsize=8)

    # K accuracy is bounded at 1, so its axis stops there and the values sit
    # inside the bars; RMSPE is unbounded, so its values sit outside.
    inside = better == "high"
    span = 1.0 if inside else max(m for _, m, _ in stats)
    for yi, (_, m, s) in zip(y, stats):
        txt = f"{fmt.format(m)}  ±{fmt.format(s)}"
        if inside:
            ax.annotate(txt, (m - s - 0.02 * span, yi), va="center", ha="right",
                        fontsize=8, color="white")
        else:
            ax.annotate(txt, (m + s + 0.02 * span, yi), va="center", fontsize=8)
    ax.set_xlim(0, span if inside else span * 1.38)
    ax.set_xlabel(xlabel); ax.set_title(title)
    ax.grid(True, axis="x", ls=":", alpha=0.5); ax.set_axisbelow(True)


def main():
    snr = load(os.path.join(RESULTS, "unified_sep8_snr_sweep.csv"), "snr_db", "snr")
    snap = load(os.path.join(RESULTS, "unified_sep8_snapshot_sweep.csv"), "snapshots", "snapshot")
    if not snr and not snap:
        print("no sep8 result CSVs — run: "
              "PYTHONPATH=. python run_eval.py --checkpoint-set sep8"); return

    rows = aggregate(snr, "snr", "snr_db") + aggregate(snap, "snapshot", "snapshots")
    out_csv = os.path.join(RESULTS, "unified_sep8_seed_avg.csv")
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    n_seeds = max(r["num_seeds"] for r in rows)
    print(f"[OK] wrote {out_csv} ({len(rows)} rows, up to {n_seeds} seeds per point)")

    for ms in sorted({r["min_sep"] for r in rows}):
        fig, ax = plt.subplots(2, 3, figsize=(17.5, 9.5))
        _panel(ax[0, 0], rows, ms, "snr", "rmspe_rad", "SNR [dB]", "RMSPE [rad]",
               f"(a) RMSPE vs SNR  (T=200, SNR >= {SNR_MIN:.0f} dB)")
        _panel(ax[0, 1], rows, ms, "snr", "k_accuracy", "SNR [dB]", "K accuracy",
               f"(b) K accuracy vs SNR  (T=200, SNR >= {SNR_MIN:.0f} dB)")
        _summary(ax[0, 2], rows, ms, "rmspe_rad", "RMSPE [rad]  (lower is better)",
                 f"(c) high-SNR (>=+{SNR_FLOOR:.0f} dB) mean RMSPE  +/- seed std")
        _panel(ax[1, 0], rows, ms, "snapshot", "rmspe_rad", "# snapshots", "RMSPE [rad]",
               "(d) RMSPE vs snapshots  (SNR=10dB)")
        _panel(ax[1, 1], rows, ms, "snapshot", "k_accuracy", "# snapshots", "K accuracy",
               "(e) K accuracy vs snapshots  (SNR=10dB)")
        _summary(ax[1, 2], rows, ms, "k_accuracy", "K accuracy  (higher is better)",
                 f"(f) high-SNR (>=+{SNR_FLOOR:.0f} dB) mean K accuracy  +/- seed std",
                 fmt="{:.3f}", better="high")
        sep = "unconstrained (off-distribution)" if ms == "none" else f"{ms} (training-matched)"
        fig.suptitle("sep8 1-bit unknown-K comparison (K-hat decoding, RMSPE in rad) | "
                     f"min-sep: {sep} | mean over {n_seeds} training seeds "
                     "(seed std = the bar error bars)",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        tag = "none" if ms == "none" else ms.replace(".", "")
        out = os.path.join(RESULTS, f"unified_sep8_compare_{tag}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print("saved:", out)


if __name__ == "__main__":
    main()
