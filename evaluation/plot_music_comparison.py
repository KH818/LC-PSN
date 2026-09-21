"""
Learned estimators vs classical MUSIC, on one canvas and in one table.

The learned block (LCPSN + the three reference models) and the classical block
(MUSIC on three covariance front-ends x three enumeration rules) are evaluated by
separate runs of run_eval.py, but on the SAME waveforms: the per-point RNG seed
and the batch schedule are fixed in `run_eval._sweep`, and asking for the analog
signal draws no extra random numbers, so run N and run N+1 see identical scenes.
That is what makes merging the two CSVs legitimate.

The SNR panels start at 0 dB, the floor of the learned rows' training SNR range;
the -10..-2 dB points stay in the CSVs but are not plotted. The curves carry no
+/-1 sigma band -- with six or seven series the bands read as fog, so the learned
rows' seed spread appears only as the error bars in (c),(f).

Inputs : results/unified_sep8_seed_avg.csv          (learned, 3-seed mean + seed sigma)
         results/unified_music_{snr,snapshot}_sweep.csv (classical, deterministic)
Outputs: results/unified_classical_vs_learned_{8deg,none}.png
         results/unified_all_{8deg,none}_summary.csv    (the paper table)

Layout, one figure per evaluation condition (6 panels):
    row 1 -- vs SNR (T=200)          (a) RMSPE  (b) K accuracy  (c) high-SNR ranking
    row 2 -- vs snapshots (SNR=10dB) (d) RMSPE  (e) K accuracy  (f) high-SNR ranking

Line panels carry the BLIND methods only -- everything that estimates K from the
data (learned heads, MDL). Putting an oracle-K curve on the same axes would
compare an estimator against a method that was told the answer; the oracle rows
appear in the ranking panels instead, hatched and labelled as such, where they
read as what they are: the angular-error floor each covariance front-end could
reach if enumeration were free. AIC stays out of the figure entirely and lives in
the summary CSV -- it is a robustness footnote for "did the criterion matter",
not a headline curve.

Colors: categorical slots 1/2/3/7 keep the assignments of the other sep8 figures
(LCPSN blue, TCN orange, Trans aqua, DA violet) so the figures agree with each
other; the two one-bit classical rows take slots 8 (red) and 4 (yellow), and the
full-precision anchor is neutral gray because it is a reference, not a competitor.
The six chromatic slots pass the lightness, chroma, CVD and normal-vision gates
in that adjacency order (worst adjacent CVD dE 9.2, normal-vision dE 20.8);
the contrast warning on aqua/yellow is answered the way the method requires --
every curve is direct-labelled and every bar carries its value as text. Dash
pattern repeats the learned/classical split, so the split survives greyscale
printing.

Usage: PYTHONPATH=. python3 plot_music_comparison.py
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
MUSIC_SNR = os.path.join(RESULTS, "unified_music_snr_sweep.csv")
MUSIC_SNAP = os.path.join(RESULTS, "unified_music_snapshot_sweep.csv")

LCPSN = "LCPSN-uncovON"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
SNR_FLOOR = 6.0                  # "high SNR" = the saturated region, as in report §33
SNR_MIN = 0.0                    # plot floor: the learned rows were trained from 0 dB
                                 # up, so -10..-2 dB is extrapolation for them. Applied
                                 # to the classical rows too, or the panels would compare
                                 # methods over different x ranges.

# series -> (color, marker, dash, label)
STYLE = {
    LCPSN:                  ("#2a78d6", "o", "-",  "LCPSN"),
    "TCN-MUSIC":            ("#eb6834", "D", "-",  "TCN"),
    "Trans-MUSIC":          ("#1baf7a", "s", "-",  "Trans"),
    "DA-MUSIC":             ("#4a3aa7", "^", "-",  "DA"),
    "1bMUSIC-naive/mdl":    ("#e34948", "v", "--", "1b-MUSIC naive"),
    "1bMUSIC-arcsine/mdl":  ("#eda100", "P", "--", "1b-MUSIC arcsine"),
    "FP-MUSIC/mdl":         ("#52514e", "*", "-.", "FP-MUSIC"),
    # oracle-K rows: bars only, drawn from the same hue with a hatch
    "1bMUSIC-naive/oracle":   ("#e34948", "v", "--", "1b-MUSIC naive (oracle K)"),
    "1bMUSIC-arcsine/oracle": ("#eda100", "P", "--", "1b-MUSIC arcsine (oracle K)"),
    "FP-MUSIC/oracle":        ("#52514e", "*", "-.", "FP-MUSIC (oracle K)"),
    # AIC rows: table only. Listed here so the loader keeps them; they are in no
    # plot order, because "MDL or AIC" is a footnote, not a curve worth an axis.
    "1bMUSIC-naive/aic":      ("#e34948", "v", "--", "1b-MUSIC naive (AIC)"),
    "1bMUSIC-arcsine/aic":    ("#eda100", "P", "--", "1b-MUSIC arcsine (AIC)"),
    "FP-MUSIC/aic":           ("#52514e", "*", "-.", "FP-MUSIC (AIC)"),
}
LEARNED = [LCPSN, "TCN-MUSIC", "Trans-MUSIC", "DA-MUSIC"]
CLASSICAL_BLIND = ["FP-MUSIC/mdl", "1bMUSIC-naive/mdl", "1bMUSIC-arcsine/mdl"]
ORACLE = ["FP-MUSIC/oracle", "1bMUSIC-naive/oracle", "1bMUSIC-arcsine/oracle"]
AIC = ["FP-MUSIC/aic", "1bMUSIC-naive/aic", "1bMUSIC-arcsine/aic"]
BAR_ORDER = LEARNED + CLASSICAL_BLIND + ORACLE
TABLE_ORDER = LEARNED + CLASSICAL_BLIND + AIC + ORACLE

# One figure per claim, rather than one figure carrying both. The paper makes two
# separate arguments -- "we beat the recent learned one-bit estimators" and "we beat
# classical MUSIC, including full-precision, under a proper enumeration rule" -- and
# a reader can only weigh one claim at a time. Splitting also drops each canvas to
# four series, back inside the validated categorical set.
#   * `classical` is the new figure: LCPSN against the MUSIC block.
#   * `all` keeps every curve on one canvas as a supplementary/appendix view.
# The learned-only figure already exists as plot_model_comparison.py.
FIGURES = {
    "classical": dict(
        lines=[LCPSN] + CLASSICAL_BLIND,
        bars=[LCPSN] + CLASSICAL_BLIND + ORACLE,
        subtitle="LCPSN vs classical MUSIC   |   blind rows use MDL; oracle-K hatched in (c),(f)"),
    "all": dict(
        lines=LEARNED + CLASSICAL_BLIND,
        bars=BAR_ORDER,
        subtitle="learned estimators and classical MUSIC on one canvas   |   "
                 "blind (unknown-K) curves; oracle-K hatched in (c),(f)"),
}


def load():
    """(min_sep, sweep, series) -> list of {x, rmspe, rmspe_sd, kacc, kacc_sd}, sorted by x."""
    acc = defaultdict(list)
    if os.path.exists(SEED_AVG):
        for r in csv.DictReader(open(SEED_AVG)):
            if r["family"] not in STYLE:
                continue
            acc[(r["min_sep"], r["sweep"], r["family"])].append(dict(
                x=float(r["x"]),
                rmspe=float(r["rmspe_deg_mean"]), rmspe_sd=float(r["rmspe_deg_std"]),
                kacc=float(r["k_accuracy_mean"]), kacc_sd=float(r["k_accuracy_std"]),
                mae=float(r["mae_deg_mean"]), mae_sd=float(r["mae_deg_std"]),
                sr5=float(r["sr5_mean"]), sr5_sd=float(r["sr5_std"]),
                maekok=float(r["mae_deg_kok_mean"]), maekok_sd=float(r["mae_deg_kok_std"])))
    for path in (MUSIC_SNR, MUSIC_SNAP):
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            if r["family"] not in STYLE:
                continue
            x = float(r["snr_db"]) if r["sweep"] == "snr" else float(r["snapshots"])
            acc[(r["min_sep"], r["sweep"], r["family"])].append(dict(
                x=x, rmspe=float(r["rmspe_deg"]), rmspe_sd=0.0,     # deterministic: no seed spread
                kacc=float(r["k_accuracy"]), kacc_sd=0.0,
                mae=float(r["mae_deg"]), mae_sd=0.0,
                sr5=float(r["sr5"]), sr5_sd=0.0,
                maekok=float(r["mae_deg_kok"]), maekok_sd=0.0))
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


def _curves(ax, acc, ms, sweep, key, xlabel, ylabel, title, order, logx=False, logy=False):
    """Series means only. The learned rows' seed spread lives in the (c),(f) bars;
    the classical rows are deterministic and have none to show."""
    ends = []
    for name in order:
        rows = acc.get((ms, sweep, name))
        if not rows:
            continue
        if sweep == "snr":
            rows = [r for r in rows if r["x"] >= SNR_MIN]
        if not rows:
            continue
        c, mk, ls, lab = STYLE[name]
        x = np.array([r["x"] for r in rows])
        mu = np.array([r[key] for r in rows])
        ax.plot(x, mu, ls, color=c, marker=mk, ms=4.5, lw=2, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=0.6)
        ends.append([x[-1], mu[-1], lab])
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _chrome(ax, xlabel, ylabel, title)

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.10 * (hi - lo))
    span = np.ptp(ax.get_ylim())
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < 0.055 * span:
            ends[i][1] = ends[i - 1][1] + 0.055 * span
    # seven curves converge in the K-accuracy panels, so the nudged stack can run
    # past the top of the axes — grow the axes to the stack, not the other way round
    if ends:
        lo, hi = ax.get_ylim()
        top = max(e[1] for e in ends)
        if top > hi - 0.03 * (hi - lo):
            ax.set_ylim(lo, top + 0.05 * (hi - lo))
    for x_end, y_lab, lab in ends:
        ax.annotate(lab, (x_end, y_lab), textcoords="offset points", xytext=(7, 0),
                    va="center", fontsize=8, color=MUTED, annotation_clip=False)


def high_snr_stats(acc, ms, key, names=None):
    """Mean over SNR >= SNR_FLOOR per series -> [(series, mean, sd)] (report §33 aggregation)."""
    out = []
    for name in (names or BAR_ORDER):
        rows = [r for r in acc.get((ms, "snr", name), []) if r["x"] >= SNR_FLOOR]
        if not rows:
            continue
        out.append((name, float(np.mean([r[key] for r in rows])),
                    float(np.mean([r[f"{key}_sd"] for r in rows]))))
    return out


def _summary(ax, acc, ms, key, xlabel, title, order, fmt="{:.2f}", better="low"):
    stats = high_snr_stats(acc, ms, key, order)
    if not stats:
        ax.set_visible(False)
        return
    stats.sort(key=lambda s: s[1], reverse=(better == "low"))     # best on top

    y = np.arange(len(stats))
    inside = better == "high"
    span = 1.0 if inside else max(m for _, m, _ in stats)
    for yi, (name, m, s) in zip(y, stats):
        c = STYLE[name][0]
        oracle = name.endswith("/oracle")
        ax.barh(yi, m, height=0.62, color=c, alpha=0.4 if oracle else 1.0,
                hatch="///" if oracle else None, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if s > 0:
            ax.errorbar(m, yi, xerr=s, fmt="none", ecolor=MUTED,
                        elinewidth=1.4, capsize=4, zorder=4)
        txt = f"{fmt.format(m)}" + (f"  ±{fmt.format(s)}" if s > 0 else "")
        if inside:
            ax.annotate(txt, (m - s - 0.02 * span, yi), va="center", ha="right",
                        fontsize=8.5, color="#ffffff" if not oracle else INK)
        else:
            ax.annotate(txt, (m + s + 0.02 * span, yi), va="center",
                        fontsize=8.5, color=MUTED)
    ax.set_yticks(y, [STYLE[n][3] for n, _, _ in stats], fontsize=8)
    ax.set_xlim(0, span if inside else span * 1.34)
    _chrome(ax, xlabel, "", title)
    ax.grid(axis="y", visible=False)


def write_summary_csv(acc, ms):
    """The paper table: high-SNR means for every method, blind and oracle, plus T=1000."""
    rmspe = {n: (m, s) for n, m, s in high_snr_stats(acc, ms, "rmspe", TABLE_ORDER)}
    kacc = {n: (m, s) for n, m, s in high_snr_stats(acc, ms, "kacc", TABLE_ORDER)}
    mae = {n: (m, s) for n, m, s in high_snr_stats(acc, ms, "mae", TABLE_ORDER)}
    sr5 = {n: (m, s) for n, m, s in high_snr_stats(acc, ms, "sr5", TABLE_ORDER)}
    maekok = {n: (m, s) for n, m, s in high_snr_stats(acc, ms, "maekok", TABLE_ORDER)}
    names = [n for n in TABLE_ORDER if n in rmspe]

    out = os.path.join(RESULTS, f"unified_all_{ms}_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "kind", "enumeration", "num_seeds",
                    "rmspe_deg_highsnr", "rmspe_deg_seed_sd",
                    "mae_deg_highsnr", "mae_deg_seed_sd",
                    "sr5_highsnr", "sr5_seed_sd",
                    "mae_deg_kok_highsnr",
                    "k_accuracy_highsnr", "k_accuracy_seed_sd",
                    "rmspe_deg_T1000", "mae_deg_T1000", "sr5_T1000",
                    "k_accuracy_T1000"])
        for n in names:
            if n not in rmspe:
                continue
            learned = n in LEARNED
            enum = ("learned K-head" if learned
                    else {"mdl": "MDL", "aic": "AIC", "oracle": "oracle K"}[n.split("/")[-1]])
            snap = [r for r in acc.get((ms, "snapshot", n), []) if r["x"] == 1000]
            w.writerow([n, "learned" if learned else "classical", enum,
                        3 if learned else 1,
                        f"{rmspe[n][0]:.4f}", f"{rmspe[n][1]:.4f}",
                        f"{mae[n][0]:.4f}", f"{mae[n][1]:.4f}",
                        f"{sr5[n][0]:.5f}", f"{sr5[n][1]:.5f}",
                        f"{maekok[n][0]:.4f}",
                        f"{kacc[n][0]:.5f}", f"{kacc[n][1]:.5f}",
                        f"{snap[0]['rmspe']:.4f}" if snap else "",
                        f"{snap[0]['mae']:.4f}" if snap else "",
                        f"{snap[0]['sr5']:.5f}" if snap else "",
                        f"{snap[0]['kacc']:.5f}" if snap else ""])
    print("saved:", out)


def main():
    acc = load()
    if not acc:
        print("no input CSVs — run:\n"
              "  PYTHONPATH=. python3 run_eval.py --checkpoint-set music\n"
              "  PYTHONPATH=. python3 plot_sep8_results.py")
        return

    for ms in ("8deg", "none"):
        if not any(k[0] == ms for k in acc):
            continue
        for tag, spec in FIGURES.items():
            render(acc, ms, tag, spec)
        write_summary_csv(acc, ms)


def render(acc, ms, tag, spec):
    lines, bars = spec["lines"], spec["bars"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.2))
    fig.patch.set_facecolor(SURFACE)
    for a in ax.ravel():
        a.set_facecolor(SURFACE)

    _curves(ax[0, 0], acc, ms, "snr", "rmspe", "SNR [dB]", "RMSPE [°]",
            f"(a) RMSPE vs SNR   (T=200, SNR ≥ {SNR_MIN:.0f} dB)", lines)
    _curves(ax[0, 1], acc, ms, "snr", "kacc", "SNR [dB]", "K accuracy",
            f"(b) K accuracy vs SNR   (T=200, SNR ≥ {SNR_MIN:.0f} dB)", lines)
    _summary(ax[0, 2], acc, ms, "rmspe", "RMSPE [°]  (lower is better)",
             f"(c) high-SNR (≥+{SNR_FLOOR:.0f} dB) mean RMSPE", bars)

    _curves(ax[1, 0], acc, ms, "snapshot", "rmspe", "# snapshots T", "RMSPE [°]",
            "(d) RMSPE vs snapshots   (SNR=10 dB)", lines, logx=True)
    _curves(ax[1, 1], acc, ms, "snapshot", "kacc", "# snapshots T", "K accuracy",
            "(e) K accuracy vs snapshots   (SNR=10 dB)", lines, logx=True)
    _summary(ax[1, 2], acc, ms, "kacc", "K accuracy  (higher is better)",
             f"(f) high-SNR (≥+{SNR_FLOOR:.0f} dB) mean K accuracy", bars,
             fmt="{:.3f}", better="high")

    ax[0, 0].legend(fontsize=8.5, frameon=False, labelcolor=MUTED,
                    ncol=2 if len(lines) > 4 else 1)

    cond = ("unconstrained — off-distribution" if ms == "none"
            else "8° separation — training-matched")
    fig.suptitle(f"{spec['subtitle']}   |   {cond}", fontsize=13, color=INK)
        # The K-hat decoding metric scores min(K_hat, K) matched angles, so a method
        # that under-counts is never charged for the sources it dropped -- usually the
        # hardest ones. An oracle-K bar can therefore sit to the RIGHT of its own blind
        # bar in (c): that gap is the metric's, not the enumeration rule's. Said here
        # rather than left for a reader to trip over.
    fig.text(0.5, 0.010,
             "RMSPE is scored on min(K̂, K) matched angles, so a blind method that under-counts is never charged for the "
             "sources it dropped — usually the hardest ones.\nThe oracle-K rows are scored on all K sources and their K "
             "accuracy is 1.0 by construction: they bound each front-end's angular error, they do not compete on enumeration.",
             ha="center", va="bottom", fontsize=8.5, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=[0, 0.052, 1, 0.96])
    out = os.path.join(RESULTS, f"unified_{tag}_vs_lcpsn_{ms}.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
