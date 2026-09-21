"""
Unified evaluation harness — ONE run produces every curve for all four models.

Sweeps (each computed for BOTH min-sep conditions: unconstrained + 8 deg):
  1. SNR sweep      : snapshots fixed at T=200,  SNR in {-10..20 step 2}
  2. Snapshot sweep : SNR fixed at 10 dB,        T in {50,100,200,300,400,600,800,1000}

Metric per (model, condition, x-value):
  * k_accuracy   = P(K_hat == K_true)
  * rmspe_rad    = permutation-min RMSPE with K-hat decoding  (also stored as deg)
  * mae_deg      = matched MAE at the SAME permutation that minimises RMSPE
  * sr5          = P(per-scene matched RMSE < 5 deg)          success rate
  * *_kok        = the same, restricted to scenes with K_hat == K_true

Checkpoint sets (--checkpoint-set):
  sep8  : training-matched 8-deg study, 15 models      -> results/unified_sep8_*.csv
          Trans/TCN/DA x seeds{0,1,2} + LCPSN-uncov{ON,OFF} x seeds{0,1,2}
          `family` groups the seeds of one model; average over `seed` to plot.

Output: two tidy CSVs (long format) under results/
Columns: model, family, seed, sweep, min_sep, snr_db, snapshots, num_tests,
         num_kok, k_accuracy, rmspe_rad, rmspe_deg, mae_deg, sr5,
         rmspe_deg_kok, mae_deg_kok

Usage:
  PYTHONPATH=. python run_eval.py                              # Part II, 4 models
  PYTHONPATH=. python run_eval.py --checkpoint-set sep8        # 15-model sep8 study
  PYTHONPATH=. python run_eval.py --smoke                      # random weights, tiny sizes
"""
import os
import csv
import argparse
import numpy as np
import torch

import signal_gen as sg
import metrics as mx
import adapters as ad

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
KS = [2, 3, 4, 5]
MIN_SEPS = [None, 8.0]                 # default: both conditions, one run
SEED = 2026


def _run_point(adapters, snr_db, snapshots, min_sep, num_tests, batch, base_seed):
    """Evaluate EVERY model at one (SNR, T, min_sep) point.

    The signal batch is generated once and handed to all adapters, so the
    models are compared on literally the same waveforms (not merely on the same
    RNG seed) and generation cost does not scale with the number of models.
    """
    rng = np.random.default_rng(base_seed)
    acc = {a.name: dict(k_true=[], k_pred=[], doa_true=[], doa_pred=[]) for a in adapters}
    # Extras some adapters ask for via `.wants` (classical MUSIC block):
    #   analog -> unquantized snapshots (full-precision reference rows)
    #   ktrue  -> true source count     (oracle-K rows)
    #   ctx    -> per-batch scratch dict for sharing covariances between rows
    # Requesting the analog signal draws no extra random numbers, so the
    # waveforms at a given point are identical with or without this block.
    need_analog = any("analog" in getattr(a, "wants", ()) for a in adapters)
    done = 0
    while done < num_tests:
        b = min(batch, num_tests - done)
        ktrue = np.array([KS[i % len(KS)] for i in range(b)], dtype=np.int64)   # balanced K
        out = sg.make_onebit_batch(ktrue, snr_db, snapshots, rng,
                                   min_sep_deg=min_sep, return_analog=need_analog)
        analog = out[3] if need_analog else None
        x1bit, truth, ktrue = out[0], out[1], out[2]
        truth_list = [truth[i][np.abs(truth[i] - np.pi) > 1e-6] for i in range(b)]
        ctx = {}
        extras = dict(analog=analog, ktrue=ktrue, ctx=ctx)
        for a in adapters:
            kw = {k: extras[k] for k in getattr(a, "wants", ())}
            khat, doa_pred = a.predict(x1bit, **kw)
            d = acc[a.name]
            for i in range(b):
                d["k_true"].append(len(truth_list[i]))
                d["k_pred"].append(int(khat[i]))
                d["doa_true"].append(np.sort(truth_list[i]))
                d["doa_pred"].append(np.sort(np.asarray(doa_pred[i])))
        done += b
    return {a.name: mx.summarize(acc[a.name]["k_true"], acc[a.name]["k_pred"],
                                 acc[a.name]["doa_true"], acc[a.name]["doa_pred"])
            for a in adapters}


def _sweep(adapters, points, sweep_name, num_tests, batch, min_seps=None):
    """points: list of dict(snr_db=, snapshots=). Returns list of CSV rows."""
    live = [a for a in adapters if a.available]
    rows = []
    for min_sep in (MIN_SEPS if min_seps is None else min_seps):
        tag = "none" if min_sep is None else f"{min_sep:g}deg"
        for p in points:
            # deterministic per-point seed so every point is reproducible
            base_seed = SEED + int(p["snr_db"]) * 131 + int(p["snapshots"]) + (0 if min_sep is None else 777)
            res = _run_point(live, p["snr_db"], p["snapshots"], min_sep,
                             num_tests, batch, base_seed)
            for a in live:
                m = res[a.name]
                rows.append(dict(model=a.name, family=getattr(a, "family", a.name),
                                 seed="" if getattr(a, "seed", None) is None else a.seed,
                                 sweep=sweep_name, min_sep=tag,
                                 snr_db=p["snr_db"], snapshots=p["snapshots"],
                                 num_tests=m["num_tests"], num_kok=m["num_kok"],
                                 k_accuracy=m["k_accuracy"],
                                 joint_success=m["joint_success"],
                                 mean_pred_len=m["mean_pred_len"],
                                 mean_true_len=m["mean_true_len"],
                                 short_pred_rate=m["short_pred_rate"],
                                 rmspe_rad=m["rmspe_rad"], rmspe_deg=m["rmspe_deg"],
                                 mae_deg=m["mae_deg"], sr5=m["sr5"],
                                 rmspe_deg_kok=m["rmspe_deg_kok"],
                                 mae_deg_kok=m["mae_deg_kok"]))
                print(f"  [{sweep_name}|{tag}] {a.name:22s} "
                      f"SNR={p['snr_db']:+3d} T={p['snapshots']:4d} "
                      f"Kacc={m['k_accuracy']:.3f} RMSPE={m['rmspe_deg']:.3f}d "
                      f"MAE={m['mae_deg']:.3f}d SR@5={m['sr5']:.3f}", flush=True)
    return rows


FIELDS = ["model", "family", "seed", "sweep", "min_sep", "snr_db", "snapshots",
          "num_tests", "num_kok", "k_accuracy", "joint_success",
          "mean_pred_len", "mean_true_len", "short_pred_rate",
          "rmspe_rad", "rmspe_deg", "mae_deg", "sr5",
          "rmspe_deg_kok", "mae_deg_kok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="random weights + tiny sizes to validate plumbing")
    ap.add_argument("--num-tests", type=int, default=None)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint-set", choices=["sep8", "music"], default="sep8",
                    help="'sep8': training-matched 8-deg study — Trans/TCN/DA x 3 seeds "
                         "plus LCPSN uncovON/uncovOFF x 3 seeds (15 models). "
                         "'music': classical baseline block — {naive,arcsine,full-precision} "
                         "covariance x {MDL,AIC,oracle-K} enumeration (9 rows, no checkpoints).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="training seeds to evaluate (--checkpoint-set sep8 only)")
    ap.add_argument("--lcpsn-variants", nargs="+", default=["uncovON", "uncovOFF"],
                    help="LCPSN sep8 variants to evaluate, i.e. the tag in "
                         "best_lcpsn_sep8_<tag>_seed<s>.pt (--checkpoint-set sep8 only)")
    ap.add_argument("--skip-three", action="store_true",
                    help="evaluate only the LCPSN variants, skipping Trans/TCN/DA. "
                         "Legitimate because each point's waveforms are fixed by "
                         "_sweep's per-point seed, so a later partial run is scored "
                         "on the same scenes and its CSV can be merged with an earlier one.")
    ap.add_argument("--out-prefix", default=None,
                    help="override the output CSV prefix (default: unified / unified_sep8 / unified_music)")
    ap.add_argument("--min-seps", nargs="+", default=None,
                    help="evaluation conditions: use none and/or numeric degrees, e.g. --min-seps 8")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.smoke:
        num_tests = args.num_tests or 128
        snr_points = [{"snr_db": s, "snapshots": 200} for s in (-10, 0, 10, 20)]
        snap_points = [{"snr_db": 10, "snapshots": t} for t in (50, 200, 1000)]
        random_init = True
    else:
        num_tests = args.num_tests or 25600
        snr_points = [{"snr_db": s, "snapshots": 200} for s in range(-10, 21, 2)]
        snap_points = [{"snr_db": 10, "snapshots": t}
                       for t in (50, 100, 200, 300, 400, 600, 800, 1000)]
        random_init = False

    if args.min_seps is None:
        eval_min_seps = [8.0]
    else:
        eval_min_seps = [None if str(x).lower() == "none" else float(x) for x in args.min_seps]

    if args.checkpoint_set == "music":
        adapters = ad.build_music(device=args.device)
        prefix = "unified_music"
    elif args.checkpoint_set == "sep8":
        adapters = ad.build_sep8(device=args.device, seeds=tuple(args.seeds),
                                 random_init=random_init,
                                 lcpsn_variants=tuple(args.lcpsn_variants),
                                 include_three=not args.skip_three)
        prefix = "unified_sep8"
    prefix = args.out_prefix or prefix
    missing = [a.name for a in adapters if not a.available]
    print(f"Models: {len(adapters)} declared, {len(adapters) - len(missing)} available"
          + (f" | MISSING: {missing}" if missing else ""))
    print(f"num_tests/point={num_tests}  device={args.device}  smoke={args.smoke} "
          f"set={args.checkpoint_set}")
    if missing and not args.smoke:
        raise FileNotFoundError("Required paper checkpoints are missing: " + ", ".join(missing))
    if not any(a.available for a in adapters):
        print("nothing to evaluate — no checkpoints found"); return

    print("\n== SNR sweep (T=200) ==")
    snr_rows = _sweep(adapters, snr_points, "snr", num_tests, args.batch, min_seps=eval_min_seps)
    with open(os.path.join(RESULTS, f"{prefix}_snr_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(snr_rows)

    print("\n== Snapshot sweep (SNR=10dB) ==")
    snap_rows = _sweep(adapters, snap_points, "snapshot", num_tests, args.batch, min_seps=eval_min_seps)
    with open(os.path.join(RESULTS, f"{prefix}_snapshot_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(snap_rows)

    print(f"\n[OK] wrote {len(snr_rows)} + {len(snap_rows)} rows to {RESULTS}/")


if __name__ == "__main__":
    main()
