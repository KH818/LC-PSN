# Reproduction guide

Run all commands from the repository root unless a command changes directory explicitly.

## Verify supplied checkpoints

```bash
bash scripts/quick_eval.sh
```

This evaluates 64 scenes per point for seed 0. It verifies checkpoint loading, model inference, common decoding, and CSV writing. It does not estimate the paper metrics accurately.

## Full learned comparison and ablation

```bash
cd evaluation
PYTHONPATH=. python run_eval.py \
  --checkpoint-set sep8 \
  --seeds 0 1 2 \
  --lcpsn-variants uncovON uncovOFF \
  --min-seps 8 \
  --out-prefix reproduced_sep8
```

The default is 25,600 test scenes per operating point. For each method and seed it evaluates SNR -10 through 20 dB at T=200 and snapshot counts 50 through 1000 at 10 dB. Paper plots display SNR results from 0 dB upward because training begins at 0 dB.

## Classical MUSIC block

```bash
cd evaluation
PYTHONPATH=. python run_eval.py \
  --checkpoint-set music \
  --min-seps 8 \
  --out-prefix reproduced_music
```

This runs naive one-bit, arcsine-corrected one-bit, and full-precision covariance front ends with MDL, AIC, and oracle-K enumeration.

## Regenerate bundled figures from the original result CSVs

```bash
cd evaluation
PYTHONPATH=. python plot_sep8_results.py
PYTHONPATH=. python plot_model_comparison.py
PYTHONPATH=. python plot_uncov_ablation.py
PYTHONPATH=. python plot_music_comparison.py
```

The plotting scripts read `evaluation/results/`. The paper-facing filtered CSV exports are preserved separately under `results/paper/`.

## Retrain

```bash
bash scripts/train_lcpsn.sh
bash scripts/train_baselines.sh
```

These are long GPU jobs. Outputs are written below `results/retrained/`; supplied paper checkpoints under `checkpoints/` are not overwritten. Reproducing identical neural weights can still depend on GPU, CUDA, cuDNN, and PyTorch behavior even with fixed seeds.
