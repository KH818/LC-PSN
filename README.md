# LCPSN: one-bit DOA estimation with unknown source count

This repository is the public-code layout for the paper **“Uncertainty-Weighted Learnable Covariance Refinement for One-Bit Direction-of-Arrival Estimation With an Unknown Number of Sources.”** It contains the proposed LCPSN method, the Trans-MUSIC, TCN-MUSIC, and DA-MUSIC comparison implementations, classical MUSIC baselines, the common evaluator, paper checkpoints, and the CSV files used for the reported tables and figures.

The paper experiment uses an 8-element half-wavelength ULA, 200 training snapshots, source count K in {2,3,4,5}, a minimum 8-degree source separation, and training SNRs {0,5,10,15,20} dB. Learned results are averaged over training seeds 0, 1, and 2.

## Paper

Read the full research paper: **[LCPSN paper (PDF)](lcpsn/LCPSN.pdf)**.

## Repository map

- `lcpsn/`: LCPSN model, arcsine covariance recovery, uncertainty approximation, loss, data generation, and training.
- `baselines/TCN_MUSIC/`: Trans-MUSIC, TCN-MUSIC, and DA-MUSIC implementations used in the comparison.
- `evaluation/`: one shared signal generator, decoder, metrics, classical MUSIC baselines, evaluation runner, and plotting scripts.
- `checkpoints/`: the 24 paper checkpoints: six LCPSN checkpoints and regressor/classifier pairs for three comparison models over three seeds.
- `results/paper/`: immutable, publication-facing CSV exports.
- `evaluation/results/`: full CSV inputs used by the plotting scripts.
- `docs/`: method-to-code mapping, baseline details, data policy, and reproduction commands.

## Setup

Python 3.10 or newer is recommended. Install a PyTorch build appropriate for the local CUDA version, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
pip install -r requirements.txt
```

## Quick check

The quick command loads the supplied seed-0 checkpoints for LCPSN, Trans-MUSIC, TCN-MUSIC, and DA-MUSIC and evaluates a small deterministic sample:

```bash
bash scripts/quick_eval.sh
```

This is a pipeline check, not a reproduction of the paper statistics.

## Reproduce the reported evaluation

```bash
bash scripts/reproduce_evaluation.sh
```

The full run evaluates 25,600 scenes per operating point and can take a long time. All learned models receive exactly the same generated waveform batch at each point. Classical MUSIC is evaluated separately on the same deterministic generation schedule.

See `docs/reproduction.md` for individual commands and `docs/paper_to_code.md` for the paper-section mapping.

## Data

Training HDF5 files are deliberately excluded because the two training/validation pairs total about 7.2 GB. Deterministic generators and fixed settings are included. Generated files are written under `data/generated/`, which is ignored by Git. The supplied checkpoints are sufficient for evaluation.

## Metric warning

The paper reports unknown-K partial matching: only `min(K_hat, K)` angles are matched, using the assignment that minimizes wrapped per-scene RMSPE. A method that under-counts is therefore not penalized for an unreported source in its angular error. Source-count accuracy is reported beside the angular metrics. The exact implementation is `evaluation/metrics.py`.

## Citation and licensing status

Citation metadata is provided in `CITATION.cff`. The project does not yet carry a public software license. Until the authors select a license and confirm the provenance of the comparison implementations, normal copyright restrictions apply. Review `LICENSE_PENDING.md` and `THIRD_PARTY.md` before publishing the repository.
