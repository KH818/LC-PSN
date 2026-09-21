# Paper-to-code map

| Paper component | Implementation | Main output |
|---|---|---|
| One-bit arcsine covariance R0 | `lcpsn/model.py` | normalized complex covariance surrogate |
| First-order plug-in uncertainty U0 | `lcpsn/uncertainty.py` | real-augmented entry-wise weights |
| Transformer covariance refinement | `lcpsn/model.py` | Hermitian refined covariance R_ref |
| Bartlett prior and probabilistic spectrum heads | `lcpsn/model.py` | spectrum mean `mu` and variance `sigma2` |
| Source-count head | `lcpsn/model.py` | logits over K={2,3,4,5} |
| Spectrum, GNLL, K, variance, and covariance losses | `lcpsn/criterion.py` | total training objective |
| LCPSN training recipe | `lcpsn/train.py` | checkpoint and epoch log |
| Trans-MUSIC | `baselines/TCN_MUSIC/Estimators/Trans_MUSIC/` | fixed-length DOA regression and K logits |
| TCN-MUSIC | `baselines/TCN_MUSIC/Estimators/TCN_MUSIC/` | fixed-length DOA regression and K logits |
| DA-MUSIC | `baselines/TCN_MUSIC/Estimators/DA_MUSIC/` | fixed-length DOA regression and K logits |
| Shared test waveform generation | `evaluation/signal_gen.py` | identical one-bit batches for every method |
| K-hat decoding and partial matching | `evaluation/decoder.py`, `evaluation/metrics.py` | K accuracy, RMSPE, MAE, SR@5 degrees |
| Classical MUSIC comparison | `evaluation/music_baselines.py` | three covariance front ends by three K rules |
| SNR and snapshot sweeps | `evaluation/run_eval.py` | tidy per-seed CSV files |

Two angle grids occur in the LCPSN path. The Bartlett steering prior uses the endpoint-exclusive 360-point grid from training. The learned spectrum target and final decoding use the endpoint-inclusive 360-point grid. `evaluation/adapters.py` keeps these grids separate.

U0 is a first-order delta-method plug-in approximation and is used only as a training weight. It is not an input feature to LCPSN.
