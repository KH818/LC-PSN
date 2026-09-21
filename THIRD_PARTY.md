# Baseline provenance notes

`baselines/TCN_MUSIC/Estimators/Trans_MUSIC` is the comparison reimplementation described in the paper. It is based on the method in:

J. Ji, W. Mao, F. Xi, and S. Chen, “TransMUSIC: A transformer-aided subspace method for DOA estimation with low-resolution ADCs,” ICASSP 2024, doi: 10.1109/ICASSP48485.2024.10447483.

The public release labels it **Trans-MUSIC** to distinguish this implementation and training protocol from an official author release. It consumes one-bit sign sequences, uses a source-count classifier trained after the regressor, and is evaluated with this repository's common unknown-K protocol.

The TCN-MUSIC and DA-MUSIC folders are included because they appear in the paper comparison. Their file-level origin and license history are not recorded in the local workspace. The authors must complete that provenance check before public distribution. Do not imply that the TransMUSIC paper authors produced or endorsed this reimplementation.
