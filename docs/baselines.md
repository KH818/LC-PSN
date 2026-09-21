# Learned comparison models

Trans-MUSIC, TCN-MUSIC, and DA-MUSIC receive the same tensor shape `[batch, 16, snapshots]`: real and imaginary sign streams for an eight-sensor ULA. Each method predicts a fixed-length DOA vector and exposes a learned feature vector. A separate classifier reads that feature and predicts one of four source-count classes, K={2,3,4,5}; evaluation truncates the DOA vector to K-hat.

The three methods differ in temporal processing:

- Trans-MUSIC applies Transformer encoding over snapshots and mean pooling.
- TCN-MUSIC applies a temporal convolutional network and global average pooling.
- DA-MUSIC uses its recurrent/eigendecomposition path.

For the paper experiment, the regressor is trained first. The classifier is then trained using the fitted regressor representation. Both checkpoint files are required for unknown-K evaluation. Every model uses the same array geometry, source-count distribution, SNR set, minimum source separation, sample counts, and training seeds. Their training datasets follow the same distribution but are generated independently of the LCPSN HDF5 files.

The Trans-MUSIC directory is a comparison reimplementation rather than a claim of byte-for-byte equivalence with an official implementation. See `THIRD_PARTY.md`.
