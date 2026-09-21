# Data and deterministic generation

The paper training distribution is:

- eight half-wavelength-spaced ULA sensors;
- 200 snapshots;
- mutually uncorrelated circular complex Gaussian sources;
- K cycling over {2,3,4,5};
- DOAs uniform in (-90, 90) degrees with at least 8 degrees pairwise separation;
- SNR in {0,5,10,15,20} dB;
- 256,000 training and 25,600 validation scenes;
- training data seed 1234 and validation data seed 4321.

Generate LCPSN data with:

```bash
cd lcpsn
python generate_paper_data.py
```

Generate Trans-MUSIC, TCN-MUSIC, and DA-MUSIC data with:

```bash
cd baselines/TCN_MUSIC
PYTHONPATH=. python exe/train_models/one_bit/sep8/generate_sep8_data.py
```

Both commands write to `data/generated/`. The LCPSN HDF5 stores complex pre-quantization observations and applies sign quantization while loading. The comparison HDF5 stores real/imaginary one-bit channels. These formats are intentionally different because they follow the respective training paths.

Test scenes are generated on demand by `evaluation/signal_gen.py`. A batch is generated once and passed to all selected models. The per-point seed is fixed, so split evaluation runs see the same scenes when their SNR, snapshot count, and separation setting match.
