# Evaluation protocol for the BRSR benchmark

## Protocol of the BRSR-OpGAN paper

The released models were trained and evaluated with this protocol, and `evaluate.py` implements it:

- Each clean and noisy signal is min–max scaled to [−1, 1] per channel (I and Q), using its own minimum and maximum.
- The network sees the scaled noisy signal.
- The network output is mapped back to physical units with the clean signal's per-channel min/max.
- SNR = 10·log10( mean|clean|² / mean|restored − clean|² ), computed per signal over both channels and then averaged.
- MSE = mean|restored − clean|² in physical units.
- PSNR = 10·log10( peak² / MSE ), where the peak is the largest |clean| value within each block of 32 consecutive test signals (the batch size of the original evaluation script).

With the released weights and the released test split, this reproduces Table 2 (within 0.03 dB SNR and 0.01 MSE), Table 3 (within 0.04 dB) and Fig. 6 (mean / median SNR improvement of 14.30 / 13.66 dB) of the paper.

## Additional metric

- **SI-SDR** (`sisdr_db`): scale-invariant SDR of the zero-mean network output against the zero-mean clean signal, with I and Q stacked. It measures how well the waveform shape is restored.

For BRSR-OpGAN-D-2P on the BRSR test split: SNR 12.36 dB and SI-SDR 10.93 dB (corrupted input: −1.94 dB and −2.02 dB).

## Reporting groups

- **BRSR**: overall; six 4-dB input-SNR bins with edges −14, −10, −6, −2, 2, 6, 10 dB (`snr_bin` in the metadata); the 7 artifact compositions (`composition`); the 12 modulation classes.
- **AWGN-Baseline** (when evaluating your own checkpoint): the 13 input-SNR levels (−14 : 2 : 10 dB); the 12 modulation classes.

`evaluate.py` writes all of these to `results/<dataset>/<dataset>_test_summary.csv` when the metadata CSV is in the data folder.
