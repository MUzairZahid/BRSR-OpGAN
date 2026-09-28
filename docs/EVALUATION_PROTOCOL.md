# Evaluation protocol for the BRSR benchmark

## Protocol of the BRSR-OpGAN paper

The released models were trained and evaluated with this protocol, and `evaluate.py` implements it:

- Each clean and noisy signal is min–max scaled to [−1, 1] per channel (I and Q), using its own minimum and maximum.
- The network sees the scaled noisy signal.
- The network output is mapped back to physical units with the **clean** signal's per-channel min/max.
- SNR = 10·log10( mean|clean|² / mean|restored − clean|² ), computed per signal over both channels and then averaged.
- MSE = mean|restored − clean|² in physical units.
- PSNR = 10·log10( peak² / MSE ), where the peak is the largest |clean| value within each block of 32 consecutive test signals (the batch size of the original evaluation script).

With the released weights and the released test split, this reproduces Table 2 (within 0.03 dB SNR and 0.01 MSE), Table 3 (within 0.04 dB) and Fig. 6 (mean / median SNR improvement of 14.30 / 13.66 dB) of the paper. On AWGN-Baseline, the released checkpoints match their original evaluation logs and are within 0.09 dB of Table 1.

## Later change: global normalization

Mapping the output back with the clean signal's min/max uses reference information that is not available in blind deployment, and it supplies the output's amplitude and offset. This per-signal practice was common at the time. We changed it in our later work: XCoRe-Net uses **global normalization** with training-split statistics.

[`norm_stats.json`](norm_stats.json) gives the per-channel mean and standard deviation of the **clean training signals** of each dataset:

```
x_norm = (x − mean[c]) / (std[c] + 1e-8)
```

Apply the same values to every split, and to clean, noisy and restored signals alike. For new comparisons we recommend this global protocol. Please state which protocol you used.

## Blind metrics reported by `evaluate.py`

- **SI-SDR** (`sisdr_db`): scale-invariant SDR of the zero-mean network output against the zero-mean clean signal, with I and Q stacked. It measures how well the waveform shape is restored, without any clean-based rescaling.
- **SNR with noisy-signal rescaling** (`snr_noisyscale_db`): SNR after mapping the output back with the **noisy** signal's min/max, the only range known at deployment. The released models were not trained to preserve amplitude, so this value is low for all of them.

For BRSR-OpGAN-D-2P on the BRSR test split: SNR (paper protocol) 12.36 dB, SI-SDR 10.93 dB (input −2.02 dB), and SNR with noisy-signal rescaling −4.51 dB.

## Reporting groups

- **BRSR**: overall; six 4-dB input-SNR bins with edges −14, −10, −6, −2, 2, 6, 10 dB (`snr_bin` in the metadata); the 7 artifact compositions (`composition`); the 12 modulation classes.
- **AWGN-Baseline**: the 13 input-SNR levels (−14 : 2 : 10 dB); the 12 modulation classes.

`evaluate.py` writes all of these to `results/<dataset>/<dataset>_test_summary.csv` when the metadata CSV is in the data folder.
