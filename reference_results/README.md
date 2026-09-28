# Reference results

Test-split results of the released generators in `pretrained_weights/`, produced with

```bash
python evaluate.py --dataset brsr
python evaluate.py --dataset awgn_baseline
```

- `*_test_summary.csv`: mean metrics overall and per group. BRSR groups: 4-dB input-SNR bin, artifact composition, modulation class. AWGN-Baseline groups: input-SNR level, modulation class.
- `*_test_per_sample.csv`: one row per test signal (`row` matches the HDF5 row and the `row` column of the dataset metadata CSV).

Columns: `snr_db`, `psnr_db`, `mse` follow the paper's protocol; `sisdr_db` is an additional, scale-invariant metric. See [../docs/EVALUATION_PROTOCOL.md](../docs/EVALUATION_PROTOCOL.md).
