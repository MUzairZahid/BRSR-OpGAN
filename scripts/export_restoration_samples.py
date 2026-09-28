"""Export the test signals shown on the restoration page, with real model outputs.

    python download_data.py --dataset brsr --splits test      # or point --test_file at your copy
    python scripts/export_restoration_samples.py               # writes scripts/restoration_samples.npz

For every selected row of the released BRSR test split this stores the clean target, the
three stored artifact components, the received signal and the outputs of the released
BRSR-OpGAN-D generator after the first and the second pass, all in physical units, plus
the per-row metadata and the SNR values of the paper's evaluation protocol
(docs/EVALUATION_PROTOCOL.md). The restored SNR of every row is checked against
reference_results/brsr_test_per_sample.csv, so the exported outputs are provably the
released model's outputs on the released data.

Row selection (deterministic, no hand-picking):
  for each of the 12 classes and each input-SNR band (target SNR within +-1 dB of
  +4, -2 and -10 dB), among the test rows whose corruption contains all three
  artifacts, the row with the median second-pass SNR improvement is taken.
Use --rows to pin specific rows instead (e.g. --rows LFM:19903 BPSK:9387).

The page builder (make_restoration_page.py) needs only NumPy and this file, so the
0.6 GB test split and PyTorch are required here only.
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from data import CLASS_NAMES  # noqa: E402
from models import ResidualGenerator  # noqa: E402

BANDS = ((4, "+4 dB"), (-2, "−2 dB"), (-10, "−10 dB"))
WEIGHTS = ("BRSR_OpGAN_Q3_Dual_Domain.pth", "BRSR_OpGAN_Q3_Dual_Domain_2ndPass.pth")
SELECTION_RULE = ("For each waveform class and each input-SNR band (target SNR within ±1 dB of +4, −2 and "
                  "−10 dB), among released test rows whose corruption contains all three artifacts, the row "
                  "with the median second-pass SNR improvement.")


def norm(x):
    """Per-channel min-max to [-1, 1] (x: [2, L]) - what the network sees."""
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    return 2 * (x - lo) / (hi - lo) - 1


def to_physical(y_norm, clean):
    """Paper protocol: map a network output back with the clean signal's per-channel min/max."""
    lo, hi = clean.min(axis=1, keepdims=True), clean.max(axis=1, keepdims=True)
    return (y_norm + 1) / 2 * (hi - lo) + lo


def snr_db(est, ref):
    return 10 * np.log10(np.mean(ref ** 2) / np.mean((est - ref) ** 2))


def load_models(weights_dir):
    out = []
    for f in WEIGHTS:
        g = ResidualGenerator(q=3)
        g.load_state_dict(torch.load(os.path.join(weights_dir, f), map_location="cpu"), strict=True)
        out.append(g.eval())
    return out


@torch.no_grad()
def run(models, x_norm):
    h = torch.from_numpy(np.asarray(x_norm, np.float32))
    p1 = models[0](h)
    p2 = models[1](p1)
    return p1.numpy().astype(np.float64), p2.numpy().astype(np.float64)


def select_rows(meta, reference):
    """Median-improvement row per class and band among all-three-artifact test rows."""
    t = meta[(meta["split"] == "test") & (meta["composition"] == "AWGN+Echo+CCI")]
    ref = reference.set_index("row")
    improvement = ref["BRSR-OpGAN-D-2P__snr_db"] - ref["input__snr_db"]
    selected = []
    for cls in CLASS_NAMES:
        g = t[t["class_name"] == cls]
        for centre, label in BANDS:
            band = g[(g["snr_target_db"] >= centre - 1) & (g["snr_target_db"] <= centre + 1)].copy()
            band["imp"] = improvement.loc[band["row"]].values
            band = band.sort_values(["imp", "row"])
            selected.append((cls, label, int(band.iloc[(len(band) - 1) // 2]["row"])))
    return selected


def load_rows(path, rows):
    with h5py.File(path, "r") as f:
        if "row" in f:                                   # small extract carrying the original row index
            index = {int(r): i for i, r in enumerate(f["row"][:])}
            idx = [index[r] for r in rows]
        else:                                            # the full brsr_test.h5
            idx = rows
        return {k: np.stack([f[k][i] for i in idx]).astype(np.float64) for k in ("clean", "noisy", "distortions")} | \
               {"label": np.array([int(f["label"][i]) for i in idx])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test_file", default=os.path.join(ROOT, "data", "brsr_test.h5"))
    ap.add_argument("--metadata", default=os.path.join(ROOT, "data", "brsr_metadata.csv"))
    ap.add_argument("--reference", default=os.path.join(ROOT, "reference_results", "brsr_test_per_sample.csv"))
    ap.add_argument("--weights_dir", default=os.path.join(ROOT, "pretrained_weights", "brsr"))
    ap.add_argument("--rows", nargs="*", default=None, help="pin rows as CLASS:ROW (replaces the selection rule)")
    ap.add_argument("--out", default=os.path.join(HERE, "restoration_samples.npz"))
    ap.add_argument("--tolerance_db", type=float, default=0.02, help="max |restored SNR - reference| allowed")
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata)
    meta = meta[meta["split"] == "test"] if "split" in meta else meta
    reference = pd.read_csv(args.reference)
    if args.rows:
        selected = []
        for item in args.rows:
            cls, row = item.split(":")
            target = float(meta.set_index("row").loc[int(row), "snr_target_db"])
            selected.append((cls, f"{target:+.0f} dB".replace("-", "−"), int(row)))
        rule = "Rows pinned by the author: " + ", ".join(f"{c} {r}" for c, _, r in selected)
    else:
        selected, rule = select_rows(meta, reference), SELECTION_RULE

    rows = [r for _, _, r in selected]
    d = load_rows(args.test_file, rows)
    p1, p2 = run(load_models(args.weights_dir), np.stack([norm(x) for x in d["noisy"]]))
    ref = reference.set_index("row")
    m = meta.set_index("row")

    arrays, records = {}, []
    for i, (cls, band, row) in enumerate(selected):
        assert CLASS_NAMES[d["label"][i] - 1] == cls, (row, cls, d["label"][i])
        clean, noisy, (awgn, echo, cci) = d["clean"][i], d["noisy"][i], d["distortions"][i]
        np.testing.assert_allclose(noisy, clean + awgn + echo + cci, atol=1e-4)
        r1, r2 = to_physical(p1[i], clean), to_physical(p2[i], clean)
        s_in, s1, s2 = snr_db(noisy, clean), snr_db(r1, clean), snr_db(r2, clean)
        for name, value, mine in (("input__snr_db", s_in, "input"), ("BRSR-OpGAN-D__snr_db", s1, "pass 1"),
                                  ("BRSR-OpGAN-D-2P__snr_db", s2, "pass 2")):
            diff = abs(value - float(ref.loc[row, name]))
            assert diff <= args.tolerance_db, f"row {row} {mine}: {value:.4f} vs reference {ref.loc[row, name]:.4f}"
        info = m.loc[row]
        key = f"r{row}"
        for k, v in (("clean", clean), ("echo", echo), ("cci", cci), ("awgn", awgn), ("restored1", r1), ("restored2", r2)):
            arrays[f"{key}_{k}"] = v.astype(np.float32)
        records.append(dict(
            cls=cls, band=band, row=int(row), label=int(d["label"][i]),
            snr_target_db=float(info["snr_target_db"]), snr_input_db=float(s_in),
            snr_pass1_db=float(s1), snr_pass2_db=float(s2),
            reference=dict(input=float(ref.loc[row, "input__snr_db"]), pass1=float(ref.loc[row, "BRSR-OpGAN-D__snr_db"]),
                           pass2=float(ref.loc[row, "BRSR-OpGAN-D-2P__snr_db"]),
                           sisdr_pass2=float(ref.loc[row, "BRSR-OpGAN-D-2P__sisdr_db"])),
            weights=dict(awgn=float(info["w_awgn"]), echo=float(info["w_echo"]), cci=float(info["w_cci"])),
            echo_delay=int(info["echo_delay"]), cci_signal_id=int(info["cci_signal_id"]),
            composition=str(info["composition"]),
        ))
        print(f"{cls:6s} {band:7s} row {row:5d}: input {s_in:6.2f} dB -> pass 1 {s1:6.2f} dB -> pass 2 {s2:6.2f} dB")

    manifest = dict(dataset="BRSR test split (Zenodo 10.5281/zenodo.23010395), v1.0", model="BRSR-OpGAN-D-2P",
                    weights=list(WEIGHTS), protocol="docs/EVALUATION_PROTOCOL.md", selection_rule=rule,
                    fs=100e6, records=records)
    np.savez_compressed(args.out, manifest=json.dumps(manifest), **arrays)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB, {len(records)} rows), all SNRs within "
          f"{args.tolerance_db} dB of reference_results.")


if __name__ == "__main__":
    main()
