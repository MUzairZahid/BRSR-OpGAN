"""Evaluate BRSR-OpGAN / CNN-GAN generators on the BRSR benchmark test split.

Examples
--------
# All released models on BRSR (reproduces Tables 2 and 3 of the paper)
python evaluate.py --dataset brsr

# Your own checkpoint (add a second path for a 2-pass model)
python evaluate.py --dataset brsr --checkpoint runs/my_run/generator_best.pth --q 3

Metrics (see docs/EVALUATION_PROTOCOL.md)
-----------------------------------------
Paper protocol: the noisy input is min-max scaled per channel to [-1, 1]; the output is mapped
back with the clean signal's per-channel min/max; SNR = 10 log10(mean|clean|^2 / mean|err|^2);
MSE in physical units; PSNR uses the peak |clean| within consecutive blocks of 32 test signals
(batch size of the original evaluation script).
Additional metric: SI-SDR of the zero-mean output vs. the zero-mean clean signal (scale-invariant).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from data import CLASS_NAMES, load_metadata, load_split
from models import ResidualGenerator

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> (q, [weight files applied in sequence])
MODEL_ZOO = {
    "brsr": {
        "CNN-GAN-T": (1, ["CNN_GAN_Time_Domain.pth"]),
        "CNN-GAN-D": (1, ["CNN_GAN_Dual_Domain.pth"]),
        "BRSR-OpGAN-T": (3, ["BRSR_OpGAN_Q3_Time_Domain.pth"]),
        "BRSR-OpGAN-D": (3, ["BRSR_OpGAN_Q3_Dual_Domain.pth"]),
        "BRSR-OpGAN-D-2P": (3, ["BRSR_OpGAN_Q3_Dual_Domain.pth", "BRSR_OpGAN_Q3_Dual_Domain_2ndPass.pth"]),
    },
}
SNR_BINS = ["[-14,-10)", "[-10,-6)", "[-6,-2)", "[-2,2)", "[2,6)", "[6,10]"]
COMPOSITIONS = ["AWGN", "Echo", "CCI", "AWGN+Echo", "AWGN+CCI", "Echo+CCI", "AWGN+Echo+CCI"]


# ---------------------------------------------------------------- metrics (numpy, float64)
def minmax(x):
    return x.min(axis=2, keepdims=True), x.max(axis=2, keepdims=True)


def snr_db(est, ref):
    return 10 * np.log10(np.mean(ref ** 2, axis=(1, 2)) / np.mean((est - ref) ** 2, axis=(1, 2)))


def mse(est, ref):
    return np.mean((est - ref) ** 2, axis=(1, 2))


def psnr_db(est, ref, block=32):
    out, err = np.empty(len(ref)), mse(est, ref)
    for a in range(0, len(ref), block):
        peak = np.abs(ref[a:a + block]).max()
        out[a:a + block] = 10 * np.log10(peak ** 2 / err[a:a + block])
    return out


def sisdr_db(est, ref):
    e = (est - est.mean(axis=2, keepdims=True)).reshape(len(est), -1)
    r = (ref - ref.mean(axis=2, keepdims=True)).reshape(len(ref), -1)
    target = (np.sum(e * r, 1) / np.sum(r * r, 1))[:, None] * r
    return 10 * np.log10(np.sum(target ** 2, 1) / np.sum((e - target) ** 2, 1))


# ---------------------------------------------------------------- inference
def load_chain(paths, q, device):
    chain = []
    for p in paths:
        G = ResidualGenerator(q=q)
        sd = torch.load(p, map_location="cpu")
        if isinstance(sd, dict) and "G" in sd:          # checkpoint_last.pth from train.py
            sd = sd["G"]
        G.load_state_dict(sd, strict=True)
        chain.append(G.to(device).eval())
    return chain


@torch.no_grad()
def restore(chain, x_norm, device, batch=256):
    y = np.empty_like(x_norm)
    for a in range(0, len(x_norm), batch):
        h = torch.from_numpy(x_norm[a:a + batch]).to(device)
        for G in chain:
            h = G(h)
        y[a:a + batch] = h.cpu().numpy()
    return y


def evaluate(models, data_dir, dataset, device):
    d = load_split(data_dir, dataset, "test")
    clean32, noisy32 = d["clean"].astype(np.float32), d["noisy"].astype(np.float32)
    cmin, cmax = minmax(clean32)
    nmin, nmax = minmax(noisy32)
    x_norm = (2 * (noisy32 - nmin) / (nmax - nmin) - 1).astype(np.float32)
    clean, noisy = clean32.astype(np.float64), noisy32.astype(np.float64)
    cmin, cmax = cmin.astype(np.float64), cmax.astype(np.float64)

    per = pd.DataFrame({"row": np.arange(len(clean)), "label": d["label"], "snr_target_db": d["snr_db"],
                        "input__snr_db": snr_db(noisy, clean), "input__psnr_db": psnr_db(noisy, clean),
                        "input__mse": mse(noisy, clean), "input__sisdr_db": sisdr_db(noisy, clean)})
    for name, (q, paths) in models.items():
        t0 = time.time()
        y = restore(load_chain(paths, q, device), x_norm, device).astype(np.float64)
        r_paper = (y + 1) / 2 * (cmax - cmin) + cmin
        per[f"{name}__snr_db"] = snr_db(r_paper, clean)
        per[f"{name}__psnr_db"] = psnr_db(r_paper, clean)
        per[f"{name}__mse"] = mse(r_paper, clean)
        per[f"{name}__sisdr_db"] = sisdr_db(y, clean)
        print(f"  {name:18s} SNR {per[f'{name}__snr_db'].mean():6.2f} dB   ({time.time() - t0:.0f}s)", flush=True)
    return per


def summarize(per, models, dataset, meta):
    per = per.copy()
    per["class_name"] = [CLASS_NAMES[int(k) - 1] for k in per["label"]]
    groups = [("overall", None, None)]
    if dataset == "brsr":
        if meta is not None:
            m = meta[meta["split"] == "test"].sort_values("row").reset_index(drop=True)
            per["snr_bin"], per["composition"] = m["snr_bin"].values, m["composition"].values
            groups += [("snr_bin", "snr_bin", SNR_BINS), ("composition", "composition", COMPOSITIONS)]
    else:
        groups += [("snr_db", "snr_target_db", sorted(per["snr_target_db"].unique()))]
    groups += [("class", "class_name", CLASS_NAMES)]
    rows = []
    for gname, col, values in groups:
        parts = [("all", per)] if col is None else [(v, per[per[col] == v]) for v in values]
        for g, sub in parts:
            for name in ["input"] + list(models):
                rows.append({"group_by": gname, "group": g, "n": len(sub),
                             "model": "Corrupted input" if name == "input" else name,
                             "snr_db": sub[f"{name}__snr_db"].mean(), "psnr_db": sub[f"{name}__psnr_db"].mean(),
                             "mse": sub[f"{name}__mse"].mean(), "sisdr_db": sub[f"{name}__sisdr_db"].mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["brsr", "awgn_baseline"], default="brsr",
                    help="released weights exist for brsr only; awgn_baseline requires --checkpoint")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--weights_dir", default=os.path.join(HERE, "pretrained_weights"),
                    help="folder with the brsr/ subfolder of released weights")
    ap.add_argument("--models", nargs="*", default=None, help="subset of released model names (default: all)")
    ap.add_argument("--checkpoint", nargs="+", default=None,
                    help="evaluate your own generator(s) instead; several paths are applied in sequence")
    ap.add_argument("--q", type=int, default=3, help="Self-ONN order of --checkpoint (1 = CNN)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default=None, help="default: results/<dataset>")
    args = ap.parse_args()

    if args.checkpoint:
        models = {"custom": (args.q, args.checkpoint)}
    else:
        if args.dataset not in MODEL_ZOO:
            ap.error(f"no released weights for {args.dataset}; pass --checkpoint")
        zoo = MODEL_ZOO[args.dataset]
        names = args.models or list(zoo)
        models = {n: (zoo[n][0], [os.path.join(args.weights_dir, args.dataset, f) for f in zoo[n][1]]) for n in names}
    out_dir = args.out_dir or os.path.join(HERE, "results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Evaluating on {args.dataset} test split ({args.device})")
    per = evaluate(models, args.data_dir, args.dataset, args.device)
    summary = summarize(per, models, args.dataset, load_metadata(args.data_dir, args.dataset))
    per.to_csv(os.path.join(out_dir, f"{args.dataset}_test_per_sample.csv"), index=False, float_format="%.4f")
    summary.to_csv(os.path.join(out_dir, f"{args.dataset}_test_summary.csv"), index=False, float_format="%.4f")

    ov = summary[summary.group_by == "overall"].set_index("model")
    print("\nOverall (paper protocol: SNR, PSNR, MSE; additional: SI-SDR)")
    print(ov[["snr_db", "psnr_db", "mse", "sisdr_db"]].round(2).to_string())
    if args.dataset == "awgn_baseline":
        t = summary[summary.group_by == "snr_db"].pivot(index="group", columns="model", values="snr_db")
        print("\nRestored SNR (dB) per input SNR level")
        print(t[[c for c in t.columns if c != "Corrupted input"]].round(2).to_string())
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
