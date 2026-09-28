"""Make the static README figures: SNR sweep and one- vs two-pass comparison.

Uses real BRSR test signals and the released BRSR-OpGAN weights:

    python download_data.py --dataset brsr --splits test
    python scripts/make_demo_figures.py

Outputs go to docs/figures/. Requires matplotlib.
The README animation is made by make_restoration_page.py and record_restoration_gif.py.

All displayed signals are min-max normalized per channel, i.e. what the network sees.
SNR values follow the paper's evaluation protocol (docs/EVALUATION_PROTOCOL.md).
"""
import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data import CLASS_NAMES  # noqa: E402
from models import ResidualGenerator  # noqa: E402

# Test-split rows used for the figures (both: AWGN + echo + CCI blends)
ROW_SWEEP, ROW_TWO_PASS = 5489, 12771
FS = 100e6
WIN = slice(0, 128)                        # default time window (1.28 us)
WIN_STATIC = slice(0, 80)                  # samples shown in the static figures (0.8 us)

# Palette (validated: CVD- and normal-vision-safe; restored/aqua is always direct-labelled)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
C_CLEAN, C_CORRUPT, C_RESTORED = "#2a78d6", "#eb6834", "#1baf7a"
SPEC_CMAP = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


# ---------------------------------------------------------------- helpers
def norm(x):
    """Per-channel min-max to [-1, 1] (x: [2, L])."""
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    return 2 * (x - lo) / (hi - lo) - 1


def db(v, sign=False):
    """Format a dB value with a typographic minus sign."""
    return (f"{v:+.1f}" if sign else f"{v:.1f}").replace("-", "\u2212")


def snr_db(est, ref):
    return 10 * np.log10(np.mean(ref ** 2) / np.mean((est - ref) ** 2))


def to_physical(y_norm, clean):
    lo, hi = clean.min(axis=1, keepdims=True), clean.max(axis=1, keepdims=True)
    return (y_norm + 1) / 2 * (hi - lo) + lo


def spectrogram_db(x, nfft=64, hop=8):
    z = x[0] + 1j * x[1]
    w = np.hanning(nfft)
    frames = np.stack([z[i:i + nfft] * w for i in range(0, len(z) - nfft + 1, hop)])
    s = np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2
    return 10 * np.log10(s.T + 1e-12)


def load_models(weights_dir, device="cpu"):
    out = []
    for f in ("BRSR_OpGAN_Q3_Dual_Domain.pth", "BRSR_OpGAN_Q3_Dual_Domain_2ndPass.pth"):
        G = ResidualGenerator(q=3)
        G.load_state_dict(torch.load(os.path.join(weights_dir, f), map_location="cpu"), strict=True)
        out.append(G.to(device).eval())
    return out


@torch.no_grad()
def run(models, x_norm):
    """Returns (first-pass, second-pass) outputs for normalized inputs [N, 2, L]."""
    h = torch.from_numpy(np.asarray(x_norm, np.float32))
    p1 = models[0](h)
    p2 = models[1](p1)
    return p1.numpy().astype(np.float64), p2.numpy().astype(np.float64)


def load_rows(path, rows):
    with h5py.File(path, "r") as f:
        if "row" in f:                                   # small extract with a 'row' index
            idx = [int(np.where(f["row"][:] == r)[0][0]) for r in rows]
        else:                                            # full brsr_test.h5
            idx = rows
        return {k: np.stack([f[k][i] for i in idx]).astype(np.float64) for k in ("clean", "noisy", "distortions")} | \
               {"label": np.array([int(f["label"][i]) for i in idx])}


def style_time_axis(ax, ylabel=True, win=WIN):
    ax.set_xlim(0, (win.stop - win.start) / FS * 1e6)
    ax.set_ylim(-1.35, 1.35)
    ax.set_yticks([-1, 0, 1])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("time (µs)", fontsize=9)
    if ylabel:
        ax.set_ylabel("I channel (normalized)", fontsize=9)


def add_key(ax, entries):
    """Direct-label key in the right margin: entries = [(label, color, lw, ls)]."""
    for k, (lab, c, lw, ls) in enumerate(entries):
        y = 0.85 - 0.2 * k
        ax.plot([1.03, 1.1], [y, y], color=c, lw=lw, ls=ls, transform=ax.transAxes, clip_on=False)
        ax.text(1.12, y, lab, color=INK2, fontsize=9, va="center", transform=ax.transAxes)


# ---------------------------------------------------------------- static figures
def make_snr_sweep(models, d, i, out_path, levels=(-12, -6, 0, 6)):
    clean, dist = d["clean"][i], d["distortions"][i].sum(axis=0)
    p_clean, p_dist = np.mean(clean ** 2), np.mean(dist ** 2)
    xs = [clean + np.sqrt(p_clean / (p_dist * 10 ** (s / 10))) * dist for s in levels]
    _, out = run(models, np.stack([norm(x) for x in xs]))
    tt = np.arange(WIN_STATIC.stop - WIN_STATIC.start) / FS * 1e6
    fig, axes = plt.subplots(2, len(levels), figsize=(12, 4.6), dpi=150, sharex=True, sharey=True)
    for j, (s, x, y) in enumerate(zip(levels, xs, out)):
        r = snr_db(to_physical(y, clean), clean)
        a, b = axes[0, j], axes[1, j]
        a.plot(tt, norm(x)[0, WIN_STATIC], color=C_CORRUPT, lw=1.3)
        a.set_title(f"Input SNR {db(s, True)} dB", loc="left", fontsize=10.5)
        b.plot(tt, norm(clean)[0, WIN_STATIC], color=C_CLEAN, lw=1.3, ls=(0, (3, 2)))
        b.plot(tt, y[0, WIN_STATIC], color=C_RESTORED, lw=1.8)
        b.set_title(f"Restored: {db(r)} dB", loc="left", fontsize=10.5)
        for ax in (a, b):
            style_time_axis(ax, ylabel=(j == 0), win=WIN_STATIC)
    for ax in axes[0]:
        ax.set_xlabel("")
    axes[0, 0].set_ylabel("received\n(normalized)", fontsize=9)
    axes[1, 0].set_ylabel("restored vs. clean\n(normalized)", fontsize=9)
    add_key(axes[0, -1], [("received", C_CORRUPT, 1.3, "-")])
    add_key(axes[1, -1], [("restored", C_RESTORED, 1.8, "-"), ("clean", C_CLEAN, 1.3, (0, (3, 2)))])
    cls = CLASS_NAMES[d["label"][i] - 1]
    fig.suptitle(f"One {cls} test signal with its echo + interference + noise mix, re-scaled to each input SNR, "
                 "restored by BRSR-OpGAN-D-2P", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def make_two_pass(models, d, i, out_path):
    clean, noisy = d["clean"][i], d["noisy"][i]
    p1, p2 = run(models, norm(noisy)[None])
    panels = [("Received", norm(noisy), snr_db(noisy, clean), C_CORRUPT),
              ("1st pass (BRSR-OpGAN-D)", p1[0], snr_db(to_physical(p1[0], clean), clean), C_RESTORED),
              ("2nd pass (BRSR-OpGAN-D-2P)", p2[0], snr_db(to_physical(p2[0], clean), clean), C_RESTORED)]
    tt = np.arange(WIN_STATIC.stop - WIN_STATIC.start) / FS * 1e6
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9), dpi=150, sharey=True)
    for j, (ax, (name, y, s, c)) in enumerate(zip(axes, panels)):
        ax.plot(tt, norm(clean)[0, WIN_STATIC], color=C_CLEAN, lw=1.3, ls=(0, (3, 2)))
        ax.plot(tt, y[0, WIN_STATIC], color=c, lw=1.6 if j else 1.3)
        ax.set_title(f"{name}: SNR {db(s)} dB", loc="left", fontsize=10.5)
        style_time_axis(ax, ylabel=(j == 0), win=WIN_STATIC)
    add_key(axes[-1], [("received", C_CORRUPT, 1.3, "-"), ("restored", C_RESTORED, 1.6, "-"),
                       ("clean", C_CLEAN, 1.3, (0, (3, 2)))])
    cls = CLASS_NAMES[d["label"][i] - 1]
    fig.suptitle(f"Second restoration pass on a {cls} test signal (echo + interference + noise)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 0.9, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test_file", default=os.path.join(ROOT, "data", "brsr_test.h5"))
    ap.add_argument("--weights_dir", default=os.path.join(ROOT, "pretrained_weights", "brsr"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "docs", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [ROW_SWEEP, ROW_TWO_PASS]
    d = load_rows(args.test_file, rows)
    models = load_models(args.weights_dir)
    _, restored = run(models, np.stack([norm(x) for x in d["noisy"]]))
    snr_in = [snr_db(d["noisy"][i], d["clean"][i]) for i in range(len(rows))]
    snr_out = [snr_db(to_physical(restored[i], d["clean"][i]), d["clean"][i]) for i in range(len(rows))]
    for i in range(len(rows)):
        print(f"row {rows[i]}: input {snr_in[i]:.2f} dB -> restored {snr_out[i]:.2f} dB")

    make_snr_sweep(models, d, 0, os.path.join(args.out_dir, "snr_sweep.png"))
    make_two_pass(models, d, 1, os.path.join(args.out_dir, "two_pass.png"))
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
