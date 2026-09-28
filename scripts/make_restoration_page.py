"""Build the interactive restoration page (docs/brsr_restoration.html).

    python download_data.py --dataset brsr --splits test
    python scripts/make_restoration_page.py

The page follows real BRSR test signals: the clean pulse, its echo, co-channel interference and AWGN
(the stored components of the test split) arrive at the receiver, and BRSR-OpGAN-D-2P restores the
received signal. Model outputs come from the released weights; SNR values follow the paper's
evaluation protocol (docs/EVALUATION_PROTOCOL.md). All test signals shown use all three artifacts.

The page is one self-contained HTML file (data and images embedded), served by GitHub Pages at
https://muzairzahid.github.io/BRSR-OpGAN/brsr_restoration.html
The README GIF is a recording of this page: scripts/record_restoration_gif.py
"""
import argparse
import base64
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_demo_figures import CLASS_NAMES, load_models, load_rows, norm, run, snr_db, to_physical  # noqa: E402

TEMPLATE = os.path.join(HERE, "restoration_template.html")
OUT = os.path.join(ROOT, "docs", "brsr_restoration.html")

# Test-split rows shown on the page: well-restored examples with all three artifacts (AWGN + echo + CCI)
# near 0 dB input SNR, one per waveform family.
ROWS = {"LFM": 19903, "BPSK": 9387, "Frank": 2266, "T4": 3502}
N_SHOW, DYN_DB = 160, 30
CMAP = LinearSegmentedColormap.from_list("ivory_blue", ["#fbfaf6", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])


def spec_db(z, nfft=64, hop=8):
    w = np.hanning(nfft)
    frames = np.stack([z[i:i + nfft] * w for i in range(0, len(z) - nfft + 1, hop)])
    return 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)).T ** 2 + 1e-12)


def spec_png(z, vmax):
    """640 x 260 px spectrogram, no axes; time left to right, frequency -50 (bottom) to +50 MHz (top)."""
    fig = plt.figure(figsize=(6.4, 2.6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(spec_db(z), aspect="auto", origin="lower", cmap=CMAP, vmin=vmax - DYN_DB, vmax=vmax, interpolation="bilinear")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    small = io.BytesIO()                                   # 64-colour palette keeps the page small
    Image.open(buf).convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT).save(small, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(small.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test_file", default=os.path.join(ROOT, "data", "brsr_test.h5"))
    ap.add_argument("--metadata", default=os.path.join(ROOT, "data", "brsr_metadata.csv"))
    ap.add_argument("--weights_dir", default=os.path.join(ROOT, "pretrained_weights", "brsr"))
    args = ap.parse_args()

    names, rows = list(ROWS), list(ROWS.values())
    d = load_rows(args.test_file, rows)
    meta = pd.read_csv(args.metadata)
    if "split" in meta:
        meta = meta[meta["split"] == "test"]
    meta = meta.set_index("row")
    _, y = run(load_models(args.weights_dir), np.stack([norm(x) for x in d["noisy"]]))

    data, buttons, imgs = {}, [], []
    for i, (name, row) in enumerate(zip(names, rows)):
        assert CLASS_NAMES[d["label"][i] - 1] == name, (row, d["label"][i])
        clean, noisy, (awgn, echo, cci) = d["clean"][i], d["noisy"][i], d["distortions"][i]
        restored = to_physical(y[i], clean)                    # paper protocol
        s_in, s_out = snr_db(noisy, clean), snr_db(restored, clean)
        cz = lambda x: x[0] + 1j * x[1]
        scale = np.sqrt(np.mean(np.abs(cz(clean)) ** 2))       # clean RMS = 1 on the page
        c, e, q, a, r = (cz(v) / scale for v in (clean, echo, cci, awgn, restored))
        stages = [c, c + e, c + e + q, c + e + q + a, r]
        ymax = max(float(np.abs(z.real[:N_SHOW]).max()) for z in stages) * 1.05
        m = meta.loc[row]
        show = lambda z: np.round(z.real[:N_SHOW], 3).tolist()
        data[name] = dict(name=name, row=row, clean=show(c), echo=show(e), cci=show(q), noise=show(a),
                          noisy=show(c + e + q + a), restored=show(r), ymax=round(ymax, 2),
                          w=[round(float(m["w_awgn"]), 2), round(float(m["w_echo"]), 2), round(float(m["w_cci"]), 2)],
                          delay=int(m["echo_delay"]), cci_id=int(m["cci_signal_id"]),
                          snr_in=round(float(s_in), 2), snr_out=round(float(s_out), 2))
        vmax = spec_db(c).max()
        for k, z in enumerate(stages):
            imgs.append(f'<img data-wf="{name}" data-k="{k}" alt="{name} spectrogram, '
                        f'{["clean", "with echo", "with echo and interference", "received", "restored"][k]}" '
                        f'src="{spec_png(z, vmax)}">')
        buttons.append(f'<button type="button" data-wf="{name}" aria-pressed="{"true" if i == 0 else "false"}">{name}</button>')
        print(f"{name}: test row {row}, input {s_in:.2f} dB -> restored {s_out:.2f} dB")

    html = open(TEMPLATE, encoding="utf-8").read()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    html = html.replace("__WF_BUTTONS__", "\n      ".join(buttons)).replace("__SPEC_IMGS__", "\n        ".join(imgs))
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"Saved {OUT} ({len(html) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
