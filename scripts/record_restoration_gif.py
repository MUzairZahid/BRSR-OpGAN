"""Record one loop of docs/brsr_restoration.html as the README GIF (docs/figures/brsr_restoration.gif).

    pip install playwright && playwright install chromium     # ffmpeg must be on PATH
    python scripts/make_restoration_page.py                    # build the page first
    python scripts/record_restoration_gif.py

The page exposes window.brsrEnv.setTime(t); every frame is rendered at an exact time, so the GIF does
not depend on the speed of the machine.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "brsr_restoration.html")
OUT = os.path.join(ROOT, "docs", "figures", "brsr_restoration.gif")


def to_gif(frame_dir, out_path, fps, width):
    pal = os.path.join(frame_dir, "pal.png")
    src = os.path.join(frame_dir, "f%04d.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", src, "-vf",
                    f"scale={width}:-1:flags=lanczos,palettegen=max_colors=160:stats_mode=full", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", src, "-i", pal, "-lavfi",
                    f"scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle",
                    "-loop", "0", out_path], check=True)


def record(out=OUT, fps=12, width=900, waveform="LFM", hold=0.0, setup=None):
    tmp = tempfile.mkdtemp()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
            if setup:
                setup(page)
            page.goto(Path(PAGE).as_uri() + f"?record=1&wf={waveform}")
            page.wait_for_function("window.brsrEnv !== undefined")
            page.evaluate("document.fonts.ready")
            page.evaluate("window.brsrEnv.freeze()")                 # stop the clock; frames are set by time
            duration = page.evaluate("window.brsrEnv.duration")
            board = page.locator("#board")
            n = int(round((duration + hold) * fps))
            for i in range(n):
                t = min(i / fps, duration - 1e-3)            # the last `hold` seconds repeat the final frame
                page.evaluate(f"window.brsrEnv.setTime({t})")
                board.screenshot(path=os.path.join(tmp, f"f{i:04d}.png"))
            browser.close()
        to_gif(tmp, out, fps, width)
    finally:
        shutil.rmtree(tmp)
    print(f"Saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--waveform", default="LFM", choices=["LFM", "BPSK", "Frank", "T4"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    record(a.out, a.fps, a.width, a.waveform)
