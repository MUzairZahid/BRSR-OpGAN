"""Record the README animations of the Restoration Observatory (Playwright + ffmpeg).

    pip install playwright && playwright install chromium      # ffmpeg must be on PATH
    python scripts/record_restoration_gifs.py

Outputs (docs/figures/restoration/), one loop of the guided reveal (12 s) plus a 2 s hold on
the restored state, in light and dark mode:
    restoration_scene_{light,dark}.gif     the radar environment + restorer card (section 01)
    restoration_compare_{light,dark}.gif   clean / received / restored / residual cards (section 02)

Frames are taken at fixed timeline positions through the page's `window.brsrPage.setTime`
hook, so every run gives the same animation regardless of machine speed. The GIFs use a
256-colour palette per file with error-diffusion dithering (ffmpeg palettegen/paletteuse).
"""
import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / 'docs' / 'brsr_restoration.html').resolve().as_uri()
OUT = ROOT / 'docs' / 'figures' / 'restoration'
FPS, HOLD_S, WIDTH = 6, 2.0, 900
TARGETS = {
    'scene': "section[aria-labelledby='sceneTitle']",
    'compare': "section[aria-labelledby='cmpTitle']",
}


def to_gif(frame_dir, out_path, fps=FPS, width=WIDTH):
    pal = os.path.join(frame_dir, 'pal.png')
    src = os.path.join(frame_dir, 'f%04d.png')
    scale = f'format=rgb24,scale={width}:-1:flags=lanczos'   # drop the alpha of element screenshots
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', src, '-vf',
                    f'{scale},palettegen=max_colors=256', pal], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', src, '-i', pal, '-lavfi',
                    f'{scale}[x];[x][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle', '-loop', '0',
                    str(out_path)], check=True)


async def record(browser, scheme, name, selector):
    ctx = await browser.new_context(viewport={'width': 1440, 'height': 1000}, color_scheme=scheme)
    page = await ctx.new_page()
    await page.goto(PAGE)
    await page.wait_for_timeout(800)
    await page.evaluate("() => { document.documentElement.style.scrollBehavior = 'auto'; }")
    duration = await page.evaluate('window.brsrPage.duration')
    element = page.locator(selector)
    await element.scroll_into_view_if_needed()
    tmp = tempfile.mkdtemp()
    n = 0
    total = int(duration * FPS)
    for i in range(total + int(HOLD_S * FPS)):
        t = min(i, total - 1) / FPS
        await page.evaluate(f'window.brsrPage.setTime({t})')
        if i == total - 1:
            await page.wait_for_timeout(1500)            # let the restorer sweep settle before the hold
        else:
            await page.wait_for_timeout(40)
        await element.screenshot(path=os.path.join(tmp, f'f{n:04d}.png'), omit_background=False)
        n += 1
    await ctx.close()
    out = OUT / f'restoration_{name}_{scheme}.gif'
    to_gif(tmp, out)
    shutil.rmtree(tmp)
    print(f'wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {n} frames)')


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for scheme in ('light', 'dark'):
            for name, selector in TARGETS.items():
                await record(browser, scheme, name, selector)
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
