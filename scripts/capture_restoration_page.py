"""Capture the Restoration Observatory preview images with Playwright (Chromium).

    pip install playwright && playwright install chromium
    python scripts/capture_restoration_page.py

Outputs (docs/figures/restoration/):
    restoration_hero_{light,dark}.png      README hero: scene with the restorer, 1600 x 1000
    restoration_compare_{light,dark}.png   clean / received / restored / residual cards, 1600 x 1120
    og_restoration.png                     1200 x 630 social card referenced by the page's og:image

The page is paused at the restored stage so the capture is deterministic.
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / 'docs' / 'brsr_restoration.html').resolve().as_uri()
OUT = ROOT / 'docs' / 'figures' / 'restoration'

PREPARE = """() => {
  const b = document.getElementById('play'); if (/Pause/.test(b.textContent)) b.click();
  document.querySelector('[data-stage="4"]').click();
  document.documentElement.style.scrollBehavior = 'auto';
}"""


async def shoot(browser, path, width, height, scheme, scroll_to, offset=12):
    ctx = await browser.new_context(viewport={'width': width, 'height': height}, color_scheme=scheme)
    page = await ctx.new_page()
    await page.goto(PAGE)
    await page.wait_for_timeout(700)
    await page.evaluate(PREPARE)
    await page.wait_for_timeout(3200)                      # let the restorer sweep finish (bars lit)
    await page.evaluate(f"() => window.scrollTo(0, document.querySelector('{scroll_to}').offsetTop - {offset})")
    await page.wait_for_timeout(400)
    await page.screenshot(path=str(path))
    await ctx.close()
    print('wrote', path)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for scheme in ('light', 'dark'):
            await shoot(browser, OUT / f'restoration_hero_{scheme}.png', 1600, 1000, scheme, '.picker')
            await shoot(browser, OUT / f'restoration_compare_{scheme}.png', 1600, 1120, scheme, '#chartgrid', 70)
        await shoot(browser, OUT / 'og_restoration.png', 1200, 630, 'light', '.workspace')
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
