"""One design system shared by BRSR-DataGen and BRSR-OpGAN figures and pages.

This file is a copy of scripts/brsr_palette.py in BRSR-DataGen with two extra roles
(``restored`` and ``residual``); keep the two in sync by hand. Every colour used by
the README figures and the interactive pages of both repositories comes from here,
so the signal components look identical everywhere, in light and in dark mode.

Component colours (categorical, fixed order, never re-assigned):

    clean  blue      x[n]  the training target
    echo   green     e[n]  benchmark echo (offset source segment)
    cci    orange    i[n]  co-channel interference
    awgn   neutral   a[n]  receiver noise
    restored  blue (solid; the clean target is then drawn dashed)  ŷ[n]  the model's estimate of x[n]
    residual  status red, always labelled                           ŷ[n] − x[n]  the error

The three chromatic slots were validated as a set for colour-vision deficiency
(protan/deutan simulated, OKLab distance >= 8 for every pair) and for contrast
against both surfaces. AWGN is deliberately neutral: noise is the "no structure"
component, so it wears the secondary-text tone and is always distinguished by a
dotted line style and a text label as well, never by colour alone. Line styles
are part of the encoding: clean solid, echo dashed, CCI dash-dot, AWGN dotted.

The spectrogram ramp is a single hue (blue), light-to-dark on the light surface
and surface-to-light on the dark surface, so magnitude always reads as "more ink".
"""

THEMES = {
    'light': dict(
        bg='#f4f3ee', surface='#fcfcfb', surface2='#f0efe9', line='#e2e0d8', line2='#d2d0c6',
        grid='#e8e6df', ink='#0b0b0b', ink2='#3f3e3a', muted='#5f5e58', faint='#8a8983',
        clean='#2a78d6', echo='#1baf7a', cci='#eb6834', awgn='#6f6e68', received='#141413',
        restored='#2a78d6', residual='#d03b3b',
        # single-hue sequential ramp, low -> high magnitude
        ramp=['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'],
        accent='#1c5cab', accent_ink='#ffffff', focus='#2a78d6',
    ),
    'dark': dict(
        bg='#0f1419', surface='#161d25', surface2='#1c2530', line='#2a3541', line2='#37434f',
        grid='#222c36', ink='#eef2f5', ink2='#c9d2da', muted='#b5bfc9', faint='#8e99a4',
        clean='#3987e5', echo='#199e70', cci='#d95926', awgn='#9aa4ae', received='#eef2f5',
        restored='#3987e5', residual='#e66767',
        ramp=['#161d25', '#132b47', '#144073', '#1c5cab', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'],
        accent='#6da7ec', accent_ink='#0f1419', focus='#9ec5f4',
    ),
}

COMPONENT_ORDER = ('clean', 'echo', 'cci', 'awgn', 'restored', 'residual')
COMPONENT_LABELS = {'clean': 'Clean target', 'echo': 'Echo', 'cci': 'Interference', 'awgn': 'AWGN', 'restored': 'Restored', 'residual': 'Residual'}
COMPONENT_SYMBOLS = {'clean': 'x[n]', 'echo': 'e[n]', 'cci': 'i[n]', 'awgn': 'a[n]'}
# Matplotlib / canvas dash patterns: secondary encoding so identity never rests on hue alone.
COMPONENT_DASHES = {'clean': (), 'echo': (6, 3), 'cci': (6, 2, 2, 2), 'awgn': (1.5, 2.5)}

FONT_SANS = "Inter, 'Segoe UI', system-ui, -apple-system, Roboto, 'Helvetica Neue', Arial, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, 'Cascadia Mono', Consolas, Menlo, monospace"


def css_tokens(indent='  '):
    """CSS custom properties for both themes (OS preference + explicit data-theme toggle)."""
    def block(theme):
        t = THEMES[theme]
        lines = [f'--{k.replace("_", "-")}:{v};' for k, v in t.items() if k != 'ramp']
        lines += [f'--ramp-{i}:{c};' for i, c in enumerate(t['ramp'])]
        lines += [f'--dash-{key}: ' + ' '.join(map(str, COMPONENT_DASHES[key])) + ';' for key in ('echo', 'cci', 'awgn')]
        lines.append(f'color-scheme:{theme};')
        return '\n'.join(indent + l for l in lines)
    return (
        ':root{\n' + block('light') + '\n}\n'
        '@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){\n' + block('dark') + '\n}}\n'
        ':root[data-theme="dark"]{\n' + block('dark') + '\n}\n'
    )


def matplotlib_cmap(theme):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('brsr_' + theme, THEMES[theme]['ramp'])


def matplotlib_rc(theme):
    t = THEMES[theme]
    return {
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': t['line2'], 'axes.labelcolor': t['ink2'],
        'xtick.color': t['ink2'], 'ytick.color': t['ink2'], 'axes.titlecolor': t['ink'], 'text.color': t['ink'],
        'figure.facecolor': t['surface'], 'axes.facecolor': t['surface'], 'savefig.facecolor': t['surface'],
        'grid.color': t['grid'],
    }


if __name__ == '__main__':
    print(css_tokens())
