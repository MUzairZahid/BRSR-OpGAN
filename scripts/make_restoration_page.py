"""Build the Restoration Observatory page (docs/brsr_restoration.html).

    python scripts/export_restoration_samples.py   # once, needs the test split + PyTorch
    python scripts/make_restoration_page.py

Embeds the released test rows and model outputs exported by
scripts/export_restoration_samples.py (scripts/restoration_samples.npz) into the page
template, quantised to 16-bit integers with a documented error bound, together with the
shared design system (scripts/brsr_palette.py, a copy of BRSR-DataGen's) and the two
fonts (docs/fonts/, SIL OFL) so the page is a single offline file.

Requires only NumPy: the 0.6 GB test split and PyTorch are needed by the exporter only.
"""
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INT16_MAX = 32767
CLASS_TITLES = {
    'LFM': 'Linear frequency modulation', 'Costas': 'Costas frequency hopping', 'BPSK': 'Binary phase-shift keying',
    'Frank': 'Frank polyphase code', 'P1': 'P1 polyphase code', 'P2': 'P2 polyphase code', 'P3': 'P3 polyphase code',
    'P4': 'P4 polyphase code', 'T1': 'T1 time code', 'T2': 'T2 time code', 'T3': 'T3 time code', 'T4': 'T4 time code',
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


palette = load('brsr_palette', ROOT / 'scripts' / 'brsr_palette.py')


def encode_component(values):
    """[2, 1024] float array -> {'scale': float, 'b64': str} holding int16 little-endian samples (I then Q)."""
    arr = np.asarray(values, dtype=np.float64)
    assert arr.shape == (2, 1024)
    scale = float(np.max(np.abs(arr))) or 1.0
    q = np.round(arr / scale * INT16_MAX).astype('<i2')
    return {'scale': scale, 'b64': base64.b64encode(q.tobytes()).decode('ascii')}


def decode_component(packed):
    """Inverse of encode_component (used by the tests); returns a [2, 1024] float64 array."""
    q = np.frombuffer(base64.b64decode(packed['b64']), dtype='<i2').reshape(2, 1024)
    return q.astype(np.float64) / INT16_MAX * packed['scale']


def load_samples(path=ROOT / 'scripts' / 'restoration_samples.npz'):
    with np.load(path) as z:
        manifest = json.loads(str(z['manifest']))
        arrays = {k: z[k] for k in z.files if k != 'manifest'}
    return manifest, arrays


def build_data(path=ROOT / 'scripts' / 'restoration_samples.npz'):
    manifest, arrays = load_samples(path)
    out = dict(fs=manifest['fs'], dataset=manifest['dataset'], model=manifest['model'],
               selectionRule=manifest['selection_rule'],
               encoding='int16-le-base64, value = int / 32767 * scale, layout [I(1024), Q(1024)]', records={})
    worst = 0.0
    for r in manifest['records']:
        key = f"r{r['row']}"
        clean = arrays[f'{key}_clean'].astype(np.float64)
        scale = float(np.sqrt(np.mean(np.sum(clean ** 2, axis=0))))      # clean RMS -> 1 on the page
        rec = dict(title=CLASS_TITLES[r['cls']], band=r['band'], row=r['row'], label=r['label'],
                   snrTarget=r['snr_target_db'], snrInput=r['snr_input_db'], snrPass1=r['snr_pass1_db'],
                   snrPass2=r['snr_pass2_db'], sisdr=r['reference']['sisdr_pass2'], reference=r['reference'],
                   weights=[r['weights']['awgn'], r['weights']['echo'], r['weights']['cci']],
                   delay=r['echo_delay'], cciId=r['cci_signal_id'], composition=r['composition'], originalRms=scale)
        for page_key, npz_key in (('clean', 'clean'), ('echo', 'echo'), ('cci', 'cci'), ('noise', 'awgn'),
                                  ('restored1', 'restored1'), ('restored2', 'restored2')):
            values = arrays[f'{key}_{npz_key}'].astype(np.float64) / scale
            rec[page_key] = encode_component(values)
            worst = max(worst, float(np.max(np.abs(decode_component(rec[page_key]) - values)) / rec[page_key]['scale']))
        # The page recomputes SNR from the quantised traces; make sure that agrees with the protocol numbers.
        received = sum(decode_component(rec[k]) for k in ('clean', 'echo', 'cci', 'noise'))
        for k, expected in (('restored1', r['snr_pass1_db']), ('restored2', r['snr_pass2_db'])):
            est = decode_component(rec[k])
            snr = 10 * np.log10(np.mean(decode_component(rec['clean']) ** 2) / np.mean((est - decode_component(rec['clean'])) ** 2))
            assert abs(snr - expected) < 0.01, (r['row'], k, snr, expected)
        snr_in = 10 * np.log10(np.mean(decode_component(rec['clean']) ** 2) / np.mean((received - decode_component(rec['clean'])) ** 2))
        assert abs(snr_in - r['snr_input_db']) < 0.01, (r['row'], snr_in, r['snr_input_db'])
        out['records'].setdefault(r['cls'], []).append(rec)
    out['maxQuantError'] = worst
    assert worst < 3.1e-5, worst
    return out


def inline_font(filename):
    return base64.b64encode((ROOT / 'docs' / 'fonts' / filename).read_bytes()).decode('ascii')


def build(out=ROOT / 'docs' / 'brsr_restoration.html'):
    data = build_data()
    n = sum(map(len, data['records'].values()))
    print(f'Verified {n} test rows across {len(data["records"])} classes; '
          f'max quantisation error {data["maxQuantError"]:.2e} of trace peak; SNRs match the protocol.')
    template = (ROOT / 'scripts' / 'restoration_template.html').read_text(encoding='utf-8')
    fonts = (
        "@font-face{font-family:'Inter';font-style:normal;font-weight:100 900;font-display:swap;"
        "src:url(data:font/woff2;base64," + inline_font('inter-latin-wght-normal.woff2') + ") format('woff2')}\n"
        "@font-face{font-family:'JetBrains Mono';font-style:normal;font-weight:100 800;font-display:swap;"
        "src:url(data:font/woff2;base64," + inline_font('jetbrains-mono-latin-wght-normal.woff2') + ") format('woff2')}\n"
    )
    dashes = {name: palette.COMPONENT_DASHES[key] for name, key in
              [('Clean', 'clean'), ('Echo', 'echo'), ('Cci', 'cci'), ('Noise', 'awgn')]}
    dashes.update(Restored=(), Model=())
    html = (template
            .replace('__THEME_CSS__', palette.css_tokens())
            .replace('__FONT_CSS__', fonts)
            .replace('__COMPONENT_DASHES__', json.dumps(dashes))
            .replace('__SIGNAL_DATA__', json.dumps(data, separators=(',', ':'))))
    for marker in ('__THEME_CSS__', '__FONT_CSS__', '__SIGNAL_DATA__', '__COMPONENT_DASHES__'):
        assert marker not in html, marker
    out.write_text(html, encoding='utf-8', newline='\n')
    print(f'Built {out} ({len(html.encode("utf-8")):,} bytes).')


if __name__ == '__main__':
    build()
