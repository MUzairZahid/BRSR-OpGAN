"""Verify the Restoration Observatory page: exported samples, embedded data, the page's own maths and playback.

These tests need only NumPy, pandas and (for the JavaScript checks) Node.js - not PyTorch or the test split.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def builder():
    return load('make_restoration_page', ROOT / 'scripts/make_restoration_page.py')


def page_source():
    return (ROOT / 'docs/brsr_restoration.html').read_text(encoding='utf-8')


def embedded_raw():
    return json.loads(page_source().split('const RAW = ', 1)[1].split(';\n', 1)[0])


def pure_math(source):
    return source[source.index('function decodeComponent('):source.index('/* ---- end pure math ---- */')]


def snr_db(est, ref):
    return 10 * np.log10(np.mean(ref ** 2) / np.mean((est - ref) ** 2))


def test_exported_samples_are_released_rows_and_outputs():
    """Every exported row is a real test row and its SNRs match the reference results of the released weights."""
    b = builder()
    manifest, arrays = b.load_samples()
    ref = pd.read_csv(ROOT / 'reference_results/brsr_test_per_sample.csv').set_index('row')
    meta_rows = set()
    assert len(manifest['records']) == 36 and manifest['model'] == 'BRSR-OpGAN-D-2P'
    for r in manifest['records']:
        key = f"r{r['row']}"
        clean, echo, cci, awgn = (arrays[f'{key}_{k}'].astype(np.float64) for k in ('clean', 'echo', 'cci', 'awgn'))
        received = clean + echo + cci + awgn
        for k, col in (('restored1', 'BRSR-OpGAN-D__snr_db'), ('restored2', 'BRSR-OpGAN-D-2P__snr_db')):
            assert abs(snr_db(arrays[f'{key}_{k}'].astype(np.float64), clean) - ref.loc[r['row'], col]) < 0.02
        assert abs(snr_db(received, clean) - ref.loc[r['row'], 'input__snr_db']) < 0.02
        assert r['composition'] == 'AWGN+Echo+CCI' and 128 <= r['echo_delay'] <= 512
        assert abs(sum(r['weights'].values()) - 1) < 1e-5  # metadata rounds to 6 decimals
        assert r['row'] not in meta_rows
        meta_rows.add(r['row'])
    classes = [r['cls'] for r in manifest['records']]
    assert classes == sorted(classes, key=classes.index) and len(set(classes)) == 12


def test_embedded_data_matches_export():
    b = builder()
    raw = embedded_raw()
    assert raw == b.build_data(), 'page must be rebuilt with scripts/make_restoration_page.py'
    for name, samples in raw['records'].items():
        assert len(samples) == 3
        for d in samples:
            clean = b.decode_component(d['clean'])
            np.testing.assert_allclose(np.mean(np.sum(clean ** 2, axis=0)), 1, atol=1e-4)  # clean RMS = 1
            received = clean + sum(b.decode_component(d[k]) for k in ('echo', 'cci', 'noise'))
            assert abs(snr_db(received, clean) - d['snrInput']) < 0.01
            assert abs(snr_db(b.decode_component(d['restored2']), clean) - d['snrPass2']) < 0.01
            assert abs(snr_db(b.decode_component(d['restored1']), clean) - d['snrPass1']) < 0.01
    assert raw['maxQuantError'] < 3.1e-5


def test_page_is_self_contained():
    html = page_source()
    assert 'data:font/woff2;base64,' in html
    for marker in ('__SIGNAL_DATA__', '__THEME_CSS__', '__FONT_CSS__', '__COMPONENT_DASHES__'):
        assert marker not in html
    assert 'fonts.googleapis.com' not in html and 'radar_environment.html' not in html
    palette = load('brsr_palette', ROOT / 'scripts/brsr_palette.py')
    for theme in palette.THEMES.values():
        for key in ('clean', 'echo', 'cci', 'awgn', 'received', 'restored', 'residual', 'surface'):
            assert theme[key] in html
    assert len(html.encode('utf-8')) < 1.6e6


def test_browser_decode_mix_and_snr_match_numpy(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required for browser math regression')
    b = builder()
    raw = embedded_raw()
    cases = []
    for name, samples in raw['records'].items():
        for sample, d in enumerate(samples):
            comps = {k: b.decode_component(d[k]) for k in ('clean', 'echo', 'cci', 'noise', 'restored1', 'restored2')}
            for mask in (0, 1, 3, 7):
                enabled = [bool(mask & (1 << k)) for k in range(3)]
                z = comps['clean'].copy()
                for on, key in zip(enabled, ('echo', 'cci', 'noise')):
                    if on:
                        z += comps[key]
                ratio = np.sum((z - comps['clean']) ** 2) / np.sum(comps['clean'] ** 2)
                cases.append(dict(wf=name, sample=sample, enabled=enabled, pass_=2, expected=z.tolist(),
                                  ratio=float(ratio), restoredSnr=float(snr_db(comps['restored2'], comps['clean']))))
    payload = tmp_path / 'cases.json'
    payload.write_text(json.dumps(dict(raw=raw, cases=cases)), encoding='utf-8')
    script = tmp_path / 'verify.cjs'
    script.write_text("""
const assert=require('node:assert/strict');
const payload=JSON.parse(require('node:fs').readFileSync(process.argv[2],'utf8'));
let state;
const record=()=>DATA.records[state.wf][state.sample];
""" + pure_math(page_source()) + """
const DATA=decodeAll(payload.raw);
const original=JSON.stringify(DATA);
for(const c of payload.cases){
  state={wf:c.wf,sample:c.sample,enabled:c.enabled,pass:c.pass_}; const z=mix(), m=measured(z), r=measured(restored());
  for(let ch=0;ch<2;ch++)for(let n=0;n<1024;n++)assert.ok(Math.abs(z[ch][n]-c.expected[ch][n])<1e-9);
  assert.ok(Math.abs(m.ratio-c.ratio)<1e-9);
  if(c.ratio===0)assert.equal(m.snr,Infinity);else assert.ok(Math.abs(m.snr+10*Math.log10(c.ratio))<1e-9);
  assert.ok(Math.abs(r.snr-c.restoredSnr)<1e-6);
}
const S=spectrum(DATA.records.LFM[0].clean);assert.equal(S.length,241);assert.equal(S[0].length,64);
assert.equal(JSON.stringify(DATA),original,'exploring never mutates the decoded data');
console.log('Verified '+payload.cases.length+' mixtures, restored SNRs, STFT layout and immutable source data');
""", encoding='utf-8')
    subprocess.run([node, str(script), str(payload)], check=True, timeout=120)


def test_restoration_playback():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required for the JavaScript timeline regression')
    source = (ROOT / 'scripts' / 'restoration_template.html').read_text(encoding='utf-8')
    stage = source.split('function setStage(stage){', 1)[1].split('\n', 1)[0]
    frame = source.split('function frame(now){', 1)[1].split('\nif(reduced.matches)', 1)[0]
    functions = 'const STAGES=5, STAGE_START=[0,2,4,6,8], T_END=12;\nconst stageAt=t=>{let s=0;for(let i=0;i<STAGES;i++)if(t>=STAGE_START[i])s=i;return s;};\nvar runs=0;function runModel(){runs++;}\nfunction setStage(stage){' + stage + '\nfunction frame(now){' + frame
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const context = {
  state: {playing:true, loop:true, t:0, stage:0, enabled:[false,false,false], inspect:'All'},
  last:0, reduced:{matches:true}, document:{hidden:false}, sceneVisible:true,
  update(){}, progress(){}, setPlotLimit(){}, requestAnimationFrame(){}
};
vm.createContext(context);
vm.runInContext(FUNCTIONS, context);
let now = 0;
const tick = count => {for(let i=0;i<count;i++) context.frame(now += 100);};
tick(25);
assert.equal(context.state.stage, 1, 'echo is revealed after 2 s');
tick(60);
assert.equal(context.state.stage, 4, 'restored stage reached after 8 s');
assert.equal(context.runs, 1, 'the model animation runs once when stage 05 is reached');
assert.equal(JSON.stringify(context.state.enabled), '[true,true,true]');
tick(40);
assert.equal(context.state.playing, true, 'default playback continues after 12 s');
assert.equal(context.state.stage, 0, 'a completed sequence starts again');
context.state.playing = false;
const paused = context.state.t;
tick(40);
assert.equal(context.state.t, paused, 'manual pause freezes the clock');
context.state.playing = true;
context.document.hidden = true;
tick(40);
assert.equal(context.state.t, paused, 'background tab must not consume the reveal');
context.document.hidden = false;
context.state.loop = false;
tick(200);
assert.equal(context.state.playing, false, 'single pass stops');
assert.equal(context.state.stage, 4);
assert.equal(context.state.t, 12);
console.log('PASS: stage timing, restore trigger, loop, pause, background visibility and single pass');
""".replace('FUNCTIONS', json.dumps(functions))
    subprocess.run([node, '-e', harness], check=True, timeout=15)
