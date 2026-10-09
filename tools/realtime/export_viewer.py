#!/usr/bin/env python3
"""Export a real-time timeline as one self-contained HTML replay viewer.

    python3 tools/realtime/export_viewer.py reports/realtime_test0.json reports/realtime_test0.html

The page embeds the timeline and loads three.js r128 from cdnjs (pinned).
It shows the pallet with the boxes of each event's snapshot (the event's
box highlighted), the conveyor window and the buffer slots beside it, and a
side panel with time, action, compute_s / replans / idle, fill; play /
pause, speed and an event slider. No ROS / simulator needed.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rt_timeline import load_timeline, ordered_events, summary_text  # noqa: E402

THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
KEYS = ("t_start", "t_end", "idle_s", "compute_s", "replans", "action", "box", "target", "pallet_closed",
        "pallet_index", "pallet_size", "moved", "visible", "current", "buffer", "pallet", "fill")


def viewer_data(timeline):
    events = [{k: e.get(k) for k in KEYS} for e in ordered_events(timeline)]
    return {"summary": summary_text(timeline), "events": events}


def render_html(timeline, title="AHEAD real-time replay"):
    data = json.dumps(viewer_data(timeline), separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.replace("__TITLE__", title).replace("__THREE__", THREE_URL).replace("__DATA__", data)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f4f5f7; --panel:#ffffff; --text:#1d2330; --muted:#5d6675; --line:#d9dde4; --accent:#d9480f; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --panel:#1d2128; --text:#e6e8ec; --muted:#9aa3b2; --line:#2e3440; --accent:#ff8a4c; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui, sans-serif; }
#wrap { display:flex; height:100vh; }
#view { flex:1; position:relative; min-width:0; }
#side { width:360px; background:var(--panel); border-left:1px solid var(--line); padding:14px 16px; overflow:auto; }
h1 { font-size:16px; margin:0 0 4px; }
.muted { color:var(--muted); font-size:12px; }
.big { font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }
.row { display:flex; gap:12px; flex-wrap:wrap; margin:8px 0; }
.kv { min-width:90px; } .kv b { display:block; font-size:16px; font-variant-numeric:tabular-nums; }
.kv span { color:var(--muted); font-size:12px; }
.action { display:inline-block; padding:2px 8px; border-radius:4px; background:var(--accent); color:#fff; font-weight:600; }
.closed { display:inline-block; padding:2px 8px; border-radius:4px; background:#c92a2a; color:#fff; margin-left:6px; }
ul { margin:4px 0 10px; padding-left:18px; } li { font-variant-numeric:tabular-nums; }
#controls { position:absolute; left:12px; right:12px; bottom:12px; background:var(--panel); border:1px solid var(--line);
  border-radius:8px; padding:8px 12px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
#controls input[type=range] { flex:1; min-width:120px; }
button { font:inherit; padding:4px 12px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--text); cursor:pointer; }
@media (max-width: 760px) { #wrap { flex-direction:column; height:auto; } #view { height:60vh; } #side { width:auto; border-left:0; } }
</style>
</head>
<body>
<div id="wrap">
  <div id="view">
    <div id="controls">
      <button id="play">Play</button>
      <label class="muted">speed <span id="speedv">10</span>x</label>
      <input id="speed" type="range" min="0" max="100" value="50">
      <input id="scrub" type="range" min="0" max="0" value="0">
    </div>
  </div>
  <div id="side">
    <h1>AHEAD real-time replay</h1>
    <div class="muted" id="summary"></div>
    <div class="row"><div class="big" id="time">t = 0.0 s</div></div>
    <div><span class="action" id="action"></span><span id="closed"></span></div>
    <div id="box" style="margin-top:6px"></div>
    <div class="row">
      <div class="kv"><b id="compute"></b><span>compute_s</span></div>
      <div class="kv"><b id="replans"></b><span>replans</span></div>
      <div class="kv"><b id="idle"></b><span>robot idle_s</span></div>
      <div class="kv"><b id="fill"></b><span>fill (pallet <span id="pidx"></span>)</span></div>
    </div>
    <div class="muted">conveyor (camera window)</div><ul id="visible"></ul>
    <div class="muted">buffer</div><ul id="buffer"></ul>
    <div class="muted" id="moved"></div>
    <div class="muted" id="eventno"></div>
  </div>
</div>
<script src="__THREE__"></script>
<script>
const DATA = __DATA__;
const EV = DATA.events;
document.getElementById('summary').textContent = DATA.summary;
const view = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(window.devicePixelRatio || 1);
view.prepend(renderer.domElement);
const scene = new THREE.Scene();
const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
scene.background = new THREE.Color(dark ? 0x14171c : 0xf4f5f7);
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
camera.up.set(0, 0, 1);
scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 0.6); sun.position.set(2, -3, 4); scene.add(sun);

const P0 = EV.length ? EV[0].pallet_size : [1.2, 1.0, 1.5];
let pallet = null, layer = new THREE.Group(); scene.add(layer);
function makePallet(size) {
  if (pallet) scene.remove(pallet);
  pallet = new THREE.Group();
  const deck = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], 0.12),
    new THREE.MeshLambertMaterial({color: 0xa47548}));
  deck.position.set(size[0] / 2, size[1] / 2, -0.06); pallet.add(deck);
  const lim = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(size[0], size[1], size[2])),
    new THREE.LineBasicMaterial({color: dark ? 0x56606e : 0xb0b6c0}));
  lim.position.set(size[0] / 2, size[1] / 2, size[2] / 2); pallet.add(lim);
  scene.add(pallet);
}
makePallet(P0);
function target() { return new THREE.Vector3(P0[0] / 2 + 0.3, P0[1] / 2 - 0.2, 0.35); }
let az = -1.0, el = 0.6, dist = 5.2;
function placeCamera() {
  const t = target();
  camera.position.set(t.x + dist * Math.cos(el) * Math.cos(az), t.y + dist * Math.cos(el) * Math.sin(az), t.z + dist * Math.sin(el));
  camera.lookAt(t);
}
let drag = null;
renderer.domElement.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; });
window.addEventListener('pointerup', () => { drag = null; });
window.addEventListener('pointermove', e => {
  if (!drag) return;
  az -= (e.clientX - drag[0]) * 0.008; el = Math.max(0.05, Math.min(1.5, el + (e.clientY - drag[1]) * 0.006));
  drag = [e.clientX, e.clientY]; placeCamera();
});
renderer.domElement.addEventListener('wheel', e => { e.preventDefault(); dist = Math.max(1.2, Math.min(10, dist * (1 + e.deltaY * 0.001))); placeCamera(); }, {passive: false});

function skuColor(sku) {
  let h = 0; for (const c of String(sku)) h = (h * 31 + c.charCodeAt(0)) % 360;
  return new THREE.Color().setHSL(h / 360, 0.45, dark ? 0.55 : 0.62);
}
function addBox(min, dims, color, highlight) {
  const g = new THREE.BoxGeometry(dims[0] * 0.995, dims[1] * 0.995, dims[2] * 0.995);
  const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({color: color}));
  m.position.set(min[0] + dims[0] / 2, min[1] + dims[1] / 2, min[2] + dims[2] / 2);
  layer.add(m);
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(g),
    new THREE.LineBasicMaterial({color: highlight ? 0xff2d2d : 0x222222}));
  e.position.copy(m.position); layer.add(e);
}
function clearLayer() {
  for (const o of [...layer.children]) { layer.remove(o); o.geometry && o.geometry.dispose(); }
}
function fmtSize(s) { return s.map(v => v.toFixed(2)).join(' x '); }
function short(id) { return String(id).split('-').pop(); }

let shown = -1, currentSize = P0.join(',');
function show(i) {
  if (i === shown || !EV.length) return;
  shown = i;
  const e = EV[i];
  if (e.pallet_size.join(',') !== currentSize) { currentSize = e.pallet_size.join(','); makePallet(e.pallet_size); }
  clearLayer();
  const hid = e.target && e.box ? e.box.box_id : null;
  for (const b of e.pallet) addBox(b.min, b.dims, b.box_id === hid ? new THREE.Color(0xff7a3d) : skuColor(b.sku), b.box_id === hid);
  // conveyor window beside the pallet (-y side), current box first
  let x = 0; const y0 = -0.75;
  const conv = [];
  if (e.box && (e.action === 'PLACE_CURRENT' || e.action === 'BUFFER_CURRENT')) conv.push(e.box);
  else if (i > 0 && EV[i - 1].current) conv.push(EV[i - 1].current);
  for (const v of e.visible || []) if (!conv.find(c => c.box_id === v.box_id)) conv.push(v);
  conv.forEach((c, k) => { addBox([x, y0 - c.size[1] / 2, 0], c.size, skuColor(c.sku), k === 0); x += c.size[0] + 0.06; });
  // buffer slots on the +x side
  (e.buffer || []).forEach((b, k) => {
    const slotY = k * 0.62;
    const pad = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.58, 0.02), new THREE.MeshLambertMaterial({color: dark ? 0x39404c : 0xc8ccd3}));
    pad.position.set(e.pallet_size[0] + 0.5, slotY + 0.29, -0.01); layer.add(pad);
    if (b) addBox([e.pallet_size[0] + 0.5 - b.size[0] / 2, slotY + 0.29 - b.size[1] / 2, 0], b.size, skuColor(b.sku), false);
  });
  document.getElementById('action').textContent = e.action;
  document.getElementById('closed').innerHTML = e.pallet_closed ? '<span class="closed">PALLET CLOSED</span>' : '';
  document.getElementById('box').textContent = e.box ? `${e.box.box_id}  ${e.box.sku}  ${fmtSize(e.box.size)} m  ${e.box.weight} kg` : '';
  document.getElementById('compute').textContent = e.compute_s.toFixed(3);
  document.getElementById('replans').textContent = e.replans;
  document.getElementById('idle').textContent = e.idle_s.toFixed(2);
  document.getElementById('fill').textContent = (100 * e.fill).toFixed(1) + '%';
  document.getElementById('pidx').textContent = e.pallet_index;
  document.getElementById('visible').innerHTML = conv.length ? conv.map((c, k) => `<li>${k === 0 ? 'pick: ' : ''}${short(c.box_id)} ${c.sku} ${fmtSize(c.size)}</li>`).join('') : '<li>-</li>';
  document.getElementById('buffer').innerHTML = (e.buffer || []).map((b, k) => `<li>slot ${k}: ${b ? short(b.box_id) + ' ' + fmtSize(b.size) : '-'}</li>`).join('');
  document.getElementById('moved').textContent = e.moved && e.moved.length ? `repack moved: ${e.moved.map(short).join(', ')}` : '';
  document.getElementById('eventno').textContent = `event ${i + 1} / ${EV.length}`;
  scrub.value = i;
}

const scrub = document.getElementById('scrub'), speedEl = document.getElementById('speed'), playBtn = document.getElementById('play');
scrub.max = Math.max(0, EV.length - 1);
let t = EV.length ? EV[0].t_start : 0, playing = false, last = null;
function speed() { return Math.pow(10, (speedEl.value - 50) / 25) * 10; }  // 1x .. 100x, middle 10x
speedEl.oninput = () => { document.getElementById('speedv').textContent = speed().toFixed(speed() < 10 ? 1 : 0); };
speedEl.oninput();
playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; last = null;
  if (playing && shown === EV.length - 1) { t = EV[0].t_start; } };
scrub.oninput = () => { const i = +scrub.value; t = EV[i].t_start; show(i); };
function indexAt(time) { let lo = 0, hi = EV.length - 1, k = 0; while (lo <= hi) { const m = (lo + hi) >> 1; if (EV[m].t_start <= time) { k = m; lo = m + 1; } else hi = m - 1; } return k; }
function resize() { const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', resize);
function frame(now) {
  if (playing && EV.length) {
    if (last !== null) t += (now - last) / 1000 * speed();
    last = now;
    if (t >= EV[EV.length - 1].t_end) { t = EV[EV.length - 1].t_end; playing = false; playBtn.textContent = 'Play'; }
    show(indexAt(t));
  }
  document.getElementById('time').textContent = `t = ${t.toFixed(1)} s`;
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
resize(); placeCamera(); show(0); requestAnimationFrame(frame);
</script>
</body>
</html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("timeline", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--title", default="AHEAD real-time replay")
    args = ap.parse_args(argv)
    html = render_html(load_timeline(args.timeline), args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"wrote {args.output} ({len(html) / 1024:.0f} KiB, {len(load_timeline(args.timeline)['events'])} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
