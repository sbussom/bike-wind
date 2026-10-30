"""Self-contained Leaflet page for the animated GustyPower wind map.

Rendered with ``st.components.v1.html``. Leaflet loads from unpkg and base
tiles from Esri's dark-gray canvas (both free, attribution shown). The wind field, slider and
play button run entirely in the browser, so scrubbing never reruns Python.
"""

from __future__ import annotations

import json
from typing import Any

from bike_wind.wind_field import COLOR_STOPS_MPH

LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"


def build_wind_map_html(
    payload: dict[str, Any],
    *,
    center: tuple[float, float],
    zoom: int = 10,
    home: tuple[float, float] | None = None,
    home_label: str = "You",
    home_series: list[list[float]] | None = None,
    route: list[list[float]] | None = None,
    route_label: str | None = None,
    predictions: dict[str, str] | None = None,
    initial_frame: int = 0,
    height_px: int = 520,
    tz_name: str = "America/New_York",
) -> str:
    """HTML document for the iframe. ``payload`` comes from ``WindField``."""
    cfg = {
        "center": list(center),
        "zoom": zoom,
        "home": list(home) if home else None,
        "homeLabel": home_label,
        "homeSeries": home_series or [],
        "route": route or [],
        "routeLabel": route_label or "",
        "predictions": predictions or {},
        "initialFrame": int(initial_frame),
        "tz": tz_name,
        "stops": [[m, list(c)] for m, c in COLOR_STOPS_MPH],
    }
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    cfg_json = json.dumps(cfg, separators=(",", ":")).replace("</", "<\\/")
    return (
        _TEMPLATE.replace("__HEIGHT__", str(int(height_px)))
        .replace("__LEAFLET_CSS__", LEAFLET_CSS)
        .replace("__LEAFLET_JS__", LEAFLET_JS)
        .replace("__DATA__", data_json)
        .replace("__CFG__", cfg_json)
    )


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="__LEAFLET_CSS__">
<style>
  html,body{margin:0;padding:0;background:#0d1117;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
  #wrap{position:relative;width:100%;height:__HEIGHT__px;border-radius:12px;overflow:hidden;border:1px solid #30363d;}
  #map{position:absolute;inset:0;background:#0d1117;}
  canvas.gp{position:absolute;left:0;top:0;pointer-events:none;z-index:450;}
  .gp-card{position:absolute;z-index:800;background:rgba(13,17,23,.82);color:#e6edf3;
    border:1px solid rgba(255,255,255,.12);border-radius:10px;backdrop-filter:blur(4px);}
  #brand{top:10px;left:10px;padding:8px 11px;max-width:62%;}
  #brand .b{font-weight:800;letter-spacing:.3px;font-size:15px;color:#7ee787;}
  #brand .b span{color:#58a6ff;}
  #brand .t{font-size:13px;margin-top:2px;}
  #brand .r{font-size:12px;color:#c9d1d9;margin-top:3px;line-height:1.35;}
  #brand .p{font-size:13px;margin-top:4px;color:#ffd866;font-weight:600;}
  #legend{top:10px;right:10px;padding:6px 8px 4px;width:170px;}
  #legend .bar{height:9px;border-radius:5px;}
  #legend .ticks{display:flex;justify-content:space-between;font-size:10px;color:#c9d1d9;margin-top:2px;}
  #legend .u{font-size:10px;color:#8b949e;text-align:right;}
  #bar{left:10px;right:10px;bottom:10px;padding:8px 10px 6px;display:flex;align-items:center;gap:10px;}
  #play{flex:0 0 auto;width:42px;height:42px;border-radius:50%;border:0;cursor:pointer;
    background:#238636;color:#fff;font-size:17px;line-height:42px;padding:0;}
  #play:active{transform:scale(.96);}
  #slide{flex:1 1 auto;min-width:0;}
  #when{font-size:13px;font-weight:600;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  input[type=range]{width:100%;margin:2px 0;accent-color:#3fb950;height:26px;}
  #days{position:relative;height:13px;font-size:10px;color:#8b949e;}
  #days span{position:absolute;top:0;transform:translateX(-50%);white-space:nowrap;}
  .leaflet-control-zoom{margin-top:72px!important;}
  .leaflet-control-attribution{font-size:9px!important;background:rgba(13,17,23,.6)!important;color:#8b949e!important;}
  .leaflet-control-attribution a{color:#8b949e!important;}
  .leaflet-bottom.leaflet-right{bottom:78px;}
  @media (max-width:520px){
    #legend{width:130px;} #brand{max-width:58%;} #brand .r{font-size:11px;}
    .leaflet-control-zoom{display:none;}
  }
</style></head>
<body>
<div id="wrap">
  <div id="map"></div>
  <canvas id="heat" class="gp"></canvas>
  <canvas id="flow" class="gp"></canvas>
  <div id="brand" class="gp-card">
    <div class="b">Gusty<span>Power</span> wind</div>
    <div class="t" id="tlabel"></div>
    <div class="r" id="readout"></div>
    <div class="p" id="pred"></div>
  </div>
  <div id="legend" class="gp-card">
    <div class="bar" id="lbar"></div>
    <div class="ticks"><span>0</span><span>10</span><span>20</span><span>30</span><span>40</span><span>50</span></div>
    <div class="u">mph · 10 m wind</div>
  </div>
  <div id="bar" class="gp-card">
    <button id="play" aria-label="Play">&#9654;</button>
    <div id="slide">
      <div id="when"></div>
      <input id="range" type="range" min="0" max="0" step="1" value="0" aria-label="Forecast time">
      <div id="days"></div>
    </div>
  </div>
</div>
<script src="__LEAFLET_JS__"></script>
<script>
const DATA = __DATA__;
const CFG = __CFG__;
const N = DATA.times.length;
const STOPS = CFG.stops;
const CELL = 6;            // px per field cell
const ALPHA = 150;         // heat opacity 0-255
let frame = Math.min(Math.max(CFG.initialFrame || 0, 0), Math.max(N - 1, 0));
let playing = false, playTimer = null;

// ---- color scale ----
function colorFor(mph){
  if (mph <= STOPS[0][0]) return STOPS[0][1];
  for (let i = 1; i < STOPS.length; i++){
    const [b, cb] = STOPS[i], [a, ca] = STOPS[i-1];
    if (mph <= b){ const t = (mph - a) / (b - a);
      return [ca[0]+(cb[0]-ca[0])*t, ca[1]+(cb[1]-ca[1])*t, ca[2]+(cb[2]-ca[2])*t]; }
  }
  return STOPS[STOPS.length-1][1];
}
(function legend(){
  const parts = STOPS.map(([m,c]) => `rgb(${c[0]},${c[1]},${c[2]}) ${Math.min(m/50*100,100)}%`);
  document.getElementById('lbar').style.background = `linear-gradient(90deg, ${parts.join(',')})`;
})();

// ---- time formatting ----
const fmtFull = new Intl.DateTimeFormat('en-US', {timeZone: CFG.tz, weekday:'short', month:'short', day:'numeric', hour:'numeric'});
const fmtDay = new Intl.DateTimeFormat('en-US', {timeZone: CFG.tz, weekday:'short'});
const fmtHour = new Intl.DateTimeFormat('en-US', {timeZone: CFG.tz, hour:'numeric', hour12:false});
const fmtTz = new Intl.DateTimeFormat('en-US',{timeZone:CFG.tz,timeZoneName:'short'});
function tzShort(d){ return (fmtTz.formatToParts(d).find(p=>p.type==='timeZoneName')||{}).value || ''; }
const TIMES = DATA.times.map(s => new Date(s.replace('Z', ':00Z')));
function label(i){ return TIMES[i] ? fmtFull.format(TIMES[i]) + ' ' + tzShort(TIMES[i]) : ''; }

// ---- map ----
const map = L.map('map', {zoomControl: true, attributionControl: true, minZoom: 5, maxZoom: 13})
  .setView(CFG.center, CFG.zoom);
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/';
L.tileLayer(ESRI + 'World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 16, attribution: 'Tiles &copy; Esri · wind: Open-Meteo'
}).addTo(map);
map.createPane('labels'); map.getPane('labels').style.zIndex = 460; map.getPane('labels').style.pointerEvents = 'none';
L.tileLayer(ESRI + 'World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {maxZoom: 16, pane: 'labels'}).addTo(map);
map.attributionControl.setPrefix(false);
if (CFG.route && CFG.route.length > 1){
  L.polyline(CFG.route, {color:'#0d1117', weight:7, opacity:.7}).addTo(map);
  L.polyline(CFG.route, {color:'#ffffff', weight:3.5, opacity:.95}).addTo(map)
    .bindTooltip(CFG.routeLabel || 'Route', {sticky:true});
}
if (CFG.home){
  L.circleMarker(CFG.home, {radius:8, color:'#fff', weight:3, fillColor:'#f0883e', fillOpacity:1}).addTo(map)
    .bindTooltip(CFG.homeLabel, {direction:'top', offset:[0,-8]});
}

// ---- wind lookup ----
const GRIDS = DATA.grids.slice().sort((a,b) => ((a.latMax-a.latMin)*(a.lonMax-a.lonMin)) - ((b.latMax-b.latMin)*(b.lonMax-b.lonMin))); // finest first
function sampleGrid(g, f, lat, lon){
  if (lat < g.latMin || lat > g.latMax || lon < g.lonMin || lon > g.lonMax) return null;
  const fy = (lat - g.latMin) / (g.latMax - g.latMin) * (g.nLat - 1);
  const fx = (lon - g.lonMin) / (g.lonMax - g.lonMin) * (g.nLon - 1);
  const y0 = Math.floor(fy), x0 = Math.floor(fx);
  const y1 = Math.min(y0+1, g.nLat-1), x1 = Math.min(x0+1, g.nLon-1);
  const ty = fy - y0, tx = fx - x0;
  const U = g.u[f], V = g.v[f];
  const k00 = y0*g.nLon+x0, k01 = y0*g.nLon+x1, k10 = y1*g.nLon+x0, k11 = y1*g.nLon+x1;
  const u = (U[k00]*(1-tx)+U[k01]*tx)*(1-ty) + (U[k10]*(1-tx)+U[k11]*tx)*ty;
  const v = (V[k00]*(1-tx)+V[k01]*tx)*(1-ty) + (V[k10]*(1-tx)+V[k11]*tx)*ty;
  return [u, v];
}
function windAt(f, lat, lon){
  for (const g of GRIDS){ const r = sampleGrid(g, f, lat, lon); if (r) return r; }
  return null;
}

// ---- field + heat ----
const heat = document.getElementById('heat'), flow = document.getElementById('flow');
const hctx = heat.getContext('2d'), fctx = flow.getContext('2d');
let W = 0, H = 0, cols = 0, rows = 0, FU = null, FV = null, FOK = null;
const off = document.createElement('canvas'), octx = off.getContext('2d');
function resize(){
  const r = document.getElementById('wrap').getBoundingClientRect();
  W = Math.round(r.width); H = Math.round(r.height);
  const dpr = window.devicePixelRatio || 1;
  for (const c of [heat, flow]){ c.width = W*dpr; c.height = H*dpr; c.style.width = W+'px'; c.style.height = H+'px'; }
  hctx.setTransform(dpr,0,0,dpr,0,0); fctx.setTransform(dpr,0,0,dpr,0,0);
}
function buildField(){
  cols = Math.ceil(W / CELL) + 1; rows = Math.ceil(H / CELL) + 1;
  FU = new Float32Array(cols*rows); FV = new Float32Array(cols*rows); FOK = new Uint8Array(cols*rows);
  off.width = cols; off.height = rows;
  const img = octx.createImageData(cols, rows);
  for (let j = 0; j < rows; j++){
    for (let i = 0; i < cols; i++){
      const ll = map.containerPointToLatLng([i*CELL, j*CELL]);
      const w = windAt(frame, ll.lat, ll.lng);
      const k = j*cols + i, p = k*4;
      if (!w){ img.data[p+3] = 0; continue; }
      FU[k] = w[0]; FV[k] = w[1]; FOK[k] = 1;
      const c = colorFor(Math.hypot(w[0], w[1]));
      img.data[p] = c[0]; img.data[p+1] = c[1]; img.data[p+2] = c[2]; img.data[p+3] = ALPHA;
    }
  }
  octx.putImageData(img, 0, 0);
  hctx.clearRect(0, 0, W, H);
  hctx.imageSmoothingEnabled = true;
  hctx.drawImage(off, 0, 0, cols, rows, 0, 0, cols*CELL, rows*CELL);
}
function fieldAt(x, y){
  const i = Math.round(x / CELL), j = Math.round(y / CELL);
  if (i < 0 || j < 0 || i >= cols || j >= rows) return null;
  const k = j*cols + i; if (!FOK[k]) return null;
  return [FU[k], FV[k]];
}

// ---- particles ----
let parts = [];
function seed(p){ p.x = Math.random()*W; p.y = Math.random()*H; p.age = Math.floor(Math.random()*80); }
function resetParticles(){
  const n = Math.min(2200, Math.round(W*H/420));
  parts = []; for (let i = 0; i < n; i++){ const p = {}; seed(p); parts.push(p); }
  fctx.clearRect(0, 0, W, H);
}
let moving = false;
function tick(){
  if (!moving && FU){
    fctx.globalCompositeOperation = 'destination-in';
    fctx.fillStyle = 'rgba(0,0,0,0.93)'; fctx.fillRect(0, 0, W, H);
    fctx.globalCompositeOperation = 'source-over';
    fctx.strokeStyle = 'rgba(255,255,255,0.78)'; fctx.lineWidth = 1.3;
    fctx.beginPath();
    for (const p of parts){
      const w = fieldAt(p.x, p.y);
      if (!w || p.age > 70){ seed(p); p.age = 0; continue; }
      const s = 0.07;   // px per frame per mph (visual only)
      const nx = p.x + w[0]*s, ny = p.y - w[1]*s;
      fctx.moveTo(p.x, p.y); fctx.lineTo(nx, ny);
      p.x = nx; p.y = ny; p.age++;
      if (p.x < 0 || p.y < 0 || p.x > W || p.y > H) seed(p);
    }
    fctx.stroke();
  }
  requestAnimationFrame(tick);
}

// ---- UI ----
const range = document.getElementById('range'), playBtn = document.getElementById('play');
range.max = Math.max(N-1, 0); range.value = frame;
function dayTicks(){
  const box = document.getElementById('days'); box.innerHTML = '';
  if (N < 2) return;
  for (let i = 0; i < N; i++){
    if (Number(fmtHour.format(TIMES[i])) % 24 === 12){
      const s = document.createElement('span'); s.textContent = fmtDay.format(TIMES[i]);
      s.style.left = (i/(N-1)*100) + '%'; box.appendChild(s);
    }
  }
}
function compass(deg){ const d=['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']; return d[Math.round(((deg%360)+360)%360/22.5)%16]; }
function updateText(){
  const l = label(frame);
  document.getElementById('when').textContent = l;
  document.getElementById('tlabel').textContent = l;
  const hs = CFG.homeSeries[frame];
  document.getElementById('readout').textContent = hs
    ? `${CFG.homeLabel}: ${Math.round(hs[0])} mph from ${compass(hs[1])}, gusts ${Math.round(hs[2])}` : '';
  const pr = CFG.predictions[DATA.times[frame]];
  document.getElementById('pred').textContent = pr || '';
}
function setFrame(i){
  frame = Math.min(Math.max(i, 0), N-1); range.value = frame;
  updateText(); buildField();
}
range.addEventListener('input', () => setFrame(Number(range.value)));
function stop(){ playing = false; clearInterval(playTimer); playBtn.innerHTML = '&#9654;'; playBtn.setAttribute('aria-label','Play'); }
playBtn.addEventListener('click', () => {
  if (playing){ stop(); return; }
  playing = true; playBtn.innerHTML = '&#10074;&#10074;'; playBtn.setAttribute('aria-label','Pause');
  if (frame >= N-1) setFrame(0);
  playTimer = setInterval(() => { if (frame >= N-1){ stop(); return; } setFrame(frame+1); }, 650);
});
map.on('movestart zoomstart', () => { moving = true; fctx.clearRect(0,0,W,H); hctx.clearRect(0,0,W,H); });
map.on('moveend zoomend', () => { moving = false; buildField(); resetParticles(); });
window.addEventListener('resize', () => { resize(); map.invalidateSize(); buildField(); resetParticles(); });

resize(); dayTicks(); updateText(); buildField(); resetParticles(); requestAnimationFrame(tick);
</script>
</body></html>
"""
