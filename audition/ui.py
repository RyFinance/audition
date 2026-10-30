"""Shared look for Audition pages: grain-paper background and a scroll selector that
shows one physical object at a time (cassette tapes for report tabs, blank CD-Rs for
the strategy library). Pure HTML/CSS/JS, no libraries."""

from __future__ import annotations

import html
import json

BG, FG, MUTED, LINE = "#f2f2f2", "#0a0a0a", "#6e6e6e", "#d6d6d6"
GREEN, AMBER, RED = "#2e7d57", "#b5760f", "#c23b2a"

GRAIN = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/>"
         "<feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .9 0'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)'/></svg>")

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Inter+Tight:wght@400;500;600'
         '&family=Permanent+Marker&display=swap" rel="stylesheet">')

BASE_CSS = (f":root{{--bg:{BG};--fg:{FG};--muted:{MUTED};--line:{LINE};--green:{GREEN};--amber:{AMBER};--red:{RED};"
            "--serif:'Instrument Serif',Georgia,'Times New Roman',serif;"
            "--sans:'Inter Tight','Helvetica Neue',Arial,sans-serif;--mono:'SF Mono',Menlo,monospace;"
            "--marker:'Permanent Marker','Marker Felt','Comic Sans MS',cursive}") + """
*{box-sizing:border-box}
html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
.grain{position:fixed;inset:-50%;width:200%;height:200%;pointer-events:none;z-index:50;opacity:.2;
  background-image:url("GRAIN_URL");animation:grain 1s steps(6) infinite}
@keyframes grain{0%{transform:translate(0,0)}20%{transform:translate(-3%,2%)}40%{transform:translate(2%,-3%)}
  60%{transform:translate(-2%,-1%)}80%{transform:translate(3%,3%)}100%{transform:translate(0,0)}}
@media (prefers-reduced-motion:reduce){.grain{animation:none}}
a{color:inherit}

/* ---------- selector ---------- */
.sel{position:fixed;inset:0;overflow:hidden;user-select:none}
.sel[hidden]{display:none}
.sel-head{position:absolute;top:28px;left:32px;right:32px;z-index:3;pointer-events:none}
.sel-title{font:400 clamp(40px,6vw,76px)/.95 var(--serif);text-transform:uppercase;letter-spacing:-.01em;
  border-bottom:1px solid var(--fg);padding-bottom:10px;max-width:760px}
.sel-meta{max-width:760px;margin-top:2px}
.sel-meta div{display:flex;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding:9px 0 7px}
.sel-meta span{font:500 10px/1.4 var(--sans);letter-spacing:.2em;text-transform:uppercase}
.sel-meta b{font:400 15px/1.2 var(--sans);text-align:right}
.sel-num{position:absolute;left:32px;top:54%;z-index:3;font:400 clamp(64px,9vw,120px)/1 var(--serif);pointer-events:none}
.sel-num small{display:block;font:500 10px/1 var(--sans);letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-top:10px}
.sel-count{position:absolute;right:24px;top:54%;transform:translateY(-50%);width:210px;height:300px;z-index:3;
  perspective:700px;cursor:ns-resize;touch-action:none;
  -webkit-mask-image:linear-gradient(transparent,#000 28%,#000 72%,transparent);mask-image:linear-gradient(transparent,#000 28%,#000 72%,transparent)}
.sel-count::before,.sel-count::after{content:"";position:absolute;right:0;width:120px;height:1px;background:var(--line);top:calc(50% - 17px)}
.sel-count::after{top:calc(50% + 17px)}
.wheel{position:absolute;inset:0;transform-style:preserve-3d}
.wheel i{position:absolute;right:0;top:50%;height:30px;margin-top:-15px;line-height:30px;font:500 13px var(--mono);font-style:normal;
  color:var(--muted);white-space:nowrap;backface-visibility:hidden;cursor:pointer;
  transition:transform .6s cubic-bezier(.2,.8,.2,1),opacity .6s,color .3s}
.wheel.live i{transition:color .3s}
.wheel i b{font:400 15px var(--serif);margin-right:12px;opacity:0;transition:opacity .35s;letter-spacing:0}
.wheel i.on{color:var(--fg);font-size:15px}
.wheel i.on b{opacity:1}
.sel-hint{position:absolute;left:0;right:0;bottom:92px;text-align:center;font:500 10px/1 var(--sans);
  letter-spacing:.22em;text-transform:uppercase;color:var(--muted);z-index:3}
.stage{position:absolute;inset:0;display:grid;place-items:center;perspective:1500px;padding-top:120px}
.holder{transform-style:preserve-3d;transition:transform .9s cubic-bezier(.2,.8,.2,1),opacity .6s;cursor:pointer}
.holder.out-up{transform:translate3d(0,-70vh,0) rotateZ(-40deg);opacity:0}
.holder.out-down{transform:translate3d(0,70vh,0) rotateZ(40deg);opacity:0}
.holder.in-up{transform:translate3d(0,70vh,0) rotateZ(30deg);opacity:0;transition:none}
.holder.in-down{transform:translate3d(0,-70vh,0) rotateZ(-30deg);opacity:0;transition:none}
.tilt{transform-style:preserve-3d;transition:transform .25s ease-out}
.drop{position:absolute;left:50%;top:50%;width:70%;height:30%;transform:translate(-50%,70%);
  background:radial-gradient(closest-side,rgba(0,0,0,.22),transparent);filter:blur(14px);z-index:-1}

/* ---------- cassette ---------- */
.tape{position:relative;width:min(440px,78vw);aspect-ratio:1.58;transform-style:preserve-3d}
.tape .ly{position:absolute;inset:0;border-radius:14px;background:var(--edge)}
.tape .face{position:absolute;inset:0;border-radius:14px;background:var(--body);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.08),inset 0 -30px 60px rgba(0,0,0,.18);transform:translateZ(0)}
.tape .screw{position:absolute;width:9px;height:9px;border-radius:50%;background:radial-gradient(#bbb,#666);box-shadow:0 0 0 1px rgba(0,0,0,.4)}
.tape .label{position:absolute;left:6%;right:6%;top:7%;height:62%;border-radius:6px;background:var(--label);
  padding:4.5% 5%;display:flex;flex-direction:column;
  background-image:linear-gradient(transparent 0 calc(100% - 1px),rgba(0,0,0,.12) 0);background-size:100% 22%}
.tape .lt{display:flex;align-items:baseline;gap:10px}
.tape .side{font:600 22px/1 var(--sans)}
.tape .ttl{font:400 clamp(20px,3.6vw,30px)/1 var(--marker);color:#1d2a6b;transform:rotate(-1.5deg);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tape .win{position:absolute;left:21%;right:21%;top:34%;height:22%;border-radius:40px;background:rgba(20,20,20,.86);
  display:flex;align-items:center;justify-content:space-between;padding:0 6%;box-shadow:inset 0 2px 8px rgba(0,0,0,.6)}
.tape .win::before{content:"";position:absolute;left:26%;right:26%;top:30%;bottom:30%;background:#4a2f1f;border-radius:3px;opacity:.9}
.tape .reel{position:relative;width:26%;aspect-ratio:1;border-radius:50%;background:
  conic-gradient(from 0deg,#eee 0 10%,transparent 0 33.3%,#eee 0 43.3%,transparent 0 66.6%,#eee 0 76.6%,transparent 0);
  -webkit-mask:radial-gradient(circle,transparent 22%,#000 23% 48%,transparent 49%);mask:radial-gradient(circle,transparent 22%,#000 23% 48%,transparent 49%);
  animation:spin 6s linear infinite;animation-play-state:paused;z-index:1}
.holder:hover .reel,.holder.spin .reel{animation-play-state:running}
@keyframes spin{to{transform:rotate(360deg)}}
.tape .sub{margin-top:auto;font:500 10px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:#555;display:flex;justify-content:space-between}
.tape .foot{position:absolute;left:24%;right:24%;bottom:0;height:20%;background:var(--edge);clip-path:polygon(8% 0,92% 0,100% 100%,0 100%);
  border-radius:4px 4px 0 0;opacity:.9}
.tape .foot i{position:absolute;bottom:22%;width:7%;aspect-ratio:1;border-radius:50%;background:#0006}

/* ---------- blank disc ---------- */
.disc{position:relative;width:min(420px,76vw);aspect-ratio:1;transform-style:preserve-3d}
.disc .ly{position:absolute;inset:0;border-radius:50%;background:#b9b9b6;
  -webkit-mask:radial-gradient(circle,transparent 7.5%,#000 7.6%);mask:radial-gradient(circle,transparent 7.5%,#000 7.6%)}
.disc .face{position:absolute;inset:0;border-radius:50%;transform:translateZ(0);
  background:
    radial-gradient(circle,transparent 0 7.5%,rgba(255,255,255,.65) 7.6% 15.5%,rgba(0,0,0,.08) 15.6% 16%,transparent 16.1% 17.5%,rgba(0,0,0,.06) 17.6% 18%,transparent 18.1%),
    conic-gradient(from 210deg,rgba(255,170,200,.18),rgba(160,220,255,.22),rgba(200,255,190,.16),rgba(255,240,170,.2),rgba(255,170,200,.18)),
    radial-gradient(circle at 35% 30%,#ffffff,#f4f4f2 60%,#e6e6e3);
  -webkit-mask:radial-gradient(circle,transparent 7.5%,#000 7.6%);mask:radial-gradient(circle,transparent 7.5%,#000 7.6%);
  box-shadow:inset 0 0 0 1px rgba(0,0,0,.14),inset 0 0 0 5px rgba(255,255,255,.7),inset 0 0 0 6px rgba(0,0,0,.06)}
.disc .ink{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;transform:translateZ(1px)}
.disc .name{position:absolute;top:13%;left:18%;right:18%;text-align:center;font:400 clamp(20px,3.4vw,34px)/1.05 var(--marker);
  color:#16205c;transform:rotate(-6deg)}
.disc .note{position:absolute;bottom:23%;left:20%;right:20%;text-align:center;font:400 clamp(13px,2vw,17px)/1.2 var(--marker);color:#333;transform:rotate(-3deg)}
.disc .ring{position:absolute;right:12%;top:44%;width:17%;aspect-ratio:1;border:3px solid var(--ink,#c23b2a);border-radius:48% 52% 45% 55%;
  display:grid;place-items:center;font:400 clamp(22px,4vw,38px)/1 var(--marker);color:var(--ink,#c23b2a);transform:rotate(8deg)}
.disc .brand{position:absolute;left:0;right:0;bottom:8%;text-align:center;font:500 9px/1 var(--sans);letter-spacing:.3em;color:#999}
.disc.blank .name{color:#bbb}

/* ---------- views ---------- */
.view{display:none}
.view.on{display:block}
.wrap{max-width:1080px;margin:0 auto;padding:56px 20px 140px;position:relative}
.dock{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);z-index:40;background:#fff;color:#000;
  display:flex;align-items:stretch;font:500 13px/1 var(--sans);box-shadow:0 1px 2px rgba(0,0,0,.06),0 8px 30px rgba(0,0,0,.06);white-space:nowrap}
.dock>a,.dock>span,.dk-mid{display:flex;align-items:center;height:48px}
.dock>a,.dock>span{padding:0 18px}
.dock>*+*{border-left:1px solid #ececec}
.dock a{text-decoration:none;color:inherit;cursor:pointer}
.dk-home{color:var(--muted)!important}.dk-home:hover{color:#000!important}
.dk-word{font:400 21px/1 var(--serif)!important;color:#000!important}
.dk-mid button{background:none;border:0;font:inherit;color:#000;cursor:pointer;letter-spacing:0;text-transform:none;height:100%}
.dk-step{padding:0 12px;font-size:20px!important;color:var(--muted)!important}
.dk-step:hover{color:#000!important}.dk-step:disabled{opacity:.25;cursor:default}
.dk-cur{display:flex;flex-direction:column;justify-content:center;align-items:center;gap:4px;min-width:150px;padding:0 6px}
.dk-cur span{font:500 14px/1 var(--sans)}
.dk-cur i{font:500 9px/1 var(--sans);font-style:normal;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
.dk-cur:hover i{color:#000}
.dk-badge{font:400 22px/1 var(--serif)}
.dk-menu{position:absolute;left:50%;bottom:calc(100% + 10px);transform:translate(-50%,6px);opacity:0;pointer-events:none;
  background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.06),0 18px 50px rgba(0,0,0,.12);padding:22px 24px 18px;
  display:grid;grid-auto-flow:column;grid-auto-columns:250px;gap:6px 28px;align-items:start;transition:opacity .18s,transform .18s;white-space:normal}
.dock.open .dk-menu{opacity:1;transform:translate(-50%,0);pointer-events:auto}
.dk-col{display:flex;flex-direction:column;gap:2px}
.dk-col h6:empty{display:none}
.dk-col h6{margin:0 0 6px;font:500 9.5px/1 var(--sans);letter-spacing:.24em;text-transform:uppercase;color:var(--muted)}
.dk-col a{padding:8px 10px;margin:0 -10px;display:block}
.dk-col a b{display:block;font:500 14px/1.2 var(--sans);color:#111}
.dk-col a span{display:block;font:400 12px/1.35 var(--sans);color:var(--muted);margin-top:3px}
.dk-col a:hover{background:#f4f4f2}
.dk-col a.on{background:#111}.dk-col a.on b{color:#fff}.dk-col a.on span{color:#bbb}
.dk-tip{position:absolute;left:50%;bottom:calc(100% + 12px);transform:translateX(-50%);background:#111;color:#fff;
  font:500 12px/1 var(--sans);padding:9px 12px;opacity:0;pointer-events:none;transition:opacity .4s}
.dk-tip::after{content:"";position:absolute;left:50%;top:100%;margin-left:-5px;border:5px solid transparent;border-top-color:#111}
.dk-tip.show{opacity:1}
.dock.open .dk-tip{opacity:0}
.tour-dim{position:fixed;inset:0;z-index:90;pointer-events:auto}
.tour-hole{position:fixed;z-index:91;box-shadow:0 0 0 9999px rgba(10,10,10,.55);border-radius:6px;pointer-events:none;
  transition:all .35s cubic-bezier(.2,.8,.2,1)}
.tour-card{position:fixed;z-index:92;width:min(360px,calc(100vw - 32px));background:#fff;color:#111;padding:20px 22px 16px;
  box-shadow:0 20px 60px rgba(0,0,0,.25);transition:top .35s,left .35s}
.tour-card .n{font:500 10px/1 var(--sans);letter-spacing:.22em;text-transform:uppercase;color:var(--muted)}
.tour-card h4{font:400 30px/1.05 var(--serif);margin:10px 0 8px}
.tour-card p{font-size:14px;line-height:1.5;margin:0 0 16px;color:#333}
.tour-card .row{display:flex;gap:8px;justify-content:space-between;align-items:center}
.tour-card button{font:500 11px/1 var(--sans);letter-spacing:.14em;text-transform:uppercase;border:0;padding:11px 14px;cursor:pointer}
.tour-card .go{background:#000;color:#fff}.tour-card .back{background:transparent;color:#000;border:1px solid #000}
.tour-card .skip{background:none;color:var(--muted);padding:11px 0}
.dk-col a:last-child[data-go=""]{border-top:1px solid #ececec;margin-top:6px;padding-top:12px}
@media (max-width:700px){.sel-head{left:16px;right:16px;top:18px}.sel-num{left:16px;top:auto;bottom:130px}
  .sel-count{right:8px;top:auto;bottom:96px;transform:none;height:190px;width:170px}.stage{padding-top:170px}.dock>a,.dock>span{padding:0 12px}.dk-cur{min-width:110px}.dk-step{padding:0 8px}
  .dk-menu{grid-auto-flow:row;grid-template-columns:1fr;width:min(92vw,380px);max-height:70vh;overflow:auto}.sel-meta div:nth-child(n+4){display:none}}
""".replace("GRAIN_URL", GRAIN.replace('"', "'"))


def tape_html(item, i):
    layers = "".join(f'<div class="ly" style="transform:translateZ({-k * 2.2}px)"></div>' for k in range(1, 8))
    screws = "".join(f'<div class="screw" style="{p}"></div>' for p in (
        "left:3%;top:4.5%", "right:3%;top:4.5%", "left:3%;bottom:6%", "right:3%;bottom:6%"))
    return (f'<div class="tape" style="--body:{item["body"]};--edge:{item["edge"]};--label:{item.get("label", "#efe9dc")}">'
            f'{layers}<div class="face">{screws}<div class="label"><div class="lt"><span class="side">{chr(65 + i % 2)}</span>'
            f'<span class="ttl">{html.escape(item["title"])}</span></div>'
            f'<div class="sub"><span>{html.escape(item.get("sub", ""))}</span><span>{i + 1:02d}</span></div></div>'
            f'<div class="win"><div class="reel"></div><div class="reel"></div></div>'
            f'<div class="foot"><i style="left:14%"></i><i style="left:32%"></i><i style="right:32%"></i><i style="right:14%"></i></div>'
            f'</div></div>')


def disc_html(item, i):
    layers = "".join(f'<div class="ly" style="transform:translateZ({-k * 1.3}px)"></div>' for k in range(1, 4))
    blank = " blank" if item.get("blank") else ""
    ring = (f'<div class="ring" style="--ink:{item["ink"]}">{html.escape(item["mark"])}</div>' if item.get("mark") else "")
    return (f'<div class="disc{blank}">{layers}<div class="face"></div><div class="ink">'
            f'<div class="name">{html.escape(item["title"])}</div>{ring}'
            f'<div class="note">{html.escape(item.get("note", ""))}</div><div class="brand">AUDITION · CD-R 700MB</div>'
            f'</div></div>')


SELECTOR_JS = r"""
(function(){
const items = JSON.parse(document.getElementById('sel-data').textContent);
const objs = [...document.querySelectorAll('#objs > template')].map(t => t.innerHTML);
const sel = document.getElementById('sel'), holder = document.getElementById('holder'), tilt = document.getElementById('tilt');
const title = document.getElementById('sel-title'), meta = document.getElementById('sel-meta');
const num = document.getElementById('sel-num'), count = document.getElementById('sel-count');
const views = [...document.querySelectorAll('.view')];
const dockLinks = [...document.querySelectorAll('.dock [data-go]')];
const dock = document.querySelector('.dock'), dkCur = document.getElementById('dk-cur');
let idx = 0, busy = false;
const pose = (x, y) => `rotateX(${54 - y * 8}deg) rotateY(${x * 8}deg) rotateZ(${-22 + x * 6}deg)`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
count.innerHTML = '<div class="wheel" id="wheel">' + items.map((it, k) =>
  `<i data-k="${k}"><b>${esc(it.title || it.head)}</b>${String(k + 1).padStart(2, '0')}</i>`).join('') + '</div>';
const wheel = document.getElementById('wheel'), cells = [...wheel.children];
const STEP = 22, R = 110;             // degrees per item, drum radius in px
let wpos = 0;                         // where the drum is, in items (fractional while scrolling)
function drum(live){
  wheel.classList.toggle('live', !!live);
  cells.forEach((c, k) => {
    const off = k - wpos, ang = -off * STEP;
    c.style.transform = `translateZ(${-R}px) rotateX(${ang}deg) translateZ(${R}px)`;
    c.style.opacity = Math.abs(ang) > 80 ? 0 : Math.max(0, 1 - Math.abs(off) * 0.22);
    c.classList.toggle('on', Math.round(wpos) === k);
  });
}
function paint(){
  const it = items[idx];
  title.textContent = it.head;
  meta.innerHTML = it.meta.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
  num.innerHTML = `${it.num}<small>${it.numLabel || ''}</small>`;
  tilt.innerHTML = objs[idx] + '<div class="drop"></div>';
  tilt.style.transform = pose(0, 0);
  wpos = idx; drum(false);
}
let queued = null;
function goTo(n){ if (busy) { queued = n; return; } go(n - idx); }
function go(d){
  const n = Math.max(0, Math.min(items.length - 1, idx + d));
  if (n === idx) return;
  wpos = n; drum(false);
  if (busy) { queued = n; return; }
  busy = true;
  holder.classList.add(d > 0 ? 'out-up' : 'out-down');
  setTimeout(() => {
    idx = n; paint();
    holder.className = 'holder ' + (d > 0 ? 'in-up' : 'in-down');
    requestAnimationFrame(() => requestAnimationFrame(() => { holder.className = 'holder spin'; }));
    setTimeout(() => { busy = false; holder.classList.remove('spin');
      if (queued !== null && queued !== idx) { const q = queued; queued = null; goTo(q); } else queued = null; }, 650);
  }, 420);
}
function open(){
  const it = items[idx];
  if (it.href) { location.href = it.href; return; }
  location.hash = it.view;
}
function route(){
  const h = location.hash.slice(1);
  const v = views.find(x => x.id === h);
  sel.hidden = !!v;
  views.forEach(x => x.classList.toggle('on', x === v));
  dockLinks.forEach(a => a.classList.toggle('on', a.dataset.go === (v ? h : '')));
  if (dkCur) { const cur = dockLinks.find(a => a.closest('.dk-menu') && a.dataset.go === (v ? h : ''));
    dkCur.firstChild.textContent = v && cur ? cur.dataset.label : dkCur.dataset.home; }
  const order = stepOrder(), k = order.indexOf(v ? h : '');
  document.querySelectorAll('.dk-step').forEach(b => b.disabled = v ? (k + +b.dataset.step < 0 || k + +b.dataset.step >= order.length) : false);
  if (dock) dock.classList.remove('open');
  if (!v) { const k = items.findIndex(i => i.view === sessionStorage.getItem('aud-last')); if (k >= 0 && k !== idx) { idx = k; paint(); } }
  else { try { sessionStorage.setItem('aud-last', h); } catch(e) {} window.scrollTo(0, 0); }
}
let snapT = null;
const clampPos = p => Math.max(-0.4, Math.min(items.length - 0.6, p));
function settle(){
  const target = Math.max(0, Math.min(items.length - 1, Math.round(wpos)));
  wpos = target; drum(false);
  if (target !== idx) goTo(target);
}
sel.addEventListener('wheel', e => {
  e.preventDefault();
  const d = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;   // lines -> px
  wpos = clampPos(wpos + d / 110);
  drum(true);
  clearTimeout(snapT); snapT = setTimeout(settle, 140);
}, {passive: false});
// drag the drum directly
let dragY = null, dragPos = 0;
count.addEventListener('pointerdown', e => { dragY = e.clientY; dragPos = wpos; count.setPointerCapture(e.pointerId); e.stopPropagation(); });
count.addEventListener('pointermove', e => { if (dragY === null) return; wpos = clampPos(dragPos + (dragY - e.clientY) / 30); drum(true); });
count.addEventListener('pointerup', e => {
  if (dragY === null) return;
  const moved = Math.abs(e.clientY - dragY) > 4; dragY = null;
  if (moved) { settle(); return; }
  const k = e.target.closest('i') && +e.target.closest('i').dataset.k;
  if (k !== undefined && k !== null && !isNaN(k)) { wpos = k; drum(false); goTo(k); }
});
let ty = null;
sel.addEventListener('touchstart', e => { ty = e.touches[0].clientY; }, {passive: true});
sel.addEventListener('touchend', e => { if (ty === null) return; const d = ty - e.changedTouches[0].clientY; if (Math.abs(d) > 40) go(d > 0 ? 1 : -1); ty = null; });
document.addEventListener('keydown', e => {
  if (sel.hidden) { if (e.key === 'Escape') location.hash = ''; return; }
  if (['ArrowDown', 'ArrowRight', 'PageDown', 'j'].includes(e.key)) go(1);
  if (['ArrowUp', 'ArrowLeft', 'PageUp', 'k'].includes(e.key)) go(-1);
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
});
sel.addEventListener('mousemove', e => {
  const x = e.clientX / innerWidth - .5, y = e.clientY / innerHeight - .5;
  tilt.style.transform = pose(x, y);
});
holder.addEventListener('click', open);

dockLinks.forEach(a => a.addEventListener('click', e => { e.preventDefault(); dock.classList.remove('open');
  if (location.hash.slice(1) === a.dataset.go) route(); else location.hash = a.dataset.go; }));
if (dkCur) dkCur.addEventListener('click', e => { e.stopPropagation(); dock.classList.toggle('open'); tipOff(); });
function stepOrder(){ return dockLinks.filter(a => a.closest('.dk-menu') && a.dataset.go).map(a => a.dataset.go); }
document.querySelectorAll('.dk-step').forEach(b => b.addEventListener('click', e => {
  e.stopPropagation(); const d = +b.dataset.step;
  if (sel.hidden) { const o = stepOrder(), k = o.indexOf(location.hash.slice(1)); if (o[k + d]) location.hash = o[k + d]; }
  else go(d);
}));
document.addEventListener('keydown', e => {
  if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
  if (e.key === 'm' || e.key === 'M') { dock.classList.toggle('open'); tipOff(); }
  if (sel.hidden && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) {
    const o = stepOrder(), k = o.indexOf(location.hash.slice(1)), n = o[k + (e.key === 'ArrowRight' ? 1 : -1)];
    if (n) location.hash = n; }
});
const tip = document.getElementById('dk-tip');
function tipOff(){ if (tip) tip.classList.remove('show'); try { localStorage.setItem('aud-tip', '1'); } catch(e) {} }
try { if (tip && !localStorage.getItem('aud-tip')) { setTimeout(() => tip.classList.add('show'), 700); setTimeout(tipOff, 6000); } } catch(e) {}
document.addEventListener('click', e => { if (dock && !dock.contains(e.target)) dock.classList.remove('open'); });
document.addEventListener('keydown', e => { if (e.key === 'Escape' && dock) dock.classList.remove('open'); });
window.addEventListener('hashchange', route);
paint(); route();
})();
"""


def dock(groups, home_label, home_href=None, badge=""):
    """One pill: home | ‹ current view (opens the menu) › | optional badge.
    groups: [(heading, [(view_id, label, description), ...]), ...]; view_id "" is the selector."""
    cols = "".join(f'<div class="dk-col"><h6>{html.escape(h)}</h6>'
                   + "".join((f'<a data-tour="1"' if v == "!tour" else f'<a data-go="{v}"')
                             + f' data-label="{html.escape(l)}"><b>{html.escape(l)}</b>'
                             f'<span>{html.escape(d)}</span></a>' for v, l, d in links) + "</div>"
                   for h, links in groups)
    home = (f'<a class="dk-home" href="{home_href}" title="Back to the shelf of strategies">← Shelf</a>' if home_href
            else '<a class="dk-home dk-word" data-go="" title="The shelf">Audition</a>')
    return (f'<nav class="dock">{home}<div class="dk-mid"><button class="dk-step" data-step="-1" title="Previous (←)">‹</button>'
            f'<button class="dk-cur" id="dk-cur" data-home="{html.escape(home_label)}" aria-haspopup="true" title="All sections and tools (M)">'
            f'<span>{html.escape(home_label)}</span><i>menu ▴</i></button>'
            f'<button class="dk-step" data-step="1" title="Next (→)">›</button></div>'
            + (f'<span class="dk-badge" title="Grade">{badge}</span>' if badge else "")
            + f'<div class="dk-menu" role="menu">{cols}</div><div class="dk-tip" id="dk-tip">Everything lives in this menu</div></nav>')


TOUR_JS = r"""
(function(){
const steps = JSON.parse(document.getElementById('tour-data').textContent || '[]');
const key = document.getElementById('tour-data').dataset.key;
if (!steps.length) return;
let i = 0, dim, hole, card;
function el(sel){ if (!sel) return null; const e = document.querySelector(sel); if (!e) return null;
  const r = e.getBoundingClientRect(); return r.width && r.height ? r : null; }
function show(){
  const st = steps[i];
  if (st.hash !== undefined && location.hash.slice(1) !== st.hash) location.hash = st.hash;
  if (st.open) document.querySelector('.dock').classList.add('open'); else document.querySelector('.dock').classList.remove('open');
  setTimeout(() => {
    const r = el(st.target), pad = 8;
    if (r) Object.assign(hole.style, {left: r.left - pad + 'px', top: r.top - pad + 'px', width: r.width + 2 * pad + 'px', height: r.height + 2 * pad + 'px'});
    else Object.assign(hole.style, {left: innerWidth / 2 + 'px', top: innerHeight / 2 + 'px', width: '0px', height: '0px'});
    card.innerHTML = `<div class="n">${st.n || 'Tutorial'} · ${i + 1} of ${steps.length}</div><h4>${st.title}</h4><p>${st.text}</p>
      <div class="row"><button class="skip">${i === steps.length - 1 ? '' : 'Skip tutorial'}</button><span style="display:flex;gap:8px">
      ${i ? '<button class="back">Back</button>' : ''}<button class="go">${i === steps.length - 1 ? 'Done' : 'Next'}</button></span></div>`;
    const cw = card.offsetWidth, ch = card.offsetHeight;
    let left, top;
    if (!r) { left = (innerWidth - cw) / 2; top = (innerHeight - ch) / 2; }
    else {
      left = Math.min(Math.max(16, r.left + r.width / 2 - cw / 2), innerWidth - cw - 16);
      top = r.top - ch - 24 > 16 ? r.top - ch - 24 : Math.min(r.bottom + 24, innerHeight - ch - 16);
      if (r.height > innerHeight * .5) { top = (innerHeight - ch) / 2; left = r.left > cw + 40 ? r.left - cw - 24 : Math.min(r.right + 24, innerWidth - cw - 16); }
    }
    card.style.left = left + 'px'; card.style.top = top + 'px';
    card.querySelector('.go').onclick = () => i === steps.length - 1 ? end() : (i++, show());
    const b = card.querySelector('.back'); if (b) b.onclick = () => { i--; show(); };
    card.querySelector('.skip').onclick = end;
  }, st.hash !== undefined || st.open ? 380 : 0);
}
function start(){
  if (dim) return;
  i = 0;
  dim = document.createElement('div'); dim.className = 'tour-dim';
  hole = document.createElement('div'); hole.className = 'tour-hole';
  card = document.createElement('div'); card.className = 'tour-card';
  document.body.append(dim, hole, card);
  dim.onclick = e => e.stopPropagation();
  show();
}
function end(){
  [dim, hole, card].forEach(x => x && x.remove()); dim = null;
  document.querySelector('.dock').classList.remove('open');
  try { localStorage.setItem(key, '1'); localStorage.setItem('aud-tip', '1'); } catch(e) {}
}
document.addEventListener('keydown', e => { if (dim && e.key === 'Escape') end(); });
window.addEventListener('resize', () => dim && show());
document.querySelectorAll('[data-tour]').forEach(a => a.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); start(); }));
let seen = false; try { seen = !!localStorage.getItem(key); } catch(e) { seen = true; }
if (!seen && !location.hash) setTimeout(start, 900);
})();
"""


def page(title, items, objects_html, views_html, dock_html, hint, extra_css="", tour=None, tour_key="aud-tour"):
    """items: dicts with head, meta [(k,v)], num, numLabel, and view (an element id) or href."""
    tmpl = "".join(f"<template>{o}</template>" for o in objects_html)
    data = json.dumps(items).replace("</", "<\\/")
    tour_json = json.dumps(tour or []).replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>{FONTS}<style>{BASE_CSS}{extra_css}</style></head>
<body><div class="grain" aria-hidden="true"></div>
<section class="sel" id="sel" aria-label="selector">
  <div class="sel-head"><div class="sel-title" id="sel-title"></div><div class="sel-meta" id="sel-meta"></div></div>
  <div class="sel-num" id="sel-num"></div><div class="sel-count mono" id="sel-count"></div>
  <div class="stage"><div class="holder" id="holder" role="button" tabindex="0" aria-label="open"><div class="tilt" id="tilt"></div></div></div>
  <div class="sel-hint">{html.escape(hint)}</div>
</section>
<div id="objs" hidden>{tmpl}</div>
{views_html}
{dock_html}
<script type="application/json" id="sel-data">{data}</script>
<script type="application/json" id="tour-data" data-key="{tour_key}">{tour_json}</script>
<script>{SELECTOR_JS}</script><script>{TOUR_JS}</script></body></html>"""
