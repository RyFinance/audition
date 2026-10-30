"""reports/index.html: every audited strategy as a white CD-R with its name written on it.
The last disc is blank and holds the template for writing a new one."""

from __future__ import annotations

import html
import json
from pathlib import Path

from . import ui

TEMPLATE = '''"""What the strategy does, in one line."""

import pandas as pd

NAME = "My strategy"
PARAMS = {"lookback": 60, "top_n": 3}          # numbers here get sensitivity-tested
# UNIVERSE = ["AAPL", "MSFT", "NVDA"]          # optional: your own tickers (home market)
# UNIVERSE = {"us": [...], "jp": [...]}        # optional: per market
# UNIVERSE = "sp500"                           # point-in-time S&P 500; use data.eligible
# UNIVERSE = "byo:my_data"                     # a dataset you uploaded under Data
# DATA = "polygon"                             # price source: yahoo (default), polygon, tiingo, alpaca
DECLARE = {
    "variants_tried": 10,          # every version you tested before this one
    "oos_start": "2022-01-01",     # where your out-of-sample period begins
    "oos_looks": 1,                # how many times you looked at it
    # "reported": {"cagr": 0.25, "sharpe": 1.6, "max_dd": -0.18},
    # "risk_controls": ["max_position", "daily_loss_limit", "kill_switch"],
    # "paper_trading_days": 0,
}


def weights(data, lookback, top_n):
    """data.open/high/low/close/volume are DataFrames (dates x tickers).
    Return target weights decided at each close; Audition fills them at the next open."""
    mom = data.close / data.close.shift(lookback) - 1
    rank = mom.rank(axis=1, ascending=False)
    return (rank <= top_n).astype(float) / top_n
'''

INK = {"A": "#2e7d57", "B": "#2e7d57", "C": "#b5760f", "D": "#c23b2a", "F": "#c23b2a"}

LIB_CSS = """
.brand{font:500 12px/1 var(--sans);letter-spacing:.32em;text-transform:uppercase}
.vt{font:400 64px/1 var(--serif);margin:14px 0 6px}
.lead{font:400 26px/1.25 var(--serif);max-width:760px;margin:20px 0 28px}
pre{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:18px 0;overflow:auto;font:13px/1.55 var(--mono);margin:0}
.cmd{display:flex;gap:12px;align-items:center;margin:22px 0;flex-wrap:wrap}
.cmd code{font:13px/1 var(--mono);background:#fff;padding:12px 14px}
button{font:500 12px/1 var(--sans);letter-spacing:.14em;text-transform:uppercase;background:#000;color:#fff;border:0;padding:13px 18px;cursor:pointer}
.steps{color:var(--muted);font-size:14px;max-width:760px}
"""


IMPORT_CSS = """
.imp{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(260px,1fr);gap:28px;align-items:start}
@media (max-width:900px){.imp{grid-template-columns:1fr}}
.dropzone{border:1px dashed #9a9a9a;padding:18px;text-align:center;color:var(--muted);font-size:13px;cursor:pointer;background:rgba(255,255,255,.35);transition:background .2s,border-color .2s}
.dropzone.over{background:#fff;border-color:#000;color:#000}
.dropzone b{color:var(--fg);font-weight:500}
.ed{width:100%;min-height:440px;margin-top:12px;resize:vertical;border:1px solid var(--line);background:#fbfbfa;color:var(--fg);
  font:13px/1.55 var(--mono);padding:16px;tab-size:4;outline:none}
.ed:focus{border-color:#000}
.side h3{font:500 10px/1 var(--sans);letter-spacing:.2em;text-transform:uppercase;margin:0 0 10px;color:var(--muted)}
.box{border-top:1px solid var(--fg);padding:12px 0 18px}
.kv{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding:8px 0;font-size:13px}
.kv span{color:var(--muted)}
.msg{font-size:13px;padding:6px 0 6px 12px;border-left:2px solid currentColor;margin:6px 0}
.msg.err{color:var(--red)}.msg.warn{color:var(--amber)}.msg.ok{color:var(--green)}
.opts label{display:inline-flex;align-items:center;gap:6px;margin:0 14px 10px 0;font-size:13px;cursor:pointer}
.opts input[type=number],.opts input[type=text]{font:13px var(--mono);border:1px solid var(--line);background:#fff;padding:8px 10px;width:100%}
.burn{width:100%;margin-top:14px;padding:16px;font-size:12px}
.burn:disabled{opacity:.35;cursor:not-allowed}
.log{font:12px/1.6 var(--mono);color:var(--muted);white-space:pre-wrap;max-height:260px;overflow:auto;margin:0;border:0;padding:0}
.spin-disc{width:84px;height:84px;border-radius:50%;margin:6px auto 14px;display:none;
  background:radial-gradient(circle,var(--bg) 0 13%,#e9e9e6 13.5% 20%,transparent 20.5%),
  conic-gradient(from 0deg,#fff,#f0f0ee,#e2f0ff,#fff2e0,#fff);box-shadow:inset 0 0 0 1px rgba(0,0,0,.12);animation:spin 1.2s linear infinite}
.burning .spin-disc{display:block}
.offline{display:none;border-left:2px solid var(--amber);padding:10px 14px;margin:0 0 18px;font-size:14px}
.offline code{font:13px var(--mono);background:#fff;padding:2px 6px}
.ghost{background:transparent;color:#000;border:1px solid #000}
button:disabled{opacity:.35;cursor:not-allowed}
.seg{display:inline-flex;border:1px solid #000;margin-bottom:12px}
.seg button{background:transparent;color:#000;padding:10px 16px}
.seg button.on{background:#000;color:#fff}
select{font:13px var(--sans);border:1px solid var(--line);background:#fff;padding:8px 10px;width:100%}
.sub{color:var(--muted);font-size:13px}
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th{font:500 10px/1 var(--sans);color:var(--muted);text-align:left;text-transform:uppercase;letter-spacing:.14em;padding:10px 10px 8px 0;border-bottom:1px solid var(--line)}
.tbl td{padding:10px 10px 10px 0;border-bottom:1px solid var(--line);vertical-align:top}
.tag{display:inline-block;font:500 10px/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;padding:5px 8px;border-radius:99px;border:1px solid currentColor;white-space:nowrap}
.v-plausible{color:var(--green)}.v-suspicious,.v-inconsistent{color:var(--amber)}.v-impossible{color:var(--red)}.v-unverifiable,.v-unreadable{color:var(--muted)}
.big{font:400 30px/1.2 var(--serif);max-width:860px;margin:6px 0 18px}
.conn{border-top:1px solid var(--line);padding:14px 0;display:grid;grid-template-columns:200px 1fr auto;gap:16px;align-items:start}
.conn .f{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.conn input{font:13px var(--mono);border:1px solid var(--line);background:#fff;padding:8px 10px;width:100%}
.conn input[type=radio]{width:auto;margin:0}
.conn .acts{display:flex;gap:8px}
.conn .acts button{padding:10px 12px}
@media (max-width:800px){.conn{grid-template-columns:1fr}}
.pane{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(260px,1fr);gap:28px;align-items:start}
@media (max-width:900px){.pane{grid-template-columns:1fr}}
h2.sec{font:500 11px/1 var(--sans);letter-spacing:.24em;text-transform:uppercase;margin:44px 0 12px;display:flex;gap:12px;align-items:center}
h2.sec::after{content:"";flex:1;height:1px;background:var(--line)}
"""

IMPORT_VIEW = r"""<main class="view" id="new"><div class="wrap"><div class="brand">Audition · blank disc</div>
<h1 class="vt">Import a strategy</h1>
<p class="lead">Drop in a .py file or paste the code. Audition checks it, runs all 71 checks, and burns the result onto a new disc.</p>
<div class="offline" id="offline">This page is open as a static file, so it can't run audits. Start the app with
<code>.venv/bin/python -m audition serve</code> in the audition folder and open <code>http://127.0.0.1:8767</code>.
You can still copy the template below.</div>
<div class="imp">
 <div>
  <div class="seg" id="seg"><button class="on" data-m="py" type="button">Python</button><button data-m="tr" type="button">Translate</button></div>
  <div id="m-py">
  <div class="dropzone" id="drop"><b>Drop a .py file here</b> or click to choose one · or paste below</div>
  <input type="file" id="file" accept=".py,text/x-python,text/plain" hidden>
  <textarea class="ed" id="code" spellcheck="false">__TEMPLATE__</textarea>
  <div style="display:flex;gap:10px;margin-top:10px;flex-wrap:wrap">
   <button class="ghost" id="tplbtn" type="button">Reset to template</button>
   <button class="ghost" id="copybtn" type="button">Copy code</button>
  </div>
  </div>
  <div id="m-tr" hidden>
   <div class="opts" style="display:grid;grid-template-columns:200px 1fr;gap:10px">
    <select id="trkind"><option value="auto">Detect automatically</option><option>TradingView Pine Script</option>
     <option>QuantConnect (Python/C#)</option><option>Backtrader / Zipline</option><option>MetaTrader MQL</option>
     <option>Plain English rule</option><option>Tweet or post</option></select>
    <input type="text" id="trhint" placeholder="Optional notes: tickers, timeframe, what to assume">
   </div>
   <textarea class="ed" id="trsrc" spellcheck="false" style="min-height:360px" placeholder="Paste a Pine script, someone else's backtest code, or a rule in words, e.g.

Buy SPY when the 14-day RSI closes under 30, sell when it closes over 55. Otherwise hold T-bills."></textarea>
   <div style="display:flex;gap:10px;margin-top:10px;align-items:center;flex-wrap:wrap">
    <button id="trbtn" type="button">Translate to Python</button><span class="sub" id="trmsg">Claude writes the file; you review it before burning.</span>
   </div>
  </div>
 </div>
 <div class="side">
  <div class="box"><h3>Check</h3><div id="val"><div class="msg warn">Waiting for code…</div></div></div>
  <div class="box opts"><h3>Run</h3>
   <div style="margin-bottom:8px;font-size:12px;color:var(--muted)">Replay in other markets</div>
   <label><input type="checkbox" value="us" checked> US</label><label><input type="checkbox" value="uk" checked> UK</label>
   <label><input type="checkbox" value="eu" checked> Europe</label><label><input type="checkbox" value="jp" checked> Japan</label>
   <label><input type="checkbox" value="crypto"> Crypto</label>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px">
    <div><div style="font-size:12px;color:var(--muted);margin-bottom:4px">Capital ($)</div><input type="number" id="capital" value="1000000" min="1000" step="1000"></div>
    <div><div style="font-size:12px;color:var(--muted);margin-bottom:4px">File name</div><input type="text" id="fname" placeholder="my_strategy.py"></div>
   </div>
   <label style="margin-top:12px"><input type="checkbox" id="quick"> Quick pass (fewer placebo and sensitivity runs)</label>
   <label><input type="checkbox" id="rehearsal" checked> Rehearsal: keep the last two years locked in the vault</label>
   <button class="burn" id="burn" disabled>Burn to disc</button>
  </div>
  <div class="box" id="progress" style="display:none"><h3>Burning</h3><div class="spin-disc"></div>
   <pre class="log" id="log"></pre><div id="done" style="margin-top:12px"></div></div>
 </div>
</div>
<p class="steps" style="margin-top:28px">Imported files run on this computer with full Python permissions, the same as
<code>audition run</code>, and are saved to <code>strategies/imported/</code>. Only import code you trust.</p>
</div></main>
<script>
(function(){
const $ = id => document.getElementById(id);
const code = $('code'), val = $('val'), burn = $('burn'), drop = $('drop'), file = $('file');
const TEMPLATE = code.value;
let online = false, valid = false, timer = null, filename = '';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
fetch('/api/ping').then(r => r.ok ? r.json() : null).then(j => { online = !!(j && j.ok); if (!online) $('offline').style.display = 'block'; check(); })
  .catch(() => { $('offline').style.display = 'block'; check(); });
function render(v){
  valid = v.ok;
  let h = '';
  if (v.name) h += `<div class="kv"><span>Name</span><b>${esc(v.name)}</b></div>`;
  const p = Object.entries(v.params || {});
  h += `<div class="kv"><span>Parameters</span><b>${p.length ? esc(p.map(([k, x]) => k + '=' + x).join(', ')) : 'none'}</b></div>`;
  h += `<div class="kv"><span>Universe</span><b>${esc(v.universe || 'Audition defaults')}</b></div>`;
  h += `<div class="kv"><span>Declared</span><b>${esc((v.declare_keys || []).join(', ') || 'nothing')}</b></div>`;
  (v.errors || []).forEach(e => h += `<div class="msg err">${esc(e)}</div>`);
  (v.warnings || []).forEach(e => h += `<div class="msg warn">${esc(e)}</div>`);
  if (v.ok) h += `<div class="msg ok">Ready to audit.</div>`;
  val.innerHTML = h;
  burn.disabled = !(valid && online);
}
function check(){
  if (!online) { val.innerHTML = '<div class="msg warn">Start the server to check and run code.</div>'; burn.disabled = true; return; }
  fetch('/api/validate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({code: code.value})})
    .then(r => r.json()).then(render).catch(() => {});
}
code.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(check, 400); });
code.addEventListener('keydown', e => {
  if (e.key === 'Tab') { e.preventDefault(); const s = code.selectionStart; code.setRangeText('    ', s, code.selectionEnd, 'end'); }
});
function load(f){
  if (!f) return;
  filename = f.name; $('fname').value = f.name;
  f.text().then(t => { code.value = t; check(); });
}
drop.addEventListener('click', () => file.click());
file.addEventListener('change', () => load(file.files[0]));
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => load(e.dataTransfer.files[0]));
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (!f || !/\.py$|\.txt$/i.test(f.name)) return;
  if (location.hash !== '#new') location.hash = 'new';   // dropping onto the shelf opens the import view
  load(f);
});
$('tplbtn').addEventListener('click', () => { code.value = TEMPLATE; filename = ''; $('fname').value = ''; check(); });
$('copybtn').addEventListener('click', e => { navigator.clipboard.writeText(code.value); e.target.textContent = 'Copied'; });
burn.addEventListener('click', () => {
  const regions = [...document.querySelectorAll('.opts input[type=checkbox][value]')].filter(x => x.checked).map(x => x.value).join(',');
  burn.disabled = true;
  $('progress').style.display = 'block'; $('progress').classList.add('burning');
  $('log').textContent = ''; $('done').innerHTML = '';
  fetch('/api/run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
    code: code.value, filename: $('fname').value || filename, regions, capital: +$('capital').value, quick: $('quick').checked,
    rehearsal: $('rehearsal').checked})})
    .then(r => r.json()).then(j => {
      if (j.error) { $('log').textContent = j.error; $('progress').classList.remove('burning'); burn.disabled = false; return; }
      $('log').textContent = 'saved ' + j.file + '\n';
      poll(j.job);
    });
});
document.querySelectorAll('#seg button').forEach(b => b.addEventListener('click', () => mode(b.dataset.m)));
function mode(m){
  document.querySelectorAll('#seg button').forEach(x => x.classList.toggle('on', x.dataset.m === m));
  $('m-py').hidden = m !== 'py'; $('m-tr').hidden = m !== 'tr';
}
$('trbtn').addEventListener('click', () => {
  if (!online) { $('trmsg').textContent = 'Start the server first.'; return; }
  const b = $('trbtn'); b.disabled = true; $('trmsg').textContent = 'Translating… (up to a minute or two)';
  fetch('/api/translate', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source: $('trsrc').value, kind: $('trkind').value, hint: $('trhint').value})})
    .then(r => r.json()).then(j => waitJob(j.job, res => {
      b.disabled = false;
      if (!res || res.error) { $('trmsg').textContent = (res && res.error) || 'Translation failed.'; return; }
      code.value = res.code; filename = ''; $('fname').value = ''; mode('py'); render(res.validation);
      $('trmsg').textContent = 'Done via ' + res.via + '. Read the translation notes at the top of the file before burning.';
    }));
});
function waitJob(id, cb){
  fetch('/api/job/' + id).then(r => r.json()).then(j => j.status === 'running' ? setTimeout(() => waitJob(id, cb), 1200) : cb(j.result));
}
function poll(id){
  fetch('/api/job/' + id).then(r => r.json()).then(j => {
    $('log').textContent = j.log.join('\n'); $('log').scrollTop = 1e9;
    if (j.status === 'running') { setTimeout(() => poll(id), 1000); return; }
    $('progress').classList.remove('burning'); burn.disabled = false;
    $('done').innerHTML = j.status === 'done'
      ? `<div class="msg ok">Burned in ${j.seconds}s.</div><div style="display:flex;gap:10px;margin-top:10px"><a href="${esc(j.report)}"><button>Open the audit</button></a><a href="index.html"><button class="ghost">Back to the shelf</button></a></div>`
      : '<div class="msg err">The audit failed. The log above has the error.</div>';
  });
}
})();
</script>"""


def _p(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


TUTORIAL = Path(__file__).resolve().parent / "tutorial"


def install_tutorial(reports_dir: Path):
    """A clean checkout ships one pre-burned disc (examples/tutorial.py) so the shelf isn't empty."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    if any(reports_dir.glob("*.json")) or not TUTORIAL.exists():
        return
    import shutil
    for f in TUTORIAL.iterdir():
        if f.suffix in (".html", ".json"):
            shutil.copy(f, reports_dir / f.name)


def build(reports_dir: Path) -> Path:
    reports_dir = Path(reports_dir)
    install_tutorial(reports_dir)
    runs = []
    for f in reports_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if "grade" in d and (reports_dir / d.get("report", f.with_suffix(".html").name)).exists():
            runs.append(d)
    runs.sort(key=lambda d: d.get("generated", ""), reverse=True)

    items, objs = [], []
    for d in runs:
        s = d["summary"]
        audited = d["waterfall"][-1][1] if d.get("waterfall") else None
        it = dict(
            href=d.get("report", ""), head=d["strategy"], title=d["strategy"], mark=d["grade"],
            ink=INK.get(d["grade"], "#c23b2a"),
            note=f"sharpe {s['sharpe']:.2f} · {d.get('generated', '')[:10]}",
            num=f"{d['score'] * 100:.0f}", numLabel="audit score",
            meta=[("Grade", d["grade"]), ("Audited Sharpe", f"{audited:.2f}" if audited is not None else "n/a"),
                  ("CAGR", _p(s.get("cagr"))), ("Max drawdown", _p(s.get("max_dd"))),
                  ("File", html.escape(d.get("file", ""))), ("Audited", d.get("generated", ""))],
        )
        items.append(it)
    blank = dict(view="new", head="Blank disc", title="import a strat", blank=True, note="drop a .py file on it",
                 num=str(len(runs) + 1), numLabel="next disc",
                 meta=[("Import", "drop or paste a .py file"), ("Needs", "weights(data, **params)"),
                       ("Checks", "71, plus Monte Carlo"), ("Saves to", "this shelf")])
    items.append(blank)
    objs = [ui.disc_html(it, i) for i, it in enumerate(items)]

    new_view = IMPORT_VIEW.replace("__TEMPLATE__", html.escape(TEMPLATE)) + TOOLS_VIEW
    dock = ui.dock([
        ("Your strategies, in order", [("new", "1 · Add a strategy", "Drop a .py file, or translate Pine / English"),
                                       ("", f"2 · Shelf ({len(runs)})", "Every audit, one disc each"),
                                       ("callback", "3 · Paper trading", "Today's orders, your paper account, live vs backtest")]),
        ("Other tools", [("claims", "Check a claim", "Did someone's posted trades really happen?"),
                         ("data", "Price data", "Price sources and your own uploaded files")]),
        ("Help", [("setup", "Setup", "What's connected, and what's optional"),
                  ("about", "How it works", "The file, the vault, the grade"),
                  ("!tour", "Tutorial", "A one-minute guided tour")]),
    ], "Shelf")
    tour = [
        dict(title="Welcome to Audition", text="It audits trading strategies the way a sceptical reviewer would: your file says what "
             "to hold, Audition owns the data, fills, costs and statistics, then grades the result."),
        dict(target="#holder", title="One disc per strategy", text="Every audited strategy is a disc on this shelf. Scroll to flip "
             "through them and click one to open its report. The tutorial disc has three classic mistakes for you to find."),
        dict(target="#sel-count", title="Where you are", text="This wheel shows your place on the shelf. Drag it, scroll, or use the arrow keys."),
        dict(target=".dk-mid", title="Everything else is in this menu", text="Add a strategy, paper trade one, check someone's posted "
             "trades, connect data. Press M any time.", open=True),
        dict(target=".dk-menu", open=True, title="In order", text="Start with 1 · Add a strategy: drop a .py file or translate a "
             "Pine script or a rule written in English. Help has Setup and How it works."),
        dict(title="Try it", text="Open the tutorial disc, read the Verdict, then fix the file and re-import it. Restart this tour "
             "from Help → Tutorial."),
    ]
    doc = ui.page("Audition · Library", items, objs, new_view, dock,
                  "Scroll the shelf · click a disc to open its audit", extra_css=LIB_CSS + IMPORT_CSS,
                  tour=tour, tour_key="aud-tour-shelf")
    out = reports_dir / "index.html"
    out.write_text(doc)
    return out


TOOLS_VIEW = r"""<main class="view" id="data"><div class="wrap"><div class="brand">Audition · data</div>
<h1 class="vt">Price data</h1>
<p class="lead">Where prices come from, and your own datasets. Keys stay in
<code>config/connections.json</code> on this computer and never go into a report.</p>
<div class="offline tool-off">Start <code>audition serve</code> to manage data sources.</div>
<h2 class="sec">Price source</h2><div id="providers"></div>
<h2 class="sec">Your datasets</h2>
<div class="pane"><div>
 <div class="dropzone" id="bdrop"><b>Drop a CSV, Parquet, Excel or JSON file</b> or click to choose<br>
  long (date, ticker, open, high, low, close, volume) or wide (date plus one close column per ticker)</div>
 <input type="file" id="bfile" accept=".csv,.txt,.parquet,.json,.xlsx,.xls" hidden>
 <div class="opts" style="display:grid;grid-template-columns:1fr auto;gap:10px;margin-top:10px">
  <input type="text" id="bname" placeholder="dataset name, e.g. my_futures"><button id="bup" type="button">Check &amp; save</button></div>
 <div id="bq" style="margin-top:14px"></div>
</div><div class="side"><div class="box"><h3>Saved</h3><div id="blist"><div class="sub">None yet.</div></div>
 <p class="sub" style="margin-top:12px">Use one in a strategy with <code>UNIVERSE = "byo:name"</code>. Pick a provider per file with
 <code>DATA = "polygon"</code>.</p></div></div></div>
</div></main>


<main class="view" id="setup"><div class="wrap"><div class="brand">Audition · setup</div>
<h1 class="vt">Setup</h1>
<p class="lead">Audition runs with nothing but Python and an internet connection. Everything else is optional; this checklist shows what's on.</p>
<div class="offline tool-off">The server isn't running. In the audition folder run <code>./setup.sh</code> once, then <code>./run.sh</code>, and open <code>http://127.0.0.1:8767</code>.</div>
<div id="checks"></div>
<h2 class="sec">From a terminal</h2>
<pre>git clone https://github.com/RyFinance/audition.git && cd audition
./setup.sh          # Python env + dependencies (once)
./run.sh            # the app at http://127.0.0.1:8767

.venv/bin/python -m audition run my_strategy.py --rehearsal
.venv/bin/python -m audition final my_strategy.py</pre>
<div style="display:flex;gap:10px;margin-top:22px"><button type="button" data-tour="1">Start the tutorial</button>
<a href="#about"><button class="ghost" type="button">How Audition works</button></a></div>
</div></main>

<main class="view" id="about"><div class="wrap"><div class="brand">Audition · how it works</div>
<h1 class="vt">How it works</h1>
<p class="lead">You write one Python function that says what to hold each day. Audition owns everything else (the data, the
clock, the fills, the costs and the statistics), so the strategy can't grade its own homework.</p>
<h2 class="sec">1 · The strategy file</h2>
<pre>NAME = "My strategy"
PARAMS = {"lookback": 60}                 # numbers here get stress-tested
UNIVERSE = ["SPY", "TLT", "GLD"]          # or "sp500", {"us": [...]}, "byo:my_data"
DECLARE = {"variants_tried": 12}          # be honest: it changes the grade

def weights(data, lookback):              # data.close etc: dates x tickers
    mom = data.close / data.close.shift(lookback) - 1
    return (mom > 0).astype(float) / 3    # decided at the close, filled next open</pre>
<h2 class="sec">2 · Rehearse, then audition</h2>
<p class="steps">The first rehearsal locks the last two years in a vault. Iterate as much as you like on the older data; every version is
counted. When it's ready, take the final audition once on the locked years. That run is sealed, and retakes are recorded.</p>
<h2 class="sec">3 · Read the report in order</h2>
<div class="kv"><span>1 · Verdict</span><b>Grade A–F, what dragged it down, the 10 checks that matter</b></div>
<div class="kv"><span>2 · Returns</span><b>Net of fees, spreads, taxes, impact and financing, plus Monte Carlo futures</b></div>
<div class="kv"><span>3 · Where the edge comes from</span><b>Known look-alikes (factors, ETFs, classic rules) and scheduled events</b></div>
<div class="kv"><span>4 · Out of sample</span><b>The holdout years, and the same rule in other markets</b></div>
<div class="kv"><span>5 · Evidence</span><b>All 71 checks with what was measured, and how to reproduce the run</b></div>
<h2 class="sec">4 · How the grade works</h2>
<p class="steps">Each check is measured and scored. Caps keep the worst problems from hiding: any failure caps it at B, a failed top-10 check
at D, lookahead or a money-losing strategy is an F, a hand-picked list of today's survivors is at most C, and a big gap between
what you reported and what Audition measures pulls it down.</p>
<h2 class="sec">5 · Then paper trade it</h2>
<p class="steps">Paper trading turns a disc's targets into orders on your own paper account (Alpaca, Interactive Brokers, Tradier,
Saxo, Trading 212, or a webhook; Schwab and Questrade read-only) and tracks live results against the backtest's range of outcomes.</p>
<p class="foot">Audition is research software. A backtest audit can't prove an edge, only find reasons to distrust one. Nothing here is
investment advice; you trade at your own risk.</p>
</div></main>

<main class="view" id="claims"><div class="wrap"><div class="brand">Audition · claim checker</div>
<h1 class="vt">Check a claim</h1>
<p class="lead">Paste trades someone posted. Audition checks each price against what actually printed that day (or that
5-minute bar, when a time is given and it's recent), the stated returns against the prices, and how suspiciously good the fills are.</p>
<div class="offline tool-off">Start <code>audition serve</code> to check claims.</div>
<div class="pane"><div>
 <textarea class="ed" id="ctext" spellcheck="false" style="min-height:260px" placeholder="One trade per line:
long NVDA 2025-03-04 @ 112.30 -> 2025-03-06 @ 120.10 +7%
short TSLA 2024-10-24 10:05 @ 214.10 -> 2024-10-25 @ 260.00
buy AAPL 6/3/2024 192.50 to 6/10/2024 196.10

or CSV with a header: ticker,side,entry_date,entry_price,exit_date,exit_price,entry_time,exit_time,claimed_return"></textarea>
 <div style="display:flex;gap:10px;margin-top:10px;align-items:center"><button id="cbtn" type="button">Check</button><span class="sub" id="cmsg"></span></div>
</div><div class="side"><div class="box"><h3>How it judges</h3>
 <div class="kv"><span>impossible</span><b>price never traded, or not a trading day</b></div>
 <div class="kv"><span>inconsistent</span><b>stated return doesn't match the prices</b></div>
 <div class="kv"><span>suspicious</span><b>both fills at the day's best price</b></div>
 <div class="kv"><span>fill quality</span><b>1 = bought the low / sold the high; luck averages 0.5</b></div>
</div></div></div>
<div id="cout"></div>
</div></main>

<main class="view" id="callback"><div class="wrap"><div class="brand">Audition · paper trading (callback)</div>
<h1 class="vt">Paper trading</h1>
<p class="lead">The paper stage. Today's targets from a disc, turned into whole-share orders against your own paper account and
sent market-on-open, the same fill the backtest assumes. Every run snapshots the account so live results can be held against
the backtest's own range of outcomes.</p>
<div class="offline tool-off">Start <code>audition serve</code> to run Callback.</div>
<div class="pane"><div class="opts">
 <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
  <div><div class="sub" style="margin-bottom:4px">Strategy</div><select id="kstrat"></select></div>
  <div><div class="sub" style="margin-bottom:4px">Broker</div><select id="kbroker">
   <option value="manual">Manual (no broker)</option></select></div>
 </div>
 <div id="kmanual" style="display:grid;grid-template-columns:160px 1fr;gap:10px;margin-top:10px">
  <div><div class="sub" style="margin-bottom:4px">Equity ($)</div><input type="number" id="kequity" value="100000"></div>
  <div><div class="sub" style="margin-bottom:4px">Current positions (shares)</div><input type="text" id="kpos" placeholder="AAPL 10, MSFT 4"></div>
 </div>
 <div style="display:flex;gap:10px;margin-top:14px"><button id="kprev" type="button">Preview orders</button>
  <button class="ghost" id="ksend" type="button">Send to paper</button><span class="sub" id="kmsg" style="align-self:center"></span></div>
</div><div class="side"><div class="box"><h3>Schedule it</h3><p class="sub">Run after each close, e.g. from cron at 16:30 ET:</p>
 <pre class="log" id="kcmd">audition callback strategy.py --broker alpaca_paper --send</pre>
 <p class="sub">Orders only go to paper, sandbox or demo accounts. Schwab and Questrade have no API paper trading, so Audition only reads them.</p></div></div></div>
<div id="kout"></div>
<h2 class="sec">Your paper account</h2><div id="brokers"></div>
</div></main>

<script>
(function(){
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct = (x, d=1) => x == null || isNaN(x) ? 'n/a' : (x * 100).toFixed(d) + '%';
const post = (u, b) => fetch(u, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(b)}).then(r => r.json());
const wait = (id, cb) => fetch('/api/job/' + id).then(r => r.json()).then(j => j.status === 'running' ? setTimeout(() => wait(id, cb), 1200) : cb(j.result || {}));
let online = false;
fetch('/api/ping').then(r => r.ok ? r.json() : null).then(j => { online = !!(j && j.ok); if (!online) off(); else { loadConn(); loadStrats(); loadSetup(); } }).catch(off);
function off(){ document.querySelectorAll('.tool-off').forEach(x => x.style.display = 'block'); }
function loadSetup(){ fetch('/api/setup').then(r => r.json()).then(list => {
  $('checks').innerHTML = list.map(c => `<div class="kv" style="align-items:center"><span style="color:var(--fg);font-weight:500">
    <span style="color:${c.ok ? 'var(--green)' : c.optional ? 'var(--muted)' : 'var(--red)'}">${c.ok ? '●' : '○'}</span> ${esc(c.name)}</span>
    <b style="font-weight:400;text-align:right">${esc(c.detail)}${c.go && !c.ok ? ` <a href="#${c.go}" style="color:var(--fg)">set up →</a>` : ''}</b></div>`).join(''); }); }
window.addEventListener('hashchange', () => { if (location.hash === '#setup' && online) loadSetup(); });

// ---------- data
let CONN = null;
function loadConn(){ fetch('/api/connections').then(r => r.json()).then(j => { CONN = j; drawConn(); drawByo(j.byo);
  $('kbroker').innerHTML = '<option value="manual">Manual (no broker)</option>' + Object.entries(j.brokers).map(([n, b]) =>
    `<option value="${n}">${esc(b.label)}${j.config[n] ? '' : ' (not linked)'}</option>`).join(''); cmd(); }); }
function connRow(name, spec, vals, isProvider){
  const fields = spec.fields.map(f => `<input data-f="${f}" placeholder="${esc((spec.placeholders || {})[f] || f.replace('_', ' '))}"
    value="${esc(vals[f] || '')}" autocomplete="off" ${(spec.secret || []).includes(f) ? 'type="password"' : ''}>`).join('');
  const def = isProvider ? `<label class="sub" style="display:flex;gap:6px;align-items:center;margin-top:8px"><input type="radio" name="defsrc" value="${name}"
    ${CONN.default === name ? 'checked' : ''}> default source</label>` : '';
  return `<div class="conn" data-name="${name}"><div><b>${esc(spec.label)}</b>${spec.kind ? `<div class="sub">${esc(spec.kind)}</div>` : ''}${def}</div>
    <div><div class="f">${fields || '<span class="sub">No key needed.</span>'}</div><div class="sub" style="margin-top:6px">${esc(spec.note)}</div>
    <div class="msg" data-msg style="display:none"></div></div>
    <div class="acts">${spec.fields.length ? '<button data-save type="button">Save</button>' : ''}<button class="ghost" data-test type="button">Test</button></div></div>`;
}
function drawConn(){
  $('providers').innerHTML = Object.entries(CONN.providers).map(([n, s]) => connRow(n, s, CONN.config[n] || {}, true)).join('');
  $('brokers').innerHTML = Object.entries(CONN.brokers).map(([n, s]) => connRow(n, s, CONN.config[n] || {}, false)).join('');
  document.querySelectorAll('.conn').forEach(row => {
    const name = row.dataset.name, msg = row.querySelector('[data-msg]');
    const say = (ok, t) => { msg.style.display = 'block'; msg.className = 'msg ' + (ok ? 'ok' : 'err'); msg.textContent = t; };
    const sv = row.querySelector('[data-save]');
    if (sv) sv.onclick = () => { const values = {}; row.querySelectorAll('[data-f]').forEach(i => values[i.dataset.f] = i.value);
      post('/api/connections', {name, values}).then(j => j.error ? say(false, j.error) : (CONN.config = j.config, say(true, 'Saved.'))); };
    row.querySelector('[data-test]').onclick = () => { say(true, 'Testing…'); post('/api/connections/test', {name}).then(j => say(j.ok, j.message)); };
  });
  document.querySelectorAll('input[name=defsrc]').forEach(r => r.onchange = () =>
    post('/api/connections', {name: 'default_source', values: {value: r.value}}).then(() => CONN.default = r.value));
}
function drawByo(list){
  $('blist').innerHTML = list && list.length ? list.map(d => `<div class="kv"><span><code>byo:${esc(d.name)}</code><br>${d.tickers} tickers · ${esc(d.start)} → ${esc(d.end)}
    ${d.issues && d.issues.length ? `<br><span style="color:var(--amber)">${d.issues.length} issue(s)</span>` : ''}</span>
    <b><button class="ghost" data-del="${esc(d.name)}" type="button" style="padding:8px 10px">Delete</button></b></div>`).join('') : '<div class="sub">None yet.</div>';
  document.querySelectorAll('[data-del]').forEach(b => b.onclick = () => { if (confirm('Delete dataset ' + b.dataset.del + '?'))
    post('/api/byo', {delete: b.dataset.del}).then(j => drawByo(j.byo)); });
}
let bFile = null;
$('bdrop').onclick = () => $('bfile').click();
$('bfile').onchange = () => pick($('bfile').files[0]);
['dragenter', 'dragover'].forEach(ev => $('bdrop').addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); $('bdrop').classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => $('bdrop').addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); $('bdrop').classList.remove('over'); }));
$('bdrop').addEventListener('drop', e => pick(e.dataTransfer.files[0]));
function pick(f){ if (!f) return; bFile = f; $('bdrop').innerHTML = '<b>' + esc(f.name) + '</b> · ' + (f.size / 1e6).toFixed(1) + ' MB';
  if (!$('bname').value) $('bname').value = f.name.replace(/\.[^.]+$/, '').toLowerCase().replace(/[^a-z0-9]+/g, '_'); }
$('bup').onclick = () => {
  if (!bFile) { $('bq').innerHTML = '<div class="msg warn">Choose a file first.</div>'; return; }
  $('bq').innerHTML = '<div class="msg warn">Checking…</div>';
  const rd = new FileReader();
  rd.onload = () => post('/api/byo', {name: $('bname').value, filename: bFile.name, content: rd.result.split(',')[1]}).then(j => {
    if (j.error) { $('bq').innerHTML = `<div class="msg err">${esc(j.error)}</div>`; return; }
    const q = j.quality;
    $('bq').innerHTML = `<div class="msg ok">Saved <code>byo:${esc(j.meta.name)}</code>: ${q.tickers} tickers, ${q.rows.toLocaleString()} rows, ${esc(q.start)} → ${esc(q.end)}.</div>`
      + (q.issues.length ? q.issues.map(i => `<div class="msg ${i.level === 'err' ? 'err' : 'warn'}">${esc(i.msg)}</div>`).join('') : '<div class="msg ok">No data problems found.</div>');
    drawByo(j.byo);
  });
  rd.readAsDataURL(bFile);
};

// ---------- claims
$('cbtn').onclick = () => {
  if (!online) return;
  $('cbtn').disabled = true; $('cmsg').textContent = 'Checking prices…'; $('cout').innerHTML = '';
  post('/api/claims', {text: $('ctext').value}).then(j => wait(j.job, r => {
    $('cbtn').disabled = false; $('cmsg').textContent = '';
    if (r.error) { $('cout').innerHTML = `<div class="msg err">${esc(r.error)}</div>`; return; }
    const rows = r.claims.map(c => `<tr><td><span class="tag v-${c.verdict}">${c.verdict}</span></td><td class="mono">${esc(c.raw)}</td>
      <td class="mono">${pct(c.return_from_prices)}</td>
      <td class="mono">${(c.legs || []).map(l => l.fill_quality == null ? '–' : l.fill_quality.toFixed(2)).join(' / ')}</td>
      <td>${(c.problems || []).map(esc).join('<br>') || '<span class="sub">all prices traded</span>'}</td></tr>`).join('');
    const cnt = Object.entries(r.counts).filter(([, v]) => v).map(([k, v]) => `<span class="v-${k}">${v} ${k}</span>`).join(' · ');
    $('cout').innerHTML = `<h2 class="sec">Result</h2><p class="big">${esc(r.verdict)}</p>
      <div class="sub" style="margin-bottom:12px">${cnt} · fill quality ${isNaN(r.hindsight) ? 'n/a' : r.hindsight.toFixed(2)} on ${r.fills} fills
      · compounded ${pct(r.compounded)}</div>
      <table class="tbl"><tr><th>Verdict</th><th>Claim</th><th>Return</th><th>Fill quality</th><th>What the tape shows</th></tr>${rows}</table>
      ${(r.unreadable || []).length ? `<div class="msg warn" style="margin-top:12px">Couldn't read: ${r.unreadable.map(esc).join(' | ')}</div>` : ''}`;
  }));
};

// ---------- callback
function loadStrats(){ fetch('/api/strategies').then(r => r.json()).then(list => {
  $('kstrat').innerHTML = list.map(s => `<option value="${esc(s.file)}">${esc(s.name)} (${esc(s.grade)})</option>`).join('') || '<option value="">No discs yet</option>';
  cmd(); }); }
function cmd(){ const f = $('kstrat').value.split('/').pop() || 'strategy.py', b = $('kbroker').value;
  $('kcmd').textContent = `audition callback ${f} --broker ${b}${b === 'manual' ? ' --equity ' + $('kequity').value : ' --send'}`;
  $('kmanual').style.display = b === 'manual' ? 'grid' : 'none'; $('ksend').disabled = b === 'manual'; }
['kstrat', 'kbroker', 'kequity'].forEach(id => $(id).addEventListener('change', cmd));
function parsePos(t){ const o = {}; t.split(/[,;\n]+/).forEach(p => { const m = p.trim().match(/^([A-Za-z.\-=^]+)\s*[: ]\s*(-?[\d.]+)$/); if (m) o[m[1].toUpperCase()] = +m[2]; }); return o; }
function cb(send){
  if (!online || !$('kstrat').value) return;
  if (send && !confirm('Send these orders to your ' + $('kbroker').selectedOptions[0].text + ' account as market-on-open orders?')) return;
  $('kmsg').textContent = send ? 'Sending…' : 'Running the strategy on the latest close…'; $('kprev').disabled = true;
  post('/api/callback', {file: $('kstrat').value, broker: $('kbroker').value, send, equity: +$('kequity').value, positions: parsePos($('kpos').value)})
    .then(j => j.error ? ($('kmsg').textContent = j.error, $('kprev').disabled = false) : wait(j.job, drawCb));
}
$('kprev').onclick = () => cb(false); $('ksend').onclick = () => cb(true);
function cone(t){
  if (!t.live) return '';
  const W = 760, H = 220, P = 34, n = t.live.length;
  const all = t.live.map(x => x[1]).concat(t.cone ? [...t.cone.p5, ...t.cone.p95] : []);
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all), X = i => P + i * (W - 2 * P) / Math.max(n - 1, 1), Y = v => H - P - (v - lo) / ((hi - lo) || 1) * (H - 2 * P);
  const line = a => a.map((v, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1)).join('');
  let band = '';
  if (t.cone) { const up = t.cone.p95.map((v, i) => [X(i), Y(v)]), dn = t.cone.p5.map((v, i) => [X(i), Y(v)]).reverse();
    band = `<path d="M${up.map(p => p.join(' ')).join('L')}L${dn.map(p => p.join(' ')).join('L')}Z" fill="#000" opacity=".07"/>
      <path d="${line(t.cone.p50)}" stroke="#999" stroke-dasharray="4 4" fill="none"/>`; }
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;margin-top:12px">${band}
    <line x1="${P}" x2="${W - P}" y1="${Y(0)}" y2="${Y(0)}" stroke="#ccc"/><path d="${line(t.live.map(x => x[1]))}" stroke="#000" stroke-width="2" fill="none"/>
    <text x="${P}" y="14" font-size="11" fill="#6e6e6e">live (black) against the backtest's 5th-95th percentile cone (grey)</text></svg>`;
}
function drawCb(r){
  $('kprev').disabled = false; $('kmsg').textContent = '';
  if (r.error) { $('kout').innerHTML = `<div class="msg err">${esc(r.error)}</div>`; return; }
  const tg = r.targets, t = r.track || {};
  const rows = r.tickets.map(k => `<tr><td>${k.shares > 0 ? 'BUY' : 'SELL'}</td><td class="mono">${esc(k.symbol)}</td><td class="mono">${Math.abs(k.shares)}</td>
    <td class="mono">$${k.price.toFixed(2)}</td><td class="mono">$${Math.abs(k.value).toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
    <td class="mono">${pct(k.target_weight)}</td><td class="mono">${k.from_shares} → ${k.to_shares}</td></tr>`).join('');
  const sent = r.sent ? `<div class="msg ok">Sent: ${esc(JSON.stringify(r.sent)).slice(0, 400)}</div>` : '';
  $('kout').innerHTML = `<h2 class="sec">Orders for the next open</h2>
    <p class="sub">${esc(tg.strategy)} · targets from the ${esc(tg.asof)} close · equity $${r.equity.toLocaleString()} · ${esc(r.broker)}</p>
    ${r.tickets.length ? `<table class="tbl"><tr><th>Side</th><th>Symbol</th><th>Shares</th><th>Price</th><th>Value</th><th>Target</th><th>Position</th></tr>${rows}</table>`
      : '<div class="msg ok">Already on target; nothing to trade.</div>'}${sent}
    <h2 class="sec">Live versus backtest</h2><p class="sub">${esc(t.message || '')}
    ${t.fill_drift_bps != null ? ` Fill drift versus the official open: ${t.fill_drift_bps.toFixed(1)} bp over ${t.fills_checked} fills.` : ''}</p>${cone(t)}`;
}
})();
</script>"""
