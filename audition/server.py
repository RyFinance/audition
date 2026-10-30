"""`audition serve`: the library and reports over HTTP on 127.0.0.1, plus an import API so a
strategy can be pasted or dropped into the page and audited without touching a terminal.

Imported files run with full Python permissions on this machine, exactly as if you ran
`audition run` on them, so the server only listens on localhost."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
IMPORTED = ROOT / "strategies" / "imported"
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
MAX_CODE = 400_000
MAX_BODY = 60_000_000  # uploads of BYO data


def validate(code: str) -> dict:
    """Static checks only: the code is parsed, never executed, until a run is started."""
    out = dict(ok=False, errors=[], warnings=[], name=None, params={}, universe=None, declare_keys=[])
    if not code.strip():
        out["errors"].append("The file is empty.")
        return out
    if len(code) > MAX_CODE:
        out["errors"].append("The file is over 400 KB.")
        return out
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        out["errors"].append(f"Syntax error on line {e.lineno}: {e.msg}")
        return out
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "weights"), None)
    if fn is None:
        out["errors"].append("No top-level weights(data, **params) function. Audition calls weights() "
                             "and fills its target weights at the next open.")
    else:
        args = [a.arg for a in fn.args.args]
        if not args or args[0] != "data":
            out["warnings"].append("weights() should take data as its first argument.")
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                val = None
            if key == "NAME" and isinstance(val, str):
                out["name"] = val
            elif key == "PARAMS":
                if isinstance(val, dict):
                    out["params"] = val
                else:
                    out["warnings"].append("PARAMS isn't a plain dict literal, so it can't be previewed.")
            elif key == "UNIVERSE":
                out["universe"] = (val if isinstance(val, str) else
                                   f"{len(val)} tickers" if isinstance(val, (list, tuple)) else
                                   f"{len(val)} markets" if isinstance(val, dict) else "computed")
            elif key == "DECLARE" and isinstance(val, dict):
                out["declare_keys"] = sorted(val)
    if fn is not None and out["params"]:
        args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        missing = [k for k in out["params"] if k not in args] if not fn.args.kwarg else []
        if missing:
            out["errors"].append(f"PARAMS has {', '.join(missing)} but weights() doesn't accept them.")
    if not out["declare_keys"]:
        out["warnings"].append("No DECLARE: multiple-testing and out-of-sample checks will be unverified.")
    out["ok"] = not out["errors"]
    return out


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return (s or "strategy")[:48]


def start_job(code: str, filename: str, regions: str, capital: float, quick: bool, rehearsal: bool = False) -> dict:
    v = validate(code)
    if not v["ok"]:
        return dict(error="; ".join(v["errors"]), validation=v)
    stem = slug(Path(filename).stem if filename else (v["name"] or "strategy"))
    IMPORTED.mkdir(parents=True, exist_ok=True)
    path = IMPORTED / f"{stem}.py"
    path.write_text(code)
    regions = ",".join(r for r in re.split(r"[,\s]+", regions or "") if r in {"us", "uk", "eu", "jp", "crypto"})
    out = REPORTS / f"{stem}.html"
    cmd = [sys.executable, "-W", "ignore", "-m", "audition", "run", str(path), "--regions", regions,
           "--capital", str(float(capital or 1e6)), "--out", str(out)]
    if quick:
        cmd.append("--quick")
    if rehearsal:
        cmd.append("--rehearsal")
    return _spawn(cmd, out, path)


def start_final(file: str) -> dict:
    path = Path(file).resolve()
    if not path.exists() or path.suffix != ".py" or ROOT not in path.parents:
        return dict(error="strategy file not found inside the Audition folder")
    out = REPORTS / f"{path.stem}_final.html"
    cmd = [sys.executable, "-W", "ignore", "-m", "audition", "final", str(path), "--out", str(out), "--retake"]
    return _spawn(cmd, out, path)


def _spawn(cmd, out, path) -> dict:
    jid = uuid.uuid4().hex[:10]
    job = dict(id=jid, status="running", log=[], report=None, file=str(path.relative_to(ROOT)),
               started=time.time(), cmd=" ".join(cmd[3:]))
    with LOCK:
        JOBS[jid] = job

    def run():
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            line = line.rstrip()
            if not line or "possibly delisted" in line or "Failed download" in line or line.startswith("["):
                continue
            with LOCK:
                job["log"].append(line)
        p.wait()
        with LOCK:
            job["status"] = "done" if p.returncode == 0 and out.exists() else "failed"
            job["report"] = out.name if job["status"] == "done" else None
            job["seconds"] = round(time.time() - job["started"])

    threading.Thread(target=run, daemon=True).start()
    return dict(job=jid, file=job["file"])


def strategy_files() -> list[dict]:
    """Every strategy file on the shelf, for the Callback picker."""
    out = []
    for f in sorted(REPORTS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        sf = d.get("manifest", {}).get("strategy_file")
        if sf and Path(sf).exists() and not any(o["file"] == sf for o in out):
            out.append(dict(name=d.get("strategy"), file=sf, grade=d.get("grade")))
    return out


def setup_checks() -> list[dict]:
    import importlib.util
    import os
    import shutil
    from . import sources
    out = []
    pk = [m for m in ("numpy", "pandas", "scipy", "matplotlib", "yfinance") if importlib.util.find_spec(m) is None]
    out.append(dict(name="Python and packages", ok=not pk, detail="all installed" if not pk else "missing " + ", ".join(pk)
                    + " (run ./setup.sh)"))
    out.append(dict(name="App server", ok=True, detail="running on 127.0.0.1 only"))
    src = sources.default_source()
    need = sources.PROVIDERS[src]["fields"]
    has = all(sources.creds(src).get(f) for f in need if f != "feed")
    out.append(dict(name="Price data", ok=has, go="data",
                    detail=f"{sources.PROVIDERS[src]['label']}" + ("" if has else " needs a key")))
    cli = shutil.which("claude") or os.path.exists(os.path.expanduser("~/.local/bin/claude"))
    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    out.append(dict(name="Translator", ok=bool(cli or key), optional=True,
                    detail="Claude Code found" if cli else "ANTHROPIC_API_KEY set" if key else
                    "optional: install Claude Code or set ANTHROPIC_API_KEY to translate Pine / English"))
    cfg = sources.load_config()
    linked = [sources.BROKERS[b]["label"] for b in sources.BROKERS if cfg.get(b)]
    out.append(dict(name="Paper broker", ok=bool(linked), optional=True, go="callback",
                    detail=", ".join(linked) if linked else "optional: link one to send paper orders"))
    discs = len([f for f in REPORTS.glob("*.json")])
    out.append(dict(name="Your first audit", ok=discs > 1, optional=True, go="new",
                    detail=f"{discs} disc(s) on the shelf" if discs > 1 else "add a strategy to burn your first disc"))
    return out


def _bg(fn, *args) -> dict:
    """Run a slow call (translation, claim check, callback) on a thread and poll it like an audit."""
    jid = uuid.uuid4().hex[:10]
    job = dict(id=jid, status="running", log=[], result=None, started=time.time())
    with LOCK:
        JOBS[jid] = job

    def go():
        try:
            res = fn(*args)
            st = "done"
        except Exception as e:
            res, st = dict(error=f"{type(e).__name__}: {e}"), "failed"
        with LOCK:
            job.update(status=st, result=res, seconds=round(time.time() - job["started"]))

    threading.Thread(target=go, daemon=True).start()
    return dict(job=jid)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(REPORTS), **kw)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/api/ping":
            return self._json(dict(ok=True, version="0.2"))
        if self.path == "/api/connections":
            from . import sources
            return self._json(dict(config=sources.masked(), providers=sources.PROVIDERS, brokers=sources.BROKERS,
                                   default=sources.default_source(), byo=sources.list_byo()))
        if self.path == "/api/setup":
            return self._json(setup_checks())
        if self.path == "/api/strategies":
            return self._json(strategy_files())
        if self.path.startswith("/api/job/"):
            with LOCK:
                job = JOBS.get(self.path.rsplit("/", 1)[-1])
                snap = json.loads(json.dumps(job, default=str)) if job else None
            return self._json(snap or dict(error="no such job"), 200 if snap else 404)
        if self.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        # same-origin only: the page is served from here
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and not origin.endswith("//" + host):
            return self._json(dict(error="cross-origin request refused"), 403)
        b = self._body()
        if self.path == "/api/validate":
            return self._json(validate(b.get("code", "")))
        if self.path == "/api/run":
            res = start_job(b.get("code", ""), b.get("filename", ""), b.get("regions", ""),
                            b.get("capital", 1e6), bool(b.get("quick")), bool(b.get("rehearsal")))
            return self._json(res, 400 if "error" in res else 200)
        if self.path == "/api/final":
            res = start_final(b.get("file", ""))
            return self._json(res, 400 if "error" in res else 200)
        if self.path == "/api/translate":
            from .translate import translate
            return self._json(_bg(translate, b.get("source", ""), b.get("kind", "auto"), b.get("hint", "")))
        if self.path == "/api/claims":
            from .claims import check, parse

            def run_claims(text):
                cl, bad = parse(text)
                if not cl:
                    return dict(error="No trades found. Use one per line, e.g. 'long NVDA 2025-03-04 @ 112.30 -> 2025-03-06 @ 120.10'.",
                                unreadable=bad)
                r = check(cl[:200])
                r["unreadable"] = bad
                return r
            return self._json(_bg(run_claims, b.get("text", "")))
        if self.path == "/api/connections":
            from . import sources
            try:
                return self._json(dict(config=sources.update_config(b.get("name", ""), b.get("values", {}))))
            except Exception as e:
                return self._json(dict(error=str(e)), 400)
        if self.path == "/api/connections/test":
            from . import sources
            return self._json(sources.test_connection(b.get("name", "")))
        if self.path == "/api/byo":
            import base64
            from . import sources
            if b.get("delete"):
                sources.delete_byo(b["delete"])
                return self._json(dict(ok=True, byo=sources.list_byo()))
            try:
                raw = base64.b64decode(b.get("content", ""))
                res = sources.save_byo(b.get("name") or Path(b.get("filename", "data")).stem, raw, b.get("filename", ""))
                return self._json(dict(res, byo=sources.list_byo()))
            except Exception as e:
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 400)
        if self.path == "/api/callback":
            from .callback import run as cb_run
            f = b.get("file", "")
            if not any(s["file"] == f for s in strategy_files()):
                return self._json(dict(error="pick a strategy from the shelf"), 400)
            pos = {k.upper(): float(v) for k, v in (b.get("positions") or {}).items()}
            return self._json(_bg(cb_run, f, b.get("broker", "manual"), bool(b.get("send")),
                                  float(b.get("equity") or 100_000), pos))
        return self._json(dict(error="not found"), 404)


def serve(port: int = 8767):
    from .library import build

    REPORTS.mkdir(exist_ok=True)
    build(REPORTS)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"audition  http://127.0.0.1:{port}", flush=True)
    httpd.serve_forever()
