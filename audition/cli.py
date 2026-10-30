"""audition run strategy.py [--regions us,uk,eu,jp,crypto] [--capital 1e6] [--quick] [--out report.html]"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "cache" / "mpl"))


def main(argv=None):
    from .regions import REGIONS

    p = argparse.ArgumentParser(prog="audition", description="Audit a trading strategy file.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="audit a strategy file")
    r.add_argument("strategy")
    r.add_argument("--regions", default="us,uk,eu,jp",
                   help=f"comma list from {','.join(REGIONS)} (first is home unless UNIVERSE says otherwise)")
    r.add_argument("--capital", type=float, default=1_000_000.0)
    r.add_argument("--start", default=None, help="override the data start date (YYYY-MM-DD)")
    r.add_argument("--quick", action="store_true", help="fewer placebo and sensitivity runs")
    r.add_argument("--out", default=None, help="report path (default reports/<strategy>.html)")
    r.add_argument("--rehearsal", action="store_true", help="stop before the vault's locked years (opens a vault the first time)")
    f = sub.add_parser("final", help="take the final audition: run once on the vault's locked years and seal it")
    f.add_argument("strategy")
    f.add_argument("--regions", default="us,uk,eu,jp")
    f.add_argument("--capital", type=float, default=1_000_000.0)
    f.add_argument("--quick", action="store_true")
    f.add_argument("--out", default=None)
    f.add_argument("--retake", action="store_true", help="allow a second final (recorded as a retake)")
    cb = sub.add_parser("callback", help="paper stage: today's orders for a strategy, optionally sent to your paper broker")
    cb.add_argument("strategy")
    cb.add_argument("--broker", default="manual",
                    help="manual, webhook, or a broker id: alpaca, ibkr, tradier, saxo, trading212 (schwab, questrade read-only)")
    cb.add_argument("--equity", type=float, default=100_000.0, help="manual mode only")
    cb.add_argument("--send", action="store_true", help="send the orders (market-on-open)")
    cl = sub.add_parser("claims", help="check posted trade claims against real prices (file or stdin)")
    cl.add_argument("file", nargs="?")
    tr = sub.add_parser("translate", help="turn a Pine script / other code / plain English into a strategy file")
    tr.add_argument("file")
    tr.add_argument("--out", default=None)
    sub.add_parser("library", help="rebuild reports/index.html from saved audits")
    sv = sub.add_parser("serve", help="open the library with strategy import at http://127.0.0.1:8767")
    sv.add_argument("--port", type=int, default=8767)
    a = p.parse_args(argv)
    if a.cmd == "serve":
        from .server import serve
        serve(a.port)
        return
    if a.cmd == "callback":
        from .callback import run as cb_run
        r = cb_run(a.strategy, a.broker, send=a.send, equity=a.equity)
        print(f"{r['targets']['strategy']}  targets as of {r['targets']['asof']}  equity ${r['equity']:,.0f}")
        for t in r["tickets"]:
            print(f"  {'BUY ' if t['shares'] > 0 else 'SELL'} {abs(t['shares']):>6} {t['symbol']:<8} ~${abs(t['value']):,.0f}")
        if r["sent"] is not None:
            print(f"sent      {r['sent']}")
        print(r["track"].get("message", ""))
        return
    if a.cmd == "claims":
        from .claims import check, parse
        text = Path(a.file).read_text() if a.file else sys.stdin.read()
        claims, bad = parse(text)
        r = check(claims)
        for c in r["claims"]:
            print(f"{c['verdict']:<13} {c['raw'][:70]}" + (f"\n              {'; '.join(c['problems'])}" if c["problems"] else ""))
        for b in bad:
            print(f"unreadable    {b[:70]}")
        print("\n" + r["verdict"])
        return
    if a.cmd == "translate":
        from .translate import translate
        r = translate(Path(a.file).read_text())
        if r.get("error"):
            p.error(r["error"])
        out = Path(a.out or Path(a.file).with_suffix(".audition.py"))
        out.write_text(r["code"])
        print(f"translated via {r['via']} -> {out}")
        for e in r["validation"]["errors"] + r["validation"]["warnings"]:
            print("  " + e)
        return
    if a.cmd == "library":
        from .library import build
        print(f"library   {build(Path('reports')).resolve()}")
        return

    from .audit import Options, grade, run_audit
    from .report import write_report

    regions = [x.strip() for x in a.regions.split(",") if x.strip()]
    bad = [x for x in regions if x not in REGIONS]
    if bad:
        p.error(f"unknown region(s) {bad}")
    mode = "final" if a.cmd == "final" else "rehearsal" if getattr(a, "rehearsal", False) else "open"
    if mode == "final":
        from .strategy import load as sload
        from .vault import load as vload
        v = vload(sload(a.strategy).name)
        if v and v["finals"] and not a.retake:
            f0 = v["finals"][-1]
            p.error(f"the final audition was already taken on {f0['time']} (Sharpe {f0.get('sharpe_locked', float('nan')):.2f} "
                    "on the locked years). Pass --retake to run it again; it will be recorded as a retake.")
    opts = Options(regions=regions, capital=a.capital, start=getattr(a, "start", None), quick=a.quick,
                   log=lambda m: print(m, file=sys.stderr), mode=mode)
    audit = run_audit(a.strategy, opts)
    out = Path(a.out) if a.out else Path("reports") / f"{Path(a.strategy).stem}{'_final' if mode == 'final' else ''}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(audit, out)
    from .library import build
    lib = build(out.parent)
    score, letter = grade(audit)
    counts = {}
    for c in audit.checks.values():
        counts[c.status] = counts.get(c.status, 0) + 1
    print(f"\n{audit.strategy.name}: grade {letter} ({score:.0%})  " +
          "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"report    {out.resolve()}")
    print(f"json      {out.with_suffix('.json').resolve()}")
    print(f"library   {lib.resolve()}")


if __name__ == "__main__":
    main()
