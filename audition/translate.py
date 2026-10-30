"""Translator: turn a TradingView Pine script, a QuantConnect/Backtrader algo, a tweet or a
plain-English rule into an Audition strategy file.

Claude does the translation. It runs through your own Claude Code login (`claude -p`) when
that's installed, otherwise through the Anthropic SDK with ANTHROPIC_API_KEY. The result is
only ever text: it's validated by the same AST check as an import and never executed here."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

MODEL = "claude-opus-5-5"

CONTRACT = '''You translate trading strategies into Audition strategy files. Reply with ONE fenced
```python block and nothing else.

The file contract:
- NAME = "..."                         short human name
- PARAMS = {...}                       every tunable number (lengths, thresholds, top_n); Audition
                                        sensitivity-tests these, so never hard-code them in the body
- UNIVERSE = [...]                     Yahoo tickers (e.g. "SPY", "BTC-USD", "ES=F", "7203.T"); or
                                        "sp500" for the point-in-time S&P 500 (then use data.eligible)
- DECLARE = {"variants_tried": N, ...} leave variants_tried as 1 with a comment asking the user to
                                        fill in the real count; add "costs" only if the source gives them
- def weights(data, **PARAMS) -> pd.DataFrame
    data.open/high/low/close/volume are DataFrames (dates x tickers), split/dividend adjusted.
    data.benchmark is a Series, data.eligible a bool DataFrame or None.
    Return target portfolio weights decided at each day's CLOSE (dates x tickers). Audition fills
    them at the NEXT open and holds open-to-open. 1.0 = 100% of equity long, -1.0 = 100% short.

Rules:
- Daily bars only. If the source is intraday, write the closest daily version and say so in the
  translation notes.
- No lookahead: only use data up to and including the current row (shift, rolling, ewm are fine;
  never shift(-k), never future-centred windows, never fit on the whole sample).
- Weights must be a pandas DataFrame with the same index as data.close; fill missing with 0.
- Use pandas and numpy only. Implement indicators (RSI, ATR, MACD, Bollinger, etc.) by hand.
- Stops/targets from the source: approximate on daily closes and note the approximation.
- Position sizing: if the source risks a fixed % per trade, convert to a weight; otherwise go
  all-in on one signal or equal-weight several.
- Start the file with a docstring: one line on what it does, then "Translation notes:" listing
  every assumption, approximation and anything the source left unclear.
'''


def _extract(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def _via_cli(prompt: str, timeout: int) -> str:
    exe = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    if not os.path.exists(exe):
        raise FileNotFoundError("claude CLI not found")
    p = subprocess.run([exe, "-p", "--model", MODEL, "--append-system-prompt", CONTRACT,
                        "--disallowedTools", "Bash,Edit,Write,Read,WebFetch,WebSearch,Glob,Grep,Agent"],
                       input=prompt, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError((p.stderr or p.stdout or "claude -p failed").strip()[-400:])
    return p.stdout


def _via_sdk(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=CONTRACT,
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined to translate this.")
    return "".join(b.text for b in resp.content if b.type == "text")


def translate(source: str, kind: str = "auto", hint: str = "", timeout: int = 240) -> dict:
    source = source.strip()
    if not source:
        return dict(error="Paste something to translate.")
    if len(source) > 60_000:
        return dict(error="That's over 60 KB; paste the strategy logic only.")
    prompt = (f"Source type: {kind}.\n" + (f"User notes: {hint}\n" if hint else "")
              + f"Translate this into an Audition strategy file:\n\n{source}")
    errors = []
    for name, fn in (("claude code", lambda: _via_cli(prompt, timeout)), ("anthropic api", lambda: _via_sdk(prompt))):
        if name == "anthropic api" and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            errors.append("anthropic api: no ANTHROPIC_API_KEY set")
            continue
        try:
            code = _extract(fn())
            from .server import validate
            return dict(code=code, via=name, validation=validate(code))
        except Exception as e:
            errors.append(f"{name}: {e}")
    return dict(error="Couldn't reach Claude. " + " | ".join(errors))
