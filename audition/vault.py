"""The holdout vault: rehearsals and the final audition.

The first time a strategy (by NAME) is rehearsed, Audition locks the most recent stretch of
history (default two years). Rehearsals only ever see data before the lock. The final
audition runs once on the locked period and is sealed with a timestamp and the file's hash.

Everything that weakens the lock is recorded, not prevented:
  - every distinct file version rehearsed counts as a variant tried (feeds check 4)
  - an ordinary full-history run after the vault exists counts as a peek
  - a second final audition (a retake) counts as another look at the locked period (check 6)
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAULT = ROOT / "vault"
LOCK_YEARS = 2


def key(name: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "strategy")[:60]


def _path(name: str) -> Path:
    return VAULT / f"{key(name)}.json"


def load(name: str) -> dict | None:
    p = _path(name)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _save(v: dict):
    VAULT.mkdir(exist_ok=True)
    _path(v["name"]).write_text(json.dumps(v, indent=2))


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def open_vault(name: str, years: float = LOCK_YEARS) -> dict:
    v = load(name)
    if v:
        return v
    sealed = (dt.date.today() - dt.timedelta(days=int(365.25 * years))).isoformat()
    v = dict(name=name, sealed_from=sealed, created=_now(), lock_years=years,
             rehearsals=[], versions=[], peeks=[], finals=[])
    _save(v)
    return v


def record_rehearsal(name: str, sha: str, sharpe: float | None):
    v = open_vault(name)
    v["rehearsals"].append(dict(time=_now(), sha=sha[:12], sharpe=sharpe))
    if sha[:12] not in v["versions"]:
        v["versions"].append(sha[:12])
    _save(v)
    return v


def record_peek(name: str, sha: str):
    v = load(name)
    if not v:
        return None
    v["peeks"].append(dict(time=_now(), sha=sha[:12]))
    _save(v)
    return v


def record_final(name: str, sha: str, result: dict):
    v = open_vault(name)
    v["finals"].append(dict(time=_now(), sha=sha[:12], **result))
    if sha[:12] not in v["versions"]:
        v["versions"].append(sha[:12])
    _save(v)
    return v


def injected_declare(v: dict, declare: dict, final: bool, sha: str = "") -> tuple[dict, list]:
    """The vault's record overrides self-reported numbers where it knows more."""
    d = dict(declare)
    notes = []
    n_ver = len(set(v["versions"]) | ({sha[:12]} if sha else set()))
    if n_ver > 1 and n_ver > int(d.get("variants_tried") or 0):
        notes.append(f"variants_tried raised from {d.get('variants_tried') or 'undeclared'} to {n_ver}: "
                     "the vault counted that many distinct versions")
        d["variants_tried"] = n_ver
    if final:
        looks = len(v["finals"]) + 1 + len(v["peeks"])
        d["oos_start"] = v["sealed_from"]
        d["oos_looks"] = looks
        notes.append(f"oos_start set to the vault seal {v['sealed_from']}, oos_looks {looks} "
                     f"({len(v['finals'])} earlier finals, {len(v['peeks'])} peeks)")
    return d, notes


def status(name: str) -> dict:
    v = load(name)
    if not v:
        return dict(state="none")
    state = "final" if v["finals"] else "rehearsing"
    return dict(state=state, **v)
