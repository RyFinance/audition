"""Re-burn the tutorial disc that ships with a clean checkout: audition/tutorial/tutorial.{html,json}.
Run from the repo root: .venv/bin/python tools/bake_tutorial.py"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "audition" / "tutorial"
OUT.mkdir(exist_ok=True)
tmp = ROOT / "cache" / "tutorial_bake"
tmp.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-W", "ignore", "-m", "audition", "run", "examples/tutorial.py",
                "--out", str(tmp / "tutorial.html")], cwd=ROOT, check=True)
for ext in ("html", "json"):
    s = (tmp / f"tutorial.{ext}").read_text()
    s = s.replace(str(ROOT) + "/", "")          # paths relative to the repo, nothing about this machine
    (OUT / f"tutorial.{ext}").write_text(s)
print("baked", OUT)
