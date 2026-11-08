"""Export screenshots of every dashboard tab with a headless browser (Edge or Chrome).

Writes outputs/figures/dashboard_<tab>.png. Uses only the browser's --headless --screenshot
switches (no extra Python packages). Tabs are opened through the URL hash (#customers ...).
"""
import shutil
import subprocess
import sys
from pathlib import Path

from common import FIGURES_DIR, ROOT

HTML = ROOT / "outputs" / "dashboard" / "retail_dashboard.html"
TABS = {"overview": 1900, "customers": 1900, "products": 1900, "recommender": 1500, "pipeline": 1500}
CANDIDATES = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              shutil.which("chromium") or "", shutil.which("google-chrome") or ""]


def main():
    browser = next((c for c in CANDIDATES if c and Path(c).exists()), None)
    if not browser:
        sys.exit("No Edge/Chrome found; open the HTML and take screenshots manually.")
    for tab, height in TABS.items():
        out = FIGURES_DIR / f"dashboard_{tab}.png"
        url = HTML.as_uri() + "#" + tab
        cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
               f"--window-size=1400,{height}", "--virtual-time-budget=8000",
               f"--screenshot={out}", url]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        ok = out.exists() and out.stat().st_size > 10_000
        print(f"{tab:12s} rc={r.returncode} ok={ok} -> {out.name}")
        if not ok:
            print(r.stderr[-800:])


if __name__ == "__main__":
    main()
