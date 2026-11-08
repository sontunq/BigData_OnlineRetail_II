"""Run the pipeline end to end:  python run_pipeline.py  [--from 02] [--to 16]

Each step is a numbered script in src/. The run stops at the first failing step.
Step timings are written to outputs/pipeline_run_log.csv.
"""
import argparse
import csv
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
LOG = ROOT / "outputs" / "pipeline_run_log.csv"
STEPS = sorted(p.name for p in SRC.glob("[0-9][0-9]_*.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="00")
    ap.add_argument("--to", dest="end", default="99")
    a = ap.parse_args()
    rows, t_all = [], time.time()
    for step in STEPS:
        if not (a.start <= step[:2] <= a.end):
            continue
        print(f"\n===== {step} =====", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, step], cwd=SRC)
        secs = round(time.time() - t0, 1)
        print(f"===== {step} finished rc={rc} in {secs}s =====", flush=True)
        rows.append({"step": step, "rc": rc, "seconds": secs,
                     "finished_at": datetime.now().isoformat(timespec="seconds"),
                     "machine": platform.node(), "python": platform.python_version()})
        if rc != 0:
            break
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["step"])
        w.writeheader()
        w.writerows(rows)
    total = round(time.time() - t_all, 1)
    print(f"\nTotal {total}s; log written to {LOG}")
    if rows and rows[-1]["rc"] != 0:
        sys.exit(rows[-1]["rc"])


if __name__ == "__main__":
    main()
