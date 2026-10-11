"""Cross-platform pipeline orchestrator (a stand-in for ``make`` on Windows).

Usage:
    python run_all.py data features models evaluate report
    python run_all.py all
    python run_all.py all --fast

Each stage shells out to the corresponding ``src`` module CLI so the behaviour
is identical to the Makefile.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STAGES = ["data", "features", "models", "evaluate", "report"]

# stage -> list of module commands
STAGE_CMDS = {
    "data": [["-m", "src.data_download"]],
    "features": [["-m", "src.data_clean"], ["-m", "src.targets"],
                 ["-m", "src.features"]],
    "models": [["-m", "src.walkforward"]],
    "evaluate": [["-m", "src.evaluate"], ["-m", "src.explain"],
                 ["-m", "src.strategy"], ["-m", "src.leakage_demo"]],
    "report": [["-m", "src.report"]],
}


def run_stage(stage: str, config: str, extra: list[str]) -> None:
    for cmd in STAGE_CMDS[stage]:
        full = [sys.executable, *cmd, "--config", config, *extra]
        print(f"\n=== RUN: {' '.join(full)} ===", flush=True)
        t0 = time.time()
        result = subprocess.run(full)
        dt = time.time() - t0
        if result.returncode != 0:
            print(f"!!! stage {stage} failed ({' '.join(cmd)}) "
                  f"exit={result.returncode}", flush=True)
            sys.exit(result.returncode)
        print(f"--- done {stage}: {' '.join(cmd)} in {dt:.1f}s ---", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline stages")
    parser.add_argument("stages", nargs="+", choices=STAGES + ["all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    stages = STAGES if "all" in args.stages else args.stages
    extra = ([] if not args.force else ["--force"]) + \
            ([] if not args.fast else ["--fast"])

    t_total = time.time()
    for s in stages:
        run_stage(s, args.config, extra)
    print(f"\nALL STAGES COMPLETE in {(time.time()-t_total)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
