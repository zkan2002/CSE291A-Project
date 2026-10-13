"""Command line.

    agentserve run CONFIG --out RUN_DIR   run one experiment into a new run directory
    agentserve analyze RUN_DIR            print its throughput and completion latency
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentserve.analysis import summarize
from agentserve.experiment import load_config, run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentserve")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one experiment from a config file")
    run.add_argument("config", type=Path)
    run.add_argument("--out", type=Path, required=True, help="run directory to create")
    analyze = commands.add_parser("analyze", help="print metrics for a run directory")
    analyze.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "run":
        run_experiment(load_config(args.config), args.out)
        print(f"wrote {args.out}")
    elif args.command == "analyze":
        print(summarize(args.run_dir).format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
