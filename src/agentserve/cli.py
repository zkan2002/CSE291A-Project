"""Command line.

    agentserve run CONFIG --out RUN_DIR   run one experiment into a new run directory
        [--unloaded-cache DIR]            reuse unloaded completion times measured before
    agentserve analyze RUN_DIR            print one run's metrics
    agentserve report PATH... --out DIR   every table and plot, from run directories
                                          (or directories holding them), across seeds
    analyze and report take [--window-start S] [--window-end S]: measure only the
    agent tasks arriving in [S, S) (default: the whole run).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentserve.analysis import MetricWindow, summarize
from agentserve.experiment import load_config, run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentserve")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one experiment from a config file")
    run.add_argument("config", type=Path)
    run.add_argument("--out", type=Path, required=True, help="run directory to create")
    run.add_argument("--unloaded-cache", type=Path, default=None, metavar="DIR",
                     help="cache unloaded completion times here, per (workload, runtime config)")
    analyze = commands.add_parser("analyze", help="print metrics for a run directory")
    analyze.add_argument("run_dir", type=Path)
    _window_args(analyze)
    report = commands.add_parser(
        "report", help="write every table and plot for a set of run directories")
    report.add_argument("paths", type=Path, nargs="+",
                        help="run directories, or directories holding them at any depth")
    report.add_argument("--out", type=Path, required=True, help="report directory (overwritten)")
    _window_args(report)
    args = parser.parse_args(argv)

    if args.command == "run":
        run_experiment(load_config(args.config), args.out,
                       unloaded_cache_dir=args.unloaded_cache)
        print(f"wrote {args.out}")
    elif args.command == "analyze":
        print(summarize(args.run_dir, _window(args)).format())
    elif args.command == "report":
        from agentserve.report import write_report  # matplotlib is only needed here
        configs = write_report(args.paths, args.out, _window(args))
        print(f"wrote {args.out} ({len(configs)} configurations, "
              f"{sum(len(c.runs) for c in configs)} runs)")
    return 0


def _window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-start", type=float, default=0.0, metavar="S",
                        help="measure agent tasks arriving at or after S seconds (default 0)")
    parser.add_argument("--window-end", type=float, default=None, metavar="S",
                        help="... and before S seconds; rates are per minute over the window "
                             "(default: no bound, rates up to the last completion)")


def _window(args: argparse.Namespace) -> MetricWindow:
    return MetricWindow(start_s=args.window_start, end_s=args.window_end)


if __name__ == "__main__":
    raise SystemExit(main())
