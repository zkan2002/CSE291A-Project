"""Command line.

    agentserve run CONFIG --out RUN_DIR   run one experiment into a new run directory
        [--unloaded-cache DIR]            reuse unloaded completion times measured before
    agentserve analyze RUN_DIR            print one run's metrics
    agentserve report PATH... --out DIR   every table and plot, from run directories
                                          (or directories holding them), across seeds
    agentserve sweep SWEEP --out DIR      run every sweep run of a sweep grid, one run directory
        [--dry-run]                       each; rerun to resume (complete sweep runs are skipped)
    analyze and report take [--window-start S] [--window-end S]: measure only the
    agent tasks arriving in [S, S) (default: each run's recorded measured window).
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
    sweep = commands.add_parser("sweep", help="run (or resume) a sweep over a grid of configurations")
    sweep.add_argument("sweep", type=Path, help="sweep config (TOML)")
    sweep.add_argument("--out", type=Path, required=True, help="sweep directory (created or resumed)")
    sweep.add_argument("--dry-run", action="store_true",
                       help="list the sweep runs and whether each is complete, then exit")
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
    elif args.command == "sweep":
        from agentserve.sweep import run_status, load_sweep, run_sweep
        sweep_cfg = load_sweep(args.sweep)
        if args.dry_run:
            sweep_runs = run_status(sweep_cfg, args.out)
            for sweep_run, done in sweep_runs:
                print(f"{'complete' if done else 'pending '}  {sweep_run.name}")
            print(f"{sum(d for _, d in sweep_runs)} of {len(sweep_runs)} sweep runs complete")
            return 0
        result = run_sweep(sweep_cfg, args.out)
        print(f"{len(result.complete)} sweep runs run, {len(result.skipped)} already complete, "
              f"{len(result.failed)} failed (see {args.out / 'sweep.jsonl'})")
        return 1 if result.failed else 0
    return 0


def _window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-start", type=float, default=None, metavar="S",
                        help="measure agent tasks arriving at or after S seconds (default 0 "
                             "if --window-end is given)")
    parser.add_argument("--window-end", type=float, default=None, metavar="S",
                        help="... and before S seconds; rates are per minute over the window "
                             "(default: no bound, rates up to the last completion). Without "
                             "either flag, each run's recorded measured window is used")


def _window(args: argparse.Namespace) -> MetricWindow | None:
    if args.window_start is None and args.window_end is None:
        return None  # each run's recorded window
    return MetricWindow(start_s=args.window_start or 0.0, end_s=args.window_end)


if __name__ == "__main__":
    raise SystemExit(main())
