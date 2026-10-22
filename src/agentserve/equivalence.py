"""Compare two machines on the same sweep runs: are they equivalent enough to pool results?

Two checks, both on runs with identical names (same configuration and seed, so the same
agent tasks):

1. Per-task speed: each agent task's unloaded completion time on machine B over machine A.
   The tasks are identical, so this is a paired comparison; the median ratio is the speed
   difference and the spread shows how uniform it is.
2. Configuration metrics: machine B's mean of each headline metric against machine A's 95%
   confidence interval across the same seeds.

    python -m agentserve.equivalence RUNS_A RUNS_B --config task_level_fifo-n4-b16384-r0.1-resident
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from agentserve.analysis import Estimate, read_jsonl, summarize

METRICS = ["goodput_tasks_per_min", "throughput_tasks_per_min", "slo_attainment", "failure_rate",
           "completion_latency_p50_s", "completion_latency_p95_s"]
SPEED_TOLERANCE = 0.05  # median per-task ratio within ±5%


def unloaded_times(run_dir: Path) -> dict[int, float]:
    return {r["task_id"]: r["unloaded_completion_s"] for r in read_jsonl(run_dir / "unloaded.jsonl")
            if r.get("unloaded_completion_s")}


def speed_ratios(runs_a: Path, runs_b: Path, names: list[str]) -> list[float]:
    """Unloaded time on B / on A, for every agent task timed successfully on both."""
    ratios = []
    for name in names:
        a, b = unloaded_times(runs_a / name), unloaded_times(runs_b / name)
        ratios += [b[t] / a[t] for t in sorted(a.keys() & b.keys())]
    return ratios


def compare(runs_a: Path, runs_b: Path, config: str, seeds: list[int]) -> tuple[list[str], bool]:
    names = [f"{config}-s{s}" for s in seeds]
    lines, ok = [], True

    ratios = speed_ratios(runs_a, runs_b, names)
    if ratios:
        q = statistics.quantiles(ratios, n=10)
        median = statistics.median(ratios)
        within = abs(median - 1) <= SPEED_TOLERANCE
        ok &= within
        lines.append(f"per-task unloaded time B/A: median {median:.3f} (p10 {q[0]:.3f}, p90 {q[-1]:.3f}), "
                     f"{len(ratios)} tasks -> {'OK' if within else 'DIFFERENT'} (tolerance ±{SPEED_TOLERANCE:.0%})")
    else:
        ok = False
        lines.append("per-task unloaded time: no agent task timed on both machines")

    for metric in METRICS:
        a = Estimate.of([summarize(runs_a / n).metrics[metric] for n in names])
        b = Estimate.of([summarize(runs_b / n).metrics[metric] for n in names])
        inside = abs(b.mean - a.mean) <= a.ci95
        ok &= inside
        lines.append(f"{metric}: A {a.mean:.3g} ± {a.ci95:.2g}, B {b.mean:.3g} ± {b.ci95:.2g} "
                     f"-> {'within' if inside else 'OUTSIDE'} A's 95% CI")
    lines.append("verdict: " + ("EQUIVALENT" if ok else "NOT EQUIVALENT"))
    return lines, ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_a", type=Path, help="machine A's runs/ directory")
    parser.add_argument("runs_b", type=Path, help="machine B's runs/ directory")
    parser.add_argument("--config", required=True, help="sweep-run name without the -sN seed suffix")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ns = parser.parse_args(argv)
    lines, ok = compare(ns.runs_a, ns.runs_b, ns.config, ns.seeds)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
