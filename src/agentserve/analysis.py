"""Analysis: reads a run directory and reports its metrics.

- Throughput: completed (non-failed) agent tasks per minute, over the run from
  t=0 to the last agent task's completion.
- Task completion latency: completion minus arrival, over completed agent tasks,
  with linearly interpolated percentiles.
Failed tasks count in ``agent_tasks`` but never as completions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentserve.recorder import TASKS_FILE


@dataclass(frozen=True)
class RunSummary:
    agent_tasks: int
    failed_tasks: int
    throughput_tasks_per_min: float
    completion_latency_p50_s: float
    completion_latency_p95_s: float

    def format(self) -> str:
        lines = []
        for name, value in vars(self).items():
            lines.append(f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}")
        return "\n".join(lines)


def summarize(run_dir: Path) -> RunSummary:
    tasks = [json.loads(line) for line in (run_dir / TASKS_FILE).read_text().splitlines() if line]
    completed = [t for t in tasks if t["failure"] is None]
    run_end_s = max((t["completed_s"] for t in tasks), default=0.0)
    latencies = sorted(t["completed_s"] - t["arrived_s"] for t in completed)
    return RunSummary(
        agent_tasks=len(tasks),
        failed_tasks=len(tasks) - len(completed),
        throughput_tasks_per_min=60.0 * len(completed) / run_end_s if run_end_s else 0.0,
        completion_latency_p50_s=percentile(latencies, 0.50),
        completion_latency_p95_s=percentile(latencies, 0.95),
    )


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
