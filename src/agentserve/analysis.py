"""Analysis: reads a run directory and reports its metrics.

- Throughput: completed (non-failed) agent tasks per minute, over the run from
  t=0 to the last agent task's completion.
- Task completion latency: completion minus arrival, over completed agent tasks,
  with linearly interpolated percentiles.
- Goodput: agent tasks completed (not failed) within their task SLO
  (``slo_s`` in ``unloaded.jsonl``), per minute over the same window as
  throughput. ``nan`` when the run has no ``unloaded.jsonl``.
- SLO attainment: the fraction of all agent tasks that count towards goodput.
Failed tasks count in ``agent_tasks`` and every denominator, but never as
completions; an agent task without a task SLO is never within it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentserve.recorder import TASKS_FILE, UNLOADED_FILE


@dataclass(frozen=True)
class RunSummary:
    agent_tasks: int
    failed_tasks: int
    throughput_tasks_per_min: float
    goodput_tasks_per_min: float
    slo_attainment: float
    completion_latency_p50_s: float
    completion_latency_p95_s: float

    def format(self) -> str:
        lines = []
        for name, value in vars(self).items():
            lines.append(f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}")
        return "\n".join(lines)


def summarize(run_dir: Path) -> RunSummary:
    tasks = _read_jsonl(run_dir / TASKS_FILE)
    completed = [t for t in tasks if t["failure"] is None]
    slos = _task_slos(run_dir)
    good = None if slos is None else [
        t for t in completed
        if slos.get(t["task_id"]) is not None
        and t["completed_s"] - t["arrived_s"] <= slos[t["task_id"]]]
    run_end_s = max((t["completed_s"] for t in tasks), default=0.0)
    latencies = sorted(t["completed_s"] - t["arrived_s"] for t in completed)
    return RunSummary(
        agent_tasks=len(tasks),
        failed_tasks=len(tasks) - len(completed),
        throughput_tasks_per_min=_per_min(len(completed), run_end_s),
        goodput_tasks_per_min=float("nan") if good is None else _per_min(len(good), run_end_s),
        slo_attainment=(float("nan") if good is None or not tasks
                        else len(good) / len(tasks)),
        completion_latency_p50_s=percentile(latencies, 0.50),
        completion_latency_p95_s=percentile(latencies, 0.95),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _task_slos(run_dir: Path) -> dict[int, float | None] | None:
    path = run_dir / UNLOADED_FILE
    if not path.exists():
        return None
    return {u["task_id"]: u["slo_s"] for u in _read_jsonl(path)}


def _per_min(count: int, run_end_s: float) -> float:
    return 60.0 * count / run_end_s if run_end_s else 0.0


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
