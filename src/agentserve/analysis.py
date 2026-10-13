"""Analysis: reads a run directory and reports its metrics.

Metrics are computed over a ``MetricWindow``: the agent tasks that *arrived*
inside it are measured (with their model requests), and rates are per minute
over the window's span. The default window starts at t=0 and ends at the last
agent task's completion, so every agent task is measured.

- Throughput: completed (non-failed) agent tasks per minute.
- Goodput: agent tasks completed (not failed) within their task SLO
  (``slo_s`` in ``unloaded.jsonl``), per minute. ``nan`` without ``unloaded.jsonl``.
- SLO attainment: the fraction of all agent tasks that count towards goodput.
- Task completion latency: completion minus arrival, over completed agent tasks,
  with linearly interpolated percentiles.
- Queueing time: admission minus proxy arrival, per model request (failed ones included).
- Failure rate: failed tasks over all agent tasks, overall and per cause.
Failed tasks count in ``agent_tasks`` and every denominator, but never as
completions; an agent task without a task SLO is never within it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from agentserve.recorder import (DECISIONS_FILE, REQUESTS_FILE, SAMPLES_FILE, TASKS_FILE,
                                 UNLOADED_FILE)
from agentserve.workload import TaskClass, WorkloadConfig

_MIB = 1024 * 1024

FAILURE_CAUSES = ("pool_exhausted", "exceeds_context", "server_error", "timeout")
"""Always reported (as 0 when absent), so runs line up; other causes are added when seen."""

Metrics = dict[str, float | int]


@dataclass(frozen=True)
class MetricWindow:
    """Agent tasks arriving in ``[start_s, end_s)`` are measured; ``end_s=None``
    means no upper bound, with rates taken up to the last completion."""

    start_s: float = 0.0
    end_s: float | None = None

    def contains(self, t: float) -> bool:
        return t >= self.start_s and (self.end_s is None or t < self.end_s)


@dataclass(frozen=True)
class RunAnalysis:
    run_dir: Path
    config: dict
    metrics: Metrics

    def format(self) -> str:
        return "\n".join(f"{name}: {_fmt(value)}" for name, value in self.metrics.items())


def summarize(run_dir: Path, window: MetricWindow = MetricWindow()) -> RunAnalysis:
    config = _read_json(run_dir / "config.json")
    all_tasks = _read_jsonl(run_dir / TASKS_FILE)
    tasks = [t for t in all_tasks if window.contains(t["arrived_s"])]
    measured = {t["task_id"] for t in tasks}
    requests = [r for r in _read_jsonl(run_dir / REQUESTS_FILE) if r["task_id"] in measured]
    end_s = window.end_s if window.end_s is not None else max(
        (t["completed_s"] for t in all_tasks), default=0.0)
    span_s = end_s - window.start_s
    slos = _task_slos(run_dir)

    m: Metrics = {}
    m.update(_task_metrics(tasks, slos, span_s))
    m.update(_fittable_goodput(tasks, fittable_classes(config), slos, span_s))
    m.update(_queueing(requests))
    m.update(_failures(tasks))
    m.update(_memory([s for s in _read_jsonl(run_dir / SAMPLES_FILE) if window.contains(s["t_s"])]))
    m.update(_cached_tokens(requests))
    m.update(_policy_overhead(
        [d for d in _read_jsonl(run_dir / DECISIONS_FILE) if window.contains(d["t_s"])]))
    return RunAnalysis(run_dir, config, m)


def fittable_classes(config: dict) -> set[str] | None:
    """Task classes whose longest agent task fits the memory budget on its own: its
    last model request (prompt plus ``n_predict``) needs at most the whole KV pool.
    ``None`` when the config does not give the classes and the budget."""
    budget = (config.get("runtime", {}).get("memory_budget_tokens")
              or config.get("server", {}).get("memory_budget"))
    if budget is None or "workload" not in config:
        return None
    classes = WorkloadConfig.from_config(config["workload"]).classes
    return {c.name for c in classes if _peak_tokens(c) <= budget}


def _peak_tokens(c: TaskClass) -> int:
    steps = c.steps[1]
    return c.initial_context_tokens + steps * c.output_tokens + (steps - 1) * c.tool_result_tokens


def _good(tasks: list[dict], slos: dict | None) -> list[dict] | None:
    """The agent tasks that completed (not failed) within their task SLO."""
    if slos is None:
        return None
    return [t for t in tasks if t["failure"] is None and slos.get(t["task_id"]) is not None
            and t["completed_s"] - t["arrived_s"] <= slos[t["task_id"]]]


def _fittable_goodput(tasks: list[dict], fits: set[str] | None, slos: dict | None,
                      span_s: float) -> Metrics:
    fittable = None if fits is None else [t for t in tasks if t["task_class"] in fits]
    good = None if fittable is None else _good(fittable, slos)
    return {
        "goodput_fittable_tasks_per_min": math.nan if good is None else _per_min(len(good), span_s),
        "slo_attainment_fittable": math.nan if good is None else _ratio(len(good), len(fittable)),
    }


def _task_metrics(tasks: list[dict], slos: dict | None, span_s: float) -> Metrics:
    completed = [t for t in tasks if t["failure"] is None]
    good = _good(tasks, slos)
    latencies = sorted(t["completed_s"] - t["arrived_s"] for t in completed)
    return {
        "agent_tasks": len(tasks),
        "failed_tasks": len(tasks) - len(completed),
        "throughput_tasks_per_min": _per_min(len(completed), span_s),
        "goodput_tasks_per_min": math.nan if good is None else _per_min(len(good), span_s),
        "slo_attainment": (math.nan if good is None or not tasks else len(good) / len(tasks)),
        "completion_latency_p50_s": percentile(latencies, 0.50),
        "completion_latency_p95_s": percentile(latencies, 0.95),
        "completion_latency_p99_s": percentile(latencies, 0.99),
    }


def _queueing(requests: list[dict]) -> Metrics:
    waits = sorted(r["admitted_s"] - r["arrived_s"] for r in requests)
    return {"queueing_mean_s": _mean(waits), "queueing_p95_s": percentile(waits, 0.95)}


def _failures(tasks: list[dict]) -> Metrics:
    causes = list(FAILURE_CAUSES)
    causes += sorted({t["failure"] for t in tasks if t["failure"]} - set(causes))
    n = len(tasks)
    m: Metrics = {"failure_rate": _ratio(sum(t["failure"] is not None for t in tasks), n)}
    for cause in causes:
        m[f"failure_rate.{cause}"] = _ratio(sum(t["failure"] == cause for t in tasks), n)
    return m


def _memory(samples: list[dict]) -> Metrics:
    def peak_mib(field: str) -> float:
        values = [s[field] for s in samples if s.get(field) is not None]
        return max(values) / _MIB if values else math.nan

    return {"peak_vram_mib": peak_mib("vram_used_bytes"),
            "peak_host_ram_mib": peak_mib("server_rss_bytes")}


def _cached_tokens(requests: list[dict]) -> Metrics:
    ok = [r for r in requests if r["error"] is None]
    cached = sum(r["n_cached_tokens"] or 0 for r in ok)
    return {"cached_tokens_per_request": _ratio(cached, len(ok)),
            "cached_token_fraction": _ratio(cached, sum(r["n_prompt_tokens"] for r in ok))}


def _policy_overhead(decisions: list[dict]) -> Metrics:
    times = sorted(d["decision_s"] for d in decisions)
    return {"policy_decisions": len(times),
            "policy_overhead_mean_us": _mean(times) * 1e6,
            "policy_overhead_p99_us": percentile(times, 0.99) * 1e6,
            "policy_overhead_total_ms": sum(times) * 1e3 if times else math.nan}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _task_slos(run_dir: Path) -> dict[int, float | None] | None:
    path = run_dir / UNLOADED_FILE
    if not path.exists():
        return None
    return {u["task_id"]: u["slo_s"] for u in _read_jsonl(path)}


def _per_min(count: int, span_s: float) -> float:
    return 60.0 * count / span_s if span_s > 0 else 0.0


def _ratio(num: float, den: float) -> float:
    return num / den if den else math.nan


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _fmt(value: float | int) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
