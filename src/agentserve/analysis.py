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
- Goodput over fitting classes, peak memory, cached tokens, policy overhead
  (from ``decisions.jsonl``) and the per-task-class breakdown: see the
  "Analysis" section of docs/run-directory.md, which defines every metric.
Failed tasks count in ``agent_tasks`` and every denominator, but never as
completions; an agent task without a task SLO is never within it.

``aggregate`` groups runs into configurations (config equal apart from the seed)
and estimates each metric across seeds: mean with a t-based 95% CI.
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
    classes: dict[str, Metrics]
    """The same task-level metrics for each task class, over the same window."""

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

    fits = fittable_classes(config)
    m: Metrics = {}
    m.update(_task_metrics(tasks, slos, span_s))
    m.update(_fittable_goodput(tasks, fits, slos, span_s))
    m.update(_queueing(requests))
    m.update(_failures(tasks))
    m.update(_memory([s for s in _read_jsonl(run_dir / SAMPLES_FILE) if window.contains(s["t_s"])]))
    m.update(_cached_tokens(requests))
    m.update(_policy_overhead(
        [d for d in _read_jsonl(run_dir / DECISIONS_FILE) if window.contains(d["t_s"])]))

    classes: dict[str, Metrics] = {}
    for name in sorted({t["task_class"] for t in tasks}):
        of_class = [t for t in tasks if t["task_class"] == name]
        ids = {t["task_id"] for t in of_class}
        c = _task_metrics(of_class, slos, span_s)
        c.update(_queueing([r for r in requests if r["task_id"] in ids]))
        c.update(_failures(of_class))
        c["fits_budget"] = math.nan if fits is None else int(name in fits)
        classes[name] = c
    return RunAnalysis(run_dir, config, m, classes)


@dataclass(frozen=True)
class Estimate:
    """Mean across seeds with the half-width of its two-sided 95% confidence interval
    (Student's t, ``n - 1`` degrees of freedom). ``nan`` values are left out of ``n``;
    ``ci95`` is ``nan`` with fewer than two values."""

    mean: float
    ci95: float
    n: int

    @classmethod
    def of(cls, values: list[float]) -> Estimate:
        xs = [float(v) for v in values if not math.isnan(v)]
        n = len(xs)
        if n == 0:
            return cls(math.nan, math.nan, 0)
        mean = sum(xs) / n
        if n == 1:
            return cls(mean, math.nan, 1)
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
        return cls(mean, t_975(n - 1) * sd / math.sqrt(n), n)


# t(0.975, df) for df = 1..30
_T_975 = (12.706205, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912, 2.364624, 2.306004,
          2.262157, 2.228139, 2.200985, 2.178813, 2.160369, 2.144787, 2.131450, 2.119905,
          2.109816, 2.100922, 2.093024, 2.085963, 2.079614, 2.073873, 2.068658, 2.063899,
          2.059539, 2.055529, 2.051831, 2.048407, 2.045230, 2.042272)


def t_975(df: int) -> float:
    """The 97.5th percentile of Student's t with ``df`` degrees of freedom."""
    if df <= len(_T_975):
        return _T_975[df - 1]
    z = 1.959964  # Cornish-Fisher expansion; error < 1e-5 beyond df = 30
    return (z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


@dataclass(frozen=True)
class Configuration:
    """The runs whose configs are equal apart from the seed."""

    config: dict
    """The shared config, without ``run.seed``."""
    params: dict
    """The swept parameters, for tables and plots (see ``config_params``)."""
    runs: list[RunAnalysis]
    seeds: list[int]
    metrics: dict[str, Estimate]
    classes: dict[str, dict[str, Estimate]]


def config_params(config: dict) -> dict:
    """The parameters a sweep varies: admission policy, N, memory budget, arrival
    rate and cache policy (``resident``, ``offload``, ``off``; ``None`` when the
    config doesn't set one)."""
    runtime, server = config.get("runtime", {}), config.get("server", {})
    cache = server if server else runtime
    if cache.get("cache_prompt") is False:
        cache_policy = "off"
    elif "cache_idle_slots" in cache:
        cache_policy = "offload" if cache["cache_idle_slots"] else "resident"
    elif runtime.get("kind") == "simulated":  # the simulated runtime's default
        cache_policy = "resident"
    else:
        cache_policy = None
    return {
        "policy": config.get("policy", {}).get("name"),
        "concurrency": config.get("policy", {}).get("concurrency"),
        "memory_budget_tokens": runtime.get("memory_budget_tokens") or server.get("memory_budget"),
        "arrival_rate_per_s": config.get("workload", {}).get("arrival_rate_per_s"),
        "cache_policy": cache_policy,
    }


def aggregate(runs: list[RunAnalysis]) -> list[Configuration]:
    """Group runs into configurations and estimate every metric across seeds."""
    groups: dict[str, list[RunAnalysis]] = {}
    for run in runs:
        groups.setdefault(_config_key(run.config), []).append(run)
    configs = []
    for key, members in groups.items():
        members.sort(key=lambda r: (_seed(r.config) is None, _seed(r.config) or 0))
        shared = json.loads(key)
        class_names = sorted({c for r in members for c in r.classes})
        configs.append(Configuration(
            config=shared, params=config_params(shared), runs=members,
            seeds=[_seed(r.config) for r in members],
            metrics=_estimates([r.metrics for r in members]),
            classes={c: _estimates([r.classes[c] for r in members if c in r.classes])
                     for c in class_names}))
    configs.sort(key=lambda c: (_sort_key(c.params), _config_key(c.config)))
    return configs


def _estimates(rows: list[Metrics]) -> dict[str, Estimate]:
    names = list(dict.fromkeys(name for row in rows for name in row))
    # a failure cause absent from a run's line-up did not occur in it
    return {name: Estimate.of([row.get(name, 0.0 if name.startswith("failure_rate.") else math.nan)
                               for row in rows]) for name in names}


def _seed(config: dict) -> int | None:
    return config.get("run", {}).get("seed")


def _config_key(config: dict) -> str:
    shared = json.loads(json.dumps(config))
    shared.get("run", {}).pop("seed", None)
    return json.dumps(shared, sort_keys=True)


def _sort_key(params: dict) -> tuple:
    order = ("policy", "cache_policy", "memory_budget_tokens", "arrival_rate_per_s", "concurrency")
    return tuple((params[k] is None, params[k] if params[k] is not None else 0) for k in order)


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
    # a model request never admitted (its agent task hit its hard timeout) queued until then
    waits = sorted((r["completed_s"] if r["admitted_s"] is None else r["admitted_s"])
                   - r["arrived_s"] for r in requests)
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
