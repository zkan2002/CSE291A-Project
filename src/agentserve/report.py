"""Report: every table and plot of the results, regenerated from raw run directories.

    agentserve report RUN_DIR_OR_SWEEP... --out REPORT_DIR

Each argument is a run directory or a directory holding run directories at any
depth. Runs whose configs differ only in the seed form one configuration; each
metric is reported as its mean across seeds with a 95% confidence interval
(``agentserve.analysis``). ``REPORT_DIR`` gets:

- ``runs.csv``: one row per run, every metric.
- ``configurations.csv``: one row per configuration, ``<metric>_mean``, ``<metric>_ci95``, ``<metric>_n``.
- ``classes.csv``: the same per configuration and task class.
- ``metrics.md``: the headline metrics, overall and per task class, as ``mean ± ci95``.
- ``goodput_vs_arrival_rate.png``, ``latency_vs_n.png``, ``memory_over_time.png``,
  ``failure_rate_vs_n.png``, ``cached_tokens_by_policy.png``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agentserve.analysis import (Configuration, Estimate, MetricWindow, RunAnalysis,  # noqa: E402
                                 aggregate, config_params, summarize)
from agentserve.recorder import SAMPLES_FILE, TASKS_FILE  # noqa: E402
from agentserve.sweep import PARTIAL_SUFFIX  # noqa: E402

PARAMS = ("policy", "concurrency", "memory_budget_tokens", "arrival_rate_per_s", "cache_policy")

HEADLINE = (
    ("goodput_tasks_per_min", "goodput /min"),
    ("goodput_fittable_tasks_per_min", "goodput (fitting classes) /min"),
    ("slo_attainment", "SLO attainment"),
    ("throughput_tasks_per_min", "throughput /min"),
    ("completion_latency_p50_s", "p50 latency s"),
    ("completion_latency_p95_s", "p95 latency s"),
    ("completion_latency_p99_s", "p99 latency s"),
    ("queueing_mean_s", "queueing s"),
    ("peak_vram_mib", "peak VRAM MiB"),
    ("peak_host_ram_mib", "peak host RAM MiB"),
    ("failure_rate", "failure rate"),
    ("cached_token_fraction", "cached fraction"),
    ("policy_overhead_mean_us", "policy overhead µs"),
)
CLASS_HEADLINE = (
    ("agent_tasks", "agent tasks"),
    ("goodput_tasks_per_min", "goodput /min"),
    ("slo_attainment", "SLO attainment"),
    ("completion_latency_p50_s", "p50 latency s"),
    ("completion_latency_p95_s", "p95 latency s"),
    ("queueing_mean_s", "queueing s"),
    ("failure_rate", "failure rate"),
    ("fits_budget", "fits budget"),
)

# Categorical slots in fixed order (dataviz reference palette, light mode).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
                 "#e34948")
LINESTYLES = ("-", "--", ":", "-.")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def find_run_dirs(paths: list[Path]) -> list[Path]:
    """Run directories given, or under the given directories at any depth. An
    unfinished sweep cell (a ``.partial`` directory, see ``agentserve.sweep``) is not one."""
    runs = set()
    for path in paths:
        if (path / TASKS_FILE).exists():
            runs.add(path)
        else:
            runs.update(p.parent for p in path.rglob(TASKS_FILE)
                        if not any(part.endswith(PARTIAL_SUFFIX)
                                   for part in p.parent.relative_to(path).parts))
    return sorted(runs)


def write_report(paths: list[Path], out_dir: Path,
                 window: MetricWindow | None = None) -> list[Configuration]:
    """``window``: measure every run over it (default: each run's recorded window)."""
    runs = [summarize(d, window) for d in find_run_dirs(paths)]
    if not runs:
        raise ValueError(f"no run directories under {', '.join(map(str, paths))}")
    configs = aggregate(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    _runs_csv(runs, out_dir / "runs.csv")
    _configurations_csv(configs, out_dir / "configurations.csv")
    _classes_csv(configs, out_dir / "classes.csv")
    (out_dir / "metrics.md").write_text(_markdown(configs, runs), encoding="utf-8")
    _plot_vs(configs, "arrival_rate_per_s", [("goodput_tasks_per_min", "goodput (tasks/min)")],
             "Goodput vs arrival rate", "arrival rate (agent tasks/s)",
             out_dir / "goodput_vs_arrival_rate.png")
    _plot_vs(configs, "concurrency",
             [("completion_latency_p50_s", "p50 task completion latency (s)"),
              ("completion_latency_p95_s", "p95 task completion latency (s)"),
              ("completion_latency_p99_s", "p99 task completion latency (s)")],
             "Task completion latency vs N", "N (concurrency)", out_dir / "latency_vs_n.png")
    _plot_vs(configs, "concurrency", [("failure_rate", "failed tasks / all agent tasks")],
             "Failure rate vs N", "N (concurrency)", out_dir / "failure_rate_vs_n.png")
    _plot_memory(configs, out_dir / "memory_over_time.png")
    _plot_cached(configs, out_dir / "cached_tokens_by_policy.png")
    return configs


# ---- tables

def _runs_csv(runs: list[RunAnalysis], path: Path) -> None:
    metric_names = _union(r.metrics for r in runs)
    rows = []
    for r in sorted(runs, key=lambda r: str(r.run_dir)):
        row = {"run_dir": str(r.run_dir), "seed": r.config.get("run", {}).get("seed"),
               **config_params(r.config)}
        row.update({name: r.metrics.get(name, "") for name in metric_names})
        rows.append(row)
    _write_csv(path, ["run_dir", "seed", *PARAMS, *metric_names], rows)


def _configurations_csv(configs: list[Configuration], path: Path) -> None:
    names = _union(c.metrics for c in configs)
    rows = [{**_param_cols(c), **_estimate_cols(c.metrics, names)} for c in configs]
    _write_csv(path, [*PARAMS, "seeds", *_estimate_headers(names)], rows)


def _classes_csv(configs: list[Configuration], path: Path) -> None:
    names = _union(m for c in configs for m in c.classes.values())
    rows = [{**_param_cols(c), "task_class": cls, **_estimate_cols(estimates, names)}
            for c in configs for cls, estimates in c.classes.items()]
    _write_csv(path, [*PARAMS, "seeds", "task_class", *_estimate_headers(names)], rows)


def _param_cols(c: Configuration) -> dict:
    return {**c.params, "seeds": " ".join(str(s) for s in c.seeds)}


def _estimate_headers(names: list[str]) -> list[str]:
    return [f"{n}_{part}" for n in names for part in ("mean", "ci95", "n")]


def _estimate_cols(estimates: dict[str, Estimate], names: list[str]) -> dict:
    cols = {}
    for n in names:
        e = estimates.get(n, Estimate(math.nan, math.nan, 0))
        cols.update({f"{n}_mean": e.mean, f"{n}_ci95": e.ci95, f"{n}_n": e.n})
    return cols


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _union(dicts) -> list[str]:
    return list(dict.fromkeys(name for d in dicts for name in d))


def _markdown(configs: list[Configuration], runs: list[RunAnalysis]) -> str:
    windows = sorted({r.window.describe() for r in runs})
    lines = ["# Results", "",
             f"Mean ± 95% CI half-width across seeds (Student's t). Agent tasks arriving in "
             f"{' or '.join(windows)} (the measured window{'s' if len(windows) > 1 else ''}, "
             f"run times in seconds) are measured.", "",
             "## Configurations", ""]
    for i, c in enumerate(configs, 1):
        lines.append(f"- **C{i}**: {_label(c.params)}; seeds {', '.join(map(str, c.seeds))}")
    lines += ["", "## Metrics", ""]
    lines += _md_table(["", *(h for _, h in HEADLINE)],
                       [[f"C{i}", *(_pm(c.metrics.get(n)) for n, _ in HEADLINE)]
                        for i, c in enumerate(configs, 1)])
    causes = _union(c.metrics for c in configs)
    causes = [n for n in causes if n.startswith("failure_rate.")]
    lines += ["", "## Failure rate by cause", ""]
    lines += _md_table(["", *(n.removeprefix("failure_rate.") for n in causes)],
                       [[f"C{i}", *(_pm(c.metrics.get(n)) for n in causes)]
                        for i, c in enumerate(configs, 1)])
    lines += ["", "## By task class", ""]
    lines += _md_table(["", "task class", *(h for _, h in CLASS_HEADLINE)],
                       [[f"C{i}", cls, *(_pm(est.get(n)) for n, _ in CLASS_HEADLINE)]
                        for i, c in enumerate(configs, 1) for cls, est in c.classes.items()])
    return "\n".join(lines) + "\n"


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(row) + " |" for row in rows])


def _pm(e: Estimate | None) -> str:
    if e is None or math.isnan(e.mean):
        return "–"
    return f"{e.mean:.3f}" if math.isnan(e.ci95) else f"{e.mean:.3f} ± {e.ci95:.3f}"


def _label(params: dict, keys=PARAMS) -> str:
    names = {"policy": "", "concurrency": "N=", "memory_budget_tokens": "budget=",
             "arrival_rate_per_s": "rate=", "cache_policy": "cache="}
    return " ".join(f"{names[k]}{params[k]}" for k in keys if params.get(k) is not None)


# ---- plots

def _varying(configs: list[Configuration]) -> list[str]:
    return [k for k in PARAMS if len({json.dumps(c.params[k]) for c in configs}) > 1]


def _style(i: int) -> dict:
    return {"color": SERIES_COLORS[i % len(SERIES_COLORS)],
            "linestyle": LINESTYLES[(i // len(SERIES_COLORS)) % len(LINESTYLES)]}


def _axes(n_panels: int, title: str):
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4), squeeze=False)
    fig.suptitle(title, color=INK)
    for ax in axes[0]:
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    return fig, axes[0]


def _empty(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, transform=ax.transAxes)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_vs(configs: list[Configuration], x: str, panels: list[tuple[str, str]], title: str,
             xlabel: str, path: Path) -> None:
    """One line per combination of the other swept parameters, with 95% CI error bars."""
    series_keys = [k for k in _varying(configs) if k != x]
    series: dict[str, list[Configuration]] = {}
    for c in configs:
        if c.params[x] is not None:
            series.setdefault(_label(c.params, series_keys) or "all runs", []).append(c)
    fig, axes = _axes(len(panels), title)
    for ax, (metric, ylabel) in zip(axes, panels):
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylabel(ylabel, color=INK)
        drawn = False
        for i, (label, members) in enumerate(series.items()):
            points = sorted((c.params[x], c.metrics.get(metric)) for c in members)
            points = [(px, e) for px, e in points if e is not None and not math.isnan(e.mean)]
            if not points:
                continue
            ax.errorbar([p for p, _ in points], [e.mean for _, e in points],
                        yerr=[0 if math.isnan(e.ci95) else e.ci95 for _, e in points],
                        marker="o", markersize=5, linewidth=2, capsize=3, label=label, **_style(i))
            drawn = True
        if not drawn:
            _empty(ax, "no data")
        elif len(series) > 1:
            ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def _plot_memory(configs: list[Configuration], path: Path) -> None:
    """VRAM and server host RAM over time: the lowest seed of each configuration."""
    keys = _varying(configs)
    fig, axes = _axes(2, "Memory over time (lowest seed of each configuration)")
    panels = (("vram_used_bytes", "VRAM in use (MiB)"), ("server_rss_bytes", "server host RAM (MiB)"))
    for ax, (field, ylabel) in zip(axes, panels):
        ax.set_xlabel("time since experiment start (s)", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        drawn = 0
        for i, c in enumerate(configs):
            samples = _read_jsonl(c.runs[0].run_dir / SAMPLES_FILE)
            points = [(s["t_s"], s[field] / 2 ** 20) for s in samples if s.get(field) is not None]
            if points:
                ax.plot([t for t, _ in points], [v for _, v in points], linewidth=2,
                        label=_label(c.params, keys) or "all runs", **_style(i))
                drawn += 1
        if not drawn:
            _empty(ax, "no system samples (simulated runs)")
        elif drawn > 1:
            ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def _plot_cached(configs: list[Configuration], path: Path) -> None:
    """Mean cached (retained-context) prompt tokens per model request, per configuration."""
    keys = ["policy", *(k for k in _varying(configs) if k != "policy")]
    fig, axes = _axes(1, "Cached prompt tokens by policy")
    fig.set_figwidth(max(5.5, 0.6 * len(configs) + 2))
    ax = axes[0]
    labels = [_label(c.params, keys) for c in configs]
    est = [c.metrics.get("cached_tokens_per_request") for c in configs]
    means = [0.0 if e is None or math.isnan(e.mean) else e.mean for e in est]
    errs = [0.0 if e is None or math.isnan(e.ci95) else e.ci95 for e in est]
    policies = list(dict.fromkeys(c.params["policy"] for c in configs))
    colors = [SERIES_COLORS[policies.index(c.params["policy"]) % len(SERIES_COLORS)]
              for c in configs]
    ax.bar(range(len(configs)), means, yerr=errs, color=colors, width=0.6, capsize=3,
           edgecolor="white", linewidth=2, error_kw={"ecolor": MUTED})
    ax.set_xticks(range(len(configs)), labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("cached prompt tokens per model request", color=INK)
    ax.set_ylim(bottom=0)
    _save(fig, path)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]
