"""Analysis is tested at the log boundary: hand-written run directories with known answers."""

import json
import math
from pathlib import Path

import pytest

from agentserve.analysis import MetricWindow, summarize
from agentserve.cli import main


def write_run(run_dir: Path, tasks: list[dict], config: dict | None = None) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config or {}))
    (run_dir / "requests.jsonl").write_text("")
    (run_dir / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))


def task(task_id, arrived, completed, failure=None, task_class="short"):
    return {"task_id": task_id, "task_class": task_class, "arrived_s": arrived,
            "step_admitted_s": [arrived], "completed_s": completed, "failure": failure}


def test_analysis_prints_throughput_and_completion_latency_percentiles(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_run(run_dir, [
        task(0, arrived=0.0, completed=10.0),
        task(1, arrived=20.0, completed=40.0),
        task(2, arrived=50.0, completed=80.0),
        task(3, arrived=80.0, completed=120.0),
        # a failed task: not a completion, and its latency is not counted
        task(4, arrived=30.0, completed=60.0, failure="pool_exhausted"),
    ])

    assert main(["analyze", str(run_dir)]) == 0

    out = capsys.readouterr().out
    # 4 completed agent tasks over 120 s of run = 2.0 per minute
    assert "throughput_tasks_per_min: 2.000" in out
    # latencies 10, 20, 30, 40 s; linear-interpolated percentiles
    assert "completion_latency_p50_s: 25.000" in out
    assert "completion_latency_p95_s: 38.500" in out
    assert "agent_tasks: 5" in out
    assert "failed_tasks: 1" in out


def write_unloaded(run_dir: Path, slos: dict[int, float | None]) -> None:
    lines = []
    for task_id, slo in slos.items():
        unloaded = None if slo is None else slo / 3
        lines.append({"task_id": task_id, "task_class": "short", "unloaded_completion_s": unloaded,
                      "slo_s": slo, "hard_timeout_s": None if slo is None else unloaded * 10,
                      "failure": None if slo is not None else "exceeds_context"})
    (run_dir / "unloaded.jsonl").write_text("".join(json.dumps(u) + "\n" for u in lines))


def test_goodput_counts_only_agent_tasks_completed_within_their_task_slo(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_run(run_dir, [
        task(0, arrived=0.0, completed=10.0),     # latency 10 <= SLO 15: within SLO
        task(1, arrived=20.0, completed=40.0),    # latency 20 == SLO 20: within SLO
        task(2, arrived=50.0, completed=80.0),    # latency 30 > SLO 15: late
        task(3, arrived=80.0, completed=120.0),   # latency 40 <= SLO 60: within SLO
        # failed well within its SLO: still not good
        task(4, arrived=30.0, completed=31.0, failure="pool_exhausted"),
        # failed even when run alone: no task SLO at all
        task(5, arrived=60.0, completed=61.0, failure="exceeds_context"),
    ])
    write_unloaded(run_dir, {0: 15.0, 1: 20.0, 2: 15.0, 3: 60.0, 4: 30.0, 5: None})

    main(["analyze", str(run_dir)])

    out = capsys.readouterr().out
    # 3 agent tasks within SLO over 120 s = 1.5 per minute; throughput counts all 4 completed
    assert "goodput_tasks_per_min: 1.500" in out
    assert "throughput_tasks_per_min: 2.000" in out
    # failed tasks stay in the denominator: 3 of 6 agent tasks met their task SLO
    assert "slo_attainment: 0.500" in out
    assert "agent_tasks: 6" in out
    assert "failed_tasks: 2" in out


def analyze(run_dir: Path, capsys, window: MetricWindow = MetricWindow()) -> dict[str, float]:
    """The run's metrics at full precision (``agentserve analyze`` prints them rounded)."""
    capsys.readouterr()
    assert main(["analyze", str(run_dir)]) == 0
    printed = dict(line.split(": ") for line in capsys.readouterr().out.splitlines())
    metrics = summarize(run_dir, window).metrics
    assert sorted(printed) == sorted(metrics)
    return metrics


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def request(task_id, step, arrived, admitted, completed, *, n_prompt=100, n_cached=0, error=None):
    return {"task_id": task_id, "step": step, "arrived_s": arrived, "admitted_s": admitted,
            "first_token_s": None if error else completed - 0.5, "completed_s": completed,
            "n_prompt_tokens": n_prompt, "n_cached_tokens": None if error else n_cached,
            "n_predict": 10, "n_output_tokens": 0 if error else 10, "slot": 0, "error": error}


def test_tail_latency_queueing_time_and_failure_rate_by_cause(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_run(run_dir, [
        task(0, arrived=0.0, completed=10.0),
        task(1, arrived=20.0, completed=40.0),
        task(2, arrived=50.0, completed=80.0),
        task(3, arrived=80.0, completed=120.0),
        task(4, arrived=30.0, completed=60.0, failure="pool_exhausted"),
        task(5, arrived=31.0, completed=90.0, failure="timeout"),
        task(6, arrived=32.0, completed=95.0, failure="pool_exhausted"),
        task(7, arrived=33.0, completed=99.0),
    ])
    # queueing (admitted - arrived) per model request: 0, 1, 2, 3, failed ones included
    write_jsonl(run_dir / "requests.jsonl", [
        request(0, 0, 0.0, 0.0, 10.0),
        request(1, 0, 20.0, 21.0, 40.0),
        request(4, 0, 30.0, 32.0, 60.0, error="pool_exhausted"),
        request(2, 0, 50.0, 53.0, 80.0),
    ])

    m = analyze(run_dir, capsys)

    # completed latencies 10, 20, 30, 40, 66 s
    assert m["completion_latency_p99_s"] == pytest.approx(40 + 0.96 * 26)
    assert m["queueing_mean_s"] == pytest.approx(1.5)
    assert m["queueing_p95_s"] == pytest.approx(2.85)
    # 3 of 8 agent tasks failed: 2 pool_exhausted, 1 timeout
    assert m["failure_rate"] == pytest.approx(3 / 8)
    assert m["failure_rate.pool_exhausted"] == pytest.approx(2 / 8)
    assert m["failure_rate.timeout"] == pytest.approx(1 / 8)
    assert m["failure_rate.exceeds_context"] == 0.0
    assert m["failure_rate.server_error"] == 0.0


MIB = 1024 * 1024


def sample(t, vram_mib, rss_mib):
    return {"t_s": t, "vram_used_bytes": None if vram_mib is None else vram_mib * MIB,
            "server_rss_bytes": None if rss_mib is None else rss_mib * MIB, "slots": None}


def decision(t, decision_s, n_pending=1, n_admitted=1):
    return {"t_s": t, "decision_s": decision_s, "n_pending": n_pending, "n_admitted": n_admitted}


def test_peak_memory_cached_tokens_and_policy_overhead(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_run(run_dir, [task(0, 0.0, 10.0), task(1, 1.0, 12.0), task(2, 2.0, 14.0)])
    write_jsonl(run_dir / "requests.jsonl", [
        request(0, 0, 0.0, 0.0, 10.0, n_prompt=100, n_cached=0),
        request(1, 0, 1.0, 1.0, 12.0, n_prompt=200, n_cached=100),
        request(2, 0, 2.0, 2.0, 14.0, n_prompt=300, n_cached=200),
        # a failed model request reports no cached tokens and is left out of both
        request(2, 1, 2.0, 2.0, 14.0, n_prompt=900, error="pool_exhausted"),
    ])
    write_jsonl(run_dir / "samples.jsonl", [
        sample(0.0, 3000, 500), sample(0.1, 5000, None), sample(0.2, None, 700),
        sample(0.3, 4000, 600)])
    write_jsonl(run_dir / "decisions.jsonl", [
        decision(0.0, 10e-6), decision(1.0, 30e-6), decision(2.0, 20e-6, n_admitted=0),
        decision(3.0, 40e-6)])

    m = analyze(run_dir, capsys)

    assert m["peak_vram_mib"] == pytest.approx(5000)
    assert m["peak_host_ram_mib"] == pytest.approx(700)
    assert m["cached_tokens_per_request"] == pytest.approx(100)
    assert m["cached_token_fraction"] == pytest.approx(0.5)
    assert m["policy_decisions"] == 4
    assert m["policy_overhead_mean_us"] == pytest.approx(25)
    assert m["policy_overhead_p99_us"] == pytest.approx(39.7)
    assert m["policy_overhead_total_ms"] == pytest.approx(0.1)


def test_runs_without_system_samples_or_decision_times_report_nan(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_run(run_dir, [task(0, 0.0, 10.0)])

    m = analyze(run_dir, capsys)

    for name in ("peak_vram_mib", "peak_host_ram_mib", "policy_overhead_mean_us"):
        assert math.isnan(m[name]), name


# "small" peaks at 100 + 2 x 10 output + 1 x 50 tool result = 170 tokens;
# "huge" at 1000 + 3 x 100 + 2 x 200 = 1700 tokens (its longest agent task).
CLASSES = [
    {"name": "small", "weight": 1.0, "initial_context_tokens": 100, "output_tokens": 10,
     "steps": 2, "tool_result_tokens": 50},
    {"name": "huge", "weight": 1.0, "initial_context_tokens": 1000, "output_tokens": 100,
     "steps": [2, 3], "tool_result_tokens": 200},
]


@pytest.mark.parametrize("budget", [
    {"runtime": {"kind": "simulated", "memory_budget_tokens": 1000}},
    {"runtime": {"kind": "llama_server"}, "server": {"memory_budget": 1000}},
], ids=["simulated", "real"])
def test_goodput_restricted_to_task_classes_that_fit_the_memory_budget(tmp_path, capsys, budget):
    run_dir = tmp_path / "run"
    write_run(run_dir, [
        task(0, 0.0, 10.0, task_class="small"),
        task(1, 10.0, 20.0, task_class="small"),
        task(2, 20.0, 60.0, task_class="small"),  # late
        # an agent task of a class that does not fit, which still made it in time
        task(3, 30.0, 40.0, task_class="huge"),
        task(4, 40.0, 41.0, failure="exceeds_context", task_class="huge"),
    ], config={"workload": {"arrival_rate_per_s": 1.0, "classes": CLASSES}, **budget})
    write_unloaded(run_dir, {0: 15.0, 1: 15.0, 2: 15.0, 3: 15.0, 4: 15.0})

    m = analyze(run_dir, capsys)

    # over 60 s: 3 agent tasks within SLO in all, 2 of them in classes that fit
    assert m["goodput_tasks_per_min"] == pytest.approx(3.0)
    assert m["slo_attainment"] == pytest.approx(3 / 5)
    assert m["goodput_fittable_tasks_per_min"] == pytest.approx(2.0)
    assert m["slo_attainment_fittable"] == pytest.approx(2 / 3)


def test_restricted_goodput_is_nan_when_the_config_does_not_give_the_classes(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_run(run_dir, [task(0, 0.0, 10.0)])
    write_unloaded(run_dir, {0: 15.0})

    m = analyze(run_dir, capsys)

    assert math.isnan(m["goodput_fittable_tasks_per_min"])


def test_a_metric_window_measures_only_agent_tasks_arriving_inside_it(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_run(run_dir, [
        task(0, 0.0, 5.0),                               # warm-up: recorded, not measured
        task(1, 10.0, 20.0),
        task(2, 20.0, 50.0),                             # completes after the window: measured
        task(3, 30.0, 35.0, failure="pool_exhausted"),
        task(4, 40.0, 45.0),                             # arrives at the window's end: not measured
    ])
    write_jsonl(run_dir / "requests.jsonl", [
        request(0, 0, 0.0, 4.0, 5.0), request(1, 0, 10.0, 11.0, 20.0),
        request(2, 0, 20.0, 23.0, 50.0), request(4, 0, 40.0, 44.0, 45.0)])
    write_jsonl(run_dir / "samples.jsonl", [
        sample(5.0, 9000, 900), sample(15.0, 4000, 400), sample(39.0, 5000, 500),
        sample(40.0, 9000, 900)])
    write_jsonl(run_dir / "decisions.jsonl", [
        decision(0.0, 1e-3), decision(10.0, 10e-6), decision(20.0, 30e-6), decision(41.0, 1e-3)])

    m = analyze(run_dir, capsys, MetricWindow(start_s=10.0, end_s=40.0))

    assert m["agent_tasks"] == 3
    assert m["failed_tasks"] == 1
    # 2 completed agent tasks over the 30 s window
    assert m["throughput_tasks_per_min"] == pytest.approx(4.0)
    assert m["completion_latency_p50_s"] == pytest.approx(20.0)
    assert m["queueing_mean_s"] == pytest.approx(2.0)
    assert m["peak_vram_mib"] == pytest.approx(5000)
    assert m["peak_host_ram_mib"] == pytest.approx(500)
    assert m["policy_overhead_mean_us"] == pytest.approx(20)


def test_the_default_window_measures_the_whole_run_up_to_the_last_completion(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_run(run_dir, [task(0, 0.0, 5.0), task(1, 50.0, 60.0)])

    m = analyze(run_dir, capsys)

    assert m["agent_tasks"] == 2
    assert m["throughput_tasks_per_min"] == pytest.approx(2.0)


def test_metrics_are_broken_down_by_task_class(tmp_path):
    run_dir = tmp_path / "run"
    write_run(run_dir, [
        task(0, 0.0, 10.0, task_class="small"),
        task(1, 10.0, 30.0, task_class="small"),     # late
        task(2, 20.0, 50.0, task_class="huge"),
        task(3, 30.0, 31.0, failure="exceeds_context", task_class="huge"),
        task(4, 40.0, 60.0, task_class="huge"),
    ], config={"workload": {"arrival_rate_per_s": 1.0, "classes": CLASSES},
               "runtime": {"memory_budget_tokens": 1000}})
    write_unloaded(run_dir, {0: 15.0, 1: 15.0, 2: 60.0, 3: 60.0, 4: 60.0})
    write_jsonl(run_dir / "requests.jsonl", [
        request(0, 0, 0.0, 1.0, 10.0), request(1, 0, 10.0, 13.0, 30.0),
        request(2, 0, 20.0, 20.0, 50.0), request(4, 0, 40.0, 50.0, 60.0)])

    classes = summarize(run_dir).classes

    assert sorted(classes) == ["huge", "small"]
    small, huge = classes["small"], classes["huge"]
    # rates are over the whole run's 60 s window
    assert small["agent_tasks"] == 2
    assert small["throughput_tasks_per_min"] == pytest.approx(2.0)
    assert small["goodput_tasks_per_min"] == pytest.approx(1.0)
    assert small["slo_attainment"] == pytest.approx(0.5)
    assert small["completion_latency_p50_s"] == pytest.approx(15.0)
    assert small["queueing_mean_s"] == pytest.approx(2.0)
    assert small["failure_rate"] == 0.0
    assert small["fits_budget"] == 1
    assert huge["agent_tasks"] == 3
    assert huge["failed_tasks"] == 1
    assert huge["goodput_tasks_per_min"] == pytest.approx(2.0)
    assert huge["slo_attainment"] == pytest.approx(2 / 3)
    assert huge["completion_latency_p95_s"] == pytest.approx(29.5)
    assert huge["queueing_mean_s"] == pytest.approx(5.0)
    assert huge["failure_rate.exceeds_context"] == pytest.approx(1 / 3)
    assert huge["fits_budget"] == 0
