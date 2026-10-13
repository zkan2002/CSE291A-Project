"""Unloaded completion time, task SLO and goodput.

Experiments run end-to-end through the one-command entry point on the simulated
runtime and virtual clock; only what lands in the run directory (and what
``agentserve analyze`` prints) is checked.
"""

import json
from pathlib import Path

import pytest

from agentserve.cli import main

CONFIG = """\
[run]
seed = {seed}
duration_s = {duration_s}

[workload]
arrival_rate_per_s = {arrival_rate}

[[workload.classes]]
name = "one-step"
weight = 1.0
initial_context_tokens = 500
output_tokens = 50
steps = 1

[[workload.classes]]
name = "three-step"
weight = 1.0
initial_context_tokens = 500
output_tokens = 50
steps = 3
tool_result_tokens = 200
tool_delay_median_s = 1.0

[policy]
name = "task_level_fifo"
concurrency = {concurrency}

[runtime]
kind = "simulated"
n_slots = {n_slots}
memory_budget_tokens = {memory_budget}
prompt_token_s = 0.0002
output_token_s = 0.01
"""


def write_config(tmp_path: Path, *, seed=1, duration_s=60, arrival_rate=0.5,
                 concurrency=4, n_slots=4, memory_budget=32768, name="config") -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(CONFIG.format(seed=seed, duration_s=duration_s, arrival_rate=arrival_rate,
                                  concurrency=concurrency, n_slots=n_slots,
                                  memory_budget=memory_budget))
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_every_agent_task_has_its_unloaded_completion_time_slo_and_timeout_recorded(tmp_path):
    run_dir = tmp_path / "run"

    main(["run", str(write_config(tmp_path)), "--out", str(run_dir)])

    tasks = read_jsonl(run_dir / "tasks.jsonl")
    unloaded = {u["task_id"]: u for u in read_jsonl(run_dir / "unloaded.jsonl")}
    assert sorted(unloaded) == sorted(t["task_id"] for t in tasks)
    for t in tasks:
        u = unloaded[t["task_id"]]
        assert u["task_class"] == t["task_class"]
        assert u["failure"] is None
        assert u["slo_s"] == pytest.approx(3 * u["unloaded_completion_s"])
        assert u["hard_timeout_s"] == pytest.approx(10 * u["unloaded_completion_s"])
        if t["task_class"] == "one-step":
            # alone: 500 prompt tokens x 0.2 ms + 50 output tokens x 10 ms
            assert u["unloaded_completion_s"] == pytest.approx(0.6)
        else:
            # three steps of compute (prompts 500, 750, 1000 tokens) plus two tool delays
            assert u["unloaded_completion_s"] > 0.45 + 1.5


def test_unloaded_completion_times_are_cached_per_workload_and_runtime_config(tmp_path):
    cache = tmp_path / "cache"
    runs = {
        "n2": write_config(tmp_path, concurrency=2, name="n2"),
        "n4": write_config(tmp_path, concurrency=4, name="n4"),         # policy differs
        "small": write_config(tmp_path, memory_budget=8192, name="small"),  # runtime differs
    }
    for name, config in runs.items():
        main(["run", str(config), "--out", str(tmp_path / name),
              "--unloaded-cache", str(cache)])

    # the admission policy does not change the measurement; the runtime config does
    assert len(list(cache.iterdir())) == 2
    assert ((tmp_path / "n2" / "unloaded.jsonl").read_bytes()
            == (tmp_path / "n4" / "unloaded.jsonl").read_bytes())

    # a cached measurement is reused, not re-measured
    for entry in cache.iterdir():
        cached = json.loads(entry.read_text())
        for r in cached["records"]:
            r["unloaded_completion_s"] = 1234.0
        entry.write_text(json.dumps(cached))
    main(["run", str(runs["n2"]), "--out", str(tmp_path / "again"),
          "--unloaded-cache", str(cache)])
    again = read_jsonl(tmp_path / "again" / "unloaded.jsonl")
    assert all(u["unloaded_completion_s"] == 1234.0 for u in again)


def analyze(run_dir: Path, capsys) -> dict[str, float]:
    capsys.readouterr()
    main(["analyze", str(run_dir)])
    metrics = {}
    for line in capsys.readouterr().out.splitlines():
        name, value = line.split(": ")
        metrics[name] = float(value)
    return metrics


def test_under_heavy_load_goodput_falls_below_throughput(tmp_path, capsys):
    run_dir = tmp_path / "run"
    # ~2 agent tasks/s against N=2: agent tasks queue for admission far longer
    # than 3x their unloaded completion time.
    main(["run", str(write_config(tmp_path, arrival_rate=2.0, concurrency=2)),
          "--out", str(run_dir)])

    metrics = analyze(run_dir, capsys)

    assert metrics["failed_tasks"] == 0
    assert metrics["throughput_tasks_per_min"] > 0
    assert metrics["goodput_tasks_per_min"] < 0.5 * metrics["throughput_tasks_per_min"]


def test_under_light_load_almost_every_agent_task_meets_its_task_slo(tmp_path, capsys):
    run_dir = tmp_path / "run"
    main(["run", str(write_config(tmp_path, arrival_rate=0.05, concurrency=4, duration_s=600)),
          "--out", str(run_dir)])

    metrics = analyze(run_dir, capsys)

    assert metrics["agent_tasks"] > 10
    assert metrics["slo_attainment"] > 0.9
