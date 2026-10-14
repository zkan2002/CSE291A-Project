"""End-to-end experiments on the simulated runtime and virtual clock.

Every test here runs a whole experiment (workload -> proxy + admission policy ->
simulated runtime -> recorder) through the one-command entry point, and checks
only what lands in the run directory.
"""

import json
import time
from pathlib import Path

from agentserve.cli import main

CONFIG = """\
[run]
seed = {seed}
duration_s = {duration_s}

[workload]
arrival_rate_per_s = {arrival_rate}

[[workload.classes]]
name = "short"
weight = 1.0
initial_context_tokens = 500
output_tokens = 50
steps = 3
tool_result_tokens = 200
tool_delay_median_s = 1.0
tool_delay_sigma = 0.5

[policy]
name = "task_level_fifo"
concurrency = {concurrency}

[runtime]
kind = "simulated"
n_slots = {n_slots}
memory_budget_tokens = 32768
prompt_token_s = 0.0002
output_token_s = 0.01
"""


def write_config(tmp_path: Path, *, seed=1, duration_s=60, arrival_rate=0.5,
                 concurrency=4, n_slots=4) -> Path:
    path = tmp_path / f"config-{seed}-{duration_s}-{concurrency}.toml"
    path.write_text(CONFIG.format(seed=seed, duration_s=duration_s,
                                  arrival_rate=arrival_rate,
                                  concurrency=concurrency, n_slots=n_slots))
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_one_command_runs_an_experiment_into_a_run_directory(tmp_path):
    run_dir = tmp_path / "run"

    exit_code = main(["run", str(write_config(tmp_path)), "--out", str(run_dir)])

    assert exit_code == 0
    config = json.loads((run_dir / "config.json").read_text())
    assert config["run"]["seed"] == 1
    assert config["policy"]["name"] == "task_level_fifo"
    tasks = read_jsonl(run_dir / "tasks.jsonl")
    requests = read_jsonl(run_dir / "requests.jsonl")
    assert len(tasks) > 0
    assert all(t["failure"] is None for t in tasks)
    # 3 agent steps per agent task, one model request each
    assert len(requests) == 3 * len(tasks)
    assert all(r["n_output_tokens"] == 50 for r in requests)


def max_active_agent_tasks(tasks: list[dict]) -> int:
    """An agent task is active from its first admission until it completes (one
    that timed out before its first admission never was)."""
    events = []
    for t in tasks:
        if not t["step_admitted_s"]:
            continue
        events.append((t["step_admitted_s"][0], 1))
        events.append((t["completed_s"], -1))
    active = peak = 0
    for _, delta in sorted(events):  # at equal times, completions (-1) sort first
        active += delta
        peak = max(peak, active)
    return peak


def test_task_level_admission_never_has_more_than_n_agent_tasks_active(tmp_path):
    run_dir = tmp_path / "run"
    # Overloaded: ~2 agent tasks/s against N=3, so admission is the bottleneck.
    config = write_config(tmp_path, arrival_rate=2.0, concurrency=3, n_slots=8)

    main(["run", str(config), "--out", str(run_dir)])

    tasks = read_jsonl(run_dir / "tasks.jsonl")
    assert max_active_agent_tasks(tasks) == 3
    # and agent tasks really did queue for admission
    assert any(t["step_admitted_s"] and t["step_admitted_s"][0] > t["arrived_s"] for t in tasks)


def test_same_config_and_seed_produce_identical_recorded_output(tmp_path):
    config = write_config(tmp_path, seed=7, arrival_rate=1.0, concurrency=3)
    first, second = tmp_path / "first", tmp_path / "second"

    main(["run", str(config), "--out", str(first)])
    main(["run", str(config), "--out", str(second)])

    for name in ("config.json", "requests.jsonl", "tasks.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_a_different_seed_produces_a_different_workload(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"

    main(["run", str(write_config(tmp_path, seed=1)), "--out", str(a)])
    main(["run", str(write_config(tmp_path, seed=2)), "--out", str(b)])

    assert (a / "tasks.jsonl").read_bytes() != (b / "tasks.jsonl").read_bytes()


def test_ten_virtual_minutes_finish_in_a_few_seconds(tmp_path):
    run_dir = tmp_path / "run"
    config = write_config(tmp_path, duration_s=600, arrival_rate=0.5, concurrency=4)

    started = time.perf_counter()
    main(["run", str(config), "--out", str(run_dir)])
    elapsed = time.perf_counter() - started

    tasks = read_jsonl(run_dir / "tasks.jsonl")
    assert max(t["arrived_s"] for t in tasks) > 540  # arrivals span ~10 virtual minutes
    assert elapsed < 5.0
