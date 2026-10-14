"""The run protocol, end to end on the simulated runtime and virtual clock: a warm-up
that is recorded but not measured, a measured window of arrivals, then a drain."""

import json
from pathlib import Path

import pytest

from agentserve.analysis import summarize
from agentserve.cli import main

CONFIG = """\
[run]
seed = 3
warmup_s = {warmup_s}
duration_s = {duration_s}

[workload]
arrival_rate_per_s = 0.5

[[workload.classes]]
name = "short"
weight = 1.0
initial_context_tokens = 500
output_tokens = 50
steps = 3

[policy]
name = "task_level_fifo"
concurrency = 4

[runtime]
kind = "simulated"
n_slots = 4
memory_budget_tokens = 32768
prompt_token_s = 0.0002
output_token_s = 0.01
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_warm_up_arrivals_are_logged_but_excluded_from_metrics(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text(CONFIG.format(warmup_s=20, duration_s=80))
    run_dir = tmp_path / "run"

    assert main(["run", str(config), "--out", str(run_dir)]) == 0

    tasks = read_jsonl(run_dir / "tasks.jsonl")
    warm_up = [t for t in tasks if t["arrived_s"] < 20]
    measured = [t for t in tasks if 20 <= t["arrived_s"] < 80]
    # the warm-up's agent tasks ran and are in the logs
    assert warm_up and measured and len(warm_up) + len(measured) == len(tasks)
    assert {r["task_id"] for r in read_jsonl(run_dir / "requests.jsonl")} >= {
        t["task_id"] for t in warm_up}
    # the drain: the run went on past the last arrival until every agent task finished
    assert max(t["completed_s"] for t in tasks) > max(t["arrived_s"] for t in tasks)

    # analysis picks the measured window up from the run directory, with no flags
    metrics = summarize(run_dir).metrics
    assert metrics["agent_tasks"] == len(measured)
    completed = sum(t["failure"] is None for t in measured)
    # rates are per minute over the 60 s measured window
    assert metrics["throughput_tasks_per_min"] == pytest.approx(completed)

    capsys.readouterr()
    assert main(["analyze", str(run_dir)]) == 0
    assert f"agent_tasks: {len(measured)}" in capsys.readouterr().out

    # and so does the report
    out = tmp_path / "report"
    assert main(["report", str(run_dir), "--out", str(out)]) == 0
    [row] = list(_csv_rows(out / "runs.csv"))
    assert int(row["agent_tasks"]) == len(measured)
    assert "[20 s, 80 s)" in (out / "metrics.md").read_text(encoding="utf-8")


def test_an_explicit_window_overrides_the_recorded_one(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text(CONFIG.format(warmup_s=20, duration_s=80))
    run_dir = tmp_path / "run"
    main(["run", str(config), "--out", str(run_dir)])
    tasks = read_jsonl(run_dir / "tasks.jsonl")

    capsys.readouterr()
    main(["analyze", str(run_dir), "--window-start", "0"])

    assert f"agent_tasks: {len(tasks)}" in capsys.readouterr().out


def _csv_rows(path: Path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)
