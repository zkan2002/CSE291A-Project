"""Analysis is tested at the log boundary: hand-written run directories with known answers."""

import json
from pathlib import Path

from agentserve.cli import main


def write_run(run_dir: Path, tasks: list[dict]) -> None:
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")
    (run_dir / "requests.jsonl").write_text("")
    (run_dir / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))


def task(task_id, arrived, completed, failure=None):
    return {"task_id": task_id, "task_class": "short", "arrived_s": arrived,
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
