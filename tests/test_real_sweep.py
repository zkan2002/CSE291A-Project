"""A tiny grid against the real pinned llama-server, through the one sweep command
(opt in: `pytest -m real_runtime`): both baselines, fresh servers for every run."""

import json
from pathlib import Path

import pytest

from agentserve.cli import main
from agentserve.report import find_run_dirs

pytestmark = pytest.mark.real_runtime

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "real-smoke.toml"

GRID = """\
[sweep]
seeds = [1]

[[grid]]
policy = ["task_level_fifo", "request_level_fifo"]
concurrency = [2]
memory_budget = [8192]
cache_policy = ["resident"]
"""


def test_one_command_runs_a_grid_on_the_real_server_with_a_restart_per_run(tmp_path):
    import tomllib
    base = tomllib.loads(SMOKE.read_text())
    if not Path(base["server"]["binary"]).exists():
        pytest.skip("pinned llama-server not present (see docs/environment.md)")
    base["run"].update(duration_s=12, warmup_s=4)
    sweep = tmp_path / "sweep.toml"
    sweep.write_text(GRID + _base_tables(base))
    out = tmp_path / "out"

    assert main(["sweep", str(sweep), "--out", str(out)]) == 0

    runs = find_run_dirs([out])
    assert len(runs) == 2
    pids = []
    for run_dir in runs:
        tasks = [json.loads(line) for line in (run_dir / "tasks.jsonl").read_text().splitlines()]
        assert tasks and all(t["failure"] is None for t in tasks)
        assert json.loads((run_dir / "window.json").read_text()) == {"start_s": 4, "end_s": 12}
        launches = json.loads((run_dir / "cold_start.json").read_text())["launches"]
        assert launches[0]["role"] == "cold_start" and launches[-1]["role"] == "experiment"
        pids += [l["pid"] for l in launches]
    assert len(set(pids)) == len(pids)  # every launch a fresh process
    log = [json.loads(line) for line in (out / "sweep.jsonl").read_text().splitlines()]
    assert [e["status"] for e in log] == ["complete", "complete"]
    assert all(e["fresh_servers"] for e in log)
    # the second run reused the first one's unloaded measurement (same runtime config)
    assert [l["role"] for l in json.loads(
        (runs[0] / "cold_start.json").read_text())["launches"]].count("unloaded") + [
        l["role"] for l in json.loads((runs[1] / "cold_start.json").read_text())["launches"]
    ].count("unloaded") == 1

    report = tmp_path / "report"
    assert main(["report", str(out), "--out", str(report)]) == 0
    assert "## Cold start" in (report / "metrics.md").read_text(encoding="utf-8")


def _base_tables(base: dict) -> str:
    lines = []
    for table, values in base.items():
        if table == "workload":
            classes = values.pop("classes")
        lines.append(f"\n[base.{table}]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in values.items()]
        if table == "workload":
            for c in classes:
                lines.append("\n[[base.workload.classes]]")
                lines += [f"{k} = {json.dumps(v)}" for k, v in c.items()]
    return "\n".join(lines) + "\n"
