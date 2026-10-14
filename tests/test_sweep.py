"""The sweep runner, end to end on the simulated runtime and virtual clock: a sweep
config expands into cells (admission policy x N x memory budget x arrival rate x
cache policy x seed), each cell is one run directory, and an interrupted sweep
resumes by running only the cells it hadn't finished."""

import json
from pathlib import Path

import pytest

from agentserve.analysis import aggregate, summarize
from agentserve.cli import main
from agentserve.experiment import run_experiment
from agentserve.report import find_run_dirs
from agentserve.sweep import load_sweep, run_sweep

SWEEP = """\
[sweep]
seeds = [1, 2]

[base.run]
warmup_s = 10
duration_s = 40

[base.workload]
arrival_rate_per_s = 0.5

[[base.workload.classes]]
name = "short"
weight = 1.0
initial_context_tokens = 500
output_tokens = 50
steps = 3

[base.policy]
name = "task_level_fifo"
concurrency = 1

[base.runtime]
kind = "simulated"
n_slots = 1
memory_budget_tokens = 32768
prompt_token_s = 0.0002
output_token_s = 0.01
cache_ram_tokens = 50000

{grids}
"""

GRID = """\
[[grid]]
policy = ["task_level_fifo", "request_level_fifo"]
concurrency = [2, 4]
memory_budget = [16384]
arrival_rate = [0.5]
cache_policy = ["resident"]
"""


def write_sweep(tmp_path: Path, grids: str = GRID) -> Path:
    path = tmp_path / "sweep.toml"
    path.write_text(SWEEP.format(grids=grids))
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_a_sweep_runs_every_cell_of_its_grid_into_its_own_run_directory(tmp_path):
    out = tmp_path / "sweep"

    assert main(["sweep", str(write_sweep(tmp_path)), "--out", str(out)]) == 0

    runs = find_run_dirs([out])
    assert len(runs) == 2 * 2 * 2  # policy x N x seed
    configs = [json.loads((r / "config.json").read_text()) for r in runs]
    cells = sorted((c["policy"]["name"], c["policy"]["concurrency"], c["run"]["seed"])
                   for c in configs)
    assert cells == sorted((p, n, s) for p in ("task_level_fifo", "request_level_fifo")
                           for n in (2, 4) for s in (1, 2))
    for c in configs:
        assert c["runtime"]["memory_budget_tokens"] == 16384
        assert c["workload"]["arrival_rate_per_s"] == 0.5
        assert c["runtime"]["n_slots"] == c["policy"]["concurrency"]  # one slot per place
        assert c["run"]["warmup_s"] == 10  # the base config's run protocol
    # runs differing only in the seed form one configuration in the analysis
    configurations = aggregate([summarize(r) for r in runs])
    assert len(configurations) == 4 and all(c.seeds == [1, 2] for c in configurations)


def test_cache_policies_map_to_the_runtimes_cache_flags(tmp_path):
    out = tmp_path / "sweep"
    grid = GRID.replace('cache_policy = ["resident"]',
                        'cache_policy = ["resident", "offload", "off"]').replace(
        'concurrency = [2, 4]', 'concurrency = [2]').replace(
        'policy = ["task_level_fifo", "request_level_fifo"]', 'policy = ["task_level_fifo"]')

    main(["sweep", str(write_sweep(tmp_path, grid)), "--out", str(out)])

    configurations = aggregate([summarize(r) for r in find_run_dirs([out])])
    by_cache = {c.params["cache_policy"]: c for c in configurations}
    assert sorted(by_cache) == ["off", "offload", "resident"]
    assert by_cache["off"].metrics["cached_tokens_per_request"].mean == 0
    assert by_cache["resident"].metrics["cached_tokens_per_request"].mean > 0
    assert by_cache["offload"].config["runtime"]["cache_idle_slots"] is True


def test_several_grids_form_one_sweep_without_duplicate_cells(tmp_path):
    out = tmp_path / "sweep"
    second = GRID.replace('concurrency = [2, 4]', 'concurrency = [4, 8]')

    main(["sweep", str(write_sweep(tmp_path, GRID + "\n" + second)), "--out", str(out)])

    # N = 4 is in both grids but runs once
    assert len(find_run_dirs([out])) == 2 * 3 * 2


def test_an_interrupted_sweep_resumes_with_only_the_cells_it_had_not_finished(tmp_path):
    sweep = load_sweep(write_sweep(tmp_path))
    out = tmp_path / "sweep"
    started = []

    def interrupted_on_the_third_cell(config, run_dir, **kwargs):
        started.append(run_dir)
        if len(started) == 3:  # killed part-way through its run
            run_dir.mkdir(parents=True)
            (run_dir / "tasks.jsonl").write_text('{"task_id": 0}\n')
            raise KeyboardInterrupt
        run_experiment(config, run_dir, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        run_sweep(sweep, out, run=interrupted_on_the_third_cell)
    finished = {r: (r / "tasks.jsonl").stat().st_mtime_ns for r in find_run_dirs([out])}
    assert len(finished) == 2  # the interrupted cell's partial run is not a result

    resumed = []

    def counting(config, run_dir, **kwargs):
        resumed.append(config)
        run_experiment(config, run_dir, **kwargs)

    run_sweep(sweep, out, run=counting)

    assert len(resumed) == 8 - 2
    runs = find_run_dirs([out])
    assert len(runs) == 8
    # the finished cells were left alone
    assert all((r / "tasks.jsonl").stat().st_mtime_ns == t for r, t in finished.items())
    # the sweep log records every attempt, the interrupted one included
    log = read_jsonl(out / "sweep.jsonl")
    assert [e["status"] for e in log].count("interrupted") == 1
    assert [e["status"] for e in log].count("complete") == 8
    assert [e["status"] for e in log].count("skipped") == 2


def test_a_failed_cell_is_recorded_and_the_sweep_goes_on(tmp_path, capsys):
    sweep = load_sweep(write_sweep(tmp_path))
    out = tmp_path / "sweep"
    calls = []

    def first_cell_crashes(config, run_dir, **kwargs):
        calls.append(run_dir)
        if len(calls) == 1:
            run_dir.mkdir(parents=True)
            raise RuntimeError("llama-server exited during startup with code 1")
        run_experiment(config, run_dir, **kwargs)

    result = run_sweep(sweep, out, run=first_cell_crashes)

    assert len(result.failed) == 1 and len(result.complete) == 7
    [failed] = [e for e in read_jsonl(out / "sweep.jsonl") if e["status"] == "failed"]
    assert "exited during startup" in failed["error"]
    assert len(find_run_dirs([out])) == 7

    # resuming retries it, and only it
    retried = []
    run_sweep(sweep, out, run=lambda c, d, **k: (retried.append(d), run_experiment(c, d, **k)))
    assert len(retried) == 1 and len(find_run_dirs([out])) == 8


def test_resuming_with_a_changed_cell_config_is_refused(tmp_path):
    out = tmp_path / "sweep"
    main(["sweep", str(write_sweep(tmp_path)), "--out", str(out)])
    changed = load_sweep(write_sweep(tmp_path))
    changed["base"]["runtime"]["output_token_s"] = 0.02

    with pytest.raises(ValueError, match="differs"):
        run_sweep(changed, out)
