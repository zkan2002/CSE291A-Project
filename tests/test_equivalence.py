"""The machine-equivalence check pairs the same agent tasks' unloaded times across machines."""

import json

from agentserve.equivalence import speed_ratios


def _unloaded(run_dir, times):
    run_dir.mkdir(parents=True)
    with open(run_dir / "unloaded.jsonl", "w", encoding="utf-8") as f:
        for task_id, t in times.items():
            f.write(json.dumps({"task_id": task_id, "unloaded_completion_s": t}) + "\n")


def test_ratios_pair_the_same_agent_tasks_and_skip_untimed_ones(tmp_path):
    _unloaded(tmp_path / "a" / "cfg-s1", {0: 10.0, 1: 20.0, 2: None, 3: 5.0})
    _unloaded(tmp_path / "b" / "cfg-s1", {0: 11.0, 1: 18.0, 2: 7.0})
    assert speed_ratios(tmp_path / "a", tmp_path / "b", ["cfg-s1"]) == [1.1, 0.9]


def test_ratios_span_every_named_run(tmp_path):
    for seed in (1, 2):
        _unloaded(tmp_path / "a" / f"cfg-s{seed}", {0: 10.0})
        _unloaded(tmp_path / "b" / f"cfg-s{seed}", {0: 10.0 * seed})
    assert speed_ratios(tmp_path / "a", tmp_path / "b", ["cfg-s1", "cfg-s2"]) == [1.0, 2.0]
