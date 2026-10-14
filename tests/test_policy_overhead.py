"""Policy overhead: the proxy times every admission-policy decision and writes it
to ``decisions.jsonl`` (wall-clock, so outside the byte-identical files), and
``agentserve analyze`` reports it."""

import json
from pathlib import Path

from agentserve.cli import main

CONFIG = """\
[run]
seed = 3
duration_s = 30

[workload]
arrival_rate_per_s = 1.0

[[workload.classes]]
name = "three-step"
weight = 1.0
initial_context_tokens = 300
output_tokens = 20
steps = 3
tool_result_tokens = 100
tool_delay_median_s = 0.5

[policy]
name = "task_level_fifo"
concurrency = 2

[runtime]
kind = "simulated"
n_slots = 2
memory_budget_tokens = 16384
prompt_token_s = 0.0002
output_token_s = 0.01
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_every_policy_decision_is_timed_in_a_separate_wall_clock_file(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(CONFIG)
    run_dir = tmp_path / "run"

    main(["run", str(config), "--out", str(run_dir)])

    decisions = read_jsonl(run_dir / "decisions.jsonl")
    requests = read_jsonl(run_dir / "requests.jsonl")
    assert decisions
    for d in decisions:
        assert d["decision_s"] >= 0.0
        assert d["n_pending"] >= 1
        assert 0 <= d["n_admitted"] <= d["n_pending"]
    # every model request was admitted by exactly one timed decision
    assert sum(d["n_admitted"] for d in decisions) == len(requests)
    # decisions are logged on the run clock, in order
    times = [d["t_s"] for d in decisions]
    assert times == sorted(times)
    # with N=2 some decisions admitted nothing (agent tasks waited for a slot)
    assert any(d["n_admitted"] == 0 for d in decisions)
