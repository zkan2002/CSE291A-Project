"""The workload, observed through recorded experiments on the simulated runtime.

Each test runs a whole experiment through the one-command entry point and checks
what lands in the run directory.
"""

import json
from pathlib import Path

from agentserve.cli import main

TWO_CLASSES = """\
[[workload.classes]]
name = "short"
weight = 1.0
initial_context_tokens = 500
output_tokens = 50
steps = [2, 3]
tool_result_tokens = 200
tool_delay_median_s = 1.0
tool_delay_sigma = 0.5

[[workload.classes]]
name = "long"
weight = 1.0
initial_context_tokens = 4000
output_tokens = 400
steps = [8, 12]
"""

POLICY_AND_RUNTIME = """\
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


def write_config(tmp_path: Path, *, run: str = "", workload: str = "",
                 classes: str = TWO_CLASSES, name: str = "config") -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(
        "[run]\nseed = 1\nduration_s = 60\n" + run
        + "\n[workload]\narrival_rate_per_s = 0.5\n" + workload
        + "\n" + classes + "\n" + POLICY_AND_RUNTIME)
    return path


def run(tmp_path: Path, config: Path, out: str = "run") -> Path:
    run_dir = tmp_path / out
    assert main(["run", str(config), "--out", str(run_dir)]) == 0
    return run_dir


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def by_task(records: list[dict]) -> dict[int, list[dict]]:
    tasks: dict[int, list[dict]] = {}
    for r in sorted(records, key=lambda r: (r["task_id"], r["step"])):
        tasks.setdefault(r["task_id"], []).append(r)
    return tasks


def test_every_agent_step_prompt_extends_the_previous_one_by_its_output_and_a_tool_result(tmp_path):
    run_dir = run(tmp_path, write_config(tmp_path, run="record_prompts = true\n"))

    prompts = by_task(read_jsonl(run_dir / "prompts.jsonl"))
    requests = {(r["task_id"], r["step"]): r for r in read_jsonl(run_dir / "requests.jsonl")}
    assert prompts
    for task_id, steps in prompts.items():
        assert [s["step"] for s in steps] == list(range(len(steps)))
        for prev, cur in zip(steps, steps[1:]):
            assert cur["prompt_tokens"][:len(prev["prompt_tokens"])] == prev["prompt_tokens"]
            grown = len(cur["prompt_tokens"]) - len(prev["prompt_tokens"])
            n_output = requests[(task_id, prev["step"])]["n_output_tokens"]
            assert grown == n_output + 200  # previous output, then the tool result
        for s in steps:
            assert len(s["prompt_tokens"]) == requests[(task_id, s["step"])]["n_prompt_tokens"]


def test_every_recorded_output_length_equals_its_n_predict(tmp_path):
    run_dir = run(tmp_path, write_config(tmp_path))

    requests = read_jsonl(run_dir / "requests.jsonl")
    tasks = {t["task_id"]: t for t in read_jsonl(run_dir / "tasks.jsonl")}
    assert requests
    for r in requests:
        assert r["n_output_tokens"] == r["n_predict"]
        assert r["n_predict"] == {"short": 50, "long": 400}[tasks[r["task_id"]]["task_class"]]
