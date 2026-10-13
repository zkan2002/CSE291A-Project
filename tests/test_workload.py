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
                 classes: str = TWO_CLASSES, name: str = "config",
                 seed: int = 1, duration_s: float = 60) -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(
        f"[run]\nseed = {seed}\nduration_s = {duration_s}\n" + run
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


SHORT_ONLY = """\
[[workload.classes]]
name = "short"
weight = 1.0
initial_context_tokens = 100
output_tokens = 10
steps = 2
"""


def arrival_counts(run_dir: Path, *, window_s: float, duration_s: float) -> list[int]:
    counts = [0] * int(duration_s / window_s)
    for t in read_jsonl(run_dir / "tasks.jsonl"):
        counts[int(t["arrived_s"] / window_s)] += 1
    return counts


def dispersion(counts: list[int]) -> float:
    """Variance-to-mean ratio of per-window arrival counts: about 1 for Poisson."""
    mean = sum(counts) / len(counts)
    return sum((c - mean) ** 2 for c in counts) / (len(counts) - 1) / mean


def test_bursty_arrivals_cluster_far_more_than_poisson_at_the_same_mean_rate(tmp_path):
    common = dict(classes=SHORT_ONLY, seed=3, duration_s=3000)
    poisson = write_config(tmp_path, name="poisson", **common)
    bursty = write_config(tmp_path, name="bursty", **common, workload=(
        'arrival_mode = "on_off"\nburst_on_s = 10.0\nburst_off_s = 30.0\n'))

    p = arrival_counts(run(tmp_path, poisson, "p"), window_s=10, duration_s=3000)
    b = arrival_counts(run(tmp_path, bursty, "b"), window_s=10, duration_s=3000)

    # Both average 0.5 agent tasks/s (~1500 arrivals); bursty totals vary more
    # from seed to seed because 3000 s holds only ~75 on/off cycles.
    assert 1350 < sum(p) < 1650
    assert 1125 < sum(b) < 1875
    # Poisson counts per 10 s window have variance ~= mean; bursty ones bunch
    # into on periods separated by silent off periods.
    assert dispersion(p) < 1.5
    assert dispersion(b) > 5.0
    assert b.count(0) > 100 > 10 > p.count(0)


