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


def workload_fingerprint(run_dir: Path) -> list[tuple]:
    """What the workload put into a run: arrivals, classes and step counts."""
    steps = by_task(read_jsonl(run_dir / "requests.jsonl"))
    return [(t["arrived_s"], t["task_class"], len(steps[t["task_id"]]))
            for t in sorted(read_jsonl(run_dir / "tasks.jsonl"), key=lambda t: t["task_id"])]


def test_development_and_evaluation_workloads_never_share_a_seed(tmp_path):
    fingerprints = {}
    for split in ("dev", "eval"):
        for seed in range(5):
            config = write_config(tmp_path, name=f"{split}{seed}", seed=seed,
                                  run=f'split = "{split}"\n')
            fingerprints[(split, seed)] = workload_fingerprint(run(tmp_path, config, f"{split}{seed}"))

    for (split, seed), fp in fingerprints.items():
        assert fp
        for other_split in ("dev", "eval"):
            if other_split != split:
                assert all(fp != other for (s, _), other in fingerprints.items() if s == other_split)


def test_same_split_and_seed_reproduce_the_same_workload(tmp_path):
    config = write_config(tmp_path, seed=4, run='split = "eval"\nrecord_prompts = true\n')

    first, second = run(tmp_path, config, "first"), run(tmp_path, config, "second")

    for name in ("prompts.jsonl", "requests.jsonl", "tasks.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


HELD_OUT_WEIGHTING = """\
[[workload.classes]]
name = "dev_only"
weight = 1.0
eval_weight = 0.0
initial_context_tokens = 100
output_tokens = 10
steps = 2

[[workload.classes]]
name = "eval_only"
weight = 0.0
eval_weight = 1.0
initial_context_tokens = 100
output_tokens = 10
steps = 2
"""


def test_evaluation_workloads_use_the_held_out_mix_weighting(tmp_path):
    dev = run(tmp_path, write_config(tmp_path, name="dev", classes=HELD_OUT_WEIGHTING), "dev")
    ev = run(tmp_path, write_config(tmp_path, name="eval", classes=HELD_OUT_WEIGHTING,
                                    run='split = "eval"\n'), "eval")

    assert {t["task_class"] for t in read_jsonl(dev / "tasks.jsonl")} == {"dev_only"}
    assert {t["task_class"] for t in read_jsonl(ev / "tasks.jsonl")} == {"eval_only"}


