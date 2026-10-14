"""Request-level FIFO admission, slot pinning and failure accounting.

Every test runs a whole experiment (workload -> proxy + admission policy ->
simulated runtime with its KV pool -> recorder) through the one-command entry
point on the virtual clock, and checks only what lands in the run directory
(and what ``agentserve analyze`` prints).
"""

import json
from pathlib import Path

import pytest

from agentserve.cli import main

CONFIG = """\
[run]
seed = {seed}
duration_s = {duration_s}

[workload]
arrival_rate_per_s = {arrival_rate}

[[workload.classes]]
name = "three-step"
weight = 1.0
initial_context_tokens = {context}
output_tokens = 50
steps = 3
tool_result_tokens = 200
tool_delay_median_s = 1.0
tool_delay_sigma = 0.5

[policy]
name = "{policy}"
concurrency = {concurrency}
{policy_extra}

[runtime]
kind = "simulated"
n_slots = {n_slots}
memory_budget_tokens = {memory_budget}
prompt_token_s = 0.0002
output_token_s = 0.01
{runtime_extra}
"""


def write_config(tmp_path: Path, *, name="config", seed=1, duration_s=60, arrival_rate=0.5,
                 context=500, policy="request_level_fifo", concurrency=4, n_slots=4,
                 memory_budget=32768, policy_extra="", runtime_extra="") -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(CONFIG.format(
        seed=seed, duration_s=duration_s, arrival_rate=arrival_rate, context=context,
        policy=policy, concurrency=concurrency, policy_extra=policy_extra, n_slots=n_slots,
        memory_budget=memory_budget, runtime_extra=runtime_extra))
    return path


def run(tmp_path: Path, name="run", **config) -> Path:
    run_dir = tmp_path / name
    assert main(["run", str(write_config(tmp_path, name=name, **config)),
                 "--out", str(run_dir)]) == 0
    return run_dir


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def max_in_flight(requests: list[dict]) -> int:
    """A model request is in flight from its admission until it completes or fails."""
    events = []
    for r in requests:
        if r["admitted_s"] is not None:
            events.append((r["admitted_s"], 1))
            events.append((r["completed_s"], -1))
    active = peak = 0
    for _, delta in sorted(events):  # at equal times, completions (-1) sort first
        active += delta
        peak = max(peak, active)
    return peak


def test_request_level_admission_never_has_more_than_n_model_requests_in_flight(tmp_path):
    # Overloaded: ~2 agent tasks/s against N=3 on 8 slots, so admission is the bottleneck.
    run_dir = run(tmp_path, arrival_rate=2.0, concurrency=3, n_slots=8)

    requests = read_jsonl(run_dir / "requests.jsonl")
    tasks = read_jsonl(run_dir / "tasks.jsonl")
    assert max_in_flight(requests) == 3
    # model requests really did queue for admission
    assert any(r["admitted_s"] is not None and r["admitted_s"] > r["arrived_s"]
               for r in requests)
    # unlike task-level admission, more than N agent tasks are under way at once
    # (a task's next step re-queues after its tool delay)
    started = [t for t in tasks if t["step_admitted_s"]]
    starts = sorted(t["step_admitted_s"][0] for t in started)
    ends = sorted(t["completed_s"] for t in started)
    assert any(sum(s <= t for s in starts) - sum(e <= t for e in ends) > 3 for t in starts)


def analyze(run_dir: Path, capsys) -> dict[str, float]:
    capsys.readouterr()
    main(["analyze", str(run_dir)])
    metrics = {}
    for line in capsys.readouterr().out.splitlines():
        name, value = line.split(": ")
        metrics[name] = float(value)
    return metrics


def test_a_large_n_on_a_small_memory_budget_fails_agent_tasks_with_pool_exhausted(
        tmp_path, capsys):
    # The overflow cliff: 16 model requests in flight on 16 slots share an
    # 8k-token pool, but their prompts alone add up to more than that.
    run_dir = run(tmp_path, arrival_rate=2.0, concurrency=16, n_slots=16, memory_budget=8192)

    tasks = read_jsonl(run_dir / "tasks.jsonl")
    requests = read_jsonl(run_dir / "requests.jsonl")
    failed = [t for t in tasks if t["failure"] is not None]
    assert any(t["failure"] == "pool_exhausted" for t in failed)
    # each failed task's failing model request is recorded with the same cause,
    # and the task stops there: no silent retry, no later agent step
    for t in failed:
        own = sorted((r for r in requests if r["task_id"] == t["task_id"]),
                     key=lambda r: r["step"])
        assert own[-1]["error"] == t["failure"]
        assert all(r["error"] is None for r in own[:-1])
        assert len(own) == len({r["step"] for r in own})

    # failed tasks stay in every denominator
    metrics = analyze(run_dir, capsys)
    assert metrics["agent_tasks"] == len(tasks)
    assert metrics["failed_tasks"] == len(failed)
    # (analyze prints three decimals)
    assert metrics["slo_attainment"] <= (len(tasks) - len(failed)) / len(tasks) + 0.0005


@pytest.mark.parametrize("policy", ["task_level_fifo", "request_level_fifo"])
def test_an_agent_task_that_exceeds_its_hard_timeout_fails_with_cause_timeout(tmp_path, policy):
    # ~2 agent tasks/s against N=1: later agent tasks wait far longer than
    # 10x their unloaded completion time.
    run_dir = run(tmp_path, policy=policy, arrival_rate=2.0, concurrency=1, duration_s=30)

    tasks = read_jsonl(run_dir / "tasks.jsonl")
    requests = read_jsonl(run_dir / "requests.jsonl")
    hard_timeout = {u["task_id"]: u["hard_timeout_s"]
                    for u in read_jsonl(run_dir / "unloaded.jsonl")}
    timed_out = [t for t in tasks if t["failure"] == "timeout"]
    assert timed_out
    for t in tasks:
        # no agent task outlives its hard timeout
        assert t["completed_s"] - t["arrived_s"] <= hard_timeout[t["task_id"]] + 1e-9
    for t in timed_out:
        assert t["completed_s"] - t["arrived_s"] == pytest.approx(hard_timeout[t["task_id"]])
        own = sorted((r for r in requests if r["task_id"] == t["task_id"]),
                     key=lambda r: r["step"])
        assert [r["step"] for r in own] == list(range(len(own)))
        assert all(r["error"] is None for r in own[:-1])
        if own[-1]["error"] is not None:
            # it timed out waiting on a model request (queued or in flight),
            # which is recorded as failed by the timeout; otherwise it timed
            # out during a tool delay
            assert own[-1]["error"] == "timeout"
            assert own[-1]["completed_s"] == pytest.approx(t["completed_s"])
    # it timed out waiting for admission, so its model request was never admitted
    assert any(r["error"] == "timeout" and r["admitted_s"] is None for r in requests)
    # capacity a timed-out task held is given back: the N=1 bound still holds
    assert max_in_flight(requests) == 1
