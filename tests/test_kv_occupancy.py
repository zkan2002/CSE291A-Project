"""KV occupancy over time: the server does not report it (pilot amendment 7), so the
proxy tracks the token footprint it has admitted and records it in ``kv.jsonl``;
a simulated run records the simulated KV pool's own occupancy beside it. End to
end on the simulated runtime and virtual clock."""

import json
from pathlib import Path

from agentserve.cli import main

CONFIG = """\
[run]
seed = 4
duration_s = {duration_s}

[workload]
arrival_rate_per_s = {arrival_rate}

[[workload.classes]]
name = "two-step"
weight = 1.0
initial_context_tokens = 500
output_tokens = 50
steps = 2
tool_result_tokens = 200
tool_delay_median_s = 1.0

[policy]
name = "{policy}"
concurrency = {concurrency}

[runtime]
kind = "simulated"
n_slots = {concurrency}
memory_budget_tokens = 32768
prompt_token_s = 0.0002
output_token_s = 0.01
cache_ram_tokens = 50000
cache_idle_slots = {offload}
"""


def run(tmp_path: Path, name: str, *, duration_s=30, arrival_rate=0.2,
        policy="task_level_fifo", concurrency=1, offload=False) -> Path:
    config = tmp_path / f"{name}.toml"
    config.write_text(CONFIG.format(duration_s=duration_s, arrival_rate=arrival_rate,
                                    policy=policy, concurrency=concurrency,
                                    offload=str(offload).lower()))
    run_dir = tmp_path / name
    assert main(["run", str(config), "--out", str(run_dir)]) == 0
    return run_dir


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_the_admitted_footprint_and_the_pools_occupancy_are_recorded_over_time(tmp_path):
    run_dir = run(tmp_path, "run")

    kv = read_jsonl(run_dir / "kv.jsonl")
    first_task = [(k["in_flight_tokens"], k["retained_tokens"], k["runtime_kv_tokens"])
                  for k in kv[:5]]
    # the first agent task, alone on the runtime and pinned to its slot:
    assert first_task == [
        (500 + 50, 0, 0),      # step 0 admitted: its prompt plus n_predict, not started yet
        (0, 550, 549),         # done: its context retained in the slot (the last token is
                               # never decoded, so the pool holds one fewer)
        (750 + 50, 0, 549),    # step 1 (prompt + output + tool result) admitted
        (0, 800, 799),
        (0, 0, 799),           # the agent task ends: the proxy pins nothing any more,
    ]                          # though the slot still holds its tokens
    assert all(k["admitted_tokens"] == k["in_flight_tokens"] + k["retained_tokens"] for k in kv)
    times = [k["t_s"] for k in kv]
    assert times == sorted(times)


def test_context_the_proxy_does_not_pin_is_not_counted_as_retained(tmp_path):
    # offload: agent tasks are not pinned; their context comes back through the
    # host-RAM prompt cache, which is outside the pool
    run_dir = run(tmp_path, "offload", policy="request_level_fifo", concurrency=2, offload=True)

    kv = read_jsonl(run_dir / "kv.jsonl")
    assert kv and all(k["retained_tokens"] == 0 for k in kv)
    assert max(k["in_flight_tokens"] for k in kv) > 0


def test_kv_occupancy_is_reproducible_on_the_virtual_clock(tmp_path):
    first = run(tmp_path, "first", duration_s=60, arrival_rate=1.0,
                policy="request_level_fifo", concurrency=3)
    second = run(tmp_path, "second", duration_s=60, arrival_rate=1.0,
                 policy="request_level_fifo", concurrency=3)

    assert (first / "kv.jsonl").read_bytes() == (second / "kv.jsonl").read_bytes()
