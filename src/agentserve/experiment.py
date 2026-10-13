"""One experiment: config file -> run directory."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from agentserve.clock import Clock, VirtualClock
from agentserve.policy import build_policy
from agentserve.proxy import Proxy
from agentserve.recorder import Recorder
from agentserve.simulated import SimulatedRuntime
from agentserve.workload import WorkloadConfig, generate


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def run_experiment(config: dict[str, Any], run_dir: Path) -> None:
    run, runtime_cfg, policy_cfg = config["run"], config["runtime"], config["policy"]
    if runtime_cfg["kind"] != "simulated":
        raise ValueError(f"unsupported runtime kind: {runtime_cfg['kind']!r}")
    clock: Clock = VirtualClock()

    workload = WorkloadConfig.from_config(config["workload"])
    tasks = generate(workload, duration_s=run["duration_s"], seed=run["seed"],
                     split=run.get("split", "dev"))
    runtime = SimulatedRuntime(clock, n_slots=runtime_cfg["n_slots"],
                               prompt_token_s=runtime_cfg["prompt_token_s"],
                               output_token_s=runtime_cfg["output_token_s"])
    recorder = Recorder(run_dir, config, record_prompts=run.get("record_prompts", False))
    try:
        proxy = Proxy(policy=build_policy(policy_cfg["name"], concurrency=policy_cfg["concurrency"]),
                      runtime=runtime, clock=clock, recorder=recorder,
                      memory_budget_tokens=runtime_cfg["memory_budget_tokens"],
                      n_slots=runtime_cfg["n_slots"])
        clock.run(proxy.run(tasks))
    finally:
        recorder.close()
