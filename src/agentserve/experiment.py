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
from agentserve import unloaded
from agentserve.workload import WorkloadConfig, generate


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def run_experiment(config: dict[str, Any], run_dir: Path, *,
                   unloaded_cache_dir: Path | None = None) -> None:
    run, runtime_cfg, policy_cfg = config["run"], config["runtime"], config["policy"]
    if runtime_cfg["kind"] == "llama_server":
        from agentserve.real_experiment import run_real_experiment
        return run_real_experiment(config, run_dir, unloaded_cache_dir=unloaded_cache_dir)
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
        # Unloaded completion times, task SLOs and hard timeouts (by task_id).
        # TODO(#6): enforce ``targets[task_id].hard_timeout_s`` in the proxy.
        targets = {r.task_id: r for r in _unloaded(config, tasks, unloaded_cache_dir)}
        recorder.unloaded(targets.values())
        proxy = Proxy(policy=build_policy(policy_cfg["name"], concurrency=policy_cfg["concurrency"]),
                      runtime=runtime, clock=clock, recorder=recorder,
                      memory_budget_tokens=runtime_cfg["memory_budget_tokens"],
                      n_slots=runtime_cfg["n_slots"])
        clock.run(proxy.run(tasks))
    finally:
        recorder.close()


def _unloaded(config: dict[str, Any], tasks, cache_dir: Path | None,
              measure_real=None) -> list[unloaded.UnloadedRecord]:
    """Each agent task run alone on a fresh, idle runtime with its own clock.
    Cached per (workload, runtime config): the admission policy is not part of the key.
    A real run passes ``measure_real``, which measures on a fresh llama-server."""
    run, runtime_cfg = config["run"], config["runtime"]
    key = {"workload": config["workload"], "seed": run["seed"], "duration_s": run["duration_s"],
           "split": run.get("split", "dev"), "runtime": runtime_cfg}
    if "server" in config:  # real runs: the server launch config is part of the runtime
        key["server"] = config["server"]

    def measure() -> list[unloaded.UnloadedRecord]:
        clock = VirtualClock()
        runtime = SimulatedRuntime(clock, n_slots=runtime_cfg["n_slots"],
                                   prompt_token_s=runtime_cfg["prompt_token_s"],
                                   output_token_s=runtime_cfg["output_token_s"])
        return unloaded.measure(tasks, clock=clock, runtime=runtime,
                                memory_budget_tokens=runtime_cfg["memory_budget_tokens"],
                                n_slots=runtime_cfg["n_slots"])

    return unloaded.cached(key, cache_dir, measure_real or measure)
