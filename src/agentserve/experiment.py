"""One experiment: config file -> run directory."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from agentserve import config as cfg
from agentserve.clock import Clock, VirtualClock
from agentserve.policy import AdmissionPolicy, build_policy
from agentserve.proxy import Proxy
from agentserve.recorder import Recorder
from agentserve.simulated import SimulatedRuntime
from agentserve import unloaded
from agentserve.workload import AgentTaskSpec, WorkloadConfig, generate


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def simulated_runtime(clock: Clock, runtime_cfg: dict[str, Any]) -> SimulatedRuntime:
    """The simulated runtime a ``[runtime]`` table with ``kind = "simulated"`` describes."""
    return SimulatedRuntime(clock, n_slots=runtime_cfg["n_slots"],
                            pool_tokens=runtime_cfg["memory_budget_tokens"],
                            prompt_token_s=runtime_cfg["prompt_token_s"],
                            output_token_s=runtime_cfg["output_token_s"],
                            cache_ram_tokens=runtime_cfg.get("cache_ram_tokens", 0),
                            cache_idle_slots=runtime_cfg.get("cache_idle_slots", False),
                            cache_prompt=runtime_cfg.get("cache_prompt", True))


def admission_policy(config: dict[str, Any]) -> AdmissionPolicy:
    """The [policy] table's admission policy, pinning agent tasks to slots
    (``id_slot``) as ``config.pin_slots`` says (pilot amendment 6)."""
    policy_cfg = config["policy"]
    return build_policy(policy_cfg["name"], concurrency=policy_cfg["concurrency"],
                        pin_slots=cfg.pin_slots(config))


def workload_tasks(config: dict[str, Any]) -> list[AgentTaskSpec]:
    """The run's agent tasks: the workload of its config, seed and split."""
    run = config["run"]
    return generate(WorkloadConfig.from_config(config["workload"]), duration_s=run["duration_s"],
                    seed=run["seed"], split=cfg.split(config))


def run_experiment(config: dict[str, Any], run_dir: Path, *,
                   unloaded_cache_dir: Path | None = None) -> None:
    runtime_cfg = config["runtime"]
    if cfg.is_real(config):
        from agentserve.real_experiment import run_real_experiment
        return run_real_experiment(config, run_dir, unloaded_cache_dir=unloaded_cache_dir)
    if runtime_cfg["kind"] != "simulated":
        raise ValueError(f"unsupported runtime kind: {runtime_cfg['kind']!r}")
    clock: Clock = VirtualClock()

    tasks = workload_tasks(config)
    runtime = simulated_runtime(clock, runtime_cfg)
    recorder = Recorder(run_dir, config, record_prompts=cfg.record_prompts(config))
    try:
        # Unloaded completion times, task SLOs and hard timeouts (by task_id).
        targets = {r.task_id: r for r in _unloaded(config, tasks, unloaded_cache_dir)}
        recorder.unloaded(targets.values())
        proxy = Proxy(policy=admission_policy(config),
                      runtime=runtime, clock=clock, recorder=recorder,
                      memory_budget_tokens=cfg.memory_budget_tokens(config),
                      n_slots=cfg.slots(config),
                      hard_timeouts={t: r.hard_timeout_s for t, r in targets.items()})
        clock.run(proxy.run(tasks))
    finally:
        recorder.close()


def _unloaded(config: dict[str, Any], tasks, cache_dir: Path | None,
              measure_real=None) -> list[unloaded.UnloadedRecord]:
    """Each agent task run alone on a fresh, idle runtime with its own clock, on
    the reference runtime config (``config.unloaded_reference``): the same for
    every admission policy, cache policy and memory budget. Cached per agent task
    and reference config (``unloaded.cached``); the slot count is not part of the
    key. A real run passes ``measure_real(reference, tasks)``, which measures on a
    fresh llama-server launched from the reference config."""
    reference = cfg.unloaded_reference(config)
    key = {"runtime": {k: v for k, v in reference["runtime"].items()
                       if k not in ("n_slots", "sample_interval_s")}}
    if "server" in reference:  # real runs: the server launch config is part of the runtime
        key["server"] = {k: v for k, v in reference["server"].items()
                         if k not in ("host", "port", "slots")}

    def measure(subset) -> list[unloaded.UnloadedRecord]:
        if measure_real is not None:
            return measure_real(reference, subset)
        clock = VirtualClock()
        runtime = simulated_runtime(clock, reference["runtime"])
        return unloaded.measure(subset, clock=clock, runtime=runtime,
                                memory_budget_tokens=cfg.memory_budget_tokens(reference),
                                n_slots=cfg.slots(reference), pin_slots=True)

    return unloaded.cached(key, cache_dir, tasks, measure)
