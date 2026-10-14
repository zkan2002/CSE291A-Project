"""One experiment: config file -> run directory."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from agentserve.clock import Clock, VirtualClock
from agentserve.policy import AdmissionPolicy, build_policy
from agentserve.proxy import Proxy
from agentserve.recorder import Recorder
from agentserve.simulated import SimulatedRuntime
from agentserve import unloaded
from agentserve.workload import WorkloadConfig, generate


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


def resident_cache_policy(config: dict[str, Any]) -> bool:
    """Whether the run's cache policy is *resident* (``--no-cache-idle-slots``,
    retained context stays in the KV pool) rather than *offload*
    (``--cache-idle-slots``). The flag is under [server] for a real run, under
    [runtime] for a simulated one."""
    return not config.get("server", config["runtime"]).get("cache_idle_slots", False)


def admission_policy(config: dict[str, Any]) -> AdmissionPolicy:
    """The [policy] table's admission policy. Agent tasks are pinned to slots
    (``id_slot``) under the resident cache policy and not under offload (pilot
    amendment 6), unless ``pin_slots`` says otherwise."""
    policy_cfg = config["policy"]
    return build_policy(policy_cfg["name"], concurrency=policy_cfg["concurrency"],
                        pin_slots=policy_cfg.get("pin_slots", resident_cache_policy(config)))


def run_experiment(config: dict[str, Any], run_dir: Path, *,
                   unloaded_cache_dir: Path | None = None) -> None:
    run, runtime_cfg = config["run"], config["runtime"]
    if runtime_cfg["kind"] == "llama_server":
        from agentserve.real_experiment import run_real_experiment
        return run_real_experiment(config, run_dir, unloaded_cache_dir=unloaded_cache_dir)
    if runtime_cfg["kind"] != "simulated":
        raise ValueError(f"unsupported runtime kind: {runtime_cfg['kind']!r}")
    clock: Clock = VirtualClock()

    workload = WorkloadConfig.from_config(config["workload"])
    tasks = generate(workload, duration_s=run["duration_s"], seed=run["seed"],
                     split=run.get("split", "dev"))
    runtime = simulated_runtime(clock, runtime_cfg)
    recorder = Recorder(run_dir, config, record_prompts=run.get("record_prompts", False))
    try:
        # Unloaded completion times, task SLOs and hard timeouts (by task_id).
        targets = {r.task_id: r for r in _unloaded(config, tasks, unloaded_cache_dir)}
        recorder.unloaded(targets.values())
        proxy = Proxy(policy=admission_policy(config),
                      runtime=runtime, clock=clock, recorder=recorder,
                      memory_budget_tokens=runtime_cfg["memory_budget_tokens"],
                      n_slots=runtime_cfg["n_slots"],
                      hard_timeouts={t: r.hard_timeout_s for t, r in targets.items()})
        clock.run(proxy.run(tasks))
    finally:
        recorder.close()


def _unloaded(config: dict[str, Any], tasks, cache_dir: Path | None,
              measure_real=None) -> list[unloaded.UnloadedRecord]:
    """Each agent task run alone on a fresh, idle runtime with its own clock.
    Cached per agent task and runtime config (``unloaded.cached``): neither the
    admission policy, the arrival process nor the slot count is part of the key.
    A real run passes ``measure_real(tasks)``, which measures on a fresh llama-server."""
    runtime_cfg = config["runtime"]
    key = {"runtime": {k: v for k, v in runtime_cfg.items() if k != "n_slots"}}
    if "server" in config:  # real runs: the server launch config is part of the runtime
        key["server"] = {k: v for k, v in config["server"].items()
                         if k not in ("host", "port", "slots")}

    def measure(subset) -> list[unloaded.UnloadedRecord]:
        clock = VirtualClock()
        runtime = simulated_runtime(clock, runtime_cfg)
        return unloaded.measure(subset, clock=clock, runtime=runtime,
                                memory_budget_tokens=runtime_cfg["memory_budget_tokens"],
                                n_slots=runtime_cfg["n_slots"],
                                pin_slots=resident_cache_policy(config))

    return unloaded.cached(key, cache_dir, tasks, measure_real or measure)
