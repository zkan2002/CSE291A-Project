"""One experiment on the real pinned llama-server ([runtime] kind = "llama_server").

Launches the server from the config's [server] table, runs the same proxy, policy
and recorder as a simulated run on the real clock, samples the system every
``sample_interval_s``, and stops the server at the end. Besides the simulated
run-directory files it writes ``samples.jsonl``, ``metadata.json`` and ``server.log``.
The memory budget and slot count come from [server].

Unloaded completion times are measured first, on a fresh server of the same config
(log: ``unloaded-server.log``), unless cached; the experiment then gets a server of
its own, so no retained context carries over from the measurement.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from agentserve import unloaded
from agentserve.clock import RealClock
from agentserve.experiment import _unloaded, admission_policy, resident_cache_policy
from agentserve.llama_runtime import LlamaServerRuntime
from agentserve.proxy import Proxy
from agentserve.recorder import SERVER_LOG_FILE, Recorder
from agentserve.run_metadata import collect_run_metadata
from agentserve.sampler import SystemSampler, real_sources
from agentserve.server import ServerConfig, start_server, stop_server
from agentserve.workload import AgentTaskSpec, WorkloadConfig, generate


UNLOADED_SERVER_LOG_FILE = "unloaded-server.log"


def run_real_experiment(config: dict[str, Any], run_dir: Path, *,
                        unloaded_cache_dir: Path | None = None) -> None:
    run = config["run"]
    server = ServerConfig(**config["server"])
    tasks = generate(WorkloadConfig.from_config(config["workload"]), duration_s=run["duration_s"],
                     seed=run["seed"], split=run.get("split", "dev"))
    recorder = Recorder(run_dir, config, record_prompts=run.get("record_prompts", False))
    try:
        metadata = collect_run_metadata(server)
        recorder.metadata(metadata)
        targets = _unloaded(config, tasks, unloaded_cache_dir,
                            measure_real=lambda: _measure_unloaded(config, server, tasks, run_dir))
        recorder.unloaded(targets)
        launched = time.perf_counter()
        proc = start_server(server, log_path=run_dir / SERVER_LOG_FILE)
        try:
            metadata["server_startup_s"] = round(time.perf_counter() - launched, 3)
            recorder.metadata(metadata)
            clock = RealClock()
            hard_timeouts = {r.task_id: r.hard_timeout_s for r in targets}
            clock.run(_run(config, server, proc.pid, tasks, recorder, clock, hard_timeouts))
        finally:
            stop_server(proc)
    finally:
        recorder.close()


async def _run(config: dict[str, Any], server: ServerConfig, server_pid: int,
               tasks: list[AgentTaskSpec], recorder: Recorder, clock: RealClock,
               hard_timeouts: dict[int, float | None]) -> None:
    runtime_cfg = config["runtime"]
    timeout = httpx.Timeout(runtime_cfg.get("request_timeout_s", 600.0), connect=10.0)
    # no connection cap: the runtime's own queue, not the HTTP client, holds waiting requests
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client, \
            httpx.AsyncClient() as sampler_client:
        sampler = SystemSampler(
            clock, real_sources(base_url=server.base_url, server_pid=server_pid, client=sampler_client),
            recorder.system_sample, interval_s=runtime_cfg.get("sample_interval_s", 0.1))
        sampling = asyncio.create_task(sampler.run())
        try:
            proxy = Proxy(policy=admission_policy(config),
                          runtime=LlamaServerRuntime(server.base_url, client=client,
                                                     cache_prompt=server.cache_prompt),
                          clock=clock, recorder=recorder,
                          memory_budget_tokens=server.memory_budget, n_slots=server.slots,
                          hard_timeouts=hard_timeouts)
            await proxy.run(tasks)
        finally:
            await sampler.stop(sampling)


def _measure_unloaded(config: dict[str, Any], server: ServerConfig, tasks: list[AgentTaskSpec],
                      run_dir: Path) -> list[unloaded.UnloadedRecord]:
    proc = start_server(server, log_path=run_dir / UNLOADED_SERVER_LOG_FILE)
    try:
        timeout = config["runtime"].get("request_timeout_s", 600.0)
        return unloaded.measure(tasks, clock=RealClock(),
                                runtime=_ClientPerRequest(server, timeout),
                                memory_budget_tokens=server.memory_budget, n_slots=server.slots,
                                pin_slots=resident_cache_policy(config))
    finally:
        stop_server(proc)


class _ClientPerRequest:
    """The llama-server runtime with an HTTP client opened inside each submit, for
    callers that own the event loop (``unloaded.measure``); requests there are sequential."""

    def __init__(self, server: ServerConfig, timeout_s: float) -> None:
        self._server = server
        self._timeout = httpx.Timeout(timeout_s, connect=10.0)

    async def submit(self, request):
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            runtime = LlamaServerRuntime(self._server.base_url, client=client,
                                         cache_prompt=self._server.cache_prompt)
            return await runtime.submit(request)
