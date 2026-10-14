"""One experiment on the real pinned llama-server ([runtime] kind = "llama_server").

Launches the server from the config's [server] table, runs the same proxy, policy
and recorder as a simulated run on the real clock, samples the system every
``sample_interval_s``, and stops the server at the end. Besides the simulated
run-directory files it writes ``samples.jsonl``, ``metadata.json`` and ``server.log``.
The memory budget and slot count come from [server].

Every server launch is a restart: a fresh llama-server process, stopped before the
next one starts, so no cached state carries over between them.

1. **Cold start** (log: ``cold-start-server.log``): a server of the run's config is
   launched, timed from launch to healthy (model load), sent ``COLD_START_PROBES``
   first model requests, and stopped.
2. **Unloaded completion times** (log: ``unloaded-server.log``), unless cached:
   measured on a fresh server, warmed first by the same probes so the first agent
   task measured doesn't pay the cold start.
3. **The experiment** (log: ``server.log``) on a server of its own, timed from launch
   to healthy but sent nothing before the experiment's own model requests (its
   first ones fall in the warm-up).

``cold_start.json`` records every launch (``role``: ``cold_start``, ``unloaded``,
``experiment``) with its ``startup_s`` and ``first_requests``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from agentserve import config as cfg
from agentserve import llama_api, unloaded
from agentserve.clock import RealClock
from agentserve.experiment import _unloaded, admission_policy, workload_tasks
from agentserve.llama_runtime import LlamaServerRuntime
from agentserve.proxy import Proxy
from agentserve.recorder import SERVER_LOG_FILE, Recorder
from agentserve.run_metadata import collect_run_metadata
from agentserve.sampler import SystemSampler, real_sources
from agentserve.server import ServerConfig, start_server, stop_server
from agentserve.workload import AgentTaskSpec


UNLOADED_SERVER_LOG_FILE = "unloaded-server.log"
COLD_START_SERVER_LOG_FILE = "cold-start-server.log"
COLD_START_PROBES = 3
"""First model requests sent to a cold server: the first pays the cold start, the
last is warm. Distinct prompts, so none reuses another's context."""
PROBE_PROMPT_TOKENS = 512
PROBE_N_PREDICT = 32


def run_real_experiment(config: dict[str, Any], run_dir: Path, *,
                        unloaded_cache_dir: Path | None = None) -> None:
    server = ServerConfig(**config["server"])
    tasks = workload_tasks(config)
    recorder = Recorder(run_dir, config, record_prompts=cfg.record_prompts(config))
    try:
        metadata = collect_run_metadata(server)
        recorder.metadata(metadata)
        launches = _Launches(server, recorder)
        stop_server(launches.start("cold_start", run_dir / COLD_START_SERVER_LOG_FILE,
                                   probes=COLD_START_PROBES))
        targets = _unloaded(config, tasks, unloaded_cache_dir,
                            measure_real=lambda subset: _measure_unloaded(
                                config, server, subset, run_dir, launches))
        recorder.unloaded(targets)
        proc = launches.start("experiment", run_dir / SERVER_LOG_FILE, probes=0)
        try:
            metadata["server_startup_s"] = launches.last_startup_s
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
    timeout = httpx.Timeout(cfg.request_timeout_s(config), connect=10.0)
    # no connection cap: the runtime's own queue, not the HTTP client, holds waiting requests
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client, \
            httpx.AsyncClient() as sampler_client:
        sampler = SystemSampler(
            clock, real_sources(base_url=server.base_url, server_pid=server_pid, client=sampler_client),
            recorder.system_sample, interval_s=cfg.sample_interval_s(config))
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


class _Launches:
    """Starts the run's servers, timing each launch and probing it, into ``cold_start.json``."""

    def __init__(self, server: ServerConfig, recorder: Recorder) -> None:
        self._server = server
        self._recorder = recorder
        self._launches: list[dict[str, Any]] = []

    @property
    def last_startup_s(self) -> float:
        return self._launches[-1]["startup_s"]

    def start(self, role: str, log_path: Path, *, probes: int):
        launched = time.perf_counter()
        proc = start_server(self._server, log_path=log_path)
        launch = {"role": role, "pid": proc.pid,
                  "startup_s": round(time.perf_counter() - launched, 3), "first_requests": []}
        self._launches.append(launch)
        try:
            for i in range(probes):
                launch["first_requests"].append(_probe(self._server, salt=1_000_003 + i))
        except BaseException:
            stop_server(proc)
            raise
        finally:
            self._recorder.cold_start(self._launches)
        return proc


def _probe(server: ServerConfig, salt: int) -> dict[str, Any]:
    payload = llama_api.completion_payload(
        llama_api.synthetic_tokens(PROBE_PROMPT_TOKENS, salt=salt), PROBE_N_PREDICT,
        cache_prompt=server.cache_prompt)
    started = time.perf_counter()
    response = httpx.post(f"{server.base_url}/completion", json=payload, timeout=120.0)
    latency = time.perf_counter() - started
    result = llama_api.parse_completion(response.status_code, response.json(),
                                        n_predict=PROBE_N_PREDICT)
    if isinstance(result, llama_api.RuntimeError_):
        raise RuntimeError(f"cold-start probe failed: {result.kind}: {result.message}")
    return {"n_prompt_tokens": result.prompt_tokens, "n_predict": PROBE_N_PREDICT,
            "n_cached_tokens": result.cached_tokens, "latency_s": round(latency, 4),
            "prompt_s": result.prompt_ms / 1000.0, "decode_s": result.predicted_ms / 1000.0}


def _measure_unloaded(config: dict[str, Any], server: ServerConfig, tasks: list[AgentTaskSpec],
                      run_dir: Path, launches: _Launches) -> list[unloaded.UnloadedRecord]:
    proc = launches.start("unloaded", run_dir / UNLOADED_SERVER_LOG_FILE, probes=COLD_START_PROBES)
    try:
        return unloaded.measure(tasks, clock=RealClock(),
                                runtime=_ClientPerRequest(server, cfg.request_timeout_s(config)),
                                memory_budget_tokens=server.memory_budget, n_slots=server.slots,
                                pin_slots=cfg.cache_policy_pins(config))
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
