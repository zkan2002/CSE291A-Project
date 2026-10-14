"""System sampler: every ``interval_s`` on the run's clock, read each source and write one sample.

A sample is ``{"t_s": <run-clock time of the tick>, <source name>: <value>, ...}``.
Sources are read concurrently; one that raises records ``null`` for that tick.
Ticks are scheduled from the start time, so slow sources don't make samples drift
(a tick that is already late is taken at once).

``real_sources`` builds the sources for a real run: device VRAM in use (NVML), the
server process's resident host RAM, and the server's ``/slots`` view.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Awaitable, Callable, Mapping

import httpx

from agentserve.clock import Clock

Source = Callable[[], Awaitable[Any]]


class SystemSampler:
    def __init__(self, clock: Clock, sources: Mapping[str, Source],
                 write: Callable[[dict[str, Any]], None], *, interval_s: float) -> None:
        self._clock = clock
        self._sources = dict(sources)
        self._write = write
        self._interval_s = interval_s

    async def run(self) -> None:
        """Sample until cancelled."""
        start = self._clock.now()
        tick = 0
        while True:
            t = self._clock.now()
            values = await asyncio.gather(*(_read(s) for s in self._sources.values()))
            self._write({"t_s": t, **dict(zip(self._sources, values))})
            # the next tick that is not already past (late ticks are skipped, not bunched)
            elapsed_ticks = (self._clock.now() - start) / self._interval_s
            tick = max(tick + 1, math.ceil(elapsed_ticks - 1e-9))
            await self._clock.sleep(start + tick * self._interval_s - self._clock.now())

    @staticmethod
    async def stop(task: asyncio.Task) -> None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _read(source: Source) -> Any:
    try:
        return await source()
    except Exception:
        return None


def real_sources(*, base_url: str, server_pid: int, client: httpx.AsyncClient,
                 gpu_index: int = 0) -> dict[str, Source]:
    import psutil
    import pynvml

    process = psutil.Process(server_pid)
    try:
        pynvml.nvmlInit()
        gpu = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except pynvml.NVMLError:
        gpu = None

    async def vram_used_bytes() -> int | None:
        return None if gpu is None else int(pynvml.nvmlDeviceGetMemoryInfo(gpu).used)

    async def server_rss_bytes() -> int:
        return int(process.memory_info().rss)

    async def slots() -> list[dict[str, Any]]:
        response = await client.get(f"{base_url}/slots", timeout=1.0)
        return [{"id": s.get("id"), "is_processing": s.get("is_processing"),
                 "n_prompt_tokens": s.get("n_prompt_tokens")} for s in response.json()]

    return {"vram_used_bytes": vram_used_bytes, "server_rss_bytes": server_rss_bytes,
            "slots": slots}
