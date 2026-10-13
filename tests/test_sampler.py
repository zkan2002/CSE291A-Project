"""The system sampler polls its sources on the run's clock and writes one sample per tick.

Run on the virtual clock with stand-in sources, so the timing is exact.
"""

import asyncio

import pytest

from agentserve.clock import VirtualClock
from agentserve.sampler import SystemSampler


def run_sampler_for(seconds: float, sources: dict, interval_s: float = 0.1) -> list[dict]:
    clock = VirtualClock()
    samples: list[dict] = []

    async def main():
        sampler = SystemSampler(clock, sources, samples.append, interval_s=interval_s)
        task = asyncio.create_task(sampler.run())
        await clock.sleep(seconds)
        await sampler.stop(task)

    clock.run(main())
    return samples


def test_samples_every_interval_on_the_run_clock():
    vram = iter(range(100, 10_000))

    async def read_vram():
        return next(vram)

    samples = run_sampler_for(1.0, {"vram_used_bytes": read_vram})

    times = [s["t_s"] for s in samples]
    assert times[:3] == pytest.approx([0.0, 0.1, 0.2])
    assert len(samples) in (10, 11)  # ticks at 0.0 .. 0.9 (and 1.0, depending on ordering)
    assert [s["vram_used_bytes"] for s in samples[:3]] == [100, 101, 102]


def test_a_failing_source_records_null_and_sampling_continues():
    async def broken():
        raise OSError("process gone")

    async def fine():
        return 1

    samples = run_sampler_for(0.35, {"server_rss_bytes": broken, "vram_used_bytes": fine})

    assert len(samples) == 4
    assert all(s["server_rss_bytes"] is None and s["vram_used_bytes"] == 1 for s in samples)


def test_slow_sources_do_not_make_samples_drift():
    async def slow():
        await asyncio.sleep(0.03)
        return 0

    samples = run_sampler_for(1.0, {"slots": slow})

    # each tick is scheduled from the start time, not from the end of the previous sample
    assert [round(s["t_s"], 6) for s in samples[:4]] == [0.0, 0.1, 0.2, 0.3]
