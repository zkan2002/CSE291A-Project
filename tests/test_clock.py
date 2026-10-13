"""The real clock times real runs, so it must resolve well below a millisecond.

(On Windows, the event loop's own time() ticks in ~15.6 ms steps.)
"""

import time

from agentserve.clock import RealClock


def test_real_clock_resolves_sub_millisecond_intervals():
    clock = RealClock()

    async def main():
        readings = []
        end = time.perf_counter() + 0.05
        while time.perf_counter() < end:
            readings.append(clock.now())
        return readings

    readings = clock.run(main())

    assert readings[0] < 0.01
    steps = sorted({b - a for a, b in zip(readings, readings[1:]) if b > a})
    assert steps and steps[0] < 0.001
    assert len(set(readings)) > 50


def test_real_clock_sleep_advances_now():
    clock = RealClock()

    async def main():
        before = clock.now()
        await clock.sleep(0.05)
        return clock.now() - before

    assert 0.04 < clock.run(main()) < 0.2
