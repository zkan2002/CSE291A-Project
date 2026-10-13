"""Clock abstraction shared by the runner and the runtime.

A clock decides how an experiment's asyncio program is driven. Inside it, code
only ever uses ``clock.now()`` and ``await clock.sleep(seconds)``.

- ``RealClock`` runs on a normal event loop: wall time (``perf_counter``), for real experiments.
- ``VirtualClock`` runs on an event loop whose time jumps straight to the next
  scheduled timer whenever nothing is ready, so ten simulated minutes take
  milliseconds and are fully deterministic.
"""

from __future__ import annotations

import asyncio
import selectors
import time
from typing import Any, Coroutine, Protocol, TypeVar

T = TypeVar("T")


class Clock(Protocol):
    def now(self) -> float:
        """Seconds since the clock started running the experiment."""

    async def sleep(self, seconds: float) -> None: ...

    def run(self, main: Coroutine[Any, Any, T]) -> T:
        """Drive ``main`` to completion on this clock."""


class _LoopClock:
    _origin = 0.0

    def now(self) -> float:
        return asyncio.get_running_loop().time() - self._origin

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))

    def _start(self, main: Coroutine[Any, Any, T]) -> Coroutine[Any, Any, T]:
        async def wrapper() -> T:
            self._origin = asyncio.get_running_loop().time()
            return await main

        return wrapper()


class RealClock:
    """Wall time from ``time.perf_counter``: the event loop's own clock ticks in
    ~15.6 ms steps on Windows, too coarse to time model requests."""

    _origin = 0.0

    def now(self) -> float:
        return time.perf_counter() - self._origin

    async def sleep(self, seconds: float) -> None:
        # the loop's timers may fire up to one of its ticks early; never wake before the deadline
        deadline = self.now() + seconds
        await asyncio.sleep(max(0.0, seconds))
        while (remaining := deadline - self.now()) > 0:
            await asyncio.sleep(remaining)

    def run(self, main: Coroutine[Any, Any, T]) -> T:
        async def wrapper() -> T:
            self._origin = time.perf_counter()
            return await main

        return asyncio.run(wrapper())


class VirtualClock(_LoopClock):
    def run(self, main: Coroutine[Any, Any, T]) -> T:
        loop = _VirtualTimeLoop()
        try:
            return loop.run_until_complete(self._start(main))
        finally:
            loop.close()


class _SkippingSelector(selectors.DefaultSelector):
    """Never blocks: instead of waiting ``timeout`` seconds, advances virtual time."""

    def __init__(self) -> None:
        super().__init__()
        self.loop: _VirtualTimeLoop | None = None

    def select(self, timeout: float | None = None):
        events = super().select(0)
        if events:
            return events
        if timeout is None:
            raise RuntimeError("virtual clock deadlock: nothing ready and no timer scheduled")
        if timeout > 0 and self.loop is not None:
            self.loop.advance(timeout)
        return []


class _VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        selector = _SkippingSelector()
        self._virtual_now = 0.0
        super().__init__(selector)
        selector.loop = self

    def time(self) -> float:
        return self._virtual_now

    def advance(self, seconds: float) -> None:
        self._virtual_now += seconds
