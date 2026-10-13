"""Clock abstraction shared by the runner and the runtime.

A clock decides how an experiment's asyncio program is driven. Inside it, code
only ever uses ``clock.now()`` and ``await clock.sleep(seconds)``.

- ``RealClock`` runs on a normal event loop: wall time, for real experiments.
- ``VirtualClock`` runs on an event loop whose time jumps straight to the next
  scheduled timer whenever nothing is ready, so ten simulated minutes take
  milliseconds and are fully deterministic.
"""

from __future__ import annotations

import asyncio
import selectors
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


class RealClock(_LoopClock):
    def run(self, main: Coroutine[Any, Any, T]) -> T:
        return asyncio.run(self._start(main))


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
        # asyncio fires every timer due within the clock's resolution early; the
        # OS's is 15.6 ms on Windows, so virtual time uses 1 ns instead (it must
        # stay > 0: a timer fires only when strictly before now + resolution).
        self._clock_resolution = 1e-9

    def time(self) -> float:
        return self._virtual_now

    def advance(self, seconds: float) -> None:
        # land exactly on the next timer: ``now + (when - now)`` can fall short of
        # ``when`` in floating point, which with no clock resolution never fires it
        target = self._virtual_now + seconds
        if self._scheduled:
            target = max(target, self._scheduled[0].when())
        self._virtual_now = target
