"""Simulated runtime: a stand-in for llama-server that runs on a clock.

For now it only enforces the slot count: at most ``n_slots`` model requests run
at once, and the rest wait FIFO in the runtime's own queue. A running request
takes ``prompt_token_s`` per prompt token plus ``output_token_s`` per output
token. The KV-pool behaviour of the pinned llama-server lands in later tickets.
"""

from __future__ import annotations

import asyncio
from collections import deque

from agentserve.clock import Clock
from agentserve.runtime import Completion, ModelRequest

_FIRST_OUTPUT_TOKEN = 1_000


class SimulatedRuntime:
    def __init__(self, clock: Clock, *, n_slots: int, prompt_token_s: float,
                 output_token_s: float) -> None:
        self._clock = clock
        self._prompt_token_s = prompt_token_s
        self._output_token_s = output_token_s
        self._free_slots = set(range(n_slots))
        self._waiting: deque[asyncio.Future[int]] = deque()

    async def submit(self, request: ModelRequest) -> Completion:
        slot = await self._acquire_slot(request.slot)
        try:
            n_prompt = len(request.prompt_tokens)
            prompt_s = n_prompt * self._prompt_token_s
            decode_s = request.n_predict * self._output_token_s
            await self._clock.sleep(prompt_s + decode_s)
            output = tuple(range(_FIRST_OUTPUT_TOKEN, _FIRST_OUTPUT_TOKEN + request.n_predict))
            return Completion(output_tokens=output, n_prompt_tokens=n_prompt,
                              n_cached_tokens=0, slot=slot,
                              prompt_s=prompt_s, decode_s=decode_s)
        finally:
            self._release_slot(slot)

    async def _acquire_slot(self, wanted: int | None) -> int:
        if not self._free_slots:
            waiter: asyncio.Future[int] = asyncio.get_running_loop().create_future()
            self._waiting.append(waiter)
            return await waiter  # the releasing request hands its slot over
        slot = wanted if wanted in self._free_slots else min(self._free_slots)
        self._free_slots.remove(slot)
        return slot

    def _release_slot(self, slot: int) -> None:
        if self._waiting:
            self._waiting.popleft().set_result(slot)
        else:
            self._free_slots.add(slot)
