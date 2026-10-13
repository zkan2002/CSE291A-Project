"""Simulated runtime: a stand-in for llama-server's shared KV pool, on a clock.

It models llama-server b11606 launched with ``--kv-unified -c POOL -np N`` as
spec #1's contract describes it, amended by the pilot
(docs/pilot/pilot-findings.md). The server is an iteration loop; so is this:

- Each iteration first starts queued model requests on slots, then builds one
  batch: one decode token per generating slot, then prompt chunks (up to
  ``n_batch`` tokens in all) for slots still processing their prompt.
- The batch must fit in the pool's free KV cells. If it doesn't, idle slots that
  hold tokens are purged one per retry, first in index order. If no such slot is
  left, every running model request fails with ``PoolExhausted`` and its KV is
  cleared; queued ones are unaffected.
- An iteration takes ``prompt_token_s`` per prompt token in the batch, plus
  ``output_token_s`` if it samples any output token.

KV bookkeeping follows the server: after a model request with prompt ``P`` and
``n`` outputs, its slot holds ``P + n - 1`` tokens (the last sampled token is
never decoded). The next request on that slot reuses the longest common prefix.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

from agentserve.clock import Clock
from agentserve.runtime import Completion, ModelRequest

_FIRST_OUTPUT_TOKEN = 1_000


@dataclass
class _Slot:
    index: int
    kv: list[int] = field(default_factory=list)
    job: _Job | None = None
    t_last_used: float = -1.0


@dataclass
class _Job:
    request: ModelRequest
    future: asyncio.Future[Completion]
    slot: _Slot | None = None
    n_cached: int = 0
    n_done: int = 0
    """Prompt tokens already in the slot's KV."""
    outputs: list[int] = field(default_factory=list)

    @property
    def prompt(self) -> tuple[int, ...]:
        return self.request.prompt_tokens

    @property
    def in_prompt(self) -> bool:
        return self.n_done < len(self.prompt)


class SimulatedRuntime:
    def __init__(self, clock: Clock, *, n_slots: int, pool_tokens: int,
                 prompt_token_s: float, output_token_s: float,
                 n_batch: int = 2048, similarity: float = 0.10) -> None:
        self._clock = clock
        self._pool_tokens = pool_tokens
        self._prompt_token_s = prompt_token_s
        self._output_token_s = output_token_s
        self._n_batch = n_batch
        self._similarity = similarity
        self._slots = [_Slot(i) for i in range(n_slots)]
        self._queue: deque[_Job] = deque()
        self._engine: asyncio.Task[None] | None = None

    # --- observable KV state ------------------------------------------------

    @property
    def kv_tokens(self) -> int:
        """Tokens currently held in the shared KV pool, over all slots."""
        return sum(len(s.kv) for s in self._slots)

    @property
    def slot_kv_tokens(self) -> tuple[int, ...]:
        """Tokens each slot currently holds in the KV pool, by slot index."""
        return tuple(len(s.kv) for s in self._slots)

    # --- runtime interface --------------------------------------------------

    async def submit(self, request: ModelRequest) -> Completion:
        job = _Job(request, asyncio.get_running_loop().create_future())
        self._queue.append(job)
        if self._engine is None or self._engine.done():
            self._engine = asyncio.create_task(self._run())
        return await job.future

    # --- the server loop ----------------------------------------------------

    async def _run(self) -> None:
        while True:
            self._start_queued()
            running = [s.job for s in self._slots if s.job is not None]
            if not running:
                return
            await self._iterate(running)

    def _start_queued(self) -> None:
        for job in list(self._queue):
            slot = self._pick_slot(job.request)
            if slot is None:
                continue
            self._queue.remove(job)
            self._start(job, slot)

    def _pick_slot(self, request: ModelRequest) -> _Slot | None:
        if request.slot is not None:
            slot = self._slots[request.slot]
            return slot if slot.job is None else None
        idle = [s for s in self._slots if s.job is None]
        if not idle:
            return None
        best, best_sim = None, self._similarity
        for slot in idle:
            if slot.kv:
                sim = _common_prefix(slot.kv, request.prompt_tokens) / len(request.prompt_tokens)
                if sim > best_sim:
                    best, best_sim = slot, sim
        if best is not None:
            return best
        lru = idle[0]
        for slot in idle[1:]:
            if slot.t_last_used <= lru.t_last_used:  # ties go to the highest index
                lru = slot
        return lru

    def _start(self, job: _Job, slot: _Slot) -> None:
        n_cached = min(_common_prefix(slot.kv, job.prompt), len(job.prompt) - 1)
        del slot.kv[n_cached:]
        slot.job, job.slot = job, slot
        job.n_cached = job.n_done = n_cached

    async def _iterate(self, running: list[_Job]) -> None:
        decoding = [j for j in running if not j.in_prompt]
        budget = self._n_batch - len(decoding)
        chunks: list[tuple[_Job, int]] = []
        for job in sorted((j for j in running if j.in_prompt), key=lambda j: j.slot.index):
            n = min(len(job.prompt) - job.n_done, budget)
            if n > 0:
                chunks.append((job, n))
                budget -= n

        for job in decoding:
            job.slot.kv.append(job.outputs[-1])
            job.outputs.append(self._next_output(job))
        samples = bool(decoding)
        for job, n in chunks:
            job.slot.kv.extend(job.prompt[job.n_done:job.n_done + n])
            job.n_done += n
            if not job.in_prompt:
                job.outputs.append(self._next_output(job))
                samples = True

        n_prompt = sum(n for _, n in chunks)
        await self._clock.sleep(n_prompt * self._prompt_token_s
                                + (self._output_token_s if samples else 0.0))
        for job in running:
            if len(job.outputs) >= job.request.n_predict:
                self._finish(job)

    def _next_output(self, job: _Job) -> int:
        return _FIRST_OUTPUT_TOKEN + len(job.outputs)

    def _finish(self, job: _Job) -> None:
        slot = job.slot
        slot.job = None
        slot.t_last_used = self._clock.now()
        n_uncached = len(job.prompt) - job.n_cached
        job.future.set_result(Completion(
            output_tokens=tuple(job.outputs), n_prompt_tokens=len(job.prompt),
            n_cached_tokens=job.n_cached, slot=slot.index,
            prompt_s=n_uncached * self._prompt_token_s,
            decode_s=len(job.outputs) * self._output_token_s))


def _common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
