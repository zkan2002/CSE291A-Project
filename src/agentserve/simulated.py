"""Simulated runtime: a stand-in for llama-server's shared KV pool, on a clock.

It models llama-server b11606 launched with ``--kv-unified -c POOL -np N`` as
spec #1's contract describes it, amended by the pilot
(docs/pilot/pilot-findings.md). The server is an iteration loop; so is this:

- Each iteration first starts queued model requests on idle slots, in arrival
  order. A request pinned (``id_slot``) to a busy slot waits for that slot. An
  unpinned one takes the idle slot holding the longest common prefix, if that
  prefix is strictly more than ``similarity`` of its prompt, else the least
  recently used idle slot (ties to the highest index).
- A prompt of ``POOL`` tokens or more is rejected with ``ExceedsContext`` only
  when it reaches a slot. A prompt that fits but leaves fewer than ``n_predict``
  cells stops early (the server's HTTP 200 ``truncated``) and is reported as
  ``ExceedsContext``, as the real adapter reports it.
- One batch per iteration: a decode token per generating slot, then prompt
  chunks (``n_batch`` tokens in all). If it doesn't fit in the free cells, idle
  slots holding tokens are purged one per retry, first in index order, KV
  discarded. If none is left, every running model request fails at once with
  ``PoolExhausted`` and its KV is cleared; queued ones are unaffected.
- A cancelled ``submit`` (the client disconnecting, as the proxy does at a hard
  timeout) leaves the queue, or gives up its slot at the end of the iteration
  in progress, its KV kept.
- An iteration takes ``prompt_token_s`` per prompt token in the batch, plus
  ``output_token_s`` if it samples any output token.

KV bookkeeping follows the server: after a model request with prompt ``P`` and
``n`` outputs, its slot holds ``P + n - 1`` tokens (the last sampled token is
never decoded). The next request on that slot reuses the longest common prefix.

Cache policies (pilot amendment 6) map to the server flags:

- ``cache_ram_tokens`` is ``--cache-ram`` in tokens (0 = off). With it on, an
  unpinned request that takes a slot by LRU, or by a prefix keeping less than
  half the slot's tokens, saves the slot's prompt to host RAM and restores the
  best-matching saved prompt into the slot. The ``id_slot`` path never does.
- ``cache_prompt = False`` is ``--no-cache-prompt`` (caching off): a slot's
  retained context is never reused, so every prompt is processed in full.
- ``cache_idle_slots`` (needs ``cache_ram_tokens``): whenever a request starts,
  every other idle slot's KV is moved to host RAM. Resident = off + pinned
  agent steps; offload = on + unpinned agent steps.

The server does not report KV occupancy (pilot amendment 7); this runtime
exposes its own as ``kv_tokens``, ``slot_kv_tokens`` and ``ram_cache_tokens``.
"""
from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import dataclass, field

from agentserve.clock import Clock
from agentserve.runtime import Completion, ExceedsContext, ModelRequest, PoolExhausted

_FIRST_OUTPUT_TOKEN = 1_000


@dataclass
class _Slot:
    index: int
    kv: list[int] = field(default_factory=list)
    req: _RunningRequest | None = None
    t_last_used: float = -1.0


@dataclass
class _RunningRequest:
    """A model request submitted to the runtime: queued, then running on a slot."""

    model_request: ModelRequest
    future: asyncio.Future[Completion]
    slot: _Slot | None = None
    n_cached: int = 0
    n_done: int = 0
    """Prompt tokens already in the slot's KV."""
    outputs: list[int] = field(default_factory=list)
    n_target: int = 0
    """Outputs it will generate: ``n_predict``, or fewer if the pool's context runs out."""
    cancelled: bool = False

    @property
    def prompt(self) -> tuple[int, ...]:
        return self.model_request.prompt_tokens

    @property
    def in_prompt(self) -> bool:
        return self.n_done < len(self.prompt)


class SimulatedRuntime:
    def __init__(self, clock: Clock, *, n_slots: int, pool_tokens: int,
                 prompt_token_s: float, output_token_s: float,
                 n_batch: int = 2048, similarity: float = 0.10,
                 cache_ram_tokens: int = 0, cache_idle_slots: bool = False,
                 cache_prompt: bool = True) -> None:
        self._clock = clock
        self._pool_tokens = pool_tokens
        self._prompt_token_s = prompt_token_s
        self._output_token_s = output_token_s
        self._n_batch = n_batch
        self._similarity = similarity
        self._cache_ram_tokens = cache_ram_tokens
        self._cache_idle_slots = cache_idle_slots and cache_ram_tokens > 0
        self._cache_prompt = cache_prompt
        self._ram: deque[tuple[int, ...]] = deque()
        self._slots = [_Slot(i) for i in range(n_slots)]
        self._queue: deque[_RunningRequest] = deque()
        self._engine: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._span: tuple[float, int, list[_RunningRequest]] | None = None
        """(start, iterations, model requests) of a decode span in progress: its KV cells are
        written when it ends, so occupancy adds the iterations started so far."""

    # --- observable KV state ------------------------------------------------

    @property
    def kv_tokens(self) -> int:
        """Tokens currently held in the shared KV pool, over all slots."""
        return sum(self.slot_kv_tokens)

    @property
    def slot_kv_tokens(self) -> tuple[int, ...]:
        """Tokens each slot currently holds in the KV pool, by slot index."""
        held = [len(s.kv) for s in self._slots]
        if self._span is not None:
            start, k, reqs = self._span
            started = min(k, math.floor((self._clock.now() - start) / self._output_token_s
                                        + 1e-9) + 1)
            for req in reqs:
                held[req.slot.index] += started
        return tuple(held)

    @property
    def ram_cache_tokens(self) -> int:
        """Tokens of saved prompts held in the host-RAM prompt cache (outside the pool)."""
        return sum(len(p) for p in self._ram)

    # --- runtime interface --------------------------------------------------

    async def submit(self, request: ModelRequest) -> Completion:
        req = _RunningRequest(request, asyncio.get_running_loop().create_future())
        self._queue.append(req)
        self._wake.set()
        if self._engine is None or self._engine.done():
            self._engine = asyncio.create_task(self._run())
        try:
            return await req.future
        except asyncio.CancelledError:
            self._cancel(req)
            raise

    def _cancel(self, req: _RunningRequest) -> None:
        """The client gave up on ``req`` (it disconnected): a queued one leaves the
        queue; a running one gives up its slot, KV kept, at the end of the
        iteration in progress, as the server releases a slot."""
        if req in self._queue:
            self._queue.remove(req)
        elif req.slot is not None and req.slot.req is req:
            req.cancelled = True
            self._wake.set()  # cut a decode span short

    # --- the server loop ----------------------------------------------------

    async def _run(self) -> None:
        while True:
            for slot in self._slots:
                if slot.req is not None and slot.req.cancelled:
                    slot.req = None
                    slot.t_last_used = self._clock.now()
            self._start_queued()
            running = [s.req for s in self._slots if s.req is not None]
            if not running:
                return
            await self._iterate(running)

    def _start_queued(self) -> None:
        for req in list(self._queue):
            slot, by_id = self._pick_slot(req.model_request)
            if slot is None:
                continue
            self._queue.remove(req)
            if len(req.prompt) >= self._pool_tokens:  # checked only once it reaches a slot
                req.future.set_exception(ExceedsContext(
                    f"request ({len(req.prompt)} tokens) exceeds the available context size "
                    f"({self._pool_tokens} tokens)"))
                continue
            if self._cache_ram_tokens and not by_id:
                self._swap_with_ram(slot, req.prompt)
            if self._cache_idle_slots:
                for other in self._slots:
                    if other is not slot and other.req is None and other.kv:
                        self._save(other)
            self._start(req, slot)

    def _pick_slot(self, request: ModelRequest) -> tuple[_Slot | None, bool]:
        """The slot for ``request`` (None if it must wait), and whether it was picked by id."""
        if request.slot is not None:
            slot = self._slots[request.slot]
            return (slot if slot.req is None else None), True
        idle = [s for s in self._slots if s.req is None]
        if not idle:
            return None, False
        best, best_sim = None, self._similarity
        for slot in idle:
            if slot.kv:
                sim = _common_prefix(slot.kv, request.prompt_tokens) / len(request.prompt_tokens)
                if sim > best_sim:
                    best, best_sim = slot, sim
        if best is not None:
            return best, False
        lru = idle[0]
        for slot in idle[1:]:
            if slot.t_last_used <= lru.t_last_used:  # ties go to the highest index
                lru = slot
        return lru, False

    # --- host-RAM prompt cache ------------------------------------------------

    def _swap_with_ram(self, slot: _Slot, prompt: tuple[int, ...]) -> None:
        """An unpinned request reusing a slot by LRU, or by a prefix that keeps less
        than half the slot's tokens, saves the slot's prompt to host RAM and loads
        the best-matching saved prompt instead (never on the id_slot path)."""
        lcp = _common_prefix(slot.kv, prompt)
        if slot.kv and lcp >= 0.5 * len(slot.kv):
            return
        if slot.kv:
            self._save(slot)
        best, best_keep, best_sim = None, 0.0, 0.0
        for saved in self._ram:
            n = _common_prefix(saved, prompt)
            keep, sim = n / len(saved), n / len(prompt)
            if keep > best_keep and sim > best_sim:
                best, best_keep, best_sim = saved, keep, sim
        if best is not None and len(best) <= self._pool_tokens - self.kv_tokens:
            self._ram.remove(best)
            slot.kv = list(best)

    def _save(self, slot: _Slot) -> None:
        """Move a slot's KV out of the pool into the host-RAM cache, oldest evicted first."""
        saved = tuple(slot.kv)
        slot.kv.clear()
        if len(saved) > self._cache_ram_tokens:
            return
        self._ram.append(saved)
        while self.ram_cache_tokens > self._cache_ram_tokens:
            self._ram.popleft()

    def _start(self, req: _RunningRequest, slot: _Slot) -> None:
        n_cached = (min(_common_prefix(slot.kv, req.prompt), len(req.prompt) - 1)
                    if self._cache_prompt else 0)
        del slot.kv[n_cached:]
        slot.req, req.slot = req, slot
        req.n_cached = req.n_done = n_cached
        req.n_target = min(req.model_request.n_predict, self._pool_tokens - len(req.prompt))

    async def _iterate(self, running: list[_RunningRequest]) -> None:
        decoding = [j for j in running if not j.in_prompt]
        budget = self._n_batch - len(decoding)
        chunks: list[tuple[_RunningRequest, int]] = []
        for req in sorted((j for j in running if j.in_prompt), key=lambda j: j.slot.index):
            n = min(len(req.prompt) - req.n_done, budget)
            if n > 0:
                chunks.append((req, n))
                budget -= n

        if not chunks and len(decoding) and await self._decode_span(decoding):
            return
        if not self._make_room(len(decoding) + sum(n for _, n in chunks)):
            self._fail_all(running)
            return
        for req in decoding:
            req.slot.kv.append(req.outputs[-1])
            req.outputs.append(self._next_output(req))
        samples = bool(decoding)
        for req, n in chunks:
            req.slot.kv.extend(req.prompt[req.n_done:req.n_done + n])
            req.n_done += n
            if not req.in_prompt:
                req.outputs.append(self._next_output(req))
                samples = True

        n_prompt = sum(n for _, n in chunks)
        await self._clock.sleep(n_prompt * self._prompt_token_s
                                + (self._output_token_s if samples else 0.0))
        for req in running:
            if len(req.outputs) >= req.n_target:
                self._finish(req)

    async def _decode_span(self, decoding: list[_RunningRequest]) -> bool:
        """Run consecutive decode-only iterations as one clock sleep, until the
        first model request finishes or the pool needs room. A ``submit`` cuts the
        span short at the end of the iteration in progress, so the new request
        starts exactly when it would have. Returns False if a span isn't worth it."""
        free = self._pool_tokens - self.kv_tokens
        k = min(min(j.n_target - len(j.outputs) for j in decoding), free // len(decoding))
        if k < 2:
            return False
        step = self._output_token_s
        start = self._clock.now()
        self._span = (start, k, decoding)
        self._wake.clear()
        sleeper = asyncio.ensure_future(self._clock.sleep(k * step))
        waker = asyncio.ensure_future(self._wake.wait())
        await asyncio.wait({sleeper, waker}, return_when=asyncio.FIRST_COMPLETED)
        n = k
        if not sleeper.done():
            sleeper.cancel()
            n = min(k, max(1, math.ceil((self._clock.now() - start) / step - 1e-9)))
            await self._clock.sleep(start + n * step - self._clock.now())
        waker.cancel()
        self._span = None
        for req in decoding:
            for _ in range(n):
                req.slot.kv.append(req.outputs[-1])
                req.outputs.append(self._next_output(req))
        for req in decoding:
            if len(req.outputs) >= req.n_target:
                self._finish(req)
        return True

    def _make_room(self, n_cells: int) -> bool:
        """Free ``n_cells`` KV cells by deleting idle slots' KV, one slot per retry,
        first in index order. The deleted KV is discarded, never saved."""
        while self._pool_tokens - self.kv_tokens < n_cells:
            victim = next((s for s in self._slots if s.req is None and s.kv), None)
            if victim is None:
                return False
            victim.kv.clear()
        return True

    def _fail_all(self, running: list[_RunningRequest]) -> None:
        for req in running:
            slot = req.slot
            slot.kv.clear()
            slot.req = None
            slot.t_last_used = self._clock.now()
            if not req.future.done():
                req.future.set_exception(PoolExhausted("Context size has been exceeded."))

    def _next_output(self, req: _RunningRequest) -> int:
        return _FIRST_OUTPUT_TOKEN + len(req.outputs)

    def _finish(self, req: _RunningRequest) -> None:
        slot = req.slot
        slot.req = None
        slot.t_last_used = self._clock.now()
        if req.future.done():  # cancelled
            return
        if len(req.outputs) < req.model_request.n_predict:
            # llama-server answers HTTP 200 with ``truncated: true``; the adapter
            # reports that as exceeds_context, and so do we
            req.future.set_exception(ExceedsContext(
                f"output truncated at {len(req.outputs)} of {req.model_request.n_predict} tokens: "
                f"prompt + n_predict exceeds the context size ({self._pool_tokens} tokens)"))
            return
        n_uncached = len(req.prompt) - req.n_cached
        req.future.set_result(Completion(
            output_tokens=tuple(req.outputs), n_prompt_tokens=len(req.prompt),
            n_cached_tokens=req.n_cached, slot=slot.index,
            prompt_s=n_uncached * self._prompt_token_s,
            decode_s=len(req.outputs) * self._output_token_s))


def _common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
