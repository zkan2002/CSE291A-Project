"""The simulated runtime's KV-pool contract, observed through the runtime interface.

Each test drives ``SimulatedRuntime.submit`` on a virtual clock and checks what
comes back (completions, typed errors, timings) plus the KV-token occupancy the
runtime exposes. The contract is spec #1's, as amended by the pilot
(docs/pilot/pilot-findings.md) against llama-server b11606.
"""

import asyncio

import pytest

from agentserve.clock import VirtualClock
from agentserve.runtime import ExceedsContext, ModelRequest, PoolExhausted
from agentserve.simulated import SimulatedRuntime

PROMPT_TOKEN_S = 0.001
OUTPUT_TOKEN_S = 0.01


def make_runtime(clock, *, n_slots=4, pool_tokens=2048, **kwargs) -> SimulatedRuntime:
    return SimulatedRuntime(clock, n_slots=n_slots, pool_tokens=pool_tokens,
                            prompt_token_s=PROMPT_TOKEN_S, output_token_s=OUTPUT_TOKEN_S,
                            **kwargs)


def tokens(n: int, start: int) -> tuple[int, ...]:
    """A prompt of ``n`` distinct token IDs; different ``start`` values never share a prefix."""
    return tuple(range(start, start + n))


def run(scenario, **runtime_kwargs):
    """Run ``scenario(clock, runtime)`` on a fresh virtual clock and runtime."""
    clock = VirtualClock()
    runtime = make_runtime(clock, **runtime_kwargs)
    return clock.run(scenario(clock, runtime))


async def outcome(runtime, request):
    """Submit and return the completion or the typed error, without raising."""
    try:
        return await runtime.submit(request)
    except (ExceedsContext, PoolExhausted) as e:
        return e


async def at(clock, t: float, coro):
    """Start ``coro`` at virtual time ``t``; return (result, finish time)."""
    await clock.sleep(t - clock.now())
    result = await coro
    return result, clock.now()


# --- retained context -------------------------------------------------------


def test_a_prefix_extending_request_on_its_pinned_slot_reuses_the_retained_context():
    a = tokens(600, 10_000)

    async def scenario(clock, runtime):
        first = await runtime.submit(ModelRequest(a, 32, slot=0))
        b = a + first.output_tokens + tokens(200, 50_000)
        second = await runtime.submit(ModelRequest(b, 32, slot=0))
        return first, second

    first, second = run(scenario)

    assert first.n_cached_tokens == 0
    assert second.slot == 0
    # the pilot's 600-token prompt + 32 outputs left 631 tokens in the slot
    # (the last sampled token is never written to the KV cache)
    assert second.n_cached_tokens == 631
    assert second.n_prompt_tokens == 832


# --- queueing -----------------------------------------------------------------


def test_requests_beyond_the_slot_count_wait_in_the_runtimes_queue():
    async def scenario(clock, runtime):
        requests = [ModelRequest(tokens(100, 10_000 * (i + 1)), 200) for i in range(6)]
        return await asyncio.gather(*(at(clock, 0.0, runtime.submit(r)) for r in requests))

    finishes = sorted(t for _, t in run(scenario, n_slots=4))

    # four run at once; the other two start when the first slots free up
    assert finishes[0] == pytest.approx(finishes[3])
    assert finishes[4] == pytest.approx(finishes[5])
    assert finishes[4] > finishes[3] + 199 * OUTPUT_TOKEN_S


def test_a_request_pinned_to_a_busy_slot_waits_for_it_even_when_other_slots_are_free():
    async def scenario(clock, runtime):
        return await asyncio.gather(
            at(clock, 0.0, runtime.submit(ModelRequest(tokens(100, 10_000), 100, slot=0))),
            at(clock, 0.1, runtime.submit(ModelRequest(tokens(100, 20_000), 100, slot=0))))

    (first, first_done), (second, second_done) = run(scenario, n_slots=4)

    assert first.slot == second.slot == 0
    assert second_done > first_done + 99 * OUTPUT_TOKEN_S


# --- slot selection -----------------------------------------------------------


def test_slot_selection_goes_by_id_slot_then_common_prefix_then_least_recently_used():
    # the pilot's prefix_extension probe, on a fresh runtime with every slot empty
    async def scenario(clock, runtime):
        a = await runtime.submit(ModelRequest(tokens(600, 10_000), 32, slot=0))
        b_prompt = tokens(600, 10_000) + a.output_tokens + tokens(200, 50_000)
        b = await runtime.submit(ModelRequest(b_prompt, 32, slot=0))
        c = await runtime.submit(ModelRequest(tokens(300, 70_000), 32))
        d = await runtime.submit(ModelRequest(b_prompt + b.output_tokens + tokens(200, 80_000), 32))
        e = await runtime.submit(ModelRequest(tokens(300, 90_000), 32))
        return a, b, c, d, e

    a, b, c, d, e = run(scenario, n_slots=4)

    assert [r.slot for r in (a, b, c, d, e)] == [0, 0, 3, 0, 2]
    assert d.n_cached_tokens == len(b.output_tokens) - 1 + 832


def test_prefix_similarity_must_be_strictly_above_one_tenth_of_the_new_prompt():
    held = tokens(100, 10_000)

    async def scenario(clock, runtime):
        await runtime.submit(ModelRequest(held, 1, slot=0))
        at_threshold = await runtime.submit(ModelRequest(held[:10] + tokens(90, 50_000), 1))
        above = await runtime.submit(ModelRequest(held[:11] + tokens(89, 60_000), 1))
        return at_threshold, above

    at_threshold, above = run(scenario, n_slots=4)

    assert (at_threshold.slot, at_threshold.n_cached_tokens) == (3, 0)
    assert (above.slot, above.n_cached_tokens) == (0, 11)


def test_least_recently_used_ties_go_to_the_highest_slot_index():
    async def scenario(clock, runtime):
        first = await runtime.submit(ModelRequest(tokens(50, 10_000), 1))
        second = await runtime.submit(ModelRequest(tokens(50, 20_000), 1))
        return first, second

    first, second = run(scenario, n_slots=4)

    assert (first.slot, second.slot) == (3, 2)


# --- context limits -----------------------------------------------------------


@pytest.mark.parametrize("n_prompt", [2048, 2100])
def test_a_prompt_that_fills_the_whole_pool_is_rejected(n_prompt):
    async def scenario(clock, runtime):
        return await outcome(runtime, ModelRequest(tokens(n_prompt, 10_000), 10))

    assert isinstance(run(scenario, pool_tokens=2048), ExceedsContext)


def test_an_oversize_prompt_is_rejected_only_once_it_reaches_a_slot():
    async def scenario(clock, runtime):
        busy = [at(clock, 0.0, runtime.submit(ModelRequest(tokens(100, 10_000 * (i + 1)), 200)))
                for i in range(4)]
        oversize = at(clock, 0.5, outcome(runtime, ModelRequest(tokens(2100, 90_000), 10)))
        return await asyncio.gather(*busy, oversize)

    *busy, (error, rejected_at) = run(scenario, n_slots=4, pool_tokens=2048)

    assert isinstance(error, ExceedsContext)
    assert rejected_at == pytest.approx(min(t for _, t in busy))


def test_a_prompt_that_fits_but_leaves_too_little_room_for_its_outputs_fails_as_truncated():
    # the server returns HTTP 200 with fewer outputs (truncated); the adapter
    # reports that as exceeds_context, and so does the simulated runtime
    async def scenario(clock, runtime):
        started = clock.now()
        error = await outcome(runtime, ModelRequest(tokens(1900, 10_000), 300, slot=0))
        return error, clock.now() - started, runtime.slot_kv_tokens[0]

    error, elapsed, kv = run(scenario, pool_tokens=2048)

    assert isinstance(error, ExceedsContext)
    # it generated 2048 - 1900 = 148 tokens before stopping, and keeps them in its slot
    assert elapsed == pytest.approx(1900 * PROMPT_TOKEN_S + 148 * OUTPUT_TOKEN_S)
    assert kv == 1900 + 148 - 1


# --- a full pool ----------------------------------------------------------------


def test_a_full_pool_deletes_the_first_idle_slot_in_index_order_not_the_least_recently_used():
    # the pilot's idle_slot_deletion probe: fill slots 2, 1, 0 (slot 2 is the
    # least recently used), then send slot 3 more than the free space
    prompts = {s: tokens(500, 10_000 * (s + 1)) for s in range(4)}
    prompts[3] = tokens(700, 40_000)

    async def scenario(clock, runtime):
        for s in (2, 1, 0):
            await runtime.submit(ModelRequest(prompts[s], 8, slot=s))
        before = runtime.slot_kv_tokens
        await runtime.submit(ModelRequest(prompts[3], 8, slot=3))
        after = runtime.slot_kv_tokens
        resent = [await runtime.submit(ModelRequest(prompts[s], 1, slot=s)) for s in (2, 1, 0)]
        return before, after, [r.n_cached_tokens for r in resent]

    before, after, cached = run(scenario, n_slots=4, pool_tokens=2048)

    assert before == (507, 507, 507, 0)
    assert after == (0, 507, 507, 707)  # one slot deleted, and only one
    assert cached == [499, 499, 0]


def test_idle_slots_are_also_deleted_when_the_pool_fills_during_generation():
    async def scenario(clock, runtime):
        await runtime.submit(ModelRequest(tokens(1000, 10_000), 1, slot=0))
        done = await runtime.submit(ModelRequest(tokens(1000, 20_000), 100, slot=1))
        return done, runtime.slot_kv_tokens

    done, kv = run(scenario, n_slots=2, pool_tokens=2048)

    assert len(done.output_tokens) == 100
    assert kv == (0, 1099)


def test_overflow_with_no_idle_slot_fails_every_running_request_at_once_and_clears_its_kv():
    # the pilot's overflow_abort probe: 4 x (400 + 400) tokens against 2048
    async def scenario(clock, runtime):
        running = [at(clock, 0.0, outcome(runtime, ModelRequest(tokens(400, 10_000 * (s + 1)), 400,
                                                                slot=s)))
                   for s in range(4)]
        queued = at(clock, 0.2, outcome(runtime, ModelRequest(tokens(100, 90_000), 16)))
        results = await asyncio.gather(*running, queued)
        return results, runtime.slot_kv_tokens

    (*failed, (late, late_done)), kv = run(scenario, n_slots=4, pool_tokens=2048)

    assert all(isinstance(error, PoolExhausted) for error, _ in failed)
    assert len({t for _, t in failed}) == 1  # one event
    # each had generated about (2048 - 1600) / 4 = 112 tokens
    assert failed[0][1] == pytest.approx(1600 * PROMPT_TOKEN_S + 113 * OUTPUT_TOKEN_S)
    # the queued request is unaffected: it runs afterwards on a cleared pool
    assert len(late.output_tokens) == 16
    assert late.n_cached_tokens == 0
    assert kv == (0, 0, 0, 100 + 15)
