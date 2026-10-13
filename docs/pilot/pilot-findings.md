# Pilot findings: KV-pool overflow on the pinned llama-server

Ticket #2, parent spec #1. Run on 2026-10-13 against llama.cpp **b11606** (commit `1b70350c9`),
Llama-3.2-3B-Instruct Q4_K_M, RTX 3060 Ti. Pins: [`configs/pins.toml`](../../configs/pins.toml).

**Bottom line:** the overflow abort does **not** crash the server. Every running sequence gets
**HTTP 500** with body
`{"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}`,
its KV is cleared, requests still queued behind them run normally, and `/health` stays `ok`.
Failed tasks therefore don't need server restarts in the baselines.

## Setup

- Server: `python -m uv run python -m agentserve.server configs/pilot.toml`, which runs
  `llama-server -m … --kv-unified -c 2048 -np 4 -ngl 99 --fit off --metrics --cache-ram 0 --no-cache-idle-slots`.
  The memory budget is 2048 tokens shared by 4 slots.
- Probe: `python -m uv run python -m agentserve.pilot configs/pilot.toml --out results/pilot`.
  Each probe starts a fresh server and sends `/completion` requests as token arrays with
  `n_predict`, `ignore_eos`, `id_slot`, `cache_prompt` and `return_tokens`. It records every response,
  the relevant server log lines, `/slots`, `/metrics`, and whether the server is still alive.
  The same probes run as assertions with `pytest -m real_runtime`
  ([`tests/test_pilot_real_runtime.py`](../../tests/test_pilot_real_runtime.py)).
- Evidence from the run quoted below: [`evidence/pilot-report.json`](evidence/pilot-report.json)
  plus one `evidence/<probe>.server.log` per probe.

### Release vs. source-reading commit

The spec's contract was read at `ff026f8`. That commit is the **direct child** of `b11606`'s
commit `1b70350c9`. Its diff, "chat: replace output string with parts" (#30389), touches only
chat parsing (`common/chat*`, `common/parsers/*`, and chat output in `tools/server/server-common.cpp`).
It doesn't change slot selection, the decode loop or the prompt cache. I re-read
`tools/server/server-context.cpp` at `b11606`, and the relevant code is the same as in the
contract: `get_available_slot`, `try_clear_idle_slots`, `decode()`, and the
`cache_idle_slots` block in `process_single_task`.

### Important launch facts at b11606

- **With `--kv-unified`, each slot's context equals the whole pool** (`n_ctx_slot = 2048`, as the
  startup log and `/slots` show). One request can use the entire memory budget. A new
  `--kv-unified-per-slot N` flag exists to cap this; we don't use it.
- **The release defaults are not "caching off".** `--cache-ram` defaults to 8192 MiB, and
  `--cache-idle-slots` defaults to **on**. `--cache-idle-slots` is silently disabled when
  `--cache-ram 0`. The launcher therefore always passes both flags explicitly.
- `/slots` no longer exposes slot token contents. For each slot it shows only `n_ctx` and
  `is_processing`, plus `id_task` and `n_prompt_tokens` of the slot's last task. `/metrics` has no
  KV-usage gauge. Its useful fields are `requests_processing`, `requests_deferred`,
  `n_tokens_max` and `prompt_tokens_cached_total`. **KV occupancy cannot be read from the
  server.** The proxy has to track it from the footprints of the requests it sent.

## Contract clauses

| # | Clause (spec #1, "Simulated runtime contract") | Verdict |
| --- | --- | --- |
| 1 | Requests beyond `-np` running sequences wait in the runtime's own queue | **Confirmed** (+ addition) |
| 2 | Slot selection: `id_slot`, then LCP above threshold, then LRU | **Confirmed** (+ details) |
| 3 | A prompt alone exceeding a slot's context is rejected immediately | **Corrected** |
| 4 | Pool full: KV of the first idle slot in index order is deleted; none idle: every running sequence fails, KV cleared | **Confirmed** (+ details) |
| 5 | `--cache-idle-slots`: idle slots offloaded from the pool whenever a new request starts | **Confirmed** (+ correction on reuse) |

### 1. Queueing beyond `-np`: confirmed

Probe `slot_queueing` sent 6 concurrent requests (100-token prompt, `n_predict` 200, no `id_slot`)
to 4 slots. Mid-flight, `/metrics` showed `requests_processing 4`, `requests_deferred 2`. Finish
times were `[2.02, 2.02, 2.06, 2.06, 3.61, 3.61]` s, so two requests waited for a free slot.

**Addition:** a request pinned with `id_slot` to a **busy** slot waits for that slot. It is not
moved to a free one. Two concurrent requests pinned to slot 0 both ran on slot 0, finishing at
5.03 s and 6.47 s. The source does the same thing: a busy `id_slot` slot is returned, and the task
is deferred.

### 2. Slot selection order: confirmed, with details

Probe `prefix_extension`, run on a fresh server with every slot empty:

| Request | id_slot | Chosen slot | Server log |
| --- | --- | --- | --- |
| A: 600-token prompt | 0 | 0 | `selected slot by id (0)` |
| B: A + A's 32 output tokens + 200-token tool result | 0 | 0 | `selected slot by id (0)` |
| C: unrelated 300 tokens | -1 | **3** | `selected slot by LRU, t_last = -1` |
| D: B + B's output + tool result | -1 | 0 | `selected slot by LCP similarity, f_sim_best = … (> 0.100 thold)` |
| E: unrelated 300 tokens | -1 | **2** | `selected slot by LRU` |

Details the simulated runtime must match:
- The similarity threshold defaults to 0.10, and the comparison is strict:
  `f_sim = lcp / len(new prompt) > 0.10`. Only idle slots that hold tokens count.
- **LRU ties go to the highest index.** The comparison is `t_last_used <= t_last`, so on a fresh
  server the first unpinned request lands on slot `np-1`, not slot 0.
- When the host-RAM prompt cache is on (`--cache-ram > 0`), LRU selection, or LCP selection that
  would keep less than half of the slot's tokens, also saves the slot's old prompt to the RAM cache.
  It then **loads the best-matching cached prompt into that slot**. This means an unpinned
  request can get cached tokens on a different slot than the one its prefix was computed on
  (see clause 5). This path is not in the contract.

### 3. Oversize prompt: corrected

Probe `oversize_prompt`:
- A 2100-token prompt on an idle server got **HTTP 400** in 16 ms with this body:
  `{"error":{"code":400,"message":"request (2100 tokens) exceeds the available context size (2048 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":2100,"n_ctx":2048}}`
- A 2048-token prompt is also rejected. **The limit is `prompt >= n_ctx`, not `>`.**
- **The rejection is not immediate.** The check runs only when the request reaches a slot. With
  all 4 slots busy, the 2100-token prompt was sent at 1.64 s and rejected at 4.42 s, the moment the
  first busy slot finished (the busy requests finished at 4.42–4.44 s). It waits in the runtime
  queue like any other request.
- Here "a slot's context" means the **whole pool** (`-c`), because of `--kv-unified`.
- **New, not in the contract:** if the prompt fits but prompt + `n_predict` does not, the request
  **succeeds with fewer output tokens**. A 1900-token prompt with `n_predict 300` returned HTTP 200
  with 148 output tokens (= 2048 − 1900), `stop_type "limit"` and `truncated: true`. Along the way it
  purged another idle slot (`purging slot 1 with 2047 tokens`). The adapter must treat
  `output_tokens < n_predict` (or `truncated`) as an error. Otherwise fixed-length outputs are
  silently broken.

### 4. Full pool: confirmed, with details

**Idle-slot deletion** (probe `idle_slot_deletion`, cache-ram 0). Slots were filled in the order
2, 1, 0, with 500 + 8 tokens each. This makes slot 2 the least recently used and slot 0 the
most recent. Then slot 3 was sent 700 + 8 tokens, more than the free space:

```
W srv  try_clear_id: purging slot 0 with 507 tokens
W srv        decode: failed to find free space in the KV cache, retrying with smaller batch size, off = 0, n_batch = 2048, ret = 1
```

The slot-3 request succeeded. Re-sending each slot's original prompt gave cached tokens of
`{slot 2: 499, slot 1: 499, slot 0: 0}`. **The slot deleted is the first idle slot in index order
(0), not the LRU one (2).** Details:
- Only one slot is purged per failed attempt, and then the batch is retried at the same size. The
  batch size is halved only when no idle slot with tokens is left.
- This happens during **prompt processing** as well as token generation (here it was a 700-token
  prompt batch). The contract says "during decode". In the server, "decode" covers both.
- The purge discards the KV. It does not save it to the RAM cache, even when `--cache-ram > 0`.

**Overflow abort** (probe `overflow_abort`). 4 requests with 400-token prompts and `n_predict 400`
were pinned to slots 0–3, for a total of 3200 tokens against a 2048-token pool. A 5th request
(100 + 16) was sent 0.2 s later, while all 4 slots were busy. The log shows the batch shrinking from
2048 down to 1 (`n_batch = 1024 … 2, 1`, at offsets 0, 2 and 3 of the 4-token decode batch), then:

```
E srv        decode: Context size has been exceeded. off = 3, n_batch = 1, ret = 1
E srv    send_error: task id = 0, error: Context size has been exceeded.
E srv    send_error: task id = 1, error: Context size has been exceeded.
E srv    send_error: task id = 2, error: Context size has been exceeded.
E srv    send_error: task id = 4, error: Context size has been exceeded.
E srv  update_slots: decode() failed: Context size has been exceeded.
```

- **What clients receive:** each of the 4 running requests got **HTTP 500**
  `{"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}`
  after 1.5 s. The error is not streamed partially, and no tokens are returned. `/metrics`
  `n_tokens_max 513` shows that each sequence had generated about 112 of its 400 tokens
  ((2048 − 1600) / 4).
- **The server does not crash.** `decode()` throws, but `update_slots` catches the exception. The
  process keeps running and `/health` returns `{"status":"ok"}`. The queued 5th request was not
  failed: it started afterwards and returned 200 with 16 tokens. A fresh follow-up request also
  succeeded. The failed slots' KV was cleared (the follow-up got 0 cached tokens and was placed by LRU).
- Error kinds for the adapter: 500 with this message means `pool_exhausted`, and 400 with
  `exceed_context_size_error` means `exceeds_context`
  ([`src/agentserve/llama_api.py`](../../src/agentserve/llama_api.py)).
- A small detail: when the batch is shrunk, part of the batch may already have been decoded before
  the abort (offsets 0–2 succeeded here). This doesn't matter, because every running sequence is
  failed anyway.

### 5. `--cache-idle-slots`: offload confirmed; the reuse it is meant to enable doesn't work for pinned requests

Probes `cache_idle_slots_*` (`--cache-ram 512`, run with `-lv 4` so prompt-cache activity is logged).
Agent pattern: A (600 + 32) on slot 0, then an unrelated X on slot 1, then B (A + output + 200-token
tool result).

- **The offload happens as the contract says.** When X starts on slot 1, every idle slot is saved to
  the RAM cache and cleared from the pool:
  ```
  I slot process_sing: id  0 | task -1 | saving idle slot to prompt cache
  I srv   prompt_save:  - saving prompt with length 631, total state size = 69.024 MiB
  I slot prompt_clear: id  0 | task -1 | clearing prompt with 631 tokens
  ```
  As a result, idle slots never hold KV in the pool, so the idle-slot deletion in clause 4 cannot
  happen while this flag is on.
- **Correction:** when B is **pinned to slot 0 with `id_slot`**, it gets **0 cached tokens** and the
  whole prompt is recomputed. The selected-by-id path never consults the RAM cache for an emptied
  slot. In the source, an empty slot gives `f_keep = 0/0 = NaN`, so the cache load is skipped.
- When B is **not pinned** (probe `cache_idle_slots_on_unpinned`), the server picks a different
  LRU slot (A ran on slot 3, B on slot 1), finds the saved prompt in the RAM cache
  (`found better prompt with f_keep = 1.000, f_sim = 0.758`), restores it, and reports **631 cached
  tokens**.
- Control (`--cache-ram 512 --no-cache-idle-slots`): slot 0 keeps its KV, and the pinned B gets
  631 cached tokens.
- Each saved 631-token prompt costs about 69 MiB of host RAM (112 KiB per token at f16: 28 layers × 8 KV heads × 128 dims × 2 (K and V) × 2 bytes).

## Differences from the spec #1 contract (summary)

1. **The oversize-prompt rejection is not immediate.** It happens when the request reaches a slot,
   so it first waits behind busy slots. The limit is `prompt >= n_ctx`, and `n_ctx` is the **whole pool**
   under `--kv-unified`.
2. **New: prompt + `n_predict` > pool size is silently truncated** (HTTP 200, `truncated: true`,
   fewer output tokens). This is not an error from the server.
3. **New: a request pinned to a busy slot waits for that slot**, even when other slots are free.
4. **LRU ties go to the highest slot index.** The LCP threshold is strict (`> 0.10`) and only counts
   idle slots that hold tokens.
5. Idle-slot deletion also happens during **prompt processing**. Only one slot is deleted per retry,
   and the batch is halved only when no idle slot with tokens is left. Deleted KV is discarded,
   not saved to the RAM cache.
6. Overflow abort: **the server survives**. Running sequences get HTTP 500 `server_error`
   "Context size has been exceeded.", and queued requests are unaffected. This matches the
   contract, and the pilot risk in spec #1 ("crash vs. fail") is resolved as "fail only".
7. **`--cache-idle-slots` is on by default** at b11606, and `--cache-ram` defaults to 8192 MiB. So
   "runtime defaults" and "defaults plus `--cache-idle-slots`" are the **same** configuration. The
   cache-policy sweep in spec #1 needs to be restated, for example as
   `--no-cache-idle-slots` vs. `--cache-idle-slots` at a fixed `--cache-ram`, plus `--cache-ram 0`.
8. **With `--cache-idle-slots`, pinned (`id_slot`) agent steps lose their retained context.** Only
   unpinned requests get it back from the host-RAM cache, through the LRU / low-`f_keep` path. This
   path, where the host-RAM prompt cache restores a saved prompt into any slot, is not in the
   contract. It matters for story 16 (pinning via `id_slot`).
9. **KV occupancy is not observable from the server** (`/slots` and `/metrics` don't report it), so
   the recorder and the policy have to track footprints themselves.
10. Each slot's context is the whole pool under `--kv-unified`, so one model request can use the
    entire memory budget. A new `--kv-unified-per-slot` flag could cap this.
