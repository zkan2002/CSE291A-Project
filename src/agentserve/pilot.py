"""Pool-overflow pilot (ticket #2): probe how the pinned llama-server behaves when its
shared KV pool fills up.

Each probe starts a fresh server from the pilot config (plus per-probe overrides), sends
`/completion` model requests as token arrays, and records exactly what every client
received, the server's own log lines about slot selection / purging / aborts, and whether
the server is still alive afterwards.

    python -m uv run python -m agentserve.pilot configs/pilot.toml [--out results/pilot] [--only NAME ...]

The JSON report (`pilot-report.json`) is written to the output directory together with
one server log per probe.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from agentserve.llama_api import (
    Completion,
    completion_payload,
    parse_completion,
    synthetic_tokens,
)
from agentserve.server import ServerConfig, load_server_config, start_server, stop_server

# Server log lines that carry evidence about the contract clauses.
LOG_EVIDENCE = re.compile(
    r"selected slot|purging slot|Context size has been exceeded|decode\(\) failed|failed to find free space"
    r"|exceeds the available context|larger than the max context|context shift|prompt cache|idle slot"
    r"|got exception|error|cache state|restored context|srv\s+load:|srv\s+update:|TEST_TAG|prompt_save|prompt_load"
    r"|saving|cache size|cached prompt",
    re.IGNORECASE,
)


class ProbeSession:
    """One fresh server plus a log of every model request sent to it."""

    def __init__(self, cfg: ServerConfig, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client
        self.t0 = time.monotonic()
        self.requests: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 4)

    async def send(self, label: str, prompt: list[int], n_predict: int, id_slot: int = -1) -> dict[str, Any]:
        record: dict[str, Any] = {
            "label": label,
            "prompt_tokens": len(prompt),
            "n_predict": n_predict,
            "id_slot_requested": id_slot,
            "t_send": self.now(),
        }
        try:
            resp = await self.client.post(
                f"{self.cfg.base_url}/completion", json=completion_payload(prompt, n_predict, id_slot)
            )
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            status: int | None = resp.status_code
            result = parse_completion(resp.status_code, body)
        except httpx.HTTPError as e:  # connection reset, server died, ...
            status, body, result = None, repr(e), None
        record["t_done"] = self.now()
        record["status"] = status
        if isinstance(body, dict) and "error" not in body:
            record["response_meta"] = {
                k: body.get(k) for k in ("stop_type", "truncated", "tokens_cached", "tokens_evaluated")
            }
        if isinstance(result, Completion):
            outcome = dataclasses.asdict(result)
            record["_tokens"] = outcome.pop("tokens")
            record["outcome"] = {"ok": True, **outcome}
        elif result is not None:
            record["outcome"] = {"ok": False, "kind": result.kind, "message": result.message, "body": result.body}
        else:
            record["outcome"] = {"ok": False, "kind": "transport_error", "message": body}
        self.requests.append(record)
        return record

    async def get_json(self, path: str) -> Any:
        try:
            r = await self.client.get(f"{self.cfg.base_url}{path}", timeout=10)
            return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
        except httpx.HTTPError as e:
            return repr(e)

    async def note(self, label: str, path: str) -> Any:
        value = await self.get_json(path)
        self.notes.append({"label": label, "t": self.now(), "path": path, "value": value})
        return value

    async def metrics(self, label: str) -> dict[str, float]:
        text = await self.get_json("/metrics")
        values = {}
        if isinstance(text, str):
            for line in text.splitlines():
                if line.startswith("llamacpp:"):
                    name, _, v = line.partition(" ")
                    values[name.removeprefix("llamacpp:")] = float(v)
        self.notes.append({"label": label, "t": self.now(), "path": "/metrics", "value": values})
        return values


def _slots_summary(slots: Any) -> Any:
    if not isinstance(slots, list):
        return slots
    return [
        {k: s.get(k) for k in ("id", "n_ctx", "is_processing", "id_task", "n_prompt_tokens") if k in s}
        for s in slots
    ]


# --------------------------------------------------------------------------- probes
# Pool: -c 2048 shared by 4 slots (each slot may use the whole pool under --kv-unified).


async def probe_prefix_extension(s: ProbeSession) -> dict[str, Any]:
    """A second request that extends the first one's prompt reports cached tokens; and
    slot selection order (id_slot, then LCP similarity, then LRU)."""
    a_prompt = synthetic_tokens(600, salt=1)
    a = await s.send("A: 600-token prompt pinned to slot 0", a_prompt, 32, id_slot=0)
    tool1 = synthetic_tokens(200, salt=101)
    b_prompt = a_prompt + a["_tokens"] + tool1
    b = await s.send("B: A + A's output + 200-token tool result, pinned to slot 0", b_prompt, 32, id_slot=0)
    c = await s.send("C: unrelated 300-token prompt, no id_slot (expect LRU)", synthetic_tokens(300, salt=2), 16)
    tool2 = synthetic_tokens(200, salt=102)
    d_prompt = b_prompt + b["_tokens"] + tool2
    d = await s.send("D: extends B, no id_slot (expect LCP similarity -> slot 0)", d_prompt, 16)
    e = await s.send("E: another unrelated prompt, no id_slot (expect LRU)", synthetic_tokens(300, salt=3), 16)
    await s.note("slots at end", "/slots")
    return {
        "b_cached_tokens": b["outcome"].get("cached_tokens"),
        "b_cached_expected_at_most": len(a_prompt) + len(a["_tokens"]),
        "slots_chosen": {r["label"][:1]: r["outcome"].get("id_slot") for r in (a, b, c, d, e)},
        "d_cached_tokens": d["outcome"].get("cached_tokens"),
    }


async def probe_slot_queueing(s: ProbeSession) -> dict[str, Any]:
    """Requests beyond -np running sequences wait in the runtime's queue; a request pinned
    to a busy slot waits for that slot."""
    prompts = [synthetic_tokens(100, salt=200 + i) for i in range(6)]
    tasks = [asyncio.create_task(s.send(f"Q{i}: 100+200, no id_slot", p, 200)) for i, p in enumerate(prompts)]
    await asyncio.sleep(0.3)
    mid = await s.metrics("metrics while 6 requests are in flight on 4 slots")
    six = await asyncio.gather(*tasks)
    pinned = await asyncio.gather(
        s.send("P0: pinned to slot 0", synthetic_tokens(100, salt=300), 200, id_slot=0),
        s.send("P1: also pinned to slot 0", synthetic_tokens(100, salt=301), 200, id_slot=0),
    )
    return {
        "six_done_at": sorted(r["t_done"] for r in six),
        "requests_processing_mid": mid.get("requests_processing"),
        "requests_deferred_mid": mid.get("requests_deferred"),
        "pinned_slots": [r["outcome"].get("id_slot") for r in pinned],
        "pinned_done_at": [r["t_done"] for r in pinned],
    }


async def probe_idle_slot_deletion(s: ProbeSession) -> dict[str, Any]:
    """Fill the pool with idle slots' KV, then start a request that needs more room.
    Fill order 2, 1, 0 so that slot 2 is least recently used but slot 0 is first in
    index order."""
    held = {}
    for slot in (2, 1, 0):
        held[slot] = synthetic_tokens(500, salt=400 + slot)
        await s.send(f"fill slot {slot} with 500+8", held[slot], 8, id_slot=slot)
    await s.note("slots after fill", "/slots")
    big = await s.send("slot 3: 700+8 needs more than the free pool", synthetic_tokens(700, salt=410), 8, id_slot=3)
    rechecks = {}
    for slot in (2, 1, 0):
        r = await s.send(f"recheck slot {slot}: resend its prompt", held[slot], 1, id_slot=slot)
        rechecks[slot] = r["outcome"].get("cached_tokens")
    return {"big_request_ok": big["outcome"]["ok"], "recheck_cached_tokens": rechecks}


async def probe_overflow_abort(s: ProbeSession) -> dict[str, Any]:
    """Four running sequences that together outgrow the pool (4 x (400 + 400) > 2048),
    plus a fifth request queued behind them."""
    running = [
        asyncio.create_task(s.send(f"R{i}: 400+400 on slot {i}", synthetic_tokens(400, salt=500 + i), 400, id_slot=i))
        for i in range(4)
    ]
    await asyncio.sleep(0.2)
    queued = asyncio.create_task(s.send("R4: 100+16 queued (all slots busy)", synthetic_tokens(100, salt=510), 16))
    results = await asyncio.gather(*running, queued)
    alive = await s.get_json("/health")
    after = await s.send("after: fresh 100+16 request", synthetic_tokens(100, salt=511), 16)
    await s.note("slots after abort", "/slots")
    await s.metrics("metrics after abort")
    return {
        "statuses": [r["status"] for r in results],
        "kinds": [r["outcome"].get("kind", "ok") for r in results],
        "health_after": alive,
        "follow_up_ok": after["outcome"]["ok"],
    }


async def probe_oversize_prompt(s: ProbeSession) -> dict[str, Any]:
    """A single prompt larger than a slot's context."""
    over = await s.send("2100-token prompt on idle server", synthetic_tokens(2100, salt=600), 8)
    exact = await s.send("2048-token prompt (= n_ctx)", synthetic_tokens(2048, salt=601), 8)
    fits = await s.send("1900-token prompt + n_predict 300 (sum > n_ctx)", synthetic_tokens(1900, salt=602), 300)
    busy = [
        asyncio.create_task(s.send(f"busy {i}: 100+300", synthetic_tokens(100, salt=610 + i), 300, id_slot=i))
        for i in range(4)
    ]
    await asyncio.sleep(0.2)
    over_busy = await s.send("2100-token prompt while all slots are busy", synthetic_tokens(2100, salt=603), 8)
    busy_done = await asyncio.gather(*busy)
    return {
        "idle_reject_latency_s": round(over["t_done"] - over["t_send"], 4),
        "idle_status": over["status"],
        "exact_status": exact["status"],
        "prompt_fits_but_output_does_not": fits["outcome"],
        "busy_reject_at": over_busy["t_done"],
        "busy_requests_done_at": sorted(r["t_done"] for r in busy_done),
    }


async def probe_cache_idle_slots(s: ProbeSession) -> dict[str, Any]:
    """With --cache-idle-slots (+ --cache-ram), starting a new task offloads idle slots.
    Does the next agent step pinned to the offloaded slot still get cached tokens?"""
    a_prompt = synthetic_tokens(600, salt=700)
    a = await s.send("A: 600+32 on slot 0", a_prompt, 32, id_slot=0)
    await s.note("slots after A", "/slots")
    await s.send("X: unrelated 300+16 on slot 1 (new task -> idle slot 0 offloaded?)", synthetic_tokens(300, salt=701), 16, id_slot=1)
    await s.note("slots after X", "/slots")
    b_prompt = a_prompt + a["_tokens"] + synthetic_tokens(200, salt=702)
    b = await s.send("B: extends A, pinned to slot 0", b_prompt, 32, id_slot=0)
    c_prompt = b_prompt + b["_tokens"] + synthetic_tokens(200, salt=703)
    c = await s.send("C: extends B, no id_slot", c_prompt, 16)
    return {
        "b_cached_tokens": b["outcome"].get("cached_tokens"),
        "c_cached_tokens": c["outcome"].get("cached_tokens"),
        "c_slot": c["outcome"].get("id_slot"),
    }


async def probe_cache_idle_slots_unpinned(s: ProbeSession) -> dict[str, Any]:
    """Same agent pattern without id_slot: after idle slot 0 is offloaded, can the next
    agent step restore its retained context from the host-RAM prompt cache?"""
    a_prompt = synthetic_tokens(600, salt=710)
    a = await s.send("A: 600+32, no id_slot", a_prompt, 32)
    await s.send("X: unrelated 300+16, no id_slot (new task -> idle slots offloaded?)", synthetic_tokens(300, salt=711), 16)
    await s.note("slots after X", "/slots")
    b_prompt = a_prompt + a["_tokens"] + synthetic_tokens(200, salt=712)
    b = await s.send("B: extends A, no id_slot", b_prompt, 32)
    return {
        "a_slot": a["outcome"].get("id_slot"),
        "b_slot": b["outcome"].get("id_slot"),
        "b_cached_tokens": b["outcome"].get("cached_tokens"),
    }


Probe = Callable[[ProbeSession], Awaitable[dict[str, Any]]]

# name -> (probe, config overrides)
PROBES: dict[str, tuple[Probe, dict[str, Any]]] = {
    "prefix_extension": (probe_prefix_extension, {}),
    "slot_queueing": (probe_slot_queueing, {}),
    "idle_slot_deletion": (probe_idle_slot_deletion, {}),
    "overflow_abort": (probe_overflow_abort, {}),
    "oversize_prompt": (probe_oversize_prompt, {}),
    # -lv 4 (debug) so the server logs its prompt-cache saves/loads
    "cache_idle_slots_on": (
        probe_cache_idle_slots, {"cache_ram_mib": 512, "cache_idle_slots": True, "extra_args": ["-lv", "4"]}
    ),
    "cache_idle_slots_on_unpinned": (
        probe_cache_idle_slots_unpinned, {"cache_ram_mib": 512, "cache_idle_slots": True, "extra_args": ["-lv", "4"]}
    ),
    "cache_idle_slots_off_cache_ram_on": (
        probe_cache_idle_slots, {"cache_ram_mib": 512, "cache_idle_slots": False, "extra_args": ["-lv", "4"]}
    ),
}


def _log_evidence(log_path: Path) -> list[str]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ln for ln in lines if LOG_EVIDENCE.search(ln)]


async def run_probe(name: str, base: ServerConfig, out_dir: Path) -> dict[str, Any]:
    probe, overrides = PROBES[name]
    cfg = dataclasses.replace(base, **overrides)
    log_path = out_dir / f"{name}.server.log"
    proc = start_server(cfg, log_path)
    try:
        async with httpx.AsyncClient(timeout=300, limits=httpx.Limits(max_connections=32)) as client:
            session = ProbeSession(cfg, client)
            try:
                findings = await probe(session)
            except Exception as e:  # keep whatever was observed
                findings = {"probe_error": repr(e)}
            server_alive = proc.poll() is None
            health = await session.get_json("/health")
    finally:
        stop_server(proc)
    for r in session.requests:
        r.pop("_tokens", None)
    return {
        "name": name,
        "doc": " ".join((probe.__doc__ or "").split()),
        "config_overrides": overrides,
        "server_alive_at_end": server_alive,
        "server_exit_code_if_dead": None if server_alive else proc.returncode,
        "health_at_end": health,
        "findings": findings,
        "requests": session.requests,
        "notes": [
            {**n, "value": _slots_summary(n["value"])} if n["path"] == "/slots" else n for n in session.notes
        ],
        "server_log_evidence": _log_evidence(log_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--out", default="results/pilot")
    parser.add_argument("--only", nargs="*", choices=sorted(PROBES))
    ns = parser.parse_args(argv)
    base = load_server_config(ns.config)
    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "server_config": dataclasses.asdict(base),
        "probes": {},
    }
    for name in ns.only or list(PROBES):
        print(f"[pilot] {name} ...", flush=True)
        result = asyncio.run(run_probe(name, base, out_dir))
        report["probes"][name] = result
        print(json.dumps({"alive": result["server_alive_at_end"], **result["findings"]}, default=str), flush=True)
    (out_dir / "pilot-report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[pilot] report: {out_dir / 'pilot-report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
