"""Pool-overflow pilot against the real pinned llama-server (opt in: `pytest -m real_runtime`).

Each test runs one pilot probe on a fresh server launched from configs/pilot.toml
(-c 2048, -np 4, --kv-unified) and pins down the behaviour recorded in
docs/pilot/pilot-findings.md, so a different llama.cpp release that behaves
differently is caught.
"""

import asyncio
from pathlib import Path

import pytest

from agentserve.pilot import run_probe
from agentserve.server import load_server_config

pytestmark = pytest.mark.real_runtime

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pilot.toml"


@pytest.fixture
def probe(tmp_path):
    base = load_server_config(CONFIG)
    if not Path(base.binary).exists() or not Path(base.model).exists():
        pytest.skip("pinned llama-server or model not present (see docs/environment.md)")

    def run(name: str) -> dict:
        result = asyncio.run(run_probe(name, base, tmp_path))
        assert result["server_alive_at_end"], result
        return result

    return run


def test_prefix_extension_reports_cached_tokens_and_slot_selection_order(probe):
    f = probe("prefix_extension")["findings"]
    # everything except the last generated token (never decoded into the KV) is reused
    assert f["b_cached_tokens"] == f["b_cached_expected_at_most"] - 1
    # id_slot honoured; unpinned prefix extension found by LCP; LRU ties go to the highest index
    assert f["slots_chosen"] == {"A": 0, "B": 0, "C": 3, "D": 0, "E": 2}
    assert f["d_cached_tokens"] > 0


def test_requests_beyond_slot_count_wait_in_the_runtime_queue(probe):
    f = probe("slot_queueing")["findings"]
    assert f["requests_processing_mid"] == 4
    assert f["requests_deferred_mid"] == 2
    assert f["pinned_slots"] == [0, 0]


def test_full_pool_purges_first_idle_slot_in_index_order(probe):
    result = probe("idle_slot_deletion")
    f = result["findings"]
    assert f["big_request_ok"]
    # slot 2 was least recently used, but slot 0 (first in index order) lost its KV
    assert f["recheck_cached_tokens"] == {2: 499, 1: 499, 0: 0}
    assert any("purging slot 0" in line for line in result["server_log_evidence"])


def test_overflow_fails_every_running_sequence_and_server_survives(probe):
    result = probe("overflow_abort")
    f = result["findings"]
    assert f["statuses"][:4] == [500] * 4
    assert f["kinds"][:4] == ["pool_exhausted"] * 4
    body = result["requests"][0]["outcome"]["body"]
    assert body == {"error": {"code": 500, "message": "Context size has been exceeded.", "type": "server_error"}}
    # the request still queued behind the running ones is untouched and later succeeds
    assert f["kinds"][4] == "ok"
    assert f["health_after"] == {"status": "ok"}
    assert f["follow_up_ok"]


def test_oversize_prompt_is_rejected_when_it_reaches_a_slot(probe):
    f = probe("oversize_prompt")["findings"]
    assert f["idle_status"] == 400 and f["exact_status"] == 400
    assert f["idle_reject_latency_s"] < 0.5
    # with all slots busy, the rejection waits for a free slot
    assert f["busy_reject_at"] >= min(f["busy_requests_done_at"]) - 0.05
    # prompt fits but prompt + n_predict does not: output is silently cut at the pool size
    fits = f["prompt_fits_but_output_does_not"]
    assert fits["ok"] and fits["output_tokens"] == 2048 - 1900


def test_cache_idle_slots_offload_loses_context_for_pinned_requests(probe):
    pinned = probe("cache_idle_slots_on")["findings"]
    assert pinned["b_cached_tokens"] == 0
    unpinned = probe("cache_idle_slots_on_unpinned")["findings"]
    assert unpinned["b_cached_tokens"] == 631 and unpinned["b_slot"] != unpinned["a_slot"]
    kept = probe("cache_idle_slots_off_cache_ram_on")["findings"]
    assert kept["b_cached_tokens"] == 631
