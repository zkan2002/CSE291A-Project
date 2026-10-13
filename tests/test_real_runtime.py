"""Smoke tests against the real pinned llama-server (opt in: `pytest -m real_runtime`).

- The llama-server adapter round trip, and cached tokens on a prefix-extending second
  model request, through the runtime interface.
- One short experiment through the one-command entry point on the real server: the run
  directory has the same format as a simulated run, plus system samples and run metadata,
  and the analysis reads it unchanged.
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from agentserve.analysis import summarize
from agentserve.cli import main
from agentserve.llama_api import synthetic_tokens
from agentserve.llama_runtime import LlamaServerRuntime
from agentserve.runtime import ModelRequest
from agentserve.server import load_server_config, start_server, stop_server

pytestmark = pytest.mark.real_runtime

PILOT = Path(__file__).resolve().parents[1] / "configs" / "pilot.toml"
SMOKE = Path(__file__).resolve().parents[1] / "configs" / "real-smoke.toml"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture
def server_cfg():
    cfg = load_server_config(PILOT, memory_budget=4096)
    if not Path(cfg.binary).exists() or not Path(cfg.model).exists():
        pytest.skip("pinned llama-server or model not present (see docs/environment.md)")
    return cfg


@pytest.fixture
def server(server_cfg, tmp_path):
    proc = start_server(server_cfg, log_path=tmp_path / "server.log")
    yield server_cfg
    stop_server(proc)


def test_adapter_round_trip_and_cached_tokens_on_a_prefix_extension(server):
    first_prompt = tuple(synthetic_tokens(600, salt=11))

    async def go():
        async with httpx.AsyncClient(timeout=120) as client:
            runtime = LlamaServerRuntime(server.base_url, client=client)
            first = await runtime.submit(ModelRequest(first_prompt, n_predict=16, slot=1))
            # next agent step: previous prompt + its output + a tool result
            second_prompt = first_prompt + first.output_tokens + tuple(synthetic_tokens(200, salt=12))
            second = await runtime.submit(ModelRequest(second_prompt, n_predict=16, slot=1))
            return first, second, second_prompt

    first, second, second_prompt = asyncio.run(go())

    assert len(first.output_tokens) == 16 and first.slot == 1
    assert first.n_prompt_tokens == 600 and first.n_cached_tokens == 0
    assert second.slot == 1 and len(second.output_tokens) == 16
    assert second.n_prompt_tokens == len(second_prompt)
    # everything but the first step's last output token (never decoded into the KV) is reused
    assert second.n_cached_tokens == 600 + 16 - 1
    assert first.decode_s > 0 and first.prompt_s > 0


def test_short_experiment_on_the_real_server_writes_a_run_directory(server_cfg, tmp_path):
    run_dir = tmp_path / "run"

    assert main(["run", str(SMOKE), "--out", str(run_dir)]) == 0

    tasks = read_jsonl(run_dir / "tasks.jsonl")
    requests = read_jsonl(run_dir / "requests.jsonl")
    assert len(tasks) > 0 and all(t["failure"] is None for t in tasks)
    assert len(requests) == sum(len(t["step_admitted_s"]) for t in tasks)
    assert all(r["n_output_tokens"] == r["n_predict"] for r in requests)
    # task-level FIFO pins each agent task to a slot, so later agent steps reuse retained context
    later = [r for r in requests if r["step"] > 0]
    assert later and all(r["n_cached_tokens"] > 0 for r in later)
    # same format as a simulated run
    sim_request_fields = {"task_id", "step", "arrived_s", "admitted_s", "first_token_s", "completed_s",
                          "n_prompt_tokens", "n_cached_tokens", "n_predict", "n_output_tokens",
                          "slot", "error"}
    assert set(requests[0]) == sim_request_fields
    assert json.loads((run_dir / "config.json").read_text())["runtime"]["kind"] == "llama_server"

    samples = read_jsonl(run_dir / "samples.jsonl")
    run_end = max(t["completed_s"] for t in tasks)
    assert len(samples) >= 0.8 * run_end / 0.1
    gaps = [b["t_s"] - a["t_s"] for a, b in zip(samples, samples[1:])]
    assert sorted(gaps)[len(gaps) // 2] == pytest.approx(0.1, abs=0.02)
    assert all(s["vram_used_bytes"] > 1e9 for s in samples)  # model weights alone are ~2 GB
    assert all(s["server_rss_bytes"] > 0 for s in samples)
    assert all(len(s["slots"]) == 4 for s in samples if s["slots"] is not None)
    assert any(slot["is_processing"] for s in samples if s["slots"] for slot in s["slots"])

    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["llama_cpp"]["release_tag"] == "b11606"
    assert meta["llama_cpp"]["commit"] == "1b70350c9"
    assert meta["model"]["sha256"] == "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff"
    assert meta["gpu"]["driver_version"] and meta["gpu"]["cuda_driver_version"]
    assert len(meta["git"]["commit"]) == 40
    assert "--kv-unified" in meta["launch_args"]
    assert (run_dir / "server.log").stat().st_size > 0

    # unloaded completion times measured on the real server, each agent task alone
    unloaded = read_jsonl(run_dir / "unloaded.jsonl")
    assert sorted(u["task_id"] for u in unloaded) == sorted(t["task_id"] for t in tasks)
    assert all(u["failure"] is None and u["unloaded_completion_s"] > 0.5 for u in unloaded)
    assert (run_dir / "unloaded-server.log").stat().st_size > 0

    summary = summarize(run_dir)
    assert summary.agent_tasks == len(tasks) and summary.throughput_tasks_per_min > 0
    assert summary.goodput_tasks_per_min > 0

    # the server was stopped after the run
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"http://127.0.0.1:{json.loads((run_dir / 'config.json').read_text())['server']['port']}/health",
                  timeout=1.0)
