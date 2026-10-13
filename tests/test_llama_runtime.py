"""The llama-server adapter behind the runtime interface.

Each test stands up a fake llama-server at the HTTP boundary (httpx.MockTransport)
that answers with response bodies copied from the pinned server during the pilot,
then submits a model request through the runtime interface.
"""

import asyncio
import json

import httpx
import pytest

from agentserve.llama_runtime import LlamaServerRuntime
from agentserve.runtime import (Completion, ExceedsContext, ModelRequest, PoolExhausted,
                                RequestTimeout, ServerError)

POOL_EXHAUSTED = (500, {"error": {"code": 500, "message": "Context size has been exceeded.",
                                  "type": "server_error"}})
OVERSIZE = (400, {"error": {
    "code": 400,
    "message": "request (2100 tokens) exceeds the available context size (2048 tokens), try increasing it",
    "type": "exceed_context_size_error", "n_prompt_tokens": 2100, "n_ctx": 2048}})


def ok(request_json: dict, *, slot: int = 1, cached: int = 0, n_output: int | None = None) -> tuple:
    n = request_json["n_predict"] if n_output is None else n_output
    return 200, {
        "id_slot": slot,
        "tokens": [7000 + i for i in range(n)],
        "tokens_predicted": n,
        "tokens_evaluated": len(request_json["prompt"]),
        "truncated": n < request_json["n_predict"],
        "timings": {"cache_n": cached, "prompt_ms": 30.0, "predicted_ms": 250.0},
    }


def submit(respond, request: ModelRequest, **kwargs):
    """Submit ``request`` to a fake server whose /completion answers ``respond(json)``."""
    sent = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/completion"
        body = json.loads(http_request.content)
        sent.append(body)
        status, payload = respond(body)
        return httpx.Response(status, json=payload)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            runtime = LlamaServerRuntime("http://fake", client=client, **kwargs)
            return await runtime.submit(request)

    return asyncio.run(go()), sent


def test_round_trip_returns_output_cached_tokens_slot_and_timings():
    completion, sent = submit(lambda j: ok(j, slot=2, cached=300),
                              ModelRequest(tuple(range(1000, 1400)), n_predict=5, slot=2))

    assert completion == Completion(
        output_tokens=(7000, 7001, 7002, 7003, 7004), n_prompt_tokens=400, n_cached_tokens=300,
        slot=2, prompt_s=0.030, decode_s=0.250)
    assert sent[0]["prompt"] == list(range(1000, 1400))
    assert sent[0]["n_predict"] == 5
    assert sent[0]["ignore_eos"] is True
    assert sent[0]["id_slot"] == 2
    assert sent[0]["cache_prompt"] is True


def test_unpinned_request_lets_the_server_choose_the_slot():
    completion, sent = submit(lambda j: ok(j, slot=3), ModelRequest((1, 2, 3), n_predict=1))
    assert sent[0]["id_slot"] == -1
    assert completion.slot == 3


def test_prompt_caching_can_be_switched_off():
    _, sent = submit(ok, ModelRequest((1, 2, 3), n_predict=1), cache_prompt=False)
    assert sent[0]["cache_prompt"] is False


@pytest.mark.parametrize("response, error", [
    (POOL_EXHAUSTED, PoolExhausted),
    (OVERSIZE, ExceedsContext),
    ((500, {"error": {"code": 500, "message": "Compute error.", "type": "server_error"}}), ServerError),
])
def test_server_errors_map_to_typed_runtime_failures(response, error):
    with pytest.raises(error):
        submit(lambda j: response, ModelRequest((1, 2, 3), n_predict=4))


def test_truncated_output_is_exceeds_context():
    with pytest.raises(ExceedsContext):
        submit(lambda j: ok(j, n_output=148), ModelRequest(tuple(range(1900)), n_predict=300))


def test_request_that_outlasts_the_timeout_is_a_timeout():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await LlamaServerRuntime("http://fake", client=client).submit(ModelRequest((1,), 1))

    with pytest.raises(RequestTimeout):
        asyncio.run(go())


def test_lost_connection_is_a_server_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await LlamaServerRuntime("http://fake", client=client).submit(ModelRequest((1,), 1))

    with pytest.raises(ServerError):
        asyncio.run(go())
