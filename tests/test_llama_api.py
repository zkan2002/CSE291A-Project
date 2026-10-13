"""Wire format for llama-server's /completion endpoint: request payloads and typed outcomes.

Response and error bodies below are copied from the pinned server (b11606) during the pilot.
"""

from agentserve.llama_api import (
    Completion,
    RuntimeError_,
    completion_payload,
    parse_completion,
    synthetic_tokens,
)


def test_payload_sends_token_array_with_fixed_output_length():
    payload = completion_payload([11, 12, 13], n_predict=50, id_slot=2)
    assert payload["prompt"] == [11, 12, 13]
    assert payload["n_predict"] == 50
    assert payload["ignore_eos"] is True
    assert payload["id_slot"] == 2
    assert payload["cache_prompt"] is True
    assert payload["return_tokens"] is True
    assert payload["temperature"] == 0


def test_payload_without_slot_lets_the_server_choose():
    assert completion_payload([1], n_predict=1)["id_slot"] == -1
    assert completion_payload([1], n_predict=1, cache_prompt=False)["cache_prompt"] is False


def test_synthetic_tokens_are_deterministic_and_salted():
    a = synthetic_tokens(500, salt=1)
    assert a == synthetic_tokens(500, salt=1)
    assert len(a) == 500
    b = synthetic_tokens(500, salt=2)
    assert a[0] != b[0], "different salts must not share a prefix"
    # plain vocabulary only: no BOS (128000) or other special tokens
    assert all(1000 <= t < 120000 for t in a + b)


def test_successful_response_reports_cached_tokens_and_slot():
    body = {
        "id_slot": 1,
        "tokens": [5, 6],
        "tokens_predicted": 2,
        "tokens_evaluated": 400,
        "tokens_cached": 401,
        "timings": {"cache_n": 300, "prompt_n": 100, "prompt_ms": 29.2, "predicted_n": 2, "predicted_ms": 14.0},
    }
    result = parse_completion(200, body)
    assert result == Completion(
        id_slot=1, tokens=[5, 6], prompt_tokens=400, cached_tokens=300, output_tokens=2,
        prompt_ms=29.2, predicted_ms=14.0,
    )


def test_pool_overflow_abort_is_pool_exhausted():
    body = {"error": {"code": 500, "message": "Context size has been exceeded.", "type": "server_error"}}
    result = parse_completion(500, body)
    assert isinstance(result, RuntimeError_)
    assert result.kind == "pool_exhausted"
    assert result.status == 500


def test_oversize_prompt_is_exceeds_context():
    body = {
        "error": {
            "code": 400,
            "message": "request (2100 tokens) exceeds the available context size (2048 tokens), try increasing it",
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 2100,
            "n_ctx": 2048,
        }
    }
    result = parse_completion(400, body)
    assert result.kind == "exceeds_context"


def test_other_errors_are_server_errors():
    body = {"error": {"code": 500, "message": "Compute error.", "type": "server_error"}}
    assert parse_completion(500, body).kind == "server_error"
    assert parse_completion(502, "not json").kind == "server_error"
