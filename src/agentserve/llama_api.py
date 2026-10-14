"""Wire format for llama-server's `/completion` endpoint.

Builds model-request payloads (token arrays, fixed output length) and turns HTTP
responses into the runtime interface's outcomes (``agentserve.runtime``): a
``Completion``, or a typed ``RuntimeFailure`` (pool_exhausted / exceeds_context /
server_error; timeouts happen at the HTTP client, not in a response).
"""

from __future__ import annotations

import random
from typing import Any

from agentserve.runtime import (Completion, ExceedsContext, PoolExhausted, RuntimeFailure,
                                ServerError)

# Plain-vocabulary token range for Llama-3 (special tokens start at 128000).
_TOKEN_LO, _TOKEN_HI = 1000, 120000

POOL_EXHAUSTED_MESSAGE = "Context size has been exceeded."

def synthetic_tokens(n: int, salt: int) -> list[int]:
    """Deterministic plain-vocabulary token IDs; different salts start differently."""
    rng = random.Random(salt)
    width = _TOKEN_HI - _TOKEN_LO
    first = _TOKEN_LO + (salt * 7919) % width
    return [first] + [rng.randrange(_TOKEN_LO, _TOKEN_HI) for _ in range(n - 1)]


def completion_payload(
    prompt: list[int], n_predict: int, id_slot: int = -1, cache_prompt: bool = True
) -> dict[str, Any]:
    return {
        "prompt": list(prompt),
        "n_predict": n_predict,
        "ignore_eos": True,
        "id_slot": id_slot,
        "cache_prompt": cache_prompt,
        "return_tokens": True,
        "temperature": 0,
    }


def parse_completion(status: int, body: Any,
                     n_predict: int | None = None) -> Completion | RuntimeFailure:
    """Map one `/completion` response to a completion or a typed failure (returned,
    not raised). ``n_cached_tokens`` is the server's ``timings.cache_n``, ``slot``
    its ``id_slot``, and ``prompt_s``/``decode_s`` its ``prompt_ms``/``predicted_ms``.

    With ``n_predict`` given, a completion with fewer output tokens is an
    ``exceeds_context`` failure: the server silently truncates when prompt +
    ``n_predict`` doesn't fit the pool (HTTP 200, ``truncated: true``).
    """
    if status == 200 and isinstance(body, dict) and "error" not in body:
        if n_predict is not None and (body["tokens_predicted"] < n_predict or body.get("truncated")):
            return ExceedsContext(
                f"output truncated: {body['tokens_predicted']} of {n_predict} tokens",
                status=status, body=body)
        timings = body.get("timings", {})
        return Completion(
            output_tokens=tuple(body.get("tokens", [])),
            n_prompt_tokens=body["tokens_evaluated"],
            n_cached_tokens=timings.get("cache_n", 0),
            slot=body["id_slot"],
            prompt_s=timings.get("prompt_ms", 0.0) / 1000.0,
            decode_s=timings.get("predicted_ms", 0.0) / 1000.0,
        )
    error = body.get("error", {}) if isinstance(body, dict) else {}
    message = error.get("message", str(body))
    if error.get("type") == "exceed_context_size_error":
        failure: type[RuntimeFailure] = ExceedsContext
    elif POOL_EXHAUSTED_MESSAGE in message:
        failure = PoolExhausted
    else:
        failure = ServerError
    return failure(message, status=status, body=body)
