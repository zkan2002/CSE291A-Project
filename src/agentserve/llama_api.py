"""Wire format for llama-server's `/completion` endpoint.

Builds model-request payloads (token arrays, fixed output length) and turns HTTP
responses into either a `Completion` or a typed `RuntimeError_`. Error kinds follow
the runtime interface in spec #1: pool_exhausted / exceeds_context / server_error /
timeout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

# Plain-vocabulary token range for Llama-3 (special tokens start at 128000).
_TOKEN_LO, _TOKEN_HI = 1000, 120000

POOL_EXHAUSTED_MESSAGE = "Context size has been exceeded."

ErrorKind = Literal["pool_exhausted", "exceeds_context", "server_error", "timeout"]


@dataclass(frozen=True)
class Completion:
    id_slot: int
    tokens: list[int]
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int
    prompt_ms: float
    predicted_ms: float


@dataclass(frozen=True)
class RuntimeError_:
    kind: ErrorKind
    status: int | None
    message: str
    body: Any = field(default=None, compare=False)


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


def parse_completion(status: int, body: Any) -> Completion | RuntimeError_:
    if status == 200 and isinstance(body, dict) and "error" not in body:
        timings = body.get("timings", {})
        return Completion(
            id_slot=body["id_slot"],
            tokens=list(body.get("tokens", [])),
            prompt_tokens=body["tokens_evaluated"],
            cached_tokens=timings.get("cache_n", 0),
            output_tokens=body["tokens_predicted"],
            prompt_ms=timings.get("prompt_ms", 0.0),
            predicted_ms=timings.get("predicted_ms", 0.0),
        )
    error = body.get("error", {}) if isinstance(body, dict) else {}
    message = error.get("message", str(body))
    if error.get("type") == "exceed_context_size_error":
        kind: ErrorKind = "exceeds_context"
    elif POOL_EXHAUSTED_MESSAGE in message:
        kind = "pool_exhausted"
    else:
        kind = "server_error"
    return RuntimeError_(kind=kind, status=status, message=message, body=body)
