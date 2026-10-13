"""The runtime interface: the one seam between the harness and an inference runtime.

``await runtime.submit(request)`` returns a ``Completion`` or raises a
``RuntimeFailure`` subclass. Implementations: the simulated runtime
(``agentserve.simulated``) and, later, the llama-server HTTP adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelRequest:
    prompt_tokens: tuple[int, ...]
    n_predict: int
    slot: int | None = None


@dataclass(frozen=True)
class Completion:
    output_tokens: tuple[int, ...]
    n_prompt_tokens: int
    n_cached_tokens: int
    slot: int
    prompt_s: float
    """Time the runtime spent processing the (uncached part of the) prompt."""
    decode_s: float
    """Time the runtime spent generating output tokens."""


class RuntimeFailure(Exception):
    """A model request that the runtime did not complete. ``kind`` is what gets recorded."""

    kind = "runtime_failure"


class PoolExhausted(RuntimeFailure):
    kind = "pool_exhausted"


class ExceedsContext(RuntimeFailure):
    kind = "exceeds_context"


class ServerError(RuntimeFailure):
    kind = "server_error"


class RequestTimeout(RuntimeFailure):
    kind = "timeout"


class Runtime(Protocol):
    async def submit(self, request: ModelRequest) -> Completion: ...
