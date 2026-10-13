"""The real runtime: the runtime interface over llama-server's `/completion` endpoint.

Request building and error mapping live in ``agentserve.llama_api`` (shared with
the pilot); this module only does the HTTP round trip and converts the outcome to
the runtime interface's ``Completion`` or a typed ``RuntimeFailure``.
"""

from __future__ import annotations

import httpx

from agentserve import llama_api
from agentserve.runtime import (Completion, ExceedsContext, ModelRequest, PoolExhausted,
                                RequestTimeout, RuntimeFailure, ServerError)

_FAILURES: dict[str, type[RuntimeFailure]] = {
    "pool_exhausted": PoolExhausted,
    "exceeds_context": ExceedsContext,
    "server_error": ServerError,
    "timeout": RequestTimeout,
}


class LlamaServerRuntime:
    def __init__(self, base_url: str, *, client: httpx.AsyncClient,
                 cache_prompt: bool = True) -> None:
        self._url = f"{base_url}/completion"
        self._client = client
        self._cache_prompt = cache_prompt

    async def submit(self, request: ModelRequest) -> Completion:
        payload = llama_api.completion_payload(
            list(request.prompt_tokens), request.n_predict,
            id_slot=-1 if request.slot is None else request.slot,
            cache_prompt=self._cache_prompt)
        try:
            response = await self._client.post(self._url, json=payload)
        except httpx.TimeoutException as e:
            raise RequestTimeout(repr(e)) from e
        except httpx.HTTPError as e:
            raise ServerError(repr(e)) from e
        try:
            body = response.json()
        except ValueError:
            body = response.text
        result = llama_api.parse_completion(response.status_code, body, n_predict=request.n_predict)
        if isinstance(result, llama_api.RuntimeError_):
            raise _FAILURES[result.kind](result.message)
        return Completion(
            output_tokens=tuple(result.tokens), n_prompt_tokens=result.prompt_tokens,
            n_cached_tokens=result.cached_tokens, slot=result.id_slot,
            prompt_s=result.prompt_ms / 1000.0, decode_s=result.predicted_ms / 1000.0)
