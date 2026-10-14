"""The real runtime: the runtime interface over llama-server's `/completion` endpoint.

Request building and response parsing live in ``agentserve.llama_api`` (shared
with the pilot); this module only does the HTTP round trip, returning the
``Completion`` or raising the typed ``RuntimeFailure`` it parsed.
"""

from __future__ import annotations

import httpx

from agentserve import llama_api
from agentserve.runtime import (Completion, ModelRequest, RequestTimeout, RuntimeFailure,
                                ServerError)


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
        if isinstance(result, RuntimeFailure):
            raise result
        return result
