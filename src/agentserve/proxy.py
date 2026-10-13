"""Proxy and runner: releases agent tasks at their arrival times, sends each
model request through the admission policy and the runtime interface, waits
out tool delays, and records everything.

The proxy never retries a failed model request: the first runtime failure
ends the agent task as a failed task.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from agentserve.clock import Clock
from agentserve.policy import (ActiveRequest, AdmissionPolicy, PendingRequest,
                               ProxyState, RequestRef)
from agentserve.recorder import Recorder, RequestRecord, TaskRecord
from agentserve.runtime import ModelRequest, Runtime, RuntimeFailure
from agentserve.workload import AgentTaskSpec


@dataclass
class _Waiting:
    pending: PendingRequest
    admitted: asyncio.Future[int | None] = field(repr=False)


class Proxy:
    def __init__(self, *, policy: AdmissionPolicy, runtime: Runtime, clock: Clock,
                 recorder: Recorder, memory_budget_tokens: int, n_slots: int) -> None:
        self._policy = policy
        self._runtime = runtime
        self._clock = clock
        self._recorder = recorder
        self._memory_budget_tokens = memory_budget_tokens
        self._n_slots = n_slots
        self._waiting: list[_Waiting] = []
        self._active: dict[RequestRef, ActiveRequest] = {}
        self._admitted_tasks: dict[int, int | None] = {}

    async def run(self, tasks: list[AgentTaskSpec]) -> None:
        running = []
        for spec in sorted(tasks, key=lambda t: t.arrival_s):
            await self._clock.sleep(spec.arrival_s - self._clock.now())
            running.append(asyncio.create_task(self._run_task(spec)))
        await asyncio.gather(*running)

    async def _run_task(self, spec: AgentTaskSpec) -> None:
        arrived = self._clock.now()
        prompt = list(spec.initial_prompt)
        step_admitted: list[float] = []
        failure = None
        for k, step in enumerate(spec.steps):
            request = ModelRequest(tuple(prompt), step.n_predict)
            try:
                output = await self._model_request(spec.task_id, k, request, step_admitted)
            except RuntimeFailure as e:
                failure = e.kind
                break
            prompt += output
            prompt += step.tool_result_tokens
            await self._clock.sleep(step.tool_delay_s)
        self._admitted_tasks.pop(spec.task_id, None)
        self._recorder.task(TaskRecord(
            task_id=spec.task_id, task_class=spec.task_class, arrived_s=arrived,
            step_admitted_s=step_admitted, completed_s=self._clock.now(), failure=failure))
        self._schedule()

    async def _model_request(self, task_id: int, step: int, request: ModelRequest,
                             step_admitted: list[float]) -> tuple[int, ...]:
        ref = RequestRef(task_id, step)
        arrived = self._clock.now()
        waiting = _Waiting(
            PendingRequest(ref, arrived, len(request.prompt_tokens), request.n_predict),
            asyncio.get_running_loop().create_future())
        self._waiting.append(waiting)
        self._schedule()
        slot = await waiting.admitted
        admitted = self._clock.now()
        step_admitted.append(admitted)
        pinned = ModelRequest(request.prompt_tokens, request.n_predict, slot)
        try:
            completion = await self._runtime.submit(pinned)
        except RuntimeFailure as e:
            self._record_request(ref, arrived, admitted, request, slot, error=e.kind)
            raise
        finally:
            del self._active[ref]
            self._schedule()
        self._record_request(ref, arrived, admitted, request, completion.slot,
                             first_token_s=self._clock.now() - completion.decode_s,
                             n_cached=completion.n_cached_tokens,
                             n_output=len(completion.output_tokens))
        return completion.output_tokens

    def _schedule(self) -> None:
        if not self._waiting:
            return
        state = ProxyState(
            now_s=self._clock.now(),
            pending=tuple(w.pending for w in self._waiting),
            active=tuple(self._active.values()),
            admitted_tasks=dict(self._admitted_tasks),
            memory_budget_tokens=self._memory_budget_tokens,
            n_slots=self._n_slots,
        )
        by_ref = {w.pending.ref: w for w in self._waiting}
        for admission in self._policy.decide(state):
            waiting = by_ref.pop(admission.ref)
            self._waiting.remove(waiting)
            p = waiting.pending
            self._active[p.ref] = ActiveRequest(p.ref, admission.slot, p.n_prompt_tokens, p.n_predict)
            self._admitted_tasks.setdefault(p.ref.task_id, admission.slot)
            waiting.admitted.set_result(admission.slot)

    def _record_request(self, ref: RequestRef, arrived: float, admitted: float,
                        request: ModelRequest, slot: int | None, *, error: str | None = None,
                        first_token_s: float | None = None, n_cached: int | None = None,
                        n_output: int = 0) -> None:
        self._recorder.request(RequestRecord(
            task_id=ref.task_id, step=ref.step, arrived_s=arrived, admitted_s=admitted,
            first_token_s=first_token_s, completed_s=self._clock.now(),
            n_prompt_tokens=len(request.prompt_tokens), n_cached_tokens=n_cached,
            n_output_tokens=n_output, slot=slot, error=error))
