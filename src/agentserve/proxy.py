"""Proxy and runner: releases agent tasks at their arrival times, sends each
model request through the admission policy and the runtime interface, waits
out tool delays, and records everything.

The proxy never retries a failed model request: the first runtime failure
ends the agent task as a failed task. So does its hard timeout, counted from
its arrival: the model request it is waiting on (queued or in flight) is
cancelled and recorded with the error ``timeout``.

The server does not report KV occupancy (pilot amendment 7), so the proxy
tracks the token footprint it has admitted: each model request in flight
(prompt plus ``n_predict``), plus the retained context of each agent task under
way whose last model request was pinned to a slot and succeeded (prompt plus
output), until that slot is given to another model request or the agent task
ends. Every change is recorded (``Recorder.kv``), with the runtime's own
occupancy beside it when ``kv_probe`` gives one (the simulated runtime).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

from agentserve.clock import Clock
from agentserve.policy import (ActiveRequest, AdmissionPolicy, PendingRequest,
                               ProxyState, RequestRef)
from agentserve.recorder import Recorder, RequestRecord, TaskRecord
from agentserve.runtime import ModelRequest, RequestTimeout, Runtime, RuntimeFailure
from agentserve.workload import AgentTaskSpec


@dataclass
class _Waiting:
    pending: PendingRequest
    admitted: asyncio.Future[int | None] = field(repr=False)


class Proxy:
    def __init__(self, *, policy: AdmissionPolicy, runtime: Runtime, clock: Clock,
                 recorder: Recorder, memory_budget_tokens: int, n_slots: int,
                 hard_timeouts: Mapping[int, float | None] | None = None,
                 kv_probe: Callable[[], int] | None = None) -> None:
        """``hard_timeouts``: per agent task (by task ID), seconds from its arrival
        after which it fails with cause ``timeout``; None or absent = no timeout.
        ``kv_probe``: the runtime's own KV-pool occupancy in tokens, if it knows it."""
        self._policy = policy
        self._runtime = runtime
        self._clock = clock
        self._recorder = recorder
        self._memory_budget_tokens = memory_budget_tokens
        self._n_slots = n_slots
        self._hard_timeouts = dict(hard_timeouts or {})
        self._waiting: list[_Waiting] = []
        self._active: dict[RequestRef, ActiveRequest] = {}
        self._admitted_tasks: dict[int, int | None] = {}
        self._retained: dict[int, tuple[int, int]] = {}
        """Agent task -> (slot, tokens) of the retained context it has pinned there."""
        self._kv_probe = kv_probe
        self._last_kv: dict | None = None

    async def run(self, tasks: list[AgentTaskSpec]) -> None:
        running = []
        for spec in sorted(tasks, key=lambda t: t.arrival_s):
            await self._clock.sleep(spec.arrival_s - self._clock.now())
            running.append(asyncio.create_task(self._run_task(spec)))
        await asyncio.gather(*running)

    async def _run_task(self, spec: AgentTaskSpec) -> None:
        arrived = self._clock.now()
        step_admitted: list[float] = []
        failure = None
        steps = self._run_steps(spec, step_admitted)
        hard_timeout = self._hard_timeouts.get(spec.task_id)
        try:
            if hard_timeout is None:
                await steps
            else:
                await self._within(steps, hard_timeout)
        except RuntimeFailure as e:
            failure = e.kind
        self._admitted_tasks.pop(spec.task_id, None)
        self._retained.pop(spec.task_id, None)
        self._record_kv()
        self._recorder.task(TaskRecord(
            task_id=spec.task_id, task_class=spec.task_class, arrived_s=arrived,
            step_admitted_s=step_admitted, completed_s=self._clock.now(), failure=failure))
        self._schedule()

    async def _run_steps(self, spec: AgentTaskSpec, step_admitted: list[float]) -> None:
        """The agent task's agent steps; the first failed model request raises."""
        prompt = list(spec.initial_prompt)
        for k, step in enumerate(spec.steps):
            request = ModelRequest(tuple(prompt), step.n_predict)
            self._recorder.prompt(spec.task_id, k, request.prompt_tokens)
            output = await self._model_request(spec.task_id, k, request, step_admitted)
            prompt += output
            prompt += step.tool_result_tokens
            await self._clock.sleep(step.tool_delay_s)

    async def _within(self, steps, seconds: float) -> None:
        """Run ``steps`` under a hard timeout: past it, cancel them (the model request
        they wait on is recorded as failed) and raise ``RequestTimeout``."""
        work = asyncio.ensure_future(steps)
        timer = asyncio.ensure_future(self._clock.sleep(seconds))
        await asyncio.wait({work, timer}, return_when=asyncio.FIRST_COMPLETED)
        if work.done():
            timer.cancel()
            return work.result()
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        raise RequestTimeout(f"agent task exceeded its hard timeout ({seconds:.3f} s)")

    async def _model_request(self, task_id: int, step: int, request: ModelRequest,
                             step_admitted: list[float]) -> tuple[int, ...]:
        ref = RequestRef(task_id, step)
        arrived = self._clock.now()
        waiting = _Waiting(
            PendingRequest(ref, arrived, len(request.prompt_tokens), request.n_predict),
            asyncio.get_running_loop().create_future())
        self._waiting.append(waiting)
        self._schedule()
        try:
            slot = await waiting.admitted
        except asyncio.CancelledError:  # hard timeout while queued
            if not waiting.admitted.cancelled():  # admitted in this same instant, never sent
                del self._active[ref]
                self._record_kv()
                self._schedule()
            else:
                self._waiting.remove(waiting)
            self._record_request(ref, arrived, None, request, None, error=RequestTimeout.kind)
            raise
        admitted = self._clock.now()
        step_admitted.append(admitted)
        pinned = ModelRequest(request.prompt_tokens, request.n_predict, slot)
        completion = None
        try:
            completion = await self._runtime.submit(pinned)
        except RuntimeFailure as e:
            self._record_request(ref, arrived, admitted, request, slot, error=e.kind)
            raise
        except asyncio.CancelledError:  # hard timeout while in flight
            self._record_request(ref, arrived, admitted, request, slot, error=RequestTimeout.kind)
            raise
        finally:
            del self._active[ref]
            if completion is not None and slot is not None:
                self._retained[task_id] = (slot, len(request.prompt_tokens)
                                           + len(completion.output_tokens))
            self._record_kv()
            self._schedule()
        self._record_request(ref, arrived, admitted, request, completion.slot,
                             first_token_s=self._clock.now() - completion.decode_s,
                             n_cached=completion.n_cached_tokens,
                             n_output=len(completion.output_tokens))
        return completion.output_tokens

    def _schedule(self) -> None:
        if not self._waiting:
            return
        decision_started = time.perf_counter()  # policy overhead: wall time, not the run clock
        state = ProxyState(
            now_s=self._clock.now(),
            pending=tuple(w.pending for w in self._waiting),
            active=tuple(self._active.values()),
            admitted_tasks=dict(self._admitted_tasks),
            memory_budget_tokens=self._memory_budget_tokens,
            n_slots=self._n_slots,
        )
        admissions = self._policy.decide(state)
        self._recorder.decision(t_s=state.now_s, decision_s=time.perf_counter() - decision_started,
                                n_pending=len(state.pending), n_admitted=len(admissions))
        by_ref = {w.pending.ref: w for w in self._waiting}
        for admission in admissions:
            waiting = by_ref.pop(admission.ref)
            self._waiting.remove(waiting)
            p = waiting.pending
            self._active[p.ref] = ActiveRequest(p.ref, admission.slot, p.n_prompt_tokens, p.n_predict)
            self._admitted_tasks[p.ref.task_id] = admission.slot
            # its context is now part of the model request in flight; a slot given
            # to it no longer holds another agent task's retained context
            self._retained.pop(p.ref.task_id, None)
            if admission.slot is not None:
                self._retained = {t: r for t, r in self._retained.items()
                                  if r[0] != admission.slot}
            waiting.admitted.set_result(admission.slot)
        if admissions:
            self._record_kv()

    def _record_kv(self) -> None:
        """The admitted token footprint, if it (or the runtime's occupancy) changed."""
        in_flight = sum(a.n_prompt_tokens + a.n_predict for a in self._active.values())
        retained = sum(n for _, n in self._retained.values())
        kv = {"admitted_tokens": in_flight + retained, "in_flight_tokens": in_flight,
              "retained_tokens": retained,
              "runtime_kv_tokens": None if self._kv_probe is None else self._kv_probe()}
        if kv != self._last_kv:
            self._last_kv = kv
            self._recorder.kv(t_s=self._clock.now(), **kv)

    def _record_request(self, ref: RequestRef, arrived: float, admitted: float | None,
                        request: ModelRequest, slot: int | None, *, error: str | None = None,
                        first_token_s: float | None = None, n_cached: int | None = None,
                        n_output: int = 0) -> None:
        self._recorder.request(RequestRecord(
            task_id=ref.task_id, step=ref.step, arrived_s=arrived, admitted_s=admitted,
            first_token_s=first_token_s, completed_s=self._clock.now(),
            n_prompt_tokens=len(request.prompt_tokens), n_cached_tokens=n_cached,
            n_predict=request.n_predict,
            n_output_tokens=n_output, slot=slot, error=error))
