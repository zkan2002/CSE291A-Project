"""Admission-policy interface and the Phase 1 baselines.

A policy is a pure decision over a snapshot of proxy state: given the pending
model requests (in proxy arrival order), the active ones, the agent tasks that
currently hold admission, the memory budget and the slot count, it returns
which pending model requests to admit now and on which slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class RequestRef:
    task_id: int
    step: int


@dataclass(frozen=True)
class PendingRequest:
    ref: RequestRef
    arrived_s: float
    n_prompt_tokens: int
    n_predict: int


@dataclass(frozen=True)
class ActiveRequest:
    ref: RequestRef
    slot: int | None
    n_prompt_tokens: int
    n_predict: int


@dataclass(frozen=True)
class ProxyState:
    now_s: float
    pending: tuple[PendingRequest, ...]
    active: tuple[ActiveRequest, ...]
    admitted_tasks: Mapping[int, int | None]
    """Agent tasks holding admission (first model request admitted, not yet
    finished), mapped to the slot they were admitted on."""
    memory_budget_tokens: int
    n_slots: int


@dataclass(frozen=True)
class Admission:
    ref: RequestRef
    slot: int | None
    """Slot to pin the model request to (``id_slot``), or None to let the runtime choose."""


class AdmissionPolicy(Protocol):
    def decide(self, state: ProxyState) -> list[Admission]: ...


class TaskLevelFifo:
    """Task-level admission: an agent task holds one of ``concurrency`` slots
    from its first admission until it finishes, tool delays included. Waiting
    agent tasks are admitted in arrival order."""

    def __init__(self, concurrency: int) -> None:
        self.concurrency = concurrency

    def decide(self, state: ProxyState) -> list[Admission]:
        held = dict(state.admitted_tasks)
        admissions = []
        for req in state.pending:
            task = req.ref.task_id
            if task not in held:
                if len(held) >= self.concurrency:
                    continue
                held[task] = _unpinned_slot(held.values(), state.n_slots)
            admissions.append(Admission(req.ref, held[task]))
        return admissions


class RequestLevelFifo:
    """Request-level admission: each model request is admitted on its own, in
    arrival order, with at most ``concurrency`` in flight. After its tool delay
    an agent task's next step joins the back of the queue."""

    def __init__(self, concurrency: int) -> None:
        self.concurrency = concurrency

    def decide(self, state: ProxyState) -> list[Admission]:
        free = self.concurrency - len(state.active)
        return [Admission(req.ref, None) for req in state.pending[:max(0, free)]]


def _unpinned_slot(pinned, n_slots: int) -> int | None:
    taken = set(pinned)
    return next((s for s in range(n_slots) if s not in taken), None)


def build_policy(name: str, *, concurrency: int) -> AdmissionPolicy:
    if name == "task_level_fifo":
        return TaskLevelFifo(concurrency)
    if name == "request_level_fifo":
        return RequestLevelFifo(concurrency)
    raise ValueError(f"unknown admission policy: {name!r}")
