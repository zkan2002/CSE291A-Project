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
    finished), mapped to the slot their latest model request was admitted on."""
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
    agent tasks are admitted in arrival order. With ``pin_slots`` every model
    request of an agent task is pinned (``id_slot``) to the slot it was admitted
    on, while there are slots to go round."""

    def __init__(self, concurrency: int, *, pin_slots: bool = True) -> None:
        self.concurrency = concurrency
        self.pin_slots = pin_slots

    def decide(self, state: ProxyState) -> list[Admission]:
        held = dict(state.admitted_tasks)
        admissions = []
        for req in state.pending:
            task = req.ref.task_id
            if task not in held:
                if len(held) >= self.concurrency:
                    continue
                held[task] = (_unpinned_slot(held.values(), state.n_slots)
                              if self.pin_slots else None)
            admissions.append(Admission(req.ref, held[task]))
        return admissions


class RequestLevelFifo:
    """Request-level admission: each model request is admitted on its own, in
    arrival order, with at most ``concurrency`` in flight. After its tool delay
    an agent task's next step joins the back of the queue.

    With ``pin_slots`` each model request is pinned (``id_slot``) where possible:
    to the slot of its agent task's previous step if that slot is free, else to
    a free slot no other agent task under way was last on (so as not to
    overwrite retained context), else to any free slot. With no slot free
    (``concurrency`` above the slot count) the runtime chooses."""

    def __init__(self, concurrency: int, *, pin_slots: bool = True) -> None:
        self.concurrency = concurrency
        self.pin_slots = pin_slots

    def decide(self, state: ProxyState) -> list[Admission]:
        free = max(0, self.concurrency - len(state.active))
        admit = state.pending[:free]
        if not self.pin_slots:
            return [Admission(req.ref, None) for req in admit]
        busy = {a.slot for a in state.active}
        last = dict(state.admitted_tasks)
        admissions = []
        for req in admit:
            task = req.ref.task_id
            slot = last.get(task)
            if slot is None or slot in busy:
                others = {s for t, s in last.items() if t != task}
                idle = [s for s in range(state.n_slots) if s not in busy]
                slot = next((s for s in idle if s not in others), idle[0] if idle else None)
            if slot is not None:
                busy.add(slot)
                last[task] = slot
            admissions.append(Admission(req.ref, slot))
        return admissions


def _unpinned_slot(pinned, n_slots: int) -> int | None:
    taken = set(pinned)
    return next((s for s in range(n_slots) if s not in taken), None)


def build_policy(name: str, *, concurrency: int, pin_slots: bool = True) -> AdmissionPolicy:
    if name == "task_level_fifo":
        return TaskLevelFifo(concurrency, pin_slots=pin_slots)
    if name == "request_level_fifo":
        return RequestLevelFifo(concurrency, pin_slots=pin_slots)
    raise ValueError(f"unknown admission policy: {name!r}")
