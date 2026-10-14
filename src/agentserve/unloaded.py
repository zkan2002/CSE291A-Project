"""Unloaded completion time, and the task SLO and hard timeout derived from it.

Each agent task's unloaded completion time is measured by running it alone on
an otherwise idle runtime, through the same proxy and runtime interface as the
experiment itself, so it works for any runtime and clock. Agent tasks are run
one after another, never overlapping, under task-level admission with N=1; the
measurement is therefore independent of the experiment's admission policy, and
every policy is judged against the same task SLOs.

- task SLO = ``SLO_MULTIPLE`` x unloaded completion time
- hard timeout = ``HARD_TIMEOUT_MULTIPLE`` x unloaded completion time

An agent task that fails even when run alone has no unloaded completion time,
task SLO or hard timeout (all ``None``) and its ``failure`` is recorded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from agentserve.clock import Clock
from agentserve.policy import TaskLevelFifo
from agentserve.proxy import Proxy
from agentserve.recorder import TaskRecord
from agentserve.runtime import Runtime
from agentserve.workload import AgentTaskSpec

SLO_MULTIPLE = 3.0
HARD_TIMEOUT_MULTIPLE = 10.0


@dataclass(frozen=True)
class UnloadedRecord:
    """One line of ``unloaded.jsonl``."""

    task_id: int
    task_class: str
    unloaded_completion_s: float | None
    slo_s: float | None
    hard_timeout_s: float | None
    failure: str | None

    @classmethod
    def from_measurement(cls, task_id: int, task_class: str, completion_s: float | None,
                         failure: str | None) -> UnloadedRecord:
        if failure is not None:
            completion_s = None
        return cls(task_id=task_id, task_class=task_class, unloaded_completion_s=completion_s,
                   slo_s=None if completion_s is None else SLO_MULTIPLE * completion_s,
                   hard_timeout_s=(None if completion_s is None
                                   else HARD_TIMEOUT_MULTIPLE * completion_s),
                   failure=failure)


def measure(tasks: list[AgentTaskSpec], *, clock: Clock, runtime: Runtime,
            memory_budget_tokens: int, n_slots: int,
            pin_slots: bool = True) -> list[UnloadedRecord]:
    """Run each agent task alone on ``runtime``, in task-ID order, on ``clock``,
    pinned to a slot or not as the cache policy has it (``pin_slots``)."""
    sink = _TaskSink()

    async def main() -> None:
        for spec in sorted(tasks, key=lambda t: t.task_id):
            proxy = Proxy(policy=TaskLevelFifo(concurrency=1, pin_slots=pin_slots),
                          runtime=runtime, clock=clock,
                          recorder=sink, memory_budget_tokens=memory_budget_tokens,
                          n_slots=n_slots)
            await proxy.run([replace(spec, arrival_s=clock.now())])

    clock.run(main())
    return [UnloadedRecord.from_measurement(t.task_id, t.task_class,
                                            t.completed_s - t.arrived_s, t.failure)
            for t in sink.tasks]


def cached(key: dict[str, Any], cache_dir: Path | None,
           measure_fn: Callable[[], list[UnloadedRecord]]) -> list[UnloadedRecord]:
    """``measure_fn()``, cached in ``cache_dir`` under ``key``: the inputs that
    determine the workload and the runtime (not the admission policy)."""
    if cache_dir is None:
        return measure_fn()
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    path = cache_dir / f"unloaded-{digest}.json"
    if path.exists():
        entry = json.loads(path.read_text())
        if entry["key"] == key:
            # Only the measurement is cached; task SLOs and hard timeouts are re-derived.
            return [UnloadedRecord.from_measurement(r["task_id"], r["task_class"],
                                                    r["unloaded_completion_s"], r["failure"])
                    for r in entry["records"]]
    records = measure_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    measured = [{"task_id": r.task_id, "task_class": r.task_class,
                 "unloaded_completion_s": r.unloaded_completion_s, "failure": r.failure}
                for r in records]
    path.write_text(json.dumps({"key": key, "records": measured}, indent=1, sort_keys=True)
                    + "\n")
    return records


class _TaskSink:
    """Recorder stand-in for measurement runs: keeps only the agent-task records."""

    def __init__(self) -> None:
        self.tasks: list[TaskRecord] = []

    def request(self, record) -> None:
        pass

    def prompt(self, task_id, step, prompt_tokens) -> None:
        pass

    def task(self, record: TaskRecord) -> None:
        self.tasks.append(record)

    def decision(self, **_) -> None:
        pass
