"""Recorder: writes one run directory. The format is described in docs/run-directory.md."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_FILE = "config.json"
REQUESTS_FILE = "requests.jsonl"
TASKS_FILE = "tasks.jsonl"
PROMPTS_FILE = "prompts.jsonl"
UNLOADED_FILE = "unloaded.jsonl"


@dataclass(frozen=True)
class RequestRecord:
    task_id: int
    step: int
    arrived_s: float
    admitted_s: float
    first_token_s: float | None
    completed_s: float
    n_prompt_tokens: int
    n_cached_tokens: int | None
    n_predict: int
    n_output_tokens: int
    slot: int | None
    error: str | None


@dataclass(frozen=True)
class TaskRecord:
    task_id: int
    task_class: str
    arrived_s: float
    step_admitted_s: list[float]
    completed_s: float
    failure: str | None


class Recorder:
    def __init__(self, run_dir: Path, config: dict[str, Any], *,
                 record_prompts: bool = False) -> None:
        run_dir.mkdir(parents=True, exist_ok=False)
        self._run_dir = run_dir
        (run_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        self._requests = _open(run_dir / REQUESTS_FILE)
        self._tasks = _open(run_dir / TASKS_FILE)
        self._prompts = _open(run_dir / PROMPTS_FILE) if record_prompts else None

    def request(self, record: RequestRecord) -> None:
        _write_line(self._requests, record)

    def task(self, record: TaskRecord) -> None:
        _write_line(self._tasks, record)

    def prompt(self, task_id: int, step: int, prompt_tokens: tuple[int, ...]) -> None:
        """The exact token array of one model request; a no-op unless prompts are recorded."""
        if self._prompts is not None:
            _write_line(self._prompts, {"task_id": task_id, "step": step,
                                        "prompt_tokens": list(prompt_tokens)})

    def unloaded(self, records) -> None:
        """Each agent task's unloaded completion time, task SLO and hard timeout
        (``agentserve.unloaded.UnloadedRecord``), one line per agent task."""
        with _open(self._run_dir / UNLOADED_FILE) as f:
            for record in records:
                _write_line(f, record)

    def close(self) -> None:
        self._requests.close()
        self._tasks.close()
        if self._prompts is not None:
            self._prompts.close()


def _open(path: Path):
    return open(path, "w", encoding="utf-8", newline="\n")


def _write_line(f, record) -> None:
    line = record if isinstance(record, dict) else asdict(record)
    f.write(json.dumps(line, sort_keys=True) + "\n")
    f.flush()
