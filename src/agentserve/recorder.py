"""Recorder: writes one run directory. The format is described in docs/run-directory.md."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_FILE = "config.json"
REQUESTS_FILE = "requests.jsonl"
TASKS_FILE = "tasks.jsonl"


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
    def __init__(self, run_dir: Path, config: dict[str, Any]) -> None:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        self._requests = open(run_dir / REQUESTS_FILE, "w", encoding="utf-8", newline="\n")
        self._tasks = open(run_dir / TASKS_FILE, "w", encoding="utf-8", newline="\n")

    def request(self, record: RequestRecord) -> None:
        _write_line(self._requests, record)

    def task(self, record: TaskRecord) -> None:
        _write_line(self._tasks, record)

    def close(self) -> None:
        self._requests.close()
        self._tasks.close()


def _write_line(f, record) -> None:
    f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    f.flush()
