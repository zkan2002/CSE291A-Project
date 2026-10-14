"""Sweep runner: expands a grid of configurations into runs, runs each one, and
resumes after an interruption by skipping the cells that are already complete.

    agentserve sweep SWEEP.toml --out SWEEP_DIR [--dry-run]

A sweep config has a ``[base]`` experiment config (the tables of an ``agentserve
run`` config, under ``base.``), ``[sweep]`` settings and one or more ``[[grid]]``
tables. Each grid lists values for the swept axes; a cell is one combination
of them plus a seed, and one run directory. Grids are unioned, so a cell named
by two grids runs once. Axes a grid leaves out keep the base config's value.

- ``policy``: admission policy (``policy.name``).
- ``concurrency``: N (``policy.concurrency``). The runtime's slot count follows
  N unless ``[sweep] slots`` fixes it.
- ``memory_budget``: KV-pool tokens (``server.memory_budget`` for a real run,
  ``runtime.memory_budget_tokens`` for a simulated one).
- ``arrival_rate``: agent tasks per second (``workload.arrival_rate_per_s``).
- ``cache_policy``: ``resident`` (``--no-cache-idle-slots``: retained context
  stays in the KV pool; agent tasks are pinned), ``offload``
  (``--cache-idle-slots``: idle context goes to the host-RAM prompt cache, sized
  by the base config; not pinned) or ``off`` (``--no-cache-prompt`` and no
  host-RAM cache).
- ``seeds`` (in ``[sweep]``, or per grid).

``SWEEP_DIR`` gets ``runs/<cell>/`` (one run directory per complete cell),
``sweep.json`` (the sweep config and its cells), ``sweep.jsonl`` (one line per
attempt at a cell: ``complete``, ``skipped``, ``failed`` or ``interrupted``) and
``unloaded-cache/`` (unloaded completion times, shared between cells). A cell
runs into ``runs/<cell>.partial/`` and is renamed when it completes, so a
directory without the suffix is always a complete run; an attempt that failed
or was interrupted stays behind as ``<cell>.<k>.partial`` and is retried on the
next invocation. ``agentserve report`` ignores ``.partial`` directories.

Every run starts its own llama-server processes and stops them (see
``agentserve.real_experiment``), so no cached state carries over between
cells. For a real sweep the runner also checks before and after every cell that
no server answers on the configured port.
"""

from __future__ import annotations

import copy
import datetime
import itertools
import json
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from agentserve import config as cfg
from agentserve.experiment import run_experiment

AXES = ("policy", "concurrency", "memory_budget", "arrival_rate", "cache_policy")
RUNS_DIR = "runs"
PARTIAL_SUFFIX = ".partial"
SWEEP_FILE = "sweep.json"
SWEEP_LOG_FILE = "sweep.jsonl"
UNLOADED_CACHE_DIR = "unloaded-cache"


def load_sweep(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        sweep = tomllib.load(f)
    unknown = set(sweep) - {"sweep", "base", "grid"}
    if unknown:
        raise ValueError(f"unknown sweep tables: {sorted(unknown)}")
    return sweep


@dataclass(frozen=True)
class Cell:
    name: str
    config: dict[str, Any]

    @property
    def is_real(self) -> bool:
        return cfg.is_real(self.config)


def expand(sweep: dict[str, Any]) -> list[Cell]:
    """Every cell of the sweep, in grid order (seeds innermost), without duplicates."""
    base, settings = sweep["base"], sweep.get("sweep", {})
    cells: dict[str, Cell] = {}
    for grid in sweep.get("grid", [{}]):
        unknown = set(grid) - set(AXES) - {"seeds"}
        if unknown:
            raise ValueError(f"unknown grid axes: {sorted(unknown)} (expected {AXES})")
        values = [grid.get(axis, [None]) for axis in AXES]
        seeds = grid.get("seeds") or settings.get("seeds") or [base["run"]["seed"]]
        for combo in itertools.product(*values, seeds):
            *axes, seed = combo
            cell = _cell(base, dict(zip(AXES, axes)), seed, settings.get("slots", "concurrency"))
            if cell.name in cells and cells[cell.name].config != cell.config:
                raise ValueError(f"two grids give cell {cell.name} different configs")
            cells.setdefault(cell.name, cell)
    return list(cells.values())


def _cell(base: dict, axes: dict, seed: int, slots: int | str) -> Cell:
    c = copy.deepcopy(base)
    c["run"]["seed"] = seed
    if axes["policy"] is not None:
        c["policy"]["name"] = axes["policy"]
    if axes["concurrency"] is not None:
        c["policy"]["concurrency"] = axes["concurrency"]
    if axes["memory_budget"] is not None:
        cfg.set_memory_budget_tokens(c, axes["memory_budget"])
    if axes["arrival_rate"] is not None:
        c["workload"]["arrival_rate_per_s"] = axes["arrival_rate"]
    cfg.set_slots(c, c["policy"]["concurrency"] if slots == "concurrency" else slots)
    if axes["cache_policy"] is not None:
        cfg.set_cache_policy(c, axes["cache_policy"])
    name = (f"{c['policy']['name']}-n{c['policy']['concurrency']}-b{cfg.memory_budget_tokens(c)}"
            f"-r{c['workload']['arrival_rate_per_s']:g}-{cfg.cache_policy(c)}-s{seed}")
    return Cell(name, c)


@dataclass
class SweepResult:
    complete: list[str] = field(default_factory=list)
    """Cells run to completion by this invocation."""
    skipped: list[str] = field(default_factory=list)
    """Cells already complete from an earlier invocation."""
    failed: list[str] = field(default_factory=list)


RunFn = Callable[..., None]


def _print(line: str) -> None:
    print(line, flush=True)


def run_sweep(sweep: dict[str, Any], out_dir: Path, *, run: RunFn = run_experiment,
              log: Callable[[str], None] = _print) -> SweepResult:
    """Run every cell of ``sweep`` that ``out_dir`` doesn't already hold complete.

    A failed cell is recorded and the sweep goes on; an interruption (Ctrl-C) is
    recorded and re-raised. ``run(config, run_dir, unloaded_cache_dir=...)``
    runs one experiment (``agentserve.experiment.run_experiment``)."""
    cells = expand(sweep)
    runs_dir = out_dir / RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    _check_complete_cells(cells, runs_dir)
    (out_dir / SWEEP_FILE).write_text(json.dumps(
        {"sweep": sweep, "cells": [c.name for c in cells]}, indent=2, sort_keys=True) + "\n")
    result = SweepResult()
    with open(out_dir / SWEEP_LOG_FILE, "a", encoding="utf-8", newline="\n") as sweep_log:
        def record(cell: Cell, status: str, **fields) -> None:
            entry = {"cell": cell.name, "status": status,
                     "at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}
            sweep_log.write(json.dumps(entry, sort_keys=True) + "\n")
            sweep_log.flush()

        for i, cell in enumerate(cells, 1):
            final = runs_dir / cell.name
            if final.exists():
                result.skipped.append(cell.name)
                record(cell, "skipped")
                continue
            partial = _fresh_partial(runs_dir, cell.name)
            log(f"[{i}/{len(cells)}] {cell.name}")
            if cell.is_real:
                _check_no_server(cell, "before")
            started = time.perf_counter()
            try:
                run(cell.config, partial, unloaded_cache_dir=out_dir / UNLOADED_CACHE_DIR)
                if cell.is_real:
                    _check_no_server(cell, "after")
            except KeyboardInterrupt:
                record(cell, "interrupted", wall_s=round(time.perf_counter() - started, 3))
                raise
            except Exception as e:  # noqa: BLE001 - recorded, and the sweep goes on
                result.failed.append(cell.name)
                record(cell, "failed", wall_s=round(time.perf_counter() - started, 3),
                       error=f"{type(e).__name__}: {e}")
                log(f"  failed: {type(e).__name__}: {e}")
                continue
            partial.rename(final)
            result.complete.append(cell.name)
            record(cell, "complete", run_dir=str(final.relative_to(out_dir)),
                   wall_s=round(time.perf_counter() - started, 3),
                   fresh_servers=cell.is_real or None)
    return result


def cell_status(sweep: dict[str, Any], out_dir: Path) -> list[tuple[Cell, bool]]:
    """Each cell, and whether ``out_dir`` holds it complete."""
    return [(c, (out_dir / RUNS_DIR / c.name).exists()) for c in expand(sweep)]


def _check_complete_cells(cells: list[Cell], runs_dir: Path) -> None:
    for cell in cells:
        config_path = runs_dir / cell.name / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != cell.config:
            raise ValueError(
                f"{runs_dir / cell.name} was run with a config that differs from this "
                f"sweep's cell {cell.name}; sweep into a new directory instead")


def _fresh_partial(runs_dir: Path, name: str) -> Path:
    """Where to run ``name``; an earlier, unfinished attempt is kept aside."""
    partial = runs_dir / f"{name}{PARTIAL_SUFFIX}"
    if partial.exists():
        k = 1
        while (runs_dir / f"{name}.{k}{PARTIAL_SUFFIX}").exists():
            k += 1
        partial.rename(runs_dir / f"{name}.{k}{PARTIAL_SUFFIX}")
    return partial


def _check_no_server(cell: Cell, when: str) -> None:
    """No llama-server may answer on the cell's port outside its run: every run
    starts fresh servers and stops them."""
    server = cell.config["server"]
    url = f"http://{server.get('host', '127.0.0.1')}:{server.get('port', 8080)}/health"
    try:
        httpx.get(url, timeout=1.0)
    except httpx.HTTPError:
        return
    raise RuntimeError(f"a server is answering at {url} {when} cell {cell.name}; "
                       "stop it so that no cached state carries over between configurations")
