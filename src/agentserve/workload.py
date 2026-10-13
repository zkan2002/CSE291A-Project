"""Workload generator: a pure function of (workload config, duration, seed, split).

It returns agent-task specs. Prompts are token-ID arrays; the prompt of agent
step k+1 is built by the proxy as prompt_k + output_k + tool_result_k, so each
prompt is an exact prefix-extension of the previous one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Token IDs drawn for synthetic context; kept clear of low special-token IDs.
_TOKEN_LOW, _TOKEN_HIGH = 100, 32_000


@dataclass(frozen=True)
class TaskClass:
    name: str
    weight: float
    initial_context_tokens: int
    output_tokens: int
    """``n_predict`` of every agent step (EOS ignored, so outputs have exactly this length)."""
    steps: tuple[int, int]
    """Inclusive range; each agent task draws its number of agent steps uniformly from it."""
    tool_result_tokens: int = 200
    tool_delay_median_s: float = 1.0
    tool_delay_sigma: float = 0.5
    """Log-normal shape: the tool delay is ``median * exp(sigma * Z)``."""
    eval_weight: float | None = None
    """Held-out mix weight used for evaluation workloads; ``None`` means ``weight``."""

    def weight_for(self, split: str) -> float:
        if split == "eval" and self.eval_weight is not None:
            return self.eval_weight
        return self.weight

    @classmethod
    def from_config(cls, c: dict) -> TaskClass:
        c = dict(c)
        steps = c.pop("steps")
        c["steps"] = (steps, steps) if isinstance(steps, int) else (steps[0], steps[1])
        return cls(**c)


ARRIVAL_MODES = ("poisson", "on_off")


@dataclass(frozen=True)
class WorkloadConfig:
    arrival_rate_per_s: float
    """Mean agent-task arrival rate, in every arrival mode."""
    classes: tuple[TaskClass, ...]
    arrival_mode: str = "poisson"
    """``poisson``: open-loop Poisson arrivals. ``on_off``: bursty arrivals that
    alternate between on periods (Poisson at ``rate * (on + off) / on``) and
    silent off periods, with exponentially distributed period lengths of mean
    ``burst_on_s`` and ``burst_off_s``; the mean rate is unchanged."""
    burst_on_s: float = 10.0
    burst_off_s: float = 30.0

    def __post_init__(self) -> None:
        if self.arrival_mode not in ARRIVAL_MODES:
            raise ValueError(f"unknown arrival mode: {self.arrival_mode!r}")

    @classmethod
    def from_config(cls, c: dict) -> WorkloadConfig:
        c = dict(c)
        c["classes"] = tuple(TaskClass.from_config(k) for k in c["classes"])
        return cls(**c)


@dataclass(frozen=True)
class AgentStepSpec:
    n_predict: int
    tool_result_tokens: tuple[int, ...]
    """Appended to the context after this step; empty for the last step."""
    tool_delay_s: float
    """Tool delay after this step; 0 for the last step."""


@dataclass(frozen=True)
class AgentTaskSpec:
    task_id: int
    task_class: str
    arrival_s: float
    initial_prompt: tuple[int, ...]
    steps: tuple[AgentStepSpec, ...]


SPLITS = ("dev", "eval")


def generate(config: WorkloadConfig, *, duration_s: float, seed: int,
             split: str = "dev") -> list[AgentTaskSpec]:
    """Open-loop agent-task arrivals over ``[0, duration_s)``.

    ``split`` is ``dev`` (development: tuning) or ``eval`` (evaluation). The
    random streams are keyed by split as well as seed, so a development
    workload and an evaluation workload never share a seed, whatever seed
    numbers they are given; evaluation workloads also use each class's held-out
    ``eval_weight``. Arrival times and task contents come from separate
    streams, so the arrival process does not shift when the task mix changes.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r} (expected one of {SPLITS})")
    arrivals = random.Random(f"agentserve/{split}/{seed}/arrivals")
    rng = random.Random(f"agentserve/{split}/{seed}/tasks")
    weights = [c.weight_for(split) for c in config.classes]
    tasks: list[AgentTaskSpec] = []
    for t in _arrival_times(config, arrivals, duration_s):
        cls = rng.choices(config.classes, weights=weights)[0]
        tasks.append(AgentTaskSpec(
            task_id=len(tasks),
            task_class=cls.name,
            arrival_s=t,
            initial_prompt=_tokens(rng, cls.initial_context_tokens),
            steps=_steps(rng, cls),
        ))
    return tasks


def _arrival_times(config: WorkloadConfig, rng: random.Random, duration_s: float) -> list[float]:
    rate = config.arrival_rate_per_s
    if config.arrival_mode == "poisson":
        return _poisson(rng, rate, 0.0, duration_s)
    on, off = config.burst_on_s, config.burst_off_s
    on_rate = rate * (on + off) / on
    times: list[float] = []
    t = 0.0
    is_on = rng.random() < on / (on + off)  # start in the stationary state
    while t < duration_s:
        end = min(duration_s, t + rng.expovariate(1.0 / (on if is_on else off)))
        if is_on:
            times += _poisson(rng, on_rate, t, end)
        t, is_on = end, not is_on
    return times


def _poisson(rng: random.Random, rate: float, start: float, end: float) -> list[float]:
    times = []
    t = start + rng.expovariate(rate)
    while t < end:
        times.append(t)
        t += rng.expovariate(rate)
    return times


def _steps(rng: random.Random, cls: TaskClass) -> tuple[AgentStepSpec, ...]:
    n = rng.randint(*cls.steps)
    return tuple(_step(rng, cls, last=(k == n - 1)) for k in range(n))


def _step(rng: random.Random, cls: TaskClass, *, last: bool) -> AgentStepSpec:
    if last:
        return AgentStepSpec(n_predict=cls.output_tokens, tool_result_tokens=(), tool_delay_s=0.0)
    delay = rng.lognormvariate(math.log(cls.tool_delay_median_s), cls.tool_delay_sigma)
    return AgentStepSpec(n_predict=cls.output_tokens,
                         tool_result_tokens=_tokens(rng, cls.tool_result_tokens),
                         tool_delay_s=delay)


def _tokens(rng: random.Random, n: int) -> tuple[int, ...]:
    return tuple(rng.randrange(_TOKEN_LOW, _TOKEN_HIGH) for _ in range(n))
