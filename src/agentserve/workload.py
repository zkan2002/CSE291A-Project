"""Workload generator: a pure function of (workload config, duration, seed).

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

    @classmethod
    def from_config(cls, c: dict) -> TaskClass:
        c = dict(c)
        steps = c.pop("steps")
        c["steps"] = (steps, steps) if isinstance(steps, int) else (steps[0], steps[1])
        return cls(**c)


@dataclass(frozen=True)
class WorkloadConfig:
    arrival_rate_per_s: float
    classes: tuple[TaskClass, ...]


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


def generate(config: WorkloadConfig, *, duration_s: float, seed: int) -> list[AgentTaskSpec]:
    """Open-loop Poisson arrivals over ``[0, duration_s)``."""
    rng = random.Random(seed)
    weights = [c.weight for c in config.classes]
    tasks: list[AgentTaskSpec] = []
    t = rng.expovariate(config.arrival_rate_per_s)
    while t < duration_s:
        cls = rng.choices(config.classes, weights=weights)[0]
        tasks.append(AgentTaskSpec(
            task_id=len(tasks),
            task_class=cls.name,
            arrival_s=t,
            initial_prompt=_tokens(rng, cls.initial_context_tokens),
            steps=_steps(rng, cls),
        ))
        t += rng.expovariate(config.arrival_rate_per_s)
    return tasks


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
