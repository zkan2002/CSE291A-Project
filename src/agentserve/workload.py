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
        """Build from the ``[workload]`` table of an experiment config.

        The task classes are either ``[[workload.classes]]`` or ``mix =
        "default"`` (the 8-class ``DEFAULT_MIX``). ``tool_result_tokens``,
        ``tool_delay_median_s`` and ``tool_delay_sigma`` set at workload level
        apply to every class that does not set its own.
        """
        c = dict(c)
        mix = c.pop("mix", None)
        shared = {k: c.pop(k) for k in _CLASS_DEFAULT_KEYS if k in c}
        if mix is not None:
            if "classes" in c:
                raise ValueError("give either a mix or [[workload.classes]], not both")
            if mix not in MIXES:
                raise ValueError(f"unknown mix: {mix!r} (expected one of {sorted(MIXES)})")
            raw_classes = MIXES[mix]
        else:
            raw_classes = c["classes"]
        c["classes"] = tuple(TaskClass.from_config({**shared, **k}) for k in raw_classes)
        return cls(**c)


_CLASS_DEFAULT_KEYS = ("tool_result_tokens", "tool_delay_median_s", "tool_delay_sigma")


def _default_mix() -> tuple[dict, ...]:
    """Short/long starting context x short/long output per agent step x
    few/many agent steps. Development weights are uniform; the held-out
    evaluation weighting doubles the many-step classes. The values are
    justified in docs/workload.md."""
    classes = []
    for ctx_name, ctx in (("short_ctx", 500), ("long_ctx", 4000)):
        for out_name, out in (("short_out", 50), ("long_out", 400)):
            for steps_name, steps in (("few_steps", [2, 3]), ("many_steps", [8, 12])):
                classes.append({
                    "name": f"{ctx_name}-{out_name}-{steps_name}",
                    "weight": 1.0,
                    "eval_weight": 2.0 if steps_name == "many_steps" else 1.0,
                    "initial_context_tokens": ctx,
                    "output_tokens": out,
                    "steps": steps,
                })
    return tuple(classes)


DEFAULT_MIX = _default_mix()
MIXES = {"default": DEFAULT_MIX}


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
