"""Stage A2's N comes from a fixed rule: highest mean goodput at the tuning rate, resident."""

import tomllib

from agentserve.best_n import best_n, grids


def _row(policy, n, budget, goodput, rate=0.1, cache="resident"):
    return {"policy": policy, "concurrency": str(n), "memory_budget_tokens": str(budget),
            "arrival_rate_per_s": str(rate), "cache_policy": cache,
            "goodput_tasks_per_min_mean": str(goodput)}


def test_picks_the_highest_mean_goodput_per_baseline_and_budget():
    rows = [
        _row("task_level_fifo", 4, 16384, 2.73), _row("task_level_fifo", 8, 16384, 2.58),
        _row("task_level_fifo", 8, 32768, 2.70), _row("task_level_fifo", 16, 32768, 3.1),
        _row("request_level_fifo", 8, 16384, 2.58), _row("request_level_fifo", 16, 16384, 1.2),
    ]
    assert best_n(rows) == {("task_level_fifo", 16384): 4, ("task_level_fifo", 32768): 16,
                            ("request_level_fifo", 16384): 8}


def test_ignores_other_rates_cache_policies_and_nan():
    rows = [
        _row("task_level_fifo", 4, 16384, 1.0),
        _row("task_level_fifo", 8, 16384, 9.0, rate=0.05),
        _row("task_level_fifo", 2, 16384, 9.0, cache="off"),
        _row("task_level_fifo", 16, 16384, float("nan")),
    ]
    assert best_n(rows) == {("task_level_fifo", 16384): 4}


def test_ties_go_to_the_smaller_n():
    rows = [_row("task_level_fifo", 8, 16384, 2.0), _row("task_level_fifo", 4, 16384, 2.0)]
    assert best_n(rows) == {("task_level_fifo", 16384): 4}


def test_grids_are_valid_sweep_toml_for_the_requested_budgets():
    text = grids({("task_level_fifo", 16384): 4, ("request_level_fifo", 8192): 16}, [0.025, 0.05],
                 budgets=[16384, 32768])
    parsed = tomllib.loads(text)["grid"]
    assert parsed == [{"policy": ["task_level_fifo"], "concurrency": [4], "memory_budget": [16384],
                       "arrival_rate": [0.025, 0.05], "cache_policy": ["resident"]}]
