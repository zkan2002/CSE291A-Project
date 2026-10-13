# Workload

The workload generator (`agentserve.workload`) is a pure function of the
`[workload]` config, the run's `duration_s`, `seed` and `split`. The same inputs
always give the same agent tasks, token for token.

## What an agent task looks like

- **Task class:** starting context length, output length per agent step
  (`n_predict`, EOS ignored, so every output has exactly that length), and a
  range of agent steps. Each agent task draws its step count uniformly from the
  range.
- **Prompts** are token-ID arrays. Agent step k+1's prompt is agent step k's
  prompt, then step k's output, then a synthetic tool result
  (`tool_result_tokens`, default 200, drawn from the seeded stream). So every
  prompt is an exact prefix-extension of the previous one, and the context grows
  by `n_predict + tool_result_tokens` per agent step.
- **Tool delays** are log-normal: `median * exp(sigma * Z)`, defaults median 1 s,
  sigma 0.5.
- **Arrivals** are open-loop. `arrival_mode = "poisson"` (default) or `"on_off"`:
  on periods of Poisson arrivals at `rate * (on + off) / on`, separated by silent
  off periods, with exponential period lengths of mean `burst_on_s` /
  `burst_off_s`. The mean rate is the same in both modes.
- **Splits:** `split = "dev"` (tuning) or `"eval"` (evaluation). The random
  streams are keyed by split and seed, so a development workload never shares a
  seed with an evaluation workload. Evaluation workloads use each class's
  held-out `eval_weight` where one is set.

Set `record_prompts = true` under `[run]` to write every prompt's token array to
`prompts.jsonl` (see `docs/run-directory.md`). It is off by default because the
file is large.

## The default mix (`mix = "default"`)

| | Starting context | Output per agent step | Agent steps |
| --- | --- | --- | --- |
| short / long | 500 tokens | 50 tokens | 2–3 (few) |
|  | 4,000 tokens | 400 tokens | 8–12 (many) |

All 2 × 2 × 2 = 8 combinations, named like `long_ctx-short_out-many_steps`.
Development weights are uniform. The held-out evaluation weighting doubles the
four many-step classes. Each agent step adds 200 tool-result tokens, so the
final context of an agent task ranges from about 750 tokens
(`short_ctx-short_out`, 2 steps) to about 10,600 tokens (`long_ctx-long_out`,
12 steps: 4,000 + 11 × 600).

## Justification against public trajectories

**Source checked:** the Hugging Face dataset
[`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories).
It is public and ungated, with 80,036 SWE-agent trajectories on SWE-bench-style
Python repositories (one `train` split, about 1.1 GB of parquet). Each
trajectory is a system prompt, the task as the first user message, then
alternating `ai` (model output) and `user` (tool observation) messages, which
maps directly onto agent steps and tool results.

**Method:** `scripts/trajectory_stats.py` reads 500 trajectories (20 pages of
25 at seeded random offsets) through the datasets-server `/rows` API and counts
tokens with tiktoken `cl100k_base`. This stands in for the Llama 3 tokenizer,
which is gated, so the counts are approximate. Re-run with
`python -m uv run --with tiktoken python scripts/trajectory_stats.py`.

**Results** (sample drawn 2026-10-13; 463 of the 500 are from the 70B model):

| Quantity | p10 | p25 | median | p75 | p90 | mean |
| --- | --- | --- | --- | --- | --- | --- |
| Agent steps per trajectory | 7 | 10 | 16 | 30 | 48 | 26 |
| Starting context (system + task), tokens | 1,854 | 1,896 | 1,945 | 2,070 | 2,421 | 2,038 |
| Output per agent step, tokens | 43 | 54 | 79 | 119 | 178 | 101 |
| Tool result per agent step, tokens | 33 | 48 | 115 | 703 | 1,053 | 372 |

At least 135 of the 500 trajectories ended at the model's context limit (exit
status `exit_context` or `submitted (exit_context)`), so the step counts are
truncated from above.

**How the mix relates to these numbers:**

- **Starting context:** 500 and 4,000 bracket the observed range (1,854–2,421
  from p10 to p90). The short class stands for agents with lighter system
  prompts than SWE-agent; the long class is a heavier-than-typical prompt that
  puts real pressure on the memory budget.
- **Output per agent step:** 50 tokens is near the observed p25 (54). 400 is
  above the observed p90 (178), on purpose, as a decode-heavy corner (for
  example agents that write whole files per step).
- **Tool result:** 200 tokens per agent step sits between the observed median
  (115) and mean (372). The distribution is heavy-tailed, and a fixed size keeps
  context growth exact. The observed context growth per agent step is about 470
  tokens on average (101 output + 372 tool result). The mix grows by 250
  (short output) or 600 (long output).
- **Agent steps:** 8–12 for "many" covers roughly p10–p25 of the observed
  distribution. The median (16) and tail (48 at p90) are longer. We cap at 12 to
  keep one agent task's final context (at most about 10.6k tokens) within the
  16k and 32k memory budgets (ADR-0001), so that memory pressure comes from
  concurrency, not from single agent tasks. **Caveat:** at the 8k budget, the
  longest `long_ctx` many-step agent tasks outgrow the whole pool on their own.
  Sweeps at 8k must either accept those as failed tasks or drop those classes.
  "Few" (2–3) is below anything in this SWE-agent sample. It stands for short
  tool-calling agents, not coding agents.
- **Held-out weighting:** the evaluation weighting doubles the many-step classes,
  moving the mix toward the long trajectories that dominate this dataset. This
  means a policy tuned on the uniform development mix is evaluated on a
  different and more memory-heavy mix.

**What this does not justify:** tool delays. The dataset records no timings, so
the 1 s log-normal median is an assumption that the sweeps can vary.
