# Run directory format

`agentserve run CONFIG --out RUN_DIR` creates `RUN_DIR` (it must not exist yet) and writes:

| File | Contents |
| --- | --- |
| `config.json` | The full resolved config the run used (keys sorted). |
| `requests.jsonl` | One line per model request, written when it completes or fails. |
| `tasks.jsonl` | One line per agent task, written when it completes or fails. |
| `unloaded.jsonl` | One line per agent task: its unloaded completion time, task SLO and hard timeout. Written before the experiment starts. |
| `prompts.jsonl` | Only with `record_prompts = true` under `[run]`: one line per model request, written when it is issued. |
| `decisions.jsonl` | One line per admission-policy decision, with its wall-clock duration (policy overhead). Not reproducible byte for byte. |

A **real run** (`[runtime] kind = "llama_server"`) writes the same files in the same format, plus:

| File | Contents |
| --- | --- |
| `samples.jsonl` | One system sample every `sample_interval_s` (default 0.1 s) while the experiment runs. |
| `metadata.json` | What the run was measured on: llama.cpp build, model hash, GPU, driver and CUDA versions, harness commit, launch arguments. |
| `server.log` | The experiment's llama-server output. |
| `unloaded-server.log` | The llama-server output while unloaded completion times were measured (absent when they came from `--unloaded-cache`). |

All times are seconds since the experiment started, on the run's clock (virtual for the simulated runtime, `perf_counter` wall time for a real run). Lines are JSON objects with sorted keys, so the same config and seed give byte-identical files on the virtual clock. A simulated run never writes the real-run files; the only wall-time measurement in its run directory is `decisions.jsonl`, which is therefore not byte-identical across repeats (every other file is). Nothing is dropped: a failed model request or failed task still gets its line.

## Admission policy and failure accounting

```toml
[policy]
name = "request_level_fifo"  # or "task_level_fifo"
concurrency = 4              # N
pin_slots = true             # optional; default: follow the cache policy
```

- `task_level_fifo` (task-level admission): an agent task holds one of N places from its first admission until it finishes, tool delays included.
- `request_level_fifo` (request-level admission): each model request queues FIFO with at most N in flight; after its tool delay an agent task's next step joins the back of the queue.

**Pinning.** With `pin_slots`, model requests are pinned to a slot with `id_slot` where possible: task-level pins each agent task to the slot it was admitted on; request-level pins each step to the slot of its task's previous step if that slot is free, else to a free slot no other agent task under way was last on, else to any free slot (with none free, the runtime chooses). By default pinning follows the cache policy (pilot amendment 6): **resident** (`cache_idle_slots = false`) pins, **offload** (`cache_idle_slots = true`) does not, so offloaded context comes back through the host-RAM prompt cache. The flag is read from `[server]` in a real run and `[runtime]` in a simulated one; the unloaded measurement pins the same way.

**Failures.** The proxy never retries a failed model request: the first error ends the agent task as a failed task with that error as its `failure`. An agent task still running `hard_timeout_s` (from `unloaded.jsonl`) after it arrived fails with `timeout`: the model request it waits on is cancelled and recorded with `error = "timeout"` (`admitted_s = null` if it was still queued for admission); one in a tool delay simply ends. A task without a hard timeout (it failed even when run alone) has none. Failed tasks stay in `tasks.jsonl` and in every denominator.

## Real runs

`agentserve run configs/real-smoke.toml --out RUN_DIR` runs the experiment against the pinned llama-server. The config has a `[server]` table in the launcher's format (`agentserve.server`; the memory budget and slot count come from there) and:

```toml
[runtime]
kind = "llama_server"
request_timeout_s = 600   # HTTP timeout per model request; recorded as `timeout`
sample_interval_s = 0.1
```

The run (1) writes `metadata.json`; (2) measures unloaded completion times on a fresh server, unless cached; (3) launches a **fresh** server for the experiment, so nothing from (2) is retained in its KV pool or prompt cache; (4) runs the proxy and the sampler on the real clock; (5) stops the server, even on failure.

The adapter (`agentserve.llama_runtime`) sends each model request to `/completion` as a token array with `n_predict`, `ignore_eos`, `id_slot` (−1 when the policy leaves the slot to the server) and `cache_prompt`. `n_cached_tokens` is the server's `timings.cache_n`, `slot` its `id_slot`, and `first_token_s` uses its `timings.predicted_ms` as the decode time. Errors map to the typed errors found in the pilot ([`pilot/pilot-findings.md`](pilot/pilot-findings.md)):

| Server response | `error` |
| --- | --- |
| HTTP 500 "Context size has been exceeded." (overflow abort) | `pool_exhausted` |
| HTTP 400 `exceed_context_size_error` (prompt ≥ pool size) | `exceeds_context` |
| HTTP 200 with fewer than `n_predict` output tokens (prompt + `n_predict` > pool size, so the server truncates) | `exceeds_context` |
| HTTP client timeout | `timeout` |
| Anything else, including a lost connection | `server_error` |

## `requests.jsonl`

| Field | Meaning |
| --- | --- |
| `task_id`, `step` | The agent task and the index of its agent step. |
| `arrived_s` | When the model request reached the proxy. |
| `admitted_s` | When the admission policy admitted it and it was sent to the runtime; `null` if its agent task hit its hard timeout first. |
| `first_token_s` | Completion time minus the runtime's decode time; `null` on error. |
| `completed_s` | When the runtime returned a completion or an error (or the hard timeout cancelled it). |
| `n_prompt_tokens` | Prompt length in tokens. |
| `n_cached_tokens` | Prompt tokens served from retained context; `null` on error. |
| `n_predict` | Output tokens requested (EOS ignored, so a completion has exactly this many). |
| `n_output_tokens` | Generated tokens (0 on error). |
| `slot` | Slot the runtime used (or the pinned slot, on error; `null` if unpinned or never admitted). |
| `error` | `null`, or the typed error: `pool_exhausted`, `exceeds_context`, `server_error`, `timeout`. |

## `tasks.jsonl`

| Field | Meaning |
| --- | --- |
| `task_id`, `task_class` | The agent task and its task class. |
| `arrived_s` | Arrival time from the workload. |
| `step_admitted_s` | Admission time of each agent step's model request, in order. |
| `completed_s` | When the agent task finished (successfully or not). |
| `failure` | `null`, or the cause of the failed task (the error of its failing model request). |

An agent task is *active* from `step_admitted_s[0]` to `completed_s` (never, if it timed out before its first admission: `step_admitted_s` is then empty).

## `prompts.jsonl`

| Field | Meaning |
| --- | --- |
| `task_id`, `step` | The agent task and the index of its agent step. |
| `prompt_tokens` | The exact token-ID array sent as the model request's prompt. |

## `unloaded.jsonl`

Before the experiment, every agent task of the workload is run alone on an otherwise idle runtime (a fresh one, on its own clock), one after another, through the same proxy and runtime interface, under task-level admission with N=1 (pinned or not as the cache policy has it, without `pin_slots`'s override). The measurement does not depend on the run's admission policy, so every policy is judged against the same task SLOs.

| Field | Meaning |
| --- | --- |
| `task_id`, `task_class` | The agent task and its task class. |
| `unloaded_completion_s` | Completion minus arrival when run alone; `null` if it failed even then. |
| `slo_s` | Task SLO: 3 x `unloaded_completion_s` (`null` if that is `null`). |
| `hard_timeout_s` | Hard timeout: 10 x `unloaded_completion_s` (`null` if that is `null`). |
| `failure` | `null`, or the error that failed the agent task when run alone. |

`agentserve run ... --unloaded-cache DIR` caches the measurement in `DIR`, keyed by the workload, seed, duration, split and runtime config (not the policy), and reuses it on later runs with the same key. Without the flag, it is measured afresh for every run.

`agentserve analyze` reports *goodput* (agent tasks that did not fail and completed within `slo_s` of arriving, per minute over the same window as throughput) and *SLO attainment* (the fraction of all agent tasks, failed ones included, that count towards goodput).

## `decisions.jsonl`

The proxy times every call into the admission policy (building its state snapshot plus `decide`) with `perf_counter`, on every run, simulated or real.

| Field | Meaning |
| --- | --- |
| `t_s` | Run-clock time of the decision. |
| `decision_s` | Wall-clock seconds the decision took. |
| `n_pending` | Model requests waiting when the policy was asked. |
| `n_admitted` | How many of them it admitted. |

## `samples.jsonl` (real runs)

| Field | Meaning |
| --- | --- |
| `t_s` | Run-clock time of the sample. Ticks are scheduled from the start, so they don't drift; a late tick is skipped, not bunched. |
| `vram_used_bytes` | Device memory in use on GPU 0, from NVML (the whole device, not only the server). |
| `server_rss_bytes` | Resident host RAM of the llama-server process (the working set, on Windows). |
| `slots` | The server's `/slots` view: per slot `id`, `is_processing`, and `n_prompt_tokens` of its last task. |

A source that fails at a tick records `null` for that tick. The server does not report KV occupancy (see the pilot findings), and `slots` doesn't give it either: KV footprints come from `requests.jsonl`.

## `metadata.json` (real runs)

| Field | Meaning |
| --- | --- |
| `llama_cpp` | From `llama-server --version`: `release_tag` (e.g. `b11606`), `commit`, `version`, and the raw `version_output`. |
| `model` | `path`, `size_bytes` and `sha256` of the model file, hashed at the start of the run. |
| `gpu` | From NVML: `name`, `memory_total_bytes`, `driver_version`, `cuda_driver_version`. |
| `git` | `commit` of the harness and `dirty` (uncommitted changes to tracked files). |
| `launch_args` | The exact llama-server command line. |
| `server_startup_s` | Launch to healthy for the experiment's server (cold start, model load included). |
| `started_at`, `python`, `platform` | Wall-clock start (UTC) and the host. |

Compare these with [`configs/pins.toml`](../configs/pins.toml).

## Analysis

```
agentserve analyze RUN_DIR [--window-start S] [--window-end S]
agentserve report PATH... --out REPORT_DIR [--window-start S] [--window-end S]
```

`analyze` prints one run's metrics. `report` regenerates every table and plot from raw run directories: each `PATH` is a run directory or a directory holding run directories at any depth. Runs whose `config.json` differ only in `run.seed` form one *configuration*, and each metric is reported as its mean across seeds with the half-width of a 95% confidence interval (Student's t, n − 1 degrees of freedom; a single seed has none). `REPORT_DIR` gets `runs.csv` (per run), `configurations.csv` and `classes.csv` (`<metric>_mean`, `_ci95`, `_n` per configuration, and per configuration and task class), `metrics.md` (the headline table, failure rate by cause and the per-class breakdown) and the plots `goodput_vs_arrival_rate.png`, `latency_vs_n.png`, `memory_over_time.png`, `failure_rate_vs_n.png`, `cached_tokens_by_policy.png`.

**Metric window.** Agent tasks that *arrive* in `[window-start, window-end)` are measured, with their model requests; rates are per minute over the window. System samples and policy decisions count when their `t_s` is in the window. The default window starts at 0 and has no end, with rates taken up to the last completion of any agent task. In code: `agentserve.analysis.MetricWindow`.

| Metric | Definition |
| --- | --- |
| `agent_tasks`, `failed_tasks` | Measured agent tasks, and how many failed. |
| `throughput_tasks_per_min` | Completed (not failed) agent tasks per minute. |
| `goodput_tasks_per_min`, `slo_attainment` | See `unloaded.jsonl` above; `nan` without it. |
| `goodput_fittable_tasks_per_min`, `slo_attainment_fittable` | The same, restricted to task classes that fit the memory budget: a class fits when its longest agent task's last model request (prompt plus `n_predict`, i.e. `initial + steps_max x output + (steps_max - 1) x tool_result` tokens) is at most the budget (`runtime.memory_budget_tokens` or `server.memory_budget`). Agent tasks of the other classes still count as failed tasks in every other metric. `nan` if the config does not give the classes and budget. |
| `completion_latency_p50_s`, `_p95_s`, `_p99_s` | Completion minus arrival over completed agent tasks (linear interpolation). |
| `queueing_mean_s`, `queueing_p95_s` | `admitted_s - arrived_s` per model request, failed ones included (`completed_s - arrived_s` for one never admitted). |
| `peak_vram_mib`, `peak_host_ram_mib` | Largest `vram_used_bytes` and `server_rss_bytes` sample; `nan` without `samples.jsonl`. |
| `failure_rate`, `failure_rate.<cause>` | Failed tasks over all measured agent tasks, overall and per cause (the four typed errors always, others when seen). |
| `cached_tokens_per_request`, `cached_token_fraction` | Over successful model requests: mean `n_cached_tokens`, and cached over prompt tokens. |
| `policy_decisions`, `policy_overhead_mean_us`, `_p99_us`, `policy_overhead_total_ms` | From `decisions.jsonl`; `nan` without it. |

The per-class breakdown repeats the agent-task metrics, queueing time and failure rates for each task class, plus `fits_budget` (1 or 0).
