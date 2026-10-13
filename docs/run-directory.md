# Run directory format

`agentserve run CONFIG --out RUN_DIR` creates `RUN_DIR` (it must not exist yet) and writes:

| File | Contents |
| --- | --- |
| `config.json` | The full resolved config the run used (keys sorted). |
| `requests.jsonl` | One line per model request, written when it completes or fails. |
| `tasks.jsonl` | One line per agent task, written when it completes or fails. |
| `unloaded.jsonl` | One line per agent task: its unloaded completion time, task SLO and hard timeout. Written before the experiment starts. |
| `prompts.jsonl` | Only with `record_prompts = true` under `[run]`: one line per model request, written when it is issued. |

A **real run** (`[runtime] kind = "llama_server"`) writes the same files in the same format, plus:

| File | Contents |
| --- | --- |
| `samples.jsonl` | One system sample every `sample_interval_s` (default 0.1 s) while the experiment runs. |
| `metadata.json` | What the run was measured on: llama.cpp build, model hash, GPU, driver and CUDA versions, harness commit, launch arguments. |
| `server.log` | The experiment's llama-server output. |
| `unloaded-server.log` | The llama-server output while unloaded completion times were measured (absent when they came from `--unloaded-cache`). |

All times are seconds since the experiment started, on the run's clock (virtual for the simulated runtime, `perf_counter` wall time for a real run). Lines are JSON objects with sorted keys, so the same config and seed give byte-identical files on the virtual clock. A simulated run never writes the real-run files, so nothing that depends on wall time lands in its run directory. Nothing is dropped: a failed model request or failed task still gets its line.

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
| `admitted_s` | When the admission policy admitted it and it was sent to the runtime. |
| `first_token_s` | Completion time minus the runtime's decode time; `null` on error. |
| `completed_s` | When the runtime returned a completion or an error. |
| `n_prompt_tokens` | Prompt length in tokens. |
| `n_cached_tokens` | Prompt tokens served from retained context; `null` on error. |
| `n_predict` | Output tokens requested (EOS ignored, so a completion has exactly this many). |
| `n_output_tokens` | Generated tokens (0 on error). |
| `slot` | Slot the runtime used (or the pinned slot, on error). |
| `error` | `null`, or the typed error: `pool_exhausted`, `exceeds_context`, `server_error`, `timeout`. |

## `tasks.jsonl`

| Field | Meaning |
| --- | --- |
| `task_id`, `task_class` | The agent task and its task class. |
| `arrived_s` | Arrival time from the workload. |
| `step_admitted_s` | Admission time of each agent step's model request, in order. |
| `completed_s` | When the agent task finished (successfully or not). |
| `failure` | `null`, or the cause of the failed task (the error of its failing model request). |

An agent task is *active* from `step_admitted_s[0]` to `completed_s`.

## `prompts.jsonl`

| Field | Meaning |
| --- | --- |
| `task_id`, `step` | The agent task and the index of its agent step. |
| `prompt_tokens` | The exact token-ID array sent as the model request's prompt. |

## `unloaded.jsonl`

Before the experiment, every agent task of the workload is run alone on an otherwise idle runtime (a fresh one, on its own clock), one after another, through the same proxy and runtime interface, under task-level admission with N=1. The measurement does not depend on the run's admission policy, so every policy is judged against the same task SLOs.

| Field | Meaning |
| --- | --- |
| `task_id`, `task_class` | The agent task and its task class. |
| `unloaded_completion_s` | Completion minus arrival when run alone; `null` if it failed even then. |
| `slo_s` | Task SLO: 3 x `unloaded_completion_s` (`null` if that is `null`). |
| `hard_timeout_s` | Hard timeout: 10 x `unloaded_completion_s` (`null` if that is `null`). |
| `failure` | `null`, or the error that failed the agent task when run alone. |

`agentserve run ... --unloaded-cache DIR` caches the measurement in `DIR`, keyed by the workload, seed, duration, split and runtime config (not the policy), and reuses it on later runs with the same key. Without the flag, it is measured afresh for every run.

`agentserve analyze` reports *goodput* (agent tasks that did not fail and completed within `slo_s` of arriving, per minute over the same window as throughput) and *SLO attainment* (the fraction of all agent tasks, failed ones included, that count towards goodput).

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
