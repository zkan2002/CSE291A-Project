# Run directory format

`agentserve run CONFIG --out RUN_DIR` creates `RUN_DIR` (it must not exist yet) and writes:

| File | Contents |
| --- | --- |
| `config.json` | The full resolved config the run used (keys sorted). |
| `requests.jsonl` | One line per model request, written when it completes or fails. |
| `tasks.jsonl` | One line per agent task, written when it completes or fails. |
| `unloaded.jsonl` | One line per agent task: its unloaded completion time, task SLO and hard timeout. Written before the experiment starts. |
| `prompts.jsonl` | Only with `record_prompts = true` under `[run]`: one line per model request, written when it is issued. |

All times are seconds since the experiment started, on the run's clock (virtual for the simulated runtime). Lines are JSON objects with sorted keys, so the same config and seed give byte-identical files on the virtual clock. Nothing is dropped: a failed model request or failed task still gets its line.

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
