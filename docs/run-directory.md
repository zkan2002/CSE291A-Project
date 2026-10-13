# Run directory format

`agentserve run CONFIG --out RUN_DIR` creates `RUN_DIR` (it must not exist yet) and writes:

| File | Contents |
| --- | --- |
| `config.json` | The full resolved config the run used (keys sorted). |
| `requests.jsonl` | One line per model request, written when it completes or fails. |
| `tasks.jsonl` | One line per agent task, written when it completes or fails. |

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
