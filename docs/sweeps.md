# Sweeps

```sh
python -m uv run agentserve sweep configs/phase1-dev.toml --out results/phase1-dev --dry-run  # list sweep runs
python -m uv run agentserve sweep configs/phase1-dev.toml --out results/phase1-dev            # run / resume
python -m uv run agentserve report results/phase1-dev --out results/phase1-dev/report
```

One command runs a whole grid. Rerunning the same command resumes an interrupted
sweep: sweep runs already complete are skipped, everything else runs. The sweep is
serial (one GPU), and each run starts fresh llama-server processes and stops them
(cold start, unloaded measurement, experiment; see [`run-directory.md`](run-directory.md)),
so no cached state carries over between configurations. Before and after every
real sweep run the runner checks that no server answers on the configured port, and
refuses to go on if one does.

## Sweep config

```toml
[sweep]
seeds = [1, 2, 3, 4, 5]
# slots = 8             # default: the runtime's slot count follows N

[[grid]]                # one or more; their sweep runs are unioned (a shared one runs once)
policy = ["task_level_fifo", "request_level_fifo"]
concurrency = [1, 2, 4, 8]
memory_budget = [16384]
arrival_rate = [0.1]
cache_policy = ["resident"]
# seeds = [1, 2]        # overrides [sweep] seeds for this grid

[base.run]              # every table of an `agentserve run` config, under `base.`
...
```

A grid that leaves an axis out keeps the base config's value.

| Axis | Sets |
| --- | --- |
| `policy` | `policy.name` |
| `concurrency` | `policy.concurrency` (N), and the slot count (`server.slots` / `runtime.n_slots`) unless `[sweep] slots` fixes it |
| `memory_budget` | `server.memory_budget` (real) or `runtime.memory_budget_tokens` (simulated) |
| `arrival_rate` | `workload.arrival_rate_per_s` |
| `cache_policy` | `resident`: `cache_idle_slots = false`, agent tasks pinned with `id_slot`. `offload`: `cache_idle_slots = true` (needs the base's host-RAM cache, `cache_ram_mib` / `cache_ram_tokens`), not pinned. `off`: `cache_prompt = false` and no host-RAM cache. See pilot amendment 6. |

## Sweep directory

| Path | Contents |
| --- | --- |
| `runs/<sweep-run>/` | One complete run directory per sweep run, named `<policy>-n<N>-b<budget>-r<rate>-<cache>-s<seed>`. |
| `runs/<sweep-run>.partial/`, `runs/<sweep-run>.<k>.partial/` | A sweep run in progress, or an attempt that failed or was interrupted (kept for inspection, retried on the next invocation). `report` ignores them. |
| `sweep.json` | The sweep config and its list of sweep runs (`sweep_runs`), as last invoked. |
| `sweep.jsonl` | One line per attempt at a sweep run: `sweep_run` (its name), `status` (`complete`, `skipped`, `failed`, `interrupted`), `at`, `wall_s`, `error`, and `fresh_servers` for real sweep runs. |
| `unloaded-cache/` | Unloaded completion times per agent task, measured on the reference runtime config, so shared by every sweep run whatever its policy, cache policy and budget (see `unloaded.jsonl` and `--unloaded-cache`). |

A failed sweep run (for example a server that won't start) is recorded and the sweep
moves on; the command then exits with status 1. Resuming with a sweep config
whose sweep runs differ from the complete ones already in the directory is refused:
sweep into a new directory instead.
