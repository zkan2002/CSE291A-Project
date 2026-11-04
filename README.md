# Memory-Aware Agent Serving

CSE 291A (Fall 2026), Option 1: *how can admission and scheduling decisions improve completed agent work while keeping memory use bounded?*

This repository contains the experiment harness, the Phase 1 study and its report. The harness drives synthetic agent workloads through an admission proxy in front of an unmodified llama.cpp server, then measures goodput, latency, failures and memory.

- **Report (PDF):** [`docs/reports/phase1-report.pdf`](docs/reports/phase1-report.pdf), with its LaTeX source in [`docs/reports/phase1-report.tex`](docs/reports/phase1-report.tex).
- **Raw data:** GitHub release [`phase1-data`](https://github.com/zkan2002/CSE291A-Project/releases/tag/phase1-data). It holds every sweep run's logs, configs and metadata; see [Data](#data).

## Phase 1 in brief

We ran two FIFO baselines with fixed concurrency on llama.cpp b11606 with Llama-3.2-3B-Instruct Q4_K_M, on an RTX 2000 Ada (16 GB). The sweep varied concurrency, memory budget (the KV-cache pool, in tokens), arrival rate and cache policy, with 5 seeds per configuration: 200 analysed sweep runs on two RunPod machines.

- **Overflow is sudden, not gradual.** When running requests together overflow the KV pool, the server fails nearly all of them at once. At the best tuned setting on a 16k budget this happened 6.4 times per 8-minute run, while mean admitted footprint was only 52% of the budget.
- **Tuning does not fix it.** Neither tuning concurrency nor doubling the budget removes it. Offloading idle context to host RAM showed no clear benefit under the tested conditions.
- **Retained context matters.** Disabling prompt caching cuts goodput about 5×.
- **Phase 2 direction:** footprint-based admission, which holds a model request when its projected footprint would not fit. It is expected to help most at 0.025–0.05 agent tasks/s; at 0.1 agent tasks/s, compute pressure may limit any gains.

The report gives the evidence, the limitations, and an AI-use statement.

## Repository layout

```
src/agentserve/       the harness (Python package)
  workload.py           synthetic agent tasks: 8 task classes, token-array prompts, open-loop arrivals
  proxy.py, policy.py   admission proxy and policies (task-level and request-level FIFO)
  runtime.py            the runtime interface; simulated.py (virtual-clock model of llama-server)
                        and llama_runtime.py / llama_api.py (the real server over HTTP)
  server.py             launches the pinned llama-server from a config
  recorder.py, sampler.py  per-request, per-task, KV, VRAM/RAM and slot logs
  unloaded.py           unloaded completion times that set task SLOs and hard timeouts
  experiment.py, real_experiment.py, sweep.py  single runs and resumable sweeps
  analysis.py, report.py   metrics with 95% CIs, tables and plots
  equivalence.py, best_n.py  machine-equivalence check and best-N selection
  pilot.py              the KV-pool overflow probe
tests/                pytest suite (simulated runtime by default; real server opt-in)
configs/              experiment, sweep and pin configs (see below)
scripts/              pod setup, Phase 1 stage scripts, data extraction and analysis
docs/
  reports/              Phase 1 report (PDF, LaTeX, generated appendix and tables, figures)
  environment.md        pinned environment, RunPod setup, running experiments
  run-directory.md      what each run writes, and every metric's definition
  sweeps.md, workload.md   sweep format; workload design and its justification
  pilot/                pilot findings on llama-server's KV-pool behaviour, with evidence
  adr/                  architecture decision records
GLOSSARY.md           project vocabulary (agent task, memory budget, goodput, ...)
```

## Requirements

- **Python 3.11** with [uv](https://docs.astral.sh/uv/). On Windows, if `uv` is not on `PATH`, use `python -m uv` or `py -3.11 -m uv`; the scripts accept `UV="py -3.11 -m uv"`.
- **Bash** for the scripts. On Windows, use Git Bash.
- **Optional:**
  - [`gh`](https://cli.github.com/), to download the data release.
  - LaTeX with `latexmk`, to rebuild the PDF.
  - An NVIDIA GPU, only to rerun the experiments.

```sh
git clone https://github.com/zkan2002/CSE291A-Project.git
cd CSE291A-Project
uv sync
uv run pytest                 # fast suite on the simulated runtime
```

## Data

The raw results are attached to the release [`phase1-data`](https://github.com/zkan2002/CSE291A-Project/releases/tag/phase1-data), not stored in the Git tree (`results/` is ignored). They are the source of truth for every number in the report.

| Asset | Machine | Size | Contents |
| --- | --- | --- | --- |
| `pod-results-2026-10-19.tar.gz` | A | 24.7 MB | 102 complete sweep runs, 1 run killed by a host reboot, 7 calibration runs, the sweep config as run, pod logs |
| `machine-b-final-2026-10-21.tar.gz` | B | 17.2 MB | 100 complete sweep runs, the sweep config as run, pod logs |
| `phase1-archives.sha256` | | | SHA-256 checksums, also in [`docs/reports/phase1-archives.sha256`](docs/reports/phase1-archives.sha256) |

Download with either of these:

```sh
gh release download phase1-data -R zkan2002/CSE291A-Project -D phase1-archives
# or without gh:
mkdir phase1-archives && cd phase1-archives
for f in pod-results-2026-10-19.tar.gz machine-b-final-2026-10-21.tar.gz phase1-archives.sha256; do
  curl -LO "https://github.com/zkan2002/CSE291A-Project/releases/download/phase1-data/$f"
done
cd ..
```

Each run directory holds:

- `config.json`, `window.json` and `metadata.json` (pinned versions, model hash, GPU and driver);
- `tasks.jsonl`, `requests.jsonl`, `kv.jsonl`, `samples.jsonl` and `decisions.jsonl`;
- `unloaded.jsonl` and `cold_start.json`;
- the server logs.

[`docs/run-directory.md`](docs/run-directory.md) documents every file and metric. The analysis leaves out machine A's interrupted run and the two finished seeds of that configuration, which was re-run in full on machine B ([ADR-0004](docs/adr/0004-pool-results-across-equivalent-machines.md)).

## Reproducing the report's tables and figures (no GPU)

With the archives in `phase1-archives/`:

```sh
bash scripts/phase1-extract.sh phase1-archives   # verify checksums, extract into results/phase1/
bash scripts/phase1-analysis.sh                   # rebuild every table, the appendix and the figures
```

`phase1-analysis.sh` runs these steps in order:

1. The per-machine reports: `agentserve report`, written to `results/phase1/report-{a,b,calibration}/`.
2. The equivalence check between the two machines.
3. The best-N selection, which must match the grids in `configs/phase1-dev-b.toml`.
4. `scripts/phase1_analysis.py`, which writes:
   - [`docs/reports/phase1-tables.md`](docs/reports/phase1-tables.md);
   - [`docs/reports/phase1-appendix.tex`](docs/reports/phase1-appendix.tex);
   - [`docs/reports/phase1-figures/`](docs/reports/phase1-figures/).

Running it from a fresh extraction reproduces the committed tables exactly.

To rebuild the PDF:

```sh
cd docs/reports && latexmk -pdf phase1-report.tex
```

## Running experiments

**On the simulated runtime** (any machine, seconds). It replays the KV-pool behaviour measured on the real server on a virtual clock:

```sh
uv run agentserve run configs/tracer.toml --out results/tracer
uv run agentserve analyze results/tracer
```

**On the real server** (NVIDIA GPU). [`docs/environment.md`](docs/environment.md) covers fetching or building the pinned llama.cpp and model, including RunPod setup with `scripts/runpod-setup.sh`. Then:

```sh
uv run pytest -m real_runtime                                    # adapter and pilot probes on the real server
uv run agentserve run configs/real-smoke.toml --out results/real-smoke
uv run agentserve sweep configs/real-grid-smoke.toml --out results/real-grid-smoke
uv run agentserve report results/real-grid-smoke --out results/real-grid-smoke/report
```

**The full Phase 1 sweep.** It takes roughly one to one and a half days of GPU time per sweep config, and resumes if interrupted:

```sh
uv run agentserve sweep configs/calibrate-rates.toml --out results/phase1/machine-a/results/calibrate-rates
uv run agentserve sweep configs/phase1-dev.toml      --out results/phase1/machine-a/results/phase1-dev
uv run agentserve sweep configs/phase1-dev-b.toml    --out results/phase1/machine-b/results/phase1-dev-b
bash scripts/phase1-analysis.sh
```

Results on other hardware will differ in absolute numbers. Section 8 of the report explains how to compare them with these results.

### Configs

| Config | Purpose |
| --- | --- |
| `tracer.toml` | Smallest end-to-end experiment, on the simulated runtime |
| `default-mix.toml` | The 8-class default workload mix, simulated |
| `pilot.toml` | KV-pool overflow probe against the real server |
| `real-smoke.toml`, `real-grid-smoke.toml` | Short real-server experiment and grid |
| `calibrate-rates.toml` | Arrival-rate calibration on the sweep GPU |
| `phase1-dev.toml`, `phase1-dev-b.toml` | The Phase 1 sweeps on machines A and B |
| `pins.toml` | Pinned llama.cpp, model and machine details |

## Design decisions

Recorded as ADRs in [`docs/adr/`](docs/adr/):

1. The memory budget is measured in KV-cache tokens.
2. The desktop GPU, superseded by (3).
3. Experiments run on a RunPod RTX 2000 Ada.
4. Results from two machines are pooled only after an equivalence check.

Project vocabulary is in [`GLOSSARY.md`](GLOSSARY.md).

## Status

- **Phase 1 is complete:** the harness, both baselines, the sweeps, the analysis and the report.
- **Phase 2 (due Dec 10, 2026) is not started.** It is the footprint-based admission policy proposed in Section 6 of the report. It will be evaluated against the tuned baselines on evaluation workloads, after a sustained-capacity measurement and an equivalence check on the new machine.
