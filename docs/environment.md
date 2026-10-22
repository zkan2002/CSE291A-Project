# Pinned environment

The exact versions live in [`configs/pins.toml`](../configs/pins.toml). Large artifacts are kept
**outside the repo** in the artifacts directory: `$AGENTSERVE_ARTIFACTS` if set, otherwise
`C:\CSE291A\artifacts\` on Windows and `/workspace/artifacts` (a RunPod network volume)
elsewhere. Server configs name them as `${AGENTSERVE_ARTIFACTS}/...`, and `llama-server` without
`.exe`, so the same configs run on both.

| What | Pin |
| --- | --- |
| llama.cpp | release `b11606` (commit `1b70350c9`): Windows CUDA 13.4 x64 prebuilt + matching cudart; on Linux, built from source at that commit |
| Model | `bartowski/Llama-3.2-3B-Instruct-GGUF` @ `5ab33fa9`, `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, SHA256 `6c1a2b41…c728ff` |
| KV cache | f16 (server default) |
| GPU / driver | RTX 3060 Ti 8 GB, NVIDIA driver 617.14, CUDA 13.4 |
| Python | 3.11 via `uv` (`python -m uv`) |

## Fetching the artifacts (Windows desktop)

```sh
mkdir -p /c/CSE291A/artifacts/llama.cpp/b11606 /c/CSE291A/artifacts/models
cd /c/CSE291A/artifacts/llama.cpp/b11606
gh release download b11606 -R ggml-org/llama.cpp \
  -p 'llama-b11606-bin-win-cuda-13.4-x64.zip' -p 'cudart-llama-bin-win-cuda-13.4-x64.zip'
sha256sum *.zip          # compare with configs/pins.toml
unzip -o llama-b11606-bin-win-cuda-13.4-x64.zip && unzip -o cudart-llama-bin-win-cuda-13.4-x64.zip

cd /c/CSE291A/artifacts/models
curl -L -o Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf
sha256sum Llama-3.2-3B-Instruct-Q4_K_M.gguf   # 6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff
```

If the artifacts live elsewhere, set `AGENTSERVE_ARTIFACTS` rather than editing the configs.

## Running on RunPod

Use a **Pod** (not Serverless): one fixed GPU for the whole sweep, with the harness starting and
restarting llama-server itself. Docker cannot run inside a pod, so llama.cpp is built from source.

1. **Network volume** (Storage → Network Volumes, ~50 GB) in a datacenter that has the GPU you
   want. Everything that must outlive the pod (artifacts, results, unloaded cache) lives on it.
2. **Pod**: one GPU of a single model kept for every comparison; the official Runpod PyTorch
   template in a `-devel` variant (it carries `nvcc`); the network volume attached at
   `/workspace`; on-demand pricing; SSH exposed.
3. **Setup** (over SSH, inside `tmux`):

   ```sh
   git clone https://github.com/zkan2002/CSE291A-Project.git /workspace/CSE291A-Project
   cd /workspace/CSE291A-Project && git checkout spec-1-memory-aware-serving
   bash scripts/runpod-setup.sh
   ```

   The script installs build tools and uv, builds `llama-server` at the pinned commit into
   `/workspace/artifacts/llama.cpp/b11606/`, downloads the model, and refuses to finish unless the
   model's SHA256 and the server's build/commit match `configs/pins.toml`. Rerunning it on a new
   pod with the same volume skips the build and download.
4. **Validate** before any sweep: `uv run pytest`, then `uv run pytest -m real_runtime`, then a
   short real grid. On a pod, `uv` is on `PATH`, so drop the `python -m` used on the desktop.
5. **Sweep** inside `tmux`, writing to the volume:
   `uv run agentserve sweep configs/phase1-dev.toml --out /workspace/results/phase1`.

Keep the pod running until the sweep finishes: a stopped pod may come back on a different
machine, which would make its results incomparable with earlier ones. Copy results off with
`scp`/`rsync`, then terminate the pod; delete the network volume once nothing on it is needed.

## Launching the server

One command launches the pinned llama-server from a config (`--kv-unified`, `-c` = memory budget,
`-np`, `--metrics`, and every cache flag passed explicitly):

```sh
python -m uv run python -m agentserve.server configs/pilot.toml
# overrides: --memory-budget 4096 --slots 8 --cache-idle-slots / --no-cache-idle-slots
# --print-only prints the command line without starting the server
```

Every cache flag is explicit because the release defaults are **not** "caching off":
at `b11606`, `--cache-ram` defaults to 8192 MiB and `--cache-idle-slots` defaults to on.
`--fit off` is also passed so the server never silently resizes anything to fit device memory.

## Running the pool-overflow pilot

```sh
python -m uv run python -m agentserve.pilot configs/pilot.toml --out results/pilot
python -m uv run pytest -m real_runtime        # the same probes as assertions
```

Findings: [`docs/pilot/pilot-findings.md`](pilot/pilot-findings.md).

## Running an experiment on the real server

```sh
python -m uv run agentserve run configs/real-smoke.toml --out results/real-smoke
python -m uv run agentserve analyze results/real-smoke
```

The run launches llama-server from the config's `[server]` table (once to measure unloaded
completion times, then fresh for the experiment), samples VRAM, server RAM and `/slots`
every 100 ms, and stops the server at the end. The extra files in the run directory
(`samples.jsonl`, `metadata.json`, server logs) are described in
[`run-directory.md`](run-directory.md). `pytest -m real_runtime` includes a smoke test
of this command and of the adapter.

## Running a sweep on the real server

```sh
python -m uv run agentserve sweep configs/real-grid-smoke.toml --out results/real-grid-smoke
python -m uv run agentserve report results/real-grid-smoke --out results/real-grid-smoke/report
```

`configs/phase1-dev.toml` is the Phase 1 sweep. Rerun the same command to resume an
interrupted sweep. Format and directory layout: [`sweeps.md`](sweeps.md).
