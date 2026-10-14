# Pinned environment

The exact versions live in [`configs/pins.toml`](../configs/pins.toml). Large artifacts are kept
**outside the repo** in `C:\CSE291A\artifacts\` and referenced by path from `configs/*.toml`.

| What | Pin |
| --- | --- |
| llama.cpp | release `b11606` (commit `1b70350c9`), Windows CUDA 13.4 x64 prebuilt + matching cudart |
| Model | `bartowski/Llama-3.2-3B-Instruct-GGUF` @ `5ab33fa9`, `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, SHA256 `6c1a2b41…c728ff` |
| KV cache | f16 (server default) |
| GPU / driver | RTX 3060 Ti 8 GB, NVIDIA driver 617.14, CUDA 13.4 |
| Python | 3.11 via `uv` (`python -m uv`) |

## Fetching the artifacts

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

If the artifacts live elsewhere, edit `binary` and `model` in the server config.

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
