"""Server launcher: start an unmodified, pinned llama-server from a declarative config.

The memory budget (ADR-0001) is the token capacity of the shared KV pool, so it maps
to `--kv-unified -c <memory_budget>`. Every cache flag is passed explicitly, so a
launch never depends on the release's defaults.

Usage (one documented command):

    python -m uv run python -m agentserve.server configs/pilot.toml [--memory-budget N] [--slots N]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass(frozen=True)
class ServerConfig:
    binary: str
    model: str
    memory_budget: int
    slots: int
    host: str = "127.0.0.1"
    port: int = 8080
    gpu_layers: int = 99
    cache_ram_mib: int = 0
    cache_idle_slots: bool = False
    cache_prompt: bool = True
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_server_config(path: str | Path, **overrides) -> ServerConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)["server"]
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return ServerConfig(**raw)


def launch_args(cfg: ServerConfig) -> list[str]:
    args = [
        cfg.binary,
        "-m", cfg.model,
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--kv-unified",
        "-c", str(cfg.memory_budget),
        "-np", str(cfg.slots),
        "-ngl", str(cfg.gpu_layers),
        "--fit", "off",
        "--metrics",
        "--cache-ram", str(cfg.cache_ram_mib),
        "--cache-idle-slots" if cfg.cache_idle_slots else "--no-cache-idle-slots",
    ]
    if not cfg.cache_prompt:
        args.append("--no-cache-prompt")
    return args + list(cfg.extra_args)


def wait_healthy(base_url: str, timeout_s: float = 120.0, proc: subprocess.Popen | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup with code {proc.returncode}")
        try:
            if httpx.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server at {base_url} not healthy after {timeout_s}s")


def start_server(cfg: ServerConfig, log_path: str | Path | None = None) -> subprocess.Popen:
    """Start llama-server, wait until healthy, return the process (caller stops it)."""
    stdout = open(log_path, "w", encoding="utf-8") if log_path else None
    proc = subprocess.Popen(launch_args(cfg), stdout=stdout, stderr=subprocess.STDOUT if stdout else None)
    try:
        wait_healthy(cfg.base_url, proc=proc)
    except BaseException:
        stop_server(proc)
        raise
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--memory-budget", type=int)
    parser.add_argument("--slots", type=int)
    parser.add_argument("--cache-idle-slots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--print-only", action="store_true", help="print the command and exit")
    ns = parser.parse_args(argv)
    cfg = load_server_config(
        ns.config, memory_budget=ns.memory_budget, slots=ns.slots, cache_idle_slots=ns.cache_idle_slots
    )
    args = launch_args(cfg)
    print(subprocess.list2cmdline(args), flush=True)
    if ns.print_only:
        return 0
    proc = subprocess.Popen(args)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        stop_server(proc)
        return 0


if __name__ == "__main__":
    sys.exit(main())
