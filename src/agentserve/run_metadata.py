"""Run metadata for a real run: what was measured on, read from the system itself.

The llama.cpp build comes from ``llama-server --version``, the model hash from the
model file, the driver and CUDA versions from NVML, and the commit from git, so a
run directory records what actually ran, not what the pins say should have.
"""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentserve.server import ServerConfig, launch_args

_VERSION = re.compile(r"version:\s*(?P<version>\S+)\s*\(build (?P<build>\d+), commit (?P<commit>[0-9a-f]+)\)")


def collect_run_metadata(cfg: ServerConfig) -> dict[str, Any]:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llama_cpp": llama_cpp_version(cfg.binary),
        "model": {"path": cfg.model, "size_bytes": Path(cfg.model).stat().st_size,
                  "sha256": sha256_file(cfg.model)},
        "gpu": gpu_info(),
        "git": git_info(),
        "launch_args": launch_args(cfg),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def llama_cpp_version(binary: str) -> dict[str, Any]:
    out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    text = (out.stdout + out.stderr).strip()
    m = _VERSION.search(text)
    if m is None:
        return {"version_output": text, "release_tag": None, "commit": None}
    return {"version_output": text, "release_tag": f"b{m['build']}", "commit": m["commit"],
            "version": m["version"]}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_info(index: int = 0) -> dict[str, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        cuda = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        return {
            "name": pynvml.nvmlDeviceGetName(handle),
            "memory_total_bytes": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
            "cuda_driver_version": f"{cuda // 1000}.{cuda % 1000 // 10}",
        }
    except Exception as e:  # no NVIDIA GPU / driver: record why, don't fail the run
        return {"error": repr(e), "driver_version": None, "cuda_driver_version": None}


def git_info() -> dict[str, Any]:
    repo = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              timeout=30).stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
