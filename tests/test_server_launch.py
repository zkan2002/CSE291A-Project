"""The server launcher turns a declarative config into a reproducible llama-server command."""

from pathlib import Path

from agentserve.server import ServerConfig, launch_args, load_server_config, resolve_artifact_path

PILOT_TOML = """
[server]
binary = "C:/artifacts/llama.cpp/b11606/llama-server.exe"
model = "C:/artifacts/models/model.gguf"
host = "127.0.0.1"
port = 8091
memory_budget = 2048
slots = 4
gpu_layers = 99
cache_ram_mib = 512
cache_idle_slots = true
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "server.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_memory_budget_and_slots_map_to_unified_pool_flags(tmp_path):
    args = launch_args(load_server_config(_write(tmp_path, PILOT_TOML)))

    assert args[0] == "C:/artifacts/llama.cpp/b11606/llama-server.exe"
    assert _flag_value(args, "-m") == "C:/artifacts/models/model.gguf"
    assert "--kv-unified" in args
    assert _flag_value(args, "-c") == "2048"
    assert _flag_value(args, "-np") == "4"
    assert "--metrics" in args
    assert _flag_value(args, "--host") == "127.0.0.1"
    assert _flag_value(args, "--port") == "8091"
    assert _flag_value(args, "-ngl") == "99"
    # the server must not silently resize anything to "fit" device memory
    assert _flag_value(args, "--fit") == "off"


def test_cache_flags_are_always_explicit(tmp_path):
    on = launch_args(load_server_config(_write(tmp_path, PILOT_TOML)))
    assert _flag_value(on, "--cache-ram") == "512"
    assert "--cache-idle-slots" in on and "--no-cache-idle-slots" not in on

    off_toml = PILOT_TOML.replace("cache_idle_slots = true", "cache_idle_slots = false")
    off = launch_args(load_server_config(_write(tmp_path, off_toml)))
    assert "--no-cache-idle-slots" in off and "--cache-idle-slots" not in off


def test_prompt_caching_can_be_switched_off(tmp_path):
    cfg = load_server_config(_write(tmp_path, PILOT_TOML + "cache_prompt = false\n"))
    assert "--no-cache-prompt" in launch_args(cfg)
    assert "--no-cache-prompt" not in launch_args(load_server_config(_write(tmp_path, PILOT_TOML)))


def test_extra_args_are_appended_verbatim(tmp_path):
    cfg = load_server_config(_write(tmp_path, PILOT_TOML + 'extra_args = ["-sps", "0.5"]\n'))
    assert launch_args(cfg)[-2:] == ["-sps", "0.5"]


def test_overrides_replace_config_values(tmp_path):
    cfg = load_server_config(_write(tmp_path, PILOT_TOML), memory_budget=4096, slots=2)
    assert isinstance(cfg, ServerConfig)
    args = launch_args(cfg)
    assert _flag_value(args, "-c") == "4096"
    assert _flag_value(args, "-np") == "2"


# --- artifact paths: one config works on the Windows desktop and on a Linux pod ---

TEMPLATE_TOML = """
[server]
binary = "${AGENTSERVE_ARTIFACTS}/llama.cpp/b11606/llama-server"
model = "${AGENTSERVE_ARTIFACTS}/models/model.gguf"
memory_budget = 2048
slots = 4
"""


def test_artifacts_variable_expands_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSERVE_ARTIFACTS", "/workspace/artifacts")
    cfg = load_server_config(_write(tmp_path, TEMPLATE_TOML))
    assert cfg.model == "/workspace/artifacts/models/model.gguf"
    assert launch_args(cfg)[0].startswith("/workspace/artifacts/llama.cpp/b11606/llama-server")


def test_artifacts_variable_defaults_per_platform():
    template = "${AGENTSERVE_ARTIFACTS}/m.gguf"
    assert resolve_artifact_path(template, env={}, platform="linux") == "/workspace/artifacts/m.gguf"
    assert resolve_artifact_path(template, env={}, platform="win32") == "C:/CSE291A/artifacts/m.gguf"


def test_windows_binary_without_extension_resolves_to_the_exe(tmp_path):
    (tmp_path / "llama-server.exe").write_bytes(b"")
    bare = str(tmp_path / "llama-server")
    assert resolve_artifact_path(bare, env={}, platform="win32") == bare + ".exe"
    # on Linux the name is used as written
    assert resolve_artifact_path(bare, env={}, platform="linux") == bare


def test_plain_paths_are_unchanged():
    assert resolve_artifact_path("C:/artifacts/x.exe", env={}, platform="win32") == "C:/artifacts/x.exe"
    assert resolve_artifact_path("/opt/m.gguf", env={"AGENTSERVE_ARTIFACTS": "/x"}, platform="linux") \
        == "/opt/m.gguf"
