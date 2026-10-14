"""Typed accessors for an experiment config (the dict an ``agentserve run`` TOML
file loads into), so no other module switches on the runtime kind for its keys.

A simulated run keeps its memory budget, slot count and cache flags under
``[runtime]`` (``memory_budget_tokens``, ``n_slots``, ``cache_ram_tokens``); a
real run (``[runtime] kind = "llama_server"``) under ``[server]`` in the
launcher's format (``memory_budget``, ``slots``, ``cache_ram_mib``). Both share
``cache_idle_slots`` and ``cache_prompt``.

Cache policies (pilot amendment 6):

- ``resident``: ``cache_idle_slots = false``, caching on; retained context stays
  in the KV pool, and agent tasks are pinned to slots by default.
- ``offload``: ``cache_idle_slots = true`` (needs a host-RAM prompt cache);
  idle context moves to host RAM, and agent tasks are not pinned by default.
- ``off``: ``cache_prompt = false`` and no host-RAM cache.
"""

from __future__ import annotations

from typing import Any

CACHE_POLICIES = ("resident", "offload", "off")

DEFAULT_SPLIT = "dev"
DEFAULT_RECORD_PROMPTS = False
DEFAULT_REQUEST_TIMEOUT_S = 600.0
DEFAULT_SAMPLE_INTERVAL_S = 0.1

Config = dict[str, Any]

_KEYS = {
    True: {"table": "server", "budget": "memory_budget", "slots": "slots",
           "ram": "cache_ram_mib"},
    False: {"table": "runtime", "budget": "memory_budget_tokens", "slots": "n_slots",
            "ram": "cache_ram_tokens"},
}


def is_real(config: Config) -> bool:
    """Whether the run uses the real llama-server rather than the simulated runtime."""
    return config.get("runtime", {}).get("kind") == "llama_server"


def _keys(config: Config) -> dict[str, str]:
    return _KEYS[is_real(config)]


def _table(config: Config) -> dict[str, Any]:
    return config.get(_keys(config)["table"], {})


def memory_budget_tokens(config: Config) -> int | None:
    """The memory budget: the KV pool's token capacity (ADR-0001)."""
    return _table(config).get(_keys(config)["budget"])


def set_memory_budget_tokens(config: Config, tokens: int) -> None:
    config.setdefault(_keys(config)["table"], {})[_keys(config)["budget"]] = tokens


def slots(config: Config) -> int | None:
    """The runtime's slot count (``-np``)."""
    return _table(config).get(_keys(config)["slots"])


def set_slots(config: Config, n: int) -> None:
    config.setdefault(_keys(config)["table"], {})[_keys(config)["slots"]] = n


def cache_policy(config: Config) -> str | None:
    """``resident``, ``offload`` or ``off`` (``None`` for a config without a runtime).
    Absent flags take the runtime's defaults (caching on, idle slots not offloaded)."""
    if "kind" not in config.get("runtime", {}):
        return None
    table = _table(config)
    if table.get("cache_prompt", True) is False:
        return "off"
    return "offload" if table.get("cache_idle_slots", False) else "resident"


def set_cache_policy(config: Config, policy: str) -> None:
    """Set the cache flags for ``policy``; ``offload`` keeps the config's host-RAM
    cache size and needs one, ``off`` turns it off."""
    keys = _keys(config)
    table = config.setdefault(keys["table"], {})
    if policy == "resident":
        table.update(cache_prompt=True, cache_idle_slots=False)
    elif policy == "offload":
        if not table.get(keys["ram"]):
            raise ValueError(f"the offload cache policy needs a host-RAM prompt cache: "
                             f"set {keys['ram']} in the base config")
        table.update(cache_prompt=True, cache_idle_slots=True)
    elif policy == "off":
        table.update(cache_prompt=False, cache_idle_slots=False, **{keys["ram"]: 0})
    else:
        raise ValueError(f"unknown cache policy {policy!r} (expected one of {CACHE_POLICIES})")


def cache_policy_pins(config: Config) -> bool:
    """Whether the cache policy pins agent tasks to slots: all but ``offload`` do,
    which instead gets retained context back through the host-RAM prompt cache."""
    return cache_policy(config) != "offload"


def pin_slots(config: Config) -> bool:
    """Whether the run's admission policy pins agent tasks to slots (``id_slot``):
    ``[policy] pin_slots`` if given, else as the cache policy has it."""
    return config.get("policy", {}).get("pin_slots", cache_policy_pins(config))


def split(config: Config) -> str:
    """``dev`` (development workloads) or ``eval`` (evaluation workloads)."""
    return config["run"].get("split", DEFAULT_SPLIT)


def record_prompts(config: Config) -> bool:
    return config["run"].get("record_prompts", DEFAULT_RECORD_PROMPTS)


def request_timeout_s(config: Config) -> float:
    """HTTP timeout per model request on a real run."""
    return config["runtime"].get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)



def sample_interval_s(config: Config) -> float:
    """Seconds between system samples on a real run."""
    return config["runtime"].get("sample_interval_s", DEFAULT_SAMPLE_INTERVAL_S)
