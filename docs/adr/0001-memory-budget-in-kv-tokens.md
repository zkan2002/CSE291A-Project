# Memory budget is measured in KV-cache tokens, not bytes

llama-server reserves its KV cache when it launches, so a physical byte cap (container/cgroup or VRAM) mostly makes it fail at startup rather than creating pressure at run time. We therefore define the memory budget as the token capacity of the shared KV pool (`--kv-unified`), vary it as the independent variable, and report physical peak memory (VRAM plus host RAM, including `--cache-ram`) as a measured outcome.
