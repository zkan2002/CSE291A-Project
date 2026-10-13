"""Summarise a sample of public agent trajectories to anchor the workload mix.

Source: the Hugging Face dataset ``nebius/SWE-agent-trajectories`` (80,036
SWE-agent trajectories on SWE-bench-style repositories), read through the public
datasets-server ``/rows`` API, so nothing has to be downloaded in bulk.

Each trajectory is a system message, a first user message (the task), then
alternating ``ai`` (model output) and ``user`` (tool observation) messages. We
count:

- agent steps: ``ai`` messages per trajectory;
- starting context: tokens of the system prompt plus the first user message;
- output per agent step: tokens of each ``ai`` message;
- tool result: tokens of each ``user`` message after the first.

Token counts use tiktoken's ``cl100k_base`` as a stand-in for the Llama 3
tokenizer (both are ~100k-vocabulary byte-level BPEs; Llama 3's tokenizer is
gated). Run (needs network):

    python -m uv run --with tiktoken python scripts/trajectory_stats.py
"""

from __future__ import annotations

import json
import random
import statistics
import time
import urllib.request
from collections import Counter

import tiktoken

DATASET = "nebius/SWE-agent-trajectories"
N_ROWS = 80_036
SAMPLE_SEED = 0
N_PAGES = 20
PAGE_ROWS = 25
URL = ("https://datasets-server.huggingface.co/rows?dataset={ds}&config=default"
       "&split=train&offset={offset}&length={length}")


def fetch(offset: int) -> list[dict]:
    url = URL.format(ds=DATASET.replace("/", "%2F"), offset=offset, length=PAGE_ROWS)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return [r["row"] for r in json.load(resp)["rows"]]
        except Exception:  # transient datasets-server errors
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not fetch {url}")


def quantiles(xs: list[int]) -> str:
    xs = sorted(xs)

    def q(p: float) -> int:
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    return (f"n={len(xs)} p10={q(0.10)} p25={q(0.25)} median={q(0.50)} "
            f"p75={q(0.75)} p90={q(0.90)} mean={statistics.fmean(xs):.0f}")


def main() -> None:
    enc = tiktoken.get_encoding("cl100k_base")

    def ntok(s: str | None) -> int:
        return len(enc.encode(s or "", disallowed_special=()))

    rng = random.Random(SAMPLE_SEED)
    offsets = sorted(rng.sample(range(0, N_ROWS - PAGE_ROWS, PAGE_ROWS), N_PAGES))
    steps, start, output, tool = [], [], [], []
    models, exits = Counter(), Counter()
    for offset in offsets:
        for row in fetch(offset):
            traj = row["trajectory"]
            system = next(m for m in traj if m["role"] == "system")
            users = [m for m in traj if m["role"] == "user"]
            ais = [m for m in traj if m["role"] == "ai"]
            models[row["model_name"]] += 1
            exits[row["exit_status"]] += 1
            steps.append(len(ais))
            start.append(ntok(system["system_prompt"]) + ntok(system["text"]) + ntok(users[0]["text"]))
            output += [ntok(m["text"]) for m in ais]
            tool += [ntok(m["text"]) for m in users[1:]]
    print(f"dataset: {DATASET}; {len(steps)} trajectories from {N_PAGES} pages of {PAGE_ROWS} "
          f"at offsets {offsets}")
    print("models:", dict(models))
    print("exit statuses:", dict(exits.most_common(5)))
    print("agent steps per trajectory:", quantiles(steps))
    print("starting context tokens:   ", quantiles(start))
    print("output tokens per step:    ", quantiles(output))
    print("tool result tokens:        ", quantiles(tool))


if __name__ == "__main__":
    main()
