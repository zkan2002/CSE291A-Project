"""Tables and figures for the Phase 1 report, from the raw runs of both machines.

Run it through scripts/phase1-analysis.sh, which first builds the per-machine reports it reads.
It expects, under --root (default results/phase1):

    machine-a/results/phase1-dev/runs    report-a/   (machine A's runs and their report)
    machine-b/results/phase1-dev-b/runs  report-b/   (machine B's)

Writes docs/reports/phase1-figures/*.png, the tables as UTF-8 Markdown (--tables, default
docs/reports/phase1-tables.md), and the same tables as the report's LaTeX appendix (--latex,
default docs/reports/phase1-appendix.tex).

Pooling follows ADR-0004: a configuration's seeds never span machines, and machine A's two-seed
request-level N = 16 at 32k is left out (it was re-run in full on machine B).
"""
import argparse
import bisect
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/phase1")
RUNS = {}


def set_root(root):
    global ROOT
    ROOT = Path(root)
    RUNS["A"] = ROOT / "machine-a/results/phase1-dev/runs"
    RUNS["B"] = ROOT / "machine-b/results/phase1-dev-b/runs"
FIGURES = Path("docs/reports/phase1-figures")
T_95_DF4 = 2.776  # Student's t, two-sided 95%, 4 degrees of freedom (5 seeds)
ABORT_GAP_S = 1.0  # pool_exhausted failures closer than this belong to one abort event
SEEDS = range(1, 6)

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
TASK_LEVEL = "#2a78d6"
REQUEST_LEVEL = "#eb6834"
TIMEOUT = "#2a78d6"
POOL = "#eb6834"
LABELS = {"task_level_fifo": "Task-level FIFO", "request_level_fifo": "Request-level FIFO"}
COLORS = {"task_level_fifo": TASK_LEVEL, "request_level_fifo": REQUEST_LEVEL}


def load_configurations():
    """Every complete configuration, keyed (policy, N, budget, rate, cache policy), with its machine."""
    table = {}
    for machine in ("A", "B"):
        path = ROOT / f"report-{machine.lower()}" / "configurations.csv"
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["seeds"] != "1 2 3 4 5":
                    continue
                row["machine"] = machine
                key = (
                    row["policy"],
                    int(row["concurrency"]),
                    int(row["memory_budget_tokens"]),
                    float(row["arrival_rate_per_s"]),
                    row["cache_policy"],
                )
                # The equivalence configuration ran on both; machine A's stays the tuning value.
                table.setdefault(key, row)
                table.setdefault(key + (machine,), row)
    return table


def value(row, metric):
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_ci95"])


def fmt(row, metric, digits=2, scale=1.0):
    mean, ci = value(row, metric)
    return f"{mean * scale:.{digits}f} ± {ci * scale:.{digits}f}"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_mechanism(machine, name, seed):
    """Abort events, KV footprint and wasted work of one run, over its measured agent tasks."""
    run = RUNS[machine] / f"{name}-s{seed}"
    window = json.loads((run / "window.json").read_text())
    start, end = window["start_s"], window["end_s"]
    tasks = [t for t in read_jsonl(run / "tasks.jsonl") if start <= t["arrived_s"] < end]
    ids = {t["task_id"] for t in tasks}
    failed = {t["task_id"] for t in tasks if t["failure"]}
    requests = [r for r in read_jsonl(run / "requests.jsonl") if r["task_id"] in ids]
    kv = read_jsonl(run / "kv.jsonl")

    events = []
    for t in sorted(r["completed_s"] for r in requests if r["error"] == "pool_exhausted"):
        if events and t - events[-1][-1] < ABORT_GAP_S:
            events[-1].append(t)
        else:
            events.append([t])

    times = [k["t_s"] for k in kv]
    before = [kv[i] for i in (bisect.bisect_left(times, e[0] - 0.05) - 1 for e in events) if i >= 0]

    # Spells where the in-flight footprint exceeds the budget, and how many of them ended in an abort.
    budget = json.loads((run / "config.json").read_text())["server"]["memory_budget"]
    spells = []
    for k, following in zip(kv, kv[1:]):
        if start <= k["t_s"] < end and k["in_flight_tokens"] > budget:
            if spells and spells[-1][1] == k["t_s"]:
                spells[-1][1] = following["t_s"]
            else:
                spells.append([k["t_s"], following["t_s"]])
    aborted = sum(any(a - 0.05 <= e[0] <= b + 0.05 for e in events) for a, b in spells)
    measured = [e for e in events if start <= e[0] < end]
    uncrossed = sum(not any(a - 0.05 <= e[0] <= b + 0.05 for a, b in spells) for e in measured)

    occupied = in_flight = covered = 0.0
    for a, b in zip(kv, kv[1:]):
        dt = min(b["t_s"], end) - max(a["t_s"], start)
        if dt > 0:
            occupied += a["admitted_tokens"] * dt
            in_flight += a["in_flight_tokens"] * dt
            covered += dt

    output = sum(r["n_output_tokens"] or 0 for r in requests)
    wasted = sum(r["n_output_tokens"] or 0 for r in requests if r["task_id"] in failed)
    return {
        "events": len(events),
        "victims": [len(e) for e in events],
        "in_flight_at_abort": [k["in_flight_tokens"] for k in before],
        "retained_at_abort": [k["retained_tokens"] for k in before],
        "mean_admitted": occupied / covered,
        "output_per_s": output / (end - start),
        "wasted_output": wasted / output if output else 0.0,
        "spells": len(spells),
        "spells_aborted": aborted,
        "aborts_measured": len(measured),
        "aborts_under_budget": uncrossed,
    }


def mechanism(machine, name):
    runs = [run_mechanism(machine, name, s) for s in SEEDS]
    def ci(values):
        return T_95_DF4 * st.stdev(values) / len(values) ** 0.5
    victims = [v for r in runs for v in r["victims"]]
    in_flight = [v for r in runs for v in r["in_flight_at_abort"]]
    retained = [v for r in runs for v in r["retained_at_abort"]]
    summary = {}
    for key in ("events", "mean_admitted", "output_per_s", "wasted_output"):
        values = [r[key] for r in runs]
        summary[key] = (st.mean(values), ci(values))
    for key in ("spells", "spells_aborted", "aborts_measured", "aborts_under_budget"):
        summary[key] = sum(r[key] for r in runs)
    summary["victims"] = st.mean(victims) if victims else float("nan")
    summary["in_flight_at_abort"] = st.mean(in_flight) if in_flight else float("nan")
    summary["retained_at_abort"] = st.mean(retained) if retained else float("nan")
    return summary


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_tuning(cfg):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, budget in zip(axes, (16384, 32768)):
        for policy in LABELS:
            ns = [n for n in (1, 2, 4, 8, 16) if (policy, n, budget, 0.1, "resident") in cfg]
            rows = [cfg[(policy, n, budget, 0.1, "resident")] for n in ns]
            ax.errorbar(
                ns,
                [value(r, "goodput_tasks_per_min")[0] for r in rows],
                yerr=[value(r, "goodput_tasks_per_min")[1] for r in rows],
                color=COLORS[policy], linewidth=2, marker="o", markersize=7, capsize=3,
                markeredgecolor="white", label=LABELS[policy],
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.minorticks_off()
        ax.set_xlabel("Concurrency N", color=INK_2)
        ax.set_title(f"{budget // 1024}k memory budget", color=INK, fontsize=10, loc="left")
        ax.set_ylim(0, 4.5)
        style(ax)
    axes[0].set_ylabel("Goodput (agent tasks/min)", color=INK_2)
    axes[1].legend(frameon=False, loc="upper left", labelcolor=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1-goodput-vs-n.png", dpi=200)
    plt.close(fig)


def figure_failures(cfg):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for i, policy in enumerate(LABELS):
        for j, budget in enumerate((16384, 32768)):
            ax = axes[i][j]
            ns = [n for n in (1, 2, 4, 8, 16) if (policy, n, budget, 0.1, "resident") in cfg]
            rows = [cfg[(policy, n, budget, 0.1, "resident")] for n in ns]
            pos = [(1, 2, 4, 8, 16).index(n) for n in ns]
            timeout = [100 * value(r, "failure_rate.timeout")[0] for r in rows]
            pool = [100 * value(r, "failure_rate.pool_exhausted")[0] for r in rows]
            ax.bar(pos, timeout, width=0.55, color=TIMEOUT, label="Hard timeout", edgecolor="white", linewidth=1)
            ax.bar(pos, pool, width=0.55, bottom=timeout, color=POOL, label="Pool exhausted", edgecolor="white", linewidth=1)
            ax.set_xticks(range(5), ["1", "2", "4", "8", "16"])
            ax.set_title(f"{LABELS[policy]}, {budget // 1024}k", color=INK, fontsize=10, loc="left")
            ax.set_ylim(0, 100)
            style(ax)
    axes[0][0].legend(frameon=False, loc="upper right", labelcolor=INK_2, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Concurrency N", color=INK_2)
    for row in axes:
        row[0].set_ylabel("Failed tasks (%)", color=INK_2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2-failure-causes.png", dpi=200)
    plt.close(fig)


def figure_rates(cfg):
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    pairs = [("task_level_fifo", 4, 16384), ("task_level_fifo", 16, 32768)]
    for policy, n, budget in pairs:
        rates = (0.025, 0.05, 0.1)
        rows = [cfg[(policy, n, budget, r, "resident", "B")] for r in rates]
        means = [100 * value(r, "slo_attainment")[0] for r in rows]
        cis = [100 * value(r, "slo_attainment")[1] for r in rows]
        dashed = budget == 32768
        ax.errorbar(
            rates, means, yerr=cis, color=TASK_LEVEL if not dashed else REQUEST_LEVEL,
            linewidth=2, marker="o", markersize=7, capsize=3, markeredgecolor="white",
            linestyle="--" if dashed else "-", label=f"{budget // 1024}k, N = {n}",
        )
    ax.set_xticks((0.025, 0.05, 0.1), ["0.025", "0.05", "0.1"])
    ax.set_xlabel("Arrival rate (agent tasks/s)", color=INK_2)
    ax.set_ylabel("Agent tasks within task SLO (%)", color=INK_2)
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK_2, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4-slo-vs-rate.png", dpi=200)
    plt.close(fig)


def figure_abort_anatomy():
    run = RUNS["A"] / "task_level_fifo-n4-b16384-r0.1-resident-s1"
    kv = read_jsonl(run / "kv.jsonl")
    start, end = 60, 540
    points = [k for k in kv if start <= k["t_s"] <= end]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.step(
        [k["t_s"] - start for k in points], [k["in_flight_tokens"] / 1000 for k in points],
        where="post", color=TASK_LEVEL, linewidth=1.2,
    )
    ax.axhline(16.384, color=INK, linewidth=1, linestyle="--")
    ax.text(108, 16.8, "Budget (16k)", color=INK_2, fontsize=9, ha="left")
    events = []
    requests = read_jsonl(run / "requests.jsonl")
    for t in sorted(r["completed_s"] for r in requests if r["error"] == "pool_exhausted"):
        if not events or t - events[-1] >= ABORT_GAP_S:
            events.append(t)
        else:
            events[-1] = t
    shown = [t - start for t in events if start <= t <= end]
    ax.plot(shown, [20.5] * len(shown), linestyle="none", marker="v", markersize=8, color=POOL,
            markeredgecolor="white", label="Overflow abort")
    ax.set_ylim(0, 22)
    ax.set_xlim(0, 480)
    ax.set_xlabel("Time in the measured window (s)", color=INK_2)
    ax.set_ylabel("In-flight footprint\n(k tokens)", color=INK_2)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), labelcolor=INK_2, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3-abort-anatomy.png", dpi=200)
    plt.close(fig)


POLICY_SHORT = {"task_level_fifo": "Task-level", "request_level_fifo": "Request-level"}


def describe(name):
    """task_level_fifo-n4-b16384-r0.025-resident -> Task-level, N=4, 16k, 0.025/s."""
    policy, n, budget, rate, cache = name.split("-")
    text = f"{POLICY_SHORT[policy]}, N={n[1:]}, {int(budget[1:]) // 1024}k, {rate[1:]}/s"
    return text if cache == "resident" else f"{text}, {cache}"


def number(value, digits=0, scale=1.0):
    value = float(value)
    return "–" if value != value else f"{value * scale:.{digits}f}"


def header_cell(text):
    """A LaTeX header cell, split onto two lines near the middle when it is long."""
    words = text.split()
    if len(text) <= 10 or len(words) < 2:
        return tex(text)
    best = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
    return f"\\makecell{{{tex(' '.join(words[:best]))}\\\\{tex(' '.join(words[best:]))}}}"


def tex(text):
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("±", "$\\pm$"), ("…", "\\ldots{}"), ("–", "--"), ("×", "$\\times$")]:
        text = text.replace(a, b)
    return text


class Doc:
    """Sections of tables, rendered both as Markdown and as the report's LaTeX appendix."""

    def __init__(self, title):
        self.title = title
        self.blocks = []

    def section(self, title, label, note=None):
        self.blocks.append(("section", title, label, note))

    def table(self, header, rows):
        self.blocks.append(("table", header, [[str(c) for c in row] for row in rows]))

    def markdown(self):
        out = [f"# {self.title}\n"]
        for block in self.blocks:
            if block[0] == "section":
                _, title, _, note = block
                out.append(f"## {title}\n")
                if note:
                    out.append(f"{note}\n")
            else:
                _, header, rows = block
                out.append("| " + " | ".join(header) + " |")
                out.append("|" + " --- |" * len(header))
                out.extend("| " + " | ".join(row) + " |" for row in rows)
                out.append("")
        return "\n".join(out) + "\n"

    def latex(self):
        out = ["% Generated by scripts/phase1_analysis.py; do not edit by hand.\n"]
        for block in self.blocks:
            if block[0] == "section":
                _, title, label, note = block
                out.append(f"\\subsection{{{tex(title)}}}\\label{{app:{label}}}\n")
                if note:
                    out.append(f"{tex(note)}\n")
            else:
                _, header, rows = block
                lead = min(2, len(header))
                spec = "l" * lead + "r" * (len(header) - lead)
                body = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
                        " & ".join(header_cell(h) for h in header) + " \\\\", "\\midrule"]
                body += [" & ".join(tex(c) for c in row) + " \\\\" for row in rows]
                body += ["\\bottomrule", "\\end{tabular}"]
                inner = "\n".join(body)
                out.append(f"\\begin{{center}}\\small\n\\begin{{adjustbox}}{{max width=\\linewidth}}\n"
                           f"{inner}\n\\end{{adjustbox}}\n\\end{{center}}\n")
        return "\n".join(out)


def tables(cfg):
    doc = Doc("Phase 1 tables (generated by scripts/phase1_analysis.py)")

    doc.section("Tuning at 0.1 agent tasks/s, resident", "tuning",
                "Every concurrency setting tried for each baseline and memory budget. "
                "Latency covers completed agent tasks only.")
    rows = []
    for policy in LABELS:
        for budget in (16384, 32768):
            for n in (1, 2, 4, 8, 16):
                r = cfg.get((policy, n, budget, 0.1, "resident"))
                if r:
                    rows.append([
                        LABELS[policy], f"{budget // 1024}k", n, r["machine"], fmt(r, "goodput_tasks_per_min"),
                        fmt(r, "throughput_tasks_per_min"), fmt(r, "failure_rate.timeout", 0, 100),
                        fmt(r, "failure_rate.pool_exhausted", 0, 100), fmt(r, "completion_latency_p50_s", 1),
                        fmt(r, "completion_latency_p95_s", 1), fmt(r, "queueing_mean_s", 1),
                    ])
    doc.table(["Baseline", "Budget", "N", "Mach.", "Goodput/min", "Throughput/min", "Timeout %", "Pool %",
               "p50 s", "p95 s", "Queue s"], rows)

    doc.section("Arrival-rate ladder at best N", "ladder")
    rows = []
    for policy, n, budget in [("task_level_fifo", 4, 16384), ("request_level_fifo", 8, 16384),
                              ("task_level_fifo", 16, 32768), ("request_level_fifo", 16, 32768)]:
        for rate in (0.025, 0.05, 0.1):
            r = cfg[(policy, n, budget, rate, "resident")]
            if policy == "task_level_fifo" and n == 4:
                r = cfg[(policy, n, budget, rate, "resident", "B")]
            rows.append([LABELS[policy], f"{budget // 1024}k", n, rate, r["machine"], fmt(r, "goodput_tasks_per_min"),
                         fmt(r, "slo_attainment", 0, 100), fmt(r, "failure_rate", 0, 100),
                         fmt(r, "completion_latency_p95_s", 1), fmt(r, "cached_token_fraction", 0, 100)])
    doc.table(["Baseline", "Budget", "N", "Rate/s", "Mach.", "Goodput/min", "SLO %", "Failed %", "p95 s",
               "Cached %"], rows)

    doc.section("Cache policies", "cache", "Each comparison stays on one machine.")
    rows = []
    for policy, n, budget, rate, cache in [
        ("task_level_fifo", 4, 16384, 0.1, "resident"), ("task_level_fifo", 4, 16384, 0.1, "off"),
        ("request_level_fifo", 4, 16384, 0.1, "resident"), ("request_level_fifo", 4, 16384, 0.1, "off"),
        ("task_level_fifo", 4, 16384, 0.025, "resident"), ("task_level_fifo", 4, 16384, 0.025, "offload"),
        ("task_level_fifo", 4, 16384, 0.05, "resident"), ("task_level_fifo", 4, 16384, 0.05, "offload"),
        ("task_level_fifo", 16, 32768, 0.05, "resident"), ("task_level_fifo", 16, 32768, 0.05, "offload"),
        ("task_level_fifo", 16, 32768, 0.1, "resident"), ("task_level_fifo", 16, 32768, 0.1, "offload"),
    ]:
        machine = "A" if rate == 0.1 and budget == 16384 else "B"  # each comparison stays on one machine
        r = cfg[(policy, n, budget, rate, cache, machine)]
        rows.append([LABELS[policy], f"{budget // 1024}k", n, rate, cache, r["machine"],
                     fmt(r, "goodput_tasks_per_min"), fmt(r, "cached_token_fraction", 0, 100),
                     fmt(r, "completion_latency_p50_s", 1), fmt(r, "peak_host_ram_mib", 0),
                     fmt(r, "peak_vram_mib", 0)])
    doc.table(["Baseline", "Budget", "N", "Rate/s", "Cache policy", "Mach.", "Goodput/min", "Cached %", "p50 s",
               "Host RAM MiB", "VRAM MiB"], rows)

    mechanisms = []
    for machine, name, budget in [
        ("B", "task_level_fifo-n4-b16384-r0.025-resident", 16384),
        ("B", "task_level_fifo-n4-b16384-r0.05-resident", 16384),
        ("A", "task_level_fifo-n4-b16384-r0.1-resident", 16384),
        ("A", "request_level_fifo-n8-b16384-r0.1-resident", 16384),
        ("B", "task_level_fifo-n16-b32768-r0.05-resident", 32768),
        ("B", "task_level_fifo-n16-b32768-r0.1-resident", 32768),
        ("A", "task_level_fifo-n4-b32768-r0.1-resident", 32768),
        ("A", "request_level_fifo-n16-b8192-r0.1-resident", 8192),
    ]:
        mechanisms.append((machine, name, budget, mechanism(machine, name)))

    doc.section("Overflow events (from raw logs)", "overflow",
                "Footprints are the proxy's projections (prompt plus n_predict), not the runtime's actual "
                "occupancy. Output figures include output generated during the drain, divided by the arrival window. "
                "Recorded output counts the tokens returned by model requests that completed. A request cut off by an abort returns nothing, so the tokens it had already generated are not counted.")
    rows = []
    for machine, name, budget, m in mechanisms:
        ev, ev_ci = m["events"]
        adm, _ = m["mean_admitted"]
        out, out_ci = m["output_per_s"]
        waste, waste_ci = m["wasted_output"]
        aborted = m["in_flight_at_abort"] == m["in_flight_at_abort"]
        rows.append([describe(name), machine, f"{ev:.1f} ± {ev_ci:.1f}",
                     f"{m['victims']:.1f}" if aborted else "–",
                     f"{m['in_flight_at_abort'] / budget * 100:.0f}" if aborted else "–",
                     f"{m['retained_at_abort']:.0f}" if aborted else "–",
                     f"{adm / budget * 100:.0f}", f"{out:.0f} ± {out_ci:.0f}",
                     f"{waste * 100:.0f} ± {waste_ci * 100:.0f}"])
    doc.table(["Configuration", "Mach.", "Aborts/run", "Failed requests/abort", "In-flight at abort % of budget",
               "Retained at abort, tokens", "Mean admitted % of budget", "Output tok per window-s",
               "Recorded output on failed tasks %"], rows)

    doc.section("Over-budget footprint spells and aborts", "spells",
                "A spell is a stretch of time in which the projected in-flight footprint exceeds the budget. "
                "Counts are summed over all five seeds. Each model request counts at its full projected size from "
                "admission, so aborts are likely to fall inside a spell by construction; these counts show the "
                "signal is consistent with the aborts, not that holding requests back would prevent them.")
    rows = [[describe(name), machine, m["spells"], m["spells_aborted"], m["aborts_measured"], m["aborts_under_budget"]]
            for machine, name, budget, m in mechanisms]
    doc.table(["Configuration", "Mach.", "Over-budget spells", "Spells ending in abort", "Aborts in window",
               "Aborts without a spell"], rows)

    doc.section("Output per arrival-window second at 0.1 agent tasks/s, resident", "output",
                "Highest six configurations. All output generated for the measured agent tasks, drain included, "
                "divided by the arrival window. This is not a steady-state capacity.")
    rows = []
    for machine in ("A", "B"):
        for run_dir in sorted(RUNS[machine].iterdir()):
            name = run_dir.name.rsplit("-s", 1)[0]
            excluded = machine == "A" and name == "request_level_fifo-n16-b32768-r0.1-resident"
            if run_dir.name.endswith("-s1") and name.endswith("-r0.1-resident") and not excluded:
                rate, ci = mechanism(machine, name)["output_per_s"]
                rows.append((rate, ci, name, machine))
    doc.table(["Configuration", "Mach.", "Output tokens per arrival-window s"],
              [[describe(n), m, f"{r:.0f} ± {c:.0f}"] for r, c, n, m in sorted(rows, reverse=True)[:6]])

    doc.section("By task class at best N, 0.1 agent tasks/s, resident", "classes",
                "Means across five seeds. p50 covers completed agent tasks only (– where none completed).")
    rows = []
    wanted = {("task_level_fifo", "4", "16384", "0.1"): "A", ("task_level_fifo", "16", "32768", "0.1"): "B"}
    for machine in ("A", "B"):
        with open(ROOT / f"report-{machine.lower()}" / "classes.csv", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["policy"], row["concurrency"], row["memory_budget_tokens"], row["arrival_rate_per_s"])
                if wanted.get(key) != machine or row["cache_policy"] != "resident":
                    continue
                config = (f"{POLICY_SHORT[row['policy']]}, N={row['concurrency']}, "
                          f"{int(row['memory_budget_tokens']) // 1024}k")
                rows.append([config, row["task_class"], machine, number(row["agent_tasks_mean"], 1),
                             number(row["failure_rate_mean"], 0, 100), number(row["slo_attainment_mean"], 0, 100),
                             number(row["completion_latency_p50_s_mean"], 1)])
    doc.table(["Configuration", "Task class", "Mach.", "Tasks/run", "Failed %", "SLO %", "p50 s"], rows)

    doc.section("8k overflow cliff by task class", "cliff",
                "Request-level FIFO, N=16, 8k, 0.1 agent tasks/s, machine A.")
    rows = []
    with open(ROOT / "report-a" / "classes.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["memory_budget_tokens"] == "8192":
                fits = row.get("fits_budget_mean")
                rows.append([row["task_class"], number(row["agent_tasks_mean"], 1),
                             number(row["failure_rate_mean"], 0, 100),
                             "yes" if fits and float(fits) == 1 else "no"])
    doc.table(["Task class", "Tasks/run", "Failed %", "Fits budget alone"], rows)

    doc.section("Unloaded completion time by task class", "unloaded",
                "Machine A, reference setup (resident, caching on, 32k), the agent tasks of all five seeds "
                "of one configuration's workload.")
    by_class = {}
    for seed in SEEDS:
        for u in read_jsonl(RUNS["A"] / f"task_level_fifo-n4-b16384-r0.1-resident-s{seed}" / "unloaded.jsonl"):
            if u["unloaded_completion_s"] is not None:
                by_class.setdefault(u["task_class"], []).append(u["unloaded_completion_s"])
    doc.table(["Task class", "Tasks", "Median s"],
              [[c, len(v), f"{st.median(v):.1f}"] for c, v in sorted(by_class.items())])

    doc.section("Cold start", "coldstart", "Medians over every server launch.")
    rows = []
    for machine in ("A", "B"):
        with open(ROOT / f"report-{machine.lower()}" / "cold_start.csv", newline="") as f:
            launches = [r for r in csv.DictReader(f) if r["role"] == "cold_start"]
        startup = [float(r["startup_s"]) for r in launches if r["startup_s"]]
        first = [float(r["first_request_s"]) for r in launches if r["first_request_s"]]
        warm = [float(r["warm_request_s"]) for r in launches if r["warm_request_s"]]
        rows.append([machine, len(startup), f"{st.median(startup):.2f}", f"{st.median(first):.2f}",
                     f"{st.median(warm):.2f}"])
    doc.table(["Machine", "Launches", "Launch to healthy s", "First probe s", "Warm probe s"], rows)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="results/phase1", help="extracted results and reports")
    parser.add_argument("--tables", default="docs/reports/phase1-tables.md", help="Markdown output")
    parser.add_argument("--latex", default="docs/reports/phase1-appendix.tex", help="LaTeX appendix output")
    args = parser.parse_args()
    set_root(args.root)
    FIGURES.mkdir(parents=True, exist_ok=True)
    cfg = load_configurations()
    figure_tuning(cfg)
    figure_failures(cfg)
    figure_rates(cfg)
    figure_abort_anatomy()
    doc = tables(cfg)
    # UTF-8 explicitly: the tables contain "±", which the Windows default code page mangles
    Path(args.tables).write_text(doc.markdown(), encoding="utf-8")
    Path(args.latex).write_text(doc.latex(), encoding="utf-8")


if __name__ == "__main__":
    main()
