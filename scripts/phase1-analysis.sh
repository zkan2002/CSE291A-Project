#!/usr/bin/env bash
# Regenerate every Phase 1 table and figure from extracted raw results (scripts/phase1-extract.sh).
#   bash scripts/phase1-analysis.sh [DEST]                 (DEST defaults to results/phase1)
# Set UV if uv is not on PATH, e.g. UV="py -3.11 -m uv" on the Windows desktop.
set -euo pipefail

DEST="${1:-results/phase1}"
UV="${UV:-uv}"
A="$DEST/machine-a/results/phase1-dev"
B="$DEST/machine-b/results/phase1-dev-b"

# 1. Per-machine reports: configurations.csv, classes.csv, cold_start.csv, metrics.md, plots
$UV run agentserve report "$A" --out "$DEST/report-a"
$UV run agentserve report "$B" --out "$DEST/report-b"
$UV run agentserve report "$DEST/machine-a/results/calibrate-rates" --out "$DEST/report-calibration"

# 2. The equivalence check that justified pooling the two machines (ADR-0004)
$UV run python -m agentserve.equivalence "$A/runs" "$B/runs" --config task_level_fifo-n4-b16384-r0.1-resident

# 3. The best-N choice, recomputed from both reports; it must match the grids in configs/phase1-dev-b.toml
$UV run python -m agentserve.best_n "$DEST/report-a/configurations.csv" "$DEST/report-b/configurations.csv" \
    --rates 0.025 0.05 > /dev/null   # the choices print to stderr; the generated grids are discarded

# 4. The report's tables and figures
$UV run python scripts/phase1_analysis.py --root "$DEST" --tables docs/reports/phase1-tables.md \
    --latex docs/reports/phase1-appendix.tex
echo "wrote docs/reports/phase1-tables.md, phase1-appendix.tex and phase1-figures/"
