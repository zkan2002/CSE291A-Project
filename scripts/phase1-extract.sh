#!/usr/bin/env bash
# Verify the two Phase 1 raw-result archives and extract them into the layout the analysis reads.
#   bash scripts/phase1-extract.sh ARCHIVE_DIR [DEST]      (DEST defaults to results/phase1)
# ARCHIVE_DIR holds pod-results-2026-10-19.tar.gz (machine A) and
# machine-b-final-2026-10-21.tar.gz (machine B); checksums are in docs/reports/phase1-archives.sha256.
set -euo pipefail

ARCHIVES="$(cd "${1:?usage: phase1-extract.sh ARCHIVE_DIR [DEST]}" && pwd)"
DEST="${2:-results/phase1}"
SUMS="$(cd "$(dirname "$0")/.." && pwd)/docs/reports/phase1-archives.sha256"

(cd "$ARCHIVES" && sha256sum -c "$SUMS")

for pair in "machine-a:pod-results-2026-10-19.tar.gz" "machine-b:machine-b-final-2026-10-21.tar.gz"; do
    target="$DEST/${pair%%:*}"
    if [ -e "$target" ] && [ -n "$(ls -A "$target")" ]; then
        echo "$target is not empty; remove it or choose another DEST" >&2
        exit 1
    fi
    mkdir -p "$target"
    # Extract from inside the target, with --force-local so GNU tar on Windows never reads
    # "C:" in a path as a remote host
    (cd "$target" && tar --force-local -xzf "$ARCHIVES/${pair#*:}")
done
echo "extracted into $DEST"
