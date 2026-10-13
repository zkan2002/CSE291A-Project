## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Development

Python 3.11 project managed with uv (`uv` is not on PATH here: invoke it as `python -m uv`). Run tests with `python -m uv run pytest`; tests marked `real_runtime` need a live pinned llama-server and are skipped by default (`-m real_runtime` to opt in).
