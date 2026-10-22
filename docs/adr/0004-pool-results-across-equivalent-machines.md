---
status: accepted
amends: ADR-0003
---

# Phase 1 results come from two RunPod machines, pooled only after an equivalence check

ADR-0003 planned a single RunPod pod and said that if it was replaced, everything would be re-measured on the new one. Partway through Phase 1, the host of machine A (EU-RO-1) rebooted and was then reported as failed. By then 102 runs had completed. The sweep was finished on machine B (EUR-IS-1), which has the same GPU model, llama.cpp commit, model file, workloads and protocol, but a different physical card, driver (595 vs 580), CUDA toolchain and CPU quota. Re-running all 102 completed runs would have taken most of the time left before the deadline, so we pool the two machines' results instead, under these rules:

- **A check before pooling** (`agentserve.equivalence`, run on B before any pooling). On one of A's configurations, re-run with the same seeds, the median per-task unloaded completion time on B over A must be within ±5%, and B's mean for each headline metric must fall inside A's 95% confidence interval.
  - Result: median ratio 1.039, and every metric inside A's interval, so the machines count as equivalent.
  - B is consistently about 4% slower per task, and the report states this.
- **No configuration spans machines.** All of a configuration's seeds run on one machine. A's incomplete request-level N=16 at 32k configuration was discarded and re-run in full on B.
- **Task SLOs and hard timeouts are per machine.** Each machine sets them from its own unloaded measurements, on the reference setup.
- **Pooling happens only where the check supports it:** choosing the best N, and a few points that are compared across machines, each flagged in the report.

If the check had failed, we would have kept ADR-0003's rule and re-run on B everything that is compared. Phase 2 must re-run the same check on its new pod against machine B before comparing its results with Phase 1.
