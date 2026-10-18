---
status: accepted
supersedes: ADR-0002
---

# Main experiments run on one RunPod RTX 2000 Ada pod, not the desktop

The desktop's GPU is shared with whatever else the machine runs, which skews latency, contaminates the unloaded measurements that set task SLOs, and lets Windows spill CUDA memory into system RAM without failing. Every experiment that is compared, in Phase 1 and Phase 2, therefore runs on a single RunPod Secure Cloud pod with one RTX 2000 Ada (16 GB) in EU-RO-1, using the same pinned llama.cpp commit and model file built and checked on the pod by `scripts/runpod-setup.sh`. The cheapest available card was chosen because run time is set mainly by the fixed protocol (8 minutes measured per run), not GPU speed, so a faster card would cost more without shortening the sweep much. Results from different machines are never compared directly: if the pod is replaced, the baselines and the tuned N are re-measured on the new one. The desktop remains for development and the simulated runtime.
