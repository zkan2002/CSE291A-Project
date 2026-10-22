---
status: superseded by ADR-0003
---

# Main experiments run on a local desktop GPU, not AWS

The course defaults to small AWS CPU instances, but we run the main experiments on a local RTX 3060 Ti (8 GB VRAM) with CUDA llama.cpp. Faster runs allow 3–5 repetitions per configuration and wider sweeps within the deadlines, and 8 GB of VRAM is a real hard limit. The $50 AWS credit is kept for an optional check that the results hold on a CPU instance. The llama.cpp commit, CUDA version and model file hash are pinned for reproducibility.
