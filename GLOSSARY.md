# Memory-Aware Agent Serving

A study of how admission and scheduling decisions in front of a local inference runtime affect completed agent work under a bounded memory budget (CSE 291A, Option 1).

## Language

### Workload

**Agent task**:
One end-to-end unit of agent work, made of an ordered sequence of agent steps. Completed work is counted in agent tasks.
_Avoid_: Job, session, conversation

**Agent step**:
One model request followed by the tool delay before the next step; its prompt carries the context accumulated by earlier steps of the same agent task.
_Avoid_: Turn, round, iteration

**Model request**:
A single call to the inference runtime, issued on behalf of one agent step.
_Avoid_: Query, inference call, request (unqualified)

**Task class**:
A kind of agent task in a workload mix, fixed by its starting context length, output length per agent step and range of agent steps.
_Avoid_: Task type, category

**Development workload / evaluation workload**:
Workloads for tuning and for reporting. They come from separate seed streams (the run's `split`), and evaluation can use a held-out mix weighting, so tuning never sees evaluation data.
_Avoid_: Train/test workload

**Tool delay**:
The idle interval between two agent steps of the same agent task while it waits on a simulated tool.
_Avoid_: Think time, gap, pause

### Scheduling

**Task-level admission**:
Admitting an agent task as a whole, so it holds one concurrency slot from arrival to completion, including its tool delays.
_Avoid_: Session admission

**Request-level admission**:
Admitting each model request on its own, so an agent task re-queues for every agent step after its tool delay.
_Avoid_: Step admission, per-call scheduling

### Outcomes

**Unloaded completion time**:
How long an agent task takes when it runs alone on an otherwise idle system; the reference against which its SLO and timeout are scaled.
_Avoid_: Baseline latency, ideal time

**Task SLO**:
A completion-time target for an agent task, set as a fixed multiple of its unloaded completion time.
_Avoid_: Deadline (unqualified), latency target

**Goodput**:
Agent tasks completed within their task SLO, per minute. The headline measure of completed agent work.
_Avoid_: Throughput (which counts all completed tasks regardless of SLO)

**Failed task**:
An agent task with at least one model request that returned an error, or that exceeded its hard timeout. Failed tasks stay in every denominator.
_Avoid_: Dropped task, skipped task

### Memory

**Memory budget**:
The capacity, in tokens, of the runtime's shared KV-cache pool; the independent variable the experiments control. Physical memory use is a measured outcome, not the budget.
_Avoid_: Memory limit, RAM budget, VRAM budget

**Retained context**:
The KV state of an agent task's earlier steps kept across a tool delay so the next agent step can reuse it instead of recomputing it.
_Avoid_: Cached prompt, session state
