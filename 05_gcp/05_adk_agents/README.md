# ADK agents

> **Concept reference:** Success Guide, Part 7.8+ · Blueprint sections 1 and 6
> **Runtime:** ~4 s · ✅ **this one actually runs** (locally, no cloud)

```bash
python 05_gcp/05_adk_agents/adk_agents.py
```

---

## The idea

**ADK is the framework you *write* an agent in. Agent Runtime is one of the places you can *run*
it.** Those are separate decisions, and conflating them is the most common confusion in this area.

The through-line of `05_gcp` still applies: **pick the least custom option that fits.** For agents
the question becomes *"who decides control flow — the model, or you?"*, and the answer changes
cost, latency and auditability.

## The three agent families

| family | control flow decided by | use when |
|---|---|---|
| **LLM agent** | **the model** | the path genuinely varies with the input |
| **Workflow agent** | **you** — Sequential, Parallel, Loop | the steps are known in advance |
| **Custom agent** | your code | neither fits |

> **A workflow agent is not a lesser agent.** If you know the steps, deciding them yourself is
> cheaper, faster and auditable. Letting a model re-derive a fixed pipeline on every request is
> the expensive way to get a less reliable answer.

## Measured: sequential vs parallel

Three independent sub-agents with deliberately uneven durations
(`sales` 0.45 s, `traffic` 0.15 s, `labour` 0.30 s):

```
Sequential  0.93 s   ->  sales, traffic, labour      (declared order)
Parallel    0.46 s   ->  traffic, labour, sales      (completion order)
```

**Two things to read here.** Parallel is ~2× faster — the sum versus the max. And its results come
back in **completion order, not declared order**: the shortest finished first.

> An earlier draft of this script asserted that parallel output arrives in declared order. The run
> disagreed. If your downstream code assumes declared order, parallel execution will reorder your
> results the first time durations differ — and they always differ.

## Tools, state, and the four deployment doors

**Tools** are how an agent does anything at all — a Python function, a built-in, another agent, or
an MCP server. **State** passes between agents via `output_key` and session state rather than
return values.

Where an agent can run:

| | |
|---|---|
| **Agent Runtime** (formerly Agent Engine) | managed, sessions and memory included |
| **Cloud Run** | a container you control, scale to zero |
| **GKE** | maximum control, existing estate |
| **locally** | development — what this script does |

## Monitoring an agent is different

Traditional model monitoring does not fit. **Trajectory** — did it call the right tools in the
right order? — is the metric that has no model equivalent.

> **A correct final answer reached by the wrong path is still a defect.** It will not generalise.
> See [`08_monitoring`](../../08_monitoring/).

## What was actually run

✅ **This script executes in full**, locally. No project, no credentials, no billing — and **no
ADK**: `google-adk` is not installed in the repository's `.venv`, and the script never imports it
at runtime. The ADK snippets (the function tool, the `LlmAgent`, the workflow agents, the
deployment call) sit in functions that are reference code in the file and are never called; the
measured section is pure `asyncio` standing in for the orchestration patterns. The class and field
names were checked by introspecting **google-adk 1.32.0** in a separate environment (see the
[parent README](../README.md)), not against this run. The timings above were measured on this
machine.
