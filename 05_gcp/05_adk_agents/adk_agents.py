"""ADK: agent types, composition, and where an agent actually runs.

The Agent Development Kit is the framework you WRITE an agent in. Agent
Runtime (formerly Agent Engine) is one of the places you can RUN it. Those
are different decisions and conflating them is the most common confusion
in this area, so section 1 separates them before anything else.

The through-line of 05_gcp applies here too: pick the least custom option
that fits. For agents the question becomes "who decides control flow - the
model, or you?", and the answer changes cost, latency and auditability.

  1. ADK vs Agent Runtime - framework and host are separate choices
  2. The three agent families, and the field that gives the game away
  3. Choosing an agent type - who decides control flow
  4. Tools - how an agent does anything at all
  5. Passing data between agents - output_key and session state
  6. MEASURED: sequential vs parallel vs loop, with real wall-clock
  7. Deployment - the four doors, three of them managed
  8. Sessions, memory, artifacts - what breaks when you deploy
  9. What the exam does and does not ask

NOT EXECUTED, with one exception. Every ADK and Vertex import sits inside
a function, so this file runs with no credentials, no project and no
google-adk installed. No model is ever called.

Section 6 DOES execute: it is a pure-asyncio simulation of the
orchestration semantics, with stub coroutines standing in for agents. It
measures the wall-clock difference between the patterns, not ADK itself.

Every class and field named here was verified against google-adk 1.32.0 by
introspection rather than recalled.

Run:
    python 05_gcp/05_adk_agents/adk_agents.py

Concept reference: Foundations volume, Part E (the model-building ladder).
"""

from __future__ import annotations

import asyncio
import random
import time

PROJECT = "your-project-id"
REGION = "us-central1"
BUCKET = "gs://your-bucket"
MODEL = "gemini-2.5-flash"


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ============================================ 1. framework vs host
def framework_vs_host() -> None:
    banner("1. ADK is a framework. Agent Runtime is a host. Separate choices.")

    print("  ADK           - an open-source Python framework. Defines agents,")
    print("                  tools, orchestration and state. Runs anywhere")
    print("                  Python runs, including your laptop.")
    print("  Agent Runtime - a managed Google Cloud service that hosts an")
    print("                  agent, with sessions and memory provided.")
    print("                  Formerly Vertex AI Agent Engine.")

    print("\n  They are genuinely independent, and this is checkable rather")
    print("  than a claim. `vertexai.agent_engines` exposes wrappers for:")
    print("      AdkApp, LangchainAgent, LanggraphAgent, AG2Agent, ModuleAgent")
    print("  - so the managed host takes non-ADK frameworks too. Equally, an")
    print("  ADK agent deploys perfectly well to Cloud Run or GKE.")

    print("\n  Consequence: 'should I use ADK?' and 'should I use Agent")
    print("  Runtime?' are two questions. Answer them separately.")

    print("\n  A rename worth knowing, because it leaks: Agent Engine became")
    print("  Agent Runtime, but the underlying API resource is STILL")
    print("  `reasoningEngines`. Resource names look like")
    print("      projects/<p>/locations/<r>/reasoningEngines/<id>")
    print("  The API endpoint is still aiplatform.googleapis.com. Expect the")
    print("  old name in IAM roles, URLs and error messages.")


# ============================================ 2. the three families
def agent_families() -> None:
    banner("2. Three agent families")

    rows = [
        ("LlmAgent", "a model decides what happens next",
         "flexible, non-deterministic"),
        ("Workflow agents", "YOU decide; no model in the control flow",
         "deterministic, auditable, cheaper"),
        ("Custom (BaseAgent)", "subclass and write _run_async_impl",
         "arbitrary logic, full control"),
    ]
    print(f"  {'family':<20} {'who drives control flow':<38} character")
    print("  " + "-" * 88)
    for a, b, c in rows:
        print(f"  {a:<20} {b:<38} {c}")

    print("\n  The three workflow agents:")
    print("      SequentialAgent - sub-agents in order, each sees the last one's state")
    print("      ParallelAgent   - sub-agents concurrently, independent branches")
    print("      LoopAgent       - repeat until max_iterations or an escalation")

    print("\n  THE FIELD THAT GIVES THE GAME AWAY")
    print("  Introspecting google-adk 1.32.0 for the declared fields on each")
    print("  class, the workflow agents carry exactly:")
    print("      name, description, sub_agents, parent_agent, and 2 callbacks")
    print("  ...and LoopAgent adds `max_iterations`. That is the whole list.")
    print("\n  There is NO `model` field on SequentialAgent, ParallelAgent or")
    print("  LoopAgent. LlmAgent has one, along with `instruction`, `tools`,")
    print("  `planner`, `output_key` and both `disallow_transfer_*` flags.")
    print("\n  So 'workflow agents contain no LLM' is not a simplification for")
    print("  teaching - it is structural. They cannot call a model, because")
    print("  there is nowhere to put one.")


# ============================================ 3. choosing a type
def choosing() -> None:
    banner("3. Choosing an agent type")

    rows = [
        ("the next step depends on what the user said", "LlmAgent"),
        ("fixed pipeline: draft, then review, then format", "SequentialAgent"),
        ("N independent lookups, then combine", "ParallelAgent"),
        ("refine until a quality bar is met", "LoopAgent"),
        ("route to one of 4 known specialists", "LlmAgent with sub_agents"),
        ("route on a field you can read yourself", "Sequential, or custom"),
        ("anything genuinely irregular", "custom BaseAgent"),
    ]
    print(f"  {'situation':<48} reach for")
    print("  " + "-" * 76)
    for a, b in rows:
        print(f"  {a:<48} {b}")

    print("\n  THE COST ARGUMENT, which decides more of these than elegance:")
    print("  every LlmAgent decision is a model call. A SequentialAgent of")
    print("  three LlmAgents is at least three calls, paid and waited for.")
    print("  Letting a model choose between two branches you could select")
    print("  with an `if` buys flexibility you may not need, at the price of")
    print("  a round trip and a non-deterministic answer.")

    print("\n  This is the same instinct as the rest of 05_gcp - pick the")
    print("  least custom option that fits - applied one level up. Replacing")
    print("  model-decided routing with a workflow agent is the agent-shaped")
    print("  version of 'do you actually need custom training?'")

    print("\n  Two ways to build a multi-agent system, worth not confusing:")
    print("    - LLM-driven delegation: give an LlmAgent `sub_agents` and it")
    print("      TRANSFERS control based on their `description` fields. Those")
    print("      descriptions become routing logic - write them accordingly.")
    print("      `disallow_transfer_to_parent` / `_to_peers` constrain it.")
    print("    - Agent-as-a-tool: wrap with AgentTool and the caller gets the")
    print("      RESULT back and keeps control. No transfer.")


# ============================================ 4. tools
def tools() -> None:
    banner("4. Tools - how an agent does anything at all")

    rows = [
        ("Function tools", "FunctionTool, LongRunningFunctionTool",
         "your own Python; the docstring is the spec"),
        ("Google-built", "BigQueryToolset, VertexAiSearchTool",
         "managed integrations"),
        ("External / standard", "MCPToolset, APIHubToolset",
         "MCP servers and registered APIs"),
        ("Agents as tools", "AgentTool",
         "call an agent, keep control"),
    ]
    print(f"  {'kind':<22} {'class in 1.32':<52} what it is")
    print("  " + "-" * 100)
    for a, b, c in rows:
        print(f"  {a:<22} {b:<52} {c}")

    print("\n  Also present in 1.32: DiscoveryEngineSearchTool, ExampleTool,")
    print("  TransferToAgentTool, and BaseTool to write your own.")

    print("\n  For a function tool the DOCSTRING AND TYPE HINTS ARE THE API")
    print("  CONTRACT - they are what the model sees when deciding whether")
    print("  and how to call it. A vague docstring is a bug, not a style")
    print("  issue: it is the difference between the tool being used")
    print("  correctly, used wrongly, or ignored.")

    print("\n  `ToolContext` is how a tool reaches the surrounding machinery -")
    print("  session state, artifacts, and `actions.escalate`, which is what")
    print("  a tool sets to break out of a LoopAgent early.")


def define_function_tool():
    """A function tool is an ordinary Python function. NOT CALLED."""
    from google.adk.tools import FunctionTool

    def weekly_apps(store_number: str, fiscal_week: str) -> dict:
        """Return credit application counts for one store and fiscal week.

        Args:
            store_number: four-digit store number, zero padded, e.g. "0121".
            fiscal_week: fiscal week id, e.g. "202634".

        Returns:
            A dict with keys `store_number`, `fiscal_week` and `applications`.
        """
        return {"store_number": store_number,
                "fiscal_week": fiscal_week,
                "applications": 0}

    return FunctionTool(func=weekly_apps)


def define_llm_agent():
    """The reasoning agent. NOT CALLED - no model is invoked."""
    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="reporting_analyst",
        model=MODEL,
        description=(
            "Answers questions about weekly application counts by store. "
            "Use for any question naming a store or a fiscal week."
        ),
        instruction=(
            "You answer questions about weekly application counts.\n"
            "Always call the weekly_apps tool rather than estimating.\n"
            "If the question does not name a store, ask for one."
        ),
        tools=[define_function_tool()],
        output_key="analyst_answer",     # writes result into session state
    )


def define_workflow_agents():
    """Composition. NOT CALLED."""
    from google.adk.agents import (LlmAgent, LoopAgent, ParallelAgent,
                                   SequentialAgent)

    drafter = LlmAgent(name="drafter", model=MODEL,
                       instruction="Draft a summary.", output_key="draft")
    critic = LlmAgent(name="critic", model=MODEL,
                      instruction="Critique the draft in {draft}.",
                      output_key="critique")

    # Sequential: critic can read {draft} because drafter wrote it first.
    review = SequentialAgent(name="draft_then_review",
                             sub_agents=[drafter, critic])

    # Parallel: branches must be INDEPENDENT. Neither may read the other's
    # output_key, because there is no ordering guarantee between them.
    fan_out = ParallelAgent(
        name="gather",
        sub_agents=[
            LlmAgent(name="sales", model=MODEL, output_key="sales"),
            LlmAgent(name="traffic", model=MODEL, output_key="traffic"),
        ],
    )

    # Loop: bounded by max_iterations, or ended early by a tool setting
    # tool_context.actions.escalate = True.
    refine = LoopAgent(name="refine", sub_agents=[drafter, critic],
                       max_iterations=3)

    return review, fan_out, refine


# ============================================ 5. state
def state() -> None:
    banner("5. Passing data between agents")

    print("  Sub-agents do not return values to each other. They share")
    print("  SESSION STATE, and the wiring is two fields:")
    print("      output_key='draft'   on the producer - writes state['draft']")
    print("      instruction='...{draft}...' on the consumer - reads it")

    print("\n  That templating is the entire data flow between agents, which")
    print("  makes one rule load-bearing:")
    print("\n    A ParallelAgent's branches MUST NOT read each other's")
    print("    output_key. They run concurrently with no ordering guarantee,")
    print("    so a branch reading a key another branch is still writing is")
    print("    a race - sometimes empty, sometimes stale, rarely an error.")

    print("\n  If branch B needs branch A's output, that is a Sequential")
    print("  dependency and using ParallelAgent is simply wrong. Section 6")
    print("  shows the completion order is not the declared order.")


# ====================================== 6. MEASURED orchestration
async def _stub_agent(name: str, seconds: float, out: list) -> str:
    """Stand-in for an LlmAgent call: I/O-bound, no CPU, no model."""
    await asyncio.sleep(seconds)
    out.append(name)
    return name


async def _sequential(branches, order):
    for name, secs in branches:
        await _stub_agent(name, secs, order)


async def _parallel(branches, order):
    await asyncio.gather(*(_stub_agent(n, s, order) for n, s in branches))


async def _loop(max_iterations: int, threshold: float, rng: random.Random):
    """A LoopAgent ends on max_iterations OR an escalation from inside."""
    for i in range(1, max_iterations + 1):
        await asyncio.sleep(0.05)
        score = rng.random()
        if score >= threshold:          # tool sets actions.escalate = True
            return i, score, "escalated"
    return max_iterations, score, "hit max_iterations"


def measured() -> None:
    banner("6. MEASURED - sequential vs parallel vs loop")

    branches = [("sales", 0.45), ("traffic", 0.15), ("labour", 0.30)]

    print("  Three stub agents, deliberately UNEVEN - real model calls are:")
    print("      " + ",  ".join(f"{n} {s:.2f} s" for n, s in branches))
    print("  I/O-bound (asyncio.sleep), standing in for model calls, which")
    print("  are also I/O-bound - so the comparison transfers. No model is")
    print("  called and ADK is not used; this measures the ORCHESTRATION")
    print("  SEMANTICS only.\n")

    seq_order: list[str] = []
    t0 = time.perf_counter()
    asyncio.run(_sequential(branches, seq_order))
    seq_s = time.perf_counter() - t0

    par_order: list[str] = []
    t0 = time.perf_counter()
    asyncio.run(_parallel(branches, par_order))
    par_s = time.perf_counter() - t0

    total = sum(s for _, s in branches)
    slowest = max(branches, key=lambda b: b[1])

    print(f"  {'pattern':<14} {'wall-clock':<14} completion order")
    print("  " + "-" * 62)
    print(f"  {'Sequential':<14} {seq_s:>8.2f} s     {' -> '.join(seq_order)}")
    print(f"  {'Parallel':<14} {par_s:>8.2f} s     {' -> '.join(par_order)}")
    print(f"\n  Speedup: {seq_s / par_s:.1f}x")

    print("\n  Two things to read off that, neither of them 'parallel is Nx")
    print("  faster':")
    print(f"\n  1. Sequential is the SUM of the branches ({total:.2f} s);")
    print(f"     parallel is the MAX ({slowest[1]:.2f} s). So the speedup is")
    print("     sum/max, which depends on how UNEVEN the branches are, not")
    print("     on how many there are. One slow branch erases the benefit:")
    print(f"     '{slowest[0]}' alone sets the floor here, and adding ten fast")
    print("     branches beside it would not move the total.")
    print("\n  2. The parallel completion order is NOT the declared order -")
    print("     it is finish order, fastest first. That is exactly why a")
    print("     parallel branch may not read another's output_key (section")
    print("     5): there is no ordering to rely on.")

    rng = random.Random(0)
    print("\n  LoopAgent, max_iterations=5, escalating at score >= 0.80:")
    for trial in range(1, 4):
        iters, score, why = asyncio.run(_loop(5, 0.80, rng))
        print(f"    trial {trial}:  {iters} iteration(s), "
              f"final score {score:.2f}  -> {why}")

    print("\n  Same agent, same bound, different iteration counts - the loop")
    print("  is data-dependent by design. `max_iterations` is not a tuning")
    print("  knob, it is the guarantee that a loop whose escalation never")
    print("  fires still terminates. Omit the escalation and you always pay")
    print("  the maximum; omit the bound and a bad run costs unboundedly.")


# ============================================ 7. deployment
def deployment() -> None:
    banner("7. Deployment - the four doors")

    print("  `adk deploy` in 1.32.0 offers exactly three targets:")
    print("      agent_engine, cloud_run, gke")
    print("  plus local running via `adk run`, `adk web`, `adk api_server`.\n")

    rows = [
        ("Local", "adk web / adk run", "everything, on your laptop",
         "development only"),
        ("Agent Runtime", "adk deploy agent_engine", "nothing - fully managed",
         "the GCP default for production"),
        ("Cloud Run", "adk deploy cloud_run", "a container and its scaling",
         "custom deps, scale to zero"),
        ("GKE", "adk deploy gke", "a cluster",
         "you already run GKE"),
    ]
    print(f"  {'door':<15} {'command':<26} {'you manage':<28} when")
    print("  " + "-" * 100)
    for a, b, c, d in rows:
        print(f"  {a:<15} {b:<26} {c:<28} {d}")

    print("\n  Agent Runtime is the managed door: sessions, memory and")
    print("  scaling are provided rather than assembled. The cost is the")
    print("  usual managed-service cost - less control, and a Google-shaped")
    print("  deployment model.")

    print("\n  Cloud Run is the interesting middle. It is an ordinary")
    print("  container, so any dependency you can install works, and it")
    print("  SCALES TO ZERO. Contrast the Vertex prediction endpoints in")
    print("  04_custom_training, where min_replica_count cannot be 0 and you")
    print("  pay per node-hour at rest. If an agent is called rarely, that")
    print("  difference dominates every other consideration.")

    print("\n  GKE is for teams already on GKE. Choosing it for an agent")
    print("  alone means adopting a cluster to run one service.")


def deploy_to_agent_runtime():
    """Managed deployment. NOT EXECUTED - nothing is created."""
    import vertexai
    from vertexai import agent_engines
    from vertexai.preview import reasoning_engines

    vertexai.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)

    # AdkApp wraps an ADK agent into the deployable shape.
    app = reasoning_engines.AdkApp(agent=define_llm_agent(),
                                   enable_tracing=True)

    return agent_engines.create(
        agent_engine=app,
        requirements=["google-adk", "google-cloud-bigquery"],
        display_name="reporting-analyst",
    )


# ============================================ 8. what breaks
def what_breaks() -> None:
    banner("8. Sessions, memory, artifacts - what breaks when you deploy")

    rows = [
        ("Sessions", "InMemorySessionService", "VertexAiSessionService",
         "the current conversation"),
        ("Memory", "InMemoryMemoryService", "VertexAiMemoryBankService",
         "recall across conversations"),
        ("Artifacts", "InMemoryArtifactService",
         "GcsArtifactService", "files the agent produces"),
    ]
    print(f"  {'service':<12} {'development':<26} {'production':<26} holds")
    print("  " + "-" * 96)
    for a, b, c, d in rows:
        print(f"  {a:<12} {b:<26} {c:<26} {d}")

    print("\n  Memory has a second managed option, VertexAiRagMemoryService,")
    print("  backed by a RAG corpus rather than Memory Bank - the choice is")
    print("  whether recall should be retrieval over documents or a curated")
    print("  store of facts about the user.")

    print("\n  THE FAILURE THIS CAUSES, and it is the common one:")
    print("  `InMemoryRunner` gives you InMemory everything, which is")
    print("  correct locally and silently wrong once deployed. State lives in")
    print("  one process, so it is lost on restart and NOT SHARED BETWEEN")
    print("  REPLICAS. On Cloud Run with more than one instance, a user's")
    print("  second message can land on an instance that has never heard of")
    print("  them - intermittently, and only under load.")

    print("\n  Nothing raises. The agent simply behaves as though the")
    print("  conversation never happened, for some fraction of requests.")
    print("  Swap the service when you deploy, not when you notice.")

    print("\n  Sessions and memory are different things: a session is THIS")
    print("  conversation, memory is what survives across them. Wanting")
    print("  'the agent to remember me next week' is memory, and a session")
    print("  service will not do it however long you keep the session.")


# ============================================ 9. exam angle
def exam_angle() -> None:
    banner("9. What the exam does and does not ask")

    print("  Said plainly, because it affects how much time this deserves:")
    print("  ADK IS NOT ON THE ML ENGINEER EXAM as of the June 2026 guide.")
    print("  Neither is Agent Garden, MCP, A2A, Memory Bank, Agent Registry")
    print("  or Agent Retrieval. Despite the Vertex AI -> Agent Platform")
    print("  rename, the stated objectives remain model-centric.")

    print("\n  Named gen-AI items on that guide are: Model Garden,")
    print("  foundation-model tuning, grounding, Gen AI Evaluation and")
    print("  LLM-as-a-judge, Model Armor and safety filters, Gemini")
    print("  cost/latency, Model Registry, endpoints, batch inference.")

    print("\n  So this file is here for practice, not for points - agents are")
    print("  what the platform is being built around, and the deployment")
    print("  trade-offs in section 7 are ordinary engineering judgement that")
    print("  outlasts any one exam guide.")

    print("\n  The exam-transferable part is section 7's cost shape, which is")
    print("  the same lesson as the rest of 05_gcp: scale-to-zero versus")
    print("  billed-at-rest decides architectures more often than capability.")


def main() -> None:
    print("=" * 72)
    print("ADK: agent types, composition, and deployment".center(72))
    print("=" * 72)
    framework_vs_host()
    agent_families()
    choosing()
    tools()
    state()
    measured()
    deployment()
    what_breaks()
    exam_angle()

    banner("Summary")
    print("  - ADK is the framework, Agent Runtime one of four hosts.")
    print("  - Workflow agents have no `model` field; that is structural.")
    print("  - Agent type = who decides control flow, and what it costs.")
    print("  - Parallel time is the MAX of its branches, sequential the SUM -")
    print("    measured above - and parallel needs independent branches.")
    print("  - Cloud Run scales to zero; a Vertex endpoint does not.")
    print("  - InMemory* services are correct locally and wrong deployed.")


if __name__ == "__main__":
    main()
