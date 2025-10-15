This lesson is the second lesson of part 2 of a course where we're teaching how to build AI agents for production.

The lesson is titled "Agent Frameworks Overview & Comparison".
Basically, this course has a part 1 where it teaches basic AI agent concepts.
Then, in part 2, the course goes more into practice and, step by step, teaches how to build two agents: a research agent and a writer agent.
The research agent is an actual agent that works with an MCP server and an MCP client, using `fastmcp`.
The writer agent is more like a workflow agent that works with `langgraph`.

So, in the two projects, we see the two main types of agents that are used in production:
- An adaptable and steerbable research agent that works with an MCP server, which can easily diverge from its main workflow to research different topics and adapt.
- A reliable writer agent whose task is better defined but harder, which needs less adaptation but more reliability.

In this lesson, we provide an overview of the main agents frameworks/libraries, and we compare them. And we also provide some insights on the experience that we had while learning about the topic and building the agents for the capstone project and in our other projects.

- When we started working on the course, we weren’t sure about whether to use agents frameworks/libraries (and which ones) or not. The field is evolving very quickly, so frameworks/libraries can rise up and down just as quickly. It’s to early to predict which will be the ones that will last for multiple years and will become the standard of the field. We're not sure that it’s possible to say that there’s a standard in the field right now, it’s too early.
- So, we started reading the documentations of the existing frameworks, trying to understand which ones are good for quick experimentation and demoing, and which ones are good for production. Also, we considered the current adoptions from the AI developers.
- LangGraph is currently doing great in both adoption and production readiness. It has features to save the state of the computations of the agent/workflow, allowing to resuming it at different points. It allows easy monitoring. And other stuff not implemented by other frameworks/libraries. In contrast, LangGraph has a steeper learning curve. LangGraph seems to be very good for workflows.
- So, we decided to give a try to LangGraph and start using it for the research agent and the writing agent.
- Next, we reasoned about the structure of the research agent and the writing agent. Should they both be agents? Or workflows? What are the pros and cons? We made some assumptions but then we preferred starting working on them and trying them, to get a feel of what would work best.
- We started the reasoning agent as mainly a workflow. Then, by trying it, we noticed that it is a rather interactive process where human feedback is useful, and so that it's possible to stop its execution and resume it from a certain point. So, to make the workflow more flexible, we moved some steps of it to tools, and, over time, the overall structure became that of a full agent. Now, the research agent is simply an MCP server that provides ~10 MCP tools and 1 MCP prompt that describes an “agentic workflow” with them, that is, a recipe of how to use the tools to do a research. We took this opportunity to create also a simple and generic MCP client for it, to teach how to create both MCP servers and MCP clients. However, the MCP server can be used with any MCP client (e.g. Cursor, Claude Code, etc).
- Then, we looked at the current best libraries for structuring MCP clients and servers, and FastMCP has both a great adoption and a lot of features. For structuring MCP servers and clients, the FastMCP python library is currently the standard of the industry, for this reason we decided to use it for the research agent (so, no more LangGraph here).
- Meanwhile, we also worked on the "writing agent", but implementing all of it as a workflow (with LangGraph). Here, we noticed that the "process of writing sections, introductions, conclusions, etc by checking if the text is adhering to style guidelines and following the provided article script" is more prone to be exactly a workflow. The process is always the same, there’s not a lot of adaptation involved. The workflow is rather complex and long, but LangGraph allows to keep it well organized.
- It's hard to decide what framework to choose. Each one has its philosophy. It's a field that is evolving a lot and many libraries and frameworks can appear and get quick traction. It's hard to predict which one will win over time. With hindsight, it seems always easy to explain why a particular framework has succeeded instead of another one, but it's hard to predict it in advance. For this reason, we try not to lose ourselves in the details of the frameworks, but rather focus on the concepts and the ideas. Also for this reason, we were in favour of building a project with two agents with the two main types (which are currently better covered by different frameworks): an adaptable and steerable research agent (better served by MCP servers and `fastmcp`) and a reliable writer agent (better served by workflows and `langgraph`). But again, while the choice of the frameworks seems to be a good one today, it's not the only one and we may change our minds in the future. Our advice is to read the main concepts and philosophies of the frameworks and libraries and choose the one that seem to better fit your needs, and experiment with a lot of them. Indeed, also for these two agents, their final design was not obvious in advance. We tried implementing the research agent as workflows at first, only to find out (while testing it) that it needed more steerability. We wanted to teach only a single framework in the course (langgraph, as it's currently the most popular framework for workflows and ready for production), but we ended up using two frameworks for the two agents as a consequence. This shows that there isn't a clear winner yet in terms of the best framework/library for AI agents.
- Moreover, at the start we weren't unsure about using LangGraph either, as it's already pretty engineered and so we were worried that, as the field evolves, it may be less flexible. So, we opted for less-opinionated frameworks/libraries, such as the OpenAI Agents SDK. It had minimal abstractions and was easy to use. However, it missed multiple production-ready features that we needed, and its adoption was still far behind LangGraph. So we sticked to LangGraph.

<agent_framework_analysis_1>
Below is a concise, opinionated overview of six popular “agent frameworks,” with emphasis on **concepts, quickstarts, and underlying philosophy**—not APIs or class names. I keep the focus on *how each thinks about the problem of building agents*, the stage it targets (prototype ↔ production), the level of abstraction, and the learning curve.

---

## TL;DR — how they “think”

* **LangGraph** – a *stateful workflow engine* for agentic systems. You explicitly design graphs (state + nodes + edges) and get persistence, human‑in‑the‑loop, and “time travel.” Strong production posture via checkpointing and an optional hosted platform. Medium learning curve because you model control flow yourself. ([LangChain Docs][1])
* **OpenAI Agents SDK** – *very few abstractions* by design. “Agent, tools, handoffs, guardrails, sessions,” plus built‑in tracing. Optimized for getting production‑grade agent flows with minimal ceremony; orchestration can be LLM‑driven or code‑driven. Low learning curve. ([OpenAI GitHub][2])
* **CrewAI** – *teams (“crews”) of role‑based agents* + *Flows* for deterministic orchestration. Opinionated scaffolding (CLI, YAML) to move fast; positions itself as production‑ready, security‑minded, and independent of LangChain. Low‑to‑medium learning curve. ([CrewAI Documentation][3])
* **PydanticAI** – “*FastAPI for agents*”: typed, validated, observability‑first. Small, Pythonic surface (agents with typed inputs/outputs, DI for tools), strong production features (durable execution with Temporal/DBOS), plus graph support when workflows get complex. Low‑to‑medium learning curve if you like type hints. ([Pydantic AI][4])
* **AutoGen** – *multi‑agent conversation patterns* first, with a split between “AgentChat” (high‑level) and “Core” (event‑driven for scalable systems), and a no‑code Studio for rapid prototyping (explicitly not production). Medium learning curve (many patterns), Studio lowers the bar. ([Microsoft GitHub][5])
* **FastMCP** – not an agent runtime; it’s the *fastest way to build MCP tool servers & clients*. Think “USB‑C for AI tools.” High‑level, Pythonic decorators; adds auth, deployment, testing, and a cloud host. Very low learning curve. ([FastMCP][6])

---

## At‑a‑glance comparison

| Framework             | Philosophy / Primary idea                                     | Sweet spot                                 | Abstractions                                               | “Agent vs. Workflow”                         | Quickstart vibe                                                          | Learning curve |
| --------------------- | ------------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------ | -------------- |
| **LangGraph**         | Explicit stateful graphs with checkpoints, HITL, time‑travel  | Production reliability for complex flows   | Low‑level *primitives* (state, nodes, edges, checkpointer) | More **workflows** (with agent loops inside) | “Define state + nodes, compile graph, run; add persistence & interrupts” | **Medium**     |
| **OpenAI Agents SDK** | Minimal primitives; production‑ready upgrade to Swarm         | Fast path to production                    | **Very few** abstractions                                  | **Agents** with optional code routing        | “Define an agent, tools, run; handoff & guardrails built‑in”             | **Low**        |
| **CrewAI**            | Role‑based agent **crews** + **Flows**                        | From prototype to prod, enterprise posture | Opinionated: Agents, Tasks, Crews/Flows                    | **Agents & flows**                           | “`crewai create` → edit YAML → run”                                      | **Low–Medium** |
| **PydanticAI**        | Type‑safe, validated, observable                              | Production apps; typed interfaces          | Small, Pythonic, strongly typed                            | **Agents** + optional graphs                 | “Define Agent with typed deps/output; add tools; run/stream; instrument” | **Low–Medium** |
| **AutoGen**           | Multi‑agent conversation framework; Studio                    | Research & prototyping → scale with Core   | Two layers: AgentChat (high), Core (event‑driven)          | **Multi‑agent** conversations                | “Install; start assistant & user proxy; try Studio”                      | **Medium**     |
| **FastMCP**           | Standardize tools via MCP; batteries‑included servers/clients | Tool surface for any agent                 | High‑level server/client utilities                         | N/A (tool layer)                             | “`FastMCP()` + `@mcp.tool` → run/host”                                   | **Low**        |

---

## Framework notes (concepts, quickstarts, and philosophy)

### LangGraph (LangChain)

* **Core concepts & philosophy.** LangGraph distinguishes **workflows** (predetermined paths) from **agents** (dynamic, tool‑using) and gives you a **graph** abstraction to encode control flow. Its value proposition is *control and reliability*: persistence (checkpoints and threads), “time travel”/replay, and **human‑in‑the‑loop** via **interrupts** that pause/resume graphs. ([LangChain Docs][1])
* **Quickstart flavor.** You define state (keys with optional reducers), functions as nodes, and conditional edges; compile; then run. The docs flag the current OSS docs as **v1‑alpha** (content evolving), while still showing both Graph and Functional APIs. ([LangChain Docs][7])
* **Production stance.** Persistence via **checkpointers** enables fault‑tolerance, long‑running work, memory across runs, and auditable histories. The **LangGraph Platform** adds deployment, scaling, and observability (distinct from the OSS graph library). ([LangChain Docs][8])
* **When it shines.** Complex, multi‑step or multi‑actor applications where you want explicit orchestration, auditing, and human approval loops baked in (not “just an agent loop”). ([LangChain Docs][1])

### OpenAI Agents SDK (Python & TypeScript)

* **Core concepts & philosophy.** The SDK is intentionally **lightweight**: a *small* set of primitives—**Agents**, **Handoffs** (delegation), **Guardrails** (validation), and **Sessions** (history). It’s designed to be quick to learn while remaining **production‑ready**, with built‑in tracing. ([OpenAI GitHub][2])
* **Quickstart flavor.** Create an Agent with instructions, attach tools (plain functions), and run; use **handoffs** to chain specialists; **guardrails** can short‑circuit on invalid input/output. There are also **Realtime/voice agents** in the TS SDK. ([OpenAI GitHub][2])
* **Orchestration stance.** You can orchestrate **via the LLM** (let it plan/delegate) or **via code** for determinism and cost control; the docs are explicit about the trade‑offs. ([OpenAI GitHub][9])
* **Ecosystem direction.** Reporting around OpenAI’s **Responses API** and **Agents SDK** indicates a shift away from the earlier Assistants API over time, part of an effort to unify tool use (e.g., web search, computer use) and make multi‑agent orchestration more coherent. *(Journalism note rather than the SDK docs themselves).* ([The Verge][10])

### CrewAI

* **Core concepts & philosophy.** CrewAI frames apps as **Crews** (teams of specialized agents with roles, tools, and goals) and **Flows** (structured, event‑driven orchestration for deterministic paths). It positions itself as **lean** (built from scratch, independent of LangChain), **fast**, and “**production‑ready by default**,” with security and enterprise posture. ([CrewAI Documentation][3])
* **Quickstart flavor.** Strong CLI ergonomics: `crewai create` scaffolds a project with YAML for agents and tasks; then you wire up a Crew/Flow and run. There’s also an enterprise Studio for visual creation/deploy. ([CrewAI Documentation][11])
* **When it shines.** Multi‑agent automations where **role clarity** and **task decomposition** matter; and cases where you want an opinionated path from *prototype → deploy* with built‑in patterns for memory, guardrails, and observability integrations. ([CrewAI Documentation][3])

### PydanticAI

* **Core concepts & philosophy.** The Pydantic team’s take on agents: bring the “**FastAPI feeling**” to GenAI—**type‑safe**, validated agents with **dependency injection**, **structured outputs**, and **first‑class observability** (Logfire). It’s model‑agnostic and aims to reduce runtime failures by moving errors to type/validation time. ([Pydantic AI][4])
* **Quickstart flavor.** Define an `Agent` with **typed dependencies** and a **typed output model**; add **tools** (functions) that automatically validate arguments; run synchronously or stream structured output; instrument with Logfire. ([Pydantic AI][4])
* **Production stance.** **Durable execution** integrations (Temporal and DBOS) support long‑running, fault‑tolerant, human‑in‑the‑loop work. There’s also **pydantic\_graph** for when you outgrow linear control flow. ([Pydantic AI][12])
* **When it shines.** You want Pythonic ergonomics, **static typing**, and strong **validation/observability** in production; you like designing your agent surface as clear, typed “signatures.” ([Pydantic AI][4])

### AutoGen

* **Core concepts & philosophy.** AutoGen treats **multi‑agent conversations** as the core building block. Docs describe it as aiming to be for agentic AI what **PyTorch** is for deep learning—support research & real apps, with agents that converse, use tools, and include humans in the loop. ([Microsoft GitHub][5])
* **Two layers.** **AgentChat** (high‑level, conversational patterns) and **Core** (event‑driven architecture for **scalable** multi‑agent systems). **AutoGen Studio** accelerates **prototyping** with a low‑code UI, but is **not production** software by design. ([Microsoft GitHub][13])
* **Quickstart flavor.** Start a chat between an AssistantAgent and a UserProxyAgent; add code execution and tools; then explore conversation patterns (e.g., evaluators, tool callers, code executors). Studio provides a visual team builder, playground, and export. ([Microsoft GitHub][5])
* **When it shines.** Multi‑agent research, pattern exploration, and rapid iteration—then graduate to **Core** if you need deterministic, event‑driven scaling. ([Microsoft GitHub][13])

### FastMCP

* **What it is (and isn’t).** FastMCP is the *standard framework* for building **Model Context Protocol (MCP)** servers and clients—i.e., the **tooling layer** that lets any agent (in any runtime) access your resources/tools via a common protocol (often described as “USB‑C for AI”). It’s not an agent runtime; it’s *how agents get high‑quality tools*. ([FastMCP][6])
* **Quickstart flavor.** Instantiate `FastMCP`, decorate Python functions with `@mcp.tool`, and run via stdio/HTTP or the CLI; there’s a hosted **FastMCP Cloud** for quick, authenticated deploys. ([FastMCP][14])
* **Production stance.** Version 2.0 goes “**beyond the protocol**”: auth, proxying, server composition, OpenAPI/FastAPI generation, testing, and integrations; v1.0 was even incorporated into the **official MCP Python SDK** in 2024. ([FastMCP][6])
* **When it shines.** You want to **standardize tools** across multiple agents/frameworks (or vendors), and you value a simple, Pythonic developer experience. ([FastMCP][6])

---

## How they’re solving “building agents,” in different ways

* **Control‑flow first vs. agent‑loop first.**

  * **Control‑flow/workflow first:** *LangGraph* (explicit graphs, checkpoints, HITL), *PydanticAI* (typed surface, optional graphs; durable execution), and *AutoGen Core* (event‑driven). Great when you need **determinism, auditability, and long‑running reliability**. ([LangChain Docs][1])
  * **Agent‑loop first:** *OpenAI Agents SDK* (minimal primitives + guardrails + handoffs), *CrewAI* (role‑based crews & flows), and *AutoGen AgentChat/Studio* (multi‑agent dialogue patterns). Great when you want **speed to value** and let the LLM handle more of the planning. ([OpenAI GitHub][2])

* **Abstractions and “future‑proofing.”**

  * *OpenAI Agents SDK* is intentionally **light on abstractions**—you can change orchestration strategies without adopting a new DSL. ([OpenAI GitHub][2])
  * *LangGraph* gives **low‑level building blocks** (state, nodes, edges, checkpointing), so you can express new agent patterns as the field evolves. ([LangChain Docs][1])
  * *PydanticAI* treats agent I/O as **typed contracts**—robust to model churn because validation, DI, and structured output remain stable. ([Pydantic AI][4])
  * *FastMCP* future‑proofs the **tooling interface** with an open protocol, so your tools are reusable across runtimes. ([FastMCP][6])

* **Prototype ↔ Production.**

  * **Prototyping:** *AutoGen Studio* (explicitly), *OpenAI Agents SDK* quickstarts, *CrewAI* CLI scaffolding. ([Microsoft GitHub][15])
  * **Production:** *LangGraph* (checkpointing, HITL, platform), *PydanticAI* (durable execution + observability), *OpenAI Agents SDK* (guardrails, tracing). ([LangChain Docs][8])
  * **Tooling for any stack:** *FastMCP* (servers/clients + cloud deploy). ([FastMCP][14])

---

## Practical chooser (rules of thumb)

* **You need explicit control, audit trails, human approvals, and the ability to rewind or branch:** pick **LangGraph** (optionally the Platform for deploy/scale). ([LangChain Docs][8])
* **You want the smallest mental model to get to a production‑grade agent (with guardrails and handoffs):** pick **OpenAI Agents SDK**. ([OpenAI GitHub][2])
* **You want role‑based multi‑agent teams and a batteries‑included CLI path from idea to deployment:** pick **CrewAI**. ([CrewAI Documentation][3])
* **You want typed, validated agents with strong observability and durable execution (Temporal/DBOS):** pick **PydanticAI**. ([Pydantic AI][4])
* **You’re exploring multi‑agent research or want a visual prototyping surface before you “get serious”:** start with **AutoGen Studio**, then move to **AutoGen Core** when scaling. ([Microsoft GitHub][15])
* **Your main problem is tools/integration rather than agent orchestration:** use **FastMCP** to expose tools/resources once and reuse everywhere. ([FastMCP][6])

---

### Sources & doc entry points

* LangGraph concepts & capabilities (workflows vs agents; persistence & interrupts; quickstart; platform positioning). ([LangChain Docs][1])
* OpenAI Agents SDK overview and orchestration approaches; TS realtime. ([OpenAI GitHub][2])
* CrewAI introduction & quickstart. ([CrewAI Documentation][3])
* PydanticAI introduction/why, agents core concepts, durable execution. ([Pydantic AI][4])
* AutoGen getting started; stable site outlining AgentChat/Core/Studio; Studio guide (prototype‑not‑prod). ([Microsoft GitHub][5])
* FastMCP welcome/quickstart/readme (beyond the protocol; SDK incorporation; cloud). ([FastMCP][6])
* Context on OpenAI’s platform direction (Responses API & Agents SDK; Assistants API sunset). ([The Verge][10])

If you want, I can tailor this to a specific use case (e.g., “customer‑support automations with human approval,” or “coding agents with strong tool safety”), and recommend a concrete starter stack and migration path.

[1]: https://docs.langchain.com/oss/python/langgraph/workflows-agents "Workflows and agents - Docs by LangChain"
[2]: https://openai.github.io/openai-agents-python/ "OpenAI Agents SDK"
[3]: https://docs.crewai.com/introduction "Introduction - CrewAI"
[4]: https://ai.pydantic.dev/ "Pydantic AI"
[5]: https://microsoft.github.io/autogen/0.2/docs/Getting-Started/ "Getting Started | AutoGen 0.2"
[6]: https://gofastmcp.com/ "Welcome to FastMCP 2.0! - FastMCP"
[7]: https://docs.langchain.com/oss/python/langgraph/quickstart "Quickstart - Docs by LangChain"
[8]: https://docs.langchain.com/oss/python/langgraph/persistence "Persistence - Docs by LangChain"
[9]: https://openai.github.io/openai-agents-python/multi_agent/ "Orchestrating multiple agents - OpenAI Agents SDK"
[10]: https://www.theverge.com/news/627556/openai-ai-agents-responses-api-agents-sdk?utm_source=chatgpt.com "OpenAI will let other apps deploy its computer-operating AI"
[11]: https://docs.crewai.com/quickstart "Quickstart - CrewAI"
[12]: https://ai.pydantic.dev/durable_execution/overview/ "Overview - Pydantic AI"
[13]: https://microsoft.github.io/autogen/stable/ "AutoGen — AutoGen"
[14]: https://gofastmcp.com/getting-started/quickstart "Quickstart - FastMCP"
[15]: https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/index.html "AutoGen Studio — AutoGen"
</agent_framework_analysis_1>

<agent_framework_analysis_2>
Below is a pragmatic, “what problem does it solve and how does it think about agents?” overview of six popular agent frameworks—now enriched with **Python‑style pseudocode** that illustrates each framework’s core philosophy (not exact APIs), a **snapshot of community adoption (GitHub stars)**, and a **forward‑looking opinion** on where each is best positioned.

---

## 1) LangGraph — “Agents as graphs; reliability first”

**Philosophy.** LangGraph treats an agent (or agent team) as a **stateful graph** of nodes/edges with explicit control over memory, retries, branching, and human‑in‑the‑loop pauses. It is intentionally **low‑level** and orchestration‑focused—emphasizing durability (checkpoints), inspectability, and **interrupts** that wait for human approval before resuming. It positions itself for **production‑grade, long‑running** agent workflows and explicitly *does not* try to abstract away prompts or architecture. ([GitHub][1])

**What the docs/quickstarts push:** human‑in‑the‑loop via **interrupts**, persistent **checkpoints**, and composable “nodes” for tools/LLM steps; you can pause, modify state, resume, and even “time‑travel” to a prior checkpoint while keeping auditability. ([LangChain Docs][2])

**Python‑style pseudocode (pseudocode ≠ real API):**

```python
# LangGraph mental model: a stateful, durable graph with HITL pauses
state = {"messages": [], "pending_tool": None}

graph = Graph(initial_state=state)

@node
def llm_step(s):
    s["messages"] += call_llm_with_tools(s["messages"])
    if tool_requested(s): 
        return "gate"   # route to human gate
    if final_answer_ready(s):
        return END
    return "llm_step"   # loops until done

@node
def gate(s):
    # Pause the graph and wait for human approval; state is checkpointed
    interrupt("approve_tool", payload=s["pending_tool"])   # HITL pause
    if approved():
        s["messages"] += run_tool(s["pending_tool"])
    else:
        s["messages"] += [{"role": "system", "content": "Tool denied"}]
    return "llm_step"

graph.add_edges({
    "llm_step": ["gate", END],
    "gate": ["llm_step"]
})

graph.use_checkpointer("sqlite://runs.db")  # durability
run = graph.invoke(input="Plan a trip to Kyoto", thread_id="u-123")
# Later, resume or replay from any checkpoint of thread u-123
```

**Learning curve:** moderate—the graph model is explicit but pays dividends in reliability and observability. **Best for:** production workflows needing **durability + human oversight**.

---

## 2) OpenAI Agents SDK — “Small set of primitives; guardrails & handoffs”

**Philosophy.** A **lightweight** SDK with “very few abstractions,” focused on real‑world **multi‑agent workflows**. Core primitives are **Agents**, **Tools**, **Guardrails** (input/output checks), **Handoffs** (agent‑to‑agent delegation), and **Sessions** (conversation memory). There’s also a realtime variant that adapts guardrail behavior for streaming voice/AV agents. It’s designed to feel simple yet production‑oriented. ([OpenAI GitHub][3])

**What the docs/quickstarts push:** define an Agent with instructions + tools; add **guardrails** for safety; compose **handoffs** between specialists; and use **sessions** to persist history. ([GitHub][4])

**Python‑style pseudocode (pseudocode ≠ real API):**

```python
# OpenAI Agents mental model: one loop, minimal primitives, safety + delegation
def is_sensitive(user_msg) -> bool: ...

assistant = Agent(
    instructions="Be concise. Use tools when needed.",
    tools=[get_weather, search_web],
    guardrails=[Guardrail(on_input=is_sensitive, action="block_or_escalate")]
)

refund_agent = Agent(instructions="Handle refunds only", tools=[refund_api])
triage = Agent(
    instructions="Route to refund_agent for refund issues; else answer directly.",
    handoffs=[refund_agent]
)

session = Session(id="user-42")  # built-in memory
result = run(triage, "I want a refund for order #123", session=session)
print(result.final_output)
```

**Learning curve:** low—few concepts. **Best for:** teams that want **fast production adoption** with opinionated safety and **agent handoffs** (esp. if you already rely on OpenAI APIs). ([OpenAI GitHub][3])

---

## 3) CrewAI — “Crews (autonomy) + Flows (control); production‑ready by default”

**Philosophy.** CrewAI organizes work around **roles and tasks** (Agents in **Crews**) and a complementary **Flows** system for precise, **procedural orchestration**. The pitch is **fast to build** multi‑agent automations, with guardrails, memory/knowledge, and observability; messaging is **“production‑ready by default.”** ([docs.crewai.com][5])

**What the docs/quickstarts push:** spin up a **Crew** with a researcher/writer pattern; add **Flows** to explicitly stage, branch, or gate with human approval and run single‑LLM steps deterministically when you don’t need full autonomy. ([docs.crewai.com][6])

**Python‑style pseudocode (pseudocode ≠ real API):**

```python
# CrewAI mental model: role-based agents (Crews) + procedural Flows
researcher = Agent(role="Researcher", goal="Find latest sources", tools=[web_search])
writer     = Agent(role="Writer", goal="Draft a brief")

crew = Crew(agents=[researcher, writer])

flow = Flow()
flow.step("research", call=researcher.run, args={"topic": "frontier LLMs"})
flow.step("review",  call=human_approval)                 # optional HITL gate
flow.step("write",   call=writer.run, input_of="research") # deterministic orchestration

report = flow.run()
```

**Learning curve:** low‑to‑moderate—great DX for prototyping **and** a path to production via **Flows** and enterprise add‑ons. ([docs.crewai.com][5])

---

## 4) PydanticAI — “Type‑safety, dependency injection, and durable execution”

**Philosophy.** PydanticAI brings the “**FastAPI feel**” to agents: **typed inputs/outputs**, **structured tools** with validation, and first‑class **dependency injection** for clean access to services. It integrates observability (Logfire) and offers **durable execution** via **Temporal** or **DBOS** wrappers so agents can survive restarts, handle human‑in‑the‑loop, and run for a long time. It’s **model‑agnostic** and explicitly “production‑grade.” ([GitHub][7])

**What the docs/quickstarts push:** guarantee shapes with `output_type`/Pydantic models, annotate tools for schema & validation, and **wrap agents** for **durability** (Temporal/DBOS) while keeping the public interface. ([Pydantic AI][8])

**Python‑style pseudocode (pseudocode ≠ real API):**

```python
# PydanticAI mental model: typed agent + DI + durability
class Deps:
    db: Database
    user_id: int

class Advice(Model):
    text: str
    risk: int  # 0-10

@tool
def balance(ctx: Ctx[Deps], include_pending: bool) -> float:
    return ctx.deps.db.balance(ctx.deps.user_id, include_pending)

agent = Agent[Deps, Advice](
    instructions="Act as a bank support agent. Validate outputs.",
    tools=[balance],
    output_type=Advice,     # retry until structured output validates
)

durable = Durable(agent)   # e.g., DBOSAgent(...) or TemporalAgent(...)
out = durable.run("I lost my card, what now?", deps=Deps(db=DB(), user_id=7))
assert isinstance(out.output, Advice)
```

**Learning curve:** moderate if you’re new to typing/DI; excellent for **teams that want correctness, audits, and reliability**. ([Pydantic AI][9])

---

## 5) AutoGen — “Research‑led multi‑agent patterns; layered; Studio for prototyping”

**Philosophy.** AutoGen provides a **layered** design: a low‑level **Core** for event‑driven, message‑passing agents and a higher‑level **AgentChat** API for common multi‑agent patterns; plus **AutoGen Studio** (a no‑code GUI) to prototype and export specs. The project emphasizes experimentation and ecosystem breadth; **AutoGen Studio is a research prototype (not production‑ready)**. ([GitHub][10])

**What the docs/quickstarts push:** declare multiple agents, wire them in a conversation, optionally add tool execution (code, web, MCP). Use Studio to explore/change the team graph, then export JSON specs. ([GitHub][10])

**Python‑style pseudocode (pseudocode ≠ real API):**

```python
# AutoGen mental model: multi-agent chat + optional event-driven Core
assistant = agent("assistant", tools=[code_exec, web_browse])
critic    = agent("critic",    instructions="Critique plans before execution")

conversation = Chat([assistant, critic])
conversation.run("Generate and review a CLI that fetches latest GDP for Japan.")

# Core (conceptual): you can register event handlers for fine-grained control
core = Runtime(agents=[assistant, critic])
core.on("tool_call", lambda evt: maybe_gate_with_human(evt))
core.run(task="same goal as above")

# Studio: visually configure agents/links; export spec for reuse (prototype)
```

**Learning curve:** low to prototype (AgentChat, Studio), higher for Core; great for **R\&D and pattern exploration**, with a path to more control. ([Microsoft GitHub][11])

---

## 6) FastMCP — “Tool/connectivity layer, not an agent framework”

**Philosophy.** FastMCP is the fast, Pythonic way to build **MCP** servers/clients—the “**USB‑C for AI**” that standardizes how agents connect to external data/tools. It **goes beyond the protocol** with features like auth, server composition, OpenAPI/FastAPI generation, and multiple transports. You bring the agent runtime (LangGraph, PydanticAI, OpenAI Agents, etc.); FastMCP supplies **portable tools/resources/prompts**. ([FastMCP][12])

**What the docs/quickstarts push:** write a tool with a decorator, run the server, then connect from any agent client that speaks MCP. ([GitHub][13])

**Python‑style pseudocode (pseudocode ≠ real API):**

```python
# FastMCP mental model: build one tool server; use it from any agent runtime
server = MCPServer(name="FinanceTools")

@server.tool
def price(ticker: str) -> float:
    return fetch_price(ticker)

server.run(transport="stdio")  # or http/sse

# In your agent app (any framework):
tools = MCPClient.connect("http://localhost:8000").tools()
agent = Agent(instructions="Use tools for facts", tools=tools)
result = agent.run("Compare AAPL and MSFT P/E and summarize risks.")
```

**Learning curve:** low for basic servers; powerful for platform teams standardizing **tooling across agent stacks**. ([FastMCP][12])

---

## Adoption snapshot (GitHub stars) — *as of Sept 17, 2025*

> Stars move quickly; treat this as a rough community‑interest proxy, not a quality metric.

| Framework             |                         Main repo stars | Related ecosystem repo(s)                                     |
| --------------------- | --------------------------------------: | ------------------------------------------------------------- |
| **AutoGen**           |           **49.9k** (microsoft/autogen) | — ([GitHub][10])                                              |
| **CrewAI**            |            **38.2k** (crewAIInc/crewAI) | crewAI‑tools **1.2k** ([GitHub][14])                          |
| **LangGraph**         |      **18.8k** (langchain‑ai/langgraph) | LangChain **116k** (context: broader ecosystem) ([GitHub][1]) |
| **FastMCP**           |              **17.8k** (jlowin/fastmcp) | MCP Python SDK **18.5k** ([GitHub][13])                       |
| **OpenAI Agents SDK** | **14.7k** (openai/openai‑agents‑python) | agents‑js **1.4k**, openai‑python **28.7k** ([GitHub][4])     |
| **PydanticAI**        |        **12.5k** (pydantic/pydantic‑ai) | (ecosystem: Pydantic core/logfire widely used) ([GitHub][7])  |

---

## How their **distinctive features** address the “build agentic systems” problem

* **Reliability & control of long‑running work**

  * *LangGraph* bakes in checkpoints, resumability, and human‑in‑the‑loop **interrupts**, letting you pause/inspect/modify state, then continue—crucial when agents use tools with side‑effects. ([LangChain Docs][2])
  * *PydanticAI* uses **typed IO + DI** and **durable execution** (Temporal/DBOS wrappers) so you can safely validate, retry, and survive crashes or human waits. ([Pydantic AI][8])

* **Simplicity with safety & team delegation**

  * *OpenAI Agents SDK* focuses on **few primitives** (Agents, Tools, **Guardrails**, **Handoffs**, **Sessions**) to cover most real cases, including **realtime** agents where guardrails emit events rather than exceptions. ([OpenAI GitHub][3])

* **Rapid multi‑agent prototyping vs. precise workflows**

  * *CrewAI* encourages starting with **Crews** (autonomous collaboration) and switching to **Flows** when you need surgical control (single‑LLM steps, deterministic branching, gates). Positioning: “**production‑ready by default**.” ([docs.crewai.com][5])
  * *AutoGen* shines for **pattern discovery and research** via AgentChat and **Studio** (visual builder). Studio is explicitly a **research prototype, not for production**, which clarifies where it fits. ([Microsoft GitHub][11])

* **Tool interoperability across stacks**

  * *FastMCP* standardizes tools/resources/prompts via MCP so any compliant agent runtime can consume them—reducing lock‑in and letting platform teams manage tools once, reuse everywhere. ([FastMCP][12])

---

## Quick “when to reach for which” (one‑liners)

* **You need durable, inspectable, human‑approved workflows:** *LangGraph* or *PydanticAI*. ([GitHub][1])
* **You want minimum concepts with built‑in safety and delegation, including realtime:** *OpenAI Agents SDK*. ([OpenAI GitHub][3])
* **You want to prototype multi‑agent teams quickly, then tighten control:** *CrewAI* (Crews → Flows). ([docs.crewai.com][6])
* **You’re exploring multi‑agent patterns or want a no‑code lab:** *AutoGen* (Studio for prototyping). ([Microsoft GitHub][11])
* **You’re a platform team standardizing tools across frameworks:** *FastMCP* (MCP servers/clients). ([FastMCP][12])

---

## My (tentative) forecast — who’s best‑positioned and why

> This is a forecast, not a verdict; the space is evolving quickly and **time will tell**.

* **Best positioned for production agent *workflows***
  **LangGraph** and **PydanticAI** look strongest where today’s pain is: *reliability, auditability, and control*. LangGraph’s graph + checkpoints + interrupts give unparalleled control over long‑running, stateful processes with humans in the loop. PydanticAI’s type‑safety/DI and **durable execution** (Temporal/DBOS) reduce “stringly typed” failures and make it easier to promote prototypes to robust services. If I had to bet on “ship something critical this quarter,” I’d start with one (or a combo: PydanticAI agents inside LangGraph graphs). ([GitHub][1])

* **Best positioned for a broad developer funnel**
  **OpenAI Agents SDK** brings a small mental model with opinionated knobs (Guardrails, Handoffs, Sessions) and good docs/examples—including **realtime**. That’s attractive for teams wanting quick wins without adopting a larger framework. The trade‑off is tighter coupling to OpenAI’s way of thinking—even though the SDK is provider‑agnostic, many users will pair it with OpenAI’s platform. ([OpenAI GitHub][3])

* **Best positioned for multi‑agent prototyping at scale**
  **CrewAI** has real momentum and a “batteries‑included” feel for **teams** of agents. The addition of **Flows** gives it a credible production path without abandoning its simple mental model. That combination—fast prototyping + procedural control—explains its rapid community adoption. ([docs.crewai.com][5])

* **Best positioned as the “tooling substrate”**
  **FastMCP** is likely to become *the* connective tissue for agent tools. Its “USB‑C for AI” positioning and beyond‑the‑protocol features (auth, composition, OpenAPI import) make it attractive for orgs that want one tool layer across multiple agent runtimes. ([FastMCP][12])

* **AutoGen’s role**
  **AutoGen** remains an influential **research & patterns** engine. The layered design is flexible, and Studio is great for exploration—but Studio’s “not for production” status suggests AutoGen will keep thriving as a *lab* that informs production implementations elsewhere (including AutoGen’s own lower‑level layers). ([Microsoft GitHub][11])

**Bottom line:** If your current blockers are **reliability, human‑gating, and observability**, pick **LangGraph** or **PydanticAI** (or combine them). If you want **minimal abstractions** with clear safety primitives and **realtime**, pick **OpenAI Agents SDK**. If your team loves **multi‑agent prototypes** but also wants a **deterministic workflow mode**, try **CrewAI**. And regardless of the framework, invest in **FastMCP** (MCP) to future‑proof your **tooling**. That said, this is a fast‑moving frontier; **the best framework a year from now may look different**.

---

### Sources

* **LangGraph** concepts/interrupts & “why LangGraph”: official docs and repo. ([LangChain Docs][2])
* **OpenAI Agents SDK** overview, Agents, Guardrails, Handoffs, Realtime guardrails. ([OpenAI GitHub][3])
* **CrewAI** docs (production‑ready; Flows; quickstart). ([docs.crewai.com][5])
* **PydanticAI** README highlights and durable execution (Temporal/DBOS). ([GitHub][7])
* **AutoGen** repo (layered design) and Studio note (not production). ([GitHub][10])
* **FastMCP** docs/repo (MCP “USB‑C” and beyond‑protocol features). ([FastMCP][12])
* **Adoption (stars)** pulled from each repo page: LangGraph 18.8k, LangChain 116k, OpenAI Agents Python 14.7k, OpenAI Agents JS 1.4k, openai‑python 28.7k, CrewAI 38.2k, crewAI‑tools 1.2k, PydanticAI 12.5k, AutoGen 49.9k, FastMCP 17.8k, MCP Python SDK 18.5k. ([GitHub][1])

If you want, I can turn this into a one‑page decision matrix (PDF) or map the pseudocode patterns into real starter templates for the framework(s) you’re leaning toward.

[1]: https://github.com/langchain-ai/langgraph "GitHub - langchain-ai/langgraph: Build resilient language agents as graphs."
[2]: https://docs.langchain.com/oss/python/langgraph/add-human-in-the-loop?utm_source=chatgpt.com "Enable human intervention - Docs by LangChain"
[3]: https://openai.github.io/openai-agents-python/?utm_source=chatgpt.com "OpenAI Agents SDK"
[4]: https://github.com/openai/openai-agents-python "GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows"
[5]: https://docs.crewai.com/?utm_source=chatgpt.com "CrewAI Documentation - CrewAI"
[6]: https://docs.crewai.com/guides/flows/first-flow?utm_source=chatgpt.com "Build Your First Flow"
[7]: https://github.com/pydantic/pydantic-ai "GitHub - pydantic/pydantic-ai: GenAI Agent Framework, the Pydantic way"
[8]: https://ai.pydantic.dev/durable_execution/temporal/?utm_source=chatgpt.com "Durable Execution with Temporal"
[9]: https://ai.pydantic.dev/durable_execution/dbos/?utm_source=chatgpt.com "Durable Execution with DBOS"
[10]: https://github.com/microsoft/autogen "GitHub - microsoft/autogen: A programming framework for agentic AI  PyPi: autogen-agentchat Discord: https://aka.ms/autogen-discord Office Hour: https://aka.ms/autogen-officehour"
[11]: https://microsoft.github.io/autogen/dev//user-guide/autogenstudio-user-guide/index.html?utm_source=chatgpt.com "AutoGen Studio - Microsoft Open Source"
[12]: https://gofastmcp.com/?utm_source=chatgpt.com "Welcome to FastMCP 2.0! - FastMCP"
[13]: https://github.com/jlowin/fastmcp "GitHub - jlowin/fastmcp:  The fast, Pythonic way to build MCP servers and clients"
[14]: https://github.com/crewAIInc/crewAI "GitHub - crewAIInc/crewAI: Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks."
</agent_framework_analysis_2>