# Lesson 13 - Choosing our Framework - Guidelines

## Global Context of the Lesson

### What We Are Planning to Share

Lesson 13 is a **guide to agent frameworks** for production‑grade AI systems. We will compare the most widely used libraries—**LangGraph**, **OpenAI Agents SDK**, **CrewAI**, **PydanticAI**, **AutoGen**, **Claude Agent SDK** and **FastMCP,** by focusing on their **philosophies, abstractions, strengths, and trade‑offs** rather than API trivia. We anchor the comparison in our capstone architecture: an **adaptable research agent** implemented with **FastMCP (client and server)**, and a **reliable writing hybrid agentic/workflow** implemented with **LangGraph (w/langchain mcp adapters) for the client + FastMCP for the server (where the agent tools live)**. You’ll learn how we evaluated the options, why we **changed direction mid‑build**, and how to reason about **framework choice under uncertainty**.

A note on Brown’s interfaces: for IDEs like **Cursor** or **Claude Code**, we **expose Brown’s workflows themselves as MCP tools**. That means any MCP client can trigger Brown’s “write”, “edit selection”, or “edit full article” flows as single commands, while LangGraph still provides checkpoints, interrupts, and replay *behind* the scenes.

### Why We Think It's Valuable

Agent frameworks are new and evolving quickly (mainly due to models getting more intelligent and more capable). Choosing them for the wrong reasons can lock you into brittle abstractions; choosing too late can stall the project. This lesson provides a **concept‑first lens** to evaluate frameworks, **separates durable ideas from transient features**, and provides a **decision process** you can reuse. You’ll also see how the **two dominant production patterns** (steerable, tool‑driven agents vs. deterministic, auditable workflows) naturally map to different libraries.

### Expected Length of the Lesson

**2,500–3000 words** (without titles and references), where we assume that 200–250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

**100% theory**

---

## Anchoring the Lesson in the Course

### Details About the Course

This piece/lesson is part of a broader course on AI agents and LLM workflows. The course consists of **4 parts**, each with multiple lessons.

Thus, it’s essential to anchor this piece into the broader course and understand where the reader is in their journey. You will be careful to consider the following:

- The points of view
- To not reintroduce concepts already taught in the previous lessons
- To be careful when talking about concepts introduced only in future lessons
- To always reference previous and future lessons when discussing topics outside the piece’s scope

### Lesson Scope

This is **Lesson 13 of the course,** which is the second lesson of part 2 (out of 4 parts) of the course on **AI Agents & LLM Workflows**. Part 2 of the course will have 22 lessons. This lesson #13, is a no‑notebook, high‑level lesson about the main frameworks and libraries used to build AI agents.

### Point of View

The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use **“we,” “our,” and “us”** to refer to the team who creates the course, and **“you” or “your”** to address the reader. Avoid singular first person and don’t use **“we”** to refer to the student.

**Example of correct point of view:**

- Instead of “Before we can choose between workflows and agents, we need a clear understanding of what they are,” write “To choose between workflows and agents, you need a clear understanding of what they are.”

### Who Is the Intended Audience

Aspiring AI engineers choosing their **first production Agent Framework**.

### Concepts Introduced in Previous Lessons

In previous lessons of the course, we introduced the following concepts:

**Part 1:**

- **Lesson 1 - AI Engineering & Agent Landscape**: Understanding the role, the stack, and why agents matter now
- **Lesson 2 - Workflows vs. Agents**: Grasping the crucial difference between predefined logic and LLM-driven autonomy
- **Lesson 3 - Context Engineering**: The art of managing information flow to LLMs
- **Lesson 4 - Structured Outputs**: Ensuring reliable data extraction from LLM responses
- **Lesson 5 - Basic Workflow Ingredients**: Implementing chaining, routing, parallel and the orchestrator-worker patterns
- **Lesson 6 - Agent Tools & Function Calling**: Giving your LLM the ability to take action
- **Lesson 7 - Planning & Reasoning**: Understanding patterns like ReAct (Reason + Act)
- **Lesson 8 - Implementing ReAct**: Building a reasoning agent from scratch
- **Lesson 9 - RAG Deep Dive**: Advanced retrieval techniques for knowledge-augmented agents
- **Lesson 10 - Agent Memory & Knowledge**: Short-term vs. long-term memory (procedural, episodic, semantic)
- **Lesson 11 - Multimodal Processing**: Working with documents, images, and complex data

Part 2:

- Lesson 12 positioned the **research** and **writing** agents at a high level.

### Concepts That Will Be Introduced in Future Lessons

In future lessons of the course, we will introduce the following concepts:

**Part 2:**

- **Lesson 14 — System‑Design Decision Framework** (LLM models comparison, costs/latency, human‑in‑the‑loop strategy). It’s a more “in depth” system-design overview of the present lesson.
- **Lessons 15–18 — Building the Research Agent** with **FastMCP** (server/client, tools, prompt; ingestion; Perplexity research loops; filtering; building the final artifact `research.md`)
- **Lessons 19–22 — Building the Writing Agent** with **LangGraph and FastMCP** (hybrid system between workflows and agents, controlling writing profiles via context engineering, reflection/self‑critique, human-in-the-loop editing via MCP)

**Part 3:**

- With the agent system built, this section focuses on the engineering practices required for production. You will learn to design and implement robust evaluation frameworks to measure and guarantee agent reliability, moving far beyond simple demos. We will cover AI observability, using specialized tools to trace, debug, and understand complex agent behaviors. Finally, you’ll explore optimization techniques for cost and performance and learn the fundamentals of deploying your agent system, ensuring it is scalable and ready for real-world use.

**Part 4:**

- In this final part of the course, you will build and submit your own advanced LLM agent, applying what you've learned throughout the previous sections. We provide a complete project template repository, enabling you to either extend our agent pipeline or build your own novel solution. Your project will be reviewed to ensure functionality, relevance, and adherence to course guidelines for the awarding of your course certification.

As lesson on the core foundation of AI engineering, we will have to make references to new terms that haven't been introduced yet. We will discuss them in a highly intuitive manner, being careful not to confuse the reader with too many terms that haven't been introduced yet in the course.

### Anchoring the Reader in the Educational Journey

Within the course we are teaching the reader multiple topics and concepts. Thus, understanding where the reader is in their educational journey is critical for this piece. You have to use only previously introduced concepts, while being reluctant about using concepts that haven't been introduced yet.

When discussing the **concepts introduced in previous lessons** listed in the `Concepts Introduced in Previous Lessons` section, avoid reintroducing them to the reader. Especially don't reintroduce the acronyms. Use them as if the reader already knows what they are.

Avoid using all the **concepts that haven't been introduced in previous lessons** listed in the `Concepts That Will Be Introduced in Future Lessons` subsection. Whenever another concept requires references to these banned concepts, instead of directly using them, use intuitive analogies or explanations that are more general and easier to understand, as you would explain them to a 7-year-old. For example:

- If the "tools" concept wasn't introduced yet and you have to talk about agents, refer to them as "actions".
- If the “OpenAI Agents SDK” framework wasn’t introduced yet, you can refer to it as “Other Agent Frameworks”
- If the "routing" concept wasn't introduced yet and you have to talk about it, refer to it as "guiding the workflow between multiple decisions". You can use the concepts that haven't been introduced in previous lessons listed in the `Concepts That Will Be Introduced in Future Lessons` subsection only if we explicitly specify them. Still, even in that case, as the reader doesn't know how that concept works, you are only allowed to use the term, while keeping the explanation extremely high-level and intuitive, as if you were explaining it to a 7-year-old. Whenever you use a concept from the `Concepts That Will Be Introduced in Future Lessons` subsection, explicitly specify in what lesson it will be explained in more detail, leveraging the particulars from the subsection. If not explicitly specified in the subsection, simply state that we will cover it in future lessons without providing a concrete lesson number.

In all use cases avoid using acronyms that aren't explicitly stated in the guidelines. 

---

## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end‑to‑end lesson:

- What problem are we learning to solve? Why is it essential to solve it?
- Why other solutions are not working and what's wrong with them.
- At a theoretical level, explain our solution or transformation. Highlight:
    - The theoretical foundations.
    - Why is it better than other solutions?
    - What methods or algorithms can we use?
- Provide simple examples or diagrams where useful.
- Go deeper into the advanced theory.
- Provide a more complex example supporting the advanced theory.
- Connect our solution to the bigger field of AI Engineering. Add course next steps.

---

## Lesson Outline

1. **Introduction**
2. **Framework choice under uncertainty and why some selection strategies fail in production**
3. **A theory for choosing: decision axes instead of brands**
4. **The landscape today: frameworks, philosophies & adoption snapshot**
5. **Framework Deep Dive: LangGraph**
6. **Framework Deep Dive: OpenAI Agents SDK**
7. **Framework Deep Dive: CrewAI**
8. **Framework Deep Dive: PydanticAI**
9. **Framework Deep Dive: AutoGen**
10. **Framework Deep Dive: FastMCP**
11. **Choosing for your project: decision matrix & tentative forecasts**
12. **Complex example — Our capstone pivots: FastMCP for research, LangGraph+FastMCP for writing**
13. **Conclusion — Concepts over brands; what you’ll build next**

---

### 1) Introduction — Framework choice under uncertainty (and why it matters now)

- **Quick reference to what we've learned in previous lessons, especially the one just before, lesson 12, in part 2 of the course:** Take the core ideas of what we've learned in earlier lessons from the `Concepts Introduced in Previous Lessons` subsection of the `Anchoring the Lesson in the Course` section.
- **Transition to what we'll learn in this lesson:** After presenting what we learned in the past, make a transition to what we will learn in this lesson. Take the core ideas of the lesson from the `What We Are Planning to Share` subsection and highlight the importance and existence of the lesson from the `Why We Think It's Valuable` subsection of the `Global Context of the Lesson` section. To remind you:
    - In this lesson, we explain how we decided **which framework(s)** to use for two capstone builds: an adaptable **research agent** and a durable **writing workflow**. We’ll compare leading options by **philosophy and fit**, not API trivia, and we’ll be transparent about why we **pivoted** during development. Keep two anchors in mind as you read: (1) **LangGraph’s interrupts and checkpoints** when you need auditable, stoppable‑and‑resumable work; (2) **MCP** as your **interoperability layer**—think “USB‑C for AI”—so tools you build today can plug into multiple runtimes tomorrow. We’ll show how this plays out in practice: our research agent ships as a **FastMCP server** (tools/resources/prompts) with a lightweight client loop, while our writing system uses a **LangGraph workflow** that **calls MCP tools** exposed by a FastMCP server. You’ll also see how we **surface Brown’s workflows as MCP commands** so IDEs like **Cursor** can drive the same durable process with a friendlier UX, while LangGraph keeps the audit trail and pause/resume intact.
- **Section length:** ~250–300 words

### 2) Framework choice under uncertainty and why some selection strategies fail in production

- Set context: the agent ecosystem is moving fast, model (LLMs) capabilities keep getting stronger, and **no universal standard** or framework that will satisfy everyone or be suitable for all use-cases. Clarify terminology so students don’t conflate things: a *framework/runtime* (e.g., LangGraph, CrewAI, PydanticAI, OpenAI Agents SDK, AutoGen) orchestrates behavior; a *protocol* (e.g., **Model Context Protocol, MCP**) standardizes how agents access tools/resources across runtimes; and a *tooling framework* (e.g., **FastMCP**) helps you implement that protocol.
- Frameworks and protocols solve different problems. A **runtime** like LangGraph orchestrates state and control‑flow; a **protocol** like MCP standardizes how agents discover and call tools/resources across runtimes; a **tooling framework** like **FastMCP** makes it fast to implement that protocol on both server and client. The production questions follow: do you need **pause/resume**, **time‑travel/replay**, **HITL gates**, and **durable execution**, or do you want a lighter orchestration with portable tools you can run in any MCP‑speaking client? (LangGraph provides **checkpoints/interrupts**; MCP provides **tool portability**; FastMCP provides **servers/clients**, transports, and quick deploys.) Beware common traps: trend‑chasing, forcing one framework onto mismatched problems, and prototyping without persistence/observability in mind. In our own build, research proved **interactive and divergent**—better served by a steerable loop and portable MCP tools—while writing benefited from **explicit state, checkpoints, and replay**.
- We initially framed research as a static LangGraph workflow; reality demanded frequent pivots and mid‑run tool changes. We moved the “work” into MCP tools and kept orchestration light; for writing, we did the opposite. (We’ll show both.)
- **Section length:** ~400 words

### 3) A theory for choosing: decision axes instead of brands

- Give students a guide to evaluate any new framework/library. Stay at high level here.
    - **Control‑flow explicitness ↔ LLM‑driven autonomy:** Graphs & gates (e.g., LangGraph) concentrate power in *your* code—great for repeatable workflows and audits. Agent loops with minimal primitives (e.g., OpenAI Agents SDK) tilt toward **autonomy with optional delegation/handoffs**. Ask: Do we need deterministic branching and pause/resume or simpler ReAct agent loops?
    - **Reliability primitives:** Look for **checkpointing, time‑travel/replay, HITL interrupts, durable execution** and where they live (framework vs. platform). LangGraph emphasizes **interrupts** and persistence; PydanticAI integrates **durable execution** via Temporal/DBOS. Clearly explain what these terms mean.
    - **Abstraction level & DX (developer experience).** Minimal surfaces (Agents/Tools/Guardrails/Handoffs/Sessions) vs. opinionated constructs (Crews/Flows). Evaluate onboarding speed, CLI scaffolding, and whether you can drop to code for precision. Abstractions are good to build fast, but the ability to go “low level” will always be great in cases where the feature you want to add is not yet present.
    - **Tooling interoperability (MCP).** Treat MCP like **USB‑C for AI**: design your tools as MCP servers so you can **swap runtimes** without rewriting integrations. In practice this means building tools/resources/prompts once (e.g., with **FastMCP**) and loading them into LangGraph (via **`langchain‑mcp‑adapters`**), OpenAI Agents SDK, or IDE clients like Cursor/Claude Code.
- Map the two capstones: **research** leans to autonomy + MCP tools + HITL; **writing** leans to explicit graphs, durability, and observability. (Note that LangGraph can do both ReAct agents and more deterministic workflows)
- **Section length:** ~400 words

### 4) The landscape today: frameworks, philosophies & adoption snapshot

- With the axes in mind, survey the terrain you’ll be choosing from.
- Introduce the six frameworks/libraries we’ll examine, in terms of how they **think** and where they best fit. Stay at high level here, as each framework/library will be explained better in the next sections.
    - **LangGraph**: Stateful graphs with **checkpoints** and **interrupts**; supports prebuilt and custom ReAct agents. Great when you need auditable, resumable execution.
    - **OpenAI Agents SDK**: Small surface area with **Agents, Tools, Guardrails, Handoffs, Sessions** and a built‑in agent loop, fast to production if you prefer minimal abstractions.
    - **CrewAI**: “Crews” (roles) + **Flows** (deterministic orchestration) with strong CLI/YAML ergonomics; useful for multi‑agent automations that later need structure.
    - **PydanticAI**: Typed agents, **durable execution** via **Temporal/DBOS**, and observability via **Logfire**; shines when schemas and correctness matter.
    - **AutoGen**: AgentChat/Core plus **Studio** for GUI prototyping; **Studio is explicitly not for production**, but great for exploration.
    - **FastMCP**: Not a runtime, **MCP servers/clients** with stdio/HTTP transports, quickstart/Cloud deploys, and OpenAPI‑to‑MCP features; ideal to standardize tools across stacks. Also enables a **“workflow‑as‑tool”** façade so IDE MCP clients (Cursor/Claude Code) can trigger **entire LangGraph flows** as single commands, with LangGraph providing durability behind the scenes.
- Provide a **directional adoption snapshot** (GitHub stars, as of Sept 17, 2025): **AutoGen ~50k**, **CrewAI ~38k**, **LangGraph ~18k**, **FastMCP ~18k**, **OpenAI Agents SDK ~14k**, **PydanticAI ~12k**. Remind students stars ≠ quality; they’re a **community‑energy proxy**, useful for ecosystem sensing and hiring risk.
- **Section length:** ~400 words

### 5) Framework Deep Dive: LangGraph

- Start with the framework most aligned to **workflows**.
- Explain how **LangGraph thinks**: you model **state + nodes + edges** and run a **graph** that you can **pause**, inspect, and **resume**
- Teach the production setup: **interrupts** for human approval, **persistence** for long‑running work, **replay/time‑travel** to earlier checkpoints, and a **Platform** for deployment/observability (Studio for prototyping).
- **Interrupts + checkpointers.** You can pause for human input and **resume from exact state**; this underpins HITL and replay.
- **ReAct agents.** You can either use the **prebuilt ReAct** helper or build custom ReAct graphs (state + nodes + edges) when you need finer control.
- Show a very simple example of LangGraph workflow (with code), ideally taken from the LangGraph official docs. It doesn’t need to be complete, but it should show the main philosophy of the framework. Explan that there’s a “graph API” and a “functional API” and what are their differences.
- **Auditability**: every transition is explicit and persisted.
- Be candid: modeling control flow is **work**; the learning curve is steeper than “just let the LLM plan,” but you get **determinism and safety** in exchange. **That said, for small scripts or one-off/simple flows, LangGraph may be overkill and impose unnecessary structure.**
- Preview that this will anchor the capstone’s **writing agent**, where the payoff of explicit state far outweighs initial complexity. Point students to the docs that show **interrupts**, **persistence**, and **Platform**/Studio so they can try the patterns hands‑on.
- **Section length:** ~400 words

### 6) Framework Deep Dive: OpenAI Agents SDK

- Shift to a **small‑surface** SDK: teach the five core ideas—**Agents**, **Tools**, **Guardrails**, **Handoffs**, **Sessions**—and how they cover common patterns without committing to a full-graph framework.
- Explain the fundamental difference between OpenAI Agents SDK and LangGraph and the from the LangGraph Functional API.
- Show the mental model: start with a single agent using **tools**; add **guardrails** to validate inputs/outputs; add **handoffs** to delegate to specialist agents; use **sessions** to maintain history. Show a very simple example of AI agent with the OpenAI Agents SDK, ideally from the official docs (with code).
- Emphasize that the SDK intentionally keeps orchestration **lightweight** so you can let the **LLM plan** or **route with code** when needed. This makes it a **fast path to production** for teams who value **few concepts** and good defaults.
- Be honest about trade‑offs: you’ll hand‑roll some reliability features (e.g., durable pause/resume), and adoption is newer than the longest‑running libraries—so expect fewer 3rd‑party templates than LangGraph today.
- Encourage students to skim the official docs to see how **guardrails/handoffs/sessions** are implemented and, if they need more guarantees, how others pair the SDK with durable backends.
- **Section length:** ~400 words

### 7) Framework Deep Dive: CrewAI

- Next, a batteries‑included approach to **multi‑agent teams**.
- Teach CrewAI’s two main concepts and why that matters: **Crews** (role‑based, autonomous collaboration) and **Flows** (event‑driven, deterministic orchestration). Show how this lets teams **start autonomous**, then **tighten control** as requirements harden.
- Show a very simple example of AI agent with CrewAI, ideally from the official docs (with code).
- Highlight the **developer experience**: CLI scaffolding (`crewai create`) and project layout (`agents.yaml`, `tasks.yaml`) makes setup fast, **YAML** for agents/tasks, and a growing set of examples.
- Be precise about where it shines: **multi‑agent automations** where role clarity matters—e.g., researcher ↔ writer with a review gate.
    - Include a small narrative: Researcher compiles sources → human **approval** gate → Writer drafts → Flow enforces a style check using a single LLM call.
- Trade-offs: opinionated abstractions enable excellent **speed**, but also sometimes **constraints**. You’ll do work around defining Crews, configuring memory (short-term/long-term/entity), writing flows, choosing and configuring persistence (e.g. via `@persist`), and specifying state schemas (structured vs unstructured). For very simple tasks, this may feel like extra overhead compared to minimal agent/tool-SDKs.
- Wrap by pointing learners to the **Flows** and **Crews** docs, the **YAML** convention, and especially the sections on **Flow State Management** and **Memory & Knowledge** so they see both what comes built-in and where they’ll need to extend.
- **Section length:** ~400 words

### 8) Framework Deep Dive: PydanticAI

- Position PydanticAI as “**FastAPI for agents**”: you design **typed inputs/outputs** and **tools**, wire them through **dependency injection (DI)**, and get **validation**, **retries until the schema passes**, and **first‑class observability** (Logfire).
- Show a very simple example of AI agent with PydanticAI, ideally from the official docs (with code).
- Explain the two production levers: (1) **durable execution** via **Temporal** or **DBOS** wrappers—agents can **survive restarts**, wait for human approval, and continue later; (2) **graph support** when control flow becomes non‑linear (see `pydantic_graph`). This is especially attractive when correctness and contracts matter; e.g., a support agent must always emit a `Resolution` object. Explain the terms above terms clearly.
- Show the ergonomics: type hints define tool schemas; docstrings become tool descriptions; structured outputs guide **automatic retries** on schema-validation failure; supports multiple providers.
- Be transparent about trade-offs: you need to define schemas up front; there’s more upfront setup—ensuring serialization, wrapping non-deterministic steps / tool functions explicitly; dealing with payload size limits and streaming constraints; teams unfamiliar with DI, typed models, or durable workflows will need a short ramp.
- Provide links for hands‑on exploration: the **durable execution** docs and the main **intro** so learners can see code‑first patterns with validation and streaming.
- **Section length:** ~400 words

### 9) Framework Deep Dive: AutoGen

- Explain AutoGen’s **layered design** clearly: **AgentChat** offers convenient multi‑agent conversation patterns; **Core** provides low‑level, event‑driven primitives for scalable systems; **Studio** (GUI) lets you prototype **without code,** including visual team/agent/workflow composition and drag-and-drop configuration.
- Clarify role in a production journey: **use Studio to explore patterns** and team topologies, then **export and harden** your architecture in code (AgentChat/Core *or elsewhere*). The key is understanding that **Studio is a research prototype and explicitly not for production**, so it’s perfect for discovery but not for critical workloads.
- Show a very simple example of AI agent with AutoGen, ideally from the official docs (with code).
- Position the trade‑offs: breadth and speed for experimentation vs. the need to re‑implement reliability, security, and production primitives. Also consider that Studio is not hardened, and the abstractions in AgentChat/Core may require custom work when you need stringent compliance, latency, or failure recovery.
- Encourage students to treat AutoGen as a **pattern laboratory**—excellent for the **research agent’s early stages**—and to graduate to a runtime that meets their durability/HITL needs once patterns stabilize.
- **Section length:** ~400 words

### 10) Framework Deep Dive: FastMCP

- Finally, the **tooling layer** most teams overlook.
- **What FastMCP is (and isn’t).** FastMCP is a fast, Pythonic way to build **MCP servers and clients**. It’s not an agent runtime; instead, it gives you the protocol‑correct “ports” (tools/resources/prompts) so any MCP‑speaking client can drive them. You can run servers over **stdio** for local use or **HTTP** for remote, call them from a **FastMCP client**, and deploy to **FastMCP Cloud** with built‑in auth. For development, FastMCP supports **local/same‑process transports** that keep iteration tight.
- Show a very simple example of AI agent with FastMCP, ideally from the official docs (with code).
- Show the mental model and architecture in words: **user ↔ MCP client ↔ FastMCP server (tools/resources/prompts)**; the client can also be an IDE, a chat app, or an agent runtime. FastMCP can also create MCP clients.
- Explain **why teams adopt it**: (1) **Interoperability**—write tools once, reuse across LangGraph, PydanticAI, OpenAI Agents, AutoGen; (2) **Beyond the protocol** features— FastMCP provides: **auth**, **server composition**, **OpenAPI‑to‑MCP generation**, **proxying**, and a **Cloud** for quick deploys; (3) **great DX**—decorate a function with `@mcp.tool`, or define `@mcp.resource` / `@mcp.prompt`, and FastMCP handles schema generation, transport, context injection etc. It abstracts away much of the boilerplate.
- Where it shines: platform teams standardizing tools; projects that expect **framework churn**; agents that must reach **enterprise systems** through a single, governed surface.
- Add the practical takeaway for this course: our **research agent** is now “just” a FastMCP server exposing ~10 tools + one MCP prompt that describes the **recipe**; any MCP client can steer it.
- Point learners to the **quickstart docs**, the **MCP spec**, and the FastMCP **OpenAPI integration** so they see how quickly existing APIs / endpoints can map to tools/resources in FastMCP.
- **Why we adopted it.** We wanted our tools to be **portable** across runtimes and IDEs, to **evolve independently** of orchestration, and to keep dev loops fast. FastMCP gave us that, and MCP adapters let those tools plug straight into LangGraph when we want orchestration features.

<aside>
💡

**Pattern: tools‑as‑workflows (Brown).** We wrap each Brown workflow entry point as a **coarse‑grained MCP tool** (e.g., `brown.write_from_scratch`, `brown.edit_selection`, `brown.edit_full_article`). The MCP tool handler simply **kicks off the corresponding LangGraph run**, returns progress/messages, and finally yields artifacts and diffs. This gives you MCP portability (Cursor, Claude Code) **without** losing LangGraph’s durability and auditability.

</aside>

- **Section length:** ~400 words

### 11) Choosing for your project: decision matrix & tentative forecasts

- Turn principles into guidance. Present a **decision matrix** with rows for common needs and a checkmark where each framework is a natural fit:
    - Durability/HITL/replay → **LangGraph** (checkpoints, interrupts).
    - **Typed contracts + durable execution** → **PydanticAI** (Temporal/DBOS).
    - Few primitives + guardrails/handoffs → **OpenAI Agents SDK** (Agents/Tools/Guardrails/Handoffs/Sessions).
    - Role‑based teams + quick scaffolding → **CrewAI** (Crews + Flows + CLI/YAML).
    - **Exploration lab** → **AutoGen (Studio/AgentChat)**; prototype patterns first.
    - Tool portability across stacks → **FastMCP (MCP)**.
- Then give a **clearly labeled forecast** (not a verdict): LangGraph and PydanticAI are strong candidates for production workflows where reliability, auditability, and correctness dominate; OpenAI Agents SDK spans a broad developer funnel that values simple mental models; CrewAI suits fast multi-agent prototyping with a path to more structured Flow-level control; FastMCP remains the durable tooling substrate regardless of runtime; AutoGen serves as the R&D lab for exploring new multi-agent patterns.
- Close by reminding students to combine axes: it’s common to use **FastMCP tools** inside **LangGraph** or **PydanticAI** workflows, or to start in **AutoGen Studio** then migrate to something more production-hardened.
- **Want IDE‑native UX (Cursor/Claude Code) for a long‑running workflow?** Keep **LangGraph** for orchestration, but **front it with FastMCP** by exposing each workflow as an **MCP command** that triggers the LangGraph run. You get MCP portability for UX and LangGraph’s durability for reliability.
- **Section length:** ~400 words

### 12) Complex example — Our capstone pivots: FastMCP for research, LangGraph for writing

- Apply the matrix to a **real** decision: our capstone project.
- **How our choices evolved.** We began with LangGraph for **both** agents. Early research runs showed that the work was **interactive and divergent**—we needed to add/replace tools, pivot mid‑run, and occasionally replan on the fly. To maximize portability, we moved the workload into **MCP tools** and shipped **Nova** as a **FastMCP server** (≈10 tools + one “agentic recipe” prompt). Any **MCP client** (IDE or runtime) can now steer it. For development, we kept a **light custom ReAct loop** (+ FastMCP Client) as the MCP client to teach you how the loop works and to benefit from tight local transports; later, you can swap this to **LangGraph** to regain **interrupts/checkpoints** without touching the tool layer.
- **Writing took the opposite path.** The process was **repeatable but long**, and we needed **auditable state**, **checkpoints**, **HITL interrupts**, and **replay**. We therefore kept **Brown** as a **LangGraph workflow** (Functional API) and **moved its actions behind MCP tools** served by FastMCP. In plain English: **“LangGraph for orchestration; FastMCP for tools.” MCP‑first mode (IDE clients):** We **expose Brown’s workflows as MCP tools** on a FastMCP server so any MCP client (e.g., **Cursor**, **Claude Code**) can run them with one command.
    
    Brown currently exposes three commands:
    
    - **Write from scratch** — generate the article from the **article guidelines** and **research**.
    - **Edit selection** — revise a **selected text span** using the **reflection loop** plus **human feedback**.
    - **Edit full article** — run the **reflection loop** (plus optional human feedback) across the entire draft.
    
    In **Cursor**, after you run an editing command, **diffs are shown like code changes**; you can accept only the chunks you want. This gives you MCP‑native ergonomics *and* keeps LangGraph’s audit trail and resumability under the hood.
    
- **Alternatives we considered.** We evaluated the **OpenAI Agents SDK** for its lightweight primitives (**agents/tools/guardrails/handoffs/sessions** and a built‑in agent loop), and it remains a solid option when you want minimal surface area; we centered the capstone on **LangGraph + FastMCP** to get first‑class **persistence/HITL** and MCP‑backed portability.

- **Section length:** ~400 words

### 13) Conclusion

- End with the meta‑lesson: **prioritize concepts over brands**. The ideas that last—**graphs & gates**, **typed contracts**, **durable execution**, **tool standardization (MCP)**—let you adapt as frameworks rise and fall.
- Encourage students to **prototype, measure, and iterate**: start with the axes, pick the smallest stack that meets today’s constraints, and keep your **tooling interface** portable via MCP to reduce lock‑in.
- Invite learners to **mix and match**: it’s normal to start a pattern in **AutoGen Studio** or **CrewAI** for speed, migrate tools to **FastMCP**, and anchor long‑running flows in **LangGraph** or **PydanticAI** as requirements solidify. Share one closing pointer so students can keep learning: the **MCP specification** is short and worth skimming; understanding it once pays dividends across every agent stack.
- In the next lesson (14), you’ll formalize the **system‑design decision framework**—how to choose models, balance cost/latency, and place human‑in‑the‑loop gates. Later in Part 2 (Lessons 15–22), you’ll build both agents end‑to‑end: **Nova** with **FastMCP** (server/client, ingestion, Perplexity loops, filtering, `research.md`) and **Brown** with **LangGraph + FastMCP** (workflow + MCP tools, profiles via context engineering, reflection/self‑critique, HITL editing).
- **Section length:** ~300 words

---

## Article Code

**No notebook or code** for this lesson.

---

## Sources

1. [LangGraph: workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
2. [LangGraph: persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
3. [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
4. [CrewAI: introduction](https://docs.crewai.com/introduction)
5. [CrewAI: quickstart](https://docs.crewai.com/quickstart) 
6. [CrewAI: first flow](https://docs.crewai.com/guides/flows/first-flow)
7. [PydanticAI](https://ai.pydantic.dev/)
8. [PydanticAI: durable execution](https://ai.pydantic.dev/durable_execution/overview/)
9. [AutoGen](https://microsoft.github.io/autogen/stable/)
10. [AutoGen Studio](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/index.html)
11. [FastMCP](https://gofastmcp.com/)
12. [FastMCP: quickstart](https://gofastmcp.com/getting-started/quickstart)

