# Research

## Research Results

<details>
<summary>What are the fundamental differences in architecture and application between the Evaluator-Optimizer pattern and the Reflection pattern for improving LLM outputs?</summary>

### Source [1]: https://www.anthropic.com/research/building-effective-agents

Query: What are the fundamental differences in architecture and application between the Evaluator-Optimizer pattern and the Reflection pattern for improving LLM outputs?

Answer: Anthropic defines the **evaluator–optimizer workflow** as one LLM call that **generates** a response and another that **evaluates and provides feedback** in a loop until a success criterion is met.[3] The core architectural idea is an explicit **separation of concerns**: a *generator* proposes solutions, and an *evaluator* critiques them against **clear, predefined evaluation criteria**.[3] This pattern is recommended when (1) such criteria exist and (2) **iterative refinement** of an answer creates measurable value.[3]

Anthropic contrasts this with simpler single-shot or tool-using workflows by emphasizing the **feedback loop**: the evaluator’s feedback is fed back into the generator, which revises its output in subsequent iterations.[3] The loop continues until the evaluator deems the response acceptable, forming a **self-improving system at the application level** rather than by retraining the underlying model.[3]

Anthropic also discusses a separate pattern of **reflection-like behavior** via *deliberation* and *critique* phases, where the same or another model reviews or comments on prior reasoning before producing a final answer.[3] In this reflection-style setup, the focus is on **introspective analysis of previous steps** (e.g., checking reasoning, spotting gaps, restructuring the approach) rather than on a strict pass/fail evaluation loop with fixed success metrics.[3] Architecturally, this often appears as intermediate *critique* or *self-review* calls inserted into a chain of reasoning, while the evaluator–optimizer pattern formalizes that review step as a dedicated **evaluator component** with explicit acceptance criteria and a loop controller.[3]

Thus, Anthropic’s official guidance frames evaluator–optimizer as a **structured, criteria-driven loop** around generation, whereas reflection-like critique is a more general **reasoning and self-review phase** that can be used even without hard evaluation metrics.[3]

-----

-----

### Source [2]: https://docs.langchain.com/oss/python/langgraph/workflows-agents

Query: What are the fundamental differences in architecture and application between the Evaluator-Optimizer pattern and the Reflection pattern for improving LLM outputs?

Answer: LangChain’s LangGraph documentation describes **evaluator–optimizer workflows** as architectures where **one LLM call creates a response and another evaluates that response**.[8] If the evaluator (or an optional human-in-the-loop) finds the response insufficient, the system **routes back to the generator** for refinement, forming an iterative loop.[8] The core components are: a **generator node**, an **evaluator node**, and **control flow** that loops until criteria are satisfied.[8]

The documentation positions this pattern among other **agentic workflows**, emphasizing that the evaluator can be implemented as an LLM, heuristic code, or a human, but in all cases it applies **explicit quality checks** before the answer is accepted.[8] The system is thus architected around explicit **validation and gating** of outputs.

By contrast, LangGraph also supports patterns that correspond to **reflection** or **self-critique**, where an LLM is prompted to **analyze its own previous outputs**, explain errors, or improve its reasoning before producing a final answer.[8] In those reflection-style workflows, the same model (or another) is used in a *critic* role, but rather than enforcing a pass/fail gate based on strict criteria, the emphasis is on **introspective improvement of the reasoning trace or answer quality**.[8] Architecturally this shows up as nodes that perform *review* or *self-reflection* within a graph, rather than a dedicated evaluator node that controls loop exit via success conditions.

LangChain thus differentiates evaluator–optimizer as a **quality-control loop with explicit success conditions**, while reflection-like nodes provide **self-review and critique** that may or may not be tied to hard acceptance thresholds.[8]

-----

-----

### Source [3]: https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-evaluators-and-reflect-refine-loops.html

Query: What are the fundamental differences in architecture and application between the Evaluator-Optimizer pattern and the Reflection pattern for improving LLM outputs?

Answer: AWS prescriptive guidance describes **evaluator workflows** and **reflect–refine loops** as related but distinct patterns.[6] In evaluator workflows, **one LLM critiques another’s output**, enabling **iterative improvement** before the result is returned.[6] The key elements are an LLM acting as **generator**, another acting as **evaluator**, and a loop that continues until an evaluation condition is satisfied or a maximum number of iterations is reached.[6] This is directly aligned with the **evaluator–optimizer** pattern: the evaluator provides structured feedback or a pass/fail signal, and the generator updates its output accordingly.[6]

AWS separately describes **reflect–refine loops** as workflows where an LLM **reflects on its own outputs**, identifies problems, and then **refines** the answer in subsequent steps.[6] Here, the same model may perform both the initial task and the reflection, often guided by prompts that ask it to review its reasoning, check for errors, or propose improvements.[6] Architecturally, this reflection pattern centers on **self-critique and internal analysis** of the model’s own response, rather than on an external evaluator component.[6]

The documentation highlights that evaluator workflows are particularly suitable where you can define **objective or semi-objective evaluation criteria** (e.g., correctness against tests, policy compliance, format constraints), allowing the evaluator to act as a quality gate.[6] Reflect–refine loops are suggested when **subjective quality** or **reasoning depth** matters, and where prompting the model to "think again" or "reflect" materially improves outputs without requiring a separate evaluator service.[6]

Thus, AWS draws a line between **evaluator–optimizer–style architectures** (two-role, criteria-driven, externally evaluated loop) and **reflection patterns** (single-role or lightly separated, self-review-focused, introspective refinement).[6]

-----

-----

### Source [4]: https://huggingface.co/blog/dcarpintero/design-patterns-for-building-agentic-workflows

Query: What are the fundamental differences in architecture and application between the Evaluator-Optimizer pattern and the Reflection pattern for improving LLM outputs?

Answer: The Hugging Face blog on "Design Patterns for Building Agentic Workflows" describes the **Evaluator–Optimizer pattern** as applying **iterative refinement** to LLM responses through **systematic feedback and improvement cycles**.[4] It explicitly notes that this pattern draws inspiration from **self-reflection** in human problem solving, where one examines past actions to gain insight and improve results.[4] In the LLM context, the pattern sets up a process to **self-reflect on past actions and implement targeted improvements**, instead of accepting first-attempt outputs.[4]

Architecturally, the pattern introduces explicit phases of **evaluation** and **optimization**: one component (often an LLM) evaluates previous outputs according to defined criteria, and another component (often the same or a different LLM call) **optimizes** the response based on that feedback.[4] This can be repeated multiple times, so the system becomes a looped workflow rather than a single call.[4]

The article contrasts this with more generic **reflection** approaches by stressing **structure and explicit feedback channels**: rather than merely prompting the model to "reflect" and try again, the Evaluator–Optimizer pattern defines a **systematic feedback mechanism** and **clear refinement steps**.[4] The evaluator’s role is to identify specific issues in the prior output (e.g., missing constraints, errors, poor structure), and the optimizer’s role is to produce a new output that addresses those concrete points.[4]

In reflection-style patterns, the blog notes that LLMs may simply be encouraged to "self-reflect" in a looser way, examining their own thoughts and actions to improve results.[4] The Evaluator–Optimizer pattern formalizes this into **two distinct roles and a repeatable loop**, which makes it easier to design agentic workflows with predictable improvement behavior and higher reliability.[4]

-----

-----

### Source [5]: https://sebgnotes.substack.com/p/evaluator-optimizer-llm-workflow

Query: What are the fundamental differences in architecture and application between the Evaluator-Optimizer pattern and the Reflection pattern for improving LLM outputs?

Answer: This article presents the **Evaluator–Optimizer LLM Workflow** as a general **pattern for self-improving AI systems**.[2] Architecturally, it consists of a **Generator**, an **Evaluator**, a **Feedback Loop**, and **Success Criteria**.[2] The **Generator** creates candidate solutions; the **Evaluator** assesses these solutions against **explicit, machine-readable criteria**; the **Feedback Loop** feeds critiques back to the Generator; and **Success Criteria** define when to exit the loop.[2]

The pattern is recommended in scenarios with **clear evaluation criteria** and where **iterative refinement** delivers value.[2] The article emphasizes **separation of concerns**: the Generator focuses on achieving goals given feedback, while the Evaluator focuses on quality assurance and governance.[2] It also stresses building **circuit breakers** and loop exit conditions to avoid infinite loops, and embedding **quality assurance** directly into the system architecture, not as a post-processing step.[2]

Regarding its relationship to **reflection**, the article notes that Evaluator–Optimizer embodies a move from **static** to **dynamic** AI implementations, enabling systems that can improve over time via structured feedback.[2] While inspired by the idea of self-reflection, the workflow differs from simple reflection prompts by requiring **explicit evaluation logic and success conditions**, often grounded in business or task metrics.[2] Reflection in a looser sense might involve a model reviewing its own answer and revising it once, whereas the Evaluator–Optimizer pattern defines a **repeatable, governance-friendly loop** with monitoring, observability, and long-term evaluation of whether improvements persist.[2]

Thus, this source characterizes Evaluator–Optimizer as a **formalized, criteria-driven refinement architecture**, compared with informal reflection patterns that rely mainly on introspective prompts without strict evaluative components or operational safeguards.[2]

-----

-----

### Source [6]: https://dev.to/clayroach/building-self-correcting-llm-systems-the-evaluator-optimizer-pattern-169p

Query: What are the fundamental differences in architecture and application between the Evaluator-Optimizer pattern and the Reflection pattern for improving LLM outputs?

Answer: This article describes implementing Anthropic’s **Evaluator–Optimizer pattern** in a production system for LLM-generated SQL.[1] Architecturally, the system is split into (1) an **evaluator** based on ClickHouse and rule-based checks, and (2) an **optimizer** using an LLM plus rules to fix problems.[1] When an LLM-generated SQL query fails, the system preserves the **original intent** (analysis goal) and iteratively **fixes syntax and semantic errors** rather than regenerating from scratch.[1]

The **evaluator component** uses cheap database operations such as `EXPLAIN AST` to validate syntax and a `SELECT ... LIMIT 1` probe to test execution, along with an error classifier that identifies specific failure modes like wrong table names or invalid aggregations.[1] These evaluations provide **precise, actionable signals** (e.g., which part of the query failed) that are then fed to the optimizer.[1]

The **optimizer component** uses the LLM, plus rule-based fallbacks, to modify the existing query to address identified issues while **preserving the original analysis intent**.[1] Examples include transforming `count() * column` into `sum(column)`, mapping incorrect table names, and moving aggregates from `WHERE` to `HAVING`.[1] This loop continues until the query passes the evaluator’s checks or a maximum number of attempts is reached.[1]

The article highlights that this Evaluator–Optimizer architecture makes the system **self-correcting without retraining** the underlying model, by "coaching" it in real time using concrete feedback from the evaluator.[1] Compared to generic **reflection prompts**, which might ask a model to "check its work" in a broad sense, this implementation relies on **external, deterministic evaluation signals** (database errors and rules) and a structured iteration mechanism, providing more reliable and production-grade improvement than purely introspective reflection.[1]

-----

</details>

<details>
<summary>What are the practical trade-offs between LangGraph's Functional API and Graph API for building complex, stateful AI workflows?</summary>

### Source [7]: https://docs.langchain.com/oss/python/langgraph/choosing-apis

Query: What are the practical trade-offs between LangGraph's Functional API and Graph API for building complex, stateful AI workflows?

Answer: The "Choosing between Graph and Functional APIs" guide explains that both APIs share the same runtime and core features (persistence, streaming, human-in-the-loop, memory), so the trade-offs are about *how* you express complex, stateful workflows rather than capabilities.[1]

For **complex workflows**, the **Graph API** is recommended when you need:
- **Complex workflow visualization** for debugging and documentation, since the graph structure (nodes/edges) is explicit and can be rendered.[1]
- **Explicit shared state management**, where a central state object is read and written by many nodes across the workflow.[1]
- **Rich conditional branching** with multiple decision points encoded as edges or conditional edge functions, making non-linear flows easier to understand.[1]
- **Parallel execution paths** that need to merge later; the graph structure naturally models fan-out/fan-in patterns.[1]
- **Team collaboration** benefits from a clear visual representation that non-authors can inspect and reason about.[1]

The **Functional API** is positioned for:
- **Minimal code changes** to existing procedural code; you decorate functions with `entrypoint`/`task` and keep standard control flow.[1]
- **Standard control flow** (if/else, loops, function calls) driven by imperative Python, without thinking in terms of nodes/edges.[1]
- **Function-scoped state** without explicit global/shared state schemas; state is typically passed via function arguments/returns.[1]
- **Rapid prototyping** and lower boilerplate for simple or mostly linear workflows.[1]
- **Linear workflows with simple branching**, where a full StateGraph would be over-engineering.[1]

The guide also describes **composability trade-offs**: you can build complex multi-agent coordination with the Graph API, then call a simpler Functional API workflow as a single node inside that graph, or refactor an over-engineered graph into a simpler Functional workflow when the process is essentially linear.[1] This supports mixing paradigms: use Graph API where structure/visualization and explicit state matter most, and Functional API where local control flow and simplicity matter most.[1]

-----

-----

### Source [8]: https://blog.langchain.com/introducing-the-langgraph-functional-api/

Query: What are the practical trade-offs between LangGraph's Functional API and Graph API for building complex, stateful AI workflows?

Answer: The "Introducing the LangGraph Functional API" blog contrasts the Functional and Graph APIs as two paradigms for building workflows and emphasizes practical trade-offs.[2]

For **control flow**, the **Functional API** lets you use normal Python constructs (loops, conditionals, function calls) without reasoning about graph topology.[2] This usually **reduces control-flow boilerplate** relative to defining nodes and edges in a StateGraph.[2] The trade-off is that the execution path is determined dynamically at run time, which means there is **no built-in workflow visualization**, unlike the Graph API where structure is explicit and easily visualized for debugging and sharing.[2]

On **state and checkpoints**, the blog notes a key difference: in the **Functional API**, checkpoints are **generated only at each `entrypoint` execution**; `task` executions update the existing checkpoint but do not create new ones.[2] As a result, **time-travel and fine-grained checkpoint navigation are better supported in the Graph API**, which can generate checkpoints at each node/edge transition.[2] For complex, long-running, stateful workflows where you need to inspect or fork from many intermediate states, this favors the Graph API.[2]

The post stresses that both APIs provide **human-in-the-loop interactions, short- and long-term memory, and streaming**, but the **Graph API excels at visualization and fine-grained time-travel**, while the **Functional API excels at ease of adoption and minimal restructuring of existing code**.[2] It also highlights that both can be **mixed**, so you can use the Graph API for complex agent orchestration and wrap simpler or more linear parts in Functional API entrypoints, balancing structural clarity with coding convenience.[2]

-----

-----

### Source [9]: https://changelog.langchain.com/announcements/functional-api-for-langgraph

Query: What are the practical trade-offs between LangGraph's Functional API and Graph API for building complex, stateful AI workflows?

Answer: The LangGraph changelog announcement for the Functional API focuses on why and when to use the Functional API and how it relates to the Graph API.[3]

It emphasizes that the **Functional API** lets you use LangGraph capabilities **without explicitly defining a graph**, which is important for practical adoption in existing applications or other frameworks.[3] This removes the overhead of modeling nodes and edges when your mental model is procedural code.[3]

Key properties of the Functional API described are:
- **Works anywhere**: it can be embedded into any app, removing dependency on graph-specific syntax.[3]
- **Seamless state management**: it still tracks user inputs, workflow progress, and historical context, so you keep statefulness without managing a central graph state object.[3]
- **Human-in-the-loop support**: workflows can be paused, collect user feedback, and resume dynamically, similar to graph-based workflows.[3]
- **Streaming-first**: supports real-time streaming updates for responsive interactions.[3]

The announcement underlines **"Graph + Functional API = Best of both worlds"**: both APIs share the same runtime, so you can **mix and match paradigms to build robust AI systems**.[3] In practice, this means you can keep using the Graph API where explicit workflow modeling, visualization, and structured branching are helpful, and introduce the Functional API where you want integration flexibility and minimal refactoring.[3]

For complex, stateful workflows, this text implies a trade-off: the **Graph API offers more explicit structure and graph-level semantics**, while the **Functional API prioritizes integration convenience and incremental adoption**, leveraging the same underlying state and interaction features.[3]

-----

-----

### Source [10]: https://docs.langchain.com/oss/python/langgraph/graph-api

Query: What are the practical trade-offs between LangGraph's Functional API and Graph API for building complex, stateful AI workflows?

Answer: The Graph API overview describes how LangGraph models workflows explicitly as **graphs** of state, nodes, and edges, which directly impacts how you build complex, stateful workflows.[4]

It defines **State** as a shared data structure capturing the current snapshot of the application; this central state can be evolved over time and checkpointed.[4] **Nodes** are functions that receive the current state, perform computation or side effects, and return updated state; **Edges** determine which node runs next based on the current state, supporting conditional or fixed transitions.[4]

By combining nodes and edges, you can create **complex, looping workflows that evolve state over time**.[4] The model is naturally suited to complex branching, cycles, and long-running processes because the transitions are explicit and interpretable.[4]

The documentation also notes **proactive vs reactive error-handling advantages** within a graph: proactive patterns (handling failures within the graph) allow saving intermediate state in checkpoints and can provide **graceful degradation**, partial results, and normal graph completion without exceptions, which is beneficial for user experience in complex flows.[4] Reactive patterns centralize error handling but are simpler to implement.[4]

Relative to the Functional API (as described in other official docs), this graph-centric design provides **fine-grained state control, explicit transitions, and easier visualization/debugging**, making it preferable for complex, multi-step, stateful workflows where you want to see and manage each stage as a node in a graph.[4] The trade-off is more up-front structure and boilerplate compared to writing imperative control flow with the Functional API.[4]

-----

</details>

<details>
<summary>What are best practices for implementing checkpointing for short-term memory in production-level AI workflows to ensure resilience and debuggability?</summary>

### Source [11]: https://learn.microsoft.com/en-us/agent-framework/tutorials/workflows/checkpointing-and-resuming

Query: What are best practices for implementing checkpointing for short-term memory in production-level AI workflows to ensure resilience and debuggability?

Answer: Microsoft’s Agent Framework documentation describes checkpointing as saving **workflow state at specific points** so execution can later be resumed after process restarts, failures, or pauses, which is critical for long‑running and human‑in‑the‑loop AI workflows.[1]

It introduces a **CheckpointManager** abstraction that centralizes checkpoint storage and retrieval.[1] In production, the default in‑memory manager is suited only for development; resilient deployments should use a durable implementation (for example, `CheckpointManager.CreateJson(store, customOptions)` backed by a persistent store) so state survives process and host restarts.[1]

Checkpointing is enabled at workflow execution time; when using `InProcessExecution.StreamAsync`, you pass a `CheckpointManager`, and **checkpoints are automatically created at super‑step boundaries**, giving a consistent, debuggable set of states over the workflow’s progression.[1] These checkpoints are exposed through events (such as `SuperStepCompletedEvent`), allowing the application to **capture checkpoint metadata** (IDs, timestamps) for later resumption or inspection.[1]

For resilience, the framework supports **resuming from a specific checkpoint** using `InProcessExecution.ResumeStreamAsync`, which restarts the workflow run from the chosen saved state while re‑emitting workflow events for downstream systems.[1] For debuggability and short‑term memory analysis, there is also **in‑place restoration**: `RestoreCheckpointAsync` can roll back an existing run instance to a prior checkpoint and continue from there, enabling step‑back debugging and replay of agent behavior.[1]

A complete sample pattern shows using a **file‑based checkpoint store**, listing available checkpoints, selecting the latest by timestamp, and resuming from it, illustrating operational practices like organizing checkpoints on disk, resuming from the most recent consistent state, and streaming events for monitoring and debugging.[1]

Key operational benefits highlighted include: fault tolerance, the ability to pause for human input and resume later, **debugging via state inspection at specific points**, and better resource management by stopping and restarting workflows without losing context.[1]

-----

-----

### Source [12]: https://run-ai-docs.nvidia.com/saas/workloads-in-nvidia-run-ai/using-training/checkpointing-preemptible-workloads

Query: What are best practices for implementing checkpointing for short-term memory in production-level AI workflows to ensure resilience and debuggability?

Answer: NVIDIA Run:ai’s guidance for **checkpointing preemptible training workloads** emphasizes patterns that generalize to production AI workflows requiring resilience and debuggability.[2]

Because preemptible jobs can be paused and resumed by the scheduler, Run:ai recommends **periodically saving checkpoints** and configuring workloads to **resume from the latest checkpoint** instead of restarting from scratch, which preserves progress and provides a concrete history of model state over time.[2]

For when to checkpoint, they stress **saving at regular intervals**, commonly at the end of each epoch in training loops, using frameworks’ checkpoint callbacks (e.g., `ModelCheckpoint` in Keras) to automatically persist weights and relevant training metadata.[2] This regular cadence gives a sequence of recent states suitable for short‑term recovery and debugging.

To handle abrupt suspension, Run:ai provides a **signal hook** triggered before a workload is suspended, giving a limited time window (default 30 seconds, configurable up to 5 minutes) to flush a final checkpoint and any additional state required to resume deterministically.[2] Production code should implement this hook to guarantee that, even under preemption, the most recent state is durably stored.

Run:ai notes that the **resume path is explicit application logic**: when a job restarts, it executes the same startup script as the first run, and it is the developer’s responsibility to check for existing checkpoint files, load them if present, and continue from that state.[2] This pattern—startup scripts that auto‑detect and restore checkpoints—improves operational robustness and makes recovery predictable.

They also show using a **shared, durable storage backend** (e.g., NFS‑mounted directory) to store checkpoints, ensuring they remain available across nodes and restarts.[2] This infrastructure practice is essential in production environments for resilient, debuggable AI training and other long‑running workloads that depend on checkpointed short‑term memory.

-----

-----

### Source [13]: https://www.seagate.com/blog/checkpointing-in-ai-workload-a-primer-for-trustworthy-ai/

Query: What are best practices for implementing checkpointing for short-term memory in production-level AI workflows to ensure resilience and debuggability?

Answer: Seagate’s primer on AI checkpoints frames checkpointing as a core mechanism for **trustworthy AI**, supporting fault tolerance, transparency, and reproducibility in AI training workflows.[3]

It defines an AI checkpoint as a **snapshot of the model and training state at a specific point in time**, enabling recovery after failures and providing a concrete artifact that can be examined or reused.[3] The article explains that checkpointing typically happens at **regular, relatively short intervals**—often every minute or every few minutes—chosen based on the complexity and needs of the training job.[3] This high‑frequency checkpointing is particularly relevant to preserving *short‑term* progress and minimizing the amount of work lost after interruptions.

Seagate highlights that, operationally, a common practice is to **write checkpoints to SSDs**, which offer high‑speed writes and rapid read access needed for active training and recovery.[3] Because SSDs are more expensive and not ideal for large‑scale long‑term storage, new checkpoints often **overwrite previous ones** to manage space efficiently, effectively maintaining a rolling window of recent model states that capture short‑term history.[3]

From a workflow‑level perspective, checkpoints also help with **optimizing models and ensuring transparency**: teams can trace how the model changed over time, reload intermediate states for analysis, or branch new experiments from a known state, which improves debuggability and auditability.[3]

The article positions checkpointing as essential for **reliable AI data workflows**—pipelines that consume dependable inputs and yield reliable outputs. This reliability comes from the ability to recover from infrastructure or software failures, reproduce intermediate results, and provide artifacts that can be inspected when diagnosing problems or validating behavior.[3]

Overall, Seagate recommends combining **frequent, SSD‑backed, rolling checkpoints** with downstream storage strategies for longer‑term retention as needed, balancing resilience, cost, and the need for debuggable, trustworthy AI operations.[3]

-----

-----

### Source [14]: https://www.cudocompute.com/blog/storage-requirements-for-ai-clusters

Query: What are best practices for implementing checkpointing for short-term memory in production-level AI workflows to ensure resilience and debuggability?

Answer: Cudo Compute’s discussion of storage requirements for AI clusters emphasizes checkpointing as a **foundational mechanism for fault tolerance and operational robustness** in modern AI workflows.[5]

They note that while checkpointing’s primary purpose is **fault tolerance**, it also enables several key capabilities: experimentation and fine‑tuning, reproducibility, and **scaling resilience** as clusters grow.[5] By saving model state at intervals, teams can launch new experiments from existing checkpoints instead of starting from scratch, and colleagues can reload the same checkpoint to **reproduce specific results**—a critical aspect of debuggability and audit trails.[5]

For large GPU clusters, the article stresses that the probability of individual node failures increases, making **frequent checkpointing a non‑negotiable requirement** to ensure that long‑running jobs can eventually complete despite interruptions.[5] This recommendation aligns with using checkpoints as short‑term memory snapshots that bound loss of work and provide clear restart points.

Operationally, they advise treating checkpointing as a first‑class concern in infrastructure design because checkpoints heavily impact storage capacity and I/O throughput requirements.[5] In particular, they recommend to **“make checkpointing a KPI”**: organizations should actively monitor checkpoint duration, I/O throughput during checkpoint writes, and the resulting GPU idle time.[5] These metrics reveal inefficiencies in the storage stack and justify investments to reduce checkpoint overhead.

By instrumenting and tracking these metrics, teams can tune checkpoint frequency and storage layout (for example, using high‑throughput storage tiers for active checkpoints) to balance resilience with performance overhead.[5] This instrumentation also supports debuggability: slow or failed checkpoint operations can be correlated with system issues and mitigated before they impact workflows.

In sum, Cudo Compute advocates designing AI clusters with **checkpoint performance and reliability as explicit, measurable objectives**, integrating them into observability and capacity planning so that short‑term model state can be saved and restored efficiently in production environments.[5]

-----

-----

### Source [15]: https://www.dbos.dev/blog/ai-agents-do-weird-stuff-and-how-to-fix-it

Query: What are best practices for implementing checkpointing for short-term memory in production-level AI workflows to ensure resilience and debuggability?

Answer: DBOS describes how **durable workflows** and fine‑grained checkpointing can improve reliability and debuggability of AI agents that exhibit unpredictable behavior.[6]

They propose **checkpointing every step an agent takes into a database**, effectively recording each action and intermediate state as part of a workflow.[6] This approach “imposes determinism on fundamentally nondeterministic agents” by ensuring that all decisions, prompts, and state transitions are durably stored.[6]

From a debuggability standpoint, this design makes it possible to **reconstruct the agent’s state at any point in time**, enabling developers to replay or step through behavior when diagnosing “weird” actions.[6] Because each step is captured, engineers can inspect the exact inputs and context that led to a problematic output, and then adjust prompts, policies, or code accordingly.

DBOS emphasizes that such durable, step‑wise checkpointing **reduces token costs** for debugging, since developers can load the precise historical context from storage rather than regenerating long conversations or re‑querying models repeatedly.[6]

In production terms, writing checkpoints to a transactional database offers strong **durability guarantees** and supports auditability and observability of agent workflows.[6] This architecture is particularly suited to short‑term memory in agents: each immediate decision and its context is captured as a checkpointed record, enabling robust post‑hoc analysis and targeted rollbacks.

Overall, DBOS recommends designing AI agent systems as **durable, database‑backed workflows** with automatic checkpointing at each step, in order to improve resilience to failures and provide fine‑grained, replayable traces for debugging and monitoring unusual agent behavior.[6]

-----

-----

### Source [16]: https://sparkco.ai/blog/mastering-langgraph-checkpointing-best-practices-for-2025

Query: What are best practices for implementing checkpointing for short-term memory in production-level AI workflows to ensure resilience and debuggability?

Answer: Sparkco’s article on mastering LangGraph checkpointing outlines concrete **best practices for production‑grade agent workflows** using LangGraph’s checkpointing system.[7]

They recommend using **production‑grade checkpointers** rather than in‑memory or local‑file implementations for critical workloads. Specifically, they highlight the `PostgresSaver` from the `langgraph-checkpoint-postgres` package as suitable for **operational workloads**, because it provides durable storage that survives process restarts and supports pause/resume across distributed environments.[7]

Using such a database‑backed saver enables **state inspection**: operators and developers can query the underlying Postgres tables to view or analyze the stored agent states, transitions, and metadata for debugging or auditing.[7] This visibility is important for understanding short‑term memory effects and reproducing problematic behaviors.

The article also notes that these production‑grade checkpointers support **pause and resume functionality**, allowing workflows to be safely halted and later continued from the exact saved state without losing context.[7] This is crucial for resilience in long‑running or resource‑intensive agent graphs, where interruptions and redeployments are common.

Beyond durability, the guidance implies attention to **safety and scalability**: choosing a robust, transactional backend like Postgres ensures that concurrent workflows can checkpoint reliably and that partial writes or corrupt states are avoided.[7] For debugging complex agent graphs, having each node or step checkpointed into a central store simplifies reconstructing execution paths.

In sum, Sparkco advocates configuring LangGraph in production with **database‑backed, durable checkpoint storage (e.g., PostgresSaver)** to provide reliable pause/resume semantics, persistent short‑term memory for agents, and rich observability into workflow state for debugging and operational monitoring.[7]

-----

</details>

<details>
<summary>According to Anthropic's prompting methodology, what is the 'prompt anatomy' and how can it be applied to create a robust system prompt for an LLM-based evaluator?</summary>

### Source [17]: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview

Query: According to Anthropic's prompting methodology, what is the 'prompt anatomy' and how can it be applied to create a robust system prompt for an LLM-based evaluator?

Answer: Anthropic’s official prompt engineering overview defines a **systematic “prompt anatomy”** composed of clearly separated sections that structure instructions and context for Claude.[6]

Key elements of this anatomy:[6]
- **Role / high-level instructions**: A brief description of *who* the model is and *what overall objective* it has (for example, “You are a careful, impartial evaluator that scores model responses against a rubric”). This should be concise and appear at the start of the prompt.[6]
- **Task description**: A direct, concrete description of the task the model must perform, avoiding ambiguity and unnecessary flourish. Instructions should state what to do, what not to do, and any constraints (like length limits or refusal criteria).[6]
- **Separation of instructions from data**: Anthropic recommends clearly separating the *instructions* from the *input content* to be analyzed, for example by using headings or XML-style tags. This helps the model distinguish between what it must follow and what it must evaluate.[6]
- **Formatting / output specification**: Explicitly describe the required output format (JSON schema, numbered steps, rubric-based scores, etc.). Anthropic emphasizes that specifying structure and fields often matters more than natural-language style.[6]
- **Examples**: Few-shot examples are part of prompt anatomy when needed. Anthropic advises that examples be placed after the core instructions, and be clearly annotated so the model can generalize them.[6]
- **Step-by-step reasoning**: A section explicitly instructing the model to reason step-by-step, deliberate, or use chain-of-thought can be included when tasks are complex, though they caution to keep such instructions concise.[6]

Applied to an **LLM-based evaluator’s system prompt**, this anatomy becomes a robust scaffold:
- Start with a short evaluator persona and objective.
- Add a precise description of the evaluation task.
- Clearly segment: evaluation rubric, input to evaluate, and any tools.
- Specify exact scoring format and justification requirements.
- Optionally, include 1–3 labeled examples of good evaluations.
- Instruct the evaluator to think step-by-step and to strictly follow the rubric and output format.[6]

-----

-----

### Source [18]: https://github.com/anthropics/prompt-eng-interactive-tutorial

Query: According to Anthropic's prompting methodology, what is the 'prompt anatomy' and how can it be applied to create a robust system prompt for an LLM-based evaluator?

Answer: Anthropic’s interactive prompt engineering tutorial operationalizes **prompt anatomy** into concrete sections and practices.[3]

The tutorial’s early chapters define basic structure as:[3]
- **Basic prompt structure**: A clear opening that frames the task, followed by any necessary context, and ending with a direct request. This corresponds to a minimal anatomy: *role → context → explicit task*.[3]
- **Being clear and direct**: Instructions should be unambiguous, using explicit verbs ("evaluate," "score," "compare") and avoiding vague phrasing. This reinforces that the instruction segment of the prompt must be concise and decisive.[3]
- **Assigning roles**: One section of the prompt explicitly assigns a role or persona (for example, "You are an expert grader following a strict rubric"). This role clause belongs at the beginning so it conditions all subsequent behavior, forming the top layer of the anatomy.[3]

Intermediate chapters refine the anatomy:[3]
- **Separating data from instructions**: The tutorial stresses that instructions and background data should be in distinct sections, often with clear markers. For an evaluator, this means using explicit tags or headings such as:
  - `Instructions:` – how to evaluate.
  - `Rubric:` – criteria.
  - `Submission:` – text to evaluate.
- **Formatting output & speaking for Claude**: A dedicated output-format section instructs the model exactly what to return (for example: a numeric score, short rationale, and decision label). This is part of the prompt anatomy that constrains the evaluator’s answers.[3]
- **Precognition (thinking step by step)**: A reasoning section instructs the model to think step-by-step before answering, improving reliability on complex tasks like rubric-based grading.[3]
- **Using examples**: An examples section (few-shot) shows full input–evaluation–output patterns. Examples should be clearly distinguishable from the live task, so they become an explicit component of the overall anatomy.[3]

For an **LLM-based evaluator system prompt**, the tutorial implies a structure like:
1) role & high-level instructions,
2) detailed evaluation instructions & rubric,
3) clearly tagged submission/content,
4) step-by-step reasoning guidance,
5) strict output-format specification,
6) one or more labeled examples.
All of these are kept in distinct, clearly marked blocks to maximize robustness.[3]

-----

-----

### Source [19]: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

Query: According to Anthropic's prompting methodology, what is the 'prompt anatomy' and how can it be applied to create a robust system prompt for an LLM-based evaluator?

Answer: Anthropic’s article on effective context engineering describes **organizing prompts into distinct sections**, effectively specifying a practical **prompt anatomy** for robust agents and evaluators.[5]

The article recommends explicitly structured sections such as:[5]
- **Background information**: A section (for example `<background_information>`) containing persistent facts, system assumptions, and any static context the model should know. For an evaluator, this can hold task domain description or general evaluation principles.[5]
- **Instructions**: A dedicated section (for example `<instructions>`) that contains the normative guidance—what the model must do, must not do, priorities when rules conflict, and how to treat tools and context. This is the core of the evaluator system prompt.[5]
- **Tool guidance**: A section (such as `## Tool guidance`) that explains how and when to call tools or APIs, if the evaluator interacts with external systems. This is part of anatomy when evaluation requires external checks.[5]
- **Output description**: A separate section specifying the exact format, fields, and style of the model’s response. This could include JSON keys for scores, explanation text, and any required tags.[5]

The article notes that using **XML tags or Markdown headers** to delineate these sections helps models keep instructions and context separate, even as models become more capable.[5]

Applied to a **system prompt for an LLM-based evaluator**, this methodology leads to a robust structure like:[5]
- `<background_information>`: explanation of the evaluation setting and any global policies.
- `<instructions>`: precise description of how to interpret the rubric, how to handle ambiguity, and how to weigh different criteria.
- `## Output description`: the scoring schema (e.g., 1–5), required rationale length, and any pass/fail logic.
- Optional `## Tool guidance`: how to use tools for fact-checking or policy checks.

Anthropic emphasizes that this disciplined organization of sections—clear boundaries between background, instructions, tools, and output—constitutes effective prompt anatomy and directly improves reliability and controllability of agent-like evaluators.[5]

-----

-----

### Source [20]: https://aws.amazon.com/blogs/machine-learning/prompt-engineering-techniques-and-best-practices-learn-by-doing-with-anthropics-claude-3-on-amazon-bedrock/

Query: According to Anthropic's prompting methodology, what is the 'prompt anatomy' and how can it be applied to create a robust system prompt for an LLM-based evaluator?

Answer: An AWS–Anthropic best-practices article lays out an explicit **10-part prompt anatomy** for working with Claude 3 models, which can be directly adapted to LLM-based evaluators.[4]

The recommended sections are:[4]
1. **Task context** – Assign a role or persona and broadly define the task. For an evaluator, this could define it as an impartial grader following a rubric.[4]
2. **Tone context** – Specify the interaction style (for example, formal, concise, neutral) to ensure consistent evaluator tone.[4]
3. **Background data (context)** – Provide documents, images, or other information required for the evaluation, separated from instructions.[4]
4. **Detailed task description and rules** – Give precise rules for how the model should act and how to perform the evaluation, including constraints and safety requirements.[4]
5. **Examples** – Show solved examples illustrating ideal evaluations and outputs.[4]
6. **Conversation history** – Include prior dialogue if necessary for multi-turn evaluation workflows.[4]
7. **Immediate task description or request** – The specific evaluation the model should perform on this turn (for example, “Evaluate the following answer according to the rubric above”).[4]
8. **Think step-by-step** – Explicitly instruct the model to reason carefully and break the task into subtasks before responding.[4]
9. **Output formatting** – Define the exact response structure (scores, labels, rationales) and any syntactic constraints like JSON or XML.[4]
10. **Prefilled response** – Optionally prefill the response with tags such as `<thinking>` and `<response>` to shape verbosity and reasoning separation.[4]

When applied as a **robust system prompt for an LLM-based evaluator**, this anatomy yields:
- Early sections (1–4) embedded in the system prompt: evaluator persona, tone, context boundaries, and strict evaluation rules.
- Sections (5–10) used in system or user messages to provide examples, current submission, and enforce deliberate, structured, and well-formatted evaluation outputs.[4]

-----

</details>

<details>
<summary>What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?</summary>

### Source [21]: https://sebgnotes.substack.com/p/evaluator-optimizer-llm-workflow

Query: What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?

Answer: This source describes the evaluator-optimizer workflow and highlights several real-world failure modes and challenges.

A major challenge is **loop control and stopping conditions**. The article stresses the need for “rigorous loop run and exit criteria” and explicit **circuit breakers** to avoid infinite or unproductive loops, especially when evaluators keep asking for incremental changes without real quality gains.[1] It recommends clear **success conditions** and **failure recovery paths**, including graceful degradation when no satisfactory solution emerges within a budgeted number of iterations or cost limit.[1]

Another issue is **evaluation subjectivity and criteria quality**. The evaluator must use criteria that are both **machine-readable** and **business-relevant**.[1] Vague, purely qualitative rubrics cause oscillation (different evaluations each pass) and make it hard to define a stable stopping rule. The author advises focusing engineering effort on robust evaluation criteria rather than “perfect generation prompts,” because weak evaluators lead to misleading scores and non-convergent loops.[1]

There are also **operational challenges**: the pattern needs quality assurance embedded into the architecture, not bolted on afterward.[1] That includes human oversight or autonomous overseer agents to intervene when the evaluator or optimizer behaves pathologically (e.g., reward hacking, degenerate solutions that technically satisfy the rubric but violate business intent).[1] The system should support **fallback behaviors** (simpler responses, less ambitious workflows) when the loop cannot find an optimal solution, to avoid user-visible failures or runaway cost.[1]

Finally, the article notes that implementing this in production requires attention to **long‑term operational considerations** such as monitoring quality over time, tuning thresholds, and ensuring the loop doesn’t overfit to narrow metrics at the expense of broader user satisfaction.[1]

-----

-----

### Source [22]: https://dev.to/clayroach/building-self-correcting-llm-systems-the-evaluator-optimizer-pattern-169p

Query: What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?

Answer: This source reports production experience using the evaluator-optimizer pattern for LLM‑generated SQL and surfaces several concrete challenges.

One key challenge is **defining objective evaluations**. The system succeeds because the evaluation is crisp: SQL either executes or returns a specific error code.[2] Where evaluation is less binary, subjective criteria would make it difficult to tell whether a new iteration is an improvement and when to stop. Here, the stopping condition is tied to a clear event: the query runs successfully.[2]

Another real-world issue is **handling external system errors and rate limits**. The team had to build intelligent retry logic for multiple LLM providers with different rate limiting patterns, using exponential backoff, respecting long retry-after headers, and only retrying on rate-limit errors (429), not on true failures.[2] Without this, the evaluator-optimizer loop could fail intermittently, stall, or amplify provider throttling.

The article also highlights **non-convergence and fallback**. Sometimes LLMs fail to self-correct; in these cases, rule‑based optimizations are used as a safety net to still produce a working query.[2] This avoids loops that keep generating variants that fail on the same error. The rule-based layer encodes common failure patterns (e.g., wrong table names, misuse of aggregates) and ensures forward progress when the optimizer stalls.[2]

A further challenge is **preserving task intent across iterations**. The system must ensure that only syntax is changed and the analysis goal stays intact, so the evaluator’s feedback (error messages, classifier outputs) is tightly scoped to syntactic and schema issues rather than reinterpreting the user’s question.[2] This containment is crucial for trust and for well-defined stopping.

Finally, the source notes that prior to this pattern, engineers faced “unpredictable failures” and repeated mistakes, illustrating that without good evaluation signals, the system could not learn from past iterations or know when it had truly fixed an issue.[2]

-----

-----

### Source [23]: https://azure.github.io/logicapps-labs/docs/logicapps-ai-course/build_multi_agent_systems/evaluator-optimizer/

Query: What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?

Answer: This Microsoft Logic Apps module explains the evaluator-optimizer pattern and emphasizes challenges around evaluation design and convergence in production workflows.

It describes the pattern as a loop where an optimizer agent generates or improves outputs and an evaluator agent provides scores or feedback until a **quality threshold or stopping condition** is met.[3] A central difficulty is **defining clear, actionable evaluation criteria** that the evaluator can apply consistently.[3] Ambiguous or overly subjective criteria undermine the pattern’s “objective assessment” goal and make it hard to determine when the loop should terminate.

The module points out that the pattern is suitable when you have **clear, actionable evaluation criteria** and quality is more important than speed.[3] In practice, this means that in domains with inherently subjective quality (e.g., creative writing, nuanced tone), it is harder to construct evaluators that yield stable, reproducible scores, and this complicates the definition of stopping rules based on reaching a threshold.

It also calls out the need to **design for convergence** by detecting when further improvement is minimal or unnecessary.[3] A practical challenge is recognizing diminishing returns from iterative refinement, especially when evaluator scores fluctuate slightly between iterations. Systems must implement logic to treat small changes as noise and stop, rather than looping indefinitely in pursuit of marginal gains.

The best practices section recommends keeping feedback **specific and actionable**; otherwise the optimizer may wander or oscillate instead of converging on a solution that passes the quality gate.[3] Another challenge in real-world systems is integrating nested agent workflows and managing their control flow, including ensuring that the evaluator’s decision to continue or stop is correctly wired and that failures in one agent do not deadlock the overall process.[3]

-----

-----

### Source [24]: https://docs.praison.ai/docs/features/evaluator-optimiser

Query: What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?

Answer: This documentation for PraisonAI’s “Agentic Evaluator Optimizer” feature outlines the pattern and includes troubleshooting guidance that reveals common failure modes.

The evaluator-optimizer is described as a **feedback loop workflow** where LLM outputs are evaluated and refined iteratively to improve accuracy and relevance.[4] Challenges arise when this loop does not behave as intended.

Under **Generation Issues**, the docs note the case where “generation is not improving.” Recommended checks include reviewing generator instructions and verifying that feedback is actually being integrated.[4] This points to a frequent failure: the optimizer ignores or misinterprets evaluator feedback, leading to repeated similar outputs and no convergence.

Another troubleshooting section, **Evaluation Flow**, covers issues where “the evaluation cycle is incorrect.” The guidance is to verify evaluation criteria, check condition mappings, and review feedback loop connections.[4] This reflects operational problems such as misconfigured thresholds, incorrect branching logic, or mismatches between evaluator outputs (scores, labels) and the conditions used to decide whether to keep iterating or stop.

These issues directly affect **stopping conditions**: if evaluation criteria or condition mappings are wrong, the loop may terminate prematurely (accepting low-quality outputs) or never terminate (never detecting success).[4]

The need to “check feedback integration” also hints at **subjectivity and ambiguity** in evaluation signals—if criteria are not sufficiently precise, the optimizer may fail to translate generic critiques into concrete changes, resulting in minimal improvement per iteration.[4] The documentation implicitly assumes that evaluation criteria must be well-defined and tightly connected to generation prompts; otherwise the evaluator-optimizer pattern will underperform or fail to converge in real deployments.[4]

-----

-----

### Source [25]: https://dylancastillo.co/til/evaluator-optimizer-pydantic-ai.html

Query: What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?

Answer: This article explains an evaluator-optimizer workflow built with Pydantic AI and illustrates practical challenges the pattern is meant to address.

The pattern is defined as having an LLM **generator** that produces a solution and an LLM **evaluator** that decides whether the solution meets predefined criteria.[5] If not, the generator receives feedback and tries again, repeating until conditions are satisfied or a maximum number of attempts is reached.[5] The explicit mention of a max-attempts limit reflects a common production requirement: guardrails against infinite or excessively long loops when improvement stalls.

The article emphasizes that outputs are evaluated “against predefined criteria,” underscoring that evaluation must be **explicit and structured**.[5] When criteria are underspecified or subjective, the evaluator cannot reliably determine pass/fail, making stopping conditions ill-defined. The example uses concrete constraints (e.g., response structure, content requirements) that the evaluator can check deterministically, showing how to avoid ambiguity.[5]

It also notes that instead of retrying the original task blindly, the generator receives **targeted feedback** from the evaluator.[5] When feedback is vague, the generator may make arbitrary changes that do not move it closer to satisfying the criteria, leading to repeated failures over multiple iterations. The pattern therefore depends on feedback that is specific enough to guide optimization.

Finally, the workflow is implemented programmatically with Pydantic models enforcing schemas.[5] This addresses a common failure mode: outputs that superficially look better but still violate hidden structural expectations. By using schema validation as part of evaluation, the system can objectively decide whether to stop (schema satisfied) or continue (violations remain) instead of relying on subjective judgments.[5]

-----

-----

### Source [73]: https://sebgnotes.substack.com/p/evaluator-optimizer-llm-workflow

Query: What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?

Answer: The article describes several practical pitfalls and safeguards when implementing the evaluator‑optimizer pattern in production.

On **evaluation subjectivity**, it stresses that evaluation criteria must be both **machine‑readable and business‑relevant**, otherwise the evaluator’s scores become inconsistent and hard to trust, which undermines the loop.[1] It recommends designing **explicit success criteria** that map directly to business goals instead of vague notions of quality, and embedding quality‑assurance direction (human or autonomous LLM oversight) into the system architecture from the start rather than as an afterthought.[1] The author notes engineers should focus on **robust evaluation criteria rather than perfect generation prompts**, implying that weak or ambiguous evaluators are a primary failure mode.[1]

Regarding **stopping conditions and loop control**, the article highlights the need for **rigorous loop run and exit criteria**.[1] It explicitly recommends implementing **circuit breakers to prevent infinite loops**, and defining **clear success conditions and failure recovery** paths so the system knows when to stop iterating, when to fall back, and how to surface unresolved failures.[1] Another recommendation is to enable **graceful degradation**, where the system can **fall back to simpler responses when optimal solutions aren’t found**, instead of looping indefinitely or failing silently.[1] This guards against non‑converging optimization or over‑iteration that wastes tokens and latency.

Operationally, the pattern is framed as requiring attention to both **immediate technical requirements** (loop control, criteria, fallbacks) and **long‑term operational considerations** (maintainable criteria, oversight mechanisms, reliability).[1] These points collectively identify common real‑world challenges: subjective or poorly specified evaluators, lack of precise exit conditions, and absence of safety mechanisms such as circuit breakers and graceful degradation.

-----

-----

### Source [74]: https://dev.to/clayroach/building-self-correcting-llm-systems-the-evaluator-optimizer-pattern-169p

Query: What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?

Answer: This production case study focuses on LLM‑generated SQL using the evaluator‑optimizer pattern and surfaces concrete failure modes and mitigations.

For **evaluation objectivity**, the system uses **ClickHouse itself as the evaluator**: SQL either executes successfully or returns a specific error code.[2] The author highlights that the pattern works well here because of **clear evaluation criteria** and **demonstrable improvement**—each iteration fixes a concrete, machine‑verifiable issue rather than relying on subjective quality judgments.[2] This suggests that using subjective evaluators instead of deterministic checks (syntax validity, execution success) would be a key challenge in other domains.

On **stopping conditions**, the loop continues until the SQL executes successfully or other safeguards trigger. The article emphasizes **selective retries** and rule‑based fixes when LLM self‑correction fails.[2] Rule‑based fallbacks apply common patterns (e.g., converting `count() * column` to `sum(column)`, moving aggregates from WHERE to HAVING) so that the system can **exit the loop with a workable query** instead of cycling indefinitely.[2] This constitutes a practical stopping strategy: success when the query runs; otherwise, exit after deterministic transformations.

The discussion of **rate limiting and retry logic** exposes another real‑world challenge: integrating the evaluator‑optimizer loop with external APIs that can return long `retry-after` headers and heterogeneous rate‑limit behaviors.[2] The solution uses intelligent retry logic (respecting long delays, exponential backoff with jitter, and only retrying on rate‑limit errors), which prevents the loop mechanism from degenerating into uncontrolled retries.

The author contrasts the **"before" state** (unpredictable failures, repeated mistakes, high operational cost) with the **"after" state** where automated recovery, consistent improvement, and caching reduce cost and increase trust, implicitly showing that without robust evaluators, fallbacks, and retry policies, the pattern fails to gain production trust.[2]

-----

-----

### Source [75]: https://azure.github.io/logicapps-labs/docs/logicapps-ai-course/build_multi_agent_systems/evaluator-optimizer/

Query: What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?

Answer: Microsoft’s Logic Apps module describes the evaluator‑optimizer pattern and provides guidance on loop design, which surfaces key challenges around evaluation and stopping.

The pattern is defined as an iterative workflow where an **optimizer agent** generates or improves outputs and an **evaluator agent** provides feedback or scores, repeating until a **quality threshold or stopping condition** is met.[3] The documentation underscores that **objective assessment** requires the evaluator to apply **clear criteria for structured, repeatable evaluation**, directly addressing subjectivity as a risk in real deployments.[3]

Under **Key Benefits**, it notes **convergence detection**, meaning the workflow should stop when **further improvement is minimal or unnecessary**.[3] This points to a common challenge: without explicit convergence logic, systems may over‑iterate, accruing latency and cost without meaningful gains.

The **Best Practices** section directly targets failure modes:
- **Keep feedback actionable**: Evaluator feedback must be clear and specific; vague or high‑level judgments impede the optimizer’s ability to improve, effectively stalling the loop.[3]
- **Define quality gates**: Implement **measurable criteria for success**, which serve as concrete stopping conditions and reduce subjective interpretation.[3]
- **Design for convergence**: Build mechanisms that detect when additional iterations do not significantly improve quality, to avoid unnecessary looping.[3]

The example scenario (iteratively improving a **sales performance report**) shows that the loop proceeds only while the evaluator deems quality insufficient; once the defined quality standard is met, the workflow returns the final report.[3] This illustrates the importance of intentionally modeling both **"pass" thresholds** and **loop exit paths**, making explicit that production implementations must treat stopping logic and evaluation design as first‑class concerns, not incidental details.

-----

-----

### Source [76]: https://docs.praison.ai/docs/features/evaluator-optimiser

Query: What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?

Answer: This documentation describes a generic **Agentic Evaluator Optimizer** feature and highlights troubleshooting points that relate to common failure modes.

The pattern is defined as a **feedback loop workflow** where LLM‑generated outputs are **evaluated, refined, and optimized iteratively** to improve **accuracy and relevance**.[4] Core capabilities include **solution generation**, **quality evaluation**, and a **feedback loop** implementing continuous improvement cycles.[4]

Under **Troubleshooting**, the docs directly identify typical implementation problems:

- **Generation issues – "generation is not improving"**: Recommended checks are to **review generator instructions**, **check feedback integration**, and enable verbose mode for debugging.[4] This points to failure modes where the evaluator’s feedback is either not connected correctly or not sufficiently specific for the optimizer to act on, leading to ineffective iterations.

- **Evaluation flow issues – "evaluation cycle is incorrect"**: The guidance is to **verify evaluation criteria**, **check condition mappings**, and **review feedback loop connections**.[4] These items map to challenges in configuring evaluation logic so that scores and conditions are interpreted properly, and ensuring the loop branches (continue vs stop) are wired correctly. If condition mappings are misconfigured, the system may loop when it should stop or terminate prematurely.

Although the documentation is high‑level, the emphasis on **verification of evaluation criteria** and **feedback loop connections** implicitly addresses challenges with evaluation subjectivity and stopping conditions: criteria must be precise enough to drive automated decisions, and the control flow that determines when to continue or stop must be correctly implemented to avoid stuck or runaway loops.[4]

-----

-----

### Source [77]: https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns

Query: What are common failure modes and challenges when implementing the evaluator-optimizer pattern in a real-world production environment, particularly regarding evaluation subjectivity and stopping conditions?

Answer: The Spring AI blog on agentic patterns describes the evaluator‑optimizer pattern and discusses design considerations that touch on evaluation and loop control.

The pattern is characterized as a **dual‑LLM process** where one model **generates responses** and another **provides evaluation and feedback**, iterating until the output meets defined standards.[9] While the article focuses on implementation in Spring ecosystems, it alludes to the importance of **clear evaluation criteria** so the evaluator can provide structured feedback that the optimizer can reliably use.[9] Implicitly, ambiguous or overly subjective criteria would limit the pattern’s effectiveness.

In the broader discussion of agentic patterns, the blog emphasizes that workflows should be designed with **explicit control over iteration and stopping** to avoid uncontrolled loops that increase latency and cost.[9] For evaluator‑optimizer, this means defining **quality thresholds** or **maximum iteration counts** to determine when the loop should terminate.[9]

The article also stresses that **quality is often more important than speed** for use cases where this pattern is appropriate, but still recommends avoiding unnecessary iterations by carefully modeling **convergence conditions** and ensuring that each evaluation cycle provides actionable guidance rather than generic critique.[9]

Although it does not enumerate failure modes explicitly, the guidance implies that common real‑world challenges include:
- Evaluators that return **non‑actionable or inconsistent feedback**, making improvement hard to measure.[9]
- Missing or weak **stopping conditions**, leading to long or ineffective loops.[9]

By framing evaluator‑optimizer as one pattern among several, the blog also suggests that misuse—e.g., applying it where clear evaluation signals are unavailable—can cause fragile systems, indirectly underscoring the risks of subjectivity and poorly defined stop criteria.[9]

-----

</details>

<details>
<summary>What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?</summary>

### Source [26]: https://bootcamptoprod.com/spring-ai-evaluator-optimizer-pattern-guide/

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This article defines the **Evaluator–Optimizer pattern** as a loop where a *Generator* produces an output, an *Evaluator* checks it against criteria, and a control signal (e.g., `PASS` vs `NEEDS_IMPROVEMENT`) decides whether to stop or iterate.[1]

It notes the pattern works **best when evaluation criteria are clear and objective**, such as word-count limits, required tone, or safety constraints.[1] In its email example, the stopping condition is a simple **80‑word rule**: the evaluator passes only drafts under the limit; otherwise it returns a verdict and suggestions, and the loop continues until the constraint is met.[1]

For more subjective tasks like **creative writing**, the article suggests that iterative refinement still adds value but warns of trade‑offs: **more rounds increase quality but also time and cost**, so developers should **cap the number of iterations** (e.g., start with 3–4 rounds) as a practical stopping condition when “perfection” is undefined.[1]

The piece also highlights implementation challenges that become acute for subjective evaluation:
- **Defining “good enough”**: the pattern assumes you can specify what a good output looks like; this is easy for format and length, but much harder for qualities like creativity or style.[1]
- **Evaluator reliability**: LLM evaluators sometimes return **invalid or unexpected formats**; systems must handle such failures gracefully instead of looping indefinitely.[1]
- **Cost–quality balancing**: because each loop invokes the model again, developers must **balance quality against latency and API cost**, using max‑iteration caps or fallback conditions as alternative stopping criteria when the evaluator never confidently returns `PASS`.[1]

The article implicitly suggests that for subjective domains, you often combine: a **hard stop by iteration limit**, plus **minimal objective checks** (e.g., no policy violations, length bounds) and accept that “good enough” may be approximated rather than formally defined.[1]

-----

-----

### Source [27]: https://dev.to/clayroach/building-self-correcting-llm-systems-the-evaluator-optimizer-pattern-169p

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This article describes an evaluator–optimizer workflow for **LLM‑generated SQL**, where evaluation is highly objective: queries either **parse and execute** or fail with specific errors.[2]

It emphasizes that the pattern works smoothly when you have:
- **Clear evaluation criteria** (“does the SQL execute without error?”).[2]
- **Demonstrable improvement per iteration** (each loop fixes one concrete issue such as a wrong table name or invalid aggregation).[2]

The practical stopping conditions in this objective domain are:
- **Success condition**: the evaluator (ClickHouse) confirms the SQL parses (`EXPLAIN AST`) and runs (`SELECT … LIMIT 1`) without errors, at which point the loop stops.[2]
- **Failure caps and fallbacks**: when LLM self‑correction fails, the system uses **rule‑based fallbacks** (e.g., rewrite certain common patterns) to avoid unbounded retries, turning potential infinite loops into deterministic correction steps.[2]

Although the task here is not creative writing, the article illustrates challenges that become harder when evaluation is subjective:
- The pattern’s strength relies on **binary or near‑binary signals** (success/failure); in creative tasks there is no equivalent automatic “executes without error” check, so termination is fuzzier.
- Their success depends on **cost‑efficient probes** (cheap checks with clear signals); for creative quality, probes often require expensive human review or model‑as‑judge scoring.

By contrast, the article’s solution hints at alternative stopping conditions applicable to subjective tasks:
- Use **bounded numbers of optimization attempts** instead of waiting for a perfect success condition.[2]
- Introduce **rule‑based constraints** (e.g., format, length, safety) that remain objective even when higher‑level quality is subjective, and stop once those are met and further changes yield diminishing returns.[2]

-----

-----

### Source [28]: https://docs.futureagi.com/future-agi/get-started/optimization/optimizers/meta-prompt

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This documentation describes a **Meta‑Prompt Optimizer** that uses a “teacher” LLM to iteratively refine prompts via systematic failure analysis.[3] It includes a dedicated example for **“Creative Writing with Constraints.”**[3]

In that example, the system optimizes prompts to generate short stories satisfying a task description:
- 200–300 words
- Clear beginning, middle, and end
- Vivid sensory details
- Tone matching the prompt
- Suitability for a general audience[3]

These become **explicit evaluation criteria** for the evaluator component, even though aspects like “vivid sensory details” and “engaging” are subjective in nature.[3] The system addresses this by using an **LLM‑based evaluator** (a teacher) that applies these criteria at scale.[3]

Key practical choices relevant to stopping conditions:
- The optimizer is configured with a fixed **`num_rounds` (e.g., 6)**, which acts as a **hard cap on iterations** regardless of whether an ideal prompt is found.[3]
- The evaluator operates on an **evaluation subset** (`eval_subset_size`), not the full dataset, trading off evaluation fidelity against cost.[3]

These decisions highlight core challenges in subjective evaluator–optimizer setups:
- **No crisp pass/fail**: for creative prompts, the system cannot rely on a binary success metric, so it runs a **fixed number of refinement rounds** and assumes improvements plateau.[3]
- **Subjective metrics encoded as checklists**: criteria like narrative structure or tone are turned into semi‑structured checklists the evaluator LLM can score, acknowledging that judgments (e.g., “engaging” or “vivid”) remain model‑dependent.[3]

Thus, for creative writing–type tasks, the documentation effectively recommends:
- Turning subjective qualities into **concrete, multi‑bullet rubrics** the evaluator can apply.
- Using **fixed iteration budgets** (`num_rounds`) as the primary stopping condition when no definitive success criterion exists.[3]

-----

-----

### Source [29]: https://www.braintrust.dev/articles/systematic-prompt-engineering

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This article argues that prompt design should be treated as **systems engineering** with **specific, measurable outcomes**, not pure creative writing.[4] For optimization, it stresses **data‑driven evaluation**.

It distinguishes between **objective** criteria (e.g., format, factual accuracy) and **subjective** criteria (e.g., tone, perceived helpfulness).[4] For subjective aspects, it recommends **model‑based evaluation** (LLM‑as‑a‑judge) or **human judgment**, often in **hybrid scoring frameworks**.[4]

Relevant to evaluator–optimizer workflows in creative settings:
- **Automated scoring** can scale evaluation beyond manual review, but subjective metrics require **sophisticated evaluators**; this introduces noise and bias into the feedback loop.[4]
- Because different performance aspects need different evaluators, multi‑dimensional scoring is needed (e.g., separate scores for structure, creativity, clarity, safety), complicating **stopping conditions**.

The article implies several practical stopping strategies under subjectivity:
- Use **quantified rubrics** where the evaluator assigns numeric scores to subjective dimensions; stop when scores exceed a chosen threshold for some number of dimensions or when improvements flatten across iterations.
- Employ **hybrid evaluation**: objective checks (format, length, safety) provide hard pass/fail gates, while subjective scores inform whether further optimization is worthwhile.[4]

It also notes that over‑focusing on a single metric can lead to **over‑optimization and metric gaming**, a significant challenge when subjective qualities are reduced to numbers.[4] This is highly relevant to creative writing, where chasing an evaluator’s taste can degrade genuine originality, suggesting a practical need for:
- **Maximum iteration limits** to avoid overfitting to the evaluator.
- **Periodic human spot checks** as an external corrective signal when subjective metrics drift.[4]

-----

-----

### Source [30]: https://askrally.com/article/dspy-with-gepa

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This article discusses using the **Evaluator–Optimizer pattern in DSPy with GEPA** to train an **LLM‑as‑a‑Judge** and then optimize **creative generative tasks** (for example, joke generation).[5]

In the described setup, the judge LLM reasons step‑by‑step (“the judge thinks first before making the decision”) and its internal reasoning is surfaced to guide optimization, effectively giving the optimizer a “look into the mind of the judge.”[5] This allows the system to **hill‑climb** toward better outputs according to the judge’s preferences.[5]

For a creative task such as **original joke generation**, the evaluation metric is inherently subjective (e.g., freshness, clear setup and punchline, topic relevance); the article notes that you can **feed the evaluation metric and the judge’s reasoning back** into the optimizer so it can iteratively improve outputs.[5]

The discussion highlights several challenges specific to subjective evaluator–optimizer loops:
- The **evaluation criteria are value‑laden** ("make it feel fresh," avoid stale dad jokes, match a comedian’s style without copying), so the judge’s taste materially shapes what the system converges to.[5]
- Since there is no objective ground truth of “funny,” the system relies on the **judge’s internal scoring**; this risks **mode collapse** toward what the judge happens to reward.[5]

Implied alternative stopping conditions for such subjective tasks include:
- **Hill‑climbing until no further improvement** is seen in the judge’s score across a few iterations (local optimum).[5]
- Applying **iteration caps** or **diminishing‑returns heuristics**: stop when incremental gains in judged quality fall below a small margin, to avoid excessive cost and overfitting to the judge.[5]

Overall, the article frames subjectively judged creative tasks as optimization over a learned, model‑defined metric, which demands explicit controls on iteration count and awareness of evaluator bias.[5]

-----

-----

### Source [53]: https://bootcamptoprod.com/spring-ai-evaluator-optimizer-pattern-guide/

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This source defines the **Evaluator–Optimizer pattern** as a loop where a *Generator* produces an output and an *Evaluator* checks it against predefined criteria, returning either a PASS or NEEDS_IMPROVEMENT verdict, with feedback used for another iteration.[1] It emphasizes that the pattern works best when there are **clear, objective evaluation criteria**, such as word count limits, thread safety, or a required tone in an email.[1]

For subjective tasks like creative writing, the article implicitly highlights several **practical challenges**:

- The pattern assumes the Evaluator can decide when the output is "good enough" based on **explicit rules**; this is straightforward for an 80‑word limit but much harder for qualities like originality or emotional impact, which are not crisply specifiable.[1]
- The **stopping condition** in the example is purely rule-based (email under 80 words and policy compliant). For creative tasks, such a crisp rule is usually absent, so it is unclear when to terminate the loop without over‑editing or cycling.[1]
- The article notes a **trade‑off between more rounds and cost/latency**: more iterations tend to improve quality but increase time and expense.[1] In subjective domains where there is no clear convergence signal, this trade‑off becomes harder to manage.
- It mentions that systems must handle **invalid or unexpected responses** from the model gracefully.[1] When evaluation criteria are fuzzy, such invalidity can extend to inconsistent or unstable judgments from the Evaluator.

As for **alternative stopping conditions**, the article suggests:

- Using a **fixed maximum number of iterations** (e.g., start with 3–4 rounds and tune from there).[1]
- Balancing **quality vs. speed/cost** as an explicit design choice, rather than waiting for perfect satisfaction of criteria.[1]

These ideas generalize to creative writing: designers may cap iterations or adopt heuristic thresholds instead of relying on a single, definitive PASS condition when evaluation is subjective.

-----

-----

### Source [54]: https://dev.to/clayroach/building-self-correcting-llm-systems-the-evaluator-optimizer-pattern-169p

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This source presents the evaluator–optimizer pattern in the context of **self‑correcting SQL generation**, stressing that the approach works particularly well when:

1. There are **clear evaluation criteria** (SQL either executes or returns an error).
2. Each iteration yields **demonstrable improvement** (fixing a specific syntax issue).
3. The **goal stays fixed** (the analysis task is unchanged; only syntax is corrected).
4. It is **cost‑efficient** to iterate on small corrections rather than regenerate everything.[3]

Although the domain here is SQL, the article implicitly exposes **limitations for subjective domains** like creative writing:

- The success of the pattern depends on having **binary or near-binary signals** (e.g., query runs vs. fails).[3] In creative writing, criteria such as style, voice, and emotional impact do not yield such crisp pass/fail feedback.
- Because the Evaluator in this setup keys off **objective failures** (syntax errors), the system can confidently continue iterating until errors disappear.[3] For creative writing, a similar objective error signal is missing, so convergence becomes ambiguous.

The article also suggests **fallback mechanisms** when the evaluator–optimizer loop cannot repair the output, such as **rule‑based fixes** or alternative patterns to ensure the query still runs.[3] Transposed to creative writing, this hints at alternative stopping or backup conditions like:

- Ending optimization after certain attempts and **falling back to simpler heuristics or templates**.
- Defining minimal “safety” or format checks that, once satisfied, allow stopping even if subjective quality may still be improvable.

In essence, this source underscores that evaluator–optimizer works best where evaluation is **objective and error-driven**, and it indirectly implies that for subjective criteria, practitioners may need **non-evaluative stopping rules** (max attempts, timeouts, or backup strategies) rather than waiting for a definitive PASS verdict.

-----

-----

### Source [55]: https://arxiv.org/pdf/2507.00769

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This paper introduces **LitBench**, a benchmark and dataset for **reliable evaluation of creative writing**, and discusses **reward models (verifiers)** for automatic assessment and optimization of creative-writing systems.[2] It directly addresses the difficulty of evaluation when criteria are inherently **subjective and preference-based**.

Key challenges highlighted:

- Even the best trained reward models, aligned to human preferences via pairwise comparisons, agree with human annotators on generated story pairs only about **57% of the time**, with a **40% disagreement rate**, indicating substantial residual subjectivity and noise in judgments.[2]
- Off‑the‑shelf LLMs used as judges can perform around chance on this benchmark, showing that naive evaluator-optimizer setups relying on generic LLM-as-judge are **unreliable for creative writing** without domain-specific fine‑tuning.[2]
- The authors find that adding chain‑of‑thought style reasoning to reward models can **degrade verdict accuracy** in creative-writing evaluation, contradicting patterns seen in math/code verification, and raising questions about how to structure evaluators in subjective domains.[2]

The paper also explores **rubric-based prompts** that break evaluation into dimensions such as **imagery, tension, structure/pattern, energy, insight, originality, and emotional impact**.[2] This shows that, for creative writing, evaluation requires **multi-dimensional, nuanced criteria**, and any single scalar score or PASS/FAIL signal may oversimplify rich stylistic trade‑offs.

Implications for evaluator–optimizer stopping conditions:

- Because automated evaluators only partially align with human taste, continuing to optimize until a model’s reward is maximized can **overfit to the evaluator** and diverge from human preferences, suggesting the need for **caps on optimization steps** or **regular human-in-the-loop checks**.[2]
- The authors note "head-room for richer supervision signals—such as rubric-based feedback or rationale distillation" to better align rewards with literary taste.[2] In a loop, this implies using **structured, rubric-level feedback** rather than a binary PASS condition, and stopping when incremental improvements on multiple dimensions plateau instead of waiting for an absolute standard.

Thus, this source documents both the **practical unreliability** of current automatic evaluators for creative writing and points toward **multi-criteria, bounded optimization strategies** as alternatives to rigid stopping rules.

-----

-----

### Source [56]: https://askrally.com/article/dspy-with-gepa

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This source discusses the **Evaluator–Optimizer pattern in DSPy with GEPA (Generalized Evaluator–Policy Architecture)**, focusing on **LLM-as-a-Judge** for optimizing creative generative tasks, including creative writing.[4] It emphasizes that an evaluator model (judge) produces scores or feedback that guide a generator (policy) toward better outputs.

The article notes several **practical challenges** for subjective domains:

- Evaluators for creative tasks rely on **learned preference models or rubric-like prompts**, which are inherently **approximate** representations of human taste.[4]
- The evaluator’s signal can be **noisy or unstable**, especially when judging open-ended, stylistic outputs, making it difficult to know when the optimization has truly converged or is just chasing evaluator idiosyncrasies.[4]

To deal with these issues, the GEPA/DSPy setup uses **alternative stopping and control strategies** beyond a single PASS condition:

- Treating the evaluator’s output as a **continuous score** and stopping optimization when **marginal gains fall below a threshold**, i.e., when additional iterations do not substantially improve the score.[4]
- Using **validation splits** or separate evaluation prompts to monitor whether the generator is simply overfitting to one evaluator; optimization can be stopped when performance diverges across evaluators, indicating misalignment.[4]
- Applying **budget-based stopping conditions**, such as a fixed number of optimization steps or a cap on API calls/compute, which is particularly important when dealing with high-variance, subjective metrics.[4]

The article frames evaluator–optimizer for creative tasks as closer to **black-box optimization over noisy, preference-based objectives**, and suggests that effective practice involves:

- Multi-metric evaluation (e.g., style, coherence, and originality) rather than a single scalar.
- Pragmatic **early stopping** rules and resource constraints, instead of waiting for a crisp notion of "perfection" that may not exist in subjective creative writing contexts.[4]

-----

-----

### Source [57]: https://ai.plainenglish.io/agentic-ai-a-deep-dive-into-the-evaluator-optimizer-workflow-and-gaia-benchmark-7c1e4257982e

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This article provides a broad overview of the **Evaluator–Optimizer workflow** within agentic AI, explaining that an LLM generator repeatedly refines its output based on feedback from an LLM evaluator until some quality threshold is reached.[5] It emphasizes that the pattern is powerful when coupled with **well-defined benchmarks** like GAIA for complex reasoning tasks.[5]

Regarding **subjective domains** such as creative writing, the article notes that the evaluator–optimizer pattern faces specific **limitations** when there is no clear ground truth.[5] Unlike question answering or code execution, creative writing does not allow simple correctness checks, so evaluators must rely on **heuristics, style rubrics, or preference models**, which may not fully capture human judgments.[5]

The article discusses **practical design choices** for stopping conditions when criteria are fuzzy:

- Using **threshold-based stopping on evaluator scores**: for example, ending the loop when a response exceeds a certain rubric-based quality score.[5]
- Employing **iteration caps** to ensure bounded cost and avoid endless refinement, particularly when gains become marginal or oscillatory.[5]
- Combining **multiple evaluators or metrics**, and stopping when improvement on one metric leads to degradation on another, signaling a trade‑off rather than unambiguous progress.[5]

It also warns about **over-optimization**: tuning outputs too aggressively to please a particular evaluator can lead to unnatural or degenerate text that scores well numerically but is less satisfying to humans.[5] This risk is especially acute in creative writing because the notion of quality is **multi-dimensional and context-dependent**.

Therefore, the article recommends treating evaluator–optimizer in subjective settings as a **tool for incremental polishing**, governed by **heuristic stopping rules** (score thresholds, max rounds, and multi-metric balance) rather than a guarantee of reaching an objectively optimal creative output.[5]

-----

-----

### Source [58]: https://dylancastillo.co/til/evaluator-optimizer-pydantic-ai.html

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This post presents a concrete implementation of the **evaluator–optimizer workflow** using PydanticAI, where an **LLM generator** proposes an answer and an **LLM evaluator** checks whether it meets certain criteria, returning structured feedback and a verdict to either accept or ask for another attempt.[6]

The author underscores that the workflow is most straightforward when the evaluator can perform **clear, rule-based checks**, such as verifying required fields, formats, or constraints.[6] In these cases, the stopping condition is simple: stop when the evaluator returns a **"pass" verdict**.[6]

The post also highlights several **engineering challenges** that become more pronounced for subjective tasks like creative writing:

- Handling **ambiguous or partial evaluator outputs**: the system must robustly parse the evaluator’s verdict and feedback, which can be less consistent when criteria are qualitative.[6]
- Preventing **infinite or unproductive loops**: if the evaluator continues to request changes that the generator fails to satisfy in a stable way, the workflow needs safeguards.[6]

As a remedy, the author suggests and demonstrates **alternative stopping conditions** beyond the evaluator’s pass/fail signal:

- Enforcing a **maximum number of optimization iterations**, after which the system returns the best attempt so far.[6]
- Optionally tracking the "best" output according to some score or heuristic and returning it if subsequent iterations do not improve it, approximating an **early stopping** criterion.[6]

While the examples are not about creative writing specifically, the implementation pattern generalizes: for subjective evaluators (e.g., style judges), systems should combine the evaluator’s qualitative feedback with **hard iteration limits and simple quality heuristics** to avoid overfitting to noisy judgments or getting stuck without a clear convergence signal.[6]

-----

-----

### Source [59]: https://www.braintrust.dev/articles/systematic-prompt-engineering

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This article describes **data-driven prompt optimization** and treats LLM prompts as objects that can be systematically improved using evaluation metrics over datasets.[7] While not exclusively about the evaluator–optimizer pattern, it outlines principles that are directly relevant when evaluation criteria are **subjective**, such as in creative writing.

The core idea is to **iterate prompts** based on feedback from evaluation metrics, but it emphasizes several **practical issues**:

- For subjective tasks, metrics are often **proxy signals**, such as human preference labels, rubric scores, or LLM-judged ratings, which can be noisy and biased.[7]
- Over‑optimization on a narrow metric can cause the system to **"chase the metric"** at the expense of true human-perceived quality, a phenomenon akin to Goodhart’s law.[7]

To mitigate this, the article proposes **alternative stopping and control strategies**:

- Use **validation sets** and stop optimization when performance on held‑out data stops improving or begins to degrade, providing an analogue of **early stopping** in machine learning.[7]
- Combine **multiple metrics** (e.g., fluency, diversity, and user satisfaction) and stop when improvements on one metric lead to regressions on others, indicating a trade‑off frontier rather than a single optimum.[7]
- Impose **budget constraints** (e.g., fixed number of prompt variants or evaluation runs), so optimization cannot continue indefinitely even if metrics are noisy.[7]

The article underscores that, in subjective domains like creative writing, prompt or system optimization should be viewed as **exploration under uncertainty**, not as convergence to a uniquely correct solution. This motivates the use of **bounded iteration, multi-metric monitoring, and validation-based early stopping** rather than waiting for a single evaluator to declare that criteria are perfectly satisfied.[7]

-----

-----

### Source [60]: https://docs.futureagi.com/future-agi/get-started/optimization/optimizers/meta-prompt

Query: What are the practical challenges and alternative stopping conditions for the evaluator-optimizer pattern when evaluation criteria are subjective, such as in creative writing?

Answer: This documentation describes the **Meta-Prompt Optimizer**, which uses a "teacher" LLM to conduct **deep reasoning-based prompt refinement** via systematic failure analysis and rewriting.[8] It is effectively an evaluator–optimizer setup where the teacher analyzes prompt failures and proposes improved prompts in multiple iterations.[8]

The doc notes several aspects relevant to **subjective evaluation** such as creative writing prompts:

- The teacher LLM’s feedback is based on **natural-language reasoning** about why a prompt failed and how it could be improved, rather than purely objective correctness checks.[8]
- Because such reasoning can be **open-ended and qualitative**, the system does not rely on a single decisive PASS condition but instead applies **iterative refinement** until certain practical limits are reached.[8]

The Meta-Prompt Optimizer defines **alternative stopping conditions** beyond satisfying exact criteria:

- A **maximum number of refinement steps** to cap computational cost.[8]
- Detecting **diminishing returns**, where subsequent refinements yield minimal improvement on chosen evaluation metrics, which can include human feedback, task success rates, or LLM-judged scores.[8]
- The ability to plug in **custom evaluators or metrics**, including preference-based or rubric-based scores, and to stop when those stabilize or cross a desired threshold rather than seeking perfect performance.[8]

The documentation also warns that overly aggressive optimization on internal metrics can lead to **overfitting to the teacher’s preferences**, especially when the metrics are subjective.[8] It recommends periodically checking performance on **external or held-out tasks** and using that as a signal for early stopping.[8]

Applied to creative writing systems, these practices imply that evaluator–optimizer loops should employ **bounded iterations, metric plateau detection, and external validation checks** instead of treating the evaluator’s internal standard as an absolute measure of literary quality.[8]

-----

</details>

<details>
<summary>How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?</summary>

### Source [31]: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview

Query: How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?

Answer: Anthropic’s prompt engineering overview introduces the idea that prompts should be treated as structured artifacts composed of distinct parts (often called *prompt anatomy*), each with a clear purpose.[4] The docs emphasize separating:
- **High-level role and behavior** (system-style instructions)
- **Task-specific instructions**
- **Context and reference material**
- **Examples / demonstrations**
- **User input and output formatting requirements**[4]

For multi-step workflows, Anthropic recommends using *different prompts for different stages* instead of trying to make a single, overloaded prompt handle everything.[4] This maps naturally to separate **evaluator** and **optimizer** agents, each with its own system-level role, instructions, and success criteria.[4]

The overview stresses **clarity, explicit constraints, and conflict resolution rules**: when there are multiple instruction sources (system, tools, history, user), the higher-level/system instructions should explicitly govern how to handle conflicts.[4] Applied to an evaluator vs. optimizer, this means clearly encoding their *distinct objectives* and telling each agent how to behave when its objective conflicts with user requests or with other instructions.[4]

Anthropic also highlights **output schemas and formatting** as first-class parts of prompt anatomy.[4] For an evaluator, this means specifying structured outputs (scores, rationales, error tags), while for an optimizer it means specifying how to revise drafts or propose changes in a machine-consumable format.[4]

The docs encourage **iterative refinement** of each component of the prompt anatomy based on observed failure modes: clarify role, adjust instructions, add or update examples, and tighten output formats as you discover issues in how multi-step agents interact.[4] This iterative decomposition is central to creating robust prompts when different agents have partially conflicting priorities (e.g., quality vs. speed, creativity vs. safety).[4]

-----

-----

### Source [32]: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices

Query: How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?

Answer: The Claude 4 prompting best practices spell out a concrete *prompt anatomy* pattern: start with **concise role and behavior instructions**, then add sections for **task description**, **constraints / policies**, **tools**, and **output format**.[7] Prompts should be organized using clear headers or tags (e.g., “Background”, “Task”, “Steps”, “Output format”) so the model can easily parse what matters for each part of its behavior.[7]

Anthropic advises making **system-level goals explicit and stable** and then using lower sections to encode per-call details.[7] For multi-step workflows, this means giving an evaluator a system role like “You are a strict evaluator whose job is to assess outputs against criteria X” and an optimizer a role like “You are an optimizer whose job is to improve drafts within constraints Y”, and keeping those roles consistent over calls.[7]

The guide notes that when there are **tradeoffs or conflicting objectives**, the prompt should define explicit **priority ordering** (e.g., safety > correctness > style > length).[7] This is directly applicable when evaluator and optimizer have different emphases: each system prompt can encode its own prioritized list, ensuring that, for example, the evaluator treats adherence to spec as higher priority than fluency, while the optimizer might treat user satisfaction and clarity as primary within evaluator-enforced constraints.[7]

Best practices also recommend **separate evaluation prompts** for grading or critiquing model outputs, with instructions to reference explicit rubrics and avoid changing the original text unless explicitly asked.[7] This is the core of an evaluator agent: its prompt anatomy encodes a rubric, scoring method, and explanation requirements, while the optimizer’s anatomy encodes how to use that feedback to revise the work in subsequent steps.[7]

-----

-----

### Source [33]: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

Query: How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?

Answer: Anthropic’s article on effective context engineering generalizes prompt engineering into managing the entire **context state** of agents, including system prompts, tools, external data, and history.[3] It recommends structuring **system prompts** into clear **sections** such as `<background_information>`, `<instructions>`, `## Tool guidance`, and `## Output description`, using XML-style tags or Markdown headers.[3] This is exactly the kind of "prompt anatomy" useful for multi-step workflows.[3]

The article stresses that system prompts should be written at the **right altitude**—not so detailed that they become brittle logic, and not so vague that the agent improvises undesirably.[3] For an evaluator, this means high-level principles like “evaluate against these criteria and do not rewrite the text” plus a concise rubric, while for an optimizer it means “revise to satisfy the evaluator’s rubric while preserving user intent.”[3]

Anthropic recommends striving for the **minimal set of information** that fully outlines expected behavior and then iterating: start from a minimal system prompt, observe failures, and add targeted instructions or examples.[3] In a pipeline with conflicting priorities, you might first specify independent roles, then add clarifying instructions for how the optimizer should respond to evaluator feedback or how the evaluator treats tradeoffs (e.g., prioritizing safety and factuality over style).[3]

The piece also introduces **compaction** of message history, which is relevant when evaluators and optimizers exchange many messages.[3] Prompts that perform compaction should be carefully tuned to capture all **relevant** evaluator feedback and constraints for downstream optimization, preserving the parts of context that encode priorities and rubrics.[3] This ensures each agent’s system prompt plus preserved history accurately reflects the intended division of responsibilities and conflict-resolution behavior.[3]

-----

-----

### Source [34]: https://www.anthropic.com/research/building-effective-agents

Query: How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?

Answer: In "Building Effective AI Agents", Anthropic explains that modern systems are **multi-step agents** operating over long horizons, where system prompts and context design are central to reliability.[6] They describe agents that decompose tasks into substeps handled by different components, such as planners, workers, and evaluators.[6]

The article highlights the importance of **clear roles and responsibilities** for each agent: prompts should spell out what an agent is responsible for, what success looks like, and what it must *not* do.[6] For an evaluator vs. optimizer, this means:
- The **evaluator** prompt defines its job as *assessing* outputs using explicit criteria, not rewriting them.
- The **optimizer** prompt defines its job as *changing* outputs to better satisfy those criteria and the user’s goals, using evaluator feedback as a guiding signal.[6]

Anthropic emphasizes encoding **feedback loops** in the agent design: evaluators should produce structured feedback and possibly scores, while optimizers consume that feedback to improve their next attempt.[6] Prompt anatomy for evaluators therefore includes rubrics, scoring scales, error categories, and explanation requirements; the optimizer’s anatomy specifies how to interpret that structure, prioritize improvements, and handle cases where criteria appear to conflict.[6]

The post also notes that agents should have **safety and reliability constraints** baked into their highest-level instructions.[6] In scenarios with conflicting priorities (e.g., speed vs. thoroughness, creativity vs. adherence to spec), prompts should encode an explicit **priority ordering** and tell agents how to behave when tradeoffs arise.[6] This yields evaluator and optimizer system prompts that are robust and predictable when orchestrated together in a multi-step workflow.[6]

-----

-----

### Source [35]: https://www.anthropic.com/engineering/claude-code-best-practices

Query: How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?

Answer: The "Claude Code" best practices article discusses building **agentic coding workflows** where Claude acts in different roles such as planner, implementer, and reviewer.[5] These roles map closely to an **optimizer** (implementer) and **evaluator** (reviewer) pattern.[5]

Anthropic recommends writing **role-specific prompts** with distinct sections for background, goals, constraints, and step-by-step behavior.[5] For a reviewer/evaluator, the prompt should instruct the model to examine code against explicit criteria (correctness, safety, style, performance) and produce *structured feedback* rather than rewriting the entire solution by default.[5] For an implementer/optimizer, the prompt anatomy should describe how to use that feedback to iteratively improve the code while preserving user intent and any invariants (like API contracts or security requirements).[5]

The article stresses that prompts for these agents should encourage **incremental changes, explicit reasoning, and verification**, and that different roles may legitimately have **different priorities** (e.g., implementer optimizes for progress and practicality, reviewer optimizes for catching subtle issues).[5] System prompts should therefore encode these differing priorities and specify that when they conflict, certain constraints (like safety and correctness) outrank others (like speed or cosmetic style).[5]

Anthropic also notes the importance of **clear output formats** for reviewer prompts (e.g., bullet-point issues, severity, suggested fixes) so that implementer prompts can reliably consume this information in subsequent steps.[5] This is a direct application of prompt anatomy to multi-step workflows with evaluator and optimizer components that must interoperate consistently.[5]

-----

-----

### Source [36]: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-engineering-guidelines.html

Query: How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?

Answer: The Amazon Bedrock prompt engineering guidelines, while not Anthropic-specific, echo the same "prompt anatomy" concepts relevant for Anthropic models.[1] They describe a prompt as composed of several parts: **task/instruction**, **context**, **demonstration examples**, and **input text**, and recommend combining these components as needed for robust behavior.[1]

For complex workflows, Bedrock suggests using **prompt templates** that explicitly separate sections like instructions, examples, and questions, and notes that Anthropic Claude prompts submitted via Bedrock should wrap main content in a consistent conversational framing (e.g., `\n\nHuman:` / `\n\nAssistant:`).[1] This encourages a disciplined anatomy where system-like instructions and role definitions are clearly separated from user content.[1]

Applied to a multi-step evaluator–optimizer pipeline with conflicting priorities, these guidelines imply designing *two templates*:
- an **evaluator template** that foregrounds evaluation criteria, relevant context, and format for feedback;
- an **optimizer template** that foregrounds the instruction to revise or improve based on that feedback and the original task.[1]

The guidelines highlight including **few-shot examples** that show the model how to handle nuanced cases and ambiguities.[1] For evaluators, this means examples of good vs. bad evaluations and how to apply rubrics; for optimizers, examples of how to respond to specific evaluator comments while respecting higher-level constraints like correctness and safety.[1]

Bedrock also stresses avoiding confusing delimiter overlaps (e.g., not reusing `Human:`/`Assistant:` inside examples), which is relevant when chaining prompts: each agent template should maintain clean, unambiguous structure so that context from previous steps does not corrupt the system-level instructions.[1]

-----

-----

### Source [69]: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview

Query: How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?

Answer: Anthropic’s prompt engineering overview describes a **“prompt anatomy”** that is directly applicable to robust system prompts for multi-step workflows.[6]

Key structural elements:
- **System / meta instructions**: Top-level rules that define role, goals, constraints, and safety expectations. For multi-step workflows, the system section should clearly separate responsibilities for an **evaluator** (e.g., "assess quality against criteria, do not modify input") versus an **optimizer** (e.g., "improve the draft while preserving intent").[6]
- **Task and input specification**: Explicitly describe what the model should do on this turn, including the exact artifacts it receives (draft, rubric, previous evaluations, etc.) and what it must treat as read‑only vs. editable.[6]
- **Style and formatting guidance**: Specify output schemas (headings, JSON fields, step lists) so that downstream steps can reliably parse evaluator vs. optimizer outputs.[6]
- **Examples (few‑shot)**: Provide worked examples of both evaluation and optimization, especially when their priorities conflict (e.g., evaluator penalizes unsafe suggestions; optimizer must never improve quality by adding disallowed content). This helps the model internalize the intended trade‑offs rather than only verbal instructions.[6]

Anthropic recommends:
- **Clear sections / labels**: Use headings or tags like `## ROLE`, `## OBJECTIVE`, `## INPUT`, `## OUTPUT FORMAT`, `## EXAMPLES` to reduce ambiguity in complex workflows.[6]
- **Conflict resolution rules**: When multiple goals exist, prompts should explicitly rank them (e.g., "Follow safety and correctness over user satisfaction; if they conflict, say so and explain the trade‑off"). This is essential when the evaluator and optimizer are driven by partially conflicting priorities such as speed vs. rigor or creativity vs. policy compliance.[6]
- **Single source of truth**: Core safety, honesty, and non-deception rules live in the system section and are shared across all steps. Evaluator and optimizer prompts both restate how these higher-level rules override local objectives.[6]

In multi-step pipelines, Anthropic’s anatomy implies writing **separate, role-specific system prompts** that share a common safety/values spine but diverge in task sections and examples, so the evaluator and optimizer each specialize without undermining overall constraints.[6]

-----

-----

### Source [70]: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

Query: How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?

Answer: Anthropic’s article on **effective context engineering for AI agents** generalizes prompt anatomy to multi-step, agentic workflows and provides guidance on system prompts when multiple components have different objectives.[2]

On **system prompts and structure**:
- System prompts should use **simple, direct language** and operate at the **right altitude**—not so low-level that they encode brittle logic, and not so high-level that behavior is underspecified.[2]
- Anthropic recommends **organizing prompts into distinct sections**, e.g. `<background_information>`, `<instructions>`, `## Tool guidance`, `## Output description`, clearly delineated via XML tags or Markdown headers.[2]
- Regardless of format, you should aim for the **minimal set of information** that fully defines expected behavior, adding details only in response to observed failure modes.[2]

Applied to an evaluator vs. optimizer with conflicting priorities:
- Use a **shared background section** encoding global objectives (safety, honesty, non-deception, policy adherence), which both roles inherit.[2]
- Have **role-specific instruction sections**:
  - Evaluator: emphasize rigorous checking against criteria, conservatism about ambiguous cases, and explicit reporting of uncertainties or violations.
  - Optimizer: emphasize improvement within constraints, deference to evaluator feedback, and rules for not overriding explicit fail flags or safety concerns.[2]
- Include an **output description** section that defines separate, parseable schemas (e.g., `evaluation_result`, `optimization_suggestions`) so that each step’s outputs can be reliably consumed by the other or by an orchestrator.[2]

On **examples and edge cases**:
- Anthropic advises **few-shot examples** but discourages stuffing prompts with exhaustive edge-case lists. Instead, curate **canonical, diverse examples** that demonstrate how the agent should handle conflicts (e.g., when user goals clash with safety, or when an optimizer’s improvement would violate evaluator criteria).[2]

On **multi-turn / multi-step context**:
- The article frames this as **context engineering**: managing system instructions, tools, external data, and message history across steps.[2]
- For conflicting priorities, the context should include prior evaluation results and rationales, and the optimizer’s instructions should explicitly reference how to interpret and prioritize those signals relative to its own objective.[2]

-----

-----

### Source [71]: https://www.anthropic.com/research/building-effective-agents

Query: How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?

Answer: In **“Building Effective AI Agents,”** Anthropic connects prompt design (including system prompts) to agent architectures that often involve multiple skills or roles, which maps to evaluator/optimizer workflows.[4]

Relevant guidance:
- Agents should be built from **modular capabilities**, each with **clear responsibilities and interfaces**.[4] This aligns with creating distinct system prompts for an evaluator module (assessment, scoring, policy checks) and an optimizer module (revision, generation) rather than a single overloaded prompt.
- The **core system prompt** acts as a global contract: it encodes overarching goals (helpfulness, harmlessness, honesty) and constraints that all skills and steps must respect.[4] Evaluator and optimizer prompts should be layered on top of this shared core.

For conflicting priorities:
- The article stresses **explicit prioritization of objectives** in system instructions (e.g., safety and truthfulness must override user satisfaction or task completion).[4]
- When designing tools or sub-agents, Anthropic recommends **precise specifications and expectations** for each, including what inputs they receive, what they must output, and how they should behave in ambiguous or conflicting situations.[4]
- System prompts should clearly define what happens when a requested optimization would violate the evaluator’s criteria or global policies—e.g., the optimizer must decline or propose an alternative instead of silently complying.[4]

Prompt/interface design practices:
- Use **structured outputs** so that orchestrators can route between evaluator and optimizer deterministically (e.g., fields like `pass_fail`, `issues`, `recommended_changes`).[4]
- Build **feedback loops**: evaluator outputs (scores, rationales) are included in the optimizer’s context with instructions for how to address them; subsequent evaluations reference prior attempts.[4]

Overall, the paper implies that robust system prompts in multi-step workflows should reflect **agent-style decomposition**: a shared core prompt for global behavior, plus tightly scoped, clearly documented role prompts for conflicting sub-goals.[4]

-----

-----

### Source [72]: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills

Query: How does Anthropic's 'prompt anatomy' apply to creating robust system prompts for multi-step LLM workflows, specifically for an evaluator and an optimizer with conflicting priorities?

Answer: Anthropic’s **Agent Skills** article describes how agents load additional, role-specific instructions at runtime, which is directly relevant to separating evaluator and optimizer behaviors while maintaining a shared core prompt.[5]

Key ideas:
- The **context window** initially includes the **core system prompt** and metadata for all installed skills, plus the user’s message.[5]
- When a skill is triggered (e.g., a PDF skill in the example), Claude reads that skill’s `SKILL.md` and any linked files to augment its behavior for that episode.[5]

Applied to evaluator/optimizer workflows:
- You can treat **"Evaluator"** and **"Optimizer"** as separate **skills**, each with its own skill file containing:
  - Role description (e.g., "You are an evaluator who..." vs. "You are an optimizer who...").
  - Detailed instructions, specific priorities, and output formats.
  - Canonical examples showing how to handle conflicts (e.g., user asks for risky optimization; evaluator flags it; optimizer then proposes a safe alternative).[5]
- The **core system prompt** remains the single source of global safety and behavior rules; evaluator/optimizer skill prompts only specialize task-level behavior and must not contradict the core.[5]

Regarding conflicting priorities:
- Agent Skills emphasizes **modularity and isolation of instructions**: each skill is only loaded when needed.[5] This reduces cross-contamination between evaluator and optimizer objectives, since only the active role’s instructions are in the immediate context.
- Orchestration logic can determine which skill to invoke next based on structured outputs (e.g., evaluator’s `needs_revision: true` leading to an optimizer call), aligning with Anthropic’s broader emphasis on structured interfaces between agent components.[5]

This pattern effectively operationalizes prompt anatomy across multiple steps: a stable global prompt plus **per-skill prompt files** that encode role-specific priorities, enabling robust behavior even when those priorities diverge.[5]

-----

</details>

<details>
<summary>What are the trade-offs between LangGraph's Functional API and its Graph API for building iterative, stateful workflows like the evaluator-optimizer loop?</summary>

### Source [37]: https://docs.langchain.com/oss/python/langgraph/choosing-apis

Query: What are the trade-offs between LangGraph's Functional API and its Graph API for building iterative, stateful workflows like the evaluator-optimizer loop?

Answer: The "Choosing between Graph and Functional APIs" guide explains that both **Functional API** and **Graph API** share the same runtime and core capabilities (persistence, streaming, human-in-the-loop, memory), but package them in different paradigms for different trade-offs.[1]

For **iterative, stateful workflows** like evaluator–optimizer loops, the Graph API is recommended when you need:
- **Explicit state management**: a shared state object is passed through nodes so multiple steps (e.g., evaluator, optimizer, controller) can read and update common fields over many iterations.[1]
- **Complex workflow visualization**: the evaluator, optimizer, stopping criteria, error handlers, and side branches are visible as nodes and edges, aiding debugging and documentation.[1]
- **Conditional branching with multiple decision points**: edges can encode when to loop back to the optimizer, when to stop, or when to route to a fallback.[1]
- **Parallel execution paths that later merge**: e.g., evaluate with multiple metrics in parallel and then aggregate before the next optimization step.[1]
- **Team collaboration**: the explicit graph makes behavior easier to understand and modify across a team.[1]

The **Functional API** is recommended when you want:
- **Standard control flow (if/else, loops)** to express iterations in normal Python, rather than as graph edges.[1]
- **Function-scoped state**: you still get state and checkpoints, but you do not explicitly design a shared state schema and edge functions; this can be simpler for more linear or moderately branched loops.[1]
- **Minimal code changes and rapid prototyping**: existing procedural evaluator–optimizer code can be wrapped with `@entrypoint` and `@task` with little refactoring.[1]

The guide notes that if a workflow becomes **over‑engineered** in the Graph API for what is effectively a linear or simple loop, it can be refactored to the Functional API; conversely, if functional code grows many decision points and shared state needs, moving to Graph can improve clarity.[1] Both APIs can also be **combined**, e.g., run a Functional workflow as a node within a larger Graph-based coordinator.[1]

-----

-----

### Source [38]: https://blog.langchain.com/introducing-the-langgraph-functional-api/

Query: What are the trade-offs between LangGraph's Functional API and its Graph API for building iterative, stateful workflows like the evaluator-optimizer loop?

Answer: The Functional API introduction post presents the Functional API as complementary to the Graph API (StateGraph), emphasizing trade-offs rather than replacement.[2]

For workflows like evaluator–optimizer loops, it highlights several **Functional API** characteristics:
- **Control flow**: you express loops and branching with *standard Python constructs* (for/while, if/else, function calls) rather than designing nodes and edges.[2] This usually **reduces control‑flow boilerplate**, making iterative patterns natural in regular code.[2]
- **Decorators only**: `@entrypoint` and `@task` wrap normal functions, letting you adopt LangGraph features (state management, human‑in‑the‑loop, streaming, persistence) *without restructuring into a graph*.[2]
- **Dynamic execution**: execution flow is generated at runtime from normal code, not from a pre-declared graph structure.[2]

However, the post also outlines **limitations versus the Graph API**:
- **Checkpointing and time travel**: in the Functional API, a checkpoint is created per `entrypoint` execution, and `task`s update that checkpoint rather than creating new ones, so **time‑travel / step‑level replay is better supported in the Graph API**.[2] For deeply iterative evaluator–optimizer loops where you want to inspect or roll back to arbitrary steps, Graph offers finer-grained history.
- **Visualization**: the **Graph API supports visualization** of the workflow graph, which aids debugging and understanding complex branching; the Functional API **does not support visualization** because control flow is dynamic at runtime.[2]

The article reiterates that both APIs share key features (human‑in‑the‑loop, state management, persistence, streaming), and they can be **mixed** because they run on the same runtime.[2] For example, you could implement a local, procedural evaluator–optimizer loop with the Functional API, and embed it as a node inside a larger Graph-based orchestration that handles higher-level routing, multi-agent coordination, or monitoring.[2]

-----

-----

### Source [39]: https://changelog.langchain.com/announcements/functional-api-for-langgraph

Query: What are the trade-offs between LangGraph's Functional API and its Graph API for building iterative, stateful workflows like the evaluator-optimizer loop?

Answer: The LangGraph Functional API changelog entry describes why and when to use the Functional API, contrasting it implicitly with the Graph API.[3]

It states that the **Functional API** allows using LangGraph’s capabilities **without explicitly defining a graph**, which is significant for iterative workflows that are already written in a procedural style.[3] You can:
- Use it **"anywhere"** in an app, integrating LangGraph features into existing code without adopting graph syntax.[3]
- Rely on **seamless state management** to keep track of user inputs, workflow progress, and historical context across calls, which is essential for iterative evaluator–optimizer loops.[3]
- Leverage **human‑in‑the‑loop support** to pause workflows, collect feedback (e.g., human evaluation of an intermediate artifact), and resume dynamically.[3]
- Take advantage of **streaming-first** behavior to stream intermediate updates, which is useful when an iterative loop produces partial results or diagnostics over time.[3]

The entry emphasizes that **Graph + Functional API = best of both worlds**, explicitly recommending **mixing paradigms** to build robust AI systems.[3] From a trade-off perspective, this suggests:
- Use the **Graph API** where explicit graphs, visualization, and fine-grained control over state transitions are beneficial (complex, branching, multi-agent, or long‑running iterative loops).[3]
- Use the **Functional API** where you want to inject LangGraph’s persistence, streaming, and human‑in‑the‑loop capabilities into existing, iterative, function-based code with minimal disruption.[3]

Although the changelog does not detail Graph API internals, it positions the Functional API primarily as a **lower-friction, more ergonomic** way to gain the same core runtime powers that the Graph API exposes more explicitly through nodes, edges, and state.[3]

-----

-----

### Source [40]: https://docs.langchain.com/oss/python/langgraph/graph-api

Query: What are the trade-offs between LangGraph's Functional API and its Graph API for building iterative, stateful workflows like the evaluator-optimizer loop?

Answer: The Graph API overview explains that LangGraph models workflows as **graphs** of `State`, `Nodes`, and `Edges`, which is central to the trade-offs for iterative, stateful loops.[5]

Key Graph API properties relevant to an evaluator–optimizer loop:
- **State**: a shared data structure that carries the current snapshot of the application; it flows through the graph and is updated by nodes.[5] This makes **global, iterative state** for evaluator metrics, candidate solutions, and history explicit and structured.
- **Nodes**: functions that implement logic and side effects; they receive and update state.[5] Evaluator and optimizer can each be separate nodes operating on shared state.
- **Edges**: functions that decide the next node based on the current state; they can implement **conditional branches** or fixed transitions.[5] Loops (e.g., repeat optimization until a quality threshold is met) are represented by edges that route back to earlier nodes.

The Graph API is designed for **complex, looping workflows that evolve state over time**.[5] This directly matches iterative improvement patterns such as evaluator–optimizer, clarifying how control flow and state changes occur at each step.

The documentation also notes advantages like:
- Ability to **save intermediate state in checkpoints**, which can improve resilience and allow inspection of intermediate steps in long-running loops.[5]
- Support for different error-handling strategies (proactive vs reactive) embedded in the graph, which can be important for robust iterative workflows where some iterations may fail and need retry or alternate paths.[5]

While this page does not mention the Functional API, by contrast with the Functional API docs it underscores that the Graph API offers **declarative, explicit control over structure and state transitions**, lending itself to more complex or heavily branched iterative workflows compared with writing everything as dynamic control flow in regular functions.

-----

-----

### Source [41]: https://langchain-ai.github.io/langgraph/guides/

Query: What are the trade-offs between LangGraph's Functional API and its Graph API for building iterative, stateful workflows like the evaluator-optimizer loop?

Answer: The LangGraph guides index distinguishes between **Graph API** and **Functional API** guides, clarifying the conceptual split and implied trade-offs.[7]

It describes:
- **Graph API**: used to define workflows via a **graph paradigm**, where you explicitly model state, nodes, and edges.[7] This paradigm is naturally suited for workflows with complex branching, looping, and multi-step coordination, such as multi-agent systems and structured evaluator–optimizer cycles.
- **Functional API**: used to build workflows via a **functional paradigm**.[7] Here, you write normal functions and use decorators to integrate with LangGraph, leveraging standard control flow for loops and conditionals instead of an explicit graph.

Although this page is high-level, it suggests the following trade-offs for iterative, stateful workflows:
- The **graph paradigm** is preferable when you want *formal structure* and potentially visualization and analysis of the workflow as a graph, which is valuable for debugging and evolving complex evaluator–optimizer loops.
- The **functional paradigm** is preferable when you prioritize ergonomic coding and incremental adoption, expressing the iterative logic directly in code while still using LangGraph’s runtime features.

By linking to separate, dedicated guides for each API, the page emphasizes that each paradigm serves different development styles and complexity levels; both are first-class paths for building iterative, stateful workflows, and teams can choose based on whether they value explicit graph structure or familiar functional/procedural code for their evaluator–optimizer loops.[7]

-----

</details>

<details>
<summary>How can LangGraph's checkpointing feature be used to implement resilient and debuggable short-term memory in a multi-iteration AI workflow?</summary>

### Source [42]: https://docs.langchain.com/oss/python/langgraph/persistence

Query: How can LangGraph's checkpointing feature be used to implement resilient and debuggable short-term memory in a multi-iteration AI workflow?

Answer: LangGraph’s checkpointing and persistence system provides **resilient, debuggable short‑term memory** for multi‑iteration workflows by automatically capturing the graph state at each super‑step and associating it with a **thread**.[6]

A **checkpoint** is the serialized state of all channels at a particular execution step; LangGraph stores a checkpoint on every super‑step, so an agent’s working memory (messages, tool results, control state) is durably persisted as the workflow iterates.[6] Because these checkpoints are tied to a thread, subsequent invocations with the same `thread_id` can *resume from the latest checkpoint* instead of starting from scratch, giving resilience against failures or restarts and enabling short‑term conversational memory across turns.[6]

After execution, threads expose the recorded checkpoints, which enables:

- **Human‑in‑the‑loop**: inspecting or modifying intermediate state before resuming execution, effectively "editing" short‑term memory between iterations.[6]
- **Memory**: treating the thread’s checkpoints as the agent’s scratchpad for multi‑turn workflows, so each new iteration reads and writes to this evolving state rather than relying only on the model context window.[6]
- **Time travel**: selecting an earlier checkpoint in the same thread and resuming execution from that point, which is useful for debugging branching behaviors or replaying a run with different prompts/tools without redoing the whole workflow.[6]
- **Fault tolerance**: if a run crashes mid‑workflow, the next run can reload the last checkpoint for that thread and continue, preserving short‑term state (partial results, in‑progress tool outputs) instead of losing it.[6]

When using the LangGraph API, this checkpointing is **automatic**; you do not have to configure a checkpointer manually, but you can still query and control threads to implement robust, debuggable short‑term memory patterns.[6]

-----

-----

### Source [43]: https://reference.langchain.com/python/langgraph/checkpoints/

Query: How can LangGraph's checkpointing feature be used to implement resilient and debuggable short-term memory in a multi-iteration AI workflow?

Answer: The LangGraph checkpointing reference defines checkpoints as **state snapshots** used to persist graph execution across interactions, and exposes interfaces that make short‑term memory both resilient and debuggable.[2]

A `Checkpoint` holds an `id` (unique, monotonically increasing) and `channel_values`, a mapping from channel names to deserialized snapshot values, representing the full working memory at that step.[2] Because IDs are ordered, they can be used to **sort checkpoints chronologically**, enabling time‑based navigation of a workflow’s short‑term memory.[2]

The **checkpointer base class** provides core methods:

- `put` / `aput`: store a checkpoint with configuration and metadata, capturing the workflow’s transient state at that iteration.[2]
- `get` / `get_tuple`: fetch a specific checkpoint (optionally by thread and checkpoint ID) or the latest one, which is how a workflow **resumes from prior state instead of recomputing**.[2]
- `list` / `alist`: iterate over checkpoints matching filters (e.g., by thread, metadata, or before a given checkpoint), which supports **debugging and state inspection** across iterations.[2]
- `adelete_thread`: delete all checkpoints for a thread, effectively clearing that thread’s short‑term memory.[2]

Checkpoints are associated with a **config** that typically includes a `thread_id` and, once created, a `checkpoint_id` and `checkpoint_ns`, which together define a specific state version.[2] In practice, this allows a multi‑iteration workflow to:

- Maintain **per‑thread short‑term memory** (by always invoking with the same `thread_id`).[2]
- Inspect or replay intermediate states for debugging, using `list`/`get` to traverse or select historical checkpoints.[2]
- Implement branching variants of a run by resuming from a chosen checkpoint and continuing along a new path.[2]

The reference emphasizes that concrete backends (Postgres, AWS, etc.) implement these methods to provide durable, queryable short‑term memory across runs.[2]

-----

-----

### Source [44]: https://docs.langchain.com/oss/python/langgraph/use-time-travel

Query: How can LangGraph's checkpointing feature be used to implement resilient and debuggable short-term memory in a multi-iteration AI workflow?

Answer: LangGraph’s **time‑travel** feature shows how checkpoints can be used to build debuggable short‑term memory by allowing execution to resume from any prior saved state within a thread.[4]

The docs describe that, during execution with a checkpointer, LangGraph creates checkpoints for the graph state; you can later **enumerate these states** using `graph.get_state_history(config)`, which returns them in reverse chronological order.[4] Each state includes `next` (the next node(s) to run) and `config` with a `checkpoint_id`, allowing you to see exactly how the workflow’s short‑term memory evolved step by step.[4]

By iterating over this history, a developer can:

- Inspect intermediate states (messages, tool outputs, control information) for debugging multi‑iteration behavior.[4]
- Identify a particular checkpoint where they want to branch or replay.[4]

Once a target checkpoint is chosen, you can **resume execution from that prior checkpoint** rather than from the most recent one, effectively “time‑traveling” the workflow back to an earlier memory state and then continuing from there.[4] This is useful for:

- Re‑running from just before a failure, improving resilience without starting over.
- Comparing behavior when prompts, tools, or parameters are changed, while holding the earlier short‑term memory fixed.
- Implementing exploratory branches from a shared history, where different paths share the same initial checkpoints but diverge later.

Because time travel is built on top of checkpointing, it turns the stored short‑term memory into a **navigable timeline** that can be queried, inspected, and replayed, directly supporting debugging, analysis, and robust iteration in complex AI workflows.[4]

-----

-----

### Source [45]: https://pypi.org/project/langgraph-checkpoint-aws/

Query: How can LangGraph's checkpointing feature be used to implement resilient and debuggable short-term memory in a multi-iteration AI workflow?

Answer: The `langgraph-checkpoint-aws` package illustrates how checkpointing can be backed by AWS services to provide **durable, session‑scoped short‑term memory** for LangGraph workflows.[5]

It exposes an AWS‑based persistence solution with multiple storage backends, including integration with **Bedrock AgentCore Memory** and other AWS services, allowing LangGraph agents to store their checkpoints in managed, scalable infrastructure.[5]

A typical usage pattern involves:

- Creating a **session‑aware checkpointer** (`session_saver`) that manages sessions and their associated checkpoints.[5]
- Creating a new session via `session_saver.session_client.create_session().session_id`, which yields a `session_id` used analogously to a `thread_id` to scope short‑term memory.[5]
- Compiling a `StateGraph` with `checkpointer=session_saver` so that every super‑step of execution persists a checkpoint to AWS storage for that session.[5]
- Invoking the graph with `config = {"configurable": {"thread_id": session_id}}`, ensuring subsequent invocations reuse the same stored state (short‑term memory) instead of starting fresh.[5]

Because checkpoints are persisted in AWS, the workflow can survive process restarts, horizontal scaling, or transient failures while preserving its in‑progress state across multiple iterations.[5] This makes short‑term memory **resilient** in production environments where local in‑memory state would be unreliable.

Additionally, by querying or managing the stored checkpoints through the AWS‑backed saver, operators can inspect or clear session state, enabling **debugging and lifecycle control** over each workflow’s short‑term memory (for example, resetting a problematic session by deleting its checkpoints, or auditing intermediate states for analysis).[5]

-----

-----

### Source [46]: https://sparkco.ai/blog/mastering-langgraph-checkpointing-best-practices-for-2025

Query: How can LangGraph's checkpointing feature be used to implement resilient and debuggable short-term memory in a multi-iteration AI workflow?

Answer: This best‑practices article explains how to implement **production‑grade checkpointing** with LangGraph, showing how it underpins resilient and debuggable short‑term memory in multi‑iteration workflows.[3]

It demonstrates setting up a durable checkpointer using `PostgresSaver` from the `langgraph-checkpoint-postgres` package:[3]

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

DB_URI = "postgresql://user:pass@host:5432/langgraph?sslmode=require"

pool = ConnectionPool(conninfo=DB_URI, max_size=10)
with pool.connection() as conn:
    saver = PostgresSaver(conn)
    saver.setup()
```

`PostgresSaver` is then passed to LangGraph as the checkpointer, so each super‑step stores a checkpoint in Postgres, allowing the workflow’s evolving state (messages, intermediate results, control metadata) to be recovered later.[3] This persistence is a foundation for **short‑term memory** that survives crashes, restarts, or horizontal scaling.

The article emphasizes:

- Using a reliable database backend (Postgres) to guarantee **durability** and **state inspection**, so developers can query checkpoints for debugging and analytics.[3]
- Integrating checkpointing with **memory management and multi‑turn conversation handling**, where each user turn appends to the stored state rather than re‑sending the full context, making multi‑iteration workflows efficient and robust.[3]
- Supporting complex architectures (e.g., tool calling, MCP protocol, vector DB integrations) by treating checkpoints as the central source of truth for the agent’s working memory.[3]

By following these practices, teams can implement checkpointing that turns LangGraph into a **stateful, resumable workflow engine**, where short‑term memory is both resilient (backed by a database) and debuggable (inspectable and queryable at each step).[3]

-----

-----

### Source [47]: https://genai.acloudfan.com/155.agent-deeper-dive/1000.langgraph/30.checkpointing/

Query: How can LangGraph's checkpointing feature be used to implement resilient and debuggable short-term memory in a multi-iteration AI workflow?

Answer: This guide describes LangGraph checkpointing as a mechanism for saving **snapshots of a graph’s state during execution**, enabling fault tolerance, reproducibility, and time travel, all of which are essential for resilient, debuggable short‑term memory in multi‑iteration workflows.[7]

Checkpointing is presented as periodically capturing the **entire state** of the graph—its channels, intermediate results, and control flow markers—so that execution can later resume from that snapshot instead of from the beginning.[7] This means that if an error or interruption occurs, the workflow can restart from the last checkpoint with its short‑term memory intact, greatly improving **fault tolerance**.[7]

The guide emphasizes several benefits:

- **Reproducibility**: by restoring a prior checkpoint, developers can reproduce the exact conditions of a past run, which helps in debugging and verifying behavior across iterations.[7]
- **Time travel**: checkpoints form a timeline of state changes; developers can roll back to an earlier state to explore alternate paths or re‑run specific segments of the workflow without repeating the entire process.[7]
- **State inspection**: since checkpoints capture full graph state, they provide a clear window into what the system "knew" at each step—useful for understanding and tuning the agent’s short‑term reasoning.[7]

In the context of multi‑iteration AI workflows, this means that short‑term memory is not just transient in‑process data; it is **persisted, navigable, and recoverable**, making it practical to run longer, more complex interactions while still being able to debug, branch, and recover as needed.[7]

-----

</details>

<details>
<summary>What are best practices for designing a centralized, type-safe configuration system (e.g., using Pydantic and YAML) for a complex AI workflow with multiple nodes and models?</summary>

### Source [48]: https://www.sarahglasmacher.com/how-to-validate-config-yaml-pydantic/

Query: What are best practices for designing a centralized, type-safe configuration system (e.g., using Pydantic and YAML) for a complex AI workflow with multiple nodes and models?

Answer: This article illustrates best practices for using **Pydantic models as a type-safe schema over YAML configuration** in ML workflows.[1]

It recommends **small, focused Pydantic models** (e.g., `DataConfig`, `SplitConfig`, `TrainingConfig`, `ModelConfig`) each with **explicit types and sensible defaults**, which provides type safety and reduces boilerplate in the pipeline code.[1] Models use `Field()` to declare **defaults, constraints (e.g. `ge`, `le`), and descriptions**, ensuring invalid configuration (e.g., invalid split ratios) fails fast at load time instead of during training.[1]

A central `AppConfig` aggregates sub-configs and provides a **single entrypoint** `from_yaml()` (typically re-exposed as `load_config()`), which loads a YAML file via `yaml.safe_load`, optionally unwraps a top-level namespace key (e.g. `ml_blueprint:`), and calls `model_validate` to build the config object.[1] This pattern supports **centralized configuration** while keeping each concern modular.[1]

The YAML file mirrors the nested Pydantic structure (e.g., `data:`, `training:`, `model:` blocks), including nested structures like `split:` and model hyperparameters, making it straightforward to reason about and version-control.[1] The article emphasizes **keeping scripts thin**: a top-level script loads the config, instantiates components by passing **config objects** (not long parameter lists) into constructors, and then calls the main steps of the workflow.[1]

The recommended repo layout isolates configuration under a `config/` directory (e.g., `config/config.yaml`) and implementation under `src/`, with a dedicated `config.py` module containing Pydantic models and loaders.[1] This encourages **reproducibility and experiment management**: changing YAML alone changes run behavior (paths, splits, hyperparameters) without touching code.[1]

Key reusable checklist items: use Pydantic models with defaults and constraints; optionally namespace YAML under a top-level key; provide a single `load_config()` entrypoint; pass full config objects to nodes/models; and rely on validation errors for early failure instead of runtime bugs.[1]

-----

-----

### Source [49]: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

Query: What are best practices for designing a centralized, type-safe configuration system (e.g., using Pydantic and YAML) for a complex AI workflow with multiple nodes and models?

Answer: The **Pydantic Settings** documentation describes best practices for managing configuration and settings with **type-safe Pydantic models** that can load from multiple sources (environment, files, secrets). It formalizes the pattern of a centralized configuration class with strict validation.[5]

A `BaseSettings` subclass is used to declare configuration fields with standard Pydantic type annotations, defaults, and validators, providing **strong typing, parsing, and validation** for configuration values.[5] Settings models can be composed and nested, enabling a structured, hierarchical configuration model suitable for complex workflows with multiple components.[5]

The settings system supports **config sources** in priority order (e.g., environment variables, `.env` files, secret files, and custom sources), making it possible to merge multiple layers of configuration: for example, base defaults in code, overrides from a YAML file via a custom source, and final overrides from environment variables for deployment.[5] This layered approach is recommended for production systems where different environments (dev/stage/prod) or nodes in a workflow may apply different overrides while sharing a common schema.[5]

Settings classes provide consistent loading APIs (e.g., instantiation with no arguments or minimal options) while internal source configuration is done via `SettingsConfigDict` or configuration methods, keeping **configuration concerns centralized**.[5] The documentation highlights that settings models use the same validation machinery as regular Pydantic models, so complex types, nested models, and parsing (e.g. URLs, enums) are supported and should be preferred over ad-hoc parsing.[5]

For secret or sensitive values, Pydantic Settings encourages using environment variables or dedicated secrets files rather than committing them into general configuration files.[5] In the context of AI workflows, this implies separating **infrastructure and secret parameters** (keys, endpoints) into environment-based settings while leaving experiment-level parameters (node options, hyperparameters) in structured config files that can be parsed into settings models.[5]

-----

-----

### Source [50]: https://github.com/pydantic/pydantic-settings/issues/185

Query: What are best practices for designing a centralized, type-safe configuration system (e.g., using Pydantic and YAML) for a complex AI workflow with multiple nodes and models?

Answer: This GitHub issue discusses **best practices for loading configuration from YAML into Pydantic (including Pydantic Settings)** and clarifies trade-offs between direct model construction and using custom settings sources.[3]

A maintainer confirms that one valid pattern is to **read YAML with `yaml.safe_load` into a dict and construct the settings or config model directly** using `SettingsModel(**data)` (or a regular `BaseModel`), which "works without a problem" when you do not need advanced settings features.[3] This is a straightforward way to centralize configuration while still getting Pydantic’s type validation and parsing for simple scenarios.[3]

However, they note that this approach **forgoes some internal functionality of `pydantic-settings`**, such as its more advanced parsing of complex values and integration with its standard source system.[3] In contrast, creating a **custom settings source class** to read YAML enables using `pydantic-settings` as intended: combining YAML with other sources (environment, secrets) and preserving behavior like complex parsing and prioritized merging.[3]

The example of a settings model `class Settings(BaseSettings): foo: Dict[int, str]` is mentioned to illustrate that complex types may benefit from the full settings machinery.[3] For a complex AI workflow with multiple nodes and models, this implies that if configuration must be layered, overridden, or combined with env-based deployment details, a **custom YAML settings source** attached to a central `BaseSettings` class is the more robust, extensible practice; direct `SettingsModel(**data)` remains suitable for simpler, file-only setups where centralization and type safety are the main goals.[3]

-----

-----

### Source [51]: https://ai.ragv.in/posts/sane-configs-with-pydantic-settings/

Query: What are best practices for designing a centralized, type-safe configuration system (e.g., using Pydantic and YAML) for a complex AI workflow with multiple nodes and models?

Answer: This article presents **practical patterns for managing configuration with Pydantic and `pydantic-settings`**, emphasizing type safety, layered sources, and clear separation between static config files and environment overrides.[4]

It shows a basic pattern where a Pydantic model (e.g., `DatabaseConfig`) is used to **parse a YAML configuration file** via `yaml.safe_load` and `DatabaseConfig(**yaml_data)`, with `ValidationError` handling to fail early on invalid configuration.[4] This illustrates how a centralized, type-safe configuration schema can be enforced over YAML files.[4]

The article then integrates this pattern with `pydantic-settings` by defining a settings class that can load from both **environment variables and a YAML file**, with **environment variables taking precedence** over YAML values.[4] This is recommended as a flexible default: project-wide defaults live in YAML (checked into version control), while deployment- or node-specific overrides are done via environment variables without changing the file.[4]

The author notes that this approach generalizes to other file formats (TOML, etc.) as long as they can be parsed into dictionaries, reinforcing that the key practice is to **express the configuration schema in Pydantic and treat external representations as interchangeable sources**.[4] In a complex AI workflow, this pattern supports centralization (one schema, multiple sources) and type safety, while allowing per-environment specialization.

The post also underscores that `pydantic-settings` is well-suited for keeping configurations "sane" by avoiding scattered constants and magic values; instead, a single settings model (potentially composed of nested models) encapsulates parameters like database connections, model hyperparameters, or service endpoints, which can then be injected into components as needed.[4]

-----

-----

### Source [52]: https://proudlynerd.vidiemme.it/mastering-python-project-configuration-with-pydantic-f924a0803dd4

Query: What are best practices for designing a centralized, type-safe configuration system (e.g., using Pydantic and YAML) for a complex AI workflow with multiple nodes and models?

Answer: This article covers **best practices for Python project configuration using Pydantic**, focusing on structured, type-safe configuration models and separation of concerns between configuration sources and application logic.[6]

It recommends modeling configuration as a dedicated Pydantic class (or classes) with **clear field types, defaults, and validation rules**, rather than scattering constants or reading raw dictionaries throughout the codebase.[6] For complex projects, configuration can be organized into multiple nested models (e.g., database, API, application behavior), promoting modularity and clarity.[6]

The article emphasizes using **environment variables for sensitive values** (such as API keys, passwords, tokens) instead of storing them in YAML or other configuration files, which might be committed to version control.[6] Non-sensitive operational parameters (like feature flags, timeouts, or model hyperparameters) can safely live in versioned configuration files, still parsed into Pydantic models for type safety.[6]

It also highlights that having a **central entrypoint for configuration loading** improves maintainability: the rest of the codebase depends only on the configuration model interface, not on how configuration is sourced (YAML, env, `.env`, etc.).[6] This allows configuration loading strategies to evolve without impacting nodes or models in the workflow.[6]

Among the best practices listed are: keeping configuration definitions close to the application domain; avoiding duplication of configuration logic; leveraging Pydantic’s validation to catch configuration errors at startup; and documenting configuration options via field descriptions or docstrings so that team members can understand which options are available and how they affect the system’s behavior.[6] These practices map well to complex AI workflows, where many interdependent parameters and nodes benefit from a single, validated configuration schema.[6]

-----

</details>

<details>
<summary>How can clean architecture principles, such as separating domain, application, and infrastructure layers, be applied when building LLM workflows with frameworks like LangGraph?</summary>

### Source [61]: https://forum.langchain.com/t/are-you-using-clean-architecture-principles-with-around-your-agents-or-not/1876

Query: How can clean architecture principles, such as separating domain, application, and infrastructure layers, be applied when building LLM workflows with frameworks like LangGraph?

Answer: The discussion explains how **Clean Architecture** can be combined with **LangGraph-based LLM workflows** by aligning graph orchestration with layered boundaries.[1]

It states that Clean Architecture promotes **separation of concerns** between **business logic**, **infrastructure**, and **interfaces**, and that LangGraph’s graph-based orchestration (nodes, edges, state) can be encapsulated inside the **domain and application layers** rather than spread across the whole system.[1] The idea is to keep **business rules and decision logic** independent of concrete frameworks, while treating LangGraph as part of the orchestration/infrastructure boundary.[1]

The post highlights that LangGraph’s **nodes and edges** can be designed as **discrete, testable units** of logic, which aligns with Clean Architecture’s emphasis on **testability** and isolating core logic from external dependencies.[1] Nodes can be implemented as pure functions or small components that operate on state, making them easier to test independently of the LLM provider or other infrastructure.[1]

It argues that LangGraph’s support for **complex, stateful, and long‑running workflows** benefits from Clean Architecture’s layered structure, which organizes this complexity into **well-defined boundaries** and improves **scalability and flexibility** when systems grow to multiple agents and workflows.[1]

The discussion also stresses **decoupling from frameworks**: Clean Architecture encourages minimizing direct dependency on concrete frameworks, so LangGraph is treated as an orchestration mechanism, not as the place where domain rules live.[1] This enables the **domain layer** to remain framework‑agnostic, with LangGraph graphs acting as the application layer that coordinates use cases, and tools/adapters implementing infrastructure concerns.[1]

Finally, the post notes trade‑offs: adding Clean Architecture around LangGraph may introduce **extra complexity, boilerplate, and indirection**, and the graph‑ and state‑driven paradigm may not always map naturally to traditional layered designs, which can lead to awkward or over‑engineered solutions in simple workflows.[1]

-----

-----

### Source [62]: https://aws.amazon.com/blogs/machine-learning/build-multi-agent-systems-with-langgraph-and-amazon-bedrock/

Query: How can clean architecture principles, such as separating domain, application, and infrastructure layers, be applied when building LLM workflows with frameworks like LangGraph?

Answer: The article describes a **multi‑agent system** built with **LangGraph** and **Amazon Bedrock**, and it implicitly demonstrates a Clean‑Architecture‑style separation between workflow orchestration and lower‑level infrastructure.[3]

It explains that LangGraph **models agent workflows as graphs**, using three main building blocks: **State**, a shared data structure representing the current snapshot of the application; **Nodes**, Python functions that encode the logic of agents; and **Edges**, Python functions that determine which node to execute next, either via conditional branching or fixed transitions.[3] This structure maps naturally to an **application layer** that coordinates use cases using state and transitions.

LangGraph also implements a **central persistence layer** that underpins cross‑cutting capabilities like **memory** (persisting arbitrary aspects of state across interactions) and **human‑in‑the‑loop** features (checkpointing state so execution can be interrupted, validated, and resumed).[3] These capabilities can be viewed as **infrastructure services** that are consumed by higher‑level workflow logic rather than embedded inside domain rules.[3]

The post describes a code organization where each agent has a **graph.py** file that defines the **workflow and decision‑making logic** via the LangGraph state machine and configures communication flow among components such as a Flight Agent, Hotel Agent, and Supervisor Agent.[3] Separate **tools** implement the concrete operations like search, booking, and modification.[3] The article explicitly notes that this separation between **graph (workflow)** and **tools (implementation)** "allows for a clean architecture where the decision‑making process is separate from the actual execution of tasks."[3]

In Clean Architecture terms, the **graphs** correspond to the **application layer** orchestrating use cases and routing between agents, while the **tools and Bedrock integrations** function as **infrastructure/adapters** that implement external concerns behind the graph‑defined workflows.[3]

-----

-----

### Source [63]: https://github.com/langchain-ai/langgraph

Query: How can clean architecture principles, such as separating domain, application, and infrastructure layers, be applied when building LLM workflows with frameworks like LangGraph?

Answer: The LangGraph repository describes the framework as a **low‑level orchestration framework** for building, managing, and deploying **long‑running, stateful agents**, which aligns it naturally with the **infrastructure/application boundary** in a Clean Architecture design.[4]

It states that LangGraph provides **low-level supporting infrastructure** for any long‑running, stateful workflow or agent and explicitly notes that it **does not abstract prompts or architecture**.[4] Instead, it focuses on core capabilities such as defining state machines as graphs, managing state, and providing persistence and control over execution.[4] This means that **domain and application architecture remain the responsibility of the user**, allowing them to enforce clean separation of domain rules from orchestration infrastructure.

By keeping architectural decisions out of the framework, LangGraph can be treated as an **orchestration layer**: developers implement **nodes** as functions or callables that encapsulate logic, while the framework manages how these nodes are connected and executed within a graph.[4] In a Clean Architecture mapping, these graphs and nodes live predominantly in the **application layer**, coordinating domain services and use cases, while **integration with LLM providers, tools, and external systems** can be placed in separate adapter/infrastructure modules invoked from within nodes.

Because LangGraph is **framework‑agnostic with respect to prompts and higher‑level patterns**, it supports building systems where **domain logic** is implemented independently, and LangGraph is only used to orchestrate when and how that logic is invoked, preserving **layered boundaries** between domain, application, and infrastructure concerns.[4]

-----

-----

### Source [64]: https://dev.to/dev_tips/build-your-first-ai-agent-with-langgraph-without-losing-your-sanity-3b31

Query: How can clean architecture principles, such as separating domain, application, and infrastructure layers, be applied when building LLM workflows with frameworks like LangGraph?

Answer: The tutorial presents LangGraph through the lens of **structured agent design**, illustrating patterns that map well to a Clean Architecture layering of **domain**, **application**, and **infrastructure**.[2]

It emphasizes that LangGraph lets you design AI agents as **graphs of nodes and edges**, where each **node** is a **step of logic** and **edges** define the flow between steps.[2] This approach promotes **modularity** and makes workflows **predictable, testable, and extendable**, which aligns with having a clear **application layer** that coordinates domain logic rather than scattering it through ad‑hoc prompt chains.[2]

The article shows how to define a **StateGraph** with a custom **AgentState** type, then add nodes such as "greet", "ask", and "answer", and connect them with edges before compiling the graph into an executable app.[2] Each node is a Python function that reads and updates the shared state.[2] In a Clean Architecture mapping, these node functions can call into **domain services** or entities, while the graph and state structure form part of the **application layer** that orchestrates the conversation or workflow.

The tutorial argues that LangGraph is **built for complexity**, making it easier to handle requirements like **memory across steps**, **dynamic decision‑making**, **error handling and retries**, and **branching workflows**.[2] These concerns can be handled within the graph orchestration while keeping external dependencies (LLMs, tools, databases) behind clearly defined interfaces, supporting a separation between **workflow coordination** and **infrastructure**.

It also notes that a clear **node/edge/state design** prevents the agent from becoming a "spaghetti monster" and encourages developers to **“think structurally”** about their agents.[2] This structural thinking is compatible with Clean Architecture, where LLM prompts and tooling can be treated as infrastructure details, and the graph defines high‑level use cases and their sequencing.[2]

-----

</details>

<details>
<summary>What are best practices for implementing polymorphic input handling in Python-based AI systems, where a single component processes different but related data types?</summary>

### Source [65]: https://www.geeksforgeeks.org/python/polymorphism-in-python/

Query: What are best practices for implementing polymorphic input handling in Python-based AI systems, where a single component processes different but related data types?

Answer: This source explains that in Python **polymorphism** means the *same function, method, or operator* behaves differently depending on the object it receives, and that this is resolved **at runtime** because Python is dynamically typed.[1]

It distinguishes:
- **Runtime polymorphism via method overriding**: subclasses implement the same method name (e.g., `sound()` in `Dog`, `Cat`, `Animal`), and the correct method is chosen based on the actual object in a loop, illustrating a common pattern where a single processing component calls the same method on different related types.[1]
- **Function polymorphism with duck typing**: functions accept different object types as long as they provide the required methods/attributes. An example is a function `perform_task()` that calls `.use()` on any object that implements it, without checking concrete types, which is a best practice for polymorphic input handling.[1]
- **Built‑in polymorphic functions**: functions like `len()` and `max()` work with multiple container or scalar types by delegating to the object protocol (e.g., `__len__`), showing that robust polymorphic input handling relies on **standard protocols and special methods** to unify behavior across types.[1]
- **Operator overloading**: operators such as `+` behave differently for integers, strings, and lists, via type‑specific implementations, emphasizing that implementing well‑defined operator/dunder methods on custom classes enables them to participate safely in polymorphic operations.[1]

The article notes that Python does not have true compile‑time overloading; instead, polymorphic behavior for a single entry point is achieved with **default/variable arguments** and **runtime dispatch based on object behavior**, which is the recommended approach for flexible, reusable components that must process related but different data types.[1]

-----

-----

### Source [66]: https://www.thedigitalcatonline.com/blog/2014/08/21/python-3-oop-part-4-polymorphism/

Query: What are best practices for implementing polymorphic input handling in Python-based AI systems, where a single component processes different but related data types?

Answer: This source characterizes Python polymorphism as being **based on delegation**: you "ask the data itself to perform the operation" instead of writing separate functions for each type.[2]

It contrasts two strategies:
- Procedural style (e.g., C++): write separate `sum_int`, `sum_float`, etc.
- Python style: write one `sum(a, b)` and rely on `a` to know how to add `b`, since `a + b` is implemented as `a.__add__(b)`.[2]

For polymorphic input handling, the guidance is:
- Design functions that **do not check explicit types**, but instead rely on the objects to implement the expected protocol (methods like `__add__`, `__len__`, etc.).[2]
- The `len()` example shows that `len(x)` simply calls `x.__len__()`. Lists, strings, and dicts work because they provide `__len__`, while `int` does not and raises `TypeError`/`AttributeError`, demonstrating how a generic component can operate on many related types by counting on a shared protocol rather than inspecting types.[2]

The article links this to the Python **EAFP (Easier to Ask Forgiveness than Permission)** principle. Instead of guarding with `hasattr` or type checks, code should:
- Attempt the polymorphic operation directly (e.g., `someobj.open()`), and
- Catch `AttributeError` or other relevant exceptions if the protocol is not supported.[2]

These patterns are directly applicable to Python-based AI components that must handle multiple related data types: define a clear protocol (set of methods/attributes), let each data type implement it, call the protocol methods directly, and use exception handling rather than pre‑emptive type inspection to manage unsupported types.[2]

-----

-----

### Source [67]: https://www.w3schools.com/python/python_polymorphism.asp

Query: What are best practices for implementing polymorphic input handling in Python-based AI systems, where a single component processes different but related data types?

Answer: This source defines **polymorphism in Python** as the use of a single interface (such as a function or method name) that works with multiple data types.[3]

Key patterns relevant to designing polymorphic input handlers include:
- **Class method polymorphism**: different classes (e.g., `Car`, `Boat`, etc.) can define the **same method names** (such as `move()`), and client code can call `obj.move()` on a collection of heterogeneous objects. This demonstrates a best practice where a processing component uses a uniform method name across related types instead of branching on type.[3]
- **Function polymorphism via built‑ins**: the built‑in `len()` function works uniformly on **strings, tuples, and dictionaries**, returning length, item count, or key/value pair count respectively.[3]

This illustrates two important guidelines:
- Expose a consistent, type‑agnostic **API surface** (e.g., all input objects implement the same method names or special methods).
- Implement type‑specific behavior **inside the types themselves**, not in the central processing component; the handler only relies on the common interface like `len(x)` or `x.move()` and doesn’t need to know the concrete type.[3]

By modeling AI data structures (e.g., different feature representations or dataset wrappers) as classes that share common method names, a single pipeline component can call those methods in a polymorphic way without explicit type checks.[3]

-----

-----

### Source [68]: https://labex.io/tutorials/python-how-to-implement-polymorphic-constructors-452346

Query: What are best practices for implementing polymorphic input handling in Python-based AI systems, where a single component processes different but related data types?

Answer: This source focuses on **polymorphic constructors** in Python and provides best practices for handling multiple input shapes or types when constructing objects, which is relevant to AI components that must accept varied but related initialization data.[6]

It notes that Python does not support classic compile‑time constructor overloading; instead, a single `__init__` handles all cases, typically via:
- Optional parameters and default values.
- `*args` and `**kwargs` to flexibly capture different input forms.[6]

The **best practices** outlined include:
- **Keep constructor logic clean and predictable**: avoid excessive branching; centralize common initialization and delegate complex or format‑specific logic to helper methods. This helps maintain clarity when a single component supports polymorphic initialization from different data structures.[6]
- **Use type hints for better readability**: annotate parameters (including unions of accepted types) so users and tools understand the polymorphic input contract, which is important for large AI systems where components accept multiple related data types.[6]
- **Handle edge cases and invalid inputs**: explicitly validate constructor arguments and raise clear exceptions when supplied data does not conform to any supported form, ensuring robust behavior when polymorphism is used at object creation time.[6]
- **Prefer composition over overly complex constructors**: if many significantly different initialization paths emerge, compose with helper objects or factory functions instead of encoding all variations into a monolithic polymorphic constructor.[6]

These guidelines help design AI data or model wrapper classes whose constructors can ingest different but related inputs (e.g., paths, in‑memory arrays, or config objects) while remaining maintainable and predictable.[6]

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>Build multi-agent systems with LangGraph and Amazon Bedrock</summary>

# Build multi-agent systems with LangGraph and Amazon Bedrock

Large language models (LLMs) have raised the bar for human-computer interaction where the expectation from users is that they can communicate with their applications through natural language. Beyond simple language understanding, real-world applications require managing complex workflows, connecting to external data, and coordinating multiple AI capabilities. Imagine scheduling a doctor’s appointment where an AI agent checks your calendar, accesses your provider’s system, verifies insurance, and confirms everything in one go—no more app-switching or hold times. In these real-world scenarios, agents can be a game changer, delivering more customized generative AI applications.

LLM agents serve as decision-making systems for application control flow. However, these systems face several operational challenges during scaling and development. The primary issues include tool selection inefficiency, where agents with access to numerous tools struggle with optimal tool selection and sequencing, context management limitations that prevent single agents from effectively managing increasingly complex contextual information, and specialization requirements as complex applications demand diverse expertise areas such as planning, research, and analysis. The solution lies in implementing a multi-agent architecture, which involves decomposing the main system into smaller, specialized agents that operate independently. Implementation options range from basic prompt-LLM combinations to sophisticated [ReAct](https://arxiv.org/abs/2210.03629) (Reasoning and Acting) agents, allowing for more efficient task distribution and specialized handling of different application components. This modular approach enhances system manageability and allows for better scaling of LLM-based applications while maintaining functional efficiency through specialized components.

This post demonstrates how to integrate open-source multi-agent framework, [LangGraph](https://www.langchain.com/langgraph), with [Amazon Bedrock](https://aws.amazon.com/bedrock/). It explains how to use LangGraph and Amazon Bedrock to build powerful, interactive multi-agent applications that use graph-based orchestration.

## Challenges with multi-agent systems

In a single-agent system, planning involves the LLM agent breaking down tasks into a sequence of small tasks, whereas a multi-agent system must have workflow management involving task distribution across multiple agents. Unlike single-agent environments, multi-agent systems require a coordination mechanism where each agent must maintain alignment with others while contributing to the overall objective. This introduces unique challenges in managing inter-agent dependencies, resource allocation, and synchronization, necessitating robust frameworks that maintain system-wide consistency while optimizing performance.

Memory management in AI systems differs between single-agent and multi-agent architectures. Single-agent systems use a three-tier structure: short-term conversational memory, long-term historical storage, and external data sources like Retrieval Augmented Generation (RAG). Multi-agent systems require more advanced frameworks to manage contextual data, track interactions, and synchronize historical records across agents. These systems must handle real-time interactions, context synchronization, and efficient data retrieval, necessitating careful design of memory hierarchies, access patterns, and inter-agent sharing.

Agent frameworks are essential for multi-agent systems because they provide the infrastructure for coordinating autonomous agents, managing communication and resources, and orchestrating workflows. Agent frameworks alleviate the need to build these complex components from scratch.

[LangGraph](https://www.langchain.com/langgraph), part of [LangChain](https://www.langchain.com/), orchestrates agentic workflows through a graph-based architecture that handles complex processes and maintains context across agent interactions. It uses supervisory control patterns and memory systems for coordination.

[LangGraph Studio](https://studio.langchain.com/) enhances development with graph visualization, execution monitoring, and runtime debugging capabilities. The integration of LangGraph with Amazon Bedrock empowers you to take advantage of the strengths of multiple agents seamlessly, fostering a collaborative environment that enhances the efficiency and effectiveness of LLM-based systems.

## Understanding LangGraph and LangGraph Studio

LangGraph implements state machines and directed graphs for multi-agent orchestration. The framework provides fine-grained control over both the flow and state of your agent applications. LangGraph models agent workflows as graphs. You define the behavior of your agents using three key components:

- **State** – A shared data structure that represents the current snapshot of your application.
- **Nodes** – Python functions that encode the logic of your agents.
- **Edges** – Python functions that determine which Node to execute next based on the current state. They can be conditional branches or fixed transitions.

LangGraph implements a central persistence layer, enabling features that are common to most agent architectures, including:

- **Memory** – LangGraph persists arbitrary aspects of your application’s state, supporting memory of conversations and other updates within and across user interactions.
- **Human-in-the-loop** – Because state is checkpointed, execution can be interrupted and resumed, allowing for decisions, validation, and corrections at key stages through human input.

LangGraph Studio is an integrated development environment (IDE) specifically designed for AI agent development. It provides developers with powerful tools for visualization, real-time interaction, and debugging capabilities. The key features of LangGraph Studio are:

- **Visual agent graphs** – The IDE’s visualization tools allow developers to represent agent flows as intuitive graphic wheels, making it straightforward to understand and modify complex system architectures.
- **Real-time debugging** – The ability to interact with agents in real time and modify responses mid-execution creates a more dynamic development experience.
- **Stateful architecture** – Support for stateful and adaptive agents within a graph-based architecture enables more sophisticated behaviors and interactions.

The following screenshot shows the nodes, edges, and state of a typical LangGraph agent workflow as viewed in LangGraph Studio.

[https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic1.png](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic1.png)

_Figure 1: LangGraph Studio UI_

In the preceding example, the state begins with `__start__` and ends with `__end__`. The nodes for invoking the model and tools are defined by you and the edges tell you which paths can be followed by the workflow.

LangGraph Studio is available as a [desktop application](https://studio.langchain.com/) for MacOS users. Alternatively, you can [run a local in-memory development server](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/) that can be used to connect a local LangGraph application with a web version of the studio.

## Solution overview

This example demonstrates the supervisor agentic pattern, where a supervisor agent coordinates multiple specialized agents. Each agent maintains its own scratchpad while the supervisor orchestrates communication and delegates tasks based on agent capabilities. This distributed approach improves efficiency by allowing agents to focus on specific tasks while enabling parallel processing and system scalability.

Let’s walk through an example with the following user query: “Suggest a travel destination and search flight and hotel for me. I want to travel on 15-March-2025 for 5 days.” The workflow consists of the following steps:

1. The Supervisor Agent receives the initial query and breaks it down into sequential tasks:
1. Destination recommendation required.
2. Flight search needed for March 15, 2025.
3. Hotel booking required for 5 days.
2. The Destination Agent begins its work by accessing the user’s stored profile. It searches its historical database, analyzing patterns from similar user profiles to recommend the destination. Then it passes the destination back to the Supervisor Agent.
3. The Supervisor Agent forwards the chosen destination to the Flight Agent, which searches available flights for the given date.
4. The Supervisor Agent activates the Hotel Agent, which searches for hotels in the destination city.
5. The Supervisor Agent compiles the recommendations into a comprehensive travel plan, presenting the user with a complete itinerary including destination rationale, flight options, and hotel suggestions.

The following figure shows a multi-agent workflow of how these agents connect to each other and which tools are involved with each agent.

[https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic2.png](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic2.png)_Figure 2: Multi-agent workflow_

## Prerequisites

You will need the following prerequisites before you can proceed with this solution. For this post, we use the `us-west-2` AWS Region. For details on available Regions, see [Amazon Bedrock endpoints and quotas](https://docs.aws.amazon.com/general/latest/gr/bedrock.html).

- A valid AWS account.
- An [AWS Identity and Access Management](https://aws.amazon.com/iam/) (IAM) role in the account that has sufficient permissions to create the necessary resources.
- Access to Anthropic’s Claude 3 Sonnet and Claude 3.5 Sonnet in Amazon Bedrock. For instructions, see [Access Amazon Bedrock foundation models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html).
- A LangGraph application up and running locally. For instructions, see [Quickstart: Launch Local LangGraph Server](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/#quickstart-launch-local-langgraph-server).

## Core components

Each agent is structured with two primary components:

- [graph.py](https://github.com/aws-samples/aim323_build_agents_with_bedrock_oss/blob/main/bedrock-multi-agent-langgraph-studio/src/flight_agent/graph.py) – This script defines the agent’s workflow and decision-making logic. It implements the LangGraph state machine for managing agent behavior and configures the communication flow between different components. For example:

  - The Flight Agent’s graph manages the flow between chat and tool operations.
  - The Hotel Agent’s graph handles conditional routing between search, booking, and modification operations.
  - The Supervisor Agent’s graph orchestrates the overall multi-agent workflow.
- [tools.py](https://github.com/aws-samples/aim323_build_agents_with_bedrock_oss/blob/main/bedrock-multi-agent-langgraph-studio/src/hotel_agent/tools.py) – This script contains the concrete implementations of agent capabilities. It implements the business logic for each operation and handles data access and manipulation. It provides specific functionalities like:

  - Flight tools: `search_flights`, `book_flights`, `change_flight_booking`, `cancel_flight_booking`.
  - Hotel tools: `suggest_hotels`, `book_hotels`, `change_hotel_booking`, `cancel_hotel_booking`.

This separation between graph (workflow) and tools (implementation) allows for a clean architecture where the decision-making process is separate from the actual execution of tasks. The agents communicate through a state-based graph system implemented using LangGraph, where the Supervisor Agent directs the flow of information and tasks between the specialized agents.

To set up Amazon Bedrock with LangGraph, refer to the following [GitHub repo](https://github.com/aws-samples/aim323_build_agents_with_bedrock_oss/blob/main/bedrock-multi-agent-langgraph-studio/README.md). The high-level steps are as follows:

1. Install the required packages:

```bash
pip install boto3 langchain-aws
```

These packages are essential for AWS Bedrock integration:

- `boto`: AWS SDK for Python, handles AWS service communication
- `langchain-aws`: Provides LangChain integrations for AWS services

2. Import the modules:

```python
from langchain_aws import ChatBedrockConverse
from langchain_aws import ChatBedrock
```

3. Create an LLM object:

```python
bedrock_client = boto3.client("bedrock-runtime", region_name="<region_name>")
llm = ChatBedrockConverse(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        temperature=0,
        max_tokens=None,
        client=bedrock_client,
        # other params...
    )
```

## LangGraph Studio configuration

This project uses a [langgraph.json](https://github.com/aws-samples/aim323_build_agents_with_bedrock_oss/blob/main/bedrock-multi-agent-langgraph-studio/langgraph.json) configuration file to define the application structure and dependencies. This file is essential for LangGraph Studio to understand how to run and visualize your agent graphs.

```json
{
"dependencies": [\
"boto3>=1.35.87",\
"langchain-aws>=0.2.10",\
"."\
],
"graphs": {
"supervisor": "./src/supervisor_agent/graph.py:graph",
"flight": "./src/flight_agent/graph.py:graph",
"hotel": "./src/hotel_agent/graph.py:graph"
},
"env": "./.env"
}
```

LangGraph Studio uses this file to build and visualize the agent workflows, allowing you to monitor and debug the multi-agent interactions in real time.

## Testing and debugging

You’re now ready to test the multi-agent travel assistant. You can start the graph using the `langgraph dev` command. It will start the LangGraph API server in development mode with hot reloading and debugging capabilities. As shown in the following screenshot, the interface provides a straightforward way to select which graph you want to test through the dropdown menu at the top left. The **Manage Configuration** button at the bottom lets you set up specific testing parameters before you begin. This development environment provides everything you need to thoroughly test and debug your multi-agent system with real-time feedback and monitoring capabilities.

[https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic3.png](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic3.png)_Figure 3: LangGraph studio with Destination Agent recommendation_

LangGraph Studio offers flexible configuration management through its intuitive interface. As shown in the following screenshot, you can create and manage multiple configuration versions (v1, v2, v3) for your graph execution. For example, in this scenario, we want to use `user_id` to fetch historic use information. This versioning system makes it simple to track and switch between different test configurations while debugging your multi-agent system.

[https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic4.png](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic4.png)_Figure 4: Runnable configuration details_

In the preceding example, we set up the `user_id` that tools can use to retrieve history or other details.

Let’s test the Planner Agent. This agent has the `compare_and_recommend_destination` tool, which can check past travel data and recommend travel destinations based on the user profile. We use `user_id` in the configuration so that can it be used by the tool.

LangGraph has concept of checkpoint memory that is managed using a thread. The following screenshot shows that you can quickly manage threads in LangGraph Studio.

[https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic5.png](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic5.png)_Figure 5: View graph state in the thread_

In this example, `destination_agent` is using a tool; you can also check the tool’s output. Similarly, you can test `flight_agent` and `hotel_agent` to verify each agent.

When all the agents are working well, you’re ready to test the full workflow. You can evaluate the state a verify input and output of each agent.

The following screenshot shows the full view of the Supervisor Agent with its sub-agents.

[https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/03/ml-18369-image6.png](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/04/02/blog-18269-Pic6.png)_Figure 6: Supervisor Agent with complete workflow_

## Considerations

Multi-agent architectures must consider agent coordination, state management, communication, output consolidation, and guardrails, maintaining processing context, error handling, and orchestration. Graph-based architectures offer significant advantages over linear pipelines, enabling complex workflows with nonlinear communication patterns and clearer system visualization. These structures allow for dynamic pathways and adaptive communication, ideal for large-scale deployments with simultaneous agent interactions. They excel in parallel processing and resource allocation but require sophisticated setup and might demand higher computational resources. Implementing these systems necessitates careful planning of system topology, robust monitoring, and well-designed fallback mechanisms for failed interactions.

When implementing multi-agent architectures in your organization, it’s crucial to align with your company’s established generative AI operations and governance frameworks. Prior to deployment, verify alignment with your organization’s AI safety protocols, data handling policies, and model deployment guidelines. Although this architectural pattern offers significant benefits, its implementation should be tailored to fit within your organization’s specific AI governance structure and risk management frameworks.

## Clean up

Delete any IAM roles and policies created specifically for this post. Delete the local copy of this post’s code. If you no longer need access to an Amazon Bedrock FM, you can remove access from it. For instructions, see [Add or remove access to Amazon Bedrock foundation models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html)

## Conclusion

The integration of LangGraph with Amazon Bedrock significantly advances multi-agent system development by providing a robust framework for sophisticated AI applications. This combination uses LangGraph’s orchestration capabilities and FMs in Amazon Bedrock to create scalable, efficient systems. It addresses challenges in multi-agent architectures through state management, agent coordination, and workflow orchestration, offering features like memory management, error handling, and human-in-the-loop capabilities. LangGraph Studio’s visualization and debugging tools enable efficient design and maintenance of complex agent interactions. This integration offers a powerful foundation for next-generation multi-agent systems, providing effective workflow handling, context maintenance, reliable results, and optimal resource utilization.

For the example code and demonstration discussed in this post, refer to the accompanying [GitHub repository](https://github.com/aws-samples/aim323_build_agents_with_bedrock_oss/tree/main/bedrock-multi-agent-langgraph-studio). You can also refer to the following [GitHub repo for Amazon Bedrock multi-agent collaboration code samples](https://github.com/awslabs/amazon-bedrock-agent-samples).

</details>

<details>
<summary>1. Why Use Config Files (in ML)?</summary>

## 1. Why Use Config Files (in ML)?

In ML projects, you always want to keep data and code separate. The code trains the model, but depending on your data, you might want to have different hyperparameters and settings for that model. Instead of hardcoding the paths and hyperparameters, we externalize them into a YAML file. This gives you:

-   **Reproducibility** → The config is included in your git commit, so you can re-run the same experiment months later.
-   **Safety** → Pydantic models can validate types and ranges, so you don’t try to train with `train_size=1.5` by mistake.
-   **Team-friendliness** → collaborators can tweak `config.yaml` without touching your code.

**Rule of Thumb:** Put everything related to your data in the config (paths, split ratios, target column). This includes your feature selection (missing value strategy, which columns to include or drop, etc.) and then also adds the model hyperparameters.

## 2. Pydantic Basics (Quickstart)

### Without Pydantic and validation it would go like this

So, let's assume you have your `.yaml` file ready with all your parameters. With a library like `pyyaml` you can read this into a dictionary very easily:

```yaml
train_size: 0.6
target_column: 'Premium Amount'
start_date: 1990-05-22
```

```python
import yaml

with open('config.yaml', 'r') as file:
	config = yaml.safe_load(file)
```

`pyyaml` will automatically convert data types as best as it can. This includes dates like `1990-05-22,` as above, but only if they follow the ISO 8601 standard for dates and times.

And that's part of the problem. If you or your colleague forgets to use this format and writes `22.05.1990` instead, the `safe_load()` method will still read it into your data. But now it will be a string and not a date 😫 You will get a weird error message deep in your code, where the date is used, and it might take you ages to trace that back to the `.yaml`.

Or you accidentally wrote `19` instead of `1.9,` and the variable is only supposed to be between 0-10. Your model might never converge, and that typo will take ages to find.

Of course, you could write assessments for all of these values by hand, but Pydantic offers you something much more convenient and less error-prone.

### What Pydantic offers you - Base Classes and Validators

Before diving into ML configs, let’s cover the **core Pydantic concepts** you’ll need:

#### `BaseModel`

Every config model inherits from the class `pydantic.BaseModel`.

This provides automatic parsing, type validation, and helpful errors.

```python
from pydantic import BaseModel

class TrainingConfig(BaseModel):
    target_column: str
    n_estimators: int
```

Usage:

```python
cfg = TrainingConfig(target_column="price", n_estimators=100)
print(cfg.n_estimators)  # → 100
```

If you input a variable with the wrong type, like this (or via the `.yaml` of course)

```python
cfg = TrainingConfig(target_column="price", n_estimators="test")
```

Pydantic will raise a clear error:

```zsh
pydantic_core._pydantic_core.ValidationError: 1 validation error for TrainingConfig
cfg.n_estimators
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='test', input_type=str]
    For further information visit https://errors.pydantic.dev/2.11/v/int_parsing
```

#### `Field`

In addition to the type of the variable, you can use `Field()` to declare:

-   **Defaults**
-   **Constraints** (e.g. ≥0 via `ge`, ≤1 via `le`)
-   **Descriptions** (optional)

```python
from pydantic import Field

class SplitConfig(BaseModel):
    train_size: float = Field(0.8, ge=0, le=1, description="Fraction of data for training")
    test_size: float = Field(0.2, ge=0, le=1, description="Fraction of data for testing")
```

This ensures only values between 0 and 1 are accepted.

#### Validators

Use validators to check for values that should fulfill certain qualities that are not simply "Numbers should be between 0 and 1".

For example, ensuring all split sizes sum to 1:

```python
from pydantic import model_validator

class SplitConfig(BaseModel):
    train_size: float = 0.6
    val_size: float = 0.2
    test_size: float = 0.2

    @model_validator(mode="after")
    def check_sum(cls, values):
        total = values.train_size + values.val_size + values.test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train + val + test must sum to 1")
        return values
```

Now, if your YAML says `train_size: 1.5`, you’ll get an immediate `ValidationError`, not a cryptic pandas slicing error 10 minutes later.

## 3. Setup: Dependencies and File Structure

Dependencies for this tutorial include `pydantic` and `pyyaml`.

```zsh
uv add pydantic pyyaml
```

You can find the full list of installed libraries in the `pyproject.toml` in the [example repository on GitHub](https://github.com/GalaxyInfernoCodes/ml-blueprint/tree/august-week2?ref=sarahglasmacher.com).

**Repo structure** building on the structure of the previous week of the [challenge](https://www.sarahglasmacher.com/ml-repo-structure-challenge/):

```
.
├─ config/
│  └─ config.yaml
├─ data/                  # small CSV for demo
├─ scripts/
│  └─ one_stop_script.py
├─ src/
│  └─ ml_blueprint/
│     ├─ __init__.py
│     ├─ config.py       # Pydantic loaders
│     ├─ data.py         # read + split
│     └─ model.py        # trainer
├─ pyproject.toml
```

This ensures:

-   `src/` is importable (`from ml_blueprint import ...` works),
-   Config lives **outside** source code,
-   Scripts are thin wiring layers.

## 4. Define the Config Models

I like to define multiple configuration classes that each encompass a specific domain. That way, I can pass the objects to the modular code for the data pipeline or the model trainer, respectively, and it's easier to find the right variable.

In `src/ml_blueprint/config.py`:

-   Use **Pydantic models** to validate YAML against a schema,
-   Create small, typed models with sensible **defaults**.
-   Provide a loader that accepts either a flat mapping or an `ml_blueprint:` namespace.
-   Fail fast with **validation errors** instead of mysterious runtime bugs.

```python
from pydantic import BaseModel, Field, model_validator
from pathlib import Path
import yaml

class SplitConfig(BaseModel):
    train_size: float = Field(0.6, ge=0, le=1)
    val_size: float = Field(0.2, ge=0, le=1)
    test_size: float = Field(0.2, ge=0, le=1)
    random_state: int = 42

    @model_validator(mode="after")
	# ... see above

class DataConfig(BaseModel):
    source_path: Path = Path("./data/playground")
    target_path: Path = Path("./data/processed")
    split: SplitConfig = SplitConfig()

# ...

class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        if "ml_blueprint" in data:
            data = data["ml_blueprint"]
        return cls.model_validate(data)
```

## 5. Write the YAML

Not much magic to this one - just add the entries that you defined in the `config.py` for your current experiment. These two files (the `.yaml` and the `config.py`) will grow next to each other as you code on your project. They will also probably be refactored a few times as the project grows and becomes more and more complex.

Here's an example from the blueprint-project: `config/config.yaml`

```yaml
ml_blueprint:
  data:
    source_path: ./data/playground-series-s4e12
    target_path: ./data
    split:
      train_size: 0.6
      val_size: 0.2
      test_size: 0.2
      random_state: 42
  training:
    target_column: Premium Amount
    drop_columns: ['Policy Start Date']
  model:
    random_state: 42
    xgb:
      n_estimators: 300
      learning_rate: 0.17
      max_depth: 7
      subsample: 0.3
      # ... more hyperparameters
```

## 6. Load & Run

Your scripts should be **as thin as possible** and orchestrate all your logic pieces. Part of that is now that you load the configuration and then pass its parts to the respective modules.

Here's an example of a simple training script (`scripts/one_stop_script.py` in the appended repository) for demonstration purposes:

```python
from ml_blueprint.data import DataHandler
from ml_blueprint.model import ModelTrainer
from ml_blueprint.config import load_config

cfg = load_config("config/config.yaml")

data_handler = DataHandler(cfg.data)
data_handler.prepare_data()
train_df, val_df, test_df = data_handler.get_data()

trainer = ModelTrainer(train_df, val_df, test_df)
trainer.separate_columns(target_col=cfg.training.target_column)
model = trainer.build_pipeline(RANDOM_STATE=cfg.model.random_state)
model.fit(trainer.X_train, trainer.y_train)
trainer.evaluate()
```

Notice:

-   Script does **not** hardcode any paths (except for the one pointing to the configuration), sizes, or hyperparams.
-   Change YAML → new run behavior → reproducibility

You can also pass the configuration file to the script using a command-line change:

```python
parser = argparse.ArgumentParser(description="Run end-to-end ML pipeline.")
parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
args = parser.parse_args()
cfg = load_config(args.config)
# rest as above
```

📗

**Note**: By default, `argparse` takes the **long flag name** (here `--config`) and creates an attribute with that name (minus the dashes). That’s why you can access the parsed value via `args.config`. If you want a different attribute name, you can override it with `dest`.

## 7. Refactor classes like DataHandler to use the passed configuration

Keep modules like `data.py` clean: it should depend only on the **DataConfig object**, not 20 separate numbers and strings.

```python
class DataHandler:
    def __init__(self, data_config: DataConfig):
        self.cfg = data_config

    def prepare_data(self):
	    # for example refer to the source_path
        df = pd.read_csv(self.cfg.source_path / "train.csv", index_col=0)

        train, temp = train_test_split(
            df, test_size=self.cfg.split.test_size, random_state=self.cfg.split.random_state
        )
	# ...
```

## 8. Validate

Ensure you have all dependencies installed:

```bash
uv add pydantic pyyaml # plus more that you need for the ML part itself
```

Run:

```bash
uv run python scripts/one_stop_script.py # or whatever your script name is
```

Try messing up the YAML (wrong type, wrong key, invalid split) → **Pydantic will catch it** before your training pipeline does.

## 9. Pattern Checklist (Reusable!)

✅ Small Pydantic models with defaults & type safety,

✅ Namespace YAML under a top-level key (optional but neat),

✅ Provide a single `load_config()` entrypoint,

✅ Pass full config objects to constructors instead of long param lists,

✅ Scripts stay thin: _config-in, wire-up, call main steps_.

## Sources and Links

-   Find my full code example at [https://github.com/GalaxyInfernoCodes/ml-blueprint/tree/august-week2](https://github.com/GalaxyInfernoCodes/ml-blueprint/tree/august-week2?ref=sarahglasmacher.com)
-   The official Pydantic docs: [https://docs.pydantic.dev/latest/concepts/models/](https://docs.pydantic.dev/latest/concepts/models/?ref=sarahglasmacher.com)
-   A cool overview from a public-repo MLOps Course: [https://mlops-coding-course.fmind.dev/2.%20Prototyping/2.2.%20Configs.html](https://mlops-coding-course.fmind.dev/2.%20Prototyping/2.2.%20Configs.html?ref=sarahglasmacher.com) on why you should use configs

</details>

<details>
<summary>```markdown</summary>

```markdown
## Setup

To build a workflow or agent, you can use any chat model that supports structured outputs and tool calling. The following example uses Anthropic:

1. Install dependencies:

```
pip install langchain_core langchain-anthropic langgraph
```

2. Initialize the LLM:

```
import os
import getpass

from langchain_anthropic import ChatAnthropic

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
```

## Evaluator-optimizer

In evaluator-optimizer workflows, one LLM call creates a response and the other evaluates that response. If the evaluator or a human-in-the-loop determines the response needs refinement, feedback is provided and the response is recreated. This loop continues until an acceptable response is generated.Evaluator-optimizer workflows are commonly used when there’s particular success criteria for a task, but iteration is required to meet that criteria. For example, there’s not always a perfect match when translating text between two languages. It might take a few iterations to generate a translation with the same meaning across the two languages.
https://mintcdn.com/langchain-5e9cc07a/-_xGPoyjhyiDWTPJ/oss/images/evaluator_optimizer.png?fit=max&auto=format&n=-_xGPoyjhyiDWTPJ&q=85&s=9bd0474f42b6040b14ed6968a9ab4e3c

Graph API

Functional API

```
# Graph state
class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )

# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)

# Nodes
def llm_call_generator(state: State):
    """LLM generates a joke"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""

    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

# Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"

# Build workflow
optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # Name returned by route_joke : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

# Show the workflow
display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))

# Invoke
state = optimizer_workflow.invoke({"topic": "Cats"})
print(state["joke"])
```
```

</details>

<details>
<summary>What are agents?</summary>

## What are agents?

"Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:

-   **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
-   **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Below, we will explore both types of agentic systems in detail. In Appendix 1 (“Agents in Practice”), we describe two domains where customers have found particular value in using these kinds of systems.

## When (and when not) to use agents

When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

When more complexity is warranted, workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale. For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough.

## When and how to use frameworks

There are many frameworks that make agentic systems easier to implement, including:

-   The [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview);
-   Amazon Bedrock's [AI Agent framework](https://aws.amazon.com/bedrock/agents/);
-   [Rivet](https://rivet.ironcladapp.com/), a drag and drop GUI LLM workflow builder; and
-   [Vellum](https://www.vellum.ai/), another GUI tool for building and testing complex workflows.

These frameworks make it easy to get started by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together. However, they often create extra layers of abstraction that can obscure the underlying prompts ​​and responses, making them harder to debug. They can also make it tempting to add complexity when a simpler setup would suffice.

We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error.

See our [cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents) for some sample implementations.

## Building blocks, workflows, and agents

In this section, we’ll explore the common patterns for agentic systems we’ve seen in production. We'll start with our foundational building block—the augmented LLM—and progressively increase complexity, from simple compositional workflows to autonomous agents.

### Building block: The augmented LLM

The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory. Our current models can actively use these capabilities—generating their own search queries, selecting appropriate tools, and determining what information to retain.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fd3083d3f40bb2b6f477901cc9a240738d3dd1371-2401x1000.png&w=3840&q=75The augmented LLM

We recommend focusing on two key aspects of the implementation: tailoring these capabilities to your specific use case and ensuring they provide an easy, well-documented interface for your LLM. While there are many ways to implement these augmentations, one approach is through our recently released [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol), which allows developers to integrate with a growing ecosystem of third-party tools with a simple [client implementation](https://modelcontextprotocol.io/tutorials/building-a-client#building-mcp-clients).

For the remainder of this post, we'll assume each LLM call has access to these augmented capabilities.

### Workflow: Prompt chaining

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F7418719e3dab222dccb379b8879e1dc08ad34c78-2401x1000.png&w=3840&q=75The prompt chaining workflow

**When to use this workflow:** This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

**Examples where prompt chaining is useful:**

-   Generating Marketing copy, then translating it into a different language.
-   Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

### Workflow: Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840&q=75The routing workflow

**When to use this workflow:** Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

**Examples where routing is useful:**

-   Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes, prompts, and tools.
-   Routing easy/common questions to smaller, cost-efficient models like Claude Haiku 4.5 and hard/unusual questions to more capable models like Claude Sonnet 4.5 to optimize for best performance.

### Workflow: Parallelization

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations:

-   **Sectioning**: Breaking a task into independent subtasks run in parallel.
-   **Voting:** Running the same task multiple times to get diverse outputs.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840&q=75The parallelization workflow

**When to use this workflow:** Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.

**Examples where parallelization is useful:**

-   **Sectioning**:
    -   Implementing guardrails where one model instance processes user queries while another screens them for inappropriate content or requests. This tends to perform better than having the same LLM call handle both guardrails and the core response.
    -   Automating evals for evaluating LLM performance, where each LLM call evaluates a different aspect of the model’s performance on a given prompt.
-   **Voting**:
    -   Reviewing a piece of code for vulnerabilities, where several different prompts review and flag the code if they find a problem.
    -   Evaluating whether a given piece of content is inappropriate, with multiple prompts evaluating different aspects or requiring different vote thresholds to balance false positives and negatives.

### Workflow: Orchestrator-workers

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840&q=75The orchestrator-workers workflow

**When to use this workflow:** This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

**Example where orchestrator-workers is useful:**

-   Coding products that make complex changes to multiple files each time.
-   Search tasks that involve gathering and analyzing information from multiple sources for possible relevant information.

### Workflow: Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75The evaluator-optimizer workflow

**When to use this workflow:** This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

**Examples where evaluator-optimizer is useful:**

-   Literary translation where there are nuances that the translator LLM might not capture initially, but where an evaluator LLM can provide useful critiques.
-   Complex search tasks that require multiple rounds of searching and analysis to gather comprehensive information, where the evaluator decides whether further searches are warranted.

### Agents

Agents are emerging in production as LLMs mature in key capabilities—understanding complex inputs, engaging in reasoning and planning, using tools reliably, and recovering from errors. Agents begin their work with either a command from, or interactive discussion with, the human user. Once the task is clear, agents plan and operate independently, potentially returning to the human for further information or judgement. During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress. Agents can then pause for human feedback at checkpoints or when encountering blockers. The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully. We expand on best practices for tool development in Appendix 2 ("Prompt Engineering your Tools").

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png&w=3840&q=75Autonomous agent

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

**Examples where agents are useful:**

The following examples are from our own implementations:

-   A coding Agent to resolve [SWE-bench tasks](https://www.anthropic.com/research/swe-bench-sonnet), which involve edits to many files based on a task description;
-   Our [“computer use” reference implementation](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo), where Claude uses a computer to accomplish tasks.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F4b9a1f4eb63d5962a6e1746ac26bbc857cf3474f-2400x1666.png&w=3840&q=75High-level flow of a coding agent

## Combining and customizing these patterns

These building blocks aren't prescriptive. They're common patterns that developers can shape and combine to fit different use cases. The key to success, as with any LLM features, is measuring performance and iterating on implementations. To repeat: you should consider adding complexity _only_ when it demonstrably improves outcomes.

## Summary

Success in the LLM space isn't about building the most sophisticated system. It's about building the _right_ system for your needs. Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short.

When implementing agents, we try to follow three core principles:

1.  Maintain **simplicity** in your agent's design.
2.  Prioritize **transparency** by explicitly showing the agent’s planning steps.
3.  Carefully craft your agent-computer interface (ACI) through thorough tool **documentation and testing**.

Frameworks can help you get started quickly, but don't hesitate to reduce abstraction layers and build with basic components as you move to production. By following these principles, you can create agents that are not only powerful but also reliable, maintainable, and trusted by their users.

## Appendix 1: Agents in practice

Our work with customers has revealed two particularly promising applications for AI agents that demonstrate the practical value of the patterns discussed above. Both applications illustrate how agents add the most value for tasks that require both conversation and action, have clear success criteria, enable feedback loops, and integrate meaningful human oversight.

### A. Customer support

Customer support combines familiar chatbot interfaces with enhanced capabilities through tool integration. This is a natural fit for more open-ended agents because:

-   Support interactions naturally follow a conversation flow while requiring access to external information and actions;
-   Tools can be integrated to pull customer data, order history, and knowledge base articles;
-   Actions such as issuing refunds or updating tickets can be handled programmatically; and
-   Success can be clearly measured through user-defined resolutions.

Several companies have demonstrated the viability of this approach through usage-based pricing models that charge only for successful resolutions, showing confidence in their agents' effectiveness.

### B. Coding agents

The software development space has shown remarkable potential for LLM features, with capabilities evolving from code completion to autonomous problem-solving. Agents are particularly effective because:

-   Code solutions are verifiable through automated tests;
-   Agents can iterate on solutions using test results as feedback;
-   The problem space is well-defined and structured; and
-   Output quality can be measured objectively.

In our own implementation, agents can now solve real GitHub issues in the [SWE-bench Verified](https://www.anthropic.com/research/swe-bench-sonnet) benchmark based on the pull request description alone. However, whereas automated testing helps verify functionality, human review remains crucial for ensuring solutions align with broader system requirements.

## Appendix 2: Prompt engineering your tools

No matter which agentic system you're building, tools will likely be an important part of your agent. [Tools](https://www.anthropic.com/news/tool-use-ga) enable Claude to interact with external services and APIs by specifying their exact structure and definition in our API. When Claude responds, it will include a [tool use block](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#example-api-response-with-a-tool-use-content-block) in the API response if it plans to invoke a tool. Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts. In this brief appendix, we describe how to prompt engineer your tools.

There are often several ways to specify the same action. For instance, you can specify a file edit by writing a diff, or by rewriting the entire file. For structured output, you can return code inside markdown or inside JSON. In software engineering, differences like these are cosmetic and can be converted losslessly from one to the other. However, some formats are much more difficult for an LLM to write than others. Writing a diff requires knowing how many lines are changing in the chunk header before the new code is written. Writing code inside JSON (compared to markdown) requires extra escaping of newlines and quotes.

Our suggestions for deciding on tool formats are the following:

-   Give the model enough tokens to "think" before it writes itself into a corner.
-   Keep the format close to what the model has seen naturally occurring in text on the internet.
-   Make sure there's no formatting "overhead" such as having to keep an accurate count of thousands of lines of code, or string-escaping any code it writes.

One rule of thumb is to think about how much effort goes into human-computer interfaces (HCI), and plan to invest just as much effort in creating good _agent_-computer interfaces (ACI). Here are some thoughts on how to do so:

-   Put yourself in the model's shoes. Is it obvious how to use this tool, based on the description and parameters, or would you need to think carefully about it? If so, then it’s probably also true for the model. A good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools.
-   How can you change parameter names or descriptions to make things more obvious? Think of this as writing a great docstring for a junior developer on your team. This is especially important when using many similar tools.
-   Test how the model uses your tools: Run many example inputs in our [workbench](https://console.anthropic.com/workbench) to see what mistakes the model makes, and iterate.
-   [Poka-yoke](https://en.wikipedia.org/wiki/Poka-yoke) your tools. Change the arguments so that it is harder to make mistakes.

While building our agent for [SWE-bench](https://www.anthropic.com/research/swe-bench-sonnet), we actually spent more time optimizing our tools than the overall prompt. For example, we found that the model would make mistakes with tools using relative filepaths after the agent had moved out of the root directory. To fix this, we changed the tool to always require absolute filepaths—and we found that the model used this method flawlessly.

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/towardsai/agentic-ai-engineering-course/tree/dev/lessons/23_evaluator_optimizer</summary>

# Repository analysis for https://github.com/towardsai/agentic-ai-engineering-course/tree/dev/lessons/23_evaluator_optimizer

## Summary
Repository: towardsai/agentic-ai-engineering-course
Branch: dev
Commit: 93cf10331fef0ff5b80e7a021794a906625d3044
Subpath: /lessons/23_evaluator_optimizer
Files analyzed: 3

Estimated tokens: 42.5k

## File tree
```Directory structure:
└── 23_evaluator_optimizer/
    ├── aritcle_guideline.md
    ├── notebook.ipynb
    └── notebook_guideline.md

```

## Extracted content
================================================
FILE: lessons/23_evaluator_optimizer/aritcle_guideline.md
================================================
1. Evaluator optimizer (workflows) vs Reflection (agents)
2. Explain more depth the article reviewer + editor prompts!!!
2. To clarify: Why we don't compute any score into the evaluator optimizer loop
3. Code structure:
    - We started introducing the app layer through the LangGraph workflows
    - And the infra layer through the memory
    - Note how we inject the infra stuff, such as the memory, only when initializing the workflow (app components)


================================================
FILE: lessons/23_evaluator_optimizer/notebook.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Lesson 23: Evaluator-Optimizer Pattern — Reviewing and Editing the Brown Agent

In this lesson, we'll explore how to implement the evaluator-optimizer pattern to review and edit generated articles. Building on the foundation from Lesson 22, we'll add a quality assurance layer that ensures the generated content meets all requirements.

Learning Objectives:

- Understand the evaluator-optimizer pattern and its real-world applications
- Implement an article reviewing system that checks content against multiple profiles
- Extend the article writer to handle review feedback
- Configure the entire system from a single YAML file
- Glue everything together into a robust LangGraph workflow

"""

"""
> [!NOTE]
> 💡 Remember that you can also run `brown` as a standalone Python package by going to `lessons/writing_workflow/` and following the instructions from there.
"""

"""
## 1. Setup

First, we define some standard Magic Python commands to autoreload Python packages whenever they change:

"""

%load_ext autoreload
%autoreload 2

"""
### Set Up Python Environment

To set up your Python virtual environment using `uv` and load it into the Notebook, follow the step-by-step instructions from the `Course Admin` lesson from the beginning of the course.

**TL/DR:** Be sure the correct kernel pointing to your `uv` virtual environment is selected.

"""

"""
### Configure Gemini API

To configure the Gemini API, follow the step-by-step instructions in the `Course Admin` lesson.

Here is a quick checklist of what you need to run this notebook:

1.  Get your key from [Google AI Studio](https://aistudio.google.com/app/api-keys).
2.  From the root of your project, run: `cp .env.example .env` 
3.  Within the `.env` file, fill in the `GOOGLE_API_KEY` variable:

Now, the code below will load the key from the `.env` file:
"""

from utils import env

env.load(required_env_vars=["GOOGLE_API_KEY"])
# Output:
#   Environment variables loaded from `/Users/pauliusztin/Documents/01_projects/TAI/agentic-ai-engineering-course/.env`

#   Environment variables loaded successfully.


"""
### Import Key Packages

"""

import nest_asyncio
from utils import pretty_print

nest_asyncio.apply()  # Allow nested async usage in notebooks

"""
### Download Required Files

We need to download the configuration files and input data that Brown uses for article generation and editing.

First, let's download the configs folder:
"""

%%capture

!rm -rf configs
!curl -L -o configs.zip https://raw.githubusercontent.com/towardsai/agentic-ai-engineering-course/main/data/configs.zip
!unzip configs.zip
!rm -rf configs.zip

"""
Now, let's download the inputs folder containing profiles, examples, and test data:
"""

%%capture

!rm -rf inputs
!curl -L -o inputs.zip https://raw.githubusercontent.com/towardsai/agentic-ai-engineering-course/main/data/inputs.zip
!unzip inputs.zip
!rm -rf inputs.zip

"""
Let's verify what we downloaded:
"""

%ls
# Output:
#   aritcle_guideline.md   [1m[36minputs[m[m/                notebook_guideline.md

#   [1m[36mconfigs[m[m/               notebook.ipynb


"""
### Set Up Directory Constants

Now let's define constants to reference these directories throughout the notebook:
"""

from pathlib import Path

CONFIGS_DIR = Path("configs")
INPUTS_DIR = Path("inputs")

print(f"Configs directory exists: {CONFIGS_DIR.exists()}")
print(f"Inputs directory exists: {INPUTS_DIR.exists()}")
# Output:
#   Configs directory exists: True

#   Inputs directory exists: True


EXAMPLES_DIR = Path("inputs/examples/course_lessons")
PROFILES_DIR = Path("inputs/profiles")

print(f"Examples directory exists: {EXAMPLES_DIR.exists()}")
print(f"Profiles directory exists: {PROFILES_DIR.exists()}")
# Output:
#   Examples directory exists: True

#   Profiles directory exists: True


"""
First, we will load a simpler example that runs faster and is easier to understand. At the end, we will load a larger sample that is closer to what we do on our end to generate professional articles:
"""

SAMPLE_DIR = Path("inputs/tests/01_sample_small")

print(f"Samples directory exists: {SAMPLE_DIR.exists()}")
# Output:
#   Samples directory exists: True


"""
## 2. How the Writing Agent Works with Review-Edit Loop

Before diving into the implementation, let's understand how the writing agent now incorporates the review-editing process through the evaluator-optimizer pattern.

### The Extended Workflow

In Lesson 22, we learned about the three-step workflow:

1. **Load Context into Memory** - Gather guidelines, research, profiles, and examples
2. **Generate Media Items** - Use the orchestrator-worker pattern to create diagrams
3. **Write the Article** - Generate the first draft using the ArticleWriter

Now we're adding a fourth and fifth step that loops multiple times:

4. **Review the Article** (Evaluator) - Check the article against all profiles and guidelines
5. **Edit the Article** (Optimizer) - Fix all identified issues based on the reviews

This review-edit pattern continues for a configurable number of iterations, gradually improving the article quality.

### The Evaluator-Optimizer Pattern Explained

The evaluator-optimizer pattern is a fundamental AI workflow pattern that mirrors real-world quality assurance processes:

- **Evaluator**: Analyzes output and identifies issues or areas for improvement
- **Optimizer**: Takes the feedback and makes targeted improvements

In our case:
- **Article Reviewer Node** = Evaluator (checks if article follows all the standards)
- **Article Writer Node** = Optimizer (edits the article based on reviews)

This approach is extremely similar to how a real-world writing process works:

1. The writer writes the article (initial draft)
2. A reviewer provides feedback from outside eyes
3. The same writer edits the article based on the provided feedback
4. Repeat steps 2-3 until satisfied
"""

"""
### Workflow Visualization

Let's visualize the complete workflow with the review-edit loop:

<img src="https://raw.githubusercontent.com/towardsai/agentic-ai-engineering-course/main/assets/images/l23_writing_workflow.png" alt="Workflow" height="800"/>
"""

"""
## 3. Review Entities: Modeling Feedback

Now let's explore the new Pydantic entities we need for the review process. In Lesson 22, we already covered the core entities like `Article`, `ArticleGuideline`, and `ArticleProfiles`. Now we need entities to represent the reviewing logic.

### Why Two Types of Reviews?

We support two review modes:

1. **Whole Article Reviews**: Review the entire article from top to bottom
2. **Selected Text Reviews**: Review only a specific portion of the article

Most of the time, only a section of the article needs editing, not the whole thing. This targeted approach saves time and reduces API costs by only reviewing what matters.

### The Review Entities

From `brown.entities.reviews`, we have these core entities:

"""

"""
### 1. The Review Entity

A `Review` represents a single piece of feedback about the article:


```python
from pydantic import BaseModel, Field
from brown.entities.mixins import ContextMixin


class Review(BaseModel, ContextMixin):
    profile: str = Field(
        description="The profile type listing the constraints based on which we will write the comment."
    )
    location: str = Field(
        description="The location from within the article where the comment is made. For example, the title of a section."
    )
    comment: str = Field(
        description="The comment made by the reviewer stating the issue relative to the profile."
    )

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    <profile>{self.profile}</profile>
    <location>{self.location}</location>
    <comment>{self.comment}</comment>
</{self.xml_tag}>
"""
```

**Key Fields:**

- **profile**: Which requirement was violated (e.g., "tonality_profile", "article_guideline", "structured_profile")
- **location**: Where in the article the issue exists, usually the title of the article section (e.g., "Introduction - Second paragraph")
- **comment**: Detailed explanation of what's wrong and why it deviates from the requirement

**Example Review:**

```python
Review(
    profile="tonality_profile",
    location="Introduction - First paragraph",
    comment="The tone is overly formal. The tonality profile specifies a conversational, friendly tone. The current opening reads like an academic paper rather than an engaging blog post."
)
```

"""

"""
### 2. The ArticleReviews Entity

`ArticleReviews` bundles multiple reviews for the whole article:


```python
class ArticleReviews(BaseModel, ContextMixin):
    article: Article
    reviews: list[Review]

    def to_context(self, include_article: bool = False) -> str:
        reviews_str = "\n".join([review.to_context() for review in self.reviews])
        return f"""
<{self.xml_tag}>
    {f"<article>{self.article}</article>" if include_article else ""}
    <reviews>
    {reviews_str}
    </reviews>
</{self.xml_tag}>
"""

    def __str__(self) -> str:
        return f"Reviews(len_reviews={len(self.reviews)})"
```

"""

"""
### 3. The SelectedText Entity

Before understanding `SelectedTextReviews`, we need to see the `SelectedText` entity from `brown.entities.articles` to understand how we will model the selected text relative to how we did for the whole article:


```python
class SelectedText(BaseModel, ContextMixin):
    article: Article
    content: str
    first_line_number: int
    last_line_number: int

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    <content>{self.content}</content>
    <first_line_number>{self.first_line_number}</first_line_number>
    <last_line_number>{self.last_line_number}</last_line_number>
</{self.xml_tag}>
"""
```

**Key Features:**

- Contains the full `article` for context
- `content`: The specific text selection to review/edit
- Line numbers help locate the selection within the full article
- This enables targeted reviews of specific sections

"""

"""
### 4. The SelectedTextReviews Entity

`SelectedTextReviews` handles reviews for just a portion of the article:


```python
class SelectedTextReviews(BaseModel, ContextMixin):
    article: Article
    selected_text: SelectedText
    reviews: list[Review]

    def to_context(self, include_article: bool = False) -> str:
        reviews_str = "\n".join([review.to_context() for review in self.reviews])
        return f"""
<{self.xml_tag}>
    {f"<article>{self.article.to_context()}</article>" if include_article else ""}
    <selected_text>{self.selected_text.to_context()}</selected_text>
    <reviews>
    {reviews_str}
    </reviews>
</{self.xml_tag}>
"""
```

**Use Case:**

When a user identifies a specific problematic section, we can:
1. Create a `SelectedText` entity pointing to that section
2. Review only that selection (faster, cheaper)
3. Edit only that selection
4. Replace the selection in the full article

This is particularly useful for human-in-the-loop workflows where humans can highlight specific sections for improvement. More on this in Lesson 24.

"""

"""
### Entity Relationships

Let's visualize how these entities relate:

```
Article
  └── ArticleReviews
       └── reviews: list[Review]

Article + SelectedText
  └── SelectedTextReviews  
       ├── selected_text: SelectedText
       └── reviews: list[Review]
```

"""

"""
## 4. The Article Reviewer Node: The Evaluator

Now let's explore the `ArticleReviewer` node, which acts as the **evaluator** in our evaluator-optimizer pattern. This node analyzes articles against all requirements and generates detailed feedback.

Remember that the core expectations are that the article follows the article guidelines and that all the writing profiles are respected.

### Node Abstraction Recap

First, a quick reminder that we leverage the same `Node` abstraction from Lesson 22 to implement all our nodes.


```python
from abc import ABC, abstractmethod
from typing import Any

from brown.nodes.base import Node, Toolkit


class Node(ABC):
    def __init__(self, model: Runnable, toolkit: Toolkit) -> None:
        self.toolkit = toolkit
        self.model = self._extend_model(model)

    def _extend_model(self, model: Runnable) -> Runnable:
        # Can be overridden to bind tools, structured output, etc.
        return model

    @abstractmethod
    async def ainvoke(self) -> Any:
        pass
```

All workflow nodes inherit from this base class, providing a consistent interface throughout the system.

"""

"""
### ArticleReviewer Class Structure

Let's examine the `ArticleReviewer` class from `brown.nodes.article_reviewer`:

**1. The Class and Initialization:**

```python
class ArticleReviewer(Node):
    system_prompt_template = """..."""  # We'll see this shortly
    selected_text_system_prompt_template = """..."""

    def __init__(
        self,
        to_review: Article | SelectedText,
        article_guideline: ArticleGuideline,
        model: Runnable,
        article_profiles: ArticleProfiles,
    ) -> None:
        self.to_review = to_review
        self.article_guideline = article_guideline
        self.article_profiles = article_profiles

        super().__init__(model, toolkit=Toolkit(tools=[]))
```

**Key Design Decisions:**

- `to_review` can be either a full `Article` or just `SelectedText` (polymorphic design)
- Takes all the requirements: guideline, profiles
- No tools needed (empty toolkit), as reviewing is a pure generation task and no tools are required

"""

"""
**2. Model Extension with Structured Output:**

```python
    def _extend_model(self, model: Runnable) -> Runnable:
        model = cast(BaseChatModel, super()._extend_model(model))
        model = model.with_structured_output(ReviewsOutput)
        
        return model
```

The reviewer uses structured output to ensure we get properly formatted reviews. First, we need an intermediate Pydantic model:

```python
class ReviewsOutput(BaseModel):
    reviews: list[Review]
```

**Why an intermediate model?**

The LLM outputs `ReviewsOutput`, but the node returns either `ArticleReviews` or `SelectedTextReviews` (which include the article/selected_text). This separation keeps the LLM output schema simple to avoid any potential LLM inference errors, while allowing richer node outputs.

"""

"""
**3. The ainvoke Method:**

```python
    async def ainvoke(self) -> ArticleReviews | SelectedTextReviews:
        # Build the main system prompt with all requirements
        system_prompt = self.system_prompt_template.format(
            human_feedback=self.human_feedback.to_context() if self.human_feedback else "",
            article=self.article.to_context(),
            article_guideline=self.article_guideline.to_context(),
            character_profile=self.article_profiles.character.to_context(),
            article_profile=self.article_profiles.article.to_context(),
            structure_profile=self.article_profiles.structure.to_context(),
            mechanics_profile=self.article_profiles.mechanics.to_context(),
            terminology_profile=self.article_profiles.terminology.to_context(),
            tonality_profile=self.article_profiles.tonality.to_context(),
        )
        
        user_input_content = self.build_user_input_content(inputs=[system_prompt])
        inputs = [{"role": "user", "content": user_input_content}]
        
        # If reviewing selected text, add additional instructions
        if self.is_selected_text:
            inputs.extend([
                {
                    "role": "user",
                    "content": self.selected_text_system_prompt_template.format(
                        selected_text=self.to_review.to_context()
                    ),
                }
            ])
        
        # Generate reviews
        reviews = await self.model.ainvoke(inputs)
        if not isinstance(reviews, ReviewsOutput):
            raise InvalidOutputTypeException(ReviewsOutput, type(reviews))
        
        # Return appropriate review type
        if self.is_selected_text:
            return SelectedTextReviews(
                article=self.article,
                selected_text=cast(SelectedText, self.to_review),
                reviews=reviews.reviews,
            )
        else:
            return ArticleReviews(
                article=self.article,
                reviews=reviews.reviews,
            )
```

**Flow:**

1. Format the system prompt with all requirements
2. If reviewing selected text, add special instructions
3. Generate structured reviews from the LLM
4. Package the output entity into the appropriate review type

"""

"""
**4. The System Prompt (Main Review Logic):**

Here's the system prompt which is carefully designed to create thorough, actionable reviews based on the article guideline and writing profiles:

```python
class ArticleReviewer(Node):
    system_prompt_template = """
You are Brown, an expert article writer, editor and reviewer specialized in reviewing technical, educative and informational articles.

Your task is to review a given article against a set of expected requirements and provide detailed feedback 
about any deviations. You will act as a quality assurance reviewer, identifying specific issues and suggesting 
how the article fails to meet the expected requirements.

These reviews will further be used to edit the article, ensuring it follows all the requirements.

## Requirements

The requirements are a set of rules, guidelines or profiles that the article should follow. Here they are:

- **article guideline:** the user intent describing how the article should look like. Specific to this particular article.
- **article profile:** rules specific to writing articles. Generic for all articles.
- **character profile:** the character you will impersonate while writing. Generic for all content.
- **structure profile:** Structure rules guiding the final output format. Generic for all content.
- **mechanics profile:** Mechanics rules guiding the writing process. Generic for all content.
- **terminology profile:** Terminology rules guiding word choice and phrasing. Generic for all content.
- **tonality profile:** Tonality rules guiding the writing style. Generic for all content.

## Article to Review

Here is the article that needs to be reviewed:

{article}

## Article Guideline

The <article_guideline> represents the user intent, describing how the actual article should look like.

The <article_guideline> will ALWAYS contain:
- all the sections of the article expected to be written, in the correct order
- a level of detail for each section, describing what each section should contain. Depending on how much detail you have in a
particular section of the <article_guideline>, you will use more or less information from the <research> tags to write the section.

The <article_guideline> can ALSO contain:
- length constraints for each section, such as the number of characters, words or reading time. If present, you will respect them.
- important (golden) references as URLs or titles present in the <research> tags. If present, always prioritize them over anything else 
from the <research>.
- information about anchoring the article into a series such as a course or a book. Extremely important when the article is part of 
something bigger and we have to anchor the article into the learning journey of the reader. For example, when introducing concepts
in previous articles that we don't want to reintroduce into the current one.
- concrete information about writing the article. If present, you will ALWAYS priotize the instructions from the <article_guideline> 
over any other instructions.

Here is the article guideline:
{article_guideline}

## Character Profile

To make the writing more personable, we impersonated the following character profile when writing the article:
{character_profile}

## Terminology Profile

Here is the terminology profile, describing how to choose the right words and phrases:
to the target audience:
{terminology_profile}

## Tonality Profile

Here is the tonality profile, describing the tone, voice and style of the writing:
{tonality_profile}

## Mechanics Profile

Here is the mechanics profile, describing how the sentences and words should be written:
{mechanics_profile}

## Structure Profile

Here is the structure profile, describing general rules on how to structure text, such as the sections, paragraphs, lists,
code blocks, or media items:
{structure_profile}

## Article Profile

Here is the article profile, describing particularities on how the end-to-end article should look like:
{article_profile}

## Reviewing Process

You will review the article against all the requirements above, creating a one-to-many relationship between each requirement and the 
number of required reviews. In other words, for each requirement, you will create 0 to N reviews. If the article follows the 
requirement 100%, you will not create any reviews for it. If it doesn't follow the requirement, you will create as many reviews 
as required to ensure the article follows the requirement.

Remember that these reviews will further be used to edit the article, ensuring it follows all the requirements. Thus, it's
important to make a thorough review, covering all the requirements and not missing any detail.

## Reviewing Rules

- **The first most important rule:** The requirements can contain some special sections labeled as "rules" or 
"correction rules". You should look for <(.*)?rules(.*)?> XML tags like <correction_media_rules>, 
<abbreviations_or_acronyms_never_to_expand_rules>, <correction_reference_rules>. These are special highlights that 
should always be prioritized over other rules during the review process. They should be respected at all costs when 
writing the article. You will always prioritize these rules over other rules from the requirements making them your 
No.1 focus.
- **The second most important rule:** The adherence to the <article_guideline>.
- **The third most important rule:** The adherence to the <article_profile>.
- **The fourth most important rule:** The adherence to the rest of the requirements.

Other more generic rules:
- Be thorough but fair - only flag genuine issues
- Emphasize WHY something is wrong, not just WHAT is wrong
- Focus on significant deviations, not minor nitpicks 

## Output Format

For each issue you identify, create a review with:
- **profile**: The requirement where the issue was found (e.g., "human_feedback", "article_guideline", "character_profile", 
"article_profile", "structure_profile", "mechanics_profile", "terminology_profile", "tonality_profile")
- **location**: The section title where the issue was found and the paragraph number. For example, "Introduction - First paragraph" 
or "Implementing GraphRAG - Third paragraph"
- **comment**: A detailed explanation of why it's wrong, what's wrong and how it deviates from the requirement.

## Chain of Thoughts

1. Read and analyze the article.
2. Read and analyze the <human_feedback>.
3. Read and analyze all the requirements considering the <human_feedback> as a guiding force.
4. Carefully compare the article against the requirements as instructed by the rules above.
5. For each requirement, create 0 to N reviews
6. Return the reviews of the article.
"""
```

**Key Prompt Engineering Techniques:**

1. **Clear Role**: Expert reviewer with specific expertise
2. **Explicit Priority System**: Rules are ranked (special rules > guideline > article profile > other profiles)
3. **Output**: Clear instructions on what we want the LLM to fill for each attribute
5. **Chain of Thought**: Explicit reasoning steps that glue together all the other sections

"""

"""
**5. The Selected Text System Prompt:**

When reviewing only a selected portion, we append additional instructions:

```python
class ArticleReviewer(Node):
    system_prompt_template = """..."""
    
    selected_text_system_prompt_template = """
You already reviewed and edited the whole article. Now we want to further review only a specific portion
of the article, which we label as the <selected_text>. Despite reviewing the selected text, instead of the
article as a whole, you will follow the exact same instructions from above as if you were reviewing the article as a whole.

## Selected Text to Review

Here is the selected text that needs to be reviewed:

{selected_text}

As pointed out before, the selected text is part of the larger <article> that is already reviewed.
You will use the full <article> as context and anchoring the reviewing process within the bigger picture.

The <first_line_number> and <last_line_number> numbers from the <selected_text> indicate the first and 
last line/row numbers of the selected text from the <article>. Use them to locate the selected text within the <article>.

## Chain of Thoughts

Here is the new chain of thoughts logic you will follow when reviewing the selected text. You can ignore the
previous chain of thoughts:

1. Read and analyze the article.
2. Locate the <selected_text> within the <article> based on the <first_line_number> and <last_line_number>.
3. Read and analyze the <human_feedback>.
4. Read and analyze all the requirements considering the <human_feedback> as a guiding force.
5. Carefully compare the selected text against the requirements as instructed by the rules above.
6. For each requirement, create 0 to N reviews
7. Return the reviews of the selected text.
"""
```

This allows focused reviews on specific sections while maintaining context of the full article. As this system prompt is passed together with the `system_prompt_template` system prompt it has to act only as an extension on explaining what to do with a selected text.

The special trick here is that it adds a new `Chain of Thoughts` section that overrides the one from the original system prompts adding specialized instructions on how to reason across the new task, while still having all the context from both system prompts.

"""

"""
### Example: Reviewing a Whole Article

Now let's see the `ArticleReviewer` in action by reviewing a sample article.

First load the sample article guideline and the standard profiles:


"""

from brown.loaders import MarkdownArticleGuidelineLoader, MarkdownArticleLoader, MarkdownArticleProfilesLoader
from brown.models import SupportedModels, get_model
from brown.nodes import ArticleReviewer

# Load the article guideline
guideline_loader = MarkdownArticleGuidelineLoader(uri=Path("article_guideline.md"))
article_guideline = guideline_loader.load(working_uri=SAMPLE_DIR)

# Load the article profiles
profiles_input = {
    "article": PROFILES_DIR / "article_profile.md",
    "character": PROFILES_DIR / "character_profiles" / "paul_iusztin.md",
    "mechanics": PROFILES_DIR / "mechanics_profile.md",
    "structure": PROFILES_DIR / "structure_profile.md",
    "terminology": PROFILES_DIR / "terminology_profile.md",
    "tonality": PROFILES_DIR / "tonality_profile.md",
}
profiles_loader = MarkdownArticleProfilesLoader(uri=profiles_input)
article_profiles = profiles_loader.load(working_uri=SAMPLE_DIR)

pretty_print.wrapped(article_guideline.content[:1000], title="Sample article guideline")
# Output:
#   [32m2025-11-29 17:56:51.665[0m | [1mINFO    [0m | [36mbrown.config[0m:[36m<module>[0m:[36m10[0m - [1mLoading environment file from `.env`[0m

#   [93m------------------------------------- Sample article guideline -------------------------------------[0m

#     ## Outline

#   

#   1. Introduction: The Critical Decision Every AI Engineer Faces

#   2. Understanding the Spectrum: From Workflows to Agents

#   3. Choosing Your Path

#   4. Conclusion: The Challenges of Every AI Engineer

#   

#   ## Section 1 - Introduction: The Critical Decision Every AI Engineer Faces

#   

#   - **The Problem:** When building AI applications, engineers face a critical architectural decision early in their development process. Should they create a predictable, step-by-step workflow where they control every action, or should they build an autonomous agent that can think and decide for itself? This is one of the key decisions that will impact everything from the product such as development time and costs to reliability and user experience.

#   - Quick walkthrough of what we'll learn by the end of this lesson

#   

#   - **Section length:** 100 words

#   

#   ## Section 2 - Understanding the Spectrum: From Workflows to Agents

#   

#   - In this section we want to take a brief look at what LLM workflows and AI agents are. At this po

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Now, load a sample article to review (that we already generated based on the same logic from lesson 22):
"""

article_loader = MarkdownArticleLoader(uri=Path("article.md"))
article = article_loader.load(working_uri=SAMPLE_DIR)

pretty_print.wrapped(f"{article.content[:3000]}...", title="Sample article to review (first 3000 characters)")
# Output:
#   [93m------------------------- Sample article to review (first 3000 characters) -------------------------[0m

#     # Workflows vs. Agents: The Critical Decision Every AI Engineer Faces

#   ### How to choose between predictable control and autonomous flexibility when building AI applications.

#   

#   When building AI applications, engineers face a critical architectural decision early on. Should you create a predictable, step-by-step workflow where you control every action, or build an autonomous agent that can think and decide for itself? This choice impacts everything from development time and cost to reliability and user experience. It is a fundamental decision that often determines if an AI application will be successful in production.

#   

#   By the end of this lesson, you will understand the fundamental differences between LLM workflows and AI agents, know when to use each, and recognize how to combine their strengths in hybrid approaches.

#   

#   ## Understanding the Spectrum: From Workflows to Agents

#   

#   To make the right choice, you first need to understand what LLM workflows and AI agents are. We will look at their core properties and how they are used, rather than their technical specifics.

#   

#   ### LLM Workflows

#   

#   An LLM workflow is a sequence of tasks orchestrated by developer-written code. It can include LLM calls, but also other operations like reading from a database or calling an API. Think of it like a recipe where each step is explicitly defined. The key characteristic is that the path is determined in advance, resulting in a deterministic or rule-based system. This gives you predictable execution, explicit control over the application's flow, and makes the system easier to test and debug. Because you control every step, you know exactly where a failure occurred and how to fix it.

#   

#   ```mermaid

#   graph TD

#       A["Start"] --> B["LLM Call"]

#       B --> C["Process Data"]

#       C --> D["Store Data"]

#       D --> E["End"]

#   ```

#   Image 1: A flowchart illustrating a deterministic LLM workflow with clear start and end points, including an LLM call and data operations.

#   

#   ### AI Agents

#   

#   AI agents are systems where an LLM dynamically decides the sequence of steps, reasoning, and actions to achieve a goal. The path is not predefined. Instead, the agent uses a reasoning process to plan its actions based on the task and the current state of its environment. This process is often modeled on frameworks like ReAct (Reason, Act, Observe). This allows agents to be adaptive and capable of handling new or unexpected situations through LLM-driven autonomy. They can select tools, execute actions, evaluate the outcomes, and correct their course until the goal is achieved [[1]](https://www.youtube.com/watch?v=kQxr-uOxw2o&t=1s).

#   

#   ```mermaid

#   graph TD

#       A["Start"] --> B["Agent (LLM) Receives Goal"]

#       B --> C["Plan/Reason (LLM)"]

#       C --> D["Select Tool"]

#       D --> E["Execute Action (Tool Call)"]

#       E --> F["Observe Environment/Feedback"]

#       F --> G{"Evaluate Outcome"}

#       G -->|"Satisfactory"| H["Stop/Achieve Goal"]

#       G -->|"Needs Adjustment"| C

#   ```

#   Image 2: Flowchart illustrating an AI agent's dynamic decis...

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Now run the article reviewer:
"""

model = get_model(SupportedModels.GOOGLE_GEMINI_25_FLASH)
reviewer = ArticleReviewer(
    to_review=article,
    article_guideline=article_guideline,
    article_profiles=article_profiles,
    model=model,
)

print("Reviewing article...")
article_reviews = await reviewer.ainvoke()

pretty_print.wrapped(f"Generated {len(article_reviews.reviews)} reviews:", title="Article reviews")
for i, review in enumerate(article_reviews.reviews, 1):
    review_dict = {
        "Profile": review.profile,
        "Location": review.location,
        "Comment": review.comment[:200] + "..." if len(review.comment) > 200 else review.comment,
    }
    pretty_print.wrapped(review_dict, title=f"Review {i}")
# Output:
#   Reviewing article...

#   [93m----------------------------------------- Article reviews -----------------------------------------[0m

#     Generated 49 reviews:

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m--------------------------------------------- Review 1 ---------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "Article level - Outline",

#     "Comment": "The article's main title is 'Workflows vs. Agents: The Critical Decision Every AI Engineer Faces'. However, the guideline specifies the title for Section 1 as 'Introduction: The Critical Decision Ever..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m--------------------------------------------- Review 2 ---------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "Introduction: The Critical Decision Every AI Engineer Faces - First paragraph",

#     "Comment": "The article's introduction contains 84 words, which is within the 100-word limit. However, the guideline specifies the content for the 'The Problem' part of the introduction. The article's content lar..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m--------------------------------------------- Review 3 ---------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "Understanding the Spectrum: From Workflows to Agents - First paragraph",

#     "Comment": "The article states 'To make the right choice, you first need to understand what LLM workflows and AI agents are. We will look at their core properties and how they are used, rather than their technica..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m--------------------------------------------- Review 4 ---------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "LLM Workflows - First paragraph",

#     "Comment": "The article's definition of LLM workflows ('An LLM workflow is a sequence of tasks orchestrated by developer-written code. It can include LLM calls, but also other operations like reading from a datab..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m--------------------------------------------- Review 5 ---------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "LLM Workflows - Mermaid diagram",

#     "Comment": "The article includes a Mermaid diagram for LLM Workflows. The guideline specifically requests 'Add a Mermaid diagram' for LLM workflows. The diagram content seems appropriate."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m--------------------------------------------- Review 6 ---------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "AI Agents - First paragraph",

#     "Comment": "The article's definition of AI agents ('AI agents are systems where an LLM dynamically decides the sequence of steps, reasoning, and actions to achieve a goal. The path is not predefined.') aligns wit..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m--------------------------------------------- Review 7 ---------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "AI Agents - Mermaid diagram",

#     "Comment": "The article includes a Mermaid diagram for AI Agents. The guideline specifically requests 'Add a Mermaid diagram' for AI agents. The diagram content seems appropriate."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m--------------------------------------------- Review 8 ---------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "Understanding the Spectrum: From Workflows to Agents - Article level",

#     "Comment": "The section 'Understanding the Spectrum: From Workflows to Agents' contains approximately 270 words (excluding diagram code and captions), which exceeds the guideline's specified length of '200 words ..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m--------------------------------------------- Review 9 ---------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "Choosing Your Path - First paragraph",

#     "Comment": "The article correctly introduces the core difference: 'The core difference between these two approaches lies in a single trade-off: developer-defined logic versus LLM-driven autonomy'. This aligns wit..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 10 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "Choosing Your Path - Image 3",

#     "Comment": "The article includes an image 'Image 3: The trade-off between an agent's level of control and application reliability.' with a link and proper citation. The guideline requests to 'Attach an image from..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 11 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "When to use LLM workflows - Examples",

#     "Comment": "The article provides examples for LLM workflows: 'Workflows are ideal for repeatable tasks with defined steps, like data extraction, report generation, or content repurposing.' The guideline's example..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 12 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "When to use LLM workflows - Strengths",

#     "Comment": "The article lists strengths as 'predictability, ensuring reliable results, easier debugging, and lower costs by using specialized models.' The guideline includes 'potentially lower operational costs a..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 13 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "When to use LLM workflows - Weaknesses",

#     "Comment": "The article mentions 'rigidity; they cannot handle unexpected scenarios, and adding features can become complex.' The guideline details more weaknesses: 'Potentially more development time required as ..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 14 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "When to use AI agents - Examples",

#     "Comment": "The article gives examples: 'Agents excel at dynamic problem-solving like open-ended research or complex customer support'. The guideline's examples are more detailed: 'Open-ended research and synthes..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 15 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "When to use AI agents - Strengths",

#     "Comment": "The article states the strength as 'flexibility in handling ambiguity.' The guideline also adds 'Adaptability to new situations and the flexibility to handle ambiguity and complexity as the steps are ..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 16 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "When to use AI agents - Weaknesses",

#     "Comment": "The article covers several weaknesses: 'less reliable, harder to debug, and costlier due to non-deterministic behavior. Without proper guardrails, they also pose security risks, especially with operat..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 17 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "Hybrid Approaches - First paragraph",

#     "Comment": "The article describes hybrid approaches: 'Most real-world systems are not purely one or the other. They often blend elements of both, creating a hybrid system. A common pattern is to use a workflow fo..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 18 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "Hybrid Approaches - Mermaid diagram",

#     "Comment": "The article includes a Mermaid diagram for a 'AI generation and human verification loop with iterative refinement.' The guideline specifically requests to 'Generate a mermaid to illustrate the AI gene..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 19 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "Choosing Your Path - Article level",

#     "Comment": "The section 'Choosing Your Path' contains approximately 460 words (excluding the image and diagram code). The guideline specifies a length of '200 words'. This section significantly exceeds the word l..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 20 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "The Challenges of Every AI Engineer - First paragraph",

#     "Comment": "The article states: 'Understanding the spectrum from workflows to agents is a core part of AI engineering. This choice helps determine if your application will succeed in production.' This aligns with..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 21 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "The Challenges of Every AI Engineer - Daily challenges",

#     "Comment": "The article presents daily challenges: 'building pipelines to pull information from Slack, web APIs, SQL databases, and data lakes; managing the cost-performance trap where sophisticated agents become..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 22 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "The Challenges of Every AI Engineer - Transition to next lesson",

#     "Comment": "The article concludes: 'In our next lesson, we will explore a foundational skill for building both workflows and agents: context engineering.' This aligns perfectly with the guideline's instruction to..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 23 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "The Challenges of Every AI Engineer - Article level",

#     "Comment": "The section 'The Challenges of Every AI Engineer' contains approximately 105 words. The guideline specifies a length of '100 words'. This is a minor deviation, but it would be ideal to be within or ve..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 24 --------------------------------------------[0m

#     {

#     "Profile": "article_guideline",

#     "Location": "References - Golden Sources",

#     "Comment": "The article includes a reference to 'Google. (n.d.). Gemini CLI. GitHub. https://github.com/google-gemini/gemini-cli/blob/main/README.md' which is listed as a Golden Source in the article guideline, b..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 25 --------------------------------------------[0m

#     {

#     "Profile": "character_profile",

#     "Location": "Article level",

#     "Comment": "The article maintains a professional, informative, and direct tone, which aligns with the 'Paul Iusztin' character profile. It avoids excessive hype and focuses on technical explanations."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 26 --------------------------------------------[0m

#     {

#     "Profile": "terminology_profile",

#     "Location": "Article level",

#     "Comment": "The article generally uses simple, direct language. However, it uses the phrase 'critical decision with confidence' in the title's subtitle and the introductory paragraph 'critical architectural decis..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 27 --------------------------------------------[0m

#     {

#     "Profile": "terminology_profile",

#     "Location": "Introduction: The Critical Decision Every AI Engineer Faces - First paragraph",

#     "Comment": "The phrase 'critical architectural decision' is used, which is close to the banned 'critical decision with confidence'. While not exactly the same, it shares the word 'critical' and implies a similar ..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 28 --------------------------------------------[0m

#     {

#     "Profile": "terminology_profile",

#     "Location": "Choosing Your Path - First paragraph",

#     "Comment": "The sentence 'The core difference between these two approaches lies in a single trade-off: developer-defined logic versus LLM-driven autonomy' uses the word 'lies', which is not explicitly banned but ..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 29 --------------------------------------------[0m

#     {

#     "Profile": "terminology_profile",

#     "Location": "When to use AI agents - First paragraph",

#     "Comment": "The article states 'Agents excel at dynamic problem-solving'. The word 'excel' is not on the banned list, but it leans towards marketing-y or exaggerated language that the terminology profile aims to ..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 30 --------------------------------------------[0m

#     {

#     "Profile": "tonality_profile",

#     "Location": "Article level",

#     "Comment": "The article maintains a conversational yet authoritative voice, using plain English with precise technical nouns, which aligns with the tonality profile. It does not sound like marketing a product rel..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 31 --------------------------------------------[0m

#     {

#     "Profile": "mechanics_profile",

#     "Location": "Workflows vs. Agents: The Critical Decision Every AI Engineer Faces - Subtitle",

#     "Comment": "The subtitle 'How to choose between predictable control and autonomous flexibility when building AI applications.' is correctly not capitalized, aligning with the 'Capitalization patterns' rule."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 32 --------------------------------------------[0m

#     {

#     "Profile": "mechanics_profile",

#     "Location": "Workflows vs. Agents: The Critical Decision Every AI Engineer Faces - Title",

#     "Comment": "The main title 'Workflows vs. Agents: The Critical Decision Every AI Engineer Faces' is capitalized as per the 'Capitalization patterns' rule."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 33 --------------------------------------------[0m

#     {

#     "Profile": "mechanics_profile",

#     "Location": "Understanding the Spectrum: From Workflows to Agents - First paragraph",

#     "Comment": "The article uses 'you' and 'we' consistently with the 'Point of View' rule, addressing the reader directly and using 'we' for the team/authors."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 34 --------------------------------------------[0m

#     {

#     "Profile": "mechanics_profile",

#     "Location": "AI Agents - First paragraph",

#     "Comment": "The article uses the citation format '[[1]](https://www.youtube.com/watch?v=kQxr-uOxw2o&t=1s)' at the end of a sentence before the period, which generally follows the 'Citation Rules'. However, the id..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 35 --------------------------------------------[0m

#     {

#     "Profile": "mechanics_profile",

#     "Location": "Choosing Your Path - First paragraph",

#     "Comment": "The article uses citations '[[2]](https://decodingml.substack.com/p/llmops-for-production-agentic-rag), [[3]](https://towardsdatascience.com/a-developers-guide-to-building-scalable-ai-workflows-vs-age..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 36 --------------------------------------------[0m

#     {

#     "Profile": "mechanics_profile",

#     "Location": "References - Article level",

#     "Comment": "The references section includes '4. Google. (n.d.). Gemini CLI. GitHub. https://github.com/google-gemini/gemini-cli/blob/main/README.md'. This source is listed in the <article_guideline> Golden Source..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 37 --------------------------------------------[0m

#     {

#     "Profile": "mechanics_profile",

#     "Location": "References - Article level",

#     "Comment": "The numbering of references in the 'References' section does not align with the provided Golden Sources. The golden sources are 1, 2, 3, 4. The article has 1, 2, 3, 4. However, the content of referenc..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 38 --------------------------------------------[0m

#     {

#     "Profile": "mechanics_profile",

#     "Location": "References - Article level",

#     "Comment": "The 'References' section lists citations 1-4. The formatting for the reference '3. (n.d.). A developer\u2019s guide to building scalable AI: Workflows vs agents. Towards Data Science. https://towardsdatasc..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 39 --------------------------------------------[0m

#     {

#     "Profile": "mechanics_profile",

#     "Location": "References - Article level",

#     "Comment": "The 'References' section lists citations 1-4. The article uses a mix of author names and '(n.d.)' for titles or dates. The `references_rules` state: 'If the author, publish date or full title is missi..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 40 --------------------------------------------[0m

#     {

#     "Profile": "structure_profile",

#     "Location": "Introduction: The Critical Decision Every AI Engineer Faces - First paragraph",

#     "Comment": "The introduction's first paragraph has 84 words. The 'Sentence and paragraph length patterns' rule states 'Keep paragraphs \u2264 80 words'. This paragraph slightly exceeds the limit. It should be condense..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 41 --------------------------------------------[0m

#     {

#     "Profile": "structure_profile",

#     "Location": "Understanding the Spectrum: From Workflows to Agents - Article level",

#     "Comment": "The section 'Understanding the Spectrum: From Workflows to Agents' has 270 words (excluding diagrams), which exceeds the 80-word paragraph rule in some of its sub-paragraphs, and also the overall sect..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 42 --------------------------------------------[0m

#     {

#     "Profile": "structure_profile",

#     "Location": "When to use AI agents - Second paragraph",

#     "Comment": "The paragraph discussing the weaknesses of AI agents ('Agents excel at dynamic problem-solving like open-ended research or complex customer support [[3]](https://towardsdatascience.com/a-developers-gu..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 43 --------------------------------------------[0m

#     {

#     "Profile": "structure_profile",

#     "Location": "Choosing Your Path - Article level",

#     "Comment": "The section 'Choosing Your Path' contains 460 words (excluding image/diagrams), significantly exceeding the article guideline's 200-word limit. Additionally, individual paragraphs within this section ..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 44 --------------------------------------------[0m

#     {

#     "Profile": "structure_profile",

#     "Location": "Image 1, Image 2, Image 3, Image 4",

#     "Comment": "The image captions follow the specified format, including 'Image <media_identifier>: <diagram description>' or 'Image <media_identifier>: <image description> (Image by <author_name> from [<citation_na..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 45 --------------------------------------------[0m

#     {

#     "Profile": "article_profile",

#     "Location": "Article level",

#     "Comment": "The article is structured with an introduction, sections, and a conclusion, using H2 and H3 headers, which aligns with the 'Article Template' and 'Sections Sub-Heading Formatting' rules. The flow betw..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 46 --------------------------------------------[0m

#     {

#     "Profile": "article_profile",

#     "Location": "Introduction: The Critical Decision Every AI Engineer Faces - First paragraph",

#     "Comment": "The introduction covers the 'why' and 'what' by presenting the problem and its impacts, aiming to captivate the reader. It sets the stage for the rest of the article as per the 'Introduction Guideline..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 47 --------------------------------------------[0m

#     {

#     "Profile": "article_profile",

#     "Location": "The Challenges of Every AI Engineer - First paragraph",

#     "Comment": "The conclusion provides a short wrap-up of what was learned and connects to future lessons, aligning with the 'Conclusion Guidelines'. It reinforces the core ideas and transitions to the bigger pictur..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 48 --------------------------------------------[0m

#     {

#     "Profile": "article_profile",

#     "Location": "References - Article level",

#     "Comment": "The 'References' section is present at the end of the article, listing all sources. This aligns with the 'General Article Structure' and 'References Rules'."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------------------------- Review 49 --------------------------------------------[0m

#     {

#     "Profile": "abbreviation_acronym_rules",

#     "Location": "Article level",

#     "Comment": "The acronyms LLM and AI are used without expansion throughout the article, which aligns with the `abbreviations_or_acronyms_never_to_expand_rules`."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
### Example: Reviewing Selected Text

Now let's review only a specific section of the article:

"""

from brown.entities.articles import SelectedText

# Let's extract a specific section to review
article_lines = article.content.split("\n")
first_line_number = 11
last_line_number = 44
selected_content = "\n".join(article_lines[first_line_number:last_line_number])

selected_text = SelectedText(
    article=article,
    content=selected_content,
    first_line_number=first_line_number,
    last_line_number=last_line_number,
)

cropped_selected_text = selected_text.model_copy()
cropped_selected_text.content = f"{cropped_selected_text.content[:1500]}\n..."

text = [
    f"Selected text: {len(selected_content)} characters",
    f"Lines: {selected_text.first_line_number}-{selected_text.last_line_number}",
]
pretty_print.wrapped("\n".join(text), title="Selected text to review")
pretty_print.wrapped(cropped_selected_text.to_context(), title="Selected text context (first 1500 characters)")
# Output:
#   [93m------------------------------------- Selected text to review -------------------------------------[0m

#     Selected text: 2414 characters

#   Lines: 11-44

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m-------------------------- Selected text context (first 1500 characters) --------------------------[0m

#     

#   <selected_text>

#       

#       <content>### LLM Workflows

#   

#   An LLM workflow is a sequence of tasks orchestrated by developer-written code. It can include LLM calls, but also other operations like reading from a database or calling an API. Think of it like a recipe where each step is explicitly defined. The key characteristic is that the path is determined in advance, resulting in a deterministic or rule-based system. This gives you predictable execution, explicit control over the application's flow, and makes the system easier to test and debug. Because you control every step, you know exactly where a failure occurred and how to fix it.

#   

#   ```mermaid

#   graph TD

#       A["Start"] --> B["LLM Call"]

#       B --> C["Process Data"]

#       C --> D["Store Data"]

#       D --> E["End"]

#   ```

#   Image 1: A flowchart illustrating a deterministic LLM workflow with clear start and end points, including an LLM call and data operations.

#   

#   ### AI Agents

#   

#   AI agents are systems where an 

[... Content truncated due to length ...]

</details>


## YouTube Video Transcripts

<details>
<summary>Here's a detailed transcript of the video:</summary>

Here's a detailed transcript of the video:

[00:00] (The screen is black)

[00:01] (A title slide appears: "Prompting 101" with "Code w/ Claude" in the top right corner. Below the title, two names are listed with small profile pictures: "Hannah Moran, Applied AI, Anthropic" and "Christian Ryan, Applied AI, Anthropic". A black staircase graphic is on the right side of the slide.)

[00:05] **Hannah Moran:** Hi everyone, thank you for joining us today for Prompting 101. (Hannah and Christian stand behind a white podium with a laptop.) My name is Hannah, I'm part of the Applied AI team here at Anthropic, and with me is Christian, also part of the Applied AI team. And what we're going to do today is take you through a little bit of prompting best practices, and we're going to use a real-world scenario and build up a prompt together.

[00:25] (The slide changes to: "What is 'prompt engineering' anyway?" The definition reads: "The practice of systematically improving prompts for LLM applications - through testing, evaluation, analysis, and optimization of prompts." Below are "Skills involved:" in a 3x3 grid: "Programming in natural language", "Clear, unambiguous, precise writing", "Conceptual engineering", "Creating evals with a scientific mindset", "Product thinking - what is the ideal model behavior for your product?", "Testing", "Understanding the LLMs", "Aggregating and analyzing failure modes + thinking of ways to fix", "Making LLMs scale to a wide range of inputs".)

[00:25] **Hannah Moran:** Uh, so a little bit about what prompt engineering is. Uh, prompt engineering, you're all probably a little bit familiar with this. This is the way that we communicate with a language model and try to get it to do what we want. So this is the practice of writing clear instructions for the model, giving the model the context that it needs to complete the task, and thinking through how we want to arrange that information in order to get the best result. Um, so there's a lot of detail here, a lot of different ways you might want to think about building out a prompt.

[00:56] (The slide changes to: "Hands-on scenario".)

[00:56] **Hannah Moran:** Um, and as always, the best way to learn this is just to practice doing it. Um, so today we're going to go through a hands-on scenario. Um, we're going to use an example inspired by a real customer that we worked with. So we've modified what the actual customer asked us to do, but this is a really interesting case of trying to analyze some images and get factual information out of the images and have Claude make a judgment about what content it finds there. And I actually do not speak the language that this content is in, but luckily Christian and Claude both do. Um, so I'm going to pass it over to Christian to talk about the scenario and the content.

[01:28] (Christian takes the remote control from Hannah.)

[01:28] **Christian Ryan:** So, for this example that we have here, it's uh intended, so, so to set the stage, imagine you're working for a Swedish insurance company, and you deal with uh car insurance claims on a daily manner. Um, and the purpose of this is that you have two pieces of information. (The slide changes to: "Prompt Claude to analyze car accident reporting forms". Two images are displayed: a car accident report form with checkboxes and a hand-drawn sketch of a car accident at an intersection with Swedish labels.) We're going to these in details well, but visually, you can see on the left-hand side, we have an a car accident report form, um, just detailing out what transpired before the accident accident actually took place. And then finally, we have a sort of human-drawn um, uh sketch of how the accident took place as well. So these two pieces of information is what we're going to try to pass on to Claude.

[02:06] (The slide changes to: "What if we just throw it into the console?" The car accident form and sketch are still displayed, with the prompt "Please analyze the images and tell me what happened".)

[02:06] **Christian Ryan:** And to begin with, we could just take these two and throw them into the console and just see what what happens. So if we transition over to your console as well, we can actually do this in a real manner. (The video cuts to a screen recording of the Anthropic Console. The prompt "You are an AI assistant who reviews accident reporting forms. You should review the form and the drawing, and determine what happened in the accident and who's at fault." is visible in the 'User' input field. Two image files, "form-numbered.jpg" and "IMG_8157.jpg", are uploaded below the prompt.) In this case here, you can see we have our shiny, beautiful Anthropic console. We're using the new Claude 4 Sonnet model as well. (Christian opens the model settings, showing "claude-sonnet-4-20250514 latest", "Temperature: 0", "Max tokens: 2000".) In this case, setting temperature to zero, and having a a huge max token budget as well, so it's helping us make sure that there's no limitations to what Claude can do. In this case, you can see I have a very simple prompt just setting the stage or Claude's supposed to do, in this case, mentioning that this is um intended to review it an accident report form. Uh, and eventually also determine um, what happened in the accident and who's at fault. So you can see here, with this very simple prompt, if I just run this, (Christian clicks 'Run', and the response starts generating in the 'Response' preview panel.) let me go to preview. Uh, (The generated response appears, starting with: "Based on my review of this Swedish accident report form and diagram, here's what happened: What Happened: This was a skiing accident that occurred on "Köpmangatan" [a street/slope name]. According to the form selections: - Person A was stationary/standing still ["parkerade/stannade till"] - marked with a checkmark - Person B was turning to the right ["tog av till höger"] - marked with a checkmark The diagram shows Person A was stationary on the slope when Person B, who was skiing and making a right turn, collided with them. Fault Determination: Person B (the skier who was turning) is at fault for this accident. Reasoning: 1. In skiing, there is a a fundamental rule that the skier who is uphill or approaching from behind has the responsibility to avoid collisions with skiers below or in front of them.")

[02:51] **Christian Ryan:** We can see here that Claude thinks that this is in relation to skiing accident, that happened on a street called Köpmangatan. It's a very common street in Sweden. Um, and in many ways, you can sort of understand this innocent mistake, in a sense that in our prompt, we actually haven't done anything to set the stage on what is actually taking place here. So, his sort of first guess is not too bad, but we still notice a lot of intuition that we can bake into Claude.

[03:22] (The video cuts back to the presentation slide: "Let's take a step back and do some prompt engineering!" with "Prompting is an empirical science: always test your prompts & iterate often!" Below is a flowchart: "Engineer preliminary prompt" -> "Test prompt against cases" -> "Refine prompt".)

[03:22] **Christian Ryan:** So, we switch back to the slides. You can see here that, um, in many ways, prompt engineering is a very iterative empirical science. In this case here, we could almost have a test case where Claude is supposed to make sure that it understands it's in a car or vehicular environment, nothing to do with skiing. Uh, and in that way, you iteratively build upon your prompt to make sure it's actually tackling the problem you're intending to solve. Um, and to do so, we go through some best practices of how we we at Anthropic break this down, uh, internally and how we recommend others to do so as well.

[03:56] (Hannah turns to Christian and they exchange the remote.)

[03:57] **Hannah Moran:** So, we're going to talk about some best practices for developing a great prompt.

[04:00] (The slide changes to: "Prompt structure". On the left, a numbered list: "1. 1-2 sentences to establish role and high-level task description", "2. Dynamic / retrieved content", "3. Detailed task instructions", "4. Examples / n-shot (optional)", "5. Repeat critical instructions (particularly useful for very long prompts)". On the right, a console screenshot shows a complex prompt for an "AI travel agent" with sections like `<location>`, `<num_days>`, `<user_preferences>`, and specific instructions for generating an itinerary.)

[04:00] **Hannah Moran:** Uh, first we want to talk a little bit about what a great prompt structure looks like. So, you might be familiar with kind of interacting with a chatbot, with Claude, going back and forth, having a more kind of conversational style interaction. When we're working with a task like this, we're probably using the API, and we kind of want to send one single message to Claude and have it nail the task the first time around without needing to, uh, kind of move back and forth. Uh, so the kind of structure that we recommend is setting the task description up front. So telling Claude, what are you here to do? What's your role? What task are you trying to accomplish today? Then we provide content. So in this case, it's the images that Christian was showing, the form and the drawing of the accident, and how they occurred. That's our dynamic content. This might also be something you're retrieving from another system, depending on what your use case is. We're going to give some detailed instructions to Claude. So almost like a step-by-step list of how we want Claude to go through the task and how we want it to, um, tackle the reasoning. We may give some examples to Claude. Here's an example of if some piece of content you might receive, here's how you should respond when given that content. And at the end, we usually recommend repeating anything that's really important for Claude to understand about this task.

[05:11] (The slide changes, refining the prompt structure to 10 points on the left side, with specific colors for each: "1. Task context" (red), "2. Tone context" (orange), "3. Background data, documents, and images" (green), "4. Detailed task description & rules" (light blue), "5. Examples" (blue), "6. Conversation history" (purple), "7. Immediate task description or request" (pink), "8. Thinking step by step / take a deep breath" (red), "9. Output formatting" (grey), "10. Prefilled response (if any)" (black). On the right is another console screenshot with a highly structured prompt, demonstrating these elements.)

[05:11] **Hannah Moran:** Kind of, uh, reviewing the information with Claude, emphasizing things that are extra critical, and then telling Claude, okay, go ahead and do your work. So here's another view. This has a little bit more detail, a little bit more of a breakdown. And we're going to walk through each of these 10 points individually and show you how we build this up, um, in the console. So the first couple things, um, Christian's going to talk about the task context and the tone context.

[05:37] **Christian Ryan:** Perfect. So yeah, if we begin with the task context. As you realized when I went through the little demo there, um, we didn't have much elaborating what what's the scenario Claude is actually working within. And because of that, you can also tell that Claude doesn't necessarily needs to guess a lot more on what you actually want from it. So in our case, we really want to break that down, make sure we can give more clear-cut instructions, um, and also make sure we understand what's the task that we're asking Claude to do. Um, secondly, as well, we also make sure we add a little bit of tone into it all. Um, key thing here is we want Claude to stay factual and to stay confident. So if, uh, Claude can't understand what it's looking at, we don't want it to guess and just sort of mislead us. We want to make sure that any assessment, and in our case, we want to make sure that we can understand who's at fault here. We want to make sure that that assessment is as clear and as confident as possible. If not, we're sort of losing track of what we're doing.

[06:29] (The video cuts back to the Anthropic console, showing version 1 of the prompt. Christian navigates to version 2 of the prompt.)

[06:29] **Christian Ryan:** So if we transition back to the to the console, um, we can jump to a V2 that we have here. So I'll just navigate to V2. (The console loads "Accident Report Form Analysis - v2". The 'User' prompt is now more detailed: "You are an AI assistant helping a human claims adjuster review car accident reporting documentation written in Swedish. Your task is to review a standard car accident report form as well as human-drawn sketch of the incident, and determine what information you can confidently conclude from these documents. After that, you should be able to state whether or not you can fully confidently determine that one vehicle was clearly at fault or if the human adjuster needs to follow up for more information." The same two image files are attached. The response panel shows more structured information: "Information from the Documents: From the Standard Form: - This is section 12 of a standard accident report asking "How did the accident happen? Hur kom det sig att olyckan hände?" - Vehicle A has marked option 1: "parkerade / stannade till" (was parking/stopping) - Vehicle B has marked option 12: "tog av till höger" (turned off to the right). From the Sketch: - Shows an intersection with "KÖPMANGATAN" (a street name) - Vehicle A is shown stationary ("still stående") - Vehicle B is shown with an arrow indicating movement/turning. - The collision point appears to be where Vehicle B was turning right while Vehicle A was stationary. Analysis: This appears to be a collision where: - Vehicle A was in the process of parking or had stopped - Vehicle B was making a right turn - The accident occurred during Vehicle B's turning maneuver while Vehicle A was stationary. Fault Determination: I cannot fully confidently determine clear fault based solely on these documents. Here's why additional investigation is needed: 1. Missing critical details: The exact positioning of vehicles, traffic control devices, right-of-way rules, and the specific circumstances of the parking/stopping maneuver are unclear. 2. Incomplete visuals: While Vehicle B was turning right, the diagram doesn't provide enough information to definitively establish fault.")

[06:38] **Christian Ryan:** And you can see here, um, I'll also just illustrate the data because we didn't really do that last time around, just to really highlight what we're looking at. So, what we're seeing here, this is the car accident report form. (Christian clicks on 'form-numbered.jpg', showing a closer view of the Swedish car accident report form with checkboxes for different actions of 'A' and 'B'.) And it's just 17 different checkboxes going through what actually happened. You can see there's a vehicle A and vehicle B, both on the left and right-hand side. And the main purpose of this is that we want to make sure that Claude can understand this manually generated data to assess what's actually going on. And that is, uh, corroborated by, (Christian closes the image and then clicks on 'IMG_8157.jpg', showing a closer view of the hand-drawn sketch of the accident.) if I navigate back here, to the sketch that we can highlight here as well. In this case, the form is just a different, um, data point for the same scenario. Um, and in this case here, we want to bake in more information into our version two. Uh, and by doing so, I'm actually elaborating a lot more on what's going on. So you can see here, I'm specifying that this AI system is supposed to help a human claims adjuster review car accident reporting documentation written in Swedish as well. Um, you can see here, I'm also elaborating that it's a human-drawn sketch of the incident, and that it should not, um, make an assessment if it's not actually fully confident. And that's really key because if we run this, you'll see that, (Christian clicks 'Run' again, with the same model settings.) and you can see it has the same settings as well. Claude 4, our new shiny model, zero temperature as well. If we run this, we can see here what actually happens. In this case, um, Claude's able to pick up that, uh, now it's relating to car accidents, not skiing accidents, which is great. We can see it's able to pick up that vehicle A was marked on on checkbox one, and then vehicle B was on 12. Um, and if we scroll down, though, we can still tell that there's some information missing for Claude to make a fully confident determination of who's at fault here. And this is great. This is pertaining to the task that we've set. Make sure you don't make anything, any claims that aren't, um, uh factual and make sure you you only cert things when you're when you're confident.

[08:24] (The video cuts back to the presentation stage, with the Anthropic Console displayed on the screen.)

[08:24] **Christian Ryan:** But there's a lot of information we're still missing here, um, regarding the form, uh, what the form actually entails, and a lot of that information is what we want to want to bake into this LLM application as well. And the best way of doing so is actually adding it to the system prompt, which Hannah will elaborate on.

[08:46] (The slide updates, highlighting "3. Background data, documents, and images" in green.)

[08:49] **Hannah Moran:** Um, so back in the slides, uh, we have the next item we're going to add to the prompt, and this is, um, background, detailed data, documents, and images. And here, as Christian was saying, we actually know a lot about this form. The form is going to be the same every single time. The form will never change. And so this is a really great type of information to provide to Claude to tell Claude, here's the structure of the form you'll be looking at. We know that will not ever alter between different queries. The way the form is filled out will change, but the form itself is not going to change. And so this is a great type of, um, information to put into the system prompt. Also a great thing to use prompt caching for. If you're considering using prompt caching, this will always be the same. And what this will help Claude do is spend less time trying to figure out what the form is the first time it sees the form each time. And it's going to do a better job of reading the form because it already knows, um, what to expect there.

[09:41] (The slide changes to: ">> how to organize information in your prompts". On the left: "Disorganized prompts are hard for Claude to comprehend", "Use delimiters like XML tags to organize", "Just like section titles and headers help humans better follow information, using XML tags (<tag></tag>) helps Claude understand the prompt's structure", "Claude understands all types of delimiters; we just prefer XML because its boundaries are clear and it's token efficient". On the right, a console screenshot shows a prompt example using XML tags for 'location', 'num_days', 'user_preferences', and 'itinerary'.)

[09:41] **Hannah Moran:** So, another thing I want to touch on here is how we like to organize information in prompts. So, Claude really loves structure, loves organization. That's why we recommend following kind of a standard structure in your prompts. And there's a couple other tools you can use to help Claude understand the information better. I also just want to mention all of this is in our docs with a lot of really great examples. So, definitely take pictures, but if you forget to take a picture, don't worry. All of this content is online with lots of examples and definitely encourage you guys to check it out there too. Um, anyway, the, uh, so some things you can use, delimiters, like XML tags. Also Markdown is pretty useful to Claude, but XML tags are nice because you can actually specify what's inside those tags. So we can tell Claude, here's, here's user preferences. Now you're going to read some content, and these XML tags are letting you know that everything wrapped in those tags is related to the user's preferences. And it helps Claude refer back to that information, maybe at later points in the prompt.

[10:46] (The video cuts back to the stage, Hannah and Christian are at the podium.)

[10:46] **Hannah Moran:** Um, so I want to show in the, back in the console, how we actually do this in this case. And Christian's going to pull up our version three. (The console loads "Accident Report Form Analysis - v3". The 'System Prompt' now includes detailed information about the form's structure, driving actions, completion rules, meaning, and interpretation, all wrapped in XML-like tags. For example, `<form_structure>` contains details about columns, rows, and headers. `<form_completion_rules>` describes how humans fill out the form. `<form_meaning>` and `<form_interpretation>` provide context and instructions for AI analysis. The user prompt and image files remain the same. The response in the preview panel is similar to v2, but with improved analysis and conclusion.) So we're keeping everything about the other part of the user prompt the same. And we've decided, in this case, to put this information in the system prompt. You could try this different ways. Um, we're doing it in the system prompt here. And we're going to tell Claude everything it needs to know about this form. So this is a Swedish car accident form, the form will be in Swedish. It'll have this title. It'll have two columns, the columns represent different vehicles. We'll tell Claude about each of the 17 rows, and what they mean. You might have noticed when we ran it before, Claude was reading individually each of the lines to understand what they are. We can provide all of that information up front. And we're also going to give Claude a little bit of information about how this form should be filled out. This is also really useful for Claude. We can tell it things like, you know, humans are filling this form out, basically. So it's not going to be perfect. People might put a circle. They might scribble. They might not put an X in the box. There could be many types of markings that you need to look for when you're reading this form. We can also give Claude a little bit of information about how to interpret this or what the purpose or meaning of this form is. And all of this is context that is hopefully really going to help Claude, um, do a better job analyzing the form.

[11:59] (Christian runs the v3 prompt.)

[11:59] **Hannah Moran:** So if we run it, everything else is still the same. So we've kept the same user prompt down here. Oh, your scroll is backwards from mine. Uh, the we have the same user prompt here, still asking Claude to do the same task, same context. And we'll see here that it's spending less time, it's kind of narrating to us a little bit less about what the form is because it already knows what that is, and it's not concerned with kind of bringing us that information back. It's going to give us a whole list of what it found to be checked, what the sketch shows. And here Claude is now becoming much more confident. With this additional context that we gave to Claude, Claude now feels it's appropriate to say vehicle B was at fault in this case, based on this drawing and based on this sketch. So already we're seeing some improvement in the way Claude is analyzing these. I think we could probably all agree if we looked at the drawing and at the list that vehicle B is at fault. Um, so we'd like to see that.

[12:55] (Hannah gestures towards Christian and they exchange the remote.)

[12:55] **Hannah Moran:** Uh, so we're going to go back to the slides and talk about a couple of other items that we're not really using in this prompt, but can be really helpful to building up a, building up your prompt and making it work better.

[13:08] (The slide updates, highlighting "6. Conversation history" in purple.)

[13:08] **Christian Ryan:** Indeed, thank you so much. So, as Hannah mentioned, uh, we sort of set the stage in this prompt to make sure that Claude's really acting on our behalf in a in a right manner. Um, and a key step that we also add towards the end of this prompt that I'm going to show you in a second is a simple sort of guidelines or reminder part as well. Just strengthening and reinforcing exactly what we want to get out of it. And one important piece is actually output formatting. You can imagine if you're a data engineer working on this LLM application, all this sort of fancy preamble is great, but at the end of the day, you want your piece of information to to be stored in, let's say, your SQL database, wherever you want to store that data, and the rest of it that is necessary for Claude to sort of give its verdict, isn't really that necessary for your application. You want the nitty-gritty information for your application.

[13:21] (The slide updates, highlighting "5. Examples" in blue.)

[13:21] **Christian Ryan:** Indeed. Thank you so much. So, as Hannah mentioned, uh, we sort of set the stage in this prompt to make sure that Claude's really acting on our behalf in a in a right manner. Um, and a key step that we also add towards the end of this prompt that I'm going to show you in a second is a simple sort of guidelines or reminder part as well. Just strengthening and reinforcing exactly what we want to get out of it. And one important piece is actually output formatting. You can imagine if you're a data engineer working on this LLM application, all this sort of fancy preamble is great, but at the end of the day, you want your piece of information to to be stored in, let's say, your SQL database, wherever you want to store that data, and the rest of it that is necessary for Claude to sort of give its verdict, isn't really that necessary for your application. You want the nitty-gritty information for your application.

[13:25] _Summary of section on prompt structure: Hannah and Christian explain a 10-point prompt structure for API interactions, emphasizing setting context (task, tone), providing static and dynamic content (background data, images), detailed instructions, examples, conversation history, immediate requests, structured thinking, and specific output formatting to guide Claude for optimal results._

[13:25] (The slide updates, highlighting "6. Conversation history" in purple.)

[13:25] **Christian Ryan:** You can see here as well, this is just a little example of how we do this. Again, really emphasizing the sort of XML structure that we we, uh, we enjoy, it gives a lot of structure to Claude, it's what it's been fine-tuned on as well. Um, and it works perfectly well for this example. In our case, we're not doing this just because it's a simple demo, but you can realistically imagine if you were building this for an insurance company, you would have tens, maybe even hundreds of examples that are quite difficult, maybe in the gray, that you'd like to make sure that Claude actually has some basis in to make the verdict next time. Um, another topic we really want to highlight, which we're not doing in this demo, is conversation history. It's in the same vein as examples. Uh, we use this to make sure that the enough context-rich information is at Claude's disposal when it when when Claude's working on on on your behalf. Um, in our case now, this isn't really a user-facing LLM application. It's more something happening in the background. You could imagine for this insurance company, they have this automated system, some data is generated out of this, and then you might have a human in the loop at towards the end. If you were to build something much more user-facing where you'd have a long conversation history that would be, um, relevant to bring in, this is a perfect place in the system prompt to include because it enriches the context that Claude works within. Um, in our case, we haven't done so. But what we do is in the next step is try to make sure we give a concrete reminder of the task at hand.

[15:55] (Christian hands the remote back to Hannah.)

[15:55] **Hannah Moran:** So now we're going to build out the final part of this prompt for Claude, and that's coming back to the reminder of what the immediate task is and giving Claude a reminder about any important guidelines that we want it to follow.

[16:09] (The slide updates, highlighting "7. Immediate task description or request" in pink.)

[16:09] **Hannah Moran:** Some reasons that we may do this are, A, preventing hallucination. (The slide changes to: ">> preventing hallucinations". Four boxes list troubleshooting tips: "Have Claude say "I don't know" if it doesn't know", "Tell Claude to answer only if it is very confident in its response", "Have Claude think before answering", "Ask Claude to find relevant quotes from long documents then answer using the quotes".) Um, so we want Claude to, uh, not invent details that it's not finding in this prompt, right? Or not finding in the data. If Claude can't tell which form is checked, we don't want Claude to take its best guess or invent the idea that a box might be checked when it's not. If the sketch is unintelligible, the person did a really bad job drawing this drawing, and even a human would not be able to figure it out, we want Claude to be able to say that. And so these are some of the things we'll include in this final reminder and kind of wrap-up step for Claude.

[16:36] (The video cuts back to the stage.)

[16:36] **Hannah Moran:** Um, remind it to do things like answer only if it's very confident. We could even ask it to refer back to what it has seen in the form, anytime it's making a factual claim. So if it wants to say vehicle B turned right, it should say, I know this based on the fact that box 2 is clearly checked or whatever it might be. We can kind of give Claude some guidelines about that. So if we go back to the console, (The console loads "Accident Report Form Analysis - v4". The system prompt now includes a `tasks` section with several ordered sub-tasks: `task id=1` "Examine the image...", `task id=2` "Provide your step-by-step analysis...", `task id=3` "Based on your analysis, create a summary...", `task id=4` "Next, review the sketch...". The output in the response panel is detailed for each task.) we can see the next version of the prompt. And we're going to keep, uh, we're going to keep everything the same here in the system prompt. So we're not changing any of that background context that we gave to Claude about the form, about how it's going to fill it all out. We're not changing anything else about the context and the role. We're just adding this detailed list of tasks, and this is how we want Claude to go about analyzing this. And a really key thing that we found here, as we were building this demo, and when we were working on the customer example, is that the order in which Claude analyzes this information is very important. And this is analogous to way you might think about doing this if you were a human. You would probably not look at the drawing first and try to understand what was going on, right? It's pretty unclear. It's a bunch of boxes and lines. We don't really know what that drawing is supposed to mean without any additional context. But if we have the form, and we can read the form first and understand that we're talking about a car accident, and that we're seeing some checkboxes that indicate what vehicles were doing at certain times, then we know a little bit more about how to understand what might be in the drawing. And so that's the kind of detail that we're going to give Claude here is to say, hey, first go look at the form. Look at it very carefully, make sure you can tell what boxes are checked, make sure you're not missing anything here, um, make a list for yourself of what you see in that, and then move on to the sketch. So after you've kind of confidently gotten information out of the form and you can say what's factually true, then you can go on and think about what you can gain from that sketch, keeping in mind your understanding of the accident so far. So whatever you've learned from the form, you're trying to match that up with the sketch. And that's how you're going to arrive, um, at your final, uh, at your final assessment of the form.

[18:45] (Christian runs the v4 prompt.)

[18:45] **Hannah Moran:** And we'll run it. And here you can see one behavior that this produced for Claude. Because I told it to very carefully examine the form, it's showing me its work as it does that. So it's telling me each individual box, is the box checked? Is it not checked? And so this is one thing you'll notice as you do prompt engineering. In our previous prompts, we were kind of letting Claude decide how much it wanted to tell us about what it saw on the form. Here, because I've told it, carefully examine each and every box, it's very carefully examining each and every box. And that might not be what we want in the end, so that's something we might change. Um, but it's also going to give me these other things that I asked for in XML tags. So a nice analysis of the form, the accident summary so far. It's going to give me a sketch analysis, and it's going to continue to say that vehicle B appears to be clearly at fault. In this in this example, it's a pretty simple example. With more complicated drawings, more, uh, less clarity in the forms, this kind of step-by-step thinking for Claude is really impactful in its ability to make a correct assessment here.

[19:54] (The video cuts back to the stage.)

[19:54] **Hannah Moran:** Um, so I think we'll go back to the slides and Christian's going to talk about a last kind of piece that we might add to this to really make it useful for a real-world task.

[20:04] (The slide updates, highlighting "9. Output formatting" in grey.)

[20:04] **Christian Ryan:** Indeed, thank you so much. So, as Hannah mentioned, um, we sort of set the stage in this prompt to make sure that Claude's really acting on our behalf in a in a right manner. Um, and a key step that we also add towards the end of this prompt that I'm going to show you in a second is a simple sort of guidelines or reminder part as well. Just strengthening and reinforcing exactly what we want to get out of it. And one important piece is actually output formatting. You can imagine if you're a data engineer working on this LLM application, all this sort of fancy preamble is great, but at the end of the day, you want your piece of information to to be stored in, let's say, your SQL database, wherever you want to store that data, and the rest of it that is necessary for Claude to sort of give its verdict, isn't really that necessary for your application. You want the nitty-gritty information for your application.

[20:06] (The slide updates, highlighting "10. Prefilled response (if any)" in black.)

[20:06] **Christian Ryan:** And with that said, I think, uh, I'd like to thank all of you for coming today. We'll be around as well. So if you have any questions around prompting, please, uh, please go ahead. I know there's a prompting. (The slide changes to: "Prefill Claude's response" with "Prefill Claude's response by writing text in the "Assistant" field / role. Claude will continue from where you left off" and "This allows you to: - Steer Claude's behavior or response - Have greater control over output formatting". An example console screenshot shows a prefilled assistant response ` <itinerary> `.)

[20:51] (The console loads "Accident Report Form Analysis - v5". The system prompt now includes a final instruction: "Wrap your final verdict in <final_verdict> XML tags." The response panel shows "Final Verdict" wrapped in ` <final_verdict> ` tags.)

[20:51] **Christian Ryan:** If we transition back to the console, you'll see here that we've just added a simple important guidelines part. And again, this is just reinforcing the sort of mechanical behavior that we want out of Claude here. We want to make sure that the summary is clear, concise, and accurate. We want to make sure that nothing is sort of impeding in in Claude's assessment, apart from the data it's analyzing. And then finally, when it comes to output formatting, in my case here, I'm just going to ask Claude to wrap its final verdict, all other stuff, I'm actually going to ignore from my application, and just look at what it's actually assessing. And that is, I can I can use this, if I want to build some sort of analytics tool afterwards as well, or if I just want to clear-cut, um, uh determination, this is the way I can do so. So if I just run this here, you'll see it's going through the same sort of process that we've seen before. In this case, it's much more succinct because we've asked it to be to summarize its findings in a in a much more straightforward manner. And then finally, towards the end, you'll see that it'll wrap my output in these final verdict XML tags. So you can see that during this demo, we've gone from a skiing accident to the sort of unconfident, insecure outputs from perhaps a car accident in the second version to now a much more strictly formatted, confident output that we can actually build an LLM application around and actually help, you know, a real-world, um, car insurance company, for example. And finally, if we transition back to the, um, slides, another key way of shaping Claude's output is actually putting words in Claude's mouth, or as we call it, prefilled responses. You could imagine that parsing XML tags is nice and all, but maybe you want a structured JSON output to make sure that, uh, it's JSON serializable and you can use this in a subsequent subsequent call, for example. Um, this is quite simple to do. You can just add that, um, Claude needs to begin its output with a certain format. This could be, for example, a, uh, open square bracket, square bracket, for example, or even in this case that we see in front of us, this would be an XML tag for itinerary. In our case, it could also be that final verdict XML tag. Um, and this is just a great way of again, shaping how Claude is supposed to respond, um, without all the preamble if you don't want that, even though that is also key in shaping its output to make sure that Claude is reasoning through the steps that we wanted. So, in our case here, we would just wrap it in the final verdict and then parse it off to us. But you can use prefill as well.

[23:23] (The slide changes to: "What about extended thinking?")

[23:23] **Christian Ryan:** Now, finally, one step that I would like to highlight here as well is that both Claude 3.5 and especially Claude 4, of course, is, uh, uh, a sort of hybrid reasoning model, meaning that there's extended thinking at your disposal. (The slide changes to: "Extended thinking vs. prompt engineering". On the left: "When do we want to use extended thinking? - Great first step to give Claude more time to think - Follow the traces to understand how Claude thinks -> Use this to steer the system prompt". On the right: "What are the cons of extended thinking? - Can often need to "reinvent the wheel" in its thinking process, which leads to higher token usage - As thinking requires temperature = 1, can at times be less reproducible".) And this is something we want to highlight because you can use extended thinking as a crutch for your prompt engineering. Basically, you can enable this to make sure that Claude actually has time to think. It adds these thinking tags and in the scratchpad. Um, and the beauty of that is that you can actually analyze that transcript to understand how Claude is going about that data. So, as we mentioned, we have these checkboxes where it goes through step-by-step of the scenario that transpired for the accident. And in many ways there, you can actually try to help Claude in building this into the system prompt itself. This is not only more token efficient, but it's a good way of understanding how these intelligent models that don't have our intuition actually go about the data that we provide them. And because of that, it's quite key in actually trying to break down how your system prompt can get a lot better.

[24:19] (Christian turns to Hannah.)

[24:19] **Christian Ryan:** Um, and with that said, I think, uh, I'd like to thank all of you for coming today. We'll be around as well. So if you have any questions around prompting, please, uh, please go ahead. I know there's a prompting...

[24:29] **Hannah Moran:** If you want to learn more about prompting, in an hour we have prompting for agents. And right now, we have an amazing demo of Claude plays Pokemon. So don't go anywhere for that. And as Christian said, we'll be around all day, so I know we didn't have time for Q&A in this session, but please come find us if you want to chat, and thank you guys for coming.

[24:45] (Christian and Hannah smile and give a slight bow, Christian picks up the laptop from the podium and they walk off stage.)

[24:46] (A loading icon appears on a black screen, followed by "c/c Thank you" on a black screen.)

_Summary of section on best practices and extended thinking: The presenters detail the structure of a good prompt for API use, including task and tone context, background information, detailed instructions, examples, conversation history, immediate requests, and structured thinking. They advocate for using delimiters like XML tags to organize information efficiently. They also introduce "extended thinking" as a debugging tool to understand Claude's reasoning process and refine prompts, though noting its potential for higher token usage and less reproducibility compared to prompt engineering itself. Finally, they highlight the importance of output formatting and prefilled responses for controlled and streamlined LLM applications._

</details>


## Additional Sources Scraped

<details>
<summary>5-llm-workflow-patterns-for-production-ai-systems</summary>

# Stop Building AI Agents. Use These 5 Patterns Instead.

### Master the workflow patterns that solve 95% of production problems.

_**Welcome to the AI Agents Foundations series**—a 9-part journey from Python developer to AI Engineer. Made by busy people. For busy people._

Everyone’s talking about AI agents. But what actually is an agent? When do we need them? How do they plan and use tools? How do we pick the correct AI tools and agentic architecture? …and most importantly, where do we even start?

To answer all these questions (and more!), We’ve started a 9-article straight-to-the-point series to build the skills and mental models to ship real AI agents in production.

We will write everything from scratch, jumping directly into the building blocks that will teach you _“how to fish”_.

**What’s ahead**:

1. [Workflows vs. Agents](https://www.decodingai.com/p/ai-workflows-vs-agents-the-autonomy)

2. [Context Engineering](https://www.decodingai.com/p/context-engineering-2025s-1-skill)

3. [Structured Outputs](https://www.decodingai.com/p/llm-structured-outputs-the-only-way)

4. **The 5 Workflow Patterns** ← _You are here_

5. [Tools](https://www.decodingai.com/p/tool-calling-from-scratch-to-production)

6. [Planning: ReAct & Plan-and-Execute](https://www.decodingai.com/p/ai-agents-planning)

7. [Writing ReAct From Scratch](https://www.decodingai.com/p/building-production-react-agents)

8. [AI Agent’s Memory](https://www.decodingai.com/p/how-does-memory-for-ai-agents-work)

9. Multimodal Data

By the end, you’ll have a deep understanding of how to design agents that think, plan, and execute—and most importantly, how to integrate them in your AI apps without being overly reliant on any AI framework.

**Let’s get started.**

---

## The 5 Workflow Patterns

When building **Brown**, my writer assistant capstone project for the AI Agents course, I built together with Towards AI, I faced a critical challenge. The first iteration worked. It could generate articles. But it was slow, expensive, and the user experience was poor. More importantly, when I tried to rewrite it from scratch for better performance, I realized something crucial: the system was trying to do too much in a single, massive LLM call. One prompt was responsible for understanding content, following structure, matching tone, and generating diagrams. When something went wrong, I could not pinpoint where. This experience taught me that complex AI systems need to be broken down, but how?

A single, large LLM call for a complex task is often problematic. It makes pinpointing errors difficult, lacks modularity, and increases the likelihood of “lost in the middle” issues where the model ignores information in long contexts [\[1\]](https://arxiv.org/abs/2307.03172). You might think that when a single prompt fails, you should jump straight to AI Agents. But that is a trap.

Jumping to agents is often overkill for deterministic tasks. Most workflows have predictable steps. Agents add unnecessary complexity with their autonomous decision-making when you just need reliable execution. They introduce too many moving parts to debug, leading to unpredictable costs and reliability issues.

When building agents, here is what actually happens at scale: the agent ignores your tools, the summarizer ignores your instructions, and the coordinator gets lost in all the options. Most agent systems break down from too much complexity, not too little.

The smarter approach is to start with simpler, more controllable patterns. Before considering agents, you should try to solve your problem using **these five core workflow patterns**:

1. Prompt Chaining

2. Parallelization

3. Routing

4. Orchestrator-Worker

5. Evaluator-Optimizer

https://substackcdn.com/image/fetch/$s_!p5MB!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7786a135-c43d-4e55-8e7d-d00ab98c6dc5_1200x1200.png

Let’s start by examining the most fundamental of these patterns: prompt chaining.

## The Power of Modularity: Chain Specialized LLM Calls

Prompt chaining is the fundamental pattern of every workflow. It involves connecting multiple LLM calls or other processing steps sequentially, where the output of one step becomes the input for the next. This divide-and-conquer approach is a more manageable solution for complex tasks. By breaking a large problem into smaller, focused sub-tasks, each LLM call has a clearer, more constrained objective. This improves reliability, as simpler prompts are less likely to confuse the model or lead to incomplete results.

https://substackcdn.com/image/fetch/$s_!I4xD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7ce86666-bc32-44c1-acd8-0b4f3469f9e6_1170x254.png Image 1: The ‘Prompt Chaining’ pattern applied to our writing workflow.

Chaining offers several benefits. It improves modularity, as each LLM call focuses on a specific, well-defined sub-task. This leads to enhanced accuracy because simpler, targeted prompts generally yield more reliable outputs. Debugging becomes easier since you can isolate issues to specific links in the chain. It also increases flexibility, allowing individual components to be swapped or optimized independently. For instance, you could use a cheaper, faster model for a simple classification step and a more powerful model for complex generation [\[2\]](https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/), [\[3\]](https://cookbook.openai.com/examples/partners/model_selection_guide/model_selection_guide).

For our writing workflow, we can chain several steps to produce a final article: creating media assets, generating the article text, crafting a title, and then producing SEO metadata.

However, chaining has downsides. Some instructions may lose meaning when split across multiple steps, and context can be lost between prompts. This approach also increases costs and latency due to multiple LLM calls. Furthermore, if one step in the chain fails, the entire workflow can break, requiring proper error handling.

In theory, this pattern sounds sooo simple you think it’s a no brainer. But in reality, there is a lot of experimentation until you find the sweet spot on how much you should divide your prompts.

For example, here is a high-level Python example illustrating the chaining pattern for our writing workflow:

```
def writing_workflow(research_and_intent: str) -> dict:
    “””
    A sequential workflow for generating an article.
    “””

    # Step 1: Create media assets based on the initial input
    media_assets = create_media_assets(research_and_intent)

    # Step 2: Generate the article using the assets and the initial input
    article_draft = generate_article(research_and_intent, media_assets)

    # Step 3: Generate a title for the draft
    title = generate_title(article_draft)

    # Step 4: Generate SEO metadata for the final article
    seo_metadata = generate_seo(article_draft, title)

    return {
        “title”: title,
        “article”: article_draft,
        “seo”: seo_metadata
    }
```

While chaining provides modularity and control, its sequential nature can create bottlenecks. The next logical step is to identify independent tasks within our workflow and execute them in parallel to improve speed and efficiency.

## Optimize Your Sequential Workflows With Proper Parallelization

While a sequential workflow is reliable, it can be slow. We can optimize it by running independent steps in parallel to reduce the total processing time. Parallelization simply means running as many steps as possible concurrently, as long as they do not depend on each other’s outputs. This is particularly effective for tasks like data extraction from multiple sources or, in our case, generating various media assets at once.

https://substackcdn.com/image/fetch/$s_!K4u7!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8c10eb06-e78e-4596-a4f1-2d4bf757d2e1_1170x570.png Image 2: The parallelization pattern applied to our writing workflow.

This pattern is useful when tasks can be processed independently. However, it can introduce challenges like race conditions or timeouts. It is also important to be mindful of API rate limits, as making many concurrent calls can lead to errors. LLM APIs such as Gemini, OpenAI or Claude often have limits, known as Requests Per Minute (RPM) quotas, which must be managed in a production application [\[4\]](https://ai.google.dev/gemini-api/docs/rate-limits), [\[5\]](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas). Thus, your application should handle these gracefully, by running a maximum number of jobs in parallel using async semaphors, thread pools or retry strategies with exponential backoff.

In our writing workflow example, we can generate all the required media assets—like diagrams and images—in parallel. After initiating all tasks, the system waits for them to complete and can retry any that fail.

The following code shows how to implement this using Python’s `asyncio` library:

```
import asyncio

async def parallel_media_generation(research_and_intent, max_concurrent=3):
    “””
    Generates all media assets in parallel with concurrency control.
    “””
    asset_types = [”diagram”, “image_1”, “image_2”, “image_3”, “image_4”]

    # Create a semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_with_limit(asset_type):
        “””Wrapper to control concurrency with semaphore.”“”
        async with semaphore:
            # Only max_concurrent tasks will execute simultaneously
            return await generate_media_asset(asset_type, research_and_intent)

    # Create a list of tasks to run concurrently (but controlled by semaphore)
    tasks = [generate_with_limit(atype) for atype in asset_types]

    # Wait for all tasks to complete
    media_assets = await asyncio.gather(*tasks)

    # Here you could add logic to retry failed tasks

    return media_assets
```

Parallelization is effective for uniform, independent tasks, but not all workflows are so straightforward. Often, the path forward depends on the nature of the input itself, requiring a more dynamic approach.

## Dynamic Decisions: Routing and Conditional Logic

Not all inputs should be processed in the same way. Routing, or conditional logic, addresses this by directing the workflow down different paths based on the input or an intermediate state. Think of it as a smart if-else statement where an LLM acts as a classifier to make branching decisions. This keeps prompts specialized and follows the “divide-and-conquer” principle, ensuring that each part of your workflow is optimized for a specific type of task.

https://substackcdn.com/image/fetch/$s_!lpYG!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fddcdb3bf-8a13-4ce6-bffc-87536563b34e_995x388.png Image 3: The routing workflow pattern applied to our writing agent.

Routing is ideal when you have distinct categories of inputs that require different handling. For example, a customer support tool could route queries to specialized handlers for technical support, billing, or general questions [\[6\]](https://googleapis.github.io/python-genai/), [\[7\]](https://github.com/google-gemini/cookbook), [\[8\]](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference). This is preferable to optimizing a single, monolithic prompt for multiple input types, where often optimizing for one use case degrades the performance on others.

A critical best practice is to include a default or catch-all route to handle unexpected cases gracefully. This ensures your system is robust and can manage inputs that do not fit neatly into predefined categories.

For our writing workflow, we can use a router to decide which type of media to generate based on the user’s intent. An LLM classifier determines if the user wants a diagram, an image, or a video, and calls the appropriate generation function.

> 💡 **Tip:** For simple classification tasks like routing, using smaller, faster, and cheaper language models is a great way to optimize your workflow [\[9\]](https://arxiv.org/html/2502.00409v2).

This Python code demonstrates how routing can be implemented:

```
def routing_workflow(user_intent):
    “””
    Routes the request to the correct media generator.
    “””
    media_type = classify_media_intent(user_intent)

    if media_type == “diagram”:
        return generate_diagram(user_intent)
    elif media_type == “image”:
        return generate_image(user_intent)
    elif media_type == “video”:
        return generate_video(user_intent)
    else:
        # Fallback for unknown types
        return handle_default(user_intent)
```

Routing manages predefined paths, but what if the tasks themselves need to be defined dynamically? This leads us to a more advanced pattern.

## Dynamic Task Decomposition: The Orchestrator-Worker

The orchestrator-worker pattern is a more advanced approach that combines routing with parallel processing. A central LLM, the orchestrator, dynamically breaks down a complex task into smaller sub-tasks and delegates them to specialized workers, which can run in parallel [\[10\]](https://www.anthropic.com/research/building-effective-agents), [\[11\]](https://langchain-ai.github.io/langgraphjs/tutorials/workflows/), [\[12\]](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns), [\[13\]](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/mixture-of-agents.html).

https://substackcdn.com/image/fetch/$s_!sm2h!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc9cf87df-9e7c-4c86-b7fa-16c47230b226_1200x1200.png Image 4: The Orchestrator-Worker pattern applied to our writing workflow.

The key difference from simple parallelization is that the number and type of the sub-tasks required to run are determined dynamically at runtime based on the input.

On a similar note, it’s different from simple routing because it “classifies” each sub-task individually, rather than a single if-else path.

The pattern is very similar to the Map-Reduce strategy from data engineering, where we have multiple jobs running in parallel, while we aggregate the result at the end. It’s similar, but with LLMs into the mix that determine what jobs should be run.

The main point of failure of this pattern is when the orchestrator creates the wrong jobs, or jobs with the wrong arguments, or not enough/too many jobs.

For our writing workflow, an orchestrator can analyze the user’s intent and generate a list of media generation jobs. An executor then takes these jobs and runs them in parallel.

> 💡 **Tip**: Implement the set of possible jobs as “tools.” The orchestrator can then output multiple tool calls with their corresponding arguments, which the executor can run in parallel.

For example, this is how the pattern looks in [Opik](https://www.comet.com/site/?utm_source=newsletter&utm_medium=partner&utm_campaign=paul), where we can see all the `mermaid_diagram_generator_tool` calls, together with the cost, latency, token usage, inputs and outputs of each step. Beautiful!

https://substackcdn.com/image/fetch/$s_!lpif!,w_2400,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8e00f379-43d1-4a8b-9235-f44cc91ee955_2932x1766.png Image 5: The Orchestrator-Worker pattern seen in [Opik](https://www.comet.com/site/?utm_source=newsletter&utm_medium=partner&utm_campaign=paul)

Here is how this pattern could look in Python:

```
from concurrent.futures import ThreadPoolExecutor

# Mock available tools
TOOLS = {
    “generate_diagram”: lambda prompt: f”Diagram generated: {prompt}”,
    “generate_image”: lambda prompt: f”Image generated: {prompt}”,
    “generate_video”: lambda prompt: f”Video generated: {prompt}”,
}

class LLM:
    “””Mocked LLM class.”“”

    @staticmethod
    def call(user_intent, tools):
        “””
        Mocked LLM call that returns tool calls based on user intent.
        In reality, this would be a call to OpenAI, Anthropic, etc.
        “””

        # Mock logic to simulate LLM deciding which tools to call
        tool_calls = []
        if “chart of performance” in user_intent:
            tool_calls.append({
                “tool”: “generate_diagram”,
                “arguments”: {”prompt”: “Generate performance chart”}
            })
        if “photo of the team” in user_intent:
            tool_calls.append({
                “tool”: “generate_image”,
                “arguments”: {”prompt”: “Find team photo”}
            })

        return tool_calls

def orchestrator(user_intent):
    “””Orchestrator that calls the LLM with available tools.”“”
    return LLM.call(user_intent=user_intent, tools=TOOLS)

def worker(tool_call):
    “””Worker that executes a single tool call.”“”
    tool_name = tool_call[”tool”]
    tool_function = TOOLS[tool_name]
    return tool_function(**tool_call[”arguments”])

def orchestrator_worker_workflow(user_intent):
    “””
    Manages the orchestrator and workers to generate media.
    “””
    # 1. Orchestrator generates tool calls via LLM
    tool_calls = orchestrator(user_intent)

    # 2. Workers execute tool calls in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, tool_calls))

    return results

# Example usage:
user_request = “Create a chart of performance and find a photo of the team”
outputs = orchestrator_worker_workflow(user_request)
# outputs: [”Diagram generated: Generate performance chart”,\
#           “Image generated: Find team photo”]
```

So far, our patterns have focused on executing tasks efficiently. But what about ensuring the quality of the final output? The next pattern introduces a feedback loop to do just that.

## Make LLMs Autocorrect Themselves: The Evaluator-Optimizer

The evaluator-optimizer pattern is the closest we get to agent-like behavior while still maintaining a structured workflow. It creates a feedback loop to automatically improve and refine the output of an LLM. This is an effective pattern when output quality matters more than speed.

https://substackcdn.com/image/fetch/$s_!8zT2!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8ff83ae6-3d06-4f0d-b68d-aaa03a875756_1025x606.png Image 6: The Evaluator-Optimizer workflow pattern applied to our writing agent.

The system typically consists of a few key components [\[14\]](https://arxiv.org/html/2410.03131v3): 1\. A **generator** LLM produces an initial output. 2. An **evaluator** LLM assesses the output against predefined criteria, generating a score and an improvement plan. 3. The generator takes the feedback from the evaluator to refine the output. 4. This loop repeats until a certain quality score is met or a maximum number of iterations is reached.

The main failure mode for this pattern is an infinite optimization loop. Therefore, it is a best practice to set clear stop conditions, such as a well-tweaked stop threshold or a maximum number of retries.

A more advanced implementation that applies the divide and conquer patterns we just learned uses multiple specialized evaluators. For instance, one evaluator could check for logical correctness, another for readability, and a third for syntax. This multifaceted feedback provides a more comprehensive critique.

In our writing workflow, we can use this pattern to refine an article draft. The generator produces the text, and an evaluator provides feedback on clarity, tone, and structure. The generator then revises the draft based on this feedback until it meets our quality standards.

As shown for the orchestrator-worker pattern, in Image 7, you can see how this pattern looks in [Opik](https://www.comet.com/site/?utm_source=newsletter&utm_medium=partner&utm_campaign=paul), along with its multimodal inputs, which are text and images in this use case. For example, in this trace, we had 2 iterations of: generating the article, reviewing it and editing it.

https://substackcdn.com/image/fetch/$s_!hX6W!,w_2400,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0da201f9-ce0e-4135-a946-db1480e5c5bb_2904x1714.png Image 7: The Evaluator-Optimizer workflow pattern seen in [Opik](https://www.comet.com/site/?utm_source=newsletter&utm_medium=partner&utm_campaign=paul)

This code demonstrates the evaluator-optimizer loop:

```
def generator(prompt, feedback=None):
    “””Generates an article draft.”“”
    if feedback:
        prompt += f”\nIncorporate this feedback: {feedback}”

    return llm_call(prompt)

def evaluator(article_draft):
    “””Evaluates the draft and provides a score and feedback.”“”
    evaluation = llm_evaluator_call(article_draft) # Returns score and feedback text

    return evaluation[”score”], evaluation[”feedback”]

def evaluator_optimizer_workflow(initial_prompt):
    “””
    Iteratively refines an article until it meets quality standards.
    “””
    max_iterations = 3
    score_threshold = 0.8

    article = generator(initial_prompt)

    for i in range(max_iterations):
        score, feedback = evaluator(article)

        if score >= score_threshold:
            print(”Quality threshold met. Finalizing article.”)
            return article

        print(f”Iteration {i+1}: Score={score}. Refining article with feedback.”)
        article = generator(initial_prompt, feedback)

    print(”Max iterations reached. Returning the latest version.”)
    return article
```

## Stop Building AI Agents. Always Start With These 5 Patterns.

With these five patterns, you can solve most of the real-world problems you will encounter when building AI applications. Choosing to build a full-fledged agent should be your last resort, reserved for when simpler, more controllable methods fail.

That’s why I fell in love with this mantra vetted by

[Hugo Bowne-Anderson](https://open.substack.com/users/63978139-hugo-bowne-anderson?utm_source=mentions)

: _“Most use cases don’t need agents. They need better workflows.”_

The strategy is simple:

1. Start with the simplest solution (a single prompt).

2. If it works, stop.

3. If it does not, try these five workflow patterns.

4. Only if all patterns fail, consider agents.

_Remember that this article is part of a longer series of 9 pieces on the AI Agents Foundations that will give you the tools to morph from a Python developer to an AI Engineer._

**Here’s our roadmap:**

1. [Workflows vs. Agents](https://www.decodingai.com/p/ai-workflows-vs-agents-the-autonomy)

2. [Context Engineering](https://www.decodingai.com/p/context-engineering-2025s-1-skill)

3. [Structured Outputs](https://www.decodingai.com/p/llm-structured-outputs-the-only-way)

4. **The 5 Workflow Patterns** _← You just finished this one._

5. [Tools](https://www.decodingai.com/p/tool-calling-from-scratch-to-production) ← _Move to this one_

6. [Planning: ReAct & Plan-and-Execute](https://www.decodingai.com/p/ai-agents-planning)

7. [Writing ReAct From Scratch](https://www.decodingai.com/p/building-production-react-agents)

8. [AI Agent’s Memory](https://www.decodingai.com/p/how-does-memory-for-ai-agents-work)

9. Multimodal Data

See you next week.

[Paul Iusztin](https://x.com/pauliusztin_)

---

## References

01. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts. _arXiv preprint arXiv:2307.03172_. [https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)

02. (n.d.). _Multi-LLM routing strategies for generative AI applications on AWS_. Amazon Web Services. [https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/](https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/)

03. (n.d.). _Model selection guide_. OpenAI Cookbook. [https://cookbook.openai.com/examples/partners/model\_selection\_guide/model\_selection\_guide](https://cookbook.openai.com/examples/partners/model_selection_guide/model_selection_guide)

04. (n.d.). _Gemini API Rate Limits_. Google AI for Developers. [https://ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)

05. (n.d.). _Generative AI on Vertex AI quotas and system limits_. Google Cloud. [https://cloud.google.com/vertex-ai/generative-ai/docs/quotas](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas)

06. (n.d.). _Python Client for Google’s Generative AI Models_. Google API Python Client. [https://googleapis.github.io/python-genai/](https://googleapis.github.io/python-genai/)

07. (n.d.). _Welcome to the Gemini API Cookbook_. GitHub. [https://github.com/google-gemini/cookbook](https://github.com/google-gemini/cookbook)

08. (n.d.). _Gemini API Reference_. Google Cloud. [https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference)

09. (n.d.). _Technical analysis of routing strategies in LLM-based systems_. arXiv. [https://arxiv.org/html/2502.00409v2](https://arxiv.org/html/2502.00409v2)

10. Schluntz, E., & Zhang, B. (n.d.). _Building effective agents_. Anthropic. [https://www.anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents)

11. (n.d.). _Workflows and Agents_. LangChain. [https://langchain-ai.github.io/langgraphjs/tutorials/workflows/](https://langchain-ai.github.io/langgraphjs/tutorials/workflows/)

12. (n.d.). _AI agent design patterns_. Azure Architecture Center. [https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)

13. (n.d.). _Mixture of Agents_. Microsoft AutoGen. [https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/mixture-of-agents.html](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/mixture-of-agents.html)

14. Patel, B., Chakraborty, S., Suttle, W. A., Wang, M., Bedi, A. S., & Manocha, D. (2024). AIME: AI System Optimization via Multiple LLM Evaluators. _arXiv preprint arXiv:2410.03131_. [https://arxiv.org/html/2410.03131v3](https://arxiv.org/html/2410.03131v3)

---

## Images

If not otherwise stated, all images are created by the author.

</details>

<details>
<summary>building-ai-agents-from-scratch-part-2-reflection-and-workin</summary>

# Building AI Agents from scratch - Part 2: Reflection and Working Memory

### Let's implement AI Agent from scratch without using any framework. Today we implement the reflection pattern coupled with simple implementation of short-term memory.

I hope you had a wonderful holiday season! As we move into the **year of AI Agents**, I also have a present for you, the second part in the series of “Building AI Agents from scratch”. If you missed the previous episode (part 1), you can find it here:

[**Building AI Agents from scratch - Part 1: Tool use**](https://www.newsletter.swirlai.com/p/building-ai-agents-from-scratch-part)

We implemented Tool Use pattern without using any LLM Orchestrator frameworks. Today, we will build on top of the previous project, I will explain how later in the article.

In this article you will learn:

- What Reflection pattern in AI Agent systems is.

- How it relates to short-term memory.

- Pros and Cons of implementing Reflection.

- How to build an Agent class that is able to implement Reflection pattern taking the memory into consideration without using any orchestration frameworks.

  - Just to remind ourselves why we are doing this. If you are using any orchestration frameworks for agentic applications, you might be abstracted away from how Agentic patterns are actually implemented there. Having clarity of how the systems actually work helps you build up systems thinking, enabling you to craft advanced applications more efficiently.
- Fix some of the hallucinations we were producing in the previous project!

You can find the code examples for this and other projects in my GitHub repository here:

[AI Engineer's Handbook](https://github.com/swirl-ai/ai-angineers-handbook)

### Defining Reflection in AI Agents.

As it is with most definitions in AI Agents nowadays, there is no single way to specifically describe Reflection. The high level definition of the pattern is:

> The ability of an Agentic System to reflect on it’s outputs and suggest improvements. Optionally, also improve the behaviour of the future actions in the system incorporating the feedback provided.

When explaining Agentic concepts I like to drop the abstraction of an Agent and think in Agentic flows - it is easier to reason when analysing flow diagrams and Agents eventually are just a number of steps interconnected via different topologies. Reflection can be applied in different steps of an Agentic flow. Let’s see how.

#### The simplest example.

https://substackcdn.com/image/fetch/$s_!kGSd!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa642fd3b-a4bb-4465-bc4b-77948c14edea_1234x704.png Reflection: Simplest Case

The above Agentic flow includes the following steps:

1. User prompts the LLM with a query.

2. Generated answer is passed to the LLM where it is asked to provide feedback to the previously generated answer and provide instructions for improvement if any.

3. The improved answer is returned to the user.

As simple as it is, even this pipeline will provide significant improvements to the accuracy of the answers in many cases. Ofcourse, similar improvements can be achieved via prompt engineering of the system prompt, but reflection pattern is usually more powerful and flexible.

#### Reflection Loop.

https://substackcdn.com/image/fetch/$s_!neQ7!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff19163d0-6ead-4bdb-95d5-fafd5acef297_1234x704.png Reflection: The Loop

The above Agentic flow is defined via the following steps:

1. A user query is passed to the LLM.

2. Generated answer is passed to the LLM where it is asked to provide feedback to the previously generated answer and provide instructions for improvement if any.

3. After applying the improvements, the improved answer is passed to the LLM again and is asked to provide feedback and suggestions for improvements once more. This loop is then repeated for a predefined amount of times or until the LLM is not able to generate more suggestions and returns a stop sign, usually a predefined string like “END”.

4. The final answer is returned to a user.

This kind of system is powerful, but requires a very specific use case. In literature it can be most often found described for code generation. It is easy to understand why - generated code can be continuously improved in multiple iterations. The system will always find where to over-engineer ;)

Andrew Ng has defined this pattern for code in one of his articles [here](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/).

In most real world use cases I would find it hard to make a business case for the use of reflection loop as it usually increases costs significantly.

#### Validating execution plans.

https://substackcdn.com/image/fetch/$s_!_6Qg!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe45a3fb8-84cd-4775-9b5f-421632628181_1194x828.png Reflection: Validation of execution plans

I find one of the useful spots to place a Reflection step is to validate an execution plan if planning is part of the Agentic flow. Let’s see how it could look like:

1. A user query with intent and a system prompt is passed to the LLM. The LLM generates an execution plan. This is also where an important point where the execution of the flow might break is. An example of the breakage would be the following:

1. Agentic flow decides if a direct answer needs to be returned to the user or a predefined tool should be used for additional context generation.

2. A decision is made that a tool should be used,

3. An incorrect set of parameters to be passed to the tool are hallucinated.

4. The tool returns an error.
2. With properly crafted Reflection step we can prompt the Agent to try and fix any hallucinations produced in previous step.

3. The plan can return a direct answer that is provided to the user,

4. Or it can prompt to use a tool which enriches the answer via another LLM call and it then is returned to the user.

We will actually be implementing this kind of flow in our hands-on example in the following sections.

#### More Complex Reflection flows.

As mentioned before, there is no predefined place for a reflection step to be invoked. In complex Agentic flows it can be used multiple times to validate intermediary answers, plans or any other part of Agentic topology.

https://substackcdn.com/image/fetch/$s_!BjrJ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F42f7fa39-9c99-41a7-a1fd-badcf622a31c_1704x930.png Reflection: Complex Agentic flows

In reality, in order to automate complex processes that are present in organisations, you would build multi-step topologies with multiple probabilistic routers connecting execution nodes. Some routers could be implemented via LLMs, some might be rule based, some might use regular ML models. Some execution nodes will have tools, some will be just LLM calls, some will have non-probabilistic executions. Reflection steps could be implemented all around the place to increase the accuracy of non-deterministic routers and executions.

### Connection between Agent Memory and Reflection pattern.

Usually, there is a need for some sort of short-term (working) memory implementation to make Reflection provide best results. Let’s see why by examining the third example of Agentic flows defined above.

https://substackcdn.com/image/fetch/$s_!7NSW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7b23ab7a-979f-4ba1-acdd-acaca8871040_1194x913.png Working Memory for Reflection

The Agentic flow until the Reflection step can be invoked includes:

_A)_ A System Prompt and a user query that will be passed to the LLM in order to generate the initial Execution Plan.

_B)_ The execution plan generated by the initial steps.

_C)_ The Reflection step will need All of this information since the System prompt will most likely have useful information about the available tools and similar, The user Query is important since it invokes the generation with context about user intent, the plan is important because it is what we are trying to improve.

In base generation implementation we are not keeping this information in memory, that is why the Working Memory needs to be implemented.

https://substackcdn.com/image/fetch/$s_!Je5w!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F397c1965-c739-4d9b-a821-a06a817e4130_2383x1276.png Working Memory: Simplest implementation

In the above picture I show the simplest implementation of Working Memory - Each of the interactions with an agent are simply stored in the list and passed as additional context to the system prompt each time the generation is invoked. This is how the simplest type of memory is implemented in the Chat Bots (e.g. ChatGPT).

If we are implementing an Agent with the capability of planning and we want to include the reflection capability, this is how it could look like:

https://substackcdn.com/image/fetch/$s_!6Stj!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdefc4bc5-bdd8-4c1d-9c31-ffc47e56434d_1642x1124.png Working Memory: Planning and Reflection

As you can see, it is very similar, the only difference is that the initial response by the agent is the execution plan and the next query by the user is a prompt to reflect on the plan given previous interactions.

### Pros and Cons of using Reflection.

As with anything that can bring improvements to the system, there is always upsides and downsides.

#### Pros.

- Almost guaranteed improvement in the accuracy of final outputs of the system. Sometimes this is the only way to make your application feasible due to extremely high accuracy requirements.

- Flexible compared to editing the initial system prompt. E.g. different Reflection methods or even agents can be utilised for different parts of the pipeline.

- It is possible to achieve more with small models when Reflection is applied. Ofcourse, you should analyse the tradeoff given all of the cons but in general there is always a phase in the project where the goal is to optimise the costs of your AI system.

#### Cons.

- Adds complexity to the application.

- Adds additional latency to the end-to-end flow since additional LLM calls are invoked.

- Adds additional cost since the LLMs are prompted at least one more additional time (usually more than one) with every Reflection step.

### Building the Reflection Agent.

As mentioned at the beginning of the article, we will be working on the example where Reflection will be used to revise an action plan generated by a LLM. And this is not without a reason!

If you followed me in the [first part](https://www.newsletter.swirlai.com/p/building-ai-agents-from-scratch-part) of the series where we implemented tool usage from scratch, you might remember that we implemented a tool that is capable of converting between two currencies on demand. We prompted the agent to only use the tool if the conversion is actually needed. It worked well, but in the example we asked for Serbian to Japanese currency conversion.

Now, this was not my initial intent. I am from Lithuania and I wanted to showcase the scenario where I would be traveling from Lithuania to Japan so I initially prompted the model with the following query:

```
I am traveling to Japan from Lithuania, I have 1500 of local currency, how much of Japanese currency will I be able to get?
```

Imagine my surprise, when the agent returned the following response:

```
Thought: I need to convert 1500 Lithuanian Litas (LTL) to Japanese Yen (JPY) using the currency conversion tool.
Plan: Use convert_currency tool to convert 1500 LTL to JPY. Return the conversion result
Results: Error: Could not fetch exchange rates
```

Ofcourse the tool call resulted in an error - Lithuania has Euro (EUR) as an official currency from 2015! The tool was obviously not able to find LTL conversion rate to JPY.

And then I took it personal, in this episode my quest is to try and fix the plan using the Reflection pattern.

https://substackcdn.com/image/fetch/$s_!wvjR!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d419729-d19c-42d0-a407-9c36ea5b9073_1194x913.png Reflection: Fixing the Plan

You can also find the code in a GitHub repository [here](https://github.com/swirl-ai/ai-angineers-handbook/tree/main/building_agents_from_scratch/planning/reflection).

You can follow the tutorial using the Jupyter notebook [here](https://github.com/swirl-ai/ai-angineers-handbook/blob/main/building_agents_from_scratch/planning/reflection/notebooks/reflection.ipynb).

#### Implementing the working memory.

We will implement a single interaction with an agent as a Dataclass:

```
@dataclass
class Interaction:
    """Record of a single interaction with the agent"""
    timestamp: datetime
    query: str
    plan: Dict[str, Any]
```

It will have the query and the plan that the agent produced.

It is important to note that we will also need the system prompt to be available for reflection step, but we will implement that separately.

We will be simplifying the Agent Class this time around by stripping any tool related functionality from it to better focus on reflecting on the plan generated by the Agent.

#### The initial system prompt.

We will keep the system prompt identical to the one we implemented in the part 1 of the series. You can find the explanation in [this](https://www.newsletter.swirlai.com/i/153433846/crafting-the-system-prompt) section of my previous article. The two differences are:

- We are now mocking the available tool instead of actually implementing it. So the tools section will look like this:

```
"tools": [\
    {\
        "name": "convert_currency",\
        "description": "Converts currency using latest exchange rates.",\
        "parameters": {\
            "amount": {\
                "type": "float",\
                "description": "Amount to convert"\
            },\
            "from_currency": {\
                "type": "str",\
                "description": "Source currency code (e.g., USD)"\
            },\
            "to_currency": {\
                "type": "str",\
                "description": "Target currency code (e.g., EUR)"\
            }\
        }\
    }\
]
```

- We are extending the capabilities and instructions of the agent by including the fourth line in both:

```
"capabilities": [\
    "Using provided tools to help users when necessary",\
    "Responding directly without tools for questions that don't require tool usage",\
    "Planning efficient tool usage sequences",\
    "If asked by the user, reflecting on the plan and suggesting changes if needed"\
],
"instructions": [\
    "Use tools only when they are necessary for the task",\
    "If a query can be answered directly, respond with a simple message instead of using tools",\
    "When tools are needed, plan their usage efficiently to minimize tool calls",\
    "If asked by the user, reflect on the plan and suggest changes if needed"\
]
```

#### Implementing the Agent Class.

The agent class will be initialised with an empty Interaction history. Interaction history IS the working memory in this example.

```
class Agent:
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize Agent with empty interaction history."""
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.interactions: List[Interaction] = []  # Working memory
        self.model = model
```

#### _Planning._

The _plan_ method prompts the LLM to create the initial execution plan and updates the working memory with the user query and the generated plan.

```
def plan(self, user_query: str) -> Dict:
    """Use LLM to create a plan and store it in memory."""
    messages = [\
        {"role": "system", "content": self.create_system_prompt()},\
        {"role": "user", "content": user_query}\
    ]

    response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=0
    )

    try:
        plan = json.loads(response.choices[0].message.content)
        # Store the interaction immediately after planning
        interaction = Interaction(
            timestamp=datetime.now(),
            query=user_query,
            plan=plan
        )
        self.interactions.append(interaction)
        return plan
    except json.JSONDecodeError:
        raise ValueError("Failed to parse LLM response as JSON")
```

#### _Reflecting on the plan._

The _reflect\_on\_plan_ method is where the interesting part is.

```
def reflect_on_plan(self) -> Dict[str, Any]:
    """Reflect on the most recent plan using interaction history."""
    if not self.interactions:
        return {"reflection": "No plan to reflect on", "requires_changes": False}

    latest_interaction = self.interactions[-1]

    reflection_prompt = {
        "task": "reflection",
        "context": {
            "user_query": latest_interaction.query,
            "generated_plan": latest_interaction.plan
        },
        "instructions": [\
            "Review the generated plan for potential improvements",\
            "Consider if the chosen tools are appropriate",\
            "Verify tool parameters are correct",\
            "Check if the plan is efficient",\
            "Determine if tools are actually needed"\
        ],
        "response_format": {
            "type": "json",
            "schema": {
                "requires_changes": {
                    "type": "boolean",
                    "description": "whether the plan needs modifications"
                },
                "reflection": {
                    "type": "string",
                    "description": "explanation of what changes are needed or why no changes are needed"
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "specific suggestions for improvements",
                    "optional": True
                }
            }
        }
    }

    messages = [\
        {"role": "system", "content": self.create_system_prompt()},\
        {"role": "user", "content": json.dumps(reflection_prompt, indent=2)}\
    ]

    response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"reflection": response.choices[0].message.content}
```

We create a new prompt with specific instructions and desired output format. Note that the prompt has both the user query and the initial plan passed as additional context via a _context_ key _._ Then we pass the prompt together with the initial system prompt to generate reflection on the initially generated plan including any suggestions for improvements.

#### _Executing the flow and generating Revised Plan._

We now stitch the flow together via the _execute_ method.

```
def execute(self, user_query: str) -> str:
    """Execute the full pipeline: plan, reflect, and potentially replan."""
    try:
        # Create initial plan (this also stores it in memory)
        initial_plan = self.plan(user_query)

        # Reflect on the plan using memory
        reflection = self.reflect_on_plan()

        # Check if reflection suggests changes
        if reflection.get("requires_changes", False):
            # Generate new plan based on reflection
            messages = [\
                {"role": "system", "content": self.create_system_prompt()},\
                {"role": "user", "content": user_query},\
                {"role": "assistant", "content": json.dumps(initial_plan)},\
                {"role": "user", "content": f"Please revise the plan based on this feedback: {json.dumps(reflection)}"}\
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )

            try:
                final_plan = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                final_plan = initial_plan  # Fallback to initial plan if parsing fails
        else:
            final_plan = initial_plan

        # Update the stored interaction with all information
        self.interactions[-1].plan = {
            "initial_plan": initial_plan,
            "reflection": reflection,
            "final_plan": final_plan
        }

        # Return the appropriate response
        if final_plan.get("requires_tools", True):
            return f"""Initial Thought: {initial_plan['thought']}
Initial Plan: {'. '.join(initial_plan['plan'])}
Reflection: {reflection.get('reflection', 'No improvements suggested')}
Final Plan: {'. '.join(final_plan['plan'])}"""
        else:
            return f"""Response: {final_plan['direct_response']}
Reflection: {reflection.get('reflection', 'No improvements suggested')}"""

    except Exception as e:
        return f"Error executing plan: {str(e)}"
```

Note that the method applies reflection improvement on the plan ONLY if requested by the reflection generation. If not, we keep the old plan.

#### Executing the Agent.

Let’s see if we managed to fix the plan! Execute the agent by running:

```
query_list = ["I am traveling to Japan from Lithuania, I have 1500 of local currency, how much of Japanese currency will I be able to get?"]

for query in query_list:
    print(f"\nQuery: {query}")
    result = agent.execute(query)
    print(result)
```

If you used the same models as me, you should be getting something similar to:

```
Query: I am traveling to Japan from Lithuania, I have 1500 of local currency, how much of Japanese currency will I be able to get?
Initial Thought: I need to convert 1500 Lithuanian Litas (LTL) to Japanese Yen (JPY) using the currency conversion tool.
Initial Plan: Use convert_currency tool to convert 1500 LTL to JPY. Return the conversion result
Reflection: The plan needs modifications because the Lithuanian Litas (LTL) is no longer in use since Lithuania adopted the Euro (EUR) in 2015. Therefore, the conversion should be from EUR to JPY instead of LTL.
Final Plan: Use convert_currency tool to convert 1500 EUR to JPY. Return the conversion result
```

Great success! we have fixed the plan and would be able to execute the tool successfully.

#### That’s it for today, we’ve learned:

- How to implement a simple type of Working Memory.

- How to construct a Reflection step to reflect on the execution plan generated by the Agent.

- How to implement the suggestions generated by the reflection step and improve the plan.

</details>

<details>
<summary>functional-api-overview-docs-by-langchain</summary>

The **Functional API** allows you to add LangGraph’s key features — [persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [memory](https://docs.langchain.com/oss/python/langgraph/add-memory), [human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/interrupts), and [streaming](https://docs.langchain.com/oss/python/langgraph/streaming) — to your applications with minimal changes to your existing code.It is designed to integrate these features into existing code that may use standard language primitives for branching and control flow, such as `if` statements, `for` loops, and function calls. Unlike many data orchestration frameworks that require restructuring code into an explicit pipeline or DAG, the Functional API allows you to incorporate these capabilities without enforcing a rigid execution model.The Functional API uses two key building blocks:

- **`@entrypoint`** – Marks a function as the starting point of a workflow, encapsulating logic and managing execution flow, including handling long-running tasks and interrupts.
- **[`@task`](https://reference.langchain.com/python/langgraph/func/#langgraph.func.task)** – Represents a discrete unit of work, such as an API call or data processing step, that can be executed asynchronously within an entrypoint. Tasks return a future-like object that can be awaited or resolved synchronously.

This provides a minimal abstraction for building workflows with state management and streaming.

For information on how to use the functional API, see [Use Functional API](https://docs.langchain.com/oss/python/langgraph/use-functional-api).

## Functional API vs. Graph API

For users who prefer a more declarative approach, LangGraph’s [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) allows you to define workflows using a Graph paradigm. Both APIs share the same underlying runtime, so you can use them together in the same application.Here are some key differences:

- **Control flow**: The Functional API does not require thinking about graph structure. You can use standard Python constructs to define workflows. This will usually trim the amount of code you need to write.
- **Short-term memory**: The **GraphAPI** requires declaring a [**State**](https://docs.langchain.com/oss/python/langgraph/graph-api#state) and may require defining [**reducers**](https://docs.langchain.com/oss/python/langgraph/graph-api#reducers) to manage updates to the graph state. `@entrypoint` and `@tasks` do not require explicit state management as their state is scoped to the function and is not shared across functions.
- **Checkpointing**: Both APIs generate and use checkpoints. In the **Graph API** a new checkpoint is generated after every [superstep](https://docs.langchain.com/oss/python/langgraph/graph-api). In the **Functional API**, when tasks are executed, their results are saved to an existing checkpoint associated with the given entrypoint instead of creating a new checkpoint.
- **Visualization**: The Graph API makes it easy to visualize the workflow as a graph which can be useful for debugging, understanding the workflow, and sharing with others. The Functional API does not support visualization as the graph is dynamically generated during runtime.

## Example

Below we demonstrate a simple application that writes an essay and [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) to request human review.

```
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt

@task
def write_essay(topic: str) -> str:
    """Write an essay about the given topic."""
    time.sleep(1) # A placeholder for a long-running task.
    return f"An essay about topic: {topic}"

@entrypoint(checkpointer=InMemorySaver())
def workflow(topic: str) -> dict:
    """A simple workflow that writes an essay and asks for a review."""
    essay = write_essay("cat").result()
    is_approved = interrupt({
        # Any json-serializable payload provided to interrupt as argument.
        # It will be surfaced on the client side as an Interrupt when streaming data
        # from the workflow.
        "essay": essay, # The essay we want reviewed.
        # We can add any additional information that we need.
        # For example, introduce a key called "action" with some instructions.
        "action": "Please approve/reject the essay",
    })

    return {
        "essay": essay, # The essay that was generated
        "is_approved": is_approved, # Response from HIL
    }
```

Detailed Explanation

This workflow will write an essay about the topic “cat” and then pause to get a review from a human. The workflow can be interrupted for an indefinite amount of time until a review is provided.When the workflow is resumed, it executes from the very start, but because the result of the `writeEssay` task was already saved, the task result will be loaded from the checkpoint instead of being recomputed.

```
import time
import uuid
from langgraph.func import entrypoint, task
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

@task
def write_essay(topic: str) -> str:
    """Write an essay about the given topic."""
    time.sleep(1)  # This is a placeholder for a long-running task.
    return f"An essay about topic: {topic}"

@entrypoint(checkpointer=InMemorySaver())
def workflow(topic: str) -> dict:
    """A simple workflow that writes an essay and asks for a review."""
    essay = write_essay("cat").result()
    is_approved = interrupt(
        {
            # Any json-serializable payload provided to interrupt as argument.
            # It will be surfaced on the client side as an Interrupt when streaming data
            # from the workflow.
            "essay": essay,  # The essay we want reviewed.
            # We can add any additional information that we need.
            # For example, introduce a key called "action" with some instructions.
            "action": "Please approve/reject the essay",
        }
    )
    return {
        "essay": essay,  # The essay that was generated
        "is_approved": is_approved,  # Response from HIL
    }

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}
for item in workflow.stream("cat", config):
    print(item)
# > {'write_essay': 'An essay about topic: cat'}
# > {
# >     '__interrupt__': (
# >        Interrupt(
# >            value={
# >                'essay': 'An essay about topic: cat',
# >                'action': 'Please approve/reject the essay'
# >            },
# >            id='b9b2b9d788f482663ced6dc755c9e981'
# >        ),
# >    )
# > }
```

An essay has been written and is ready for review. Once the review is provided, we can resume the workflow:

```
from langgraph.types import Command

# Get review from a user (e.g., via a UI)
# In this case, we're using a bool, but this can be any json-serializable value.
human_review = True

for item in workflow.stream(Command(resume=human_review), config):
    print(item)
```

```
{'workflow': {'essay': 'An essay about topic: cat', 'is_approved': False}}
```

The workflow has been completed and the review has been added to the essay.

## Entrypoint

The [`@entrypoint`](https://docs.langchain.com/oss/python/langgraph/functional-api#entrypoint) decorator can be used to create a workflow from a function. It encapsulates workflow logic and manages execution flow, including handling _long-running tasks_ and [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts).

### Definition

An **entrypoint** is defined by decorating a function with the `@entrypoint` decorator.The function **must accept a single positional argument**, which serves as the workflow input. If you need to pass multiple pieces of data, use a dictionary as the input type for the first argument.Decorating a function with an `entrypoint` produces a [`Pregel`](https://reference.langchain.com/python/langgraph/pregel/#langgraph.pregel.Pregel.stream) instance which helps to manage the execution of the workflow (e.g., handles streaming, resumption, and checkpointing).You will usually want to pass a **checkpointer** to the `@entrypoint` decorator to enable persistence and use features like **human-in-the-loop**.

```
from langgraph.func import entrypoint

@entrypoint(checkpointer=checkpointer)
def my_workflow(some_input: dict) -> int:
    # some logic that may involve long-running tasks like API calls,
    # and may be interrupted for human-in-the-loop.
    ...
    return result
```

**Serialization**
The **inputs** and **outputs** of entrypoints must be JSON-serializable to support checkpointing. Please see the [serialization](https://docs.langchain.com/oss/python/langgraph/functional-api#serialization) section for more details.

### Injectable parameters

When declaring an `entrypoint`, you can request access to additional parameters that will be injected automatically at run time. These parameters include:

| Parameter | Description |
| --- | --- |
| **previous** | Access the state associated with the previous `checkpoint` for the given thread. See [short-term-memory](https://docs.langchain.com/oss/python/langgraph/functional-api#short-term-memory). |
| **store** | An instance of \[BaseStore\]\[langgraph.store.base.BaseStore\]. Useful for [long-term memory](https://docs.langchain.com/oss/python/langgraph/use-functional-api#long-term-memory). |
| **writer** | Use to access the StreamWriter when working with Async Python < 3.11. See [streaming with functional API for details](https://docs.langchain.com/oss/python/langgraph/use-functional-api#streaming). |
| **config** | For accessing run time configuration. See [RunnableConfig](https://python.langchain.com/docs/concepts/runnables/#runnableconfig) for information. |

Declare the parameters with the appropriate name and type annotation.

Requesting Injectable Parameters

```
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

in_memory_store = InMemoryStore(...)  # An instance of InMemoryStore for long-term memory

@entrypoint(
    checkpointer=checkpointer,  # Specify the checkpointer
    store=in_memory_store  # Specify the store
)
def my_workflow(
    some_input: dict,  # The input (e.g., passed via `invoke`)
    *,
    previous: Any = None, # For short-term memory
    store: BaseStore,  # For long-term memory
    writer: StreamWriter,  # For streaming custom data
    config: RunnableConfig  # For accessing the configuration passed to the entrypoint
) -> ...:
```

### Executing

Using the [`@entrypoint`](https://docs.langchain.com/oss/python/langgraph/functional-api#entrypoint) yields a [`Pregel`](https://reference.langchain.com/python/langgraph/pregel/#langgraph.pregel.Pregel.stream) object that can be executed using the `invoke`, `ainvoke`, `stream`, and `astream` methods.

```
config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}
my_workflow.invoke(some_input, config)  # Wait for the result synchronously
```

### Resuming

Resuming an execution after an [interrupt](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt) can be done by passing a **resume** value to the [`Command`](https://reference.langchain.com/python/langgraph/types/#langgraph.types.Command) primitive.

```
from langgraph.types import Command

config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}

my_workflow.invoke(Command(resume=some_resume_value), config)
```

**Resuming after an error**To resume after an error, run the `entrypoint` with a `None` and the same **thread id** (config).This assumes that the underlying **error** has been resolved and execution can proceed successfully.

```

config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}

my_workflow.invoke(None, config)
```

### Short-term memory

When an `entrypoint` is defined with a `checkpointer`, it stores information between successive invocations on the same **thread id** in [checkpoints](https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints).This allows accessing the state from the previous invocation using the `previous` parameter.By default, the `previous` parameter is the return value of the previous invocation.

```
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> int:
    previous = previous or 0
    return number + previous

config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}

my_workflow.invoke(1, config)  # 1 (previous was None)
my_workflow.invoke(2, config)  # 3 (previous was 1 from the previous invocation)
```

#### `entrypoint.final`

[`entrypoint.final`](https://reference.langchain.com/python/langgraph/func/#langgraph.func.entrypoint.final) is a special primitive that can be returned from an entrypoint and allows **decoupling** the value that is **saved in the checkpoint** from the **return value of the entrypoint**.The first value is the return value of the entrypoint, and the second value is the value that will be saved in the checkpoint. The type annotation is `entrypoint.final[return_type, save_type]`.

```
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    # This will return the previous value to the caller, saving
    # 2 * number to the checkpoint, which will be used in the next invocation
    # for the `previous` parameter.
    return entrypoint.final(value=previous, save=2 * number)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

my_workflow.invoke(3, config)  # 0 (previous was None)
my_workflow.invoke(1, config)  # 6 (previous was 3 * 2 from the previous invocation)
```

## Task

A **task** represents a discrete unit of work, such as an API call or data processing step. It has two key characteristics:

- **Asynchronous Execution**: Tasks are designed to be executed asynchronously, allowing multiple operations to run concurrently without blocking.
- **Checkpointing**: Task results are saved to a checkpoint, enabling resumption of the workflow from the last saved state. (See [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) for more details).

### Definition

Tasks are defined using the `@task` decorator, which wraps a regular Python function.

```
from langgraph.func import task

@task()
def slow_computation(input_value):
    # Simulate a long-running operation
    ...
    return result
```

**Serialization**
The **outputs** of tasks must be JSON-serializable to support checkpointing.

### Execution

**Tasks** can only be called from within an **entrypoint**, another **task**, or a [state graph node](https://docs.langchain.com/oss/python/langgraph/graph-api#nodes).Tasks _cannot_ be called directly from the main application code.When you call a **task**, it returns _immediately_ with a future object. A future is a placeholder for a result that will be available later.To obtain the result of a **task**, you can either wait for it synchronously (using `result()`) or await it asynchronously (using `await`).

```
@entrypoint(checkpointer=checkpointer)
def my_workflow(some_input: int) -> int:
    future = slow_computation(some_input)
    return future.result()  # Wait for the result synchronously
```

## When to use a task

**Tasks** are useful in the following scenarios:

- **Checkpointing**: When you need to save the result of a long-running operation to a checkpoint, so you don’t need to recompute it when resuming the workflow.
- **Human-in-the-loop**: If you’re building a workflow that requires human intervention, you MUST use **tasks** to encapsulate any randomness (e.g., API calls) to ensure that the workflow can be resumed correctly. See the [determinism](https://docs.langchain.com/oss/python/langgraph/functional-api#determinism) section for more details.
- **Parallel Execution**: For I/O-bound tasks, **tasks** enable parallel execution, allowing multiple operations to run concurrently without blocking (e.g., calling multiple APIs).
- **Observability**: Wrapping operations in **tasks** provides a way to track the progress of the workflow and monitor the execution of individual operations using [LangSmith](https://docs.smith.langchain.com/).
- **Retryable Work**: When work needs to be retried to handle failures or inconsistencies, **tasks** provide a way to encapsulate and manage the retry logic.

## Serialization

There are two key aspects to serialization in LangGraph:

1. `entrypoint` inputs and outputs must be JSON-serializable.
2. `task` outputs must be JSON-serializable.

These requirements are necessary for enabling checkpointing and workflow resumption. Use python primitives like dictionaries, lists, strings, numbers, and booleans to ensure that your inputs and outputs are serializable.Serialization ensures that workflow state, such as task results and intermediate values, can be reliably saved and restored. This is critical for enabling human-in-the-loop interactions, fault tolerance, and parallel execution.Providing non-serializable inputs or outputs will result in a runtime error when a workflow is configured with a checkpointer.

## Determinism

To utilize features like **human-in-the-loop**, any randomness should be encapsulated inside of **tasks**. This guarantees that when execution is halted (e.g., for human in the loop) and then resumed, it will follow the same _sequence of steps_, even if **task** results are non-deterministic.LangGraph achieves this behavior by persisting **task** and [**subgraph**](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) results as they execute. A well-designed workflow ensures that resuming execution follows the _same sequence of steps_, allowing previously computed results to be retrieved correctly without having to re-execute them. This is particularly useful for long-running **tasks** or **tasks** with non-deterministic results, as it avoids repeating previously done work and allows resuming from essentially the same.While different runs of a workflow can produce different results, resuming a **specific** run should always follow the same sequence of recorded steps. This allows LangGraph to efficiently look up **task** and **subgraph** results that were executed prior to the graph being interrupted and avoid recomputing them.

## Idempotency

Idempotency ensures that running the same operation multiple times produces the same result. This helps prevent duplicate API calls and redundant processing if a step is rerun due to a failure. Always place API calls inside **tasks** functions for checkpointing, and design them to be idempotent in case of re-execution. Re-execution can occur if a **task** starts, but does not complete successfully. Then, if the workflow is resumed, the **task** will run again. Use idempotency keys or verify existing results to avoid duplication.

## Common Pitfalls

### Handling side effects

Encapsulate side effects (e.g., writing to a file, sending an email) in tasks to ensure they are not executed multiple times when resuming a workflow.

In this example, a side effect (writing to a file) is directly included in the workflow, so it will be executed a second time when resuming the workflow.

```
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    # This code will be executed a second time when resuming the workflow.
    # Which is likely not what you want.
    with open("output.txt", "w") as f:
        f.write("Side effect executed")
    value = interrupt("question")
    return value
```

### Non-deterministic control flow

Operations that might give different results each time (like getting current time or random numbers) should be encapsulated in tasks to ensure that on resume, the same result is returned.

- In a task: Get random number (5) → interrupt → resume → (returns 5 again) → …
- Not in a task: Get random number (5) → interrupt → resume → get new random number (7) → …

This is especially important when using **human-in-the-loop** workflows with multiple interrupts calls. LangGraph keeps a list of resume values for each task/entrypoint. When an interrupt is encountered, it’s matched with the corresponding resume value. This matching is strictly **index-based**, so the order of the resume values should match the order of the interrupts.If order of execution is not maintained when resuming, one [`interrupt`](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt) call may be matched with the wrong `resume` value, leading to incorrect results.Please read the section on [determinism](https://docs.langchain.com/oss/python/langgraph/functional-api#determinism) for more details.

In this example, the workflow uses the current time to determine which task to execute. This is non-deterministic because the result of the workflow depends on the time at which it is executed.

```
from langgraph.func import entrypoint

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    t0 = inputs["t0"]
    t1 = time.time()

    delta_t = t1 - t0

    if delta_t > 1:
        result = slow_task(1).result()
        value = interrupt("question")
    else:
        result = slow_task(2).result()
        value = interrupt("question")

    return {
        "result": result,
        "value": value
    }
```

</details>

<details>
<summary>reflection-agents</summary>

Reflection is a prompting strategy used to improve the quality and success rate of agents and similar AI systems. It involves prompting an LLM to reflect on and critique its past actions, sometimes incorporating additional external information such as tool observations.

People like to talk about "System 1" and "System 2" thinking, where System 1 is reactive or instinctive and System 2 is more methodical and reflective. When applied correctly, reflection can help LLM systems break out of purely System 1 "thinking" patterns and closer to something exhibiting System 2-like behavior.

Reflection takes time! All the approaches in this post trade off a bit of extra compute for a shot at better output quality. While this may not be appropriate for low-latency applications, it _is_ worthwhile for knowledge intensive tasks where response _quality_ is more important than speed.

The three examples are outlined below:

## Basic Reflection

This simple example composes two LLM calls: a generator and a reflector. The generator tries to respond directly to the user's requests. The reflector is prompted to role play as a teacher and offer constructive criticism for the initial response.

The loop proceeds a fixed number of times, and the final generated output is returned.

https://blog.langchain.com/content/images/2024/02/reflection.png
Simple Reflection Loop

We can define the loop in LangGraph below:

```python
from langgraph.graph import MessageGraph

builder = MessageGraph()
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("generate")

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return "reflect"

builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
graph = builder.compile()
```

The `MessageGraph` represents a stateful graph, where the "state" is simply a list of messages. Each time the generator or reflector node is called, it appends a message to the end of the state. The final result is returned from the generator node.

This simple type of reflection can sometimes improve performance by giving the LLM multiple attempts at refining its output and by letting the reflection node adopt a different persona while critiquing the output.

However, since the reflection step isn't grounded in any external process, the final result may not be significantly better than the original. Let's explore some other techniques that can ameliorate that.

## Reflexion

[Reflexion](https://arxiv.org/abs/2303.11366?ref=blog.langchain.com) by Shinn, et. al., is an architecture designed to learn through verbal feedback and self-reflection. Within reflexion, the _actor_ agent explicitly critiques each response and grounds its criticism in external data. It is forced to generate citations and explicitly enumerate superfluous and missing aspects of the generated response. This makes the content of the reflections more constructive and better steers the generator in responding to the feedback.

In the [linked](https://github.com/langchain-ai/langgraph/blob/main/examples/reflexion/reflexion.ipynb?ref=blog.langchain.com) example, we stop after a fixed number of steps, though you can also offload this decision to the reflection LLM call.

An overview of the agent loop is shown below:

https://blog.langchain.com/content/images/2024/02/reflexion.png
Reflexion Actor Overview

For each step, the responder is tasked with generating a response, along with additional actions in the form of search queries. Then the revisor is prompted to reflect on the current state. The logic can be defined in LangGraph as follows:

```python
from langgraph.graph import END, MessageGraph

MAX_ITERATIONS = 5
builder = MessageGraph()
builder.add_node("draft", first_responder.respond)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor.respond)
# draft -> execute_tools
builder.add_edge("draft", "execute_tools")
# execute_tools -> revise
builder.add_edge("execute_tools", "revise")

# Define looping logic:
def event_loop(state: List[BaseMessage]) -> str:
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state)
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

# revise -> execute_tools OR end
builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")
graph = builder.compile()
```

This agent can effectively use explicit reflections and web-based citations to improve the quality of the final response. It only pursues one fixed trajectory, however, so if it makes a misstep, that error can impact subsequent decisions.

## Language Agent Tree Search

[Language Agent Tree Search](https://arxiv.org/abs/2310.04406?ref=blog.langchain.com) (LATS), by Zhou, et. al, is a general LLM agent search algorithm that combines reflection/evaluation and search (specifically Monte-Carlo trees search) to achieve better overall task performance compared to similar techniques like ReACT, Reflexion, or even Tree of Thoughts. It adopts a standard reinforcement learning (RL) task framing, replacing the RL agents, value functions, and optimizer all with calls to an LLM. This is meant to help the agent adapt and problem solve for complex tasks, avoiding getting stuck in repetitive loops.

The search process is outlined in the diagram below:

https://blog.langchain.com/content/images/2024/02/lats.png

The search has four main steps:

1.  **Select**: pick the best next actions based on the aggregate rewards from step (2) below. Either respond (if a solution is found or the max search depth is reached) or continue searching.
2.  **Expand and simulate:** generate N (5 in our case) potential actions to take and execute them in parallel.
3.  **Reflect + evaluate**: observe the outcomes of these actions and score the decisions based on reflection (and possibly external feedback)
4.  **Backpropagate**: update the scores of the root trajectories based on the outcomes.

If the agent has a tight feedback loop (through high quality environment rewards or reliable reflection scores), the search is able to accurately distinguish between different action trajectories and pick the best path. The final trajectory can then be saved to external memory (or used for model fine-tuning) to improve the model in the future.

The "selection" step picks the node with the highest upper confidence bound (UCT), which just balances the expected reward (the first term) with an incentive to explore new paths (the second term).

$$
UCT = \frac{\text{value}}{\text{visits}} + c \sqrt{\frac{\ln(\text{parent.visits})}{\text{visits}}}
$$

Check out the [code](https://github.com/langchain-ai/langgraph/blob/main/examples/lats/lats.ipynb?ref=blog.langchain.com) to see how it's implemented. In our LangGraph implementation, we put generation + reflection steps in a single node each, and check the tree state on each loop to see if the task is solved. The (abbreviated) graph definition looks something like below:

```python
from langgraph.graph import END, StateGraph

class Node:
    def __init__(
        self,
        messages: List[BaseMessage],
        reflection: Reflection,
        parent: Optional[Node] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
    # Additional methods are defined here. Check the code for more!

class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str

def should_loop(state: TreeState):
    """Determine whether to continue the tree search."""
    root = state["root"]
    if root.is_solved:
        return END
    if root.height > 5:
        return END
    return "expand"

builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand)
builder.set_entry_point("start")

builder.add_conditional_edges(
    "start",
    # Either expand/rollout or finish
    should_loop,
)
builder.add_conditional_edges(
    "expand",
    # Either continue to rollout or finish
    should_loop,
)

graph = builder.compile()
```

LATS Graph

Once you've created the basic outline, it's easy to expand to other tasks! For instance, this technique would suit code generation tasks well, where you the agent can write explicit unit tests and score trajectories based on test quality.

LATS unifies the reasoning, planning, and reflection components of other agent architectures, such as Reflexion, Tree of Thoughts, and [plan-and-execute](https://blog.langchain.com/planning-agents/) agents. LATS also from the backpropagation of reflective and environment-based feedback for an improved search process. While it can be sensitive to the reward scores, the general algorithm can be flexibly applied to a variety of tasks.

https://blog.langchain.com/content/images/2024/02/image-17.png
Comparison of LATS with other agent architectures

## Video Tutorial

Reflection Agents - YouTube

LangChain

161K subscribers

Reflection Agents

LangChain

## Conclusion

Thanks for reading! All of these examples can be found in the [LangGraph](https://github.com/langchain-ai/langgraph/tree/main?ref=blog.langchain.com) repository, and we will port these to [LangGraphJS](https://github.com/langchain-ai/langgraphjs?ref=blog.langchain.com) soon (maybe by the time you read this post).

All of the techniques above leverage additional LLM inference to increase the likelihood of generating a higher quality output, or of responding correctly to a more complex reasoning task. While this takes extra time, it can be appropriate when output quality matters more than response time, and if you save the trajectories to memory (or as [fine-tuning data](https://docs.smith.langchain.com/tracing/use_cases/few-shot-datasets?ref=blog.langchain.com)), you can update the model to avoid repeat mistakes in the future.

</details>

<details>
<summary>use-the-functional-api-docs-by-langchain</summary>

```
The Functional API allows you to add LangGraph’s key features — [persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [memory](https://docs.langchain.com/oss/python/langgraph/add-memory), [human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/interrupts), and [streaming](https://docs.langchain.com/oss/python/langgraph/streaming) — to your applications with minimal changes to your existing code.

For conceptual information on the functional API, see [Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api).

## Creating a simple workflow

When defining an `entrypoint`, input is restricted to the first argument of the function. To pass multiple inputs, you can use a dictionary.

Copy

```
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    value = inputs["value"]
    another_value = inputs["another_value"]
    ...

my_workflow.invoke({"value": 1, "another_value": 2})
```

Extended example: simple workflow

Copy

```
import uuid
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

# Task that checks if a number is even
@task
def is_even(number: int) -> bool:
    return number % 2 == 0

# Task that formats a message
@task
def format_message(is_even: bool) -> str:
    return "The number is even." if is_even else "The number is odd."

# Create a checkpointer for persistence
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(inputs: dict) -> str:
    """Simple workflow to classify a number."""
    even = is_even(inputs["number"]).result()
    return format_message(even).result()

# Run the workflow with a unique thread ID
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = workflow.invoke({"number": 7}, config=config)
print(result)
```

Extended example: Compose an essay with an LLM

This example demonstrates how to use the `@task` and `@entrypoint` decorators
syntactically. Given that a checkpointer is provided, the workflow results will
be persisted in the checkpointer.

Copy

```
import uuid
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

model = init_chat_model('gpt-3.5-turbo')

# Task: generate essay using an LLM
@task
def compose_essay(topic: str) -> str:
    """Generate an essay about the given topic."""
    return model.invoke([\
        {"role": "system", "content": "You are a helpful assistant that writes essays."},\
        {"role": "user", "content": f"Write an essay about {topic}."}\
    ]).content

# Create a checkpointer for persistence
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(topic: str) -> str:
    """Simple workflow that generates an essay with an LLM."""
    return compose_essay(topic).result()

# Execute the workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = workflow.invoke("the history of flight", config=config)
print(result)
```

## Parallel execution

Tasks can be executed in parallel by invoking them concurrently and waiting for the results. This is useful for improving performance in IO bound tasks (e.g., calling APIs for LLMs).

Copy

```
@task
def add_one(number: int) -> int:
    return number + 1

@entrypoint(checkpointer=checkpointer)
def graph(numbers: list[int]) -> list[str]:
    futures = [add_one(i) for i in numbers]
    return [f.result() for f in futures]
```

Extended example: parallel LLM calls

This example demonstrates how to run multiple LLM calls in parallel using `@task`. Each call generates a paragraph on a different topic, and results are joined into a single text output.

Copy

```
import uuid
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

# Initialize the LLM model
model = init_chat_model("gpt-3.5-turbo")

# Task that generates a paragraph about a given topic
@task
def generate_paragraph(topic: str) -> str:
    response = model.invoke([\
        {"role": "system", "content": "You are a helpful assistant that writes educational paragraphs."},\
        {"role": "user", "content": f"Write a paragraph about {topic}."}\
    ])
    return response.content

# Create a checkpointer for persistence
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(topics: list[str]) -> str:
    """Generates multiple paragraphs in parallel and combines them."""
    futures = [generate_paragraph(topic) for topic in topics]
    paragraphs = [f.result() for f in futures]
    return "\n\n".join(paragraphs)

# Run the workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = workflow.invoke(["quantum computing", "climate change", "history of aviation"], config=config)
print(result)
```

This example uses LangGraph’s concurrency model to improve execution time, especially when tasks involve I/O like LLM completions.

## Calling graphs

The **Functional API** and the [**Graph API**](https://docs.langchain.com/oss/python/langgraph/graph-api) can be used together in the same application as they share the same underlying runtime.

Copy

```
from langgraph.func import entrypoint
from langgraph.graph import StateGraph

builder = StateGraph()
...
some_graph = builder.compile()

@entrypoint()
def some_workflow(some_input: dict) -> int:
    # Call a graph defined using the graph API
    result_1 = some_graph.invoke(...)
    # Call another graph defined using the graph API
    result_2 = another_graph.invoke(...)
    return {
        "result_1": result_1,
        "result_2": result_2
    }
```

Extended example: calling a simple graph from the functional API

Copy

```
import uuid
from typing import TypedDict
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

# Define the shared state type
class State(TypedDict):
    foo: int

# Define a simple transformation node
def double(state: State) -> State:
    return {"foo": state["foo"] * 2}

# Build the graph using the Graph API
builder = StateGraph(State)
builder.add_node("double", double)
builder.set_entry_point("double")
graph = builder.compile()

# Define the functional API workflow
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(x: int) -> dict:
    result = graph.invoke({"foo": x})
    return {"bar": result["foo"]}

# Execute the workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
print(workflow.invoke(5, config=config))  # Output: {'bar': 10}
```

## Call other entrypoints

You can call other **entrypoints** from within an **entrypoint** or a **task**.

Copy

```
@entrypoint() # Will automatically use the checkpointer from the parent entrypoint
def some_other_workflow(inputs: dict) -> int:
    return inputs["value"]

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    value = some_other_workflow.invoke({"value": 1})
    return value
```

Extended example: calling another entrypoint

Copy

```
import uuid
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

# Initialize a checkpointer
checkpointer = InMemorySaver()

# A reusable sub-workflow that multiplies a number
@entrypoint()
def multiply(inputs: dict) -> int:
    return inputs["a"] * inputs["b"]

# Main workflow that invokes the sub-workflow
@entrypoint(checkpointer=checkpointer)
def main(inputs: dict) -> dict:
    result = multiply.invoke({"a": inputs["x"], "b": inputs["y"]})
    return {"product": result}

# Execute the main workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
print(main.invoke({"x": 6, "y": 7}, config=config))  # Output: {'product': 42}
```

## Streaming

The **Functional API** uses the same streaming mechanism as the **Graph API**. Please
read the [**streaming guide**](https://docs.langchain.com/oss/python/langgraph/streaming) section for more details.Example of using the streaming API to stream both updates and custom data.

Copy

```
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def main(inputs: dict) -> int:
    writer = get_stream_writer()
    writer("Started processing")
    result = inputs["x"] * 2
    writer(f"Result is {result}")
    return result

config = {"configurable": {"thread_id": "abc"}}

for mode, chunk in main.stream(
    {"x": 5},
    stream_mode=["custom", "updates"],
    config=config
):
    print(f"{mode}: {chunk}")
```

1. Import [`get_stream_writer`](https://reference.langchain.com/python/langgraph/config/#langgraph.config.get_stream_writer) from `langgraph.config`.
2. Obtain a stream writer instance within the entrypoint.
3. Emit custom data before computation begins.
4. Emit another custom message after computing the result.
5. Use `.stream()` to process streamed output.
6. Specify which streaming modes to use.

Copy

```
('updates', {'add_one': 2})
('updates', {'add_two': 3})
('custom', 'hello')
('custom', 'world')
('updates', {'main': 5})
```

**Async with Python < 3.11**
If using Python < 3.11 and writing async code, using [`get_stream_writer`](https://reference.langchain.com/python/langgraph/config/#langgraph.config.get_stream_writer) will not work. Instead please
use the `StreamWriter` class directly. See [Async with Python < 3.11](https://docs.langchain.com/oss/python/langgraph/streaming#async) for more details.

Copy

```
from langgraph.types import StreamWriter

@entrypoint(checkpointer=checkpointer)
async def main(inputs: dict, writer: StreamWriter) -> int:
...
```

## Retry policy

Copy

```
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

# This variable is just used for demonstration purposes to simulate a network failure.
# It's not something you will have in your actual code.
attempts = 0

# Let's configure the RetryPolicy to retry on ValueError.
# The default RetryPolicy is optimized for retrying specific network errors.
retry_policy = RetryPolicy(retry_on=ValueError)

@task(retry_policy=retry_policy)
def get_info():
    global attempts
    attempts += 1

    if attempts < 2:
        raise ValueError('Failure')
    return "OK"

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def main(inputs, writer):
    return get_info().result()

config = {
    "configurable": {
        "thread_id": "1"
    }
}

main.invoke({'any_input': 'foobar'}, config=config)
```

Copy

```
'OK'
```

## Caching Tasks

Copy

```
import time
from langgraph.cache.memory import InMemoryCache
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy

@task(cache_policy=CachePolicy(ttl=120))
def slow_add(x: int) -> int:
    time.sleep(1)
    return x * 2

@entrypoint(cache=InMemoryCache())
def main(inputs: dict) -> dict[str, int]:
    result1 = slow_add(inputs["x"]).result()
    result2 = slow_add(inputs["x"]).result()
    return {"result1": result1, "result2": result2}

for chunk in main.stream({"x": 5}, stream_mode="updates"):
    print(chunk)

#> {'slow_add': 10}
#> {'slow_add': 10, '__metadata__': {'cached': True}}
#> {'main': {'result1': 10, 'result2': 10}}
```

1. `ttl` is specified in seconds. The cache will be invalidated after this time.

## Resuming after an error

Copy

```
import time
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import StreamWriter

# This variable is just used for demonstration purposes to simulate a network failure.
# It's not something you will have in your actual code.
attempts = 0

@task()
def get_info():
    """
    Simulates a task that fails once before succeeding.
    Raises an exception on the first attempt, then returns "OK" on subsequent tries.
    """
    global attempts
    attempts += 1

    if attempts < 2:
        raise ValueError("Failure")  # Simulate a failure on the first attempt
    return "OK"

# Initialize an in-memory checkpointer for persistence
checkpointer = InMemorySaver()

@task
def slow_task():
    """
    Simulates a slow-running task by introducing a 1-second delay.
    """
    time.sleep(1)
    return "Ran slow task."

@entrypoint(checkpointer=checkpointer)
def main(inputs, writer: StreamWriter):
    """
    Main workflow function that runs the slow_task and get_info tasks sequentially.

    Parameters:
    - inputs: Dictionary containing workflow input values.
    - writer: StreamWriter for streaming custom data.

    The workflow first executes `slow_task` and then attempts to execute `get_info`,
    which will fail on the first invocation.
    """
    slow_task_result = slow_task().result()  # Blocking call to slow_task
    get_info().result()  # Exception will be raised here on the first attempt
    return slow_task_result

# Workflow execution configuration with a unique thread identifier
config = {
    "configurable": {
        "thread_id": "1"  # Unique identifier to track workflow execution
    }
}

# This invocation will take ~1 second due to the slow_task execution
try:
    # First invocation will raise an exception due to the `get_info` task failing
    main.invoke({'any_input': 'foobar'}, config=config)
except ValueError:
    pass  # Handle the failure gracefully
```

When we resume execution, we won’t need to re-run the `slow_task` as its result is already saved in the checkpoint.

Copy

```
main.invoke(None, config=config)
```

Copy

```
'Ran slow task.'
```

## Human-in-the-loop

The functional API supports [human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/interrupts) workflows using the [`interrupt`](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt) function and the `Command` primitive.

### Basic human-in-the-loop workflow

We will create three [tasks](https://docs.langchain.com/oss/python/langgraph/functional-api#task):

1. Append `"bar"`.
2. Pause for human input. When resuming, append human input.
3. Append `"qux"`.

Copy

```
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt

@task
def step_1(input_query):
    """Append bar."""
    return f"{input_query} bar"

@task
def human_feedback(input_query):
    """Append user input."""
    feedback = interrupt(f"Please provide feedback: {input_query}")
    return f"{input_query} {feedback}"

@task
def step_3(input_query):
    """Append qux."""
    return f"{input_query} qux"
```

We can now compose these tasks in an [entrypoint](https://docs.langchain.com/oss/python/langgraph/functional-api#entrypoint):

Copy

```
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def graph(input_query):
    result_1 = step_1(input_query).result()
    result_2 = human_feedback(result_1).result()
    result_3 = step_3(result_2).result()

    return result_3
```

[interrupt()](https://docs.langchain.com/oss/python/langgraph/interrupts#pause-using-interrupt) is called inside a task, enabling a human to review and edit the output of the previous task. The results of prior tasks— in this case `step_1`— are persisted, so that they are not run again following the [`interrupt`](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt).Let’s send in a query string:

Copy

```
config = {"configurable": {"thread_id": "1"}}

for event in graph.stream("foo", config):
    print(event)
    print("\n")
```

Note that we’ve paused with an [`interrupt`](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt) after `step_1`. The interrupt provides instructions to resume the run. To resume, we issue a [`Command`](https://docs.langchain.com/oss/python/langgraph/interrupts#resuming-interrupts) containing the data expected by the `human_feedback` task.

Copy

```
# Continue execution
for event in graph.stream(Command(resume="baz"), config):
    print(event)
    print("\n")
```

After resuming, the run proceeds through the remaining step and terminates as expected.

### Review tool calls

To review tool calls before execution, we add a `review_tool_call` function that calls [`interrupt`](https://docs.langchain.com/oss/python/langgraph/interrupts#pause-using-interrupt). When this function is called, execution will be paused until we issue a command to resume it.Given a tool call, our function will [`interrupt`](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt) for human review. At that point we can either:

- Accept the tool call
- Revise the tool call and continue
- Generate a custom tool message (e.g., instructing the model to re-format its tool call)

Copy

```
from typing import Union

def review_tool_call(tool_call: ToolCall) -> Union[ToolCall, ToolMessage]:
    """Review a tool call, returning a validated version."""
    human_review = interrupt(
        {
            "question": "Is this correct?",
            "tool_call": tool_call,
        }
    )
    review_action = human_review["action"]
    review_data = human_review.get("data")
    if review_action == "continue":
        return tool_call
    elif review_action == "update":
        updated_tool_call = {**tool_call, **{"args": review_data}}
        return updated_tool_call
    elif review_action == "feedback":
        return ToolMessage(
            content=review_data, name=tool_call["name"], tool_call_id=tool_call["id"]
        )
```

We can now update our [entrypoint](https://docs.langchain.com/oss/python/langgraph/functional-api#entrypoint) to review the generated tool calls. If a tool call is accepted or revised, we execute in the same way as before. Otherwise, we just append the [`ToolMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolMessage) supplied by the human. The results of prior tasks — in this case the initial model call — are persisted, so that they are not run again following the [`interrupt`](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt).

Copy

```
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def agent(messages, previous):
    if previous is not None:
        messages = add_messages(previous, messages)

    model_response = call_model(messages).result()
    while True:
        if not model_response.tool_calls:
            break

        # Review tool calls
        tool_results = []
        tool_calls = []
        for i, tool_call in enumerate(model_response.tool_calls):
            review = review_tool_call(tool_call)
            if isinstance(review, ToolMessage):
                tool_results.append(review)
            else:  # is a validated tool call
                tool_calls.append(review)
                if review != tool_call:
                    model_response.tool_calls[i] = review  # update message

        # Execute remaining tool calls
        tool_result_futures = [call_tool(tool_call) for tool_call in tool_calls]
        remaining_tool_results = [fut.result() for fut in tool_result_futures]

        # Append to message list
        messages = add_messages(
            messages,
            [model_response, *tool_results, *remaining_tool_results],
        )

        # Call model again
        model_response = call_model(messages).result()

    # Generate final response
    messages = add_messages(messages, model_response)
    return entrypoint.final(value=model_response, save=messages)
```

## Short-term memory

Short-term memory allows storing information across different **invocations** of the same **thread id**. See [short-term memory](https://docs.langchain.com/oss/python/langgraph/functional-api#short-term-memory) for more details.

### Manage checkpoints

You can view and delete the information stored by the checkpointer.

#### View thread state

Copy

```
config = {
    "configurable": {
        "thread_id": "1",
        # optionally provide an ID for a specific checkpoint,
        # otherwise the latest checkpoint is shown
        # "checkpoint_id": "1f029ca3-1f5b-6704-8004-820c16b69a5a"  #

    }
}
graph.get_state(config)
```

Copy

```
StateSnapshot(
    values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today?), HumanMessage(content="what's my name?"), AIMessage(content='Your name is Bob.')]}, next=(),
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1f5b-6704-8004-820c16b69a5a'}},
    metadata={
        'source': 'loop',
        'writes': {'call_model': {'messages': AIMessage(content='Your name is Bob.')}},
        'step': 4,
        'parents': {},
        'thread_id': '1'
    },
    created_at='2025-05-05T16:01:24.680462+00:00',
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},
    tasks=(),
    interrupts=()
)
```

#### View the history of the thread

Copy

```
config = {
    "configurable": {
        "thread_id": "1"
    }
}
list(graph.get_state_history(config))
```

Copy

```
[\
    StateSnapshot(\
        values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?'), HumanMessage(content="what's my name?"), AIMessage(content='Your name is Bob.')]},\
        next=(),\
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1f5b-6704-8004-820c16b69a5a'}},\
        metadata={'source': 'loop', 'writes': {'call_model': {'messages': AIMessage(content='Your name is Bob.')}}, 'step': 4, 'parents': {}, 'thread_id': '1'},\
        created_at='2025-05-05T16:01:24.680462+00:00',\
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},\
        tasks=(),\
        interrupts=()\
    ),\
    StateSnapshot(\
        values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?'), HumanMessage(content="what's my name?")]},\
        next=('call_model',),\
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},\
        metadata={'source': 'loop', 'writes': None, 'step': 3, 'parents': {}, 'thread_id': '1'},\
        created_at='2025-05-05T16:01:23.863421+00:00',\
        parent_config={...}\
        tasks=(PregelTask(id='8ab4155e-6b15-b885-9ce5-bed69a2c305c', name='call_model', path=('__pregel_pull', 'call_model'), error=None, interrupts=(), state=None, result={'messages': AIMessage(content='Your name is Bob.')}),),\
        interrupts=()\
    ),\
    StateSnapshot(\
        values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')]},\
        next=('__start__',),\
        config={...},\
        metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': "what's my name?"}]}}, 'step': 2, 'parents': {}, 'thread_id': '1'},\
        created_at='2025-05-05T16:01:23.863173+00:00',\
        parent_config={...}\
        tasks=(PregelTask(id='24ba39d6-6db1-4c9b-f4c5-682aeaf38dcd', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'messages': [{'role': 'user', 'content': "what's my name?"}]}),),\
        interrupts=()\
    ),\
    StateSnapshot(\
        values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')]},\
        next=(),\
        config={...},\
        metadata={'source': 'loop', 'writes': {'call_model': {'messages': AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')}}, 'step': 1, 'parents': {}, 'thread_id': '1'},\
        created_at='2025-05-05T16:01:23.862295+00:00',\
        parent_config={...}\
        tasks=(),\
        interrupts=()\
    ),\
    StateSnapshot(\
        values={'messages': [HumanMessage(content="hi! I'm bob")]},\
        next=('call_model',),\
        config={...},\
        metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}, 'thread_id': '1'},\
        created_at='2025-05-05T16:01:22.278960+00:00',\
        parent_config={...}\
        tasks=(PregelTask(id='8cbd75e0-3720-b056-04f7-71ac805140a0', name='call_model', path=('__pregel_pull', 'call_model'), error=None, interrupts=(), state=None, result={'messages': AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')}),),\
        interrupts=()\
    ),\
    StateSnapshot(\
        values={'messages': []},\
        next=('__start__',),\
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-0870-6ce2-bfff-1f3f14c3e565'}},\
        metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': "hi! I'm bob"}]}}, 'step': -1, 'parents': {}, 'thread_id': '1'},\
        created_at='2025-05-05T16:01:22.277497+00:00',\
        parent_config=None,\
        tasks=(PregelTask(id='d458367b-8265-812c-18e2-33001d199ce6', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'messages': [{'role': 'user', 'content': "hi! I'm bob"}]}),),\
        interrupts=()\
    )\
]
```

### Decouple return value from saved value

Use `entrypoint.final` to decouple what is returned to the caller from what is persisted in the checkpoint. This is useful when:

- You want to return a computed result (e.g., a summary or status), but save a different internal value for use on the next invocation.
- You need to control what gets passed to the previous parameter on the next run.

Copy

```
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def accumulate(n: int, *, previous: int | None) -> entrypoint.final[int, int]:
    previous = previous or 0
    total = previous + n
    # Return the *previous* value to the caller but save the *new* total to the checkpoint.
    return entrypoint.final(value=previous, save=total)

config = {"configurable": {"thread_id": "my-thread"}}

print(accumulate.invoke(1, config=config))  # 0
print(accumulate.invoke(2, config=config))  # 1
print(accumulate.invoke(3, config=config))  # 3
```

### Chatbot example

An example of a simple chatbot using the functional API and the [`InMemorySaver`](https://reference.langchain.com/python/langgraph/checkpoints/#langgraph.checkpoint.memory.InMemorySaver) checkpointer.The bot is able to remember the previous conversation and continue from where it left off.

Copy

```
from langchain.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

@task
def call_model(messages: list[BaseMessage]):
    response = model.invoke(messages)
    return response

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(inputs: list[BaseMessage], *, previous: list[BaseMessage]):
    if previous:
        inputs = add_messages(previous, inputs)

    response = call_model(inputs).result()
    return entrypoint.final(value=response, save=add_messages(inputs, response))

config = {"configurable": {"thread_id": "1"}}
input_message = {"role": "user", "content": "hi! I'm bob"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()

input_message = {"role": "user", "content": "what's my name?"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```

## Long-term memory

[long-term memory](https://docs.langchain.com/oss/python/concepts/memory#long-term-memory) allows storing information across different **thread ids**. This could be useful for learning information about a given user in one conversation and using it in another.

## Workflows

- [Workflows and agent](https://docs.langchain.com/oss/python/langgraph/workflows-agents) guide for more examples of how to build workflows using the Functional API.

<h2>Integrate with other libraries</h2>

- [Add LangGraph’s features to other frameworks using the functional API](https://docs.langchain.com/langsmith/autogen-integration): Add LangGraph features like persistence, memory and streaming to other agent frameworks that do not provide them out of the box.
```

</details>

<details>
<summary>www-anthropic-com</summary>

# Building effective agents

We've worked with dozens of teams building LLM agents across industries. Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks.

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.

## What are agents?

"Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:

- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
- **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Below, we will explore both types of agentic systems in detail.

## When (and when not) to use agents

When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

When more complexity is warranted, workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale. For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough.

## When and how to use frameworks

There are many frameworks that make agentic systems easier to implement, including:

- The [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview);
- Amazon Bedrock's [AI Agent framework](https://aws.amazon.com/bedrock/agents/);
- [Rivet](https://rivet.ironcladapp.com/), a drag and drop GUI LLM workflow builder; and
- [Vellum](https://www.vellum.ai/), another GUI tool for building and testing complex workflows.

These frameworks make it easy to get started by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together. However, they often create extra layers of abstraction that can obscure the underlying prompts ​​and responses, making them harder to debug. They can also make it tempting to add complexity when a simpler setup would suffice.

We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error.

See our [cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents) for some sample implementations.

## Building blocks, workflows, and agents

In this section, we’ll explore the common patterns for agentic systems we’ve seen in production. We'll start with our foundational building block—the augmented LLM—and progressively increase complexity, from simple compositional workflows to autonomous agents.

### Building block: The augmented LLM

The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory. Our current models can actively use these capabilities—generating their own search queries, selecting appropriate tools, and determining what information to retain.

The augmented LLM

We recommend focusing on two key aspects of the implementation: tailoring these capabilities to your specific use case and ensuring they provide an easy, well-documented interface for your LLM. While there are many ways to implement these augmentations, one approach is through our recently released [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol), which allows developers to integrate with a growing ecosystem of third-party tools with a simple [client implementation](https://modelcontextprotocol.io/tutorials/building-a-client#building-mcp-clients).

For the remainder of this post, we'll assume each LLM call has access to these augmented capabilities.

### Workflow: Prompt chaining

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.

The prompt chaining workflow

**When to use this workflow:** This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

**Examples where prompt chaining is useful:**

- Generating Marketing copy, then translating it into a different language.
- Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

### Workflow: Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.

The routing workflow

**When to use this workflow:** Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

**Examples where routing is useful:**

- Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes, prompts, and tools.
- Routing easy/common questions to smaller, cost-efficient models like Claude Haiku 4.5 and hard/unusual questions to more capable models like Claude Sonnet 4.5 to optimize for best performance.

### Workflow: Parallelization

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations:

- **Sectioning**: Breaking a task into independent subtasks run in parallel.
- **Voting:** Running the same task multiple times to get diverse outputs.

The parallelization workflow

**When to use this workflow:** Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.

**Examples where parallelization is useful:**

- **Sectioning**:
  - Implementing guardrails where one model instance processes user queries while another screens them for inappropriate content or requests. This tends to perform better than having the same LLM call handle both guardrails and the core response.
  - Automating evals for evaluating LLM performance, where each LLM call evaluates a different aspect of the model’s performance on a given prompt.
- **Voting**:
  - Reviewing a piece of code for vulnerabilities, where several different prompts review and flag the code if they find a problem.
  - Evaluating whether a given piece of content is inappropriate, with multiple prompts evaluating different aspects or requiring different vote thresholds to balance false positives and negatives.

### Workflow: Orchestrator-workers

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

The orchestrator-workers workflow

**When to use this workflow:** This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

**Example where orchestrator-workers is useful:**

- Coding products that make complex changes to multiple files each time.
- Search tasks that involve gathering and analyzing information from multiple sources for possible relevant information.

### Workflow: Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.

The evaluator-optimizer workflow

**When to use this workflow:** This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

**Examples where evaluator-optimizer is useful:**

- Literary translation where there are nuances that the translator LLM might not capture initially, but where an evaluator LLM can provide useful critiques.
- Complex search tasks that require multiple rounds of searching and analysis to gather comprehensive information, where the evaluator decides whether further searches are warranted.

### Agents

Agents are emerging in production as LLMs mature in key capabilities—understanding complex inputs, engaging in reasoning and planning, using tools reliably, and recovering from errors. Agents begin their work with either a command from, or interactive discussion with, the human user. Once the task is clear, agents plan and operate independently, potentially returning to the human for further information or judgement. During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress. Agents can then pause for human feedback at checkpoints or when encountering blockers. The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully. We expand on best practices for tool development in Appendix 2 ("Prompt Engineering your Tools").

Autonomous agent

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

**Examples where agents are useful:**

The following examples are from our own implementations:

- A coding Agent to resolve [SWE-bench tasks](https://www.anthropic.com/research/swe-bench-sonnet), which involve edits to many files based on a task description;
- Our [“computer use” reference implementation](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo), where Claude uses a computer to accomplish tasks.

High-level flow of a coding agent

## Combining and customizing these patterns

These building blocks aren't prescriptive. They're common patterns that developers can shape and combine to fit different use cases. The key to success, as with any LLM features, is measuring performance and iterating on implementations. To repeat: you should consider adding complexity _only_ when it demonstrably improves outcomes.

## Summary

Success in the LLM space isn't about building the most sophisticated system. It's about building the _right_ system for your needs. Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short.

When implementing agents, we try to follow three core principles:

1. Maintain **simplicity** in your agent's design.
2. Prioritize **transparency** by explicitly showing the agent’s planning steps.
3. Carefully craft your agent-computer interface (ACI) through thorough tool **documentation and testing**.

Frameworks can help you get started quickly, but don't hesitate to reduce abstraction layers and build with basic components as you move to production. By following these principles, you can create agents that are not only powerful but also reliable, maintainable, and trusted by their users.

### Acknowledgements

Written by Erik Schluntz and Barry Zhang. This work draws upon our experiences building agents at Anthropic and the valuable insights shared by our customers, for which we're deeply grateful.

## Appendix 1: Agents in practice

Our work with customers has revealed two particularly promising applications for AI agents that demonstrate the practical value of the patterns discussed above. Both applications illustrate how agents add the most value for tasks that require both conversation and action, have clear success criteria, enable feedback loops, and integrate meaningful human oversight.

### A. Customer support

Customer support combines familiar chatbot interfaces with enhanced capabilities through tool integration. This is a natural fit for more open-ended agents because:

- Support interactions naturally follow a conversation flow while requiring access to external information and actions;
- Tools can be integrated to pull customer data, order history, and knowledge base articles;
- Actions such as issuing refunds or updating tickets can be handled programmatically; and
- Success can be clearly measured through user-defined resolutions.

Several companies have demonstrated the viability of this approach through usage-based pricing models that charge only for successful resolutions, showing confidence in their agents' effectiveness.

### B. Coding agents

The software development space has shown remarkable potential for LLM features, with capabilities evolving from code completion to autonomous problem-solving. Agents are particularly effective because:

- Code solutions are verifiable through automated tests;
- Agents can iterate on solutions using test results as feedback;
- The problem space is well-defined and structured; and
- Output quality can be measured objectively.

In our own implementation, agents can now solve real GitHub issues in the [SWE-bench Verified](https://www.anthropic.com/research/swe-bench-sonnet) benchmark based on the pull request description alone. However, whereas automated testing helps verify functionality, human review remains crucial for ensuring solutions align with broader system requirements.

## Appendix 2: Prompt engineering your tools

No matter which agentic system you're building, tools will likely be an important part of your agent. [Tools](https://www.anthropic.com/news/tool-use-ga) enable Claude to interact with external services and APIs by specifying their exact structure and definition in our API. When Claude responds, it will include a [tool use block](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#example-api-response-with-a-tool-use-content-block) in the API response if it plans to invoke a tool. Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts. In this brief appendix, we describe how to prompt engineer your tools.

There are often several ways to specify the same action. For instance, you can specify a file edit by writing a diff, or by rewriting the entire file. For structured output, you can return code inside markdown or inside JSON. In software engineering, differences like these are cosmetic and can be converted losslessly from one to the other. However, some formats are much more difficult for an LLM to write than others. Writing a diff requires knowing how many lines are changing in the chunk header before the new code is written. Writing code inside JSON (compared to markdown) requires extra escaping of newlines and quotes.

Our suggestions for deciding on tool formats are the following:

- Give the model enough tokens to "think" before it writes itself into a corner.
- Keep the format close to what the model has seen naturally occurring in text on the internet.
- Make sure there's no formatting "overhead" such as having to keep an accurate count of thousands of lines of code, or string-escaping any code it writes.

One rule of thumb is to think about how much effort goes into human-computer interfaces (HCI), and plan to invest just as much effort in creating good _agent_-computer interfaces (ACI). Here are some thoughts on how to do so:

- Put yourself in the model's shoes. Is it obvious how to use this tool, based on the description and parameters, or would you need to think carefully about it? If so, then it’s probably also true for the model. A good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools.
- How can you change parameter names or descriptions to make things more obvious? Think of this as writing a great docstring for a junior developer on your team. This is especially important when using many similar tools.
- Test how the model uses your tools: Run many example inputs in our [workbench](https://console.anthropic.com/workbench) to see what mistakes the model makes, and iterate.
- [Poka-yoke](https://en.wikipedia.org/wiki/Poka-yoke) your tools. Change the arguments so that it is harder to make mistakes.

While building our agent for [SWE-bench](https://www.anthropic.com/research/swe-bench-sonnet), we actually spent more time optimizing our tools than the overall prompt. For example, we found that the model would make mistakes with tools using relative filepaths after the agent had moved out of the root directory. To fix this, we changed the tool to always require absolute filepaths—and we found that the model used this method flawlessly.

</details>
