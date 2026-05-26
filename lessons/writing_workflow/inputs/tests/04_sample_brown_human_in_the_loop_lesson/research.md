# Research

## Research Results

<details>
<summary>What are the best practices for designing user interfaces for human-in-the-loop AI systems to maximize efficiency and minimize user cognitive load?</summary>

### Source [1]: https://www.lewis-lin.com/blog/designing-effective-human-in-the-loop-systems-with-llms-a-practical-guide

Query: What are the best practices for designing user interfaces for human-in-the-loop AI systems to maximize efficiency and minimize user cognitive load?

Answer: This guide outlines structural and UI best practices for human-in-the-loop (HITL) systems built around large language models to maximize efficiency and reduce cognitive load.[1]

It describes an effective HITL system as having: an **LLM integration** that handles bulk tasks, **task routing and prioritization** to direct only important/ambiguous items to humans, an intuitive **human interface** for reviewing/modifying AI outputs, explicit **feedback mechanisms**, and **learning/improvement cycles** so feedback refines the model over time.[1]

For workflow design, it recommends identifying **critical decision points** where human intervention adds most value, creating **clear escalation criteria** for when AI output must be reviewed, balancing automation vs. human review so routine tasks are automated and attention is reserved for complex edge cases, and ensuring **smooth handoffs** between AI and humans to avoid confusion and delay.[1]

Best practices for HITL system design include **clearly defining roles and responsibilities** so humans know when and how to intervene, **training human operators** to interpret AI outputs and make consistent decisions, and **maintaining consistency** via documented guidelines so different reviewers behave predictably.[1]

It stresses **continuous monitoring and optimization**: track performance, error types, and review load, then adjust thresholds and UI flows to reduce unnecessary escalations and mental effort.[1]

Overall, the article implies that an efficient, low–cognitive load interface: surfaces only the right subset of cases; presents AI suggestions and key context clearly; guides users with escalation rules and decision guidelines; and turns simple corrections into structured feedback so humans do minimal extra work beyond their core judgement.[1]

-----

-----

### Source [2]: https://hai.stanford.edu/news/humans-loop-design-interactive-ai-systems

Query: What are the best practices for designing user interfaces for human-in-the-loop AI systems to maximize efficiency and minimize user cognitive load?

Answer: This Stanford HAI article frames human‑in‑the‑loop as primarily a **human–computer interaction design** problem, emphasizing interaction patterns that reduce cognitive load while preserving human agency.[2]

It argues that automation should be seen as **selective inclusion of human participation**, asking where human judgement and preferences most improve the system and user experience.[2] An effective HITL UI therefore exposes **interaction points** where users can easily steer or correct the AI instead of confronting a monolithic “Big Red Button” control.[2]

The piece highlights **granularity of control** as a key principle: break large tasks into smaller steps so humans can intervene at specific stages, which makes interfaces easier to understand and reduces the mental burden of supervising an opaque full pipeline.[2]

It stresses **valuing human agency**: the UI should make it clear how the human can shape outcomes, not just approve or reject them.[2] In a music‑separation example, users give lightweight annotations (like drawing over spectrogram regions) and can iteratively listen and refine; these small, intuitive actions guide complex algorithms while keeping cognitive load low.[2]

The article notes that human‑in‑the‑loop designs improve **transparency** because each human interaction step forces the system to be understandable enough for the next action.[2] They also ease pressure to build “perfect” algorithms; the system only needs to make **meaningful progress to the next interaction point**, which UIs can present as intermediate results the user can inspect and refine.[2]

Key design questions it poses for interfaces include: where to incorporate **human curation**, at what junctures judgement improves effectiveness, what **interaction model or UI** supports that, and how AI models must be structured to support these interactions.[2] It concludes that there is no single pattern but encourages designing interfaces around human curation, granular control, and iterative feedback loops.[2]

-----

-----

### Source [3]: https://www.permit.io/blog/human-in-the-loop-for-ai-agents-best-practices-frameworks-use-cases-and-demo

Query: What are the best practices for designing user interfaces for human-in-the-loop AI systems to maximize efficiency and minimize user cognitive load?

Answer: This article on HITL for AI agents provides concrete UI and UX patterns that both improve safety and reduce user cognitive effort in supervisory roles.[3]

It emphasizes designing for **decision points, not just prompts**: explicitly identify where human input is critical (access approvals, configuration changes, destructive actions) and make these points explicit checkpoints in the UI, often enforced with mechanisms like `interrupt()` for pausing agents.[3]

For approval interfaces, it recommends **keeping prompts contextual and lightweight**: when asking for human approval, show a clear, focused summary, explain why input is needed, and avoid dumping raw JSON or excessive detail.[3] The goal is to let reviewers make high‑quality decisions quickly with minimal mental parsing.

The article describes patterns such as **approval gates**, **human‑as‑a‑tool**, **fallback escalation**, and **audit‑first** design.[3] Human‑as‑a‑tool UIs allow agents to route questions to humans only when uncertain, keeping **human load low but available** while preserving efficiency.[3] Approval‑gate UIs ensure sensitive actions are always surfaced in a clear, actionable view.

It also advises teams to **think asynchronously when needed**: not every approval must be real time, so interfaces should support routing review tasks to asynchronous channels (e.g., dashboards, email, chat integrations) for low‑priority flows.[3] This reduces interruption cost and lets humans batch decisions, improving efficiency.

The piece frames HITL as a **foundational pattern for trustworthy agents**: UI checkpoints, summaries, and audit trails help keep LLMs within safe boundaries and maintain human control as autonomy increases.[3] Effective interfaces thus: surface only key decisions; summarize context clearly; support real‑time or async review; and integrate HITL patterns without forcing users to understand the full internal complexity of the agentic system.[3]

-----

-----

### Source [4]: https://parseur.com/blog/hitl-best-practices

Query: What are the best practices for designing user interfaces for human-in-the-loop AI systems to maximize efficiency and minimize user cognitive load?

Answer: This article focuses on human‑in‑the‑loop AI in document workflows and highlights UI practices that increase efficiency and reduce reviewer cognitive load.[5]

It states that **successful HITL implementation requires clear review points, intuitive UI design, defined exception rules, and measurable KPIs**.[5] Clear review points ensure users always know when they must intervene, avoiding confusion or constant vigilance.

For interfaces, it recommends **leveraging efficient tools and intuitive, low‑code collaboration platforms**.[5] Helpful features include:
- **Dashboards for real‑time review**, giving users a centralized, organized view of items needing attention instead of scattered alerts.[5]
- **Automated alerts for low‑confidence fields**, so humans only review uncertain cases; this directly reduces workload and mental fatigue compared to checking everything.[5]
- **In‑app correction and annotation tools**, letting users fix extraction errors quickly and provide structured feedback without leaving the workflow.[5]

The article emphasizes **seamless integration** of human intervention so it occurs naturally and predictably, which reduces friction and enhances trust.[5] Interfaces should make it obvious when the AI is confident vs. uncertain and provide one‑click ways to correct data.

It also mentions **common pitfalls** such as unclear processes, overreliance on manual checks, and lack of performance metrics.[5] To avoid these, UI design should support clear exception handling rules and surface KPIs (accuracy, review time, exception rates), allowing teams to optimize thresholds and flows over time.[5]

Overall, the source suggests that for document‑centric HITL systems, high‑efficiency, low‑cognitive‑load UIs center on: targeted review (confidence‑based), consolidated dashboards, inline corrections, and predictable integration of human steps into the automation pipeline.[5]

-----

-----

### Source [5]: https://www.thesys.dev/blogs/designing-human-in-the-loop-ai-interfaces-that-empower-users

Query: What are the best practices for designing user interfaces for human-in-the-loop AI systems to maximize efficiency and minimize user cognitive load?

Answer: This article explicitly addresses how to design **human‑in‑the‑loop AI interfaces that empower users**, outlining several UI principles for efficiency and cognitive ease.[6]

It states that designing for HITL involves key principles: **make the AI’s role transparent**, **allow user intervention**, and **provide clear feedback loops**.[6] Transparency means interfaces should clearly indicate what the AI did, what it is suggesting, and its level of confidence, so users do not need to guess or mentally reconstruct system behavior.

Allowing user intervention requires UIs that support easy overrides, corrections, and adjustments at appropriate points, instead of all‑or‑nothing control.[6] This can include editable fields, toggles to accept/reject AI suggestions, and granular controls over how strongly AI outputs influence final decisions.

The article emphasizes **feedback loops**: interfaces should show how user corrections and inputs affect the AI over time, for example by acknowledging received feedback or indicating improved behavior on similar future cases.[6] This supports user trust and reduces cognitive dissonance by making the learning process visible.

It also stresses that **interfaces should reduce complexity, not add to it**.[6] AI‑augmented tools must simplify workflows and help users focus on high‑value decisions, not force them to manage additional configuration or interpret opaque outputs.

Designers are encouraged to think about how users will understand, monitor, and guide the AI: what information needs to be surfaced, what controls are necessary, and how to avoid overwhelming users with options.[6] By clearly framing AI assistance, enabling lightweight intervention, and visually reinforcing the feedback loop, HITL UIs can both empower users and keep cognitive load manageable.[6]

-----

-----

### Source [6]: https://www.reforge.com/guides/design-human-centered-ai-interfaces

Query: What are the best practices for designing user interfaces for human-in-the-loop AI systems to maximize efficiency and minimize user cognitive load?

Answer: This Reforge guide discusses principles for **human‑centered AI interfaces**, many of which directly apply to human‑in‑the‑loop AI systems focused on efficiency and minimizing cognitive load.[7]

It underscores that AI interfaces should be designed around **user goals and mental models**, not around model capabilities.[7] This includes presenting AI outputs in familiar formats, aligning with existing workflows, and avoiding forcing users to think in terms of prompts or internal model concepts.

The guide emphasizes **robust error‑handling mechanisms** and **clear feedback on how to resolve issues**.[7] When AI fails or is uncertain, the UI should state what happened, why (when possible), and what the user can do next; this direct guidance reduces the mental effort of diagnosing problems.

It recommends providing **explanations and transparency** at appropriate levels of detail, so users can understand why AI made a suggestion and decide whether to trust it without excessive investigation.[7] Interfaces might surface rationales, highlight key factors, or show confidence indicators.

The article also discusses **progressive disclosure**: reveal complexity only when needed, starting with simple summaries and allowing users to drill down into more detailed information on demand.[7] This approach is particularly important for HITL systems where too much detail at once can overwhelm supervisors.

Additionally, it highlights the value of **clear affordances and guidance**—labels, examples, and constraints that show how to interact with AI features—and using **sensible defaults** so users can operate effectively without extensive configuration.[7]

Overall, the guide advocates for interfaces that: align with user tasks; handle errors gracefully; communicate uncertainty and rationale; and manage information density through progressive disclosure, all of which contribute to efficient, low‑cognitive‑load human‑in‑the‑loop AI experiences.[7]

-----

</details>

<details>
<summary>How does the 'evaluator-optimizer' pattern in AI agentic workflows specifically benefit from incorporating direct human feedback?</summary>

### Source [7]: https://www.anthropic.com/research/building-effective-agents

Query: How does the 'evaluator-optimizer' pattern in AI agentic workflows specifically benefit from incorporating direct human feedback?

Answer: Anthropic describes the **evaluator‑optimizer** workflow as a loop where one LLM call generates a response and another provides evaluation and feedback, iterating until quality is satisfactory.[3] This pattern is recommended specifically when there are **clear evaluation criteria** and when **iterative refinement yields measurable value**.[3]

According to Anthropic, a strong signal that evaluator‑optimizer is appropriate is that **LLM outputs can be substantially improved when a human articulates feedback**, and that an LLM can in turn play that evaluator role.[3] Direct human feedback thus becomes the *prototype* of the evaluator: if humans can reliably say what is wrong or missing, those same criteria can be encoded or mimicked in the evaluator step. This makes the workflow particularly well‑suited for **human‑sensitive tasks** like literary translation or complex research, where subtle human judgments (nuance, completeness, trade‑offs) matter and can be expressed as feedback.[3]

Anthropic’s framing implies several concrete benefits of incorporating human feedback into this pattern:

- Human feedback helps define **explicit evaluation rubrics**, which the evaluator can then apply consistently in the loop.[3]
- The evaluator‑optimizer loop can approximate a human **iterative editing process**, where a human would successively critique and refine drafts; formalizing that process as criteria lets an LLM evaluator stand in for the human when appropriate.[3]
- For complex search and analysis tasks, humans can specify **when further searching or refinement is warranted**, and those conditions can be turned into evaluator rules that decide whether another iteration is needed.[3]

Thus, Anthropic positions human feedback as both the **design input** (to shape evaluation criteria and stopping conditions) and a **gold standard** against which the performance of the evaluator‑optimizer loop is judged.[3]

-----

-----

### Source [8]: https://docs.praison.ai/docs/features/evaluator-optimiser

Query: How does the 'evaluator-optimizer' pattern in AI agentic workflows specifically benefit from incorporating direct human feedback?

Answer: PraisonAI defines the **Evaluator‑Optimizer** pattern as a feedback‑loop workflow where LLM‑generated outputs are **evaluated, refined, and optimized iteratively** to improve accuracy and relevance.[1] The pattern explicitly supports **feedback‑driven optimization** and **continuous improvement loops**.[1]

In PraisonAI’s reference implementation, there is a **Generator** agent whose goal is to "generate initial solutions and incorporate feedback" and an **Evaluator** agent whose goal is to "evaluate and provide feedback."[1] Tasks are wired so that the evaluator’s output (e.g., "more" vs "done") directly controls whether another generation step occurs, forming a closed loop.[1]

PraisonAI’s design shows several ways direct human feedback can be integrated and why it is beneficial in this pattern:

- The **instructions** for both generator and evaluator include detailed criteria and formats; these can be derived from human feedback about what constitutes a good solution or a good evaluation.[1]
- The evaluator task’s **expected output** (such as "more or done" or specific quality checks) can encode human judgment about sufficiency and quality, making the loop reflect human standards.[1]
- Because the system exposes **configuration options** and process control for the optimization loop, humans can adjust thresholds, criteria, or stopping rules based on observed behavior, effectively injecting feedback at the workflow level, not just at the content level.[1]

By structuring the workflow around evaluation and refinement, PraisonAI’s pattern is naturally aligned with **direct human supervision**: humans can either act as evaluators themselves or translate their feedback into evaluator prompts and conditions, which then guide the optimizer to systematically move outputs closer to human‑defined expectations.[1]

-----

-----

### Source [9]: https://sageflow.ai/glossary/evaluator-optimizer

Query: How does the 'evaluator-optimizer' pattern in AI agentic workflows specifically benefit from incorporating direct human feedback?

Answer: SageFlow defines the **Evaluator‑Optimizer** pattern as a workflow where **two distinct LLMs** collaborate in a continuous feedback loop: a **Generator** creates initial responses and an **Evaluator** "meticulously" assesses them and provides **constructive criticism and guidance for refinement**.[2] This loop continues until the Evaluator deems the output satisfactory, representing convergence toward **optimal quality**.[2]

SageFlow emphasizes that this pattern "excels" when **well‑defined evaluation criteria exist** and where **incremental improvement per iteration yields tangible benefits**.[2] Those evaluation criteria are typically derived from **human domain expertise and feedback**—for example, human reviewers specifying what counts as comprehensive research, safe code, or on‑brand copy.[2]

The article likens the workflow to "a meticulous writer seeking feedback from a discerning editor".[2] In that analogy, the editor corresponds to a human‑informed Evaluator: the more precise and high‑quality the human editorial standards, the more the evaluator can enforce them through the loop. Concrete examples given include:

- **Code review**, where the evaluator highlights bugs, vulnerabilities, or optimization opportunities—mirroring how human reviewers comment on code quality and safety.[2]
- **Complex research tasks**, where the evaluator checks for comprehensiveness and can suggest additional searches if needed, akin to human researchers pointing out missing angles or sources.[2]

SageFlow also notes that one should "start simple, rigorously evaluate performance, and only introduce more complex agentic systems when simpler approaches fall short".[2] Here, human feedback about where single‑shot LLMs fail is what motivates and shapes an evaluator‑optimizer setup: humans identify failure modes and quality gaps, and those insights are encoded into evaluator behavior and iteration logic, directly aligning the loop with human expectations.[2]

-----

-----

### Source [10]: https://javaaidev.com/docs/agentic-patterns/patterns/evaluator-optimizer

Query: How does the 'evaluator-optimizer' pattern in AI agentic workflows specifically benefit from incorporating direct human feedback?

Answer: Java AI Dev describes the **Evaluator‑Optimizer** pattern as allowing an LLM to improve generation quality by **optimizing a previous generation with feedback from an evaluator**, potentially across multiple iterations.[4] The workflow is decomposed into subtasks: initialize input, generate initial result, **evaluate and provide feedback**, optimize based on that feedback, and finalize the response.[4]

The evaluation subtask explicitly produces **feedback** that is then consumed by the optimization subtask, whose goal is "Improve existing code based on feedback" and whose prompt requires the system to **address all concerns in the feedback**.[4] This structure directly mirrors how human review comments drive revision.

Human feedback benefits this pattern in several ways implied by the design:

- The evaluation criteria (e.g., pass/fail or numeric scores 0–100) can be calibrated from human assessments of what is acceptable, enabling evaluators to approximate **human quality judgments**.[4]
- Numeric scoring makes it possible to set **thresholds** that reflect human tolerance for imperfections and to average multiple evaluations, similar to aggregating several human reviewers’ opinions.[4]
- The example of multi‑round **code review** (generation → evaluation not passed → optimization → evaluation passed) reflects the typical human process of iterative review and fix, showing how human‑style feedback can be operationalized.[4]

The guidance to **limit the maximum number of evaluations** balances quality with latency and cost.[4] In practice, humans decide appropriate trade‑offs between more iterations and responsiveness; those decisions can be encoded as configuration, so the pattern not only incorporates human feedback on content but also on the workflow’s **performance and cost trade‑offs**.[4]

By structuring evaluation outputs as actionable feedback and clear scores or decisions, this pattern is well‑suited to absorbing **direct human review comments** (or criteria derived from them) and using them to systematically steer the optimizer toward human‑aligned results.[4]

-----

</details>

<details>
<summary>What are the architectural patterns for decoupling AI workflows to support flexible human-in-the-loop (HITL) interventions, particularly in complex writing or content generation systems?</summary>

### Source [11]: https://aws.amazon.com/blogs/compute/part-2-serverless-generative-ai-architectural-patterns/

Query: What are the architectural patterns for decoupling AI workflows to support flexible human-in-the-loop (HITL) interventions, particularly in complex writing or content generation systems?

Answer: AWS describes several **serverless generative AI patterns** that naturally decouple workflows and enable human-in-the-loop (HITL) in content-generation systems.[7]

It recommends separating concerns into distinct managed components: 
- **Front-end / interaction layer** (e.g., web or mobile client using API Gateway + Amazon CloudFront) that handles user prompts, streaming outputs, and human feedback collection.[7]
- **Orchestration layer** using AWS Step Functions or Amazon Bedrock Flows to model AI workflows as state machines or graphs, with explicit states for LLM calls, tool invocations, and human approval steps.[7]
- **Model invocation layer** via Amazon Bedrock or Amazon SageMaker for calling different foundation models behind a stable interface, so model changes do not affect business or orchestration logic.[7]
- **Context and data layer** using services like Amazon RDS, DynamoDB, or S3 for documents, user profiles, and conversation history, decoupled from the model and orchestration components.[7]

For HITL specifically, AWS patterns insert **human review states** in the workflow, where:
- Generated content is stored (e.g., S3) and a task token or event is sent to a human review UI (e.g., via Amazon SNS/EventBridge + custom app) for approval, edit, or rejection.[7]
- The workflow waits in a paused state until the human action is completed, then resumes the state machine, passing the edited content forward.[7]

AWS also advocates **event-driven decoupling** for complex workflows: each major step (prompt enrichment, safety checks, generation, post-processing, review) emits and consumes events via Amazon EventBridge, allowing independent evolution and selective insertion of HITL stages (e.g., human moderation only for high-risk categories).[7]

This architecture supports experimentation with **different models or guardrails** without changing the human-review logic, and vice versa, because HITL review is modeled as its own service or state in the workflow rather than embedded in model-calling code.[7]

-----

-----

### Source [12]: https://dev.to/leena_malhotra/the-architecture-of-ai-workflows-designing-beyond-the-model-layer-45ld

Query: What are the architectural patterns for decoupling AI workflows to support flexible human-in-the-loop (HITL) interventions, particularly in complex writing or content generation systems?

Answer: This article argues that robust AI systems require **workflow-level architecture** that decouples the model from surrounding components, enabling flexible interventions like human review.[6]

It highlights problems when application logic is tightly coupled to a specific LLM API (e.g., prompt formats and response structures are assumed everywhere), making it difficult to switch models, inject new stages, or add moderation/HITL checks.[6]

The author proposes designing **AI workflows as explicit pipelines** composed of distinct stages:
- **Input gathering and validation** (collecting user intent, parameters, constraints).[6]
- **Context assembly** (retrieving documents, past messages, or external data for RAG-style prompts).[6]
- **Model interaction** (one or more calls to LLMs/tools behind a stable abstraction).[6]
- **Post-processing** (parsing, formatting, classification, or additional transformations).[6]
- **Output delivery and feedback capture** (returning results and collecting human ratings or edits).[6]

By separating these stages, HITL can be introduced at clear boundaries—for example:
- Manual approval between model output and final delivery for sensitive content.
- Editorial review after post-processing but before publishing.
- Human correction loops whose results feed back into prompt templates or fine-tuning data.[6]

The article emphasizes **model-agnostic design**: use an abstraction layer or gateway that normalizes request/response formats so that workflow definitions and HITL logic do not depend on a single provider.[6]

It also notes that for complex writing systems, multiple model calls (drafting, fact-checking, style normalization) should be orchestrated explicitly, with the possibility of human intervention after each major phase, instead of hiding everything in one monolithic “generate” function.[6]

-----

-----

### Source [13]: https://codewithcaptain.com/how-to-decouple-ai-architecture/

Query: What are the architectural patterns for decoupling AI workflows to support flexible human-in-the-loop (HITL) interventions, particularly in complex writing or content generation systems?

Answer: This source lays out a **layered architecture** for decoupled AI systems that maps cleanly to human-in-the-loop writing workflows.[1]

It distinguishes four main layers:[1]
- **Model layer**: capability providers (e.g., Claude, GPT-4, Gemini) hidden behind a **unified interface** and common output schema (often JSON Schema), so business and workflow logic stay unchanged when swapping models.[1]
- **Agent layer**: manages tools, memory, tone, and policies—*how* to act—separate from concrete tasks or prompts.[1]
- **Task layer**: discrete tasks like summarize, enrich, classify, generate, each with **fixed input/output contracts** and evaluation rules that remain stable as models or prompts evolve.[1]
- **Orchestration layer**: controls flows, routing, fallbacks, and A/B testing, often implemented via tools like Amazon Bedrock Flows.[1]

For HITL, this layering enables interventions at different points without breaking others:
- Editors can review or modify outputs at the **task boundary**, because each task has a clear schema and success criteria.[1]
- Policy or compliance reviewers can adjust **agent behavior** (e.g., stricter tool usage, guardrails) independently of specific prompts or models.[1]
- Product teams can experiment with **orchestration strategies** (e.g., when to escalate to humans, when to auto-publish) by changing flows rather than touching model integrations.[1]

The article stresses **decoupling contracts from implementations**: tasks and schemas define what the system expects; adapters handle how each model or tool satisfies that contract.[1]

It recommends patterns such as:
- Output normalization using JSON Schema for all models.[1]
- Capability-based registries (e.g., “long-form writer,” “style rewriter”) to swap implementations transparently.[1]
- Feature flags/configuration to change models or flows at runtime, enabling rapid experimentation with where and how to place human review in content pipelines.[1]

-----

-----

### Source [14]: https://paelladoc.com/blog/stop-guessing-5-ai-architecture-patterns/

Query: What are the architectural patterns for decoupling AI workflows to support flexible human-in-the-loop (HITL) interventions, particularly in complex writing or content generation systems?

Answer: This article lists several AI architecture patterns and introduces the **“decoupled context pipeline”** as a way to manage complex context separately from model calls, which is important for human-in-the-loop content systems.[2]

The **decoupled context pipeline** addresses scenarios where context comes from multiple sources (user input, databases, external APIs) and must be processed, enriched, and provided consistently to different models or agents.[2]

Key ideas:[2]
- Context ingestion, enrichment, and storage should be implemented as an independent pipeline, not tangled with LLM prompts or application UI.
- Downstream AI components (writers, summarizers, editors) consume a **standardized context representation**, allowing HITL steps that correct or curate context without changing model code.
- Since context preparation can be expensive and domain-specific, decoupling lets teams build dedicated context services with their own monitoring, caching, and quality checks.[2]

For HITL, this pattern enables:
- Human review of **sources and context** (e.g., which documents feed a long-form generation) before the generation step, because context is a first-class artifact in the system.[2]
- Editorial tools that let users adjust or annotate context objects, rather than rewriting prompts, improving controllability in complex writing workflows.[2]

Within the same article, other patterns (like using orchestrated workflows and agents) are positioned as building blocks: workflows define **when** models or humans engage; agents define **how** AI components reason; the decoupled context pipeline ensures they all see the same curated information.[2]

The overall recommendation is to avoid “context-aware monoliths” and instead use separate context services that can evolve—and include human curation—independently of model orchestration and UI layers.[2]

-----

-----

### Source [15]: https://www.prompts.ai/en/blog/decoupled-ai-pipelines-dependency-management-best-practices

Query: What are the architectural patterns for decoupling AI workflows to support flexible human-in-the-loop (HITL) interventions, particularly in complex writing or content generation systems?

Answer: This source discusses **decoupled AI pipelines** and dependency management patterns that support modular, adaptable workflows where HITL can be inserted without breaking the system.[3]

It describes breaking AI workflows into independent modules such as **data preprocessing, model training, and inference**, each communicating via well-defined interfaces.[3]

Core architectural principles:[3]
- **Loose coupling**: components should not rely on each other’s internal details; they interact through stable contracts so modules (including human-review services) can be swapped or upgraded independently.
- **Dependency Inversion Principle**: high-level modules (e.g., workflow orchestrators, editorial tools) should depend on **abstractions**, not specific libraries or models. Both high-level and low-level modules depend on shared interfaces.[3]
- **Dependency injection**: components receive implementations (e.g., a specific LLM, a human-review adapter, a safety filter) from the outside, enabling easy replacement and testing.[3]
- **Factory methods and centralized registries**: object creation and dependency lookup are centralized so different implementations (e.g., “AI reviewer” vs “human reviewer”) can be selected at runtime.[3]

Applied to complex writing systems, these patterns allow:
- Defining abstract interfaces like `IContentGenerator`, `IContentReviewer`, `ISafetyChecker`, with concrete implementations that may be AI-only, human-only, or hybrid.[3]
- Switching from automated review to HITL (or vice versa) by configuration, without changing core workflow code.
- Managing versioned dependencies (models, tools, review policies) in a **central registry**, keeping complex content pipelines consistent across teams and environments.[3]

The article emphasizes that such decoupling reduces the risk that adding a new human-approval step or changing a reviewer implementation will cascade into failures elsewhere in the AI pipeline.[3]

-----

-----

### Source [16]: https://www.getknit.dev/blog/orchestrating-complex-ai-workflows-advanced-integration-patterns

Query: What are the architectural patterns for decoupling AI workflows to support flexible human-in-the-loop (HITL) interventions, particularly in complex writing or content generation systems?

Answer: This source focuses on **orchestration patterns for complex AI workflows**, introducing designs that support multiple tools, parallelism, and explicit task graphs—all of which facilitate flexible HITL points.[4]

One highlighted pattern is a **planner–executor DAG architecture**:[4]
- A **Planner** analyzes the user request and produces a **Directed Acyclic Graph (DAG)** of tasks and dependencies.
- A **Task Fetching Unit** executes tasks as soon as their dependencies are satisfied, often in parallel and using multiple tools or models.
- A **Joiner** aggregates task results into a final response once all required nodes have completed.[4]

This graph-based orchestration decouples *what needs to be done* (the plan) from *how each step is implemented* (specific tools, models, or humans).[4]

For HITL in complex writing systems, the same pattern can:
- Represent **human tasks as nodes** in the DAG (e.g., “legal review,” “editorial polishing”), scheduled alongside AI tasks.
- Allow the planner to decide dynamically when to route a subtask (like fact-checking or tone adjustment) to a human vs an AI tool, based on complexity or risk.[4]
- Enable partial progress and parallelism: AI tools can draft or summarize while human editors work on other sections, with the Joiner merging everything coherently.[4]

The article also covers **advanced integration beyond basic RAG**, including multi-tool sequences and multiple app instances, and positions orchestration as a separate concern from model selection.[4]

By modeling workflows as DAGs with interchangeable node implementations, teams can evolve HITL policies—where humans intervene and for which tasks—without rewriting the planner or core application interface.[4]

-----

-----

### Source [17]: https://aipmguru.substack.com/p/ai-architecture-patterns-101-workflows

Query: What are the architectural patterns for decoupling AI workflows to support flexible human-in-the-loop (HITL) interventions, particularly in complex writing or content generation systems?

Answer: This piece categorizes modern AI architecture into four main patterns—**Workflows, Agents, MCPs (Model Context Protocol), and A2A (Agent-to-Agent)**—and explains when to use each, which informs how to design decoupled HITL interventions.[5]

Key pattern roles:[5]
- **Workflows**: best when you need *reliability and predictability*. AI steps are defined in a structured flow (often as state machines or pipelines) that can easily include **human approval or escalation steps** at known points.
- **Agents**: used when *flexibility and autonomous reasoning* are required. Agents can decide whether to call tools, request human input, or perform additional reasoning in response to ambiguous content-generation tasks.[5]
- **MCP**: used when multiple AI components need to work together seamlessly, by standardizing how tools and models exchange context.[5]
- **A2A**: focuses on complex tasks that benefit from multiple specialized agents collaborating, potentially including agents that represent human roles (editor, fact-checker).[5]

For complex writing systems, the article’s guidance implies:
- Start with **workflow-based architecture** to define a stable backbone where HITL steps such as editorial review, compliance checks, or user approvals are explicit stages.[5]
- Introduce **agents on top of workflows** when you need more adaptive routing—for instance, an agent may decide to trigger a human-in-the-loop step only if a confidence or risk threshold is crossed.[5]

By treating workflows, agents, and multi-agent coordination as separate patterns rather than mixing them ad hoc, teams can keep human interaction logic (who reviews what, when) decoupled from low-level model calls and prompt engineering.[5]

-----

</details>

<details>
<summary>What are the trade-offs between fully autonomous AI agents and human-in-the-loop systems, especially regarding cost, latency, and reliability for practical applications?</summary>

### Source [18]: https://www.auxiliobits.com/blog/how-to-choose-between-autonomous-and-human-in-the-loop-agents/

Query: What are the trade-offs between fully autonomous AI agents and human-in-the-loop systems, especially regarding cost, latency, and reliability for practical applications?

Answer: Autonomous agents excel in high-volume, repetitive, and rule-based environments, delivering unmatched speed, cost-efficiency, and scalability with minimal human involvement. They operate entirely independently, making decisions and executing actions without direct human intervention once deployed, which reduces labor costs and enables 24/7 operation. However, they are best suited for tasks with clear rules and low ambiguity, where errors are tolerable or easily corrected.

Human-in-the-loop (HITL) agents are essential where decisions are complex, ethical, or emotionally nuanced, combining AI efficiency with human oversight for greater accuracy and trust. HITL systems handle the bulk of data processing and initial analysis but escalate critical, ambiguous, or sensitive decisions to human experts. This improves reliability and reduces the risk of harmful errors, especially in high-stakes domains like legal, medical, or financial contexts. However, HITL introduces higher operational costs due to human staffing and can increase latency because of the time required for human review and approval. The model also supports continuous learning through human feedback, enhancing long-term reliability and adaptability.

The choice between autonomous and HITL agents depends on task complexity, error tolerance, regulatory needs, and customer experience goals. Hybrid models are increasingly favored, automating routine work with autonomous agents while reserving human intervention for exceptions, edge cases, or sensitive interactions, balancing cost, latency, and reliability.

-----

-----

### Source [19]: https://www.redhat.com/en/blog/classifying-human-ai-agent-interaction

Query: What are the trade-offs between fully autonomous AI agents and human-in-the-loop systems, especially regarding cost, latency, and reliability for practical applications?

Answer: In the Human-Out-of-the-Loop (HOOTL) model, AI operates fully autonomously with no human involvement at any stage, handling the entire loop independently in low-risk scenarios. This approach maximizes speed and minimizes human labor costs, making it suitable for predictable, routine tasks like basic automated sorting or data processing in controlled environments. However, it carries the risk of unaddressed errors due to the lack of safeguards, which can compromise reliability, especially in dynamic or high-stakes settings.

In contrast, Human-in-the-Loop (HITL) and AI-in-the-Loop (AITL) models involve active human participation. HITL embeds human judgment in decision-making or validation, improving accuracy and accountability but introducing latency due to human review cycles and higher operational complexity from staffing and shift coverage. AITL flips the dynamic: humans lead workflows while AI provides assistive inputs (e.g., suggestions in diagnostics or creative tools), enhancing human productivity without full automation. This human-centric augmentation improves reliability and interpretability but may not achieve the same throughput or cost savings as fully autonomous systems. The trade-offs center on balancing full automation’s efficiency against the reliability and ethical safeguards provided by human involvement.

-----

-----

### Source [20]: https://skywork.ai/blog/agent-vs-human-in-the-loop-2025-comparison/

Query: What are the trade-offs between fully autonomous AI agents and human-in-the-loop systems, especially regarding cost, latency, and reliability for practical applications?

Answer: Autonomous AI agents can plan and execute multi-step work across tools and APIs, favoring real-time, high-volume, and 24/7 workloads with low latency and no queueing delays. They are strong when ground truth is available, automatic checks can validate steps, and rollback is possible, enabling high throughput and cost efficiency per resolution. However, their reliability depends on clear, unambiguous correctness; in domains with high nuance or where harm from error is high, autonomous agents may produce hallucinations or errors that are hard to detect without human oversight.

Human-in-the-Loop (HITL) systems embed human judgment at training, validation, and/or runtime, which is needed when correctness is ambiguous, domain nuance matters, or error consequences are severe. HITL improves reliability, accountability, and auditability by adding a second pair of eyes and documented rationale, but it increases latency due to human wait times and operational complexity from staffing, shift coverage, and queue management. HITL also supports better quality control in high-risk or sensitive contexts.

The trade-offs are: agents offer lower cost and latency with good reliability in well-defined tasks, while HITL increases cost and latency but significantly improves reliability and safety in complex, high-risk, or ambiguous scenarios. Progressive autonomy—starting with HITL and expanding autonomy only when KPIs and audits are consistently positive—helps balance these trade-offs.

-----

-----

### Source [21]: https://yodaplus.com/blog/human-in-the-loophitl-vs-agentic-autonomy-striking-the-right-balance/

Query: What are the trade-offs between fully autonomous AI agents and human-in-the-loop systems, especially regarding cost, latency, and reliability for practical applications?

Answer: Human-in-the-Loop (HITL) systems involve active human involvement in AI decision-making or validation, ensuring accountability and supervision. HITL is especially useful in domains requiring ethical reasoning, interpretability, or complex judgment, where humans act as gatekeepers. This improves reliability and reduces the risk of harmful errors in critical sectors, but it introduces higher operational costs due to staffing and can create bottlenecks that increase latency, especially in high-volume workflows.

Agentic autonomy, by contrast, uses autonomous agents that can independently perceive, decide, and act based on contextual goals. These agents enable self-correction, broader scope, and higher speed, making them suitable for domains like retail automation, document digitization, and FinTech where constant human supervision is neither scalable nor efficient. Agentic autonomy reduces labor costs and latency, enabling 24/7 operation and faster throughput, but carries higher risk if deployed in critical areas without safeguards, potentially compromising reliability.

The trade-offs are: HITL provides higher reliability and accountability at the cost of higher operational expense and latency, while agentic autonomy offers lower cost and lower latency with higher speed and scalability, but requires careful risk management to maintain reliability. The optimal approach is often a hybrid that combines the strengths of both, rather than choosing one exclusively.

-----

-----

### Source [22]: https://dev.to/camelai/agents-with-human-in-the-loop-everything-you-need-to-know-3fo5

Query: What are the trade-offs between fully autonomous AI agents and human-in-the-loop systems, especially regarding cost, latency, and reliability for practical applications?

Answer: Human-in-the-Loop (HITL) frameworks integrate human expertise at key decision points to improve AI efficiency, accuracy, and accountability. HITL systems balance automation and human judgment by allowing AI to handle routine tasks autonomously while escalating uncertain or critical decisions to human experts. This improves reliability, especially in dynamic environments, by minimizing errors through conformal prediction, iterative feedback loops, and interactive validation. However, the need for human review introduces latency and increases operational complexity and cost due to staffing and coordination requirements.

Autonomous agents, in contrast, operate independently in an "outer loop," actively working toward goals using tools and functions without requiring human initiation. This design maximizes efficiency and autonomy, reducing latency and operational costs associated with constant human involvement. Communication with humans is agent-initiated, occurring only when a critical function requires human approval or feedback, which limits human overhead while preserving oversight where it matters most.

The trade-offs are: fully autonomous agents offer lower cost and lower latency by minimizing human involvement, but may be less reliable in ambiguous or high-stakes situations. HITL systems improve reliability and accountability through human oversight but at the expense of higher cost and increased latency due to human review cycles. The evolution of HITL aims to enhance autonomy, accountability, flexibility, and ethical alignment, moving toward agent-initiated collaboration rather than constant human control.

-----

-----

### Source [23]: https://aws.amazon.com/blogs/aws-insights/the-rise-of-autonomous-agents-what-enterprise-leaders-need-to-know-about-the-next-wave-of-ai/

Query: What are the trade-offs between fully autonomous AI agents and human-in-the-loop systems, especially regarding cost, latency, and reliability for practical applications?

Answer: Truly autonomous agents are distinguished by their capacity to reason iteratively, evaluate outcomes, adapt plans, and pursue goals without constant human intervention. They excel at tireless execution, statistical pattern recognition, and goal-directed autonomy at scale, which translates into high throughput, low latency, and reduced operational costs over time, especially in high-volume, repetitive workflows. This makes them economically attractive for enterprises seeking efficiency and scalability.

However, autonomous agents operate differently from humans: while they are strong in execution and pattern recognition, they lack lived experience, moral reasoning, and intuitive creativity grounded in ambiguity and emotion. In practical applications, this can affect reliability, particularly in complex, ambiguous, or ethically sensitive situations where human judgment is critical. Fully autonomous systems may make statistically sound but contextually inappropriate or harmful decisions if not properly constrained.

The emerging paradigm is a "human-AI partnership" that reimagines the traditional "human-in-the-loop" model. Instead of constant oversight, humans act more like managers who coach, clarify, or redirect autonomous agents when needed. This balances the cost and latency advantages of autonomy with the reliability, ethical alignment, and contextual understanding that humans provide, especially for high-risk decisions. The trade-off is between the efficiency and scalability of full autonomy and the reliability, safety, and ethical safeguards of human involvement.

-----

</details>

<details>
<summary>How is the concept of a 'human-in-the-loop' being implemented in modern AI-powered developer tools like IDE extensions or code assistants?</summary>

### Source [24]: https://ai-sdk.dev/cookbook/next/human-in-the-loop

Query: How is the concept of a 'human-in-the-loop' being implemented in modern AI-powered developer tools like IDE extensions or code assistants?

Answer: This cookbook shows how to implement **human-in-the-loop (HITL)** in a web-based, agentic developer tool using the Vercel AI SDK with Next.js.[3] The pattern is aimed at AI agents that call tools (for example, a `weather` tool) and need user approval before executing potentially impactful operations.[3]

By default, a tool definition includes an `execute` function that runs automatically when the model issues a `ToolCall` with parameters (such as `location`).[3] To introduce HITL, the guide removes the `execute` function from the tool definition, so the frontend can intercept the tool call instead of executing it directly.[3] The agent still returns a structured tool invocation (with name and arguments), but the application inserts a **confirmation step** between the tool call and the tool result.[3]

In practice, the UI surfaces the pending tool call to the human (for example, in an IDE-like panel or modal) and lets the user **approve, modify, or reject** the action before it is carried out.[3] Once approved, the frontend (or server) explicitly invokes the underlying capability (such as an API request) and then feeds the result back into the agent’s context.[3]

The cookbook also sketches an abstraction layer you can build on top of this low-level pattern, so that tool calls requiring confirmation can be consistently routed through a HITL workflow.[3] This demonstrates how modern AI-powered developer tools can embed user checkpoints around tool execution, enabling fine-grained control over what the agent is allowed to do while still benefiting from automated planning and reasoning.[3]

-----

-----

### Source [25]: https://aws.amazon.com/blogs/machine-learning/implement-human-in-the-loop-confirmation-with-amazon-bedrock-agents/

Query: How is the concept of a 'human-in-the-loop' being implemented in modern AI-powered developer tools like IDE extensions or code assistants?

Answer: This Amazon Bedrock Agents blog describes built-in mechanisms for **human-in-the-loop confirmation** when AI agents orchestrate multi-step tasks and invoke APIs, which is directly applicable to code assistants and IDE extensions that call developer tooling.[2]

Bedrock Agents use foundation models to break down user requests into multiple steps, then call company APIs or knowledge bases to fulfill them.[2] In sensitive scenarios (for example, actions that change state, cost money, or affect production systems), the post emphasizes that **HITL interaction is essential**, with humans approving actions, providing feedback, or reviewing responses offline.[2] Human oversight establishes ground truth, validates responses before they go live, and supports continuous learning via feedback loops.[2]

The post introduces two concrete HITL patterns:

- **User confirmation**: A Boolean confirmation gate where the agent pauses before executing a selected tool and asks the end-user to approve or deny the action.[2] Developers can configure which tools are auto-executable and which require explicit user confirmation, adding an extra safety and control layer in automated workflows.[2]

- **Review and oversee control (ROC)**: A more advanced mode in which the agent proposes the task and parameters but delegates actual execution to the developer.[2] The developer can validate or change the agent’s decisions, add context, and modify parameters before the action is run, with ROC configured at the action-group level.[2]

Both mechanisms provide structured **checkpoints** where humans remain in control of critical decisions while still leveraging the agent’s planning and orchestration capabilities.[2] These patterns map directly to IDE or code-assistant settings where tool calls that modify repositories, infrastructure, or CI/CD pipelines are gated by developer review.[2]

-----

-----

### Source [26]: https://www.ibm.com/think/tutorials/human-in-the-loop-ai-agent-langraph-watsonx-ai

Query: How is the concept of a 'human-in-the-loop' being implemented in modern AI-powered developer tools like IDE extensions or code assistants?

Answer: IBM’s tutorial demonstrates a **human-in-the-loop (HITL)** feedback architecture for an agent built with **LangGraph** and **watsonx.ai**, focused on prior-art patent search—a workflow similar to developer research assistants.[6]

The tutorial defines HITL as an architectural pattern where **human feedback is required** to guide LLM decision-making and provide supervision at some stage in the AI workflow, ensuring precision, safety, and accountability.[6] The agent uses an LLM (IBM Granite) plus tools (Google Patents via SerpAPI) to search and summarize patents.[6]

Within this graph-based agent, HITL is implemented as an explicit **feedback mechanism**: the human user reviews the agent’s patent findings and can approve, refine, or correct them before the workflow proceeds.[6] LangGraph’s structure allows these review steps to be integrated as nodes where the graph can pause, gather human input, and then resume with updated context or instructions.[6]

This architecture is analogous to modern developer tools that embed review stages into agentic flows—such as letting developers inspect generated code or search results before they are accepted or used in subsequent automated steps. IBM highlights HITL as a way to keep humans as **supervisors and decision-makers**, while the agent automates tedious exploration and drafting tasks.[6]

The pattern illustrates how tools like LangGraph can be used in IDE-like or code-assistant settings to insert mandatory review checkpoints, enforce human oversight on critical outputs, and maintain a clear separation between automated suggestion and human approval.[6]

-----

-----

### Source [27]: https://www.permit.io/blog/human-in-the-loop-for-ai-agents-best-practices-frameworks-use-cases-and-demo

Query: How is the concept of a 'human-in-the-loop' being implemented in modern AI-powered developer tools like IDE extensions or code assistants?

Answer: This post discusses **human-in-the-loop (HITL)** as a control and safety mechanism for AI agents, outlining patterns and frameworks that can be applied to AI-powered developer tools.[1]

HITL is described as a way to insert humans at key decision points to **prevent irreversible mistakes**, ensure accountability, comply with audit requirements, and build trust by keeping the AI as a supervised assistant rather than an unsupervised actor.[1] The article argues that in many agentic workflows, HITL is the only responsible way forward.[1]

It surveys frameworks like **LangGraph**, **CrewAI**, **HumanLayer**, **LangChain MCP adapters**, and **Permit.io + MCP**, highlighting their HITL features:[1]

- **LangGraph**: Graph-based control with native `interrupt()` support to pause workflows and wait for human input before resuming.[1]
- **CrewAI**: Multi-agent orchestration with `human_input` and a **HumanTool** that lets agents route questions to humans when they need guidance.[1]
- **HumanLayer**: SDK/API for injecting human decisions across channels (Slack, email, Discord) via decorators like `@require_approval()` and `human_as_tool()`.[1]
- **Permit.io + MCP**: Policy-driven approval flows and delegated access control, including UI/API-based approvals for operations and access requests.[1]

The post describes several concrete **HITL patterns** relevant to developer tools:[1]

- **Human-as-a-tool**: Agents treat a human endpoint as a callable tool when unsure, using the human’s response as context.[1]
- **Approval gates**: Human checkpoints before sensitive actions like deployments or data access.[1]
- **Fallback escalation**: When an agent fails or lacks permission, it escalates tasks to humans through channels like Slack or dashboards.[1]

A demo system integrates LangGraph, MCP adapters, Permit.io, and an LLM to show policy-driven approvals, real-time interruptions, logging, and auditable workflows—all patterns that can be embedded into IDE extensions or code assistants.[1]

-----

-----

### Source [28]: https://developer.nvidia.com/blog/build-your-first-human-in-the-loop-ai-agent-with-nvidia-nim/

Query: How is the concept of a 'human-in-the-loop' being implemented in modern AI-powered developer tools like IDE extensions or code assistants?

Answer: NVIDIA’s post outlines how to build a **human-in-the-loop AI agent** using **NVIDIA NIM** microservices, focusing on a cognitive workflow where agents assist and humans make final decisions.[4]

The architecture defines a **human–agent decision-making workflow** in which AI agents handle perception, analysis, or proposal steps, while humans retain control over final approvals or high-impact actions.[4] Figure 1 (described in the post) depicts interactions between human decision-makers and agents, showing how tasks move from the user to the agent and back to the user for verification or refinement.[4]

Using NIM—NVIDIA’s accelerated inference microservices—the system can integrate multiple models and tools, with humans positioned as overseers who confirm or adjust the agent’s outputs before they are enacted.[4] The focus is on creating a loop: the agent proposes solutions or actions, the human evaluates them, and feedback is then used to refine subsequent model behavior.[4]

Although the blog is not IDE-specific, the described pattern maps directly to AI-powered developer tools that:

- Let the agent draft code, configuration, or changes.
- Surface these drafts to the developer inside the tool for inspection.
- Require explicit human confirmation before applying changes to codebases or environments.

The post presents HITL as an intentional workflow design—rather than an afterthought—where human judgment is embedded as a **mandatory step** in the decision chain, ensuring that automated reasoning remains subordinate to human oversight.[4]

-----

-----

### Source [29]: https://cloud.google.com/discover/human-in-the-loop

Query: How is the concept of a 'human-in-the-loop' being implemented in modern AI-powered developer tools like IDE extensions or code assistants?

Answer: Google Cloud’s article defines **human-in-the-loop (HITL)** as a design approach where humans are **actively involved in the training and operation** of AI systems, providing input and feedback alongside automated processing.[5]

HITL addresses the limits of ML models by bringing in human expertise for tasks that require nuanced judgment, contextual understanding, or handling incomplete information.[5] Humans can interact with HITL systems in multiple ways, including reviewing model outputs, correcting mistakes, labeling or re-labeling data, and providing feedback that becomes part of a continuous improvement loop.[5]

The article emphasizes that human collaboration makes ML systems more **adaptable** and better aligned with changing real-world conditions and user preferences.[5] By integrating humans into the pipeline, AI systems can navigate complexities that purely algorithmic approaches struggle with, such as ambiguous requests or domain-specific edge cases.[5]

Although the piece is general rather than tool-specific, the patterns it describes underpin how modern AI developer tools implement HITL:

- Having developers review and approve code suggestions before they are committed.
- Using user feedback and corrections in IDEs to refine future model behavior.
- Combining automated code analysis with human judgment for edge cases and policy-sensitive decisions.

Google positions HITL as a core **design principle** for safe, reliable AI systems rather than a narrow technical feature, encouraging developers to build workflows where human supervision and feedback are first-class components throughout the model lifecycle.[5]

-----

</details>

<details>
<summary>What are best practices for designing a 'coding-like' user experience for human-in-the-loop content generation workflows in IDEs like Cursor?</summary>

### Source [30]: https://cursor.com/learn/context

Query: What are best practices for designing a 'coding-like' user experience for human-in-the-loop content generation workflows in IDEs like Cursor?

Answer: Cursor’s **Context** system is designed to keep AI assistance feeling like normal coding by tightly integrating model actions with code navigation and editing.[4]

Key practices relevant to coding‑like, human‑in‑the‑loop UX:

- **Plan + execute inside the IDE**: Cursor supports having a *plan* (sometimes model‑suggested, sometimes human‑written) that breaks work into concrete coding tasks, which are then executed step by step in the editor.[4]
- **Todo list as first‑class artifact**: Plans become a visible todo list that the agent checks off as it completes tasks, mirroring how developers track work items, and keeping progress legible to the user.[4]
- **Contextual grounding in files**: The AI is explicitly driven by the open files, selected code, and project context, so that generated content is always anchored in the existing codebase rather than a detached chat thread.[4]
- **Iterative, not monolithic, changes**: The system is oriented toward small, incremental edits tied to todo items, which makes it easy for humans to review diffs and stay in control of the code evolution.[4]
- **User‑driven flow**: Humans can supply or adjust the plan, decide which tasks to run, and when to stop or redirect, preserving a tight feedback loop between human intent and AI actions.[4]

For designing coding‑like workflows, these patterns suggest: expose plans as editable checklists, keep AI actions file‑ and selection‑scoped, present progress via task completion, and structure interactions as incremental diffs that users can inspect and accept inside the IDE.

-----

-----

### Source [31]: https://cursor.com/docs/cookbook/web-development

Query: What are best practices for designing a 'coding-like' user experience for human-in-the-loop content generation workflows in IDEs like Cursor?

Answer: Cursor’s **Web Development** guide shows how to make AI assistance feel like an extension of normal coding flow rather than a separate UI.[7]

Relevant best practices from this guide:

- **Tight feedback loop**: The guide emphasizes configuring Cursor so that code changes can be generated, run, and validated quickly, mirroring a standard dev workflow where you edit, refresh, and test continuously.[7]
- **Use chat for orchestration, editor for code**: Chat is framed as a place to *orchestrate* work (describe goals, ask for refactors, request components), while the resulting code is applied as diffs in the editor, preserving the familiar edit‑review‑commit cycle.[7]
- **Context from running app & files**: The workflows assume the model leverages current project files and relevant config so that suggestions are grounded in the real app structure, not abstract snippets.[7]
- **Incremental refactors and features**: Suggested flows involve repeatedly asking the AI for small, focused changes (e.g., build a component, adjust styling, wire an API) and then inspecting the changes in the diff view, which keeps the human in the loop on every step.[7]
- **Leaning on existing tools**: The guide assumes continued use of normal dev tooling (framework CLIs, local dev server, browser devtools), with AI integrated around them instead of replacing them. This keeps the UX close to how developers already work.[7]

For human‑in‑the‑loop content generation, this implies: use chat to define goals and constraints, always materialize model output as reviewable diffs, keep iterations small, and align AI flows with existing run/test/debug cycles.

-----

-----

### Source [32]: https://www.datacamp.com/tutorial/cursor-ai-code-editor

Query: What are best practices for designing a 'coding-like' user experience for human-in-the-loop content generation workflows in IDEs like Cursor?

Answer: The DataCamp guide on Cursor describes concrete UI and interaction patterns that make AI assistance feel like coding rather than generic text generation.[2]

Key UX practices:

- **Inline generation bound to selections**: Pressing `Cmd+K` opens a small inline prompt attached to the current cursor position or selected code, allowing users to request edits (e.g., refactors) in place.[2] This keeps the interaction localized and code‑centric.
- **Diff‑based review**: Cursor presents proposed changes as a diff, with red lines for deletions and green lines for additions.[2] Users explicitly accept or reject the edit, maintaining human control over code integration.
- **Keyboard‑first flow**: Shortcuts like `Cmd+K` for inline generation and `Cmd+L` for chat integrate AI actions into the same command palette mental model developers already use.[2]
- **Autocomplete & predictive edits**: Multi‑line autocomplete, smart rewrites, and natural‑language‑to‑code suggestions appear inline and are accepted with `Tab`, mimicking traditional IDE completion but powered by LLMs.[2]
- **Separation of inline vs chat**: Inline generation is optimized for small, localized edits; the chat window is positioned as a more versatile area for broader questions and larger code generation, with an explicit **Apply** button to bring code into the project.[2]
- **Model transparency and choice**: The inline generator exposes a model dropdown (e.g., selecting `claude-3.5-sonnet`), making the underlying model visible and selectable while keeping the main workflow consistent.[2]

These patterns support human‑in‑the‑loop content generation by: anchoring prompts to concrete code selections, always showing structured diffs, enabling keyboard‑driven acceptance, and providing distinct spaces for local edits vs higher‑level planning.

-----

-----

### Source [33]: https://blog.sshh.io/p/how-cursor-ai-ide-works

Query: What are best practices for designing a 'coding-like' user experience for human-in-the-loop content generation workflows in IDEs like Cursor?

Answer: Shrivu Shankar’s analysis of how Cursor works surfaces several design patterns that support coding‑like, human‑in‑the‑loop workflows.[1]

Key points:

- **Tool‑driven agent architecture**: Cursor wraps an LLM with explicit tools like `read_file(full_path: str)` and `write_file(full_path: str, content: str)` so that AI operations are concrete file actions, not opaque text generation.[1]
- **Static, cacheable system prompts**: Cursor uses a largely static system prompt and tool descriptions so calls can leverage prompt caching, reducing latency and making iterative, stepwise interactions feasible in an IDE context.[1]
- **Explicit edit protocol**: The `edit_file` tool requires representing unchanged code as language comments, which supports an “apply model” that can reliably map model output back to precise file edits.[1]
- **High‑signal lint feedback**: The article notes that lint feedback is extremely high signal for the agent and highlights investment in a strong linter; this lets the AI align with the same feedback signals developers already use (errors, warnings) when generating or fixing code.[1]
- **File naming & structure for disambiguation**: Recommended practices include using unique filenames, preferring full paths in docs, and co‑locating hot‑path code, so that AI‑driven tools can select and edit the correct files deterministically.[1]
- **Prompt discipline and UX**: The Cursor system prompt explicitly instructs the model to avoid unnecessary apologies and to follow specific conventions when emitting edits, which improves readability and predictability for users.[1]

For designing similar workflows, these patterns suggest: expose file‑level tools, enforce a structured edit format, integrate compiler/linter signals into the loop, optimize for low‑latency iterative calls, and design prompts to yield predictable, reviewable code edits rather than free‑form text.

-----

-----

### Source [34]: https://cursor.com/students

Query: What are best practices for designing a 'coding-like' user experience for human-in-the-loop content generation workflows in IDEs like Cursor?

Answer: Cursor’s students page highlights how the product’s workflow maintains a coding‑like experience even when AI is heavily involved.[6]

Relevant aspects:

- **Flow preservation**: Testimonials emphasize that Cursor helps users “clean datasets, debug pipelines, and write complex queries without breaking flow,” indicating that AI interactions are designed to occur within the normal editing environment rather than forcing context switches.[6]
- **Focus on core problem‑solving**: By automating routine code generation and debugging, Cursor frees users to focus on research or higher‑level problem‑solving, aligning with a human‑in‑the‑loop model where humans set direction and review results.[6]
- **Integration in familiar tooling**: The site presents Cursor as an AI‑powered code editor, not a separate assistant, reinforcing that AI features are embedded directly in the IDE so the UX remains that of a typical coding environment with enhanced capabilities.[6]

These points support designing experiences where AI augments, rather than replaces, standard coding workflows: keep AI actions in‑editor, minimize context switching, and ensure users remain responsible for directing and validating generated content.

-----

-----

### Source [35]: https://www.latent.space/p/cursor

Query: What are best practices for designing a 'coding-like' user experience for human-in-the-loop content generation workflows in IDEs like Cursor?

Answer: The Latent Space interview with Cursor co‑founder Aman Sanger provides insight into UX principles behind AI‑first IDEs.[3]

Key practices for coding‑like, human‑in‑the‑loop workflows:

- **Retrieval over monolithic context**: Cursor uses retrieval models to surface the specific file, function, or class relevant to a query, rather than depending on users to craft massive prompts, which keeps interactions close to how developers naturally navigate code.[3]
- **Addable docs and URLs as context**: Users can “add files or add documentation” (e.g., Next.js docs) directly into chat or Command‑K, and Cursor crawls and embeds those docs in the background.[3] This lets users gradually build a working context in flow instead of front‑loading specification.
- **Step‑by‑step change flows**: Aman notes that agentic systems that produce large amounts of code at once can be hard to trust, and that Cursor instead focuses on flows where users “see a change, then go step by step from there,” which is more efficient and easier to validate.[3]
- **Emphasis on fun and ease of use**: The product philosophy is that the more fun and easy‑to‑use coding assistant will win, assuming similar capabilities, which motivates designs that let users iteratively explore, tweak, and inspect changes rather than manage complex prompts.[3]
- **Local, user‑specific knowledge**: Crawled documentation is stored locally for the user, allowing personalized, project‑specific assistance that feels like working with a teammate familiar with the same code and docs.[3]

These ideas suggest best practices such as: rely on retrieval to infer relevant context, let users incrementally add knowledge sources, design UIs for small, inspectable changes, and structure interactions to reduce prompt‑writing burden while keeping humans in the approval loop.

-----

</details>

<details>
<summary>How can LangGraph's Functional API, specifically the '@entrypoint' and 'interrupt' functions, be used to manage state and resume complex, multi-step AI workflows after a human-in-the-loop event?</summary>

### Source [36]: https://docs.langchain.com/oss/python/langgraph/use-functional-api

Query: How can LangGraph's Functional API, specifically the '@entrypoint' and 'interrupt' functions, be used to manage state and resume complex, multi-step AI workflows after a human-in-the-loop event?

Answer: The "Use the functional API" guide explains that the **`@entrypoint` decorator turns a normal Python function into a LangGraph workflow** that automatically gets persistence, state management, and human‑in‑the‑loop support.[1] Within this entrypoint you can call graphs built with the Graph API (`StateGraph`) or other Functional‑API tasks, and LangGraph manages the workflow state and checkpoints behind the scenes.[1]

To manage state across calls, the entrypoint can be configured with a **checkpointer**, for example `InMemorySaver()`, and a **`thread_id`** in the runtime `config`.

```python
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(x: int) -> dict:
    result = graph.invoke({"foo": x})
    return {"bar": result["foo"]}

config = {"configurable": {"thread_id": str(uuid.uuid4())}}
workflow.invoke(5, config=config)  # state/checkpoints tied to this thread
```

Because each invocation is associated with a thread, LangGraph can **resume the workflow from checkpoints** instead of recomputing everything, which is critical for long‑running, multi‑step flows.[1]

The guide shows human‑in‑the‑loop with **`interrupt` inside a `@task` used from an `@entrypoint`**:

```python
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt

@task
def human_feedback(input_query):
    feedback = interrupt(f"Please provide feedback: {input_query}")
    return f"{input_query} {feedback}"
```

When `interrupt(...)` is reached, **execution pauses and a checkpoint is written** capturing current state and location.[1] The application (e.g., a server) then surfaces the prompt to a human. After the human supplies feedback, the runtime is called again with that answer, and LangGraph **automatically resumes execution from the interrupt point**, using the stored state in the configured checkpointer and thread.[1]

Because entrypoints can orchestrate multiple graphs and tasks with normal control flow, you can implement complex, multi‑step workflows that pause for human review at arbitrary points and later resume from exactly where they left off, without manual state plumbing.[1]

-----

-----

### Source [37]: https://docs.langchain.com/oss/python/langgraph/functional-api

Query: How can LangGraph's Functional API, specifically the '@entrypoint' and 'interrupt' functions, be used to manage state and resume complex, multi-step AI workflows after a human-in-the-loop event?

Answer: The **Functional API overview** describes that `@entrypoint` and `@task` functions execute on the same LangGraph runtime as StateGraphs, providing **persistence, memory, human‑in‑the‑loop, and streaming** with minimal changes to existing code.[3]

An **entrypoint** is the top‑level workflow function. When it is decorated with `@entrypoint(checkpointer=...)`, each call is associated with a **thread plus a sequence of checkpoints** that capture inputs, outputs, and internal progress.[3] This gives **automatic state management** for complex multi‑step workflows: you call other tasks or graphs, and the runtime stores the intermediate state and where in the call stack you are.

**Tasks** can only be called from an entrypoint, another task, or a StateGraph node and **return futures** immediately; calling `.result()` on a future blocks until completion.[3] This allows long chains of dependent steps while LangGraph continues to **checkpoint at task boundaries**, enabling resumption and caching.[3]

For human‑in‑the‑loop, the overview explains that you use the **`interrupt` primitive (from `langgraph.types`) inside a task**. When `interrupt` is executed, LangGraph:

- Writes a **checkpoint containing the current state and the interrupt request** (e.g., a question for the user).
- Returns control to the caller, so the application can present this request to a human.[3]

Later, when the human response is available, you **call back into the same entrypoint/thread configuration**, providing the answer. The runtime **restores the checkpointed state and resumes exactly after the `interrupt` call**, passing the provided value back into the task as the result of `interrupt(...)`.[3]

This pattern lets you model complex, multi‑step AI workflows with arbitrary Python control flow, while relying on `@entrypoint` + checkpointer for state and on `interrupt` to pause and resume after human input, without hand‑rolled persistence logic.[3]

-----

-----

### Source [38]: https://blog.langchain.com/introducing-the-langgraph-functional-api/

Query: How can LangGraph's Functional API, specifically the '@entrypoint' and 'interrupt' functions, be used to manage state and resume complex, multi-step AI workflows after a human-in-the-loop event?

Answer: The Functional‑API introduction blog states that the API is built around **two decorators: `entrypoint` and `task`**, which let you define workflows in standard imperative Python while still using LangGraph’s runtime.[4] This enables features including **human‑in‑the‑loop, short‑term and long‑term memory, and streaming** in existing codebases.[4]

**`@entrypoint`**: wraps a top‑level function so that it runs under the LangGraph engine, gaining:

- **Stateful execution and checkpointing**.
- The same `.invoke`, `.stream`, `.batch` interface as graphs.
- Integration with checkpointers and thread IDs for persistent workflows.[4]

**`@task`**: marks internal steps that should be part of the workflow graph; their calls are tracked, checkpointed, and observable.[4]

The blog emphasizes **human‑in‑the‑loop** support as a key use case: by combining entrypoints and tasks with the **`interrupt` primitive**, you can pause a workflow at any step (e.g., to review a tool call or get user feedback), then resume it later with additional context.[4]

Because both Graph and Functional APIs share the same runtime, the blog notes you can **mix paradigms**: an entrypoint can orchestrate multiple StateGraphs and tasks, and human‑in‑the‑loop interruptions can occur **inside either functional tasks or graph nodes** while still benefiting from centralized state management and resumption.[4]

Observability is also highlighted: inputs/outputs and intermediate steps of entrypoints and tasks are automatically logged by LangGraph, which makes it straightforward to inspect checkpoints around `interrupt` points and understand how and where workflows resume after human interaction.[4]

-----

-----

### Source [39]: https://changelog.langchain.com/announcements/functional-api-for-langgraph

Query: How can LangGraph's Functional API, specifically the '@entrypoint' and 'interrupt' functions, be used to manage state and resume complex, multi-step AI workflows after a human-in-the-loop event?

Answer: The Functional‑API changelog announcement explicitly lists **"Seamless state management"** and **"Human‑in‑the‑loop support"** as core motivations.[7]

- **Seamless state management**: With the Functional API, you can "keep track of user inputs, workflow progress, and historical context" without needing explicit graph syntax.[7] This is handled by the LangGraph runtime when using `@entrypoint` (typically with a checkpointer and thread IDs), which stores and restores workflow state as it advances.

- **Human‑in‑the‑loop support**: The announcement says you can "pause workflows, collect feedback, and resume dynamically".[7] In practice this is accomplished via the `interrupt` mechanism used within tasks or nodes that are orchestrated by an entrypoint. When `interrupt` is hit, the workflow pauses, a checkpoint is written, and later the same workflow can be resumed from that point with the human response supplied.[7]

The announcement also stresses that the Functional API **"works anywhere"** and can be used in any app without graph syntax, and that you can **mix Graph + Functional APIs** to "build robust AI systems".[7] For complex, multi‑step AI workflows, this means you can:

- Use an `@entrypoint` function as the orchestration layer that maintains state across many calls and sessions.
- Place `interrupt` calls at key decision points where human oversight is needed.
- Rely on LangGraph’s underlying engine to checkpoint, restore state, and continue the workflow once the human‑in‑the‑loop event is resolved.[7]

-----

-----

### Source [40]: https://www.youtube.com/watch?v=NXhyWJozM8A

Query: How can LangGraph's Functional API, specifically the '@entrypoint' and 'interrupt' functions, be used to manage state and resume complex, multi-step AI workflows after a human-in-the-loop event?

Answer: In the "LangGraph Functional API Overview" video, the presenter explains that the Functional API **"manages state for you, in particular checkpointing and human‑in‑the‑loop"** when using `@entrypoint` and `@task`.[5]

They note that marking the top function with **`@entrypoint(checkpointer=...)`** causes every step inside (including downstream tasks) to be **checkpointed into a thread**, keyed by a `thread_id` passed via the `config` argument.[5] This means you can stop and later restart the workflow from a given checkpoint without re‑running previous steps, which is important for long, multi‑step flows.

The video highlights that the **`@task` decorator ensures functions called by the entrypoint are saved as checkpoints**, which is important both for **caching results** and for resuming after interruptions.[5]

For human‑in‑the‑loop, the presenter demonstrates a task that includes an `interrupt` call (in a tool‑calling scenario). When execution reaches the `interrupt`, the workflow **pauses and emits an interrupt event**; a checkpoint is written that encodes the current state and the pending request to the human.[5] The external application then:

1. Shows this request to the human (e.g., "Approve this tool call?").
2. Collects the human’s decision.
3. Calls back into the same workflow (same entrypoint/thread/checkpoint) with that decision.

LangGraph then **re‑executes from that checkpoint**, automatically injecting the human’s response as the return value of `interrupt`, and the rest of the workflow continues (e.g., the approved tool is executed, the LLM call completes, and a final answer is produced).[5]

Later in the video, they show using the checkpoint list to **fork from any prior checkpoint**, demonstrating how stateful management plus interrupts enable sophisticated, resumable workflows and branches.[5]

-----

</details>

<details>
<summary>What are the benefits of decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) when building systems that require flexible human oversight?</summary>

### Source [41]: https://www.finops.org/wg/model-context-protocol-mcp-ai-for-finops-use-case/

Query: What are the benefits of decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) when building systems that require flexible human oversight?

Answer: Decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) enables context modularity, where model usage is separated from data access. This separation allows AI applications to remain lightweight and focused on interaction logic, while delegating data access and tool execution to the server layer. For systems requiring flexible human oversight, this modularity simplifies how context is managed and updated, ensuring that oversight mechanisms can be applied consistently across different tools and data sources without being tightly coupled to any single implementation. MCP servers inject accurate, up-to-date cloud service provider (CSP) information into the model’s context, which reduces hallucinations and improves technical accuracy, making AI responses more reliable and easier for human overseers to validate. The protocol also supports workflow integration with tools like CDK, Terraform, or CLI, exposing reusable workflows that the model can invoke directly. This reduces manual interventions for repetitive tasks, freeing human oversight to focus on higher-level decisions. MCP servers also support tools, agents, and capabilities discovery: clients can interrogate the server for available tools and capabilities, which the server curates and keeps updated. This allows oversight systems to dynamically adapt to new or changed tools without requiring code changes, supporting flexible and evolving oversight requirements.

-----

-----

### Source [42]: https://www.bcg.com/publications/2025/put-ai-to-work-faster-using-model-context-protocol

Query: What are the benefits of decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) when building systems that require flexible human oversight?

Answer: Using a standardized framework like MCP to decouple AI agent tools allows integrations to be reused, so development effort increases only linearly as more agents are created, making scaling cheaper and faster. This decoupling helps build better agents that evolve from rigid, prompt-based workflows to more autonomous agents capable of dynamic, session-based interactions that reference prior activity. For systems requiring flexible human oversight, this means agents can reason about which tools to use and how to sequence them, based on capabilities announced by MCP servers, including prompt templates that guide effective tool use. This improves task autonomy and execution, as agents can dynamically coordinate across complex, multi-tool workflows without brittle, hard-coded logic, allowing human oversight to focus on exceptions and strategic decisions rather than micromanaging steps. MCP also improves memory and knowledge use by enabling agents to access real-time transaction data or vector databases, augmenting their contextual understanding. This richer context supports more informed agent behavior and more meaningful human oversight, as decisions are based on current, accurate data rather than static or outdated model knowledge.

-----

-----

### Source [43]: https://modelcontextprotocol.io

Query: What are the benefits of decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) when building systems that require flexible human oversight?

Answer: The Model Context Protocol (MCP) is an open standard that enables secure, two-way connections between data sources and AI-powered tools, decoupling the AI model from direct integration with each individual tool. This decoupling allows developers to build AI applications or agents that can access an ecosystem of data sources, tools, and apps through a single, standardized interface. For systems that require flexible human oversight, this means that oversight mechanisms do not need to be rebuilt or reconfigured for every new tool or data source; instead, they can rely on the MCP layer to uniformly expose capabilities and context. MCP enables agents to access real-world systems such as calendars, documents, databases, and even physical devices (like 3D printers), acting as more capable, personalized assistants. Because MCP provides a consistent way for agents to discover and use tools, human overseers can more easily understand and audit what actions an agent is capable of and has taken, improving transparency and control. This standardization reduces complexity and friction when linking different systems, allowing oversight to be more adaptive and scalable as new tools are added to the environment.

-----

-----

### Source [44]: https://www.moveworks.com/us/en/resources/blog/model-context-protocol-mcp-explained

Query: What are the benefits of decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) when building systems that require flexible human oversight?

Answer: Decoupling AI agent tools using a standardized framework like MCP reduces the complexity of linking different systems, which is especially valuable in environments that require flexible human oversight. MCP provides a standardized way to integrate APIs, so developers can use a common protocol to connect various systems instead of creating separate, specific integrations for each tool. This standardization simplifies development and accelerates the rollout of new features, allowing oversight mechanisms to be implemented more consistently across different tools and data sources. For systems with human oversight, this means that oversight policies, monitoring, and auditing can be applied uniformly through the MCP layer, rather than being duplicated or adapted for each individual integration. MCP also minimizes friction when connecting AI to enterprise apps, databases, or customer service platforms, enabling smoother integrations that are easier to manage and supervise. By eliminating the need for separate integrations for each tool, MCP reduces the workload for developers and allows them to focus on building robust oversight and governance features. This leads to more effective and responsive AI experiences where human oversight can be more agile, adapting quickly as new tools or data sources are added to the system.

-----

-----

### Source [45]: https://www.merge.dev/blog/model-context-protocol

Query: What are the benefits of decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) when building systems that require flexible human oversight?

Answer: The Model Context Protocol (MCP) simplifies the build process by providing a single, standard protocol for LLM providers and SaaS applications to integrate with one another, which directly benefits systems that require flexible human oversight. By decoupling AI agent tools through this standardized framework, developers avoid the need to create custom, point-to-point integrations for each tool, reducing complexity and making it easier to manage and audit agent behavior. MCP supports workflow definitions by providing a structured way for LLMs to retain, update, and get context, enabling LLMs to manage and progress workflows autonomously while still operating within a governed framework. For human oversight, this means that workflows and context are explicitly defined and accessible, making it easier to monitor, intervene, or adjust agent behavior as needed. MCP also enhances LLM efficiency by standardizing context management, which reduces unnecessary processing and makes agent actions more predictable and interpretable for human overseers. Additionally, MCP strengthens security and compliance by offering standardized governance over how context is stored, shared, and updated across environments. This standardized governance is critical for oversight, as it ensures consistent enforcement of access controls, data handling policies, and audit trails, even as the set of connected tools evolves over time.

-----

-----

### Source [46]: https://cloud.google.com/discover/what-is-model-context-protocol

Query: What are the benefits of decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) when building systems that require flexible human oversight?

Answer: The Model Context Protocol (MCP) helps make LLMs more versatile, reliable, and capable by enabling them to connect with many ready-made tools and integrations, such as business software, content repositories, and development environments. When AI agent tools are decoupled using a standardized framework like MCP, LLMs are no longer limited to their training data, which can quickly become outdated, but can instead access real-time, external data sources and tools. This reduces hallucinations and improves the truthfulness of responses, which is essential for systems requiring flexible human oversight, as it makes agent behavior more predictable and easier to validate. MCP allows AI to handle more complex, real-world tasks that involve interacting with external systems, such as updating customer records or looking up current information, while still operating within a standardized, governed interface. For oversight, this means that human operators can rely on more accurate and up-to-date information when reviewing or intervening in agent actions. The protocol’s ability to connect AI with a wide range of tools in a consistent way also supports flexible oversight by enabling a uniform way to monitor, audit, and control which tools the agent can access and how it uses them, even as the toolset evolves.

-----

-----

### Source [47]: https://www.hiberus.com/en/blog/the-future-of-connected-ai-what-is-an-mcp-server/

Query: What are the benefits of decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) when building systems that require flexible human oversight?

Answer: The Model Context Protocol enables AI to connect with data sources via an MCP server, which facilitates communication between the model and external systems without requiring the model to manage direct integrations. This decoupling of tools through a standardized framework like MCP provides real-time access to databases and APIs, eliminating outdated responses and reliance on re-indexing processes that are common in architectures like RAG. For systems that require flexible human oversight, this real-time access ensures that agents operate on current, accurate data, making their decisions and actions more reliable and easier for human overseers to understand and validate. Because the MCP server retrieves the appropriate data source and returns information without additional processing, the oversight layer can focus on interpreting and acting on that information rather than managing data retrieval logic. This architecture simplifies how oversight systems interact with AI agents, as they can rely on a consistent, standardized interface to monitor agent context and actions. The separation of concerns between the AI model and the data/tools layer also makes it easier to implement and evolve oversight policies, such as access controls and audit trails, independently of the agent’s core logic.

-----

-----

### Source [48]: https://www.anthropic.com/news/model-context-protocol

Query: What are the benefits of decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) when building systems that require flexible human oversight?

Answer: The Model Context Protocol (MCP) is an open standard that enables developers to build secure, two-way connections between their data sources and AI-powered tools. By decoupling AI agent tools using this standardized framework, developers can create AI systems where the model is not directly responsible for managing each individual integration with external systems. This separation allows the AI agent to focus on reasoning and interaction, while the MCP layer handles secure access to data and tools. For systems that require flexible human oversight, this architecture supports clearer boundaries between the agent’s decision-making and the execution of actions, making it easier to monitor, audit, and intervene in agent behavior. Because MCP defines a standard way to connect to data sources and tools, oversight mechanisms can be built once and applied consistently across different integrations, rather than being customized for each tool. This standardization also enhances security and control, as access to data and tools can be governed centrally through the MCP layer, ensuring that human overseers have a unified view of what the agent can access and what actions it can perform, even as new tools are added.

-----

-----

### Source [49]: https://www.ibm.com/think/topics/model-context-protocol

Query: What are the benefits of decoupling AI agent tools using a standardized framework like the Model Context Protocol (MCP) when building systems that require flexible human oversight?

Answer: The Model Context Protocol (MCP) allows AI agents to be context-aware while complying with a standardized protocol for tool integration. By decoupling AI agent tools using this standardized framework, agents can dynamically access and use various tools and data sources through a consistent interface, rather than being tightly coupled to specific implementations. This supports flexible human oversight because it enables a uniform way to manage, monitor, and audit how agents use tools and interpret context across different systems. The standardized protocol ensures that agents operate within defined boundaries for tool access and data handling, which simplifies the implementation of governance, compliance, and security policies that human overseers rely on. Because MCP provides a clear, structured way for agents to interact with external systems, oversight mechanisms can more easily track agent actions, understand the context in which decisions are made, and intervene when necessary. This standardization also makes it easier to adapt oversight policies as new tools or data sources are integrated, since the protocol abstracts away the specifics of each integration and presents a consistent model of capabilities and constraints to both the agent and the oversight layer.

-----

</details>

<details>
<summary>How does Andrej Karpathy's concept of the 'AI generation/human validation loop' apply to practical AI engineering, particularly the principles of 'speeding up verification' and 'keeping the AI on a leash'?</summary>

### Source [50]: https://www.latent.space/p/s3

Query: How does Andrej Karpathy's concept of the 'AI generation/human validation loop' apply to practical AI engineering, particularly the principles of 'speeding up verification' and 'keeping the AI on a leash'?

Answer: Latent Space’s writeup of Karpathy’s “Software 3.0” talk summarizes his **human–AI generation–verification loop** as a core pattern for partial autonomy in AI products.[1]

Karpathy frames modern software as moving toward **“partial autonomy”**: AI systems generate proposals or actions, while humans verify and decide.[1] The loop is:
- **Generation**: the AI proposes code, content, or actions.
- **Verification**: the human rapidly inspects, corrects, or rejects.
- Iterate, with the **tightness and speed** of this loop directly determining productivity and safety.[1]

To make this loop practical in engineering, he emphasizes **two product principles** for AI tools:[1]

1. **Speed up verification (make it easy, fast to win)**
   - The workflow must be designed so that humans can verify AI output *very quickly*.[1]
   - Verification UX should exploit human strengths (e.g., visual inspection, pattern recognition) so that checking is cheaper than doing the task from scratch.[1]
   - In coding tools, this means things like **diff views**, inline annotations, and structured presentation so the engineer can see the impact of AI changes at a glance instead of reading a raw text blob.[1]
   - In knowledge or research tools, this includes surfacing **sources, context, and structure** so that factual claims are easy to audit.[1]

2. **Keep the AI on a tight leash**
   - Karpathy warns against free-roaming agents that execute long, opaque sequences of actions or produce huge outputs without checkpoints.[1]
   - Practical systems instead use **short action horizons**: the AI performs a small step, then stops for human review before proceeding.[1]
   - This keeps failure modes bounded: if the model drifts or hallucinates, the damage is local and caught early in the loop.[1]
   - He characterizes this as designing **copilots**, not fully autonomous agents: the AI extends the engineer (like an **Iron Man suit**) but never replaces human control.[1]

Applied to AI engineering, this loop becomes a design principle: integrate LLMs as tools that generate small, reviewable units (patches, queries, plans), optimize the interfaces and tooling around **fast human verification**, and restrict autonomy so that humans remain the final arbiter at each iteration.[1]

-----

-----

### Source [51]: https://catalaize.substack.com/p/andrej-karpathy-software-is-changing

Query: How does Andrej Karpathy's concept of the 'AI generation/human validation loop' apply to practical AI engineering, particularly the principles of 'speeding up verification' and 'keeping the AI on a leash'?

Answer: This article distills Karpathy’s “Software is changing” talk and explicitly details his **AI generation / human verification loop** and its engineering implications.[2]

Karpathy is described as advocating **partial autonomy applications**: traditional software augmented with LLM **copilots** that operate under constant human supervision.[2] He likens this to an **“Iron Man suit”**: AI provides superhuman speed and knowledge, but the human remains the pilot making key decisions.[2]

The generation–verification loop is presented as the central design pattern for these systems:[2]
- AI **generates** candidate solutions or actions (code edits, research summaries, plans).
- Human **verifies** and either accepts, edits, or rejects.
- The pair iterates, forming a tight feedback loop.

Two explicit principles structure that loop:[2]

1. **“Make verification fast and easy” (speed up verification)**
   - Interfaces must be optimized so humans can **audit AI suggestions quickly**, not read through long unstructured outputs.[2]
   - For coding copilots, the article cites **colored diff views** showing additions and deletions, allowing developers to accept or reject changes with a click.[2]
   - For research assistants, Karpathy emphasizes **source citations for each factual claim**, with links, so users can verify statements efficiently.[2]
   - These choices leverage human strengths—visual pattern recognition and judgment—to counter AI weaknesses like hallucinations and subtle errors.[2]

2. **“Keep the AI on a tight leash”**
   - Karpathy is skeptical of unconstrained agents that autonomously produce **thousands of lines of code or long action chains**, because they tend to **drift off course or make harmful errors** and leave users with massive, opaque outputs to debug.[2]
   - Instead, he recommends **small, controlled increments**: have the AI do a modest step, stop, then let the human verify before continuing.[2]
   - This yields a **tight, iterative loop** where quality and safety stay under human control while still benefiting from AI speed.[2]

The article concludes that, for the near term, Karpathy expects the industry to focus on **copilots over fully autonomous agents**, slowly pushing the **“autonomy slider”** only as verification tooling and interfaces mature.[2] In practical AI engineering, this means designing systems around constrained, inspectable actions and rich verification UX, rather than delegating long, opaque tasks to unchecked agents.[2]

-----

-----

### Source [52]: https://uxdesign.cc/how-to-be-the-designer-in-the-ai-loop-e14a5fb8c375

Query: How does Andrej Karpathy's concept of the 'AI generation/human validation loop' apply to practical AI engineering, particularly the principles of 'speeding up verification' and 'keeping the AI on a leash'?

Answer: This UX-focused article builds directly on Karpathy’s **AI generation / human verification loop**, translating it into concrete interface and workflow patterns for practitioners.[3]

The author references Karpathy’s loop as a way to combine AI **generation** with human **verification** inside broader product development cycles (like the build–measure–learn loop).[3] The goal is to design tools where AI handles repetitive or exploratory work while humans remain responsible for evaluation and direction.[3]

In applying **“speeding up verification”**, the article stresses:
- Present AI outputs in **structured, scannable formats** (tables, highlights, visual summaries) instead of dense text, so humans can verify quickly.[3]
- Use UI mechanisms such as **diffs, side-by-side comparisons, and confidence indicators** to help users see what changed and where to focus their attention.[3]
- Provide **traceability**—e.g., links back to source data or steps the AI took—so humans can drill down only where needed, minimizing total verification time.[3]

To **“keep the AI on a leash,”** the article emphasizes interaction constraints drawn from Karpathy’s framing:[3]
- Limit each AI interaction to **small, reversible steps**, such as suggesting edits rather than directly applying them.[3]
- Insert explicit **checkpoints** where the human must confirm or adjust before the AI proceeds.[3]
- Expose clear **controls** (approve, reject, refine) instead of letting the AI run long autonomous workflows.[3]

For practical AI engineering, this means designing systems and UIs so that:
- AI operates as a **co-designer or co-developer**, always proposing, never silently deciding.[3]
- The **cost of saying “no”** to the AI is low: users can quickly revert, tweak prompts, or narrow scope.[3]
- The loop becomes tight and reliable enough that teams can safely lean on AI for speed without surrendering quality control.[3] The article frames this as the designer’s role in operationalizing Karpathy’s loop: shaping the environment where the human’s verification work is as fast and low-friction as possible, and where the AI’s autonomy is carefully bounded.[3]

-----

-----

### Source [53]: https://www.dwarkesh.com/p/andrej-karpathy

Query: How does Andrej Karpathy's concept of the 'AI generation/human validation loop' apply to practical AI engineering, particularly the principles of 'speeding up verification' and 'keeping the AI on a leash'?

Answer: In this interview, Karpathy discusses his broader view of **agents, copilots, and human oversight**, which aligns with his AI generation / human verification loop, even when not named explicitly.[4]

Karpathy distinguishes between:
- **Intermediate workflows**, where humans still “write a lot from scratch” but use LLM autocomplete as an assistant.[4]
- More autonomous **“vibe coding”** or agents, where you describe what you want and let the model implement it.[4]

Even as he acknowledges the appeal of agents, his comments imply the importance of **keeping AI on a leash** in practice:
- He notes that current agents struggle when they go **far off the data manifold** of the internet and can behave unreliably.[4]
- He points out that LLMs lack an internal process analogous to a human’s **self-review and critical reflection** over many rollouts; they do not naturally stop and inspect their own work.[4]
- Because of this, relying on long, unchecked action sequences is risky; human review remains necessary.[4]

In terms of **verification**, Karpathy contrasts humans and current agents:
- Humans do not blindly generate hundreds of rollouts; they **review, compare, and refine**.[4]
- Present-day LLM agents lack such built-in evaluative loops, so external **human verification** or additional scaffolding is required to achieve similar reliability.[4]

For AI engineering, his remarks support designs where:
- LLMs act as **copilots** (autocomplete, suggestion engines) that humans continuously supervise.[4]
- Systems explicitly insert **review points** after each significant AI action, since the model will not self-audit in a humanlike way.[4]
- Attempts to build more autonomous agents must compensate for missing self-verification mechanisms—either by human-in-the-loop checks or by added evaluation models—rather than trusting long, autonomous runs.[4]

These positions are consistent with his public advocacy for **tight, human-controlled loops**, rather than unconstrained, fully autonomous AI agents.[4]

-----

-----

### Source [54]: https://www.youtube.com/shorts/hTT5o2AoewQ

Query: How does Andrej Karpathy's concept of the 'AI generation/human validation loop' apply to practical AI engineering, particularly the principles of 'speeding up verification' and 'keeping the AI on a leash'?

Answer: In this short interview clip, Karpathy briefly explains why **humans must remain in the loop** when working with AI systems and why agents should be kept on a **tight leash**.[5]

He emphasizes the role of **GUIs (graphical user interfaces)** as essential mediators between humans and powerful AI systems.[5] Rather than letting agents operate invisibly and autonomously, he argues that we need interfaces where humans can **see, understand, and control** what the AI is doing.[5]

This maps directly onto his **AI generation / human verification loop**:
- AI systems generate proposals or actions, but these must be surfaced through a GUI where the human can **inspect and approve** them.[5]
- The GUI is the place where **verification is sped up**: by showing changes, actions, or recommendations in a clear, visual format that makes it easy to verify quickly.[5]

Karpathy warns that letting agents run without such interfaces or oversight is dangerous; they can take actions or make changes that the user does not fully grasp.[5] Keeping AI **“on a leash”** in this context means:
- Constraining what agents can do without explicit user confirmation.[5]
- Ensuring there is always a **human-visible step** before consequential actions are taken.[5]

For practical AI engineering, this implies designing:
- Agent systems that route all significant steps through **transparent, inspectable UI surfaces**.
- Workflows where the human is always empowered to **pause, inspect, and override** the AI, rather than being surprised by autonomous behavior.

Thus, the clip reinforces that the combination of **strong UI/UX and human-in-the-loop control** is how we both accelerate verification and maintain a tight leash on AI agents in real-world tools.[5]

-----

</details>

<details>
<summary>What is the role of the 'evaluator-optimizer' pattern in workflows that iteratively refine content based on a combination of predefined rules and dynamic human feedback?</summary>

### Source [55]: https://www.anthropic.com/news/building-effective-agents

Query: What is the role of the 'evaluator-optimizer' pattern in workflows that iteratively refine content based on a combination of predefined rules and dynamic human feedback?

Answer: The **evaluator-optimizer pattern** is described as a workflow where one LLM call **generates** a response and another LLM call **evaluates and provides feedback**, running in a loop until the output meets the desired standard.[3]

Its role in iterative workflows is to:

- **Structure refinement as a dialogue between two roles**: a *generator* that proposes a draft answer or plan, and an *evaluator* that critiques it against explicit criteria, suggesting concrete improvements.[3]
- **Operationalize clear evaluation criteria**: the pattern is recommended when you can define what “good” looks like—such as correctness, completeness, safety, style, or alignment—and measure these reliably in each iteration.[3]
- **Mimic human editing workflows**: it is explicitly compared to a human writer getting repeated feedback from an editor, turning rough drafts into polished outputs through successive revisions.[3]
- **Exploit human feedback where available**: Anthropic notes it works best when you know that LLM responses can be *demonstrably improved* by human feedback and when the LLM itself can express similar feedback, so human preferences and rules can be encoded as evaluation rubrics that the evaluator applies each round.[3]
- **Provide a mechanism for stopping and convergence**: the evaluator decides when the response is “good enough” given the criteria, at which point the loop stops, effectively optimizing toward those rules and feedback signals.[3]

In workflows combining **predefined rules and dynamic human feedback**, the evaluator-optimizer pattern serves as the **control mechanism** that repeatedly measures the output against both: fixed rule-based checks are baked into the evaluator’s criteria, while new human feedback is incorporated into subsequent evaluations and optimization prompts, gradually steering the generator toward content that better satisfies both static constraints and evolving human expectations.[3]

-----

-----

### Source [56]: https://sageflow.ai/glossary/evaluator-optimizer

Query: What is the role of the 'evaluator-optimizer' pattern in workflows that iteratively refine content based on a combination of predefined rules and dynamic human feedback?

Answer: This source defines the **Evaluator-Optimizer** as an **iterative workflow pattern** where **two distinct LLMs** collaborate in a **continuous feedback loop** to enhance the quality of generated outputs.[1]

Role and structure:

- One LLM acts as the **Generator**, responsible for producing the initial response based on the prompt.[1]
- A second LLM acts as the **Evaluator**, which **meticulously assesses** the Generator’s work and provides **constructive criticism and guidance** for refinement.[1]
- The process repeats until the Evaluator **deems the output satisfactory**, representing **convergence toward optimal quality**.[1]

In iterative refinement settings:

- The pattern is said to **excel when well-defined evaluation criteria exist**, because the evaluator can systematically check outputs against these criteria and request targeted improvements.[1]
- It is compared to a **writer–editor relationship**, where cycles of critique and revision lead to a more polished final product than single-pass generation.[1]
- The evaluator can implicitly reflect **predefined rules** (for example, style guides, correctness checks, or domain constraints) by encoding them into its evaluation rubric and feedback.[1]
- Dynamic **human feedback** can be translated into updated evaluation instructions or criteria for the Evaluator LLM, so future iterations enforce both original rules and newly introduced preferences or corrections.[1]

The source highlights **applications** that naturally mix rules and nuanced judgment—such as literary translation, code refinement, and complex research tasks—where the Evaluator’s role is to repeatedly enforce objective constraints (accuracy, security, coverage) while also improving subjective qualities (faithfulness, clarity, style) through iterative feedback.[1]

-----

-----

### Source [57]: https://docs.praison.ai/docs/features/evaluator-optimiser

Query: What is the role of the 'evaluator-optimizer' pattern in workflows that iteratively refine content based on a combination of predefined rules and dynamic human feedback?

Answer: This documentation presents **Agentic Evaluator-Optimizer** as a **feedback loop workflow** where LLM-generated outputs are **evaluated, refined, and optimized iteratively** to improve **accuracy and relevance**.[2]

Defined role in workflows:

- The pattern **enables iterative solution generation and refinement**, with a **Generator** agent producing candidate solutions and an **Evaluator** agent assessing them.[2]
- It provides **automated quality evaluation**, where the evaluator checks outputs against specified requirements (for example, “are there 10 points about AI?”) and returns signals like *more* vs *done*.[2]
- The **feedback-driven optimization** loop continues while the evaluator indicates that the solution is incomplete or suboptimal, and stops when the evaluator returns a satisfactory verdict (for example, *done*).[2]

Encoding rules and human feedback:

- **Predefined rules** are expressed via the evaluator’s **description, instructions, and expected_output**, which define evaluation criteria and the feedback format.[2]
- The workflow configuration supports **decision-type tasks** where evaluator outputs control process flow (e.g., branching on *more* or *done*), effectively using evaluation results as optimization signals.[2]
- **Dynamic human feedback** can be incorporated by updating the evaluator’s instructions or task descriptions, so future evaluations enforce both existing rules and newly added human preferences.[2]

Operational role:

- The Evaluator-Optimizer acts as **process control** for the optimization cycle, monitoring each generation and deciding whether another refinement step is required.[2]
- It supports **continuous improvement loops**, where successive iterations gradually align the generated content with both formal criteria and evolving human expectations, making it suitable for workflows that must respond to changing or interactive feedback while still obeying fixed constraints.[2]

-----

-----

### Source [58]: https://javaaidev.com/docs/agentic-patterns/patterns/evaluator-optimizer

Query: What is the role of the 'evaluator-optimizer' pattern in workflows that iteratively refine content based on a combination of predefined rules and dynamic human feedback?

Answer: This source describes the **Evaluator-Optimizer pattern** as a way for an LLM to **improve the quality of a previous generation using feedback from an evaluator**, running an **evaluation–optimization loop** multiple times if needed.[4]

Workflow structure and role:

- The pattern is broken into up to **five subtasks**: initializing input, generating the initial result, evaluating the result and providing feedback, optimizing the previous result using that feedback, and finalizing the response.[4]
- Only the **initial generation** is strictly required; evaluation, optimization, and finalization are optional but enable richer iterative refinement.[4]
- The **evaluation task** returns structured results such as a boolean *passed/not passed* or a numeric score (0–100), which then guide whether optimization continues.[4]

Use in rule- and feedback-based refinement:

- **Predefined rules** (for example, coding standards, correctness checks, security rules) are encoded into the evaluation logic and prompts so the evaluator systematically tests outputs against these constraints and issues targeted feedback.[4]
- The **optimization task** instructs the generator to *address all concerns in the feedback*, explicitly using evaluator comments to drive the next revision.[4]
- The loop repeats **until the result passes evaluation or hits a maximum number of evaluations**, making the evaluator-optimizer the mechanism that balances quality against cost/latency.[4]
- The documentation recommends using **different models** for generation and evaluation to improve reliability, reinforcing the distinct roles of creator vs critic.[4]

Dynamic human feedback can be incorporated by modifying evaluation criteria or by injecting human-authored feedback into the optimization prompt, allowing the same loop to integrate both **static rule-based checks** and **new human guidance** as additional constraints the optimizer must satisfy.[4]

-----

-----

### Source [59]: https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-evaluators-and-reflect-refine-loops.html

Query: What is the role of the 'evaluator-optimizer' pattern in workflows that iteratively refine content based on a combination of predefined rules and dynamic human feedback?

Answer: AWS describes an **evaluator workflow / reflect-refine loop** where **one LLM generates a result and another evaluates or critiques it**, forming a **feedback loop** that promotes **self-reflection, optimization, and iterative improvements**.[5]

Role in iterative, rule- and feedback-driven workflows:

- The workflow is **ideal when output quality, accuracy, and alignment are important**, and when single-pass generation is not sufficient.[5]
- The **evaluator** checks outputs against both **subjective metrics** (style, tone, readability) and **objective criteria** (correctness, safety, performance), then suggests improvements or indicates the need for revision.[5]
- This pattern supports **reasoning about trade-offs, constraints, and optimization toward a goal**, using evaluation results to steer the next generation step.[5]
- It helps build **redundancy and quality assurance**, especially in regulated, customer-facing, or creative domains where consistent adherence to rules and standards is critical.[5]

Integration of rules and human feedback:

- **Policy enforcement and alignment checking** are highlighted use cases; here, the evaluator embodies **predefined organizational or regulatory rules**, flagging violations and guiding corrections.[5]
- The workflow also underpins **self-improving agents**, where continuous evaluator feedback shapes better responses over time, including adapting to **human-in-the-loop** reviews when those are available.[5]
- Human feedback signals can be translated into updated evaluation prompts or criteria, and the loop then repeatedly refines future outputs to better match those evolving expectations while remaining within rule-based constraints.[5]

Overall, the evaluator-optimizer role is to provide a **systematic critique-and-revise mechanism** that turns both static rules and dynamic human feedback into concrete guidance for subsequent generations, improving reliability and trustworthiness across iterations.[5]

-----

</details>

<details>
<summary>What are the primary challenges and trade-offs when designing AI-assisted workflows for editing entire documents versus specific, user-selected text sections?</summary>

### Source [60]: https://magai.co/how-ai-improves-content-revision-workflows/

Query: What are the primary challenges and trade-offs when designing AI-assisted workflows for editing entire documents versus specific, user-selected text sections?

Answer: This source describes AI-assisted revision for **entire documents**, emphasizing speed, consistency, and workflow automation. It noted that AI tools can cut overall editing time by 30–60% because they handle grammar, style, and readability across the whole document in a single pass, instead of multiple human passes.[1] Designing full-document workflows therefore trades **coverage and speed** against the risk of over-editing or altering sections the user did not intend to change, making human oversight “crucial” to preserve brand voice and natural tone.[1] Whole-document operations are positioned as best for enforcing **global style guides and compliance standards**, since AI can automatically apply custom rules across all sections and versions, which individual snippet edits cannot guarantee.[1] The article also highlights that many tools are user-friendly with free cores and premium advanced options, implying a design trade-off between exposing powerful full-document controls versus keeping the UI simple for non-experts.[1]

From a workflow perspective, the source stresses that AI should be treated as an **assistant, not a replacement**, which affects how full-document features are surfaced: they support humans rather than fully automating publication.[1] It explains that AI is effective at centralizing feedback, tracking revisions, and routing entire documents to the right reviewers automatically, indicating that full-document context simplifies collaboration and approvals.[1] However, because AI can route based on content type and compliance needs, designers must manage **permissions and review stages** carefully when changes affect complete files, not just local sections.[1] The platform’s ability to “retain context throughout your work” suggests that full-document workflows benefit from persistent conversational context, while selective-section editing must correctly re-anchor that context when users jump between parts.[1] Overall, the source frames full-document AI revision as high-leverage but requiring guardrails, clear approval steps, and persistent context to avoid unintended global changes.[1]

-----

-----

### Source [61]: https://document360.com/blog/ai-solve-the-challenges-in-technical-writing/

Query: What are the primary challenges and trade-offs when designing AI-assisted workflows for editing entire documents versus specific, user-selected text sections?

Answer: This source focuses on AI assistance for **large, frequently updated documentation**, which highlights core trade-offs between editing whole documents and specific sections. It states that with vast documentation and many contributors, maintaining **clarity and consistency** in tone, terminology, and style is a major challenge.[2] AI assistants are described as “game-changers” for this, implying that workflows which operate at the **document or corpus level** are better suited for enforcing uniformity across distributed content than purely local, user-selected edits.[2]

For editing and proofreading, the author notes that complex technical documents slow human proofreading to 4–6 pages per hour, especially for specialized materials like API docs.[2] This motivates full-document AI passes for tasks like error detection and language cleanup. However, because technical content is complex and high-risk, the implicit trade-off is that **aggressive full-document rewrites may endanger accuracy**, so designs should allow constrained, section-level suggestions where precision matters (e.g., code snippets, parameter tables).[2] The source also mentions evolving search behavior and the need for content tailored to user knowledge levels, reinforcing that AI workflows must respect **contextual relevance**—editing a single user-selected section may not account for how users navigate or query the entire knowledge base.[2]

Thus, official guidance here suggests designing workflows where AI can operate over entire document sets to maintain global consistency and searchability, while still enabling writers to apply or reject changes **granularly**, section by section, to preserve correctness and intended audience targeting.[2] The tension between scale (whole-document and corpus operations) and precision (section-level, expert-reviewed changes) is central to the described challenges.[2]

-----

-----

### Source [62]: https://www.docupilot.com/blog/ai-document-automation

Query: What are the primary challenges and trade-offs when designing AI-assisted workflows for editing entire documents versus specific, user-selected text sections?

Answer: This source discusses AI document automation at a workflow level, which clarifies challenges tied to **full-document processing**. It explains that AI-driven processing improves reliability by eliminating **manual inconsistencies**, such as typos, missed fields, and formatting mismatches, resulting in consistent extraction and analysis for entire documents.[3] This underlines a benefit of whole-document flows: they can systematically enforce structure and data integrity in one pass, compared with piecemeal, user-selected edits that may leave inconsistencies.[3]

The article also notes common challenges in AI document automation, especially **cost and resource constraints related to AI training**.[3] Supporting robust whole-document understanding (layout, fields, semantics) demands sizable training investment; designing fine-grained, section-level tools may reduce some complexity but at the cost of weaker global understanding.[3] The piece highlights **error reduction and consistency** as core advantages of AI across complete documents, indicating that workflows should be optimized to let AI see whole forms and templates, not only fragments, for accurate field mapping and compliance checks.[3]

Moreover, it stresses “frictionless workflow automation,” where once a document is generated, AI triggers downstream workflows automatically (e-signature, billing, storage).[3] This model assumes the AI operates at the granularity of a whole document object. It implies a trade-off: designing UIs that allow users to tweak small sections interactively may complicate these straight-through, fully automated flows, requiring mechanisms to re-validate the entire document after local changes.[3] Overall, the source supports architectures where AI mainly processes documents end-to-end, with careful handling of costs, training, and validation when integrating finer-grained, user-selected edits into automated pipelines.[3]

-----

-----

### Source [63]: https://arya.ai/blog/document-workflow-automation

Query: What are the primary challenges and trade-offs when designing AI-assisted workflows for editing entire documents versus specific, user-selected text sections?

Answer: This source examines **document workflow automation** and surfaces challenges relevant to designing AI workflows at different granularities. It highlights the “**complexity of processes and implementation**,” noting that different document types (contracts, invoices, etc.) have varied formats and layouts, making it difficult to find a one-size-fits-all automation solution.[4] For full-document AI editing, this means models must generalize across heterogeneous structures; section-level tools can be simpler but may miss layout-dependent logic (e.g., cross-references, multi-page clauses).[4]

The article stresses **security and compliance** as another major challenge, requiring robust access controls, encryption, and audit trails.[4] Whole-document operations raise higher stakes because a single AI action can alter or expose large amounts of sensitive content; fine-grained, user-selected editing offers a narrower blast radius but complicates auditing many small changes.[4] The guidance to “identify repetitive tasks” and validate them with **human-in-the-loop** suggests that designs should combine automated passes—often at document scale—with checkpoints where users approve or refine specific sections.[4]

It also advises **choosing the right tools** based on scalability, integration, and user-friendliness.[4] Full-document AI pipelines are attractive for scalability and system integration (batch processing, routing), while section-level, interactive tools are more user-friendly for nuanced edits. The source further recommends **standardizing data** via standard document formats and processes to ensure consistency and accuracy.[4] Such standardization favors whole-document workflows (uniform templates, schemas), yet user-selected editing must still respect those standards, implying constraints on what local edits are allowed. Overall, the source underscores balancing full-document automation (for efficiency and compliance) with human-controlled, granular edits (for correctness and usability), supported by strong governance and standardization.[4]

-----

-----

### Source [64]: https://provectus.com/document-automation-with-ai-major-challenges-opportunities/

Query: What are the primary challenges and trade-offs when designing AI-assisted workflows for editing entire documents versus specific, user-selected text sections?

Answer: This source analyzes **AI-powered document processing** and its main challenges, offering insights for designing workflows across entire documents versus sections. It emphasizes the need to **assess and prepare data** by cleaning, formatting, and categorizing it so AI models can learn effectively.[5] For full-document workflows, this entails large-scale preparation of diverse document types, whereas section-level tools may be deployed with more limited context but risk misinterpretation if the broader document structure is not well-modeled.[5]

The article advises training models on **diverse datasets** so they can recognize and classify documents with varying types, formats, and complexity.[5] This supports comprehensive, whole-document understanding, which is essential for tasks such as classification, extraction, and routing. However, the broader and more varied the target documents, the more resources and time are required—a trade-off for designing general full-document systems compared to more scoped, feature-level or section-based helpers.[5]

A key challenge it identifies is **dependency on external providers**, suggesting that organizations should seek solutions with in-house customization and integration capabilities.[5] Full-document workflows integrated deeply into existing systems may magnify this dependency risk, especially when external models control end-to-end processing. Section-level assistance embedded in user tools (e.g., editors) can localize risk but offers less automation.[5]

The source further highlights **budget and resource uncertainty** when adopting Intelligent Document Processing, linked to factors such as evolving technology, specialized talent needs, and unclear data availability.[5] Comprehensive, full-document AI systems generally require higher upfront investment and clearer scoping than incremental, section-focused helpers. It concludes that successful adoption demands addressing operational and UX challenges, hinting that workflows should balance automation scope (document vs section), user experience, and cost, while partnering with providers that can tailor solutions to specific business objectives.[5]

-----

-----

### Source [65]: https://www.foxit.com/blog/demystifying-ai-in-document-workflows-what-cios-and-it-admins-need-to-know/

Query: What are the primary challenges and trade-offs when designing AI-assisted workflows for editing entire documents versus specific, user-selected text sections?

Answer: This source covers how AI transforms **document workflows** for CIOs and IT admins, emphasizing integration considerations that affect whole-document vs section-level designs. It states that integrating AI into existing workflows brings challenges around **compatibility, data privacy, and user adoption**.[6] Whole-document AI features (e.g., auto-summarization, classification, end-to-end redaction) may require deeper integration with repositories and content management systems, raising compatibility and migration concerns. Section-level, in-editor assistance can be easier to roll out but delivers less automation on repository-level tasks.[6]

The article frames AI’s role in workflows as spanning **content creation, review, and approval**, where operations often run at document granularity (e.g., routing, signing, archiving).[6] Applying AI to entire PDFs or documents aligns with these lifecycle stages but demands strong **privacy controls**, since full content is processed. In contrast, user-selected snippets reduce the data exposed to AI services but limit tasks that depend on global context, such as detecting missing signatures or inconsistent clauses.[6]

It stresses the need to navigate **user adoption** challenges: users must trust AI suggestions and understand how it modifies documents.[6] Full-document, automatic changes can be opaque and harder for users to review; workflows may need explicit diff views, versioning, and approvals. Section-based suggestions, by contrast, are more transparent and reviewable inline, but they can be slower for large-scale editing and may miss global consistency checks.[6]

Overall, the guidance suggests architecting AI document workflows so full-document features are carefully integrated with governance, privacy, and review mechanisms, while section-level tools focus on usability and incremental adoption. Balancing these two modes is critical for achieving both efficiency and trust.[6]

-----

-----

### Source [66]: https://flowtrics.com/automated-document-processing-with-ai/

Query: What are the primary challenges and trade-offs when designing AI-assisted workflows for editing entire documents versus specific, user-selected text sections?

Answer: This source explains **automated document processing with AI** and underscores trade-offs inherent in full-document automation. It notes that AI boosts efficiency and reduces errors in document workflows by automating tasks such as data extraction and classification.[9] These tasks typically process documents as whole entities, enabling straight-through workflows that move files into downstream systems without manual intervention.[9] Such design favors speed and consistency but leaves less room for fine-grained, user-selected adjustments within individual documents.

The article also discusses how AI transforms business workflows by enabling documents to be **routed, validated, and stored** automatically.[9] These capabilities rely on a comprehensive view of each document’s structure and content; partial or section-only analysis would undermine routing accuracy or compliance validation. Therefore, the trade-off is between strong, automated orchestration at document scale and interactive, section-level editing that might require re-running validations for the entire file after any local change.[9]

Furthermore, the source highlights that automation reduces **human error** but must be balanced with appropriate oversight.[9] For full-document workflows, a single configuration mistake or model error can affect entire batches of documents, whereas section-level intervention distributes risk across many smaller, user-reviewed decisions. This implies that robust monitoring, exception handling, and human-in-the-loop checkpoints are especially important when designing AI that operates over complete documents rather than narrowly scoped selections.[9] Overall, the source frames AI document automation as inherently full-document-centric, with design attention needed to integrate localized user edits without breaking automated flows.[9]

-----

</details>

<details>
<summary>What are the established design patterns for creating a 'coding-like' user experience in AI-powered content generation tools that are not specifically for software development?</summary>

### Source [67]: https://www.koruux.com/ai-patterns-for-ui-design/

Query: What are the established design patterns for creating a 'coding-like' user experience in AI-powered content generation tools that are not specifically for software development?

Answer: This article outlines **14 AI UX patterns** that apply broadly to AI-powered tools, including non-programming content generation, and several of them create a "coding-like" experience by enabling iterative, controllable manipulation of outputs.

Relevant patterns include:

- **Refinement & editing controls**: Users can select generated text and apply structured operations such as **regenerate, make longer/shorter, change tone**, etc., via contextual menus or buttons. The UI should support **iterative refinement**, where users repeatedly adjust and re-run transformations on the same content, resembling code edit–run loops.[1]

- **Parameter control / sliders & presets**: Interfaces expose **explicit parameters** (e.g., creativity level, detail, language complexity, tone presets like formal/casual/technical). These are adjusted through sliders, toggles, or dropdowns, mirroring code-like configuration rather than opaque one-shot prompts.[1]

- **Branching exploration of alternatives**: Users can explore **branching paths**—variants, consequences, or reasons—in a hierarchical/tree structure, with navigation to expand/collapse branches. This mimics branching in version control or exploratory coding, where users follow and compare alternative solution paths.[1]

- **Layered explanations and drill-down**: Explanations are presented in **layers**: simple summaries, then deeper technical details on demand. Users can "drill down" into local explanations for a specific output or global explanations of system behavior, analogous to inspecting logs or stack traces in development workflows.[1]

- **Feedback loops & co-training**: Interfaces collect **granular feedback** (ratings, flags, corrections) on outputs, building a continuous improvement loop where users iteratively correct the system, similar to refactoring and test–fix cycles.[1]

- **Context retention & history**: The system maintains a **conversational or interaction history** that users can view and manage without being overwhelmed, enabling stateful refinement akin to an evolving codebase with history.[1]

- **Inline insights & annotations**: The UI can surface **contextual insights** or explanations alongside content, with controls to reveal more detail. This supports a "live IDE" feel, where the environment explains and annotates outputs inline rather than acting as a black box.[1]

-----

-----

### Source [68]: https://www.infoq.com/articles/practical-design-patterns-modern-ai-systems/

Query: What are the established design patterns for creating a 'coding-like' user experience in AI-powered content generation tools that are not specifically for software development?

Answer: InfoQ describes **UX design patterns for user-facing AI applications** that are particularly relevant to creating a coding-like experience around generative content.

- **Editable Output Pattern**: Generated results are **directly editable** by the user rather than treated as final. The article notes that for GenAI there is rarely a single correct answer; the best result depends on user context and preferences, so tools should let users **modify or rewrite** what the AI produced. This supports a **human–AI collaboration** model where users treat AI output as draft code or scaffolding to be refined.[2]

- The pattern emphasizes that allowing edits **reduces the "black box" perception** and increases user control over the final artifact. It likens this to environments like coding IDEs where suggestions (e.g., Copilot) are edited inline, but the same principle applies to any text-based or creative content.[2]

- **Iterative Exploration Pattern**: Interfaces should assume the **first output will often be insufficient**, so they must provide quick ways to **regenerate** or "try again." Buttons for regeneration, plus support for **multiple options at once** (e.g., several text or image variants), encourage a loop of trial, comparison, and selection that parallels coding cycles of run–inspect–modify–rerun.[2]

- The article highlights the need to allow users to **revert to previous generations** or **combine parts from different outputs**, because later prompts can be worse than earlier ones. Preserving and navigating prior results acts like version history or diffs in development tools.[2]

- More broadly, the author lists generic UX best practices such as **suggested follow-ups, clear onboarding examples, uncertainty signaling, and explicit confirmation of critical intents**, all of which structure the interaction so users progressively refine outcomes rather than issuing one-off opaque prompts.[2]

-----

-----

### Source [69]: https://jakobnielsenphd.substack.com/p/prompt-augmentation

Query: What are the established design patterns for creating a 'coding-like' user experience in AI-powered content generation tools that are not specifically for software development?

Answer: Jakob Nielsen’s article on **Prompt Augmentation** defines UX patterns that make prompting feel more structured and tool-like, closer to configuring or composing code than to free-form chatting.

He identifies **six patterns**:

- **Style Galleries**: Predefined styles the user can select (e.g., professional, casual, humorous). Instead of crafting detailed text prompts, users **choose from a catalog of styles**, similar to choosing configuration options or templates in a coding environment.[3]

- **Prompt Rewrite**: The system automatically **rewrites the user’s prompt** into a more effective one while preserving intent. This is analogous to code refactoring tools that optimize or clean up user-written code to better match compiler or runtime expectations.[3]

- **Targeted Prompt Rewrite**: The tool rewrites only specific aspects of the prompt (like tone or level of detail), enabling granular control that resembles editing specific parameters or functions in code rather than changing the entire specification.[3]

- **Related Prompts**: The UI offers **suggested related prompts** to guide exploration. This builds an interaction pattern of incremental, guided refinement akin to exploring related API calls or code snippets suggested by an IDE.[3]

- **Prompt Builders**: Graphical or form-based builders **assemble a full prompt from structured inputs**, such as dropdowns or fields, instead of relying on free text alone. This pattern mirrors code generation wizards or configuration UIs that output code-like specifications under the hood.[3]

- **Parametrization**: Users adjust **explicit parameters** (for example, creativity, length, or emphasis) around a base prompt. Nielsen groups this with **hybrid interfaces** that merge **text-based prompting with GUI controls**, seeking to preserve NLU flexibility while reducing articulation burden. This supports a coding-like experience where users tweak **parameters and options** rather than constantly rewriting entire prompts.[3]

-----

-----

### Source [70]: https://uxplanet.org/7-key-design-patterns-for-ai-interfaces-893ab96988f6

Query: What are the established design patterns for creating a 'coding-like' user experience in AI-powered content generation tools that are not specifically for software development?

Answer: This UX Planet article presents **7 key design patterns for AI interfaces**, several of which describe high-level interaction models that can be used to give non-programming tools a coding-like workflow.

- **Collaborative Canvas**: A shared, often spatial canvas where users and AI **co-create artifacts** (documents, visuals, plans). Users iteratively **manipulate elements, request AI changes, and reorganize content**, echoing how developers work with code on a shared workspace or whiteboard-like IDE.[4]

- **Chatbot & Copilot**: An AI that acts as a **companion or assistant embedded in a workflow**, not just a standalone chat. The pattern stresses continuous assistance, contextual understanding, and **inline suggestions**. For content tools, this leads to an experience similar to coding with a copilot: AI proposes text or structures, and the user accepts, edits, or rejects suggestions in place.[4]

- **Input Form + Prompt-Engineered Forms**: Instead of free text, the system provides **structured forms and fields** that map to sophisticated underlying prompts. This hides complexity while still enabling **precise control**, like supplying arguments to a function call. Users can adjust constraints and parameters without needing to know prompt syntax, yielding a more deterministic, "coding-adjacent" feeling.[4]

- Additional patterns (such as **video & multimodal interfaces**) highlight how AI can be woven into existing media pipelines, but the core takeaway for coding-like UX is that tools should encourage **iterative, dialog-based, and parameterized control** rather than one-shot magic. The patterns advocate for designs where AI is part of a **tight feedback loop** of propose–inspect–revise, analogous to software development workflows.[4]

-----

-----

### Source [71]: https://www.aiuxpatterns.com

Query: What are the established design patterns for creating a 'coding-like' user experience in AI-powered content generation tools that are not specifically for software development?

Answer: This site presents a **pattern library for AI UX**, emphasizing that recurring, effective interaction solutions can be abstracted as **design patterns** to handle the new AI paradigm.

The site explains that AI tools often combine multiple patterns into a **compiled experience**. One example is a typical AI result workflow: after generation, the interface provides **help, feedback options, and additional actions** on the output, rather than a static one-off answer. This compiled pattern illustrates how tools should structure the post-generation phase to support further user control and iteration.[6]

The stated goal is to **abstract core elements and interactions** so designers can reuse best-practice solutions in new contexts. For content-generation tools seeking a coding-like UX, this means selecting and assembling patterns that support:

- **Explicit next actions** after each result (e.g., refine, transform, branch),
- **User feedback mechanisms** that loop back into the interaction,
- **Context-aware assistance** rather than purely linear conversations.[6]

The site frames this as a response to a "paradigm shift" in UX brought by AI. It argues that by **peeling back layers** of concrete products, designers can identify the underlying patterns—like history views, refinement controls, or parameter panels—and then apply them systematically across different domains, echoing how design patterns are used in software engineering.[6]

Although it does not list every individual pattern on the landing explanation, the resource positions AI UX as a field where **pattern-based thinking** is essential to creating predictable, learnable, and controllable AI experiences, akin to design systems and pattern catalogs in development environments.[6]

-----

-----

### Source [72]: https://www.shapeof.ai

Query: What are the established design patterns for creating a 'coding-like' user experience in AI-powered content generation tools that are not specifically for software development?

Answer: Shape of AI is a **pattern library of AI UX patterns** for designing AI products. It focuses on **best practices** and reusable interaction structures rather than ad hoc designs, paralleling software design pattern catalogs.[7]

The library positions AI UX patterns as **abstracted solutions** that can be applied across products. For AI content generation tools, these patterns can be used to:

- Define how users **specify intent** (prompting, forms, mixed-initiative dialogs),
- Structure **feedback and refinement loops** (editing, rating, branching, retrying),
- Present **context and memory** (conversation history, stateful sessions),
- Expose **controls and parameters** that modulate AI behavior.[7]

By cataloging such recurrent solutions, the site encourages teams to **compose experiences** from patterns instead of reinventing each interaction. This mirrors how developers compose software using design patterns, resulting in a more predictable, "tool-like" UX where users can transfer knowledge from one AI product to another.[7]

The resource emphasizes "UX patterns for artificial intelligence design" as a distinct discipline, implying that patterns like copilot-style assistance, parameter panels, example-driven onboarding, iteration controls, and history views can be considered standard components of modern AI-powered tools, including those for non-coding content generation.[7]

-----

</details>

<details>
<summary>How can AI workflow orchestration frameworks like LangGraph be used to dynamically switch between fully automated loops and human-in-the-loop interventions based on task complexity or model confidence?</summary>

### Source [73]: https://docs.langchain.com/oss/python/langgraph/overview

Query: How can AI workflow orchestration frameworks like LangGraph be used to dynamically switch between fully automated loops and human-in-the-loop interventions based on task complexity or model confidence?

Answer: According to the LangGraph overview, the framework is designed for **long‑running, stateful workflows or agents** and gives developers low‑level control over execution flow, state, and interruptions.[6] This control is what enables dynamic switching between fully automated loops and human‑in‑the‑loop.

LangGraph models a workflow as a **graph of nodes** (LLM calls, tools, decision logic, etc.) connected by edges that determine control flow.[6] Because the graph is explicit, developers can insert special nodes that implement **conditional routing** based on task complexity or model signals such as confidence scores or safety/mode flags. A node can examine the current state (for example, a score returned by a previous model call) and choose either to continue through an automated loop or to branch into a human‑review path.[6]

The core abstraction, often referred to as a stateful graph, keeps **persistent state** across steps: intermediate outputs, metadata, and control variables.[6] This lets the system implement **loops**—repeating parts of the graph until some condition over the state is met (for example, “model confidence > threshold” or “max_iterations reached”). As long as the condition is unmet, the workflow stays in an automated refinement loop; when it is met or if a risk flag is set, the graph transitions to another branch, such as a human approval node.[6]

Because LangGraph is intentionally low‑level and does not prescribe a specific agent pattern, teams can encode arbitrary **human‑in‑the‑loop patterns** at particular nodes, including pausing for review, collecting human feedback, and then resuming automated execution with an updated state.[6] This composability is what allows systems to dynamically adapt between full automation and human intervention based on complexity or confidence criteria encoded directly in the graph logic.

-----

-----

### Source [74]: https://www.langchain.com/langgraph

Query: How can AI workflow orchestration frameworks like LangGraph be used to dynamically switch between fully automated loops and human-in-the-loop interventions based on task complexity or model confidence?

Answer: The LangChain LangGraph product page emphasizes that LangGraph helps teams **guide, moderate, and control** agents with *human‑in‑the‑loop* capabilities.[3] It explicitly states that developers can "add human‑in‑the‑loop checks to steer and approve agent actions," which directly supports workflows that switch between automated execution and human intervention.[3]

LangGraph is described as enabling **expressive, customizable agent workflows**, including single, multi‑agent, and hierarchical control flows within one framework.[3] This flexibility is crucial for dynamic switching: developers can define branches in the workflow where certain classes of tasks (for example, complex, ambiguous, or high‑risk ones) are routed to humans, while simpler or high‑confidence tasks continue through automated loops.

Moderation and quality‑control nodes can be placed strategically in the graph so that the agent’s outputs are checked before taking consequential actions.[3] Based on the results of these checks—such as safety assessments, rule‑based evaluations, or model‑generated confidence indicators—the workflow can either stay in a **fully automated loop** (retrying, refining, or escalating to a more capable model) or **pause for human approval**.

Because the page stresses that LangGraph is the foundation for scalable AI workloads ranging from conversational agents to complex task automation, it implies that these human‑in‑the‑loop checks are meant to be part of **production orchestration**, not only development‑time debugging.[3] In practice, this means teams can encode policies like "if task category is high‑risk or confidence below threshold, send to human" directly into the graph, making the system dynamically toggle between automation and human review at runtime.

-----

-----

### Source [75]: https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025

Query: How can AI workflow orchestration frameworks like LangGraph be used to dynamically switch between fully automated loops and human-in-the-loop interventions based on task complexity or model confidence?

Answer: The multi‑agent orchestration guide explains that LangGraph organizes workflows as a **directed acyclic graph (DAG)** where nodes are agents, tools, or decision points and edges define data and control flow.[2] A centralized **StateGraph** stores intermediate results and metadata, enabling conditional branching and loops that depend on the evolving state.[2]

For dynamic switching, the guide highlights **conditional edges** that evaluate the current state to choose the next path.[2] Conditions can be simple checks or more complex evaluations such as analyzing **agent confidence scores** or the status of external systems.[2] This allows developers to encode logic like: if confidence ≥ threshold, remain in an automated processing loop; if confidence < threshold or an anomaly flag is set, route to a human‑review node.

The framework supports **loop constructs** via recursive graph patterns with explicit termination criteria.[2] A loop can repeatedly call an agent or group of agents (for example, refinement cycles, retrieval‑and‑reasoning loops) until a condition is met, such as achieving sufficient confidence, reaching a maximum iteration count, or obtaining human approval.[2]

The guide describes explicit **human‑in‑the‑loop patterns**: workflows can pause at specific nodes, present the current state to an operator for review, and then resume based on the operator’s input.[2] It also mentions **interrupt mechanisms** that let operators pause execution at any node, inspect or adjust state, and then continue processing.[2] Human‑in‑the‑loop checkpoints are recommended for high‑stakes contexts: when monitoring detects anomalies, execution can be stopped and passed to a human, turning a fully automated pipeline into a human‑supervised one dynamically.[2]

Together, conditional edges, loop constructs, and human‑in‑the‑loop checkpoints provide a concrete mechanism for switching between autonomous loops and human intervention as a function of task complexity, confidence metrics, or anomaly detection encoded in the state.[2]

-----

-----

### Source [76]: https://www.getzep.com/ai-agents/langchain-agents-langgraph/

Query: How can AI workflow orchestration frameworks like LangGraph be used to dynamically switch between fully automated loops and human-in-the-loop interventions based on task complexity or model confidence?

Answer: The Zep guide on LangChain agents and LangGraph states that LangGraph uses a **graph‑based approach** to orchestration, supporting **branching, looping, and conditional logic**.[4] Each agent or function is a node, and edges define the flow between them, enabling workflows that can "branch based on specific conditions or repeat certain steps as needed."[4]

It specifically notes that LangGraph supports **human‑in‑the‑loop workflows** as a key feature alongside cycles and persistent state management.[4] This means that at design time, developers can mark certain nodes as human checkpoints where execution pauses and awaits a human decision before continuing.

The document’s example of loops and conditionals shows how an agent might process input, check some property (such as input length), and then either continue processing or end the task based on that condition.[4] By analogy, the same mechanism can be applied to **model confidence or task complexity**: a node can compute or receive a confidence value from the model and then use conditional edges to either:

- stay in an automated refinement loop (for example, run another tool, re‑query a model, or escalate to a more capable model), or
- exit the loop and direct control to a human review node when the condition (e.g., low confidence or high risk) is met.[4]

Because LangGraph includes **persistent state management**, the full history of automated steps and intermediate artifacts is available to the human when they are brought into the loop.[4] This allows humans to make informed decisions and then update the state so that, when control returns to the automated graph, subsequent nodes can take into account both prior model outputs and human feedback, effectively blending automated loops with selective human oversight.

-----

-----

### Source [77]: https://www.indapoint.com/blog/langgraph-revolutionizing-ai-workflows-with-graph-based-orchestration.html

Query: How can AI workflow orchestration frameworks like LangGraph be used to dynamically switch between fully automated loops and human-in-the-loop interventions based on task complexity or model confidence?

Answer: The Indapoint article describes LangGraph as introducing **graph‑based, stateful workflows** that replace traditional linear chains.[1] It emphasizes features such as **stateful orchestration**, dynamic graph‑based workflows with cycles and branching, **advanced state management**, explicit **human‑in‑the‑loop integration**, and multi‑agent support.[1]

Stateful orchestration allows the system to maintain context across multiple interactions and **revisit previous decisions**, which is fundamental for implementing **automated loops** that refine outputs or iterate through multi‑step reasoning processes.[1] Dynamic workflows with cycles and branching support decision paths that can adapt on‑the‑fly as conditions change.[1]

The human‑in‑the‑loop integration feature is highlighted as a mechanism for adding **human oversight at key decision points**.[1] At these points, the workflow can pause, surface the current state and AI outputs to a human, and proceed based on their input, which may approve, modify, or override the automated decision.[1] This pattern enables workflows where **routine or low‑risk tasks run fully automatically**, while more complex or ambiguous tasks are escalated for human judgment.

The article notes that LangGraph is well‑suited for complex applications such as research assistants, autonomous decision‑making systems, and customer service flows with escalation paths.[1] In such systems, branching logic can route standard, high‑confidence cases through automated loops and direct edge cases or high‑impact decisions into **escalation paths** that include human agents.[1] Multi‑agent support further allows specialized agents (for analysis, retrieval, summarization, etc.) to collaborate automatically, with humans stepping in at specific orchestration nodes when conditions encoded in the graph—such as task complexity or detected uncertainty—trigger human review.[1]

-----

</details>

<details>
<summary>What are the emerging best practices for orchestrating multiple specialized AI agents, such as a research agent and a writing agent, into a single, cohesive workflow?</summary>

### Source [78]: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns

Query: What are the emerging best practices for orchestrating multiple specialized AI agents, such as a research agent and a writing agent, into a single, cohesive workflow?

Answer: Microsoft’s AI Agent Orchestration Patterns describe **emerging best practices** for coordinating multiple specialized agents (for example, a research agent and a writing agent) into cohesive workflows.[5]

They recommend selecting from proven **coordination patterns** based on the task:
- **Sequential orchestration**: one agent (e.g., research) completes its task and passes structured outputs to the next (e.g., writing) via well-defined interfaces and schemas.[5]
- **Concurrent orchestration**: multiple agents (such as several research agents using different data sources) work in parallel, with an orchestrator aggregating and reconciling results.[5]
- **Group chat / collaborative patterns**: agents interact in a shared context (for example, research, writing, and fact-checking agents “discussing” content), suited to brainstorming, decision-making, and iterative refinement.[5]
- **Handoff patterns**: explicit transfer of control between agents or from agent to human, useful where roles are clearly separated, such as researcher → writer → compliance reviewer.[5]

Microsoft emphasizes **optimization through specialization**, where each agent can use distinct models, tools, knowledge, and compute aligned to its role, instead of relying on a single generalist agent.[5]

They highlight scenarios particularly relevant to research–writing workflows:
- **Collaborative scenarios**: creative brainstorming, multidisciplinary problems, and decision processes where agents build on each other’s contributions through iterative dialogue.[5]
- **Validation and quality-control scenarios**: separate reviewer or editor agents that perform structured review, compliance checks, and iterative improvement of content created by a primary writing agent.[5]

Across patterns, the guidance stresses:
- Making **coordination logic explicit** in an orchestrator (or workflow engine) rather than hidden in individual agents.
- Using **structured, inspectable messages** between agents to support auditability and debugging.
- Treating orchestration patterns as complements to traditional cloud patterns, integrating agents into broader systems with reliability, observability, and governance.[5]

-----

-----

### Source [79]: https://support.talkdesk.com/hc/en-us/articles/39096730105115-AI-Agent-Platform-Best-Practices

Query: What are the emerging best practices for orchestrating multiple specialized AI agents, such as a research agent and a writing agent, into a single, cohesive workflow?

Answer: Talkdesk’s AI Agent Platform best practices document provides detailed guidance for **multi–AI-agent design and orchestration**.[2]

Core instruction design principles:
- **Write clear, specific instructions without conflicts** so each agent (e.g., research vs. writing) has unambiguous responsibilities.[2]
- **Provide relevant examples** in the instructions to steer agent behavior toward desired outputs.[2]
- **Split complex tasks into simpler subtasks**, mapping each to a specialized agent; this directly supports workflows like research → outline → draft → review.[2]

For structuring orchestration:
- Use **Skills** for structured tasks and let Skills manage variable assignment, ensuring consistent context management across agents.[2]
- Use **variables to provide context**, but **only share relevant information** from variables with each agent, minimizing distraction and risk of confusion.[2]

Operational best practices:
- **Test changes systematically** when updating agent instructions or orchestration logic, rather than making large, unvalidated changes.[2]
- **Monitor and iterate based on real usage**, using logs and performance signals to refine agent boundaries, prompts, and data sharing.[2]

For multi-agent design specifically, the document emphasizes:
- Designing agents around **narrow, well-defined capabilities** instead of large, monolithic agents.[2]
- Creating a **clear handoff contract** between agents (for instance, rigid schemas for what the research agent returns and what the writing agent expects).[2]
- Keeping orchestration logic in the **platform’s orchestration layer**, not inside individual agents, to maintain transparency and easier evolution.[2]

Applied to research–writing workflows, these practices mean:
- Defining separate Skills/agents for research, synthesis, drafting, and review.
- Passing structured, minimal, and relevant context between those agents.
- Continuously tightening instructions and examples as monitoring reveals gaps.[2]

-----

-----

### Source [80]: https://www.uipath.com/blog/ai/agent-builder-best-practices

Query: What are the emerging best practices for orchestrating multiple specialized AI agents, such as a research agent and a writing agent, into a single, cohesive workflow?

Answer: UiPath’s "10 best practices for building reliable AI agents in 2025" offers platform-agnostic guidance for orchestrating multiple specialized agents into robust workflows.[3]

For system design:
- **Modularize into multiple specialized agents**: complex workflows should combine several narrow agents and robots instead of a single “do-everything” agent, which simplifies scaling, debugging, and reuse.[3]
- **Align agent goals and measurable outcomes**: define clear objectives, metrics, and success criteria for each agent (e.g., research accuracy vs. writing clarity).[3]

For context and reasoning:
- **Index enterprise context** and choose appropriate search strategies (semantic, structured, or DeepRAG) so research agents retrieve high-quality evidence, which then feeds reliably into writing agents.[3]
- Start with a system prompt that clearly defines each agent’s **role, goal, context, success metrics, guardrails, and constraints**.[3]
- Use **structured, multi-step reasoning** and explicit task decomposition; specify how agents break large jobs into smaller steps and what outputs they must produce.[3]

For orchestration and governance:
- **Run agents via UiPath Orchestrator or Maestro** to get lifecycle management, auditing, and governance for multi-agent workflows.[3]
- Apply an **AI trust layer** for permissions, PII redaction, audit logs, throttling, and usage controls across all agents.[3]
- Maintain **human-in-the-loop** escalation for high-risk decisions and use these interactions to update agent memory and design.[3]
- Use **guardrails** and deterministic confirmations before irreversible actions (e.g., publishing content).[3]
- **Version everything**—prompts, tools, datasets, and evaluations—so changes to one agent in a chain (like research) can be tracked against downstream impacts (like writing quality).[3]

They also recommend optimizing **model selection and token usage**, batching operations, and using tracing and evaluation tools to iteratively improve the combined agent workflow.[3]

-----

-----

### Source [81]: https://kanerika.com/blogs/ai-agent-orchestration/

Query: What are the emerging best practices for orchestrating multiple specialized AI agents, such as a research agent and a writing agent, into a single, cohesive workflow?

Answer: Kanerika’s 2025 overview of AI Agent Orchestration outlines how to coordinate multiple specialized agents into a unified, scalable workflow.[1]

They define **multi-agent orchestration** as the systematic coordination of multiple AI agents, managing **communication, task allocation, shared memory, conflict resolution, and workflow sequencing** so all agents contribute effectively to common objectives.[1]

Key orchestration lifecycle practices:
- An **orchestrator agent** interprets incoming requests, breaks them into smaller components, and determines the scope of work.[1]
- The orchestrator **selects appropriate specialized agents** for each component, considering capabilities, availability, workload, and performance history.[1]
- It **divides work with clear instructions and priorities**, sending each agent the context it needs plus information about how its output fits into the overall workflow.[1]
- Agents work while **maintaining communication** with each other and the orchestrator, sharing progress updates, intermediate results, and issues in real time.[1]
- The orchestrator **combines individual outputs** into a comprehensive response, ensuring consistency, removing duplication, formatting results, and performing quality checks before delivery.[1]
- A **feedback loop** analyzes workflow performance, updates metrics, reinforces successful patterns, and addresses problematic areas, improving orchestration efficiency over time.[1]

Benefits relevant to research–writing setups:
- **Specialized agents focus on narrow tasks** where they excel (e.g., research vs. writing), minimizing mistakes often seen in general-purpose systems.[1]
- Agents can **validate outputs from each other**, creating multiple checkpoints (for example, a fact-checking agent verifying the writing agent’s claims) before content reaches end users.[1]
- Orchestration supports **scalability and fault tolerance**, automatically redistributing tasks when volume grows or individual agents fail, avoiding single points of failure.[1]

Kanerika also distinguishes **generative AI orchestration**, where multiple generative models or tools are coordinated to create comprehensive content while maintaining consistency and quality.[1]

-----

-----

### Source [82]: https://www.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions/2026/ai-agent-orchestration.html

Query: What are the emerging best practices for orchestrating multiple specialized AI agents, such as a research agent and a writing agent, into a single, cohesive workflow?

Answer: Deloitte’s report on AI agent orchestration focuses on **enterprise-scale, multiagent systems** and the practices needed to unlock their value.[6]

They define agent orchestration as the **effective coordination of role-specific agents** across domains, enabling multiagent systems to interpret requests, design workflows, delegate and coordinate tasks, and continuously validate and enhance outcomes.[6]

For cohesive workflows (like research + writing), Deloitte highlights:
- The importance of **thoughtful orchestrators** that interpret high-level user intent, translate it into a multi-step workflow, and allocate responsibilities to the right agents.[6]
- Continuous **validation and enhancement** of outputs: downstream agents (e.g., reviewers, compliance checkers) improve quality rather than acting only as passive recipients.[6]

They identify key **infrastructure and platform practices**:
- Use of **agent registries** for trusted discovery of agents, capability descriptions, and workload balancing—notably important when many specialized agents coexist.[6]
- **Asynchronous messaging** and support for **chained and nested workflows** to coordinate agents that may run at different speeds or depend on each other’s outputs.[6]
- Designing for **high throughput and low latency**, so orchestration remains responsive even as more agents are added.[6]

On risk and governance:
- Implement **authentication, secure messaging, and access control** between agents to mitigate security risks.[6]
- Capture **inter-agent messages and explanations** for auditability and error traceability, providing transparency into how research findings translated into written outputs.[6]

Deloitte positions orchestration as central to building **intelligent workflows** in which multiple specialized agents interact seamlessly, augmenting or replacing traditional process automation with flexible, adaptive, multi-step reasoning pipelines.[6]

-----

-----

### Source [83]: https://blog.n8n.io/ai-agent-orchestration-frameworks/

Query: What are the emerging best practices for orchestrating multiple specialized AI agents, such as a research agent and a writing agent, into a single, cohesive workflow?

Answer: n8n’s article on AI Agent Orchestration Frameworks describes patterns and tooling for coordinating multiple specialized agents into end-to-end workflows.[4]

They advocate **splitting large agents into smaller, specialized ones**, where each agent is an expert in a domain (for example, data analysis, research, drafting, scheduling) and agents coordinate via a framework rather than acting monolithically.[4]

The article identifies **essential orchestration components**:
- **State management**: persistent memory that survives across agent interactions so that, when a research agent finishes and a writing agent starts, relevant context is transferred seamlessly without manual copying.[4]
- **Communication protocols**: standardized ways for agents to talk to each other, including structured handoffs, shared chat threads, or event-driven messages.[4]

Applied to research–writing workflows, these components imply:
- Maintaining a **shared state store** (for example, a knowledge object containing research findings and decisions) that both research and writing agents can read and update.[4]
- Using **structured messages** for handoffs, so outputs from research (claims, citations, summaries) are machine-readable for the writing agent.

The article also emphasizes the role of **AI agent orchestration frameworks** (such as graph- or workflow-based systems) that:
- Manage **task delegation** to the right agent at the right time based on the current state.[4]
- Handle **error paths and retries**, ensuring that failures in one agent trigger compensating steps or reruns where appropriate.
- Allow designers to **visually model agent workflows**, making multi-agent pipelines more understandable and maintainable.

Overall, n8n frames emerging best practice as using dedicated orchestration frameworks with strong state and communication abstractions, instead of ad hoc scripting between individual agents.[4]

-----

-----

### Source [84]: https://www.moveworks.com/us/en/resources/blog/improve-workflow-efficiency-with-ai-agent-orchestration

Query: What are the emerging best practices for orchestrating multiple specialized AI agents, such as a research agent and a writing agent, into a single, cohesive workflow?

Answer: Moveworks’ discussion of AI Agent Orchestration for enterprise workflows outlines how orchestrated agents improve efficiency, reliability, and user experience across complex processes.[9]

They describe orchestration as coordinating multiple agents to handle tasks such as IT, HR, and facilities support, which often map closely to research, triage, and response-generation activities.[9]

Relevant emerging practices include:
- Using orchestration to **route work to the most appropriate specialized agent**, based on task type and context, rather than relying on a single generic assistant.[9]
- Designing agents around **clear roles** (for example, a diagnosis/troubleshooting agent vs. a communication/response agent) that together form a cohesive workflow.[9]
- Implementing **end-to-end workflows** in which agents handle different steps but the overall interaction remains seamless for the user.[9]

Moveworks emphasizes operational aspects:
- **Reducing manual intervention** by ensuring agents can autonomously hand off tasks to each other as needed, while preserving history and context.[9]
- **Improving response times and accuracy** by letting specialized agents do what they are best at and orchestrating them through a central platform.[9]

While the examples are focused on enterprise support, the same orchestration principles apply to research–writing pipelines: one agent gathers and interprets information, another generates tailored responses or content, and the orchestration layer manages routing, state, and user-facing continuity.[9]

-----

-----

### Source [85]: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

Query: What are the emerging best practices for orchestrating multiple specialized AI agents, such as a research agent and a writing agent, into a single, cohesive workflow?

Answer: OpenAI’s "A practical guide to building agents" (business guide PDF) presents frameworks and patterns for designing agent logic and orchestration.[7]

The guide outlines **patterns for orchestrating multiple agents** to accomplish complex tasks, rather than relying on a single agent.[7]

Key best practices include:
- **Identifying promising multi-step use cases** where decomposition into stages (e.g., research, planning, drafting, review) is natural, and designing multiple agents accordingly.[7]
- Making **agent roles explicit**, with clear definitions of responsibilities, tools, and success criteria for each agent in the workflow.[7]
- Implementing **orchestration logic outside the agents**—in a controller or workflow layer that sequences calls to agents, handles branching, and manages shared state.[7]
- Using **structured inputs and outputs** so agents can reliably consume each other’s results; for example, a research agent returning JSON summaries and citations that the writing agent transforms into narrative text.[7]

The guide stresses **reliability and safety** through:
- **Tool usage and constrained actions**: using tools/APIs to perform deterministic operations (like data retrieval) that feed agents, rather than allowing unconstrained free-form actions.[7]
- **Evaluation and monitoring**: instrumenting multi-agent workflows with telemetry and evaluation datasets to measure quality and catch regressions as agents or prompts change.[7]
- **Human oversight** on higher-risk steps, which can be placed at specific points in the orchestrated chain (for example, before final publication).[7]

For multi-agent setups like research + writing, the guide recommends **iterative refinement** patterns, where a reviewer or critic agent evaluates drafts and a writer agent revises based on that feedback, under the control of an orchestrating process that loops until quality thresholds are met.[7]

-----

</details>

<details>
<summary>Beyond Andrej Karpathy's "AI generation/human validation loop", what other theoretical frameworks or models exist for designing effective human-AI collaboration in practical applications?</summary>

### Source [86]: https://arxiv.org/html/2407.19098v2

Query: Beyond Andrej Karpathy's "AI generation/human validation loop", what other theoretical frameworks or models exist for designing effective human-AI collaboration in practical applications?

Answer: This paper proposes a **structured, domain-agnostic framework for Human-AI Collaboration (HAIC)** that goes beyond simple generation/validation loops by treating collaboration as a multi-factor system.[1]

The framework organizes evaluation and design of HAIC along three **primary factors** with specialized **subfactors**:[1]
- **Interaction factor**: focuses on communication and feedback channels between humans and AI.
  - Subfactors: **Communication methods**, **feedback mechanisms**, **adaptability**, **trust and safety**.[1]
  - Emphasizes bidirectional information flow, mutual adaptation, and mechanisms to maintain calibrated trust.

- **Task / allocation factor** (implied through modes of collaboration): defines how work is split and coordinated.
  - Identifies **three modes of collaboration** based on task allocation: **AI‑centric**, **human‑centric**, and **symbiotic**.[1]
  - In AI‑centric mode, AI leads decision-making with limited human oversight.
  - In human‑centric mode, AI acts mainly as a tool or recommender while humans retain primary control.
  - In **symbiotic mode**, humans and AI maintain a **balanced partnership**, with two‑way interaction, shared decision‑making, continuous feedback exchange, and joint pursuit of collective goals.[1]

- **Goals / performance factor**: emphasizes setting **measurable goals**, clarifying tasks, and defining **reporting metrics** to evaluate outcomes across domains.[1]

A key conceptual model embedded in this framework is **Learning to Defer (L2D)** as an explicit collaboration paradigm:[1]
- The AI system is trained to **recognize its own limitations** and **defer decisions to humans** when appropriate.
- Recent work extends L2D to **cost‑sensitive** and **multi‑expert** settings with workload constraints, highlighting **dynamic task allocation** based on expertise and availability.[1]
- Another line trains models specifically to **complement multiple human experts**, aligning the AI’s competence profile with human strengths and weaknesses.[1]

Overall, this framework encourages designers to explicitly choose a **collaboration mode**, specify **interaction subfactors**, and integrate **deferral and complementarity mechanisms**, rather than treating AI as a static tool.[1]

-----

-----

### Source [87]: https://thedecisionlab.com/reference-guide/computer-science/human-ai-collaboration

Query: Beyond Andrej Karpathy's "AI generation/human validation loop", what other theoretical frameworks or models exist for designing effective human-AI collaboration in practical applications?

Answer: This reference describes a **conceptual framework for Human-AI collaboration** built around three tightly coupled elements: **shared goals**, **interaction**, and **task allocation**.[3]

Core components:[3]
- **Shared goals**: Human and AI systems should be aligned on overarching objectives; collaboration is framed as a partnership to achieve these goals.

- **Interaction**: Effective HAIC depends on **sound communication and feedback** between humans and AI agents.
  - How well humans understand AI outputs depends on how well the AI can model **human intentions, skills, and limits**.[3]
  - Emphasizes iterative feedback loops, explanation quality, and transparency so humans can interpret and contest AI suggestions.

- **Task allocation**: Tasks are delegated between humans and AI based on their **respective strengths**.
  - The framework highlights **dynamic task allocation**, with **real-time changes in duties** depending on context, difficulty, or risk.[3]
  - Rather than a static pipeline, roles can shift as the system learns about user capabilities or as conditions change.

Within this structure, the article points to **key mechanisms** studied in HAIC literature as design patterns:[3]
- **AI delegation**: The AI **passes tasks to humans** in contexts where humans are likely to perform better or where human involvement improves satisfaction.[3]
  - Example pattern: a chatbot automatically handles routine inquiries but **routes complex or emotionally charged cases to human agents**.

The overall model encourages practitioners to:[3]
- Define clear, **shared objectives** for human and AI.
- Design robust **interaction channels** that support understanding and feedback.
- Implement **adaptive delegation policies** (human-to-AI and AI-to-human) instead of rigid workflows.

This yields a broader collaboration framework than simple validation loops by explicitly treating collaboration as **goal alignment + communication + dynamic division of labor**.[3]

-----

-----

### Source [88]: https://clanx.ai/glossary/human-ai-colaboration

Query: Beyond Andrej Karpathy's "AI generation/human validation loop", what other theoretical frameworks or models exist for designing effective human-AI collaboration in practical applications?

Answer: This source outlines **Human-AI collaboration theory** as a guiding framework for designing practical collaboration, emphasizing **synergy between human intuition and AI analytics**.[2]

Key theoretical ideas:[2]
- **Human-AI collaboration theory** focuses on principles and frameworks for **balancing human and machine capabilities**.
  - It stresses combining **human intuition, judgment, and contextual understanding** with **AI-driven analytics, pattern recognition, and scale**.
  - The goal is a **cohesive, synergistic partnership**, not mere tool use.

- A central concept is that collaboration design must consider **role definition, decision authority, and interaction protocols**, so that humans neither over‑rely on nor ignore AI systems.[2]

The article highlights **Collaborative Machine Learning** as a concrete framework:[2]
- Humans and AI **work together iteratively** to refine and improve models.
- Human experts provide **feedback, labels, corrections, and domain constraints**, while AI systems propose patterns, predictions, or candidate models.[2]
- This creates an **iterative loop** where the model evolves to better match **real-world scenarios and domain expectations**.

In practice, this theory suggests design patterns beyond generation/validation:[2]
- Use AI to **surface analytics and hypotheses**, then let humans **steer model objectives and constraints**.
- Maintain an ongoing **co‑evolution** of models and workflows, where human feedback is a first‑class input to model updates, not just a one-off validation step.

Overall, the theory conceptualizes effective collaboration as **continuous, bidirectional learning and adjustment** between humans and AI, with explicit attention to how their capabilities are combined across time and tasks.[2]

-----

-----

### Source [89]: https://digitalcommons.harrisburgu.edu/cgi/viewcontent.cgi?article=1028&context=dandt

Query: Beyond Andrej Karpathy's "AI generation/human validation loop", what other theoretical frameworks or models exist for designing effective human-AI collaboration in practical applications?

Answer: This work reviews **theoretical frameworks underpinning human-AI collaboration in project management**, drawing on broader organizational and socio-technical theories.[4]

Key frameworks used to analyze and design Human-AI collaboration include:[4]

- **Socio-Technical Systems Theory (STS)**:
  - STS examines the **interdependence of social and technical elements** in organizations.[4]
  - For human-AI collaboration, it implies AI systems must be designed and integrated considering **organizational structures, workflows, roles, and human needs**, rather than as isolated tools.[4]
  - STS emphasizes **joint optimization**: both the social (people, culture, processes) and technical (AI systems, tools) subsystems should be co‑designed for overall performance and well-being.[4]
  - The paper connects STS to the importance of **trust and transparency**, noting that transparent, user‑friendly AI can foster more effective collaboration.[4]

- **Trust as a cross-cutting construct**:
  - The work highlights **trust and transparency** as central to successful human-AI collaboration in project settings.[4]
  - It suggests that building trust involves not only technical reliability but also **explainability, usability, and alignment with project team norms**.[4]

Design implications from these frameworks include:[4]
- Treat human-AI collaboration as a **system design problem** across people, processes, and technology.
- Explicitly consider **organizational impacts, role changes, and communication structures** when introducing AI.
- Use STS principles to guide **task allocation, interfaces, and training**, ensuring that AI augments rather than disrupts human teamwork.

This positions human-AI collaboration frameworks within established organizational theory, offering a lens that goes beyond interaction loops to **organization-level co‑design and governance**.[4]

-----

-----

### Source [90]: https://thesai.org/Downloads/Volume16No7/Paper_1-Enhancing_Trust_in_Human_AI_Collaboration.pdf

Query: Beyond Andrej Karpathy's "AI generation/human validation loop", what other theoretical frameworks or models exist for designing effective human-AI collaboration in practical applications?

Answer: This conceptual review focuses on **trust in human-AI collaboration**, synthesizing multiple **theoretical frameworks** to model trust dynamics in operator–AI teams.[5]

It identifies several core theories relevant for designing effective collaboration:

- **Social Exchange Theory (SET)**:
  - Views relationships as exchanges where individuals seek to **maximize benefits and minimize costs**.[5]
  - Applied to human-AI teams, operators are more willing to collaborate when perceived benefits (e.g., improved performance) outweigh costs (e.g., effort, risk, loss of control).[5]
  - Factors like **perceived fairness, reliability, and competence of the AI** influence this cost–benefit assessment and thus trust.[5]

- **Cognitive Load Theory (CLT)**:
  - Focuses on **limitations of human cognitive resources**.[5]
  - In HAIC, well-designed AI support should **reduce unnecessary cognitive load**, enabling operators to better evaluate AI outputs and maintain appropriate trust.[5]
  - Managing cognitive load (e.g., via interface design, information filtering) is seen as **crucial for effective collaboration and trust calibration**.[5]

- **Human-Computer Interaction (HCI) and Technology Acceptance Models**:
  - The paper references diverse **HCI and technology acceptance frameworks** as foundations for understanding **user attitudes, perceived usefulness, ease of use, and adoption** of AI systems.[5]
  - These models help explain how **system design, feedback, and transparency** shape trust and reliance behaviors in human-AI teams.[5]

The review concludes that trust in HAIC is a **multi-factor phenomenon** influenced by:
- Perceived utility and reliability of AI.
- Cognitive feasibility and workload.
- Interface and interaction design.
- User psychology and team dynamics.[5]

It proposes building a **composite framework for enhancing trust** that integrates these theories, guiding design of AI systems that support **dynamic trust calibration, appropriate reliance, and effective teaming processes** beyond simplistic validation loops.[5]

-----

-----

### Source [91]: https://www.authorea.com/users/828898/articles/1223041-human-ai-collaboration-exploring-synergies-and-future-directions

Query: Beyond Andrej Karpathy's "AI generation/human validation loop", what other theoretical frameworks or models exist for designing effective human-AI collaboration in practical applications?

Answer: This article surveys **Human-AI collaboration** and presents a high-level **theoretical framework** for future research and system design.[6]

In the section on **Theoretical Framework – Theories of Human-AI Interaction**, it discusses how existing theories of human–computer interaction and organizational behavior can be adapted to human-AI settings.[6]

Central ideas include:[6]
- Viewing Human-AI collaboration as **teaming** between heterogeneous agents, where concepts from **team cognition, shared mental models, and joint intention theory** can be applied.
- Emphasizing **complementarity of capabilities**: AI contributes computation, pattern detection, and scalability, while humans provide context sensitivity, ethics, and creativity.
- Framing collaboration as an ongoing process of **coordination, communication, and mutual adaptation**, not a one-off handoff.

The framework highlights design elements such as:[6]
- **Role clarity**: specifying when AI is a decision-maker, recommender, or assistant.
- **Explainability and transparency**: enabling humans to understand AI reasoning sufficiently to coordinate and override when needed.
- **Feedback and learning loops**: allowing both humans and AI components to adapt over time based on performance and interaction outcomes.

The article also outlines **future directions** such as more sophisticated models of **shared control**, **interactive learning**, and **ethical governance** to ensure that collaborative systems remain aligned with human values and societal goals.[6]

Overall, it positions Human-AI collaboration within a broader **multi-theory framework** that integrates team science, HCI, and AI, and encourages system designers to treat collaboration as **joint problem solving and co-adaptation** rather than simple validation of AI outputs.[6]

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>[00:00] (Audience applauding as speaker walks on stage)</summary>

[00:00] (Audience applauding as speaker walks on stage)

[00:01] Please welcome, former Director of AI, Tesla, Andrej Karpathy.

[00:10] (Audience applauding loudly) Hello.

[00:18] (Speaker smiles and waves to the audience)

[00:19] Wow, a lot of people here. Hello. Um, okay, yeah, so I'm excited to be here today to talk to you about software in the era of AI. [00:30] (Slide with title: "Software in the era of AI", and a picture of Golden Gate Bridge with lavender field) And I'm told that many of you are students, like bachelors, masters, PhD, and so on, and you're about to enter the industry. And I think it's actually like an extremely unique and very interesting time to enter the industry right now. And I think fundamentally, the reason for that is that, um, software is changing.

[00:48] (Slide changes to white background with "Software is changing. (again)" in black text) Again. And I say again because I actually gave this talk already. Um, but the problem is that software keeps changing, so I actually have a lot of material to create new talks. And I think it's changing quite fundamentally. I think roughly speaking, software has not changed much on such a fundamental level for 70 years, and then it's changed, I think, about twice quite rapidly in the last few years. [01:00] And so there's just a huge amount of work to do, a huge amount of software to write and rewrite. So let's take a look at maybe the realm of software.

[01:10] (Slide with a dark blue background titled "Map of GitHub", showing a scattered graph of blue nodes representing software repositories) So if we kind of think of this as like the map of software, this is a really cool tool called Map of GitHub. Um, this is kind of like all the software that's written. These are instructions to the computer for carrying out tasks in the digital space. So if you zoom in here, these are all different kinds of repositories, [01:30] (Zoomed-in section highlights a cluster of repositories, showing interconnected nodes and text labels) and this is all the code that has been written. And a few years ago, I kind of observed that, um, software was kind of changing, and there was kind of like a new type of software around, and I called this software 2.0 at the time. [01:38] (Slide titled "Software 2.0" with a blog post header by Andrej Karpathy, showing "Software 1.0 = code" next to an image of computer code and "Software 2.0 = weights" next to a diagram of a neural network) And the idea here was that software 1.0 is the code you write for the computer. Software 2.0 are basically neural networks, and in particular, the weights of a neural network. And you're not writing this code directly. You are most, you are more kind of like tuning the data sets and then you're running an optimizer to create the parameters of this neural net. [02:00] And I think like at the time, neural nets were kind of seen as like just a different kind of classifier, like a decision tree or something like that, and so I think, uh, it was kind of like, um, I, I think this framing was a lot more appropriate. [02:11] (Slide showing "Map of GitHub" (Software 1.0) on the left, and "HuggingFace Model Atlas" (Software 2.0) on the right, which is a dense, colorful, interconnected graph of neural network weights) And now actually what we have is kind of like an equivalent of GitHub in the realm of software 2.0. And I think, uh, the HuggingFace, uh, is basically equivalent of GitHub in software 2.0. And there's also model atlas and you can visualize all the code written there. In case you're curious, by the way, the giant circle, the point in the middle, uh, these are the parameters of Flux, the image generator. [02:40] And so anytime someone tunes a Laura on top of a Flux model, you basically create a Git commit, uh, in this space and, uh, you create a different kind of, uh, image generator. So basically what we have is software 1.0 is the computer code that programs a computer. [02:45] (Slide with two columns: "Software 1.0" (computer code) on the left with a flowchart "programs -> computer" and an old photo of a person using a computer; "Software 2.0" (weights) on the right with a flowchart "programs -> neural net" and a diagram of AlexNet) Software 2.0 are the weights, which program neural networks. Uh, and here's an example of AlexNet image recognizer neural network. Now, so far, all of the neural networks that we've been familiar with until recently, were kind of like fixed function computers, [03:00] image to categories or something like that. And I think what's changed, and I think it's quite a fundamental change, is that neural networks became programmable with large language models. [03:09] (A third column "Software 3.0" (prompts) is added on the right, with a flowchart "programs -> LLM" and a diagram of a Large Language Model) And so I, I see this as quite new, unique, it's new kind of a computer. And, uh, so in my mind, it's, uh, worth giving it a new designation of Software 3.0. And basically, your prompts are now programs that program the LLM. And, uh, remarkably, uh, these, uh, prompts are written in English. So it's kind of a very interesting programming language. [03:33] (Slide with three columns for "Software 1.0", "Software 2.0", and "Software 3.0" illustrating sentiment classification. Software 1.0 shows Python code, Software 2.0 shows data examples and "train binary classifier", Software 3.0 shows a prompt for an LLM.) Um, so maybe, uh, to, uh, summarize the difference, if you're doing sentiment classification, for example, you can imagine writing some, uh, amount of Python to, to basically do sentiment classification, or you can train a neural net, or you can prompt a large language model. Uh, so here I'm, this is a few shot prompt, and you can imagine changing it and programming the computer in a slightly different way. [03:54] (Slide returns to "Map of GitHub" (Software 1.0) on the left, "HuggingFace Model Atlas" (Software 2.0) on the right, and "Software 3.0" represented by a cluster of yellow arrows emerging from a central point at the bottom right) So basically we have software 1.0, software 2.0, and I think we're seeing, I, maybe you've seen a lot of GitHub code is not just like code anymore. There's a bunch of like English interspersed with code. And so I think kind of there's a growing category of new kind of code. So not only is it a new programming paradigm, it's also remarkable to me that it's in our native language of English. [04:16] (Slide showing a pinned tweet from Andrej Karpathy @karpathy Jan 24, 2023: "The hottest new programming language is English") And so when this blew my mind, uh, a few, uh, I guess years ago now, uh, I tweeted this and, um, I think it captured the attention of a lot of people, and this is my currently pinned tweet, uh, is that remarkably we're now programming computers in English. [04:30] (Slide titled "Software is eating the world. Software 2.0 eating Software 1.0", showing a diagram of a car's autopilot system on the left, with a pink box representing "1.0 code" and a smaller blue area for "2.0 code". On the right is a detailed diagram of a BEV Net neural network processing camera inputs.) Now, when I was at, uh, Tesla, um, we were working on the, uh, autopilot and, uh, we were trying to get the car to drive. And I sort of showed this slide at the time, where you can imagine that the inputs to the car are on the bottom, and they're going through a software stack to produce the steering and acceleration. [04:56] And I made the observation at the time that there was a ton of C++ code around, uh, in the autopilot, which was the software 1.0 code, and then there was some neural nets in there doing image recognition. And I kind of observed that over time as we made the autopilot better, basically, the neural network grew in capability and size. [05:15] And, in addition to that, all the C++ code was being deleted, and kind of like was, um, and a lot of the kind of capabilities and functionality that was originally written 1.0 was migrated to 2.0. So as an example, a lot of the stitching up of information across images from the different cameras and across time was done by neural network, and we were able to delete a lot of code. [05:27] (The blue area representing "2.0 code" in the pink box of the autopilot system grows, consuming most of the "1.0 code" area) And so the, the software 2.0 stack literally ate through the software stack of the autopilot. So I thought this was really remarkable at the time. [05:36] (Slide titled "A huge amount of Software will be (re-)written.", showing a square divided into three areas: "1.0" (red, smallest), "2.0" (blue, larger), and "3.0" (yellow, emerging and growing into 2.0 and 1.0)) And I think we're seeing the same thing again, where, uh, basically we have a new kind of software and it's eating through the stack. We have three completely different programming paradigms. And I think if you're entering the industry, it's a very good idea to be fluent in all of them, because they all have slight pros and cons and you may want to program some functionality in 1.0 or 2.0 or 3.0. Are you going to train a neural net? Are you going to just prompt an LLM? [06:00] Should this be a piece of code that's explicit, et cetera? So we'll have to make these decisions and actually potentially, uh, fluidly transition between these paradigms. So what I wanted to get into now is, [06:10] (Slide with title "Part 1: How to think about LLMs") first, I want to, in the first part, talk about LLMs and how to kind of like think of this new paradigm and the ecosystem and what that looks like. Like, what are, what is this new computer? What does it look like? And what does the ecosystem look like? Um, I was struck by this quote from Andrew, actually, uh, many years ago now, I think. [06:30] (Slide with title "'AI is the new electricity' -Andrew Ng") And I think Andrew is going to, uh, be speaking right after me. Uh, but he said at the time AI is the new electricity. And I do think that it, um, kind of captures something very interesting, in that LLMs certainly feel like they have properties of utilities right now. [06:42] (Slide with title "LLMs have properties of utilities...", listing points like "Huge CAPEX", "OPEX to serve intelligence", "Metered access", "Demand for low latency, high uptime, consistent quality", "OpenRouter ~= Transfer Switch", "Intelligence 'brownouts'". An image of an electricity substation and a graph of LLM rankings are shown) So, um, LLM labs like OpenAI, Gemini, Anthropic, et cetera, they spend CAPEX to train the LLMs, and this is kind of equivalent to building out a grid. And then there's OPEX to serve that intelligence over APIs to all of us. And this is done through metered access where we pay per million tokens or something like that. And we have a lot of demands that are very utility-like demands out of this API. We demand low latency, high uptime, consistent quality, et cetera. [07:09] In electricity, you would have a transfer switch so you can transfer your electricity source from like grid and solar or battery or generator. In LLMs, we have maybe OpenRouter and easily switch between the different types of LLMs that exist. Because the LLMs are software, they don't compete for physical space. So it's okay to have basically like six electricity providers, and you can switch between them, right, because they don't compete in such a direct way. [07:32] And I think what's also a little fascinating, and we saw this in the last few days, actually, a lot of the LLMs went down, and people were kind of like stuck and unable to work. And, uh, I think it's kind of fascinating to me that when the state of the art LLMs go down, it's actually kind of like an intelligence brownout in the world. It's kind of like when the voltage is unreliable in the grid, and, uh, the planet just gets dumber, [07:54] (Speaker gestures with hands as if describing the planet getting dumber) the more reliance we have on these models, which already is like really dramatic, and I think will continue to grow. But LLMs don't only have properties of utilities. I think it's also fair to say that they have some properties of fabs. [08:05] (Slide with title "LLMs have properties of fabs...", listing points like "Huge CAPEX", "Deep tech tree R&D, secrets", "4nm process node ~= 10^20 FLOPS cluster", "Anyone training on NVIDIA GPUs ~= fabless", "Google training on TPUs ~= owns fab (e.g. Intel)". Images of a large semiconductor factory and a data center are shown) And the reason for this is that the CAPEX required for building LLMs is actually quite large. Uh, it's not just like building some, uh, power station or something like that, right? Uh, you're investing a huge amount of money. And I think the tech tree and it, for the technology is growing quite rapidly. So we're in a world where we have sort of deep tech trees, research and development, secrets that are centralizing inside the LLM labs. [08:30] Um, and I, but I think the analogy muddies a little bit also, because as I mentioned, this is software, and software is a bit less defensible, uh, because it is so malleable. [08:45] And so, um, I think it's just an interesting kind of thing to think about potentially. There's many analogies you can make, like a four nanometer process node maybe is something like a cluster with certain max flops. You can think about when you're using, when you're using Nvidia GPUs, and you're only doing the software, and you're not doing the hardware, that's kind of like the fabless model. [09:00] But if you're actually also building your own hardware, and you're training on TPUs, if you're Google, that's kind of like the Intel model where you own your fab. So I think there's some analogies here that make sense. But actually, I think the analogy that makes the most sense, perhaps, is that in my mind, LLMs have very strong kind of analogies to operating systems. [09:16] (Slide titled "LLMs have properties of Operating Systems...", listing points about LLMs as complex software ecosystems, software, switching friction, and system/user space. Images of Windows 11, macOS, various Linux distributions, and a complex network diagram representing LLM ecosystem are shown) Uh, in that, this is not just electricity or water. It's not something that comes out of a tap as a commodity. Uh, this is, these are now increasingly complex software ecosystems. Right? So, uh, they're not just like simple commodities like electricity. And it's kind of interesting to me that the ecosystem is shaping in a very similar kind of way where you have a few closed source providers like Windows or macOS, and then you have an open source alternative like Linux. [09:40] And I think for, uh, neural, for LLMs as well, we have a kind of a few competing closed source, uh, providers. And then maybe the Llama ecosystem is currently like maybe a close approximation to something that may grow into something like Linux. Again, I think it's still very early because these are just simple LLMs, but we're starting to see that these are going to get a lot more complicated. It's not just about the LLM itself, it's about all the tool use, [10:00] and the multimodalities, and how all of that works. And so when I sort of had this realization a while back, I tried to sketch it out. [10:10] (Slide with title "LLM OS", showing a block diagram similar to a computer architecture, with LLM at the center connected to CPU, RAM (context window), disk (file system + embeddings), peripheral devices (video, audio), ethernet (browser, other LLMs), and "Software 1.0 tools ('classical computer')" like calculator, Python interpreter, terminal) And it kind of seemed to me like LLMs are kind of like a new operating system, right? So the LLM is a new kind of a computer. It's sitting, it's kind of like the CPU equivalent. Uh, the context windows are kind of like the memory, and then the LLM is orchestrating memory and compute, uh, for problem solving, um, using all of these, uh, capabilities here. So definitely, if you look at it, it looks very much like operating system from that perspective. [10:37] (Slide with title "You can run an app like VS Code on:", listing "Windows 10, 11", "Mac 10.15", "Linux", and showing logos for Windows, Linux (Tux), and Apple (Mac) with download options for VS Code) Um, a few more analogies. For example, if you want to download an app, say I go to VS Code and I go to download. You can download VS Code and you can run it on Windows, Linux, or, [10:50] or Mac. In the same way, as you can take an LLM app like Cursor, [10:52] (Slide adds "Just like you can run an LLM app like Cursor on:", listing "GPT o3", "Claude 4-sonet", "Gemini 2.5-pro", "DeepSeek", and showing a screenshot of Cursor chat interface with a dropdown for selecting models) and you can run it on GPT or Claude or Gemini series, right? There's just a drop down. So it's kind of like similar in that way as well. [11:02] (Slide with title "1950s - 1970s time-sharing era", listing characteristics: "OS runs in the cloud", "I/O is streamed back and forth over the network", "compute is batched over users". Images show people interacting with early mainframe computers.) Uh, more analogies that I think strike me is that we're kind of like in this 1960s-ish era where LLM compute is still very expensive for this new kind of a computer. And that forces the LLMs to be centralized in the cloud, and we're all just, uh, sort of thin clients that interact with it over the network. [11:20] And none of us have full utilization of these computers, and therefore, it makes sense to use time sharing where we're all just, you know, a dimension of the batch when they're running the computer in the cloud. And this is very much what computers used to look like during this time. The operating systems were in the cloud, everything was streamed around, and there was batching. [11:40] And so the, the personal computing revolution hasn't happened yet because it's just not economical, it doesn't make sense, but I think some people are trying. [11:48] (Slide titled "Early hints of Personal Computing v2", showing tweets about running Llama on Apple Silicon Macs, with images of stacked Mac Minis acting as compute clusters) And it turns out that Mac Minis, for example, are a very good fit for some of the LLMs, because it's all, if you're doing batch one inference, this is all super memory bound, so this actually works. And, uh, I think these are some early indications maybe of personal computing, but this hasn't really happened yet. It's not clear what this looks like. Maybe some of you get to invent what, what this is, or how it works, or, uh, what it should, what it should be. [12:10] (Slide showing a meme comparing an old terminal screen and a ChatGPT screenshot, with a character from The Office saying "They're the same picture.") Maybe one more analogy that I'll mention is, whenever I talk to ChatGPT or some LLM directly in text, I feel like I'm talking to an operating system through the terminal. Like it just, it's, it's text, it's direct access to the operating system, and I think a GUI hasn't yet really been invented in like a general way. [12:30] Like should ChatGPT have a GUI? What, different than just the text bubbles? Uh, certainly some of the apps that we're going to go into in a bit have GUI, but there's no like GUI across all the tasks that make sense. [12:44] (Speaker gestures with hands) Um, there are some ways in which LLMs are different from kind of operating systems in some fairly unique way and from early computing. [12:53] (Slide titled "Power to the people: How LLMs flip the script on technology diffusion", with a meme of a three-headed dragon (Government, Corporations, Consumer) and text arrows indicating technology diffusion direction. "LLMs" is at the bottom, with an arrow pointing up towards "Consumer" head, which is happy and sticking its tongue out. Examples of technology and their usual diffusion path are listed.) And I wrote about, uh, this, uh, one particular property that strikes me as very different, uh, this time around. It's that LLMs like flip, they flip the direction of technology diffusion, uh, that is usually, uh, present in technology. So, for example, with electricity, cryptography, computing, flight, internet, GPS, lots of new transformative technologies that have not been around. Typically, it is the government and corporations that are the first users, because it's new and expensive, et cetera. And it only later diffuses to consumer. [13:21] But I feel like LLMs are kind of like flipped around. So, maybe with early computers, [13:24] (An image of military ballistics calculations is added next to the dragon meme) it was all about ballistics and military use. But with LLMs, it's all about how do you boil an egg or something like that. [13:30] (An image of a boiling egg is added next to the dragon meme) This is certainly like a lot of my use. So it's really fascinating to me that we have a new magical computer, and it's like helping me boil an egg. It's not helping the government do something really crazy like some military ballistics or some special technology. Indeed, corporations or governments are lagging behind the adoption of all of us of all of these technologies. [13:50] So it's just backwards, and I think it informs maybe some of the uses of how we want to use this technology or like where are some of the first apps and so on. So in summary so far, [14:00] (Slide with title "Part 1 Summary", listing: "LLM labs: - Fab LLMs", "LLMs ~= Operating Systems (circa 1960s)", "Available via time-sharing, distributed like utility", "NEW: Billions of people have sudden access to them! It is our time to program them.") LLM labs, I think, accurate language to use. But LLMs are complicated operating systems. They're circa 1960s in computing and we're redoing computing all over again. And they're currently available via time-sharing and distributed like a utility. What is new and unprecedented is that they're not in the hands of a few governments and corporations. They're in the hands of all of us because we all have a computer and it's all just software. [14:20] And ChatGPT was beamed down to our computers like to billions of people like instantly and overnight, and this is insane. Uh, and it's kind of insane to me that this is the case, and now it is our time to enter the industry and program these computers. This is crazy. So I think this is quite remarkable. [14:39] (Slide with title "Part 2: LLM Psychology") Before we program LLMs, we have to kind of like spend some time to think about what these things are, and I especially like to kind of talk about their psychology. So, the way I like to think about LLMs is that they're kind of like people spirits. [14:53] (Slide with title "LLMs are 'people spirits': stochastic simulations of people. Simulator = autoregressive Transformer", showing an artistic representation of a translucent human figure in a data stream, with a Transformer model architecture diagram on the right) Um, they are stochastic simulations of people, um, and the simulator in this case happens to be an autoregressive Transformer. So Transformer is a neural net. Uh, it's and it just kind of like is a goes on the level of tokens, it goes chunk, chunk, chunk, chunk. And there's an almost equal amount of compute for every single chunk. [15:10] Um, and, um, this simulator, of course, is, is just, is basically there's some weights involved, and we fit it to all of text that we have on the internet and so on. And you end up with this kind of a simulator. And because it is trained on humans, it's got this emergent psychology that is human-like. [15:28] (Slide titled "Encyclopedic knowledge/memory, ...", showing a young man reading in a library and a movie poster for "Rain Man") So the first thing you'll notice is, of course, LLMs have encyclopedic knowledge and memory, uh, and they can remember lots of things, a lot more than any single individual human can, because they have read so many things. It, it actually kind of reminds me of this movie Rain Man, which I actually really recommend people watch, it's an amazing movie, I love this movie. Um, and Dustin Hoffman here is an autistic savant, who has almost perfect memory. So he can read it, a, he can read like a phone book and remember all of the names and phone numbers. And I kind of feel like LLMs are kind of like very similar. They can remember SHA hashes and lots of different kinds of things very, very easily. [16:01] So they certainly have superpowers in some in some respects. But they also have a bunch of, I would say, cognitive deficits. So they hallucinate quite a bit, um, and they kind of make up stuff and don't have a very good, uh, sort of internal model of self-knowledge, not sufficient at least, and this has gotten better, but not perfect. [16:20] (Slide titled "Jagged intelligence", showing a student looking frustrated at homework, with "2 + 2 = 5" on a whiteboard. Text examples listed: "9.11 > 9.9", "two 'r' in 'strawberry'") They display jagged intelligence. So they're going to be superhuman in some problem solving domains. And then they're going to make mistakes that basically no human will make. [16:30] Like, you know, they will insist that 9.11 is greater than 9.9, or that there are two Rs in strawberry. These are some famous examples. But basically, there are rough edges that you can trip on. So that's kind of, I think, also kind of unique. [16:42] (Slide titled "Anterograde amnesia", showing a student looking lost with a paper saying "What did you eat for breakfast?". Text explanation: "Context windows ~= working memory. No continual learning, no equivalent of 'sleep' to consolidate knowledge, insight or expertise into weights.") Uh, they also kind of suffer from anterograde amnesia. Um, so, uh, and I think I'm alluding to the fact that if you have a co-worker who joins your organization, this co-worker will over time learn your organization and, uh, they will understand and gain like a huge amount of context on the organization, and they go home and they sleep, and they consolidate knowledge and they develop expertise over time. [17:02] LLMs don't natively do this, and this is not something that has really been solved in the R&D of LLMs, I think. Um, and so context windows are really kind of like working memory, and you have to sort of program the working memory quite directly because they don't just kind of like get smarter by, uh, by default. And I think a lot of people get tripped up by the analogies, um, in this way. [17:21] (Slide titled "In popular culture...", showing movie posters for "Memento" and "50 First Dates") In popular culture, I recommend people watch these two movies, uh, Memento and 50 First Dates. In both of these movies, the protagonists, their weights are fixed, and their context windows gets wiped every single morning, and it's really problematic to go to work or have relationships when this happens, and this happens to LLMs all the time. [17:39] (Slide titled "Gullibility", showing a young man with a stack of books with "TRUST ME" on them. Text: "=> Prompt injection risks, e.g. of private data") I guess one more thing I would point to is security, uh, kind of related limitations of the use of LLMs. So, for example, LLMs are quite gullible. Uh, they are susceptible to prompt injection risks. They might leak your data, et cetera. [17:52] And so, um, and there's many other considerations, uh, security related. So, so basically, long story short, [18:00] (Slide with title "Part 2 Summary", showing a collage of images from previous slides illustrating various psychological aspects of LLMs. Text: "Kind of a lossy simulation of a savant with cognitive issues.") you have to load your, you have to load your, you have to simultaneously think through this superhuman thing that has a bunch of cognitive deficits and issues. How do we, and yet, they are extremely like useful. And so how do we program them, and how do we work around their deficits and enjoy their superhuman powers. [18:18] (Slide with title "Part 3: Opportunities") So what I want to switch to now is talking about the opportunities of how do we use these models and what are some of the biggest opportunities. This is not a comprehensive list, just some of the things that I thought were interesting for this talk. [18:27] (Speaker makes a gesture with his hands like "here are the opportunities") The first thing I'm kind of excited about is, [18:30] (Slide with title "Partial autonomy apps", listing '"Copilot" / "Cursor for AI"') what I would call partial autonomy apps. So, for example, let's work with the example of coding. [18:34] (Slide with title "Example: you could go to an LLM to chat about code...", showing a ChatGPT chat interface and an old terminal) You can certainly go to ChatGPT directly and you can start copy-pasting code around, and copy-pasting, uh, bug reports and stuff around and getting code and copy-pasting everything around. Why would you, why would you do that? Why would you go directly through the operating system? [18:49] (Slide titled "Example: Anatomy of Cursor", showing the VS Code interface with Cursor's LLM integration. Text points to "Traditional interface" and "LLM integration") It makes a lot more sense to have an app dedicated for this. And so I think many of you, uh, use, uh, Cursor. I do as well. Uh, and, uh, Cursor is kind of like the thing you want instead. You don't want to just directly go to the ChatGPT. [19:07] And I think, uh, Cursor is a very good example of an early LLM app that has a bunch of properties that I think are, um, useful across all the LLM apps. So in particular, you will notice that we have a traditional interface [19:20] (Points 1, 2, and 3 are highlighted on the slide next to the "LLM integration" section) that allows a human to go in and do all the work manually, just as before. But in addition to that, we now have this LLM integration that allows us to go in bigger chunks. And so some of the properties of LLM apps that I think are shared and useful to point out. Number one, the LLMs basically do a ton of the context management, um, number two, they orchestrate multiple calls to LLMs. [19:35] Right? So in the case of Cursor, there's under the hood embedding models for all your files, the actual chat models, models that apply diffs to the code, and this is all orchestrated for you. A really big one that, uh, I think also maybe not fully appreciated always, is application specific GUI, and the importance of it. [19:53] Um, because you don't just want to talk to the operating system directly in text. Text is very hard to read, interpret, understand. And also like, you don't want to take some of these actions natively in text. [20:02] So, it's much better to just see a diff as like red and green change, and you can see what's being added and subtracted. It's much easier to just do command Y to accept or command N to reject. I shouldn't have to type it in text, right? So a GUI allows a human to audit the work of these fallible systems and to go faster. I'm going to come back to this point a little bit later as well. [20:23] (Point 4 is added: "Autonomy slider: Tab -> Cmd+K -> Cmd+L -> Cmd+I (agent mode)") And the last kind of feature I want to point out is that there's what I call the autonomy slider. So, for example, in Cursor, you can just do tab completion. You're mostly in charge. You can select a chunk of code and command K to change just that chunk of code. You can do command L to change the entire file, or you can do command I, which just, you know, let her rip, do whatever you want in the entire repo. And that's the sort of full autonomy agent, agentic version. [20:47] And so you are in charge of the autonomy slider, and depending on the complexity of the task at hand, you can, uh, tune the amount of autonomy that you're willing to give up for that task. Maybe to show one more example of a fairly successful LLM app, [20:59] (Slide titled "Example: Anatomy of Perplexity", showing a screenshot of Perplexity AI with a search result and an "autonomy slider". Features listed: "Package information into a context window", "Orchestrate multiple LLM models", "Application-specific GUI for Input/Output UIUX", "autonomy slider (search, research, deep research + suggested followup questions)") Perplexity, um, it also has very similar features to what I've just pointed out in Cursor. Uh, it packages up a lot of the information, it orchestrates multiple LLMs. It's got a GUI that allows you to audit some of its work. So, for example, it will cite sources, and you can imagine inspecting them. And it's got an autonomy slider. You can either just do a quick search, or you can do research, or you can do deep research, and come back 10 minutes later. [21:26] So this is all just varying levels of autonomy that you give up to the tool. So, I guess my question is, [21:29] (Slide titled "What does all software look like in the partial autonomy world?", showing screenshots of Adobe Photoshop and Unreal Engine, with questions below: "Can an LLM "see" all the things the human can?", "Can an LLM "act" in all the ways a human can?", "How can a human supervise and stay in the loop?") I feel like a lot of software will become partially autonomous. And I'm trying to think through like, what does that look like? And for many of you who maintain products and services, how are you going to make your products and services partially autonomous? Can an LLM "see" everything that a human can see? Can an LLM "act" in all the ways that a human can act? And can humans supervise and stay in the loop of this activity, because again, these are fallible systems that aren't yet perfect. [21:54] And what does a diff look like in Photoshop or something like that, you know? And also, a lot of the traditional software right now, it has all these switches and all this kind of stuff that's all designed for human. All of this has to change and become accessible to LLMs. [22:08] (Slide titled "Consider the full workflow of partial autonomy UIUX", showing a circular diagram with "AI" and "HUMAN" arrows rotating. "Generation" arrow points from AI, "Verification" arrow points from Human.) So, one thing I want to stress with a lot of these LLM apps that I'm not sure gets, uh, as much attention as it should is, um, we're now kind of like cooperating with AIs, and usually they are doing the generation, and we as humans are doing the verification. [22:23] It is in our interest to make this loop go as fast as possible, so we're getting a lot of work done. There are two major ways that I, uh, think, uh, this can be done. Number one, you can speed up verification a lot, um, and I think GUIs, for example, are extremely important to this. [22:35] Because a GUI utilizes your computer vision GPU in all of our head. Reading text is effortful, and it's not fun, but looking at stuff is fun, and it's, it's just a kind of like a highway to your brain. So I think GUIs are very useful for auditing systems and visual representations in general. [22:54] (Slide updates to show point 1: "Make this EASY FAST to win" and point 2: "Keep AI 'on a tight leash' to increase the probability of successful verification") And number two, I would say is, we have to keep the AI on the leash. We, I think a lot of people are getting way over excited with AI agents. And, uh, it's not useful to me to get a diff of 1,000 lines of code to my repo. Like, I have to, I'm still the bottleneck, right? Even though the 1,000 lines come out instantly, I have to make sure that this thing is not introducing bugs, is just like, and that it is doing the correct thing, right? And that there's no security issues and so on. [23:20] So I'm, I think that, um, yeah, basically, you, we have to sort of like, it's in our interest to make the, the flow of these two go very, very fast. And we have to somehow keep the AI on the leash because it gets way too overreactive. It's, uh, it's kind of like this. [23:33] (Slide shows a cartoon of a human coder on a leash held by a large robot with multiple monitors, representing "Human+AI UIUX for Coding") This is how I feel when I do AI assisted coding. If I'm just vibe coding, everything is nice and great, but if I'm actually trying to get work done, it's not so great to have an overreactive, uh, agents doing all this kind of stuff. So, [23:46] (Slide shows a bulleted list titled "Example: keeping agents on the leash", outlining steps for "AI-assisted coding" workflows) this slide is not very good. I'm sorry, but I guess I'm trying to develop like many of you, some ways of utilizing these agents in my coding workflow, and to do AI assisted coding. And in my own work, I'm always scared to get way too big diffs. [24:00] I always go in small incremental chunks. I want to make sure that everything is good. I want to spin this loop very, very fast, and, uh, I sort of work on small chunks of single concrete thing. Uh, and so I think, uh, many of you probably are developing similar ways of working with the, with LLMs. [24:18] (Slide shows a blog post excerpt titled "Example: keeping agents on the leash", discussing prompt engineering techniques to ensure predictable LLM behavior) Um, I also saw a number of blog posts that tried to develop these best practices for working with LLMs. And here's one that I read recently, and I thought was quite good, and it kind of discusses some techniques. And some of them have to do with how you keep the AI on the leash. So, as an example, if you are prompting, if your prompt is big, then, uh, the AI might not do exactly what you wanted. [24:40] And in that case, verification will fail. You're going to ask for something else. If verification fails, then you're going to start spinning. So it makes a lot more sense to spend a bit more time to be more concrete in your prompts, which increases the probability of successful verification, and you can move forward. And so I think a lot of us are going to end up finding, um, kind of techniques like this. [24:56] (Slide titled "Example: keeping agents on the leash", showing two images: a textbook ("1. App for course creation (for teacher)") and two students studying ("2. App for course serving (for student)"). Below the title is a formula: "AI = Education / LLM101h") I think in my own work as well. I'm currently interested in, uh, what education looks like in, uh, together with, uh, kind of like now that we have AI, uh, and LLMs, what does education look like? And I think a lot, a large amount of thought for me goes into how we keep AI on the leash. I don't think it just works to go to ChatGPT and be like, hey, teach me physics. I don't think this works because the AI is like gets lost in the woods. [25:18] And so for me, this is actually two separate apps, for example. Uh, there's an app for a teacher that creates courses, and then there's an app that takes courses and serves them to students. And in both cases, we now have this intermediate artifact of a course that is auditable, and we can make sure it's good. We can make sure it's consistent. And the AI is kept on the leash with respect to a certain syllabus, a certain like, um, progression of projects, and so on. And so this is one way of keeping the AI on the leash, and I think has a much higher likelihood of working. [25:47] And the AI is not getting lost in the woods. [25:49] (Slide shows a Tesla car interior with a hand on the steering wheel and a "Autonomy slider" on the right, listing tasks from "Keep the lane" to "Take turns at intersections") One more kind of analogy I wanted to sort of allude to is, I'm not, I'm no stranger to partial autonomy, and I'm kind of worked on this, I think, for five years at Tesla. And this is also a partial autonomy product, and shares a lot of the features. Like, for example, right there in the instrument panel is the GUI of the autopilot. So it's showing me what the what the neural network sees and so on. [26:10] And we have the autonomy slider, where over the course of my tenure there, we did more and more autonomous tasks for the user. And maybe the story that I wanted to tell very briefly is, [26:20] (Slide shows a photo of a white car parked on a street with text "2015 - 2025 was the decade of "driving agents" 2013: my first demo drive in a Waymo around Palo Alto (it was perfect).") uh, actually, the first time I drove a self-driving vehicle was in 2013. And I had a friend who worked at Waymo, and, uh, he offered to give me a drive around Palo Alto, highways, uh, streets, and so on. And this drive was perfect. [26:45] There was zero interventions. And this was 2013, which is now 12 years ago. And it's kind of struck me because at the time when I had this perfect drive, this perfect demo, I felt like, wow, self-driving is imminent, because this just worked. This is incredible. [27:03] Um, but here we are 12 years later, and we are still working on autonomy. Um, we are still working on driving agents. And even now, we haven't actually like really solved the problem. Like, you may see Waymos going around, and they look driverless, but, you know, there's still a lot of teleoperation and a lot of human in the loop of a lot of this driving. [27:20] So we still haven't even like declared success, but I think it's definitely like going to succeed at this point, but it just took a long time. And so I think like, like this is, uh, software [27:30] is really tricky, I think, in the same way that driving is tricky. And so when I see things like, oh, 2025 is the year of agents, I get very concerned, and I kind of feel like, you know, this is the decade of agents, and this is going to be quite some time. [27:45] We need humans in the loop. We need to do this carefully. This is software. Let's be serious here. One more kind of analogy that I always think through is [27:55] (Slide titled "THE IRON MAN SUIT", showing Iron Man in different forms: augmentation (human in suit), a blueprint/model of the suit with a slider, and an agent (suit flying independently)) the Iron Man suit. Uh, I think this is, I always love Iron Man. I think it's like so, um, correct in a bunch of ways with respect to technology and how it will play out. And what I love about the Iron Man suit is that it's both an augmentation, and Tony Stark can drive it, and it's also an agent. [28:10] And in some of the movies, the Iron Man suit is quite autonomous and can fly around and find Tony and all this kind of stuff. And so this is the autonomy slider is we can be, we can build augmentations, or we can build agents. And we kind of want to do a bit of both. But at this stage, I would say working with fallible LLMs and so on, [28:24] (Slide titled "Building Autonomous Software", with two columns: "X Iron Man robots", "X Flashy demos of autonomous agents", "X AGI 2027" on the left, and "✅ Iron Man suits", "✅ Partial autonomy products", "✅ Custom GUI and UIUX", "✅ Fast Generation - Verification loop", "✅ Autonomy slider" on the right) I would say, you know, it's less Iron Man robots, and more Iron Man suits that you want to build. It's less like building flashy demos of autonomous agents, and more building partial autonomy products. And these products have custom GUIs and UIUX, and we're trying to, um, and this is done so that the generation verification loop with the human is very, very fast. [28:47] But we are not losing the sight of the fact that it is in principle possible to automate this work, and there should be an autonomy slider in your product, and you should be thinking about how you can slide that autonomy slider, and make your product, uh, sort of, um, more autonomous. [29:02] But this is kind of how, I think, there's lots of opportunities in these kinds of products. I want to now switch gears a little bit, and talk about one other dimension that I think is very unique. [29:09] (Slide with title "Make software highly accessible 😉") Not only is there a new type of programming language that allows for autonomy in software, but also, as I mentioned, it's programmed in English, which is this natural interface. And suddenly, everyone is a programmer because everyone speaks natural language like English. So this is extremely bullish and very interesting to me, and also completely unprecedented, I would say. [29:30] It used to be the case that you need to spend five to 10 years studying something to be able to do something in software. This is not the case anymore. So I don't know if by any chance, anyone has heard of vibe coding. [29:40] (Slide showing a tweet by Andrej Karpathy @karpathy "There's a new kind of coding I call "vibe coding", where you fully give in to the vibes, embrace exponentials, and forget that the code even exists. It's possible because the LLMs (e.g. Cursor Composer w Sonnet) are getting too good...") Uh, this, this is the tweet that kind of like introduced this. But I'm told that this is now like a major meme. Um, fun story about this is that, [29:50] I've been on Twitter for like 15 years or something like that at this point, and I still have no clue which tweet will become viral, and which tweet like fizzles and no one cares. And I thought that this tweet was going to be the latter. I don't know, it was just like a shower of thoughts, but this became like a total meme. [30:20] (Audience laughs and applauds) And I really just can't tell, but I guess like it struck a chord, and it gave a name to something that everyone was feeling, but couldn't quite say in words. [30:13] (Slide shows a Wikipedia page for "Vibe coding", with a note that the article may contain an excessive number of citations) So now there's a Wikipedia page and everything. [30:25] (Speaker laughs) This is like a major contribution now or something like that. So, Um, [30:31] (Slide shows a tweet by Thomas Wolf @Thom_Wolf with an embedded video. The video shows young children sitting at desks with computers, working on a coding project collaboratively. They clap and cheer when their code works. The tweet text mentions "9-13 yo vibe-coding event" and AI unleashing "wildly creative builders") so Thomas Wolf from HuggingFace shared this beautiful video that I really love. Um, (Sound of children cheering) Yeah. These are kids vibe coding. (Sound of children cheering again) I, I find that this is such a wholesome video, like I love this video. Like, how can you look at this video and feel bad about the future? The future is great. [30:53] I think this will end up being like a gateway drug to software development. Um, I'm not a doomer about the future of the generation. And I think, yeah, I love this video. So, I tried vibe coding a little bit [31:05] (Slide shows a smartphone screenshot of a "Vibe Coding iOS app", which appears to be a calorie tracker with buttons to add or subtract 100 calories) as well because it's so fun. Uh, so vibe coding is so great when you want to build something super duper custom that doesn't appear to exist, and you just want to wing it because it's a Saturday or something like that. So, I built this, uh, iOS app. And I don't, I can't actually program in Swift, but I was really shocked that I was able to build like a super basic app, and I'm not going to explain it, it's really, uh, dumb. [31:25] (The app on the screen shows calories being added and subtracted, eventually settling on a negative number) But, uh, I kind of like, this was just like a day of work, and this was running on my phone like later that day, and I was like, wow, this is amazing. I didn't have to like read through Swift for like five days or something like that to like get started. [31:37] (Slide shows a restaurant menu on the left and a webpage on the right with images corresponding to menu items. Text: "Vibe coding MenuGen https://www.menugen.app/") I also vibe coded this app called MenuGen. And this is live. You can try it in menugen.app. And I basically had this problem where I show up at a restaurant, I read through the menu, and I have no idea what any of the things are. And I need pictures. So this doesn't exist, so I was like, hey, I'm going to vibe code it. [31:54] (Video plays on the slide, showing a user taking a photo of a menu, uploading it to the MenuGen app, and then the app generates images for each menu item) So, um, this is what it looks like. You go to menugen.app. Um, and, uh, you take a picture of a, of a menu, and then MenuGen generates the images. And everyone gets $5 in credits for free when you sign up. And therefore, [32:12] (Speaker laughs) this is a major cost center in my life. So, (Speaker laughs again) this is a, at negative, negative, uh, revenue app for me right now. I've lost a huge amount of money on MenuGen. Okay. [32:20] (Slide with text "The code was the easiest part! :O Most of the work was in the browser clicking things." with smiling emoji. A checklist below shows: "LLM API keys ☹️", "Flux (image generation) API keys ☹️", "Running locally (ez) ✅", "Vercel deployments ☹️", "Domain names ☹️", "Authentication ☹️", "Payments ☹️". A link to the blog post "Vibe coding MenuGen" is at the bottom right) But the fascinating thing about MenuGen for me, is that the code of, the vibe, the vibe coding part, the code was actually the easy part of, of vibe coding MenuGen. And most of it actually was when I tried to make it real so that you can actually have authentication and payments and a domain name and a Vercel deployment. [32:40] This was really hard, and all of this was not code. All of this DevOps stuff was in me, in the browser, clicking stuff. And this was extreme slog and took another week. So it was really fascinating that I had the MenuGen, um, basically demo working on my laptop in a few hours. [33:00] And then it took me a week because I was trying to make it real. And the reason for this is, this was just really annoying. Um, so, for example, if you try to add Google login to your webpage, I know this is very small, but just a huge amount of instructions of this Clerk library telling me how to integrate this. [33:14] And this is crazy, like it's telling me, go to this URL, click on this drop down, choose this, go to this and click on that, and it's like telling me what to do. Like a computer is telling me the actions I should be taking. Like, you do it. Why am I doing this? (Speaker laughs) What the hell? (Speaker laughs again) [33:33] (Speaker makes a puzzled expression) I had to follow all these instructions. This was crazy. So I think the last part of my talk, [33:38] (Slide with title "Build for agents 🤖") therefore, focuses on, can we just build for agents? I don't want to do this work. Can agents do this? Thank you. (Audience applauds) Okay. [33:48] (Slide with title "There is new category of consumer/manipulator of digital information:", listing "1. Humans (GUIs)", "2. Computers (APIs)", "3. NEW: Agents <-- computers... but human-like") So roughly speaking, I think there's a new category of consumer/manipulator of digital information. It used to be just humans through GUIs, or computers through APIs. And now we have a completely new thing. And agents, they're computers, but they are human-like. [34:00] Kind of, right? They're people spirits. There's people spirits on the internet, and they need to interact with our software infrastructure. Like, can we build for them? It's a new thing. So as an example, [34:11] (Slide with title "robots.txt -> /llms.txt file", showing a snippet of a text file on the left and a FastHTML documentation page on the right. The text file contains a "proposal to standardize on using an /llms.txt file to provide information to help LLMs use a website at inference time") you can have robots.txt on your domain, and you can instruct, uh, or like advise, I suppose, um, web crawlers on how to behave on your website. In the same way, you can have maybe llms.txt file, which is just a simple markdown that's telling LLMs what this domain is about. And this is very readable to a, to an LLM. [34:30] If it had to instead get the HTML of your web page and try to parse it, this is very error prone and difficult, and it will screw it up, and it's not going to work. So we can just directly speak to the LLM, it's worth it. [34:39] (Slide titled "Docs for people LLMs", showing a tweet about Vercel's documentation being available in markdown, and a "Build on Stripe with LLMs" doc page. The tweet highlights a URL to a .llms.net file) Um, a huge amount of documentation is currently written for people. So you will see things like lists, and bold, and pictures, and this is not directly accessible by an LLM. So I see some of the services now are transitioning a lot of their docs to be specifically for LLMs. So Vercel and Stripe, as an example, are early movers here. But there are, uh, a few more that I've seen already. [35:01] And they offer their documentation in markdown. Markdown is super easy for LLMs to understand. This is great. [35:09] (Slide titled "Manim: Mathematical Animation Engine", showing a screenshot of Manim's code and its visual output, along with a picture of 3Blue1Brown) Um, maybe one simple example from from, uh, my experience as well. Maybe some of you know Three Blue One Brown. He makes beautiful animation videos on YouTube. (Audience applauds) Yeah, I love this library, uh, so that he wrote, uh, Manim. And I wanted to make my own, and, uh, there's extensive documentation on how to use Manim. [35:32] And, uh, so I didn't want to actually read through it. So I copy-pasted the whole thing to an LLM, and I described what I wanted, and it just worked out of the box. Like LLM just vibe coded me an animation exactly what I wanted. And I was like, wow, this is amazing. So if we can make docs legible to LLMs, it's going to unlock a huge amount of, um, kind of use. And, um, I think this is wonderful and should, should, uh, happen more. [35:55] (Slide with title "Actions for people LLMs", showing a tweet about adding cURL commands to Vercel's documentation and a "Stripe Model Context Protocol (MCP) Server" page) The other thing I wanted to point out is that you do unfortunately have to, it's not just about taking your docs and making them appear in markdown. That's the easy part. We actually have to change the docs. Because anytime your docs say click, this is bad. An LLM will not be able to natively take this action right now. So Vercel, for example, is replacing every occurrence of click with the equivalent cURL command that your LLM agent could take on your behalf. [36:18] Um, and so I think this is very interesting. And then of course, there's, uh, Model Context Protocol from Anthropic, and this is also another way, it's a protocol of speaking directly to agents as this new consumer and manipulator of digital information. [36:30] So I'm very bullish on these ideas. The other thing I really like is, [36:32] (Slide titled "Context builders, e.g.: Gitingest", showing a GitHub repo page on the left and a Gitingest interface on the right, which simplifies the repo's content for LLMs) a number of little tools here and there that are helping ingest data that in like very LLM friendly formats. So, for example, when I go to a GitHub repo, like my NanoGPT repo, I can't feed this to an LLM and ask questions about it, uh, because it's, you know, this is a human interface on GitHub. [36:48] So when you just change the URL from GitHub to Gitingest, then, uh, this will actually concatenate all the files into a single giant text, and it will create a directory structure, et cetera. And this is ready to be copy-pasted into your favorite LLM, and you can do stuff. Maybe even more dramatic example of this is Devin DeepWiki, [37:05] (Slide titled "Context builders, e.g.: Devin DeepWiki", showing a GitHub repo page on the left and a DeepWiki page for NanoGPT on the right, which provides a system's architecture flowchart and overview) where it's not just the raw content of these files. Uh, this is from Devin. But also like they have Devin basically do analysis of the GitHub repo, and Devin basically builds up a whole docs, uh, pages just for your repo, and you can imagine that this is even more helpful to copy-paste into your LLM. [37:22] I love all the little tools that basically where you just change the URL and it makes something accessible to an LLM. So this is all well and great and, uh, I think there should be a lot more of it. [37:32] (Slide with title "Introducing Operator", showing a webpage with a detailed description of an "Operator" agent, including a transcript of its actions and a screenshot of Tripadvisor. A keyboard and mouse are shown at the bottom) One more note I wanted to make is that it is absolutely possible that in the future, LLMs will be able to, this is not even future, this is today. They'll be able to go around and they'll be able to click stuff and so on. But I still think it's very worth, uh, basically meeting LLM halfway, LLMs halfway, and making it easier for them to access all this information, uh, because this is still fairly expensive, I would say, to use, and, uh, a lot more difficult. [37:55] And so I do think that lots of software, there will be a long tail where it won't, like, adapt, because these are not like live player sort of repositories or digital infrastructure. And we will need these tools. Uh, but I think for everyone else, I think it's very worth kind of like meeting them at some middle point. So I'm bullish on both, if that makes sense. [38:15] (Slide showing a collage of images from previous slides, summarizing the talk. Text on the left lists "Partial autonomy LLM apps:", "Package context", "Orchestrate LLM calls", "Custom GUI", "Autonomy slider". Text on the right lists "speed up the full generation-verification flow" and "Build for agents 🤖") So in summary, what an amazing time to get into the industry. We need to rewrite a ton of code, a ton of code will be written by professionals and vibe coders. These LLMs are kind of like utilities, kind of like fabs, but they're kind of especially like operating systems. But it's so early. It's like 1960s of operating systems. And, uh, and I think a lot of the analogies cross over. Um, and these LLMs are kind of like these fallible, uh, you know, people spirits that we have to learn to work with. [38:45] And in order to do that properly, we need to adjust our infrastructure towards it. So when you're building these LLM apps, I described some of the ways of working effectively with these LLMs, and some of the tools that make that kind of possible. And how you can spin this loop very, very quickly. And basically, create partial autonomy products. And then, um, yeah, a lot of code has to also be written for the agents more directly. But in any case, going back to the Iron Man suit analogy, [39:09] (The Iron Man suit slider animation plays on the screen, moving from "Augmentation" to "Agent") I think what we'll see over the next decade, roughly, is we're going to take the slider from left to right. And I'm very interesting, it's going to be very interesting to see what that looks like. And I can't wait to build it with all of you. Thank you. (Audience applauds)
_This detailed transcription covers Andrej Karpathy's talk on the evolution of software, from Software 1.0 (code) to Software 2.0 (weights/neural networks), and now Software 3.0 (prompts/LLMs), highlighting LLMs' properties as utilities, fabs, and operating systems, and emphasizing the opportunities in building partially autonomous applications with custom GUIs and fast human-AI verification loops, while keeping AI on a "tight leash."_

</details>

<details>
<summary>The provided markdown content discusses "Context" in AI models, including system/user prompts, context limits, and tool integration.</summary>

The provided markdown content discusses "Context" in AI models, including system/user prompts, context limits, and tool integration.
The article guidelines, however, focus on "adding humans in the loop into our Brown writing workflow," with a specific outline covering the AI generation/human validation loop, human feedback, editing workflows, and MCP server integration.

The content of the provided markdown is a general explanation of AI context, which is a foundational concept but is not directly pertinent to the specific human-in-the-loop writing workflow outlined in the article guidelines. It does not contain any information about the "Brown writing workflow," "human-in-the-loop," "MCP," or specific editing workflows mentioned in the guidelines.

Therefore, according to the instruction "Focus on keeping only the core textual content (and code content if there are code sections) that is pertinent to the article guidelines provided below," the entire provided markdown content is irrelevant to the specified article guidelines.

</details>

<details>
<summary>AI agents are no longer just passive observers in our applications—they’re active participants in our systems. With the rapid rise of LLMs, agents today can query APIs, modify infrastructure, respond to users, and trigger workflows. They’re increasingly embedded in customer support, developer tools, operations, and even decision-making pipelines.</summary>

AI agents are no longer just passive observers in our applications—they’re active participants in our systems. With the rapid rise of LLMs, agents today can query APIs, modify infrastructure, respond to users, and trigger workflows. They’re increasingly embedded in customer support, developer tools, operations, and even decision-making pipelines.

But with this autonomy comes a critical question:

> **Can you trust an agent to act without oversight?**

The short answer: **no.**

Agents may hallucinate actions, misinterpret prompts, or overstep boundaries. And when those actions touch sensitive systems—like access control, financial operations, or customer data—it's not a risk that you should take.

It also doesn’t mean that AI agents have no place in performing these actions. We all know that they will have the capability to do it sooner or later, and it’s only a question of how we adequately prepare for it, especially from a security perspective.

That’s where the concept of **Human-in-the-Loop (HITL)** comes in. By **delegating final decisions to a human**, developers can combine the efficiency of automation with the judgment of real people. That’s what we're here to talk about. This article explores:

- Why HITL is essential in agentic systems

- What it means to delegate permissions to humans

- The best tools and frameworks for building HITL workflows

- Real-world use cases where oversight is non-negotiable

- And a short glimpse into a working demo that ties it all together


Let’s get into it -

## Why Human Oversight Is Essential in Agentic Workflows

AI agents are incredibly useful—but they’re not infallible. Even the most advanced LLMs operate without true understanding. They can simulate reasoning, but they can’t evaluate risk, interpret social nuance, or take accountability for decisions.

When agents are empowered to take actions, not just generate text, that gap becomes a liability.

### What can go wrong?

- **Hallucinated actions**: The agent makes up nonexistent commands, tools, or resource IDs.

- **Misused permissions**: A vague prompt leads the agent to act outside its intended scope.

- **Overreach**: The agent tries to approve its own access or bypass a restriction.

- **Lack of traceability**: No one knows who authorized what, or why it was allowed.


Now imagine those behaviors in the context of changing user roles in a production system, approving infrastructure changes in a CI/CD flow, accessing knowledge bases, deleting customer data, or issuing refunds. You can see how quickly risks can escalate.

### Human-in-the-loop (HITL) isn’t just about safety—it’s about **control**

Inserting humans at key decision points allows you to **prevent irreversible mistakes** before they happen, **ensure accountability**, so that every action has a reviewer or approver, **comply with audit requirements**, such as SOC 2 policies and internal governance, as well as **build trust** by making your AI a supervised assistant, instead of a a black box.

This isn't a nice-to-have - in many agentic workflows, **HITL is the only responsible way forward**.

## What Delegating Permissions to Humans Really Means

When we talk about **delegating permissions** in AI systems, we don’t mean giving humans more buttons to press. We mean teaching agents to **ask for permission**—and wait.

This is the core of human-in-the-loop (HITL):

> **The agent doesn’t act until a human explicitly approves the request.**

### The HITL Control Loop

The HITL workflow follows a predictable and repeatable pattern:

https://us-west-2.graphassets.com/AuGrs0mztRH6ldTYKJkSAz/cmbj7g4pvwl6y07n3mhq38v9g

1. **Agent receives a task by** processing a prompt or trigger.

2. **Agent proposes an action,** like “I need access to this document”.

3. **Agent pauses and** routes the request to a human approver using built-in `interrupt()` .

4. **Human reviews** the context and either approves or rejects the proposed action.

5. **Agent resumes,** executing the action only if it has been approved.


This pattern works across roles, industries, and use cases—from DevOps and security to customer service and compliance.

## Frameworks and Libraries for HITL Integration

The agent ecosystem is evolving fast—and so are the tools designed to add **human judgment** into the loop.

Below is a breakdown of the most effective frameworks and libraries you can use today to build human-in-the-loop (HITL) workflows for AI agents. Each has a different focus, but all support some form of permission delegation or real-time human approval.

### Quick Comparison

| Framework / Tool | Strengths | HITL Support |
| --- | --- | --- |
| **LangGraph** | Graph-based control, modular nodes, native interrupt/resume support | `interrupt()` pauses the graph for human input |
| **CrewAI** | Multi-agent task orchestration with role-based design | `human_input` flag and `HumanTool` support |
| **HumanLayer** | SDK/API for integrating human decisions across channels (Slack, Email, Discord) | `@require_approval()` decorator and `human_as_tool()` support |
| **LangChain MCP Adapters** | Bridges LangChain agents with external access/approval flows like [Permit.io](http://permit.io/ "http://permit.io/") | HITL when combined with LangGraph or LangChain callbacks |
| [**Permit.io**](http://permit.io/ "http://permit.io/") **\+ MCP** | Real-world authorization engine with UI/API-based access and operation approvals | Built-in support for access requests and delegated approvals |

### Framework Highlights

[**LangGraph**](https://langchain-ai.github.io/langgraph/ "https://langchain-ai.github.io/langgraph/") is ideal for building structured workflows where you need full control over how an agent reasons, routes, and pauses. Its `interrupt()` function lets you **pause the graph mid-execution**, wait for human input, and resume cleanly, ****making it a top choice for inserting HITL checkpoints. Use it when you need **custom routing logic**, deterministic, debuggable behavior, or when you're managing multiple agents/tool types.

[**CrewAI**](https://www.crewai.com/ "https://www.crewai.com/") focuses on **collaborative, role-based agent teams**. It’s great for decomposing tasks among agents with different goals or capabilities. HITL comes in via `human_input`, or by defining a **HumanTool** the agent can call for guidance. Use it when your workflow involves multiple agents, or when you want to keep humans in the loop as decision-makers or fallback experts.

The [**HumanLayer**](https://www.humanlayer.dev/ "https://www.humanlayer.dev/") SDK enables agents to communicate with humans via familiar tools (Slack, Email, Discord). Its decorators (`@require_approval`, `human_as_tool`) wrap functions to make **approval logic seamless**. Use it when you want asynchronous human decisions, notifications/multi-channel communication, or when you need an orchestration-agnostic implementation.

[**LangChain MCP Adapters**](https://python.langchain.com/docs/integrations/providers/permit/ "https://python.langchain.com/docs/integrations/providers/permit/") connect LangChain agents to real-world access systems (like [Permit.io](http://permit.io/ "http://permit.io/")). They convert access request and approval tools into LangChain-compatible tools, so agents can ask for access, but require policy-defined human approval before proceeding. Use it when you're already using LangChain, you want policy-driven access control, or when you’re integrating with Permit MCP workflows.

### [Permit.io](http://permit.io/ "http://permit.io/") \+ MCP

[Permit.io](http://permit.io/ "http://permit.io/") offers a **complete authorization-as-a-service layer**. With its **Model Context Protocol (MCP)** server, you can turn access and approval workflows into tools that an LLM can call but only execute after human approval via built-in logic or dashboard controls. Use it when you need to manage sensitive permissions, full auditability, and role enforcement, and want UI + API + agent-level integration.

https://us-west-2.graphassets.com/AuGrs0mztRH6ldTYKJkSAz/cmbj7gtdlwlfg07n3nymly51s

**These tools aren’t mutually exclusive.**

In fact, some of the strongest HITL setups use **LangGraph + MCP Adapters +** [**Permit.io**](http://permit.io/ "http://permit.io/") together for full control, flexibility, and policy enforcement.

## Key Design Patterns for HITL in Agentic Workflows

Building a human-in-the-loop (HITL) workflow isn’t just about picking a framework—it’s about choosing **where**, **when**, and **how** to involve humans in the agent's decision-making process.

Below are the most effective HITL design patterns used across real-world agentic systems.

**Interrupt & Resume:**

- **Used in:** LangGraph

- **How it works:** The agent is paused mid-execution using an `interrupt()` call. Human input is collected (yes/no, select from options, etc.), and then the workflow resumes based on the response.

- **Best for:**



- Approving tool calls (e.g. `approve_access_request`)

- Pausing long-running workflows

- Inserting human checkpoints before final actions


**Human-as-a-Tool:**

- **Used in:** LangChain, CrewAI, HumanLayer

- **How it works:** The agent sees a "human" as just another callable tool. When it's unsure, it routes a question to the human tool and uses the returned response in context.

- **Best for:**



- Ambiguous prompts

- Fact-checking or contextual clarification

- Letting humans fill in gaps (e.g. “What should I do here?”)


**Approval Flows:**

- **Used in:** Permit.io, ReBAC systems

- **How it works:** Permissions are structured such that only specific human roles (e.g. “Reviewer”) can approve access or actions. Agents can initiate requests, but only users with the right roles can approve them via UI or API.

- **Best for:**



- Fine-grained, policy-backed access control

- Scenarios where human approval is required by design (e.g. legal, financial ops)

- Systems with delegated authority and role hierarchies


**Fallback Escalation:**

- **Used in:** Hybrid agent-human systems

- **How it works:** The agent attempts to complete a task. If it fails, lacks permissions, or gets stuck, it escalates the task to a human via Slack, email, or a dashboard for resolution.

- **Best for:**



- Reducing friction while keeping a safety net

- Complex queries with low LLM confidence

- Keeping human load low but available


These patterns are **not mutually exclusive**. The best HITL architectures often combine:

- `interrupt()` for real-time decisions

- `approval roles` for policy gating

- `fallback escalation` for graceful recovery

- `audit-first` for transparency


Use them modularly. And always design around the question:

> **"Would I be okay if the agent did this without asking me?"**

## From Theory to Practice: The Food Ordering System Demo

To see all of this in action, let’s take a look at a working example:

A **family food ordering system** where AI agents handle routine tasks—but humans remain in control of sensitive decisions.

This system demonstrates a real-world implementation of HITL with:

- **LangGraph** for agent orchestration

- [**Permit.io**](http://permit.io/ "http://permit.io/") **MCP server** for access requests and approval workflows

- **LangChain MCP Adapters** to expose those workflows as tools

- **Gemini 2.0 Flash** as the reasoning engine

- **Python** for the orchestration layer and CLI interface


### Scenario: Delegated Access in a Family System

- **Parents** can view and manage all restaurants and dishes.

- **Children** can only view a subset of restaurants.

- If a child wants to access a restricted restaurant or order a premium dish:



- The agent initiates an **access request** or **operation approval**.

- The agent then **pauses** using LangGraph’s `interrupt()` function.

- A human reviewer (a parent) is prompted for approval.

- The workflow **resumes only if approval is granted**.


This means even though the LLM handles interactions and tool selection, **it never executes sensitive actions on its own.**

### What Makes This a Strong HITL Demo

- **Policy-driven**: All permissions and approval logic are managed via Permit.io

- **Real-time approvals**: `interrupt()` ensures decisions are deliberate

- **Auditable and extensible**: Every access request is logged, every tool call is human-verified

- **Modular design**: Easily adaptable to other domains—DevOps, finance, internal tools


### Try it Yourself

Explore the full implementation in the [**open-source repo**](https://github.com/permitio/permit-mcp "https://github.com/permitio/permit-mcp")

## Best Practices & Final Thoughts

As AI agents take on more responsibility in our workflows, **building for oversight isn’t optional—it’s foundational**. Human-in-the-loop (HITL) systems strike the right balance between speed and safety, automation and accountability.

Whether you're adding simple approval gates or designing complex agent workflows with delegated authority, here are the best practices to follow:

### Design for Decision Points, Not Just Prompts

Identify **where** human input is critical—access approvals, configuration changes, destructive actions—and design explicit checkpoints. Use tools like `interrupt()` to enforce those pauses.

### Keep Prompts Contextual and Lightweight

When asking humans for approval, keep the request clear, focused, and explain why it's needed. Don't overload reviewers with raw JSON—summarize context when possible.

### Use Policies, Not If-Statements

Hardcoding access rules leads to weak, non-scalable logic. Delegate approval logic to a policy engine, where changes are declarative, versioned, and enforceable across systems.

### Log Everything

Audit trails aren’t just for compliance—they’re part of the HITL loop. Ensure that every access request, approval, and denial is tracked and reviewable.

### Think Asynchronously When Needed

Not every human needs to approve something in real time. For low-priority or non-blocking flows, route to async review channels (Slack, email, dashboards) using frameworks like HumanLayer.

Human-in-the-loop is not a temporary workaround—it’s a long-term pattern for **building AI agents we can trust**. It ensures that LLMs stay within safe operational boundaries, sensitive actions don’t slip through automation, and teams remain in control—even as autonomy grows.

With tools like **LangGraph**, **Permit.io**, and **LangChain MCP Adapters**, it’s easier than ever to give agents the ability to ask for permission—without hardcoding approvals or sacrificing usability.

The best agents aren’t just intelligent. They’re responsible.

</details>

<details>
<summary>Building effective agents</summary>

# Building effective agents

Published Dec 19, 2024

We've worked with dozens of teams building LLM agents across industries. Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks.

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.

## What are agents?

"Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:

- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
- **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Below, we will explore both types of agentic systems in detail. In Appendix 1 (“Agents in Practice”), we describe two domains where customers have found particular value in using these kinds of systems.

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

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fd3083d3f40bb2b6f477901cc9a240738d3dd1371-2401x1000.png&w=3840&q=75
The augmented LLM

We recommend focusing on two key aspects of the implementation: tailoring these capabilities to your specific use case and ensuring they provide an easy, well-documented interface for your LLM. While there are many ways to implement these augmentations, one approach is through our recently released [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol), which allows developers to integrate with a growing ecosystem of third-party tools with a simple [client implementation](https://modelcontextprotocol.io/tutorials/building-a-client#building-mcp-clients).

For the remainder of this post, we'll assume each LLM call has access to these augmented capabilities.

### Workflow: Prompt chaining

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F7418719e3dab222dccb379b8879e1dc08ad34c78-2401x1000.png&w=3840&q=75
The prompt chaining workflow

**When to use this workflow:** This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

**Examples where prompt chaining is useful:**

- Generating Marketing copy, then translating it into a different language.
- Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

### Workflow: Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840&q=75
The routing workflow

**When to use this workflow:** Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

**Examples where routing is useful:**

- Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes, prompts, and tools.
- Routing easy/common questions to smaller, cost-efficient models like Claude Haiku 4.5 and hard/unusual questions to more capable models like Claude Sonnet 4.5 to optimize for best performance.

### Workflow: Parallelization

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations:

- **Sectioning**: Breaking a task into independent subtasks run in parallel.
- **Voting:** Running the same task multiple times to get diverse outputs.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840&q=75
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

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840&q=75
The orchestrator-workers workflow

**When to use this workflow:** This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

**Example where orchestrator-workers is useful:**

- Coding products that make complex changes to multiple files each time.
- Search tasks that involve gathering and analyzing information from multiple sources for possible relevant information.

### Workflow: Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75
The evaluator-optimizer workflow

**When to use this workflow:** This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

**Examples where evaluator-optimizer is useful:**

- Literary translation where there are nuances that the translator LLM might not capture initially, but where an evaluator LLM can provide useful critiques.
- Complex search tasks that require multiple rounds of searching and analysis to gather comprehensive information, where the evaluator decides whether further searches are warranted.

### Agents

Agents are emerging in production as LLMs mature in key capabilities—understanding complex inputs, engaging in reasoning and planning, using tools reliably, and recovering from errors. Agents begin their work with either a command from, or interactive discussion with, the human user. Once the task is clear, agents plan and operate independently, potentially returning to the human for further information or judgement. During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress. Agents can then pause for human feedback at checkpoints or when encountering blockers. The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully. We expand on best practices for tool development in Appendix 2 ("Prompt Engineering your Tools").

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png&w=3840&q=75
Autonomous agent

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

**Examples where agents are useful:**

The following examples are from our own implementations:

- A coding Agent to resolve [SWE-bench tasks](https://www.anthropic.com/research/swe-bench-sonnet), which involve edits to many files based on a task description;
- Our [“computer use” reference implementation](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo), where Claude uses a computer to accomplish tasks.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F4b9a1f4eb63d5962a6e1746ac26bbc857cf3474f-2400x1666.png&w=3840&q=75
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


## Code Sources

<details>
<summary>Repository analysis for https://github.com/towardsai/agentic-ai-engineering-course/blob/dev/lessons/24_human_in_the_loop/notebook.ipynb</summary>

# Repository analysis for https://github.com/towardsai/agentic-ai-engineering-course/blob/dev/lessons/24_human_in_the_loop/notebook.ipynb

## Summary
Repository: towardsai/agentic-ai-engineering-course
Branch: dev
Commit: 93cf10331fef0ff5b80e7a021794a906625d3044
File: notebook.ipynb
Lines: 3,347

Estimated tokens: 34.5k

## File tree
```Directory structure:
└── notebook.ipynb

```

## Extracted content
================================================
FILE: lessons/24_human_in_the_loop/notebook.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Lesson 24: Human-in-the-Loop for Brown Writing Workflow

In this lesson, we'll explore how to implement human-in-the-loop capabilities in the Brown writing workflow. We'll learn how to integrate human feedback into the article review and editing process, expose the workflows as MCP tools for seamless integration with AI assistants like Claude and Cursor, and create a collaborative writing experience where AI and human expertise combine.

**Learning Objectives:**

- Understand the importance of human feedback and human-in-the-loop in AI writing workflows
- Learn how to implement the HumanFeedback entity and integrate it into the ArticleReviewer node
- Explore two new editing workflows: edit article and edit selected text
- Discover how to expose Brown as an MCP server with tools, prompts, and resources
- See how to integrate Brown with MCP clients like Cursor for a coding-like writing experience

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
#   Environment variables loaded from `/Users/pauliusztin/Documents/01_projects/TAI/course-ai-agents/.env`

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
#   article_guideline.md   [1m[36minputs[m[m/                notebook_guideline.md

#   [1m[36mconfigs[m[m/               notebook.ipynb


"""
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
## 2. The Importance of Human-in-the-Loop

After generating an article using the writing workflow we explained in Lessons 22 and 23, you'll likely want to refine it further. Writing is highly subjective, and even the best AI-generated content benefits from human review and editing.

The perfect balance between AI and human expertise is to use AI to generate and automate parts of your work, then have you, as the domain expert, review and refine it. Known as the AI generation - human validation loop. 

This is exactly what we've designed the Brown writing workflow to support.

### The Human-in-the-Loop Design

We designed Brown to easily introduce humans into the loop between generating the first version of an article and refining it through additional review and editing cycles with human feedback. This means:

1. We can use a low number of review loops during initial article generation to reduce costs and latency
2. After reviewing the generated article, we can dynamically run additional review and editing workflows with human feedback
3. We can edit either the entire article or just selected sections based on your needs

### Decoupling Workflows with MCP

To enable this human-in-the-loop approach, we needed to decouple the article generation workflow from the editing workflows. We used MCP servers to achieve this separation, where:

- The `generate_article` workflow is one independent MCP tool
- The `edit_article` workflow is another independent MCP tool
- The `edit_selected_text` workflow is a third independent MCP tool

This architecture allows you to generate an article, review it, and then selectively apply additional editing workflows with your human feedback until you're satisfied with the results.

Here's how the workflow looks:
"""

"""
<img src="https://raw.githubusercontent.com/towardsai/agentic-ai-engineering-course/main/assets/images/l24_writing_workflow.png" alt="Workflow" height="700"/>
"""

"""
The diagram shows how Brown as an MCP server exposes three main tools, with a human feedback loop that allows iterative refinement until you're satisfied with the article.
"""

"""
## 3. Introducing Human Feedback into the Article Reviewer

Let's see how we introduced human feedback into our Article Reviewer Node. We'll start by explaining the `HumanFeedback` entity, then show how it's integrated into the `ArticleReviewer` node, and finally demonstrate it with a working example.
"""

"""
### 3.1 The HumanFeedback Entity

The `HumanFeedback` entity is a simple but Pydantic model that encapsulates human feedback for the article review process.

Source: `brown.entities.reviews`
```python
class HumanFeedback(BaseModel, ContextMixin):
    content: str

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    {self.content}
</{self.xml_tag}>
"""
```
"""

"""
### 3.2 Human Feedback in ArticleReviewer

Now let's see how the `ArticleReviewer` node integrates human feedback into the review process. We'll focus only on the relevant sections.

Source: `brown.nodes.article_reviewer`

1. **Initialization with Human Feedback**
```python
def __init__(
    self,
    to_review: Article | SelectedText,
    article_guideline: ArticleGuideline,
    model: Runnable,
    article_profiles: ArticleProfiles,
    human_feedback: HumanFeedback | None = None,
) -> None:
    self.to_review = to_review
    self.article_guideline = article_guideline
    self.article_profiles = article_profiles
    self.human_feedback = human_feedback

    super().__init__(model, toolkit=Toolkit(tools=[]))
```

The `ArticleReviewer` now accepts an optional `human_feedback` parameter. This allows the reviewer to work with or without human input.
"""

"""
2. **Human Feedback in the System Prompt**

The system prompt includes a dedicated section for human feedback:
```python
system_prompt_template = """
You are Brown, an expert article writer, editor and reviewer specialized in reviewing technical, educative and informational articles.

...

## Human Feedback

Along with the expected requirements, a human already reviewed the article and provided the following feedback:

{human_feedback}

If empty, completely ignore it, otherwise the feedback will ALWAYS be used in two ways:
1. First you will use the <human_feedback> to guide your reviewing process against the requirements. This will help you understand 
on what rules to focus on as this directly highlights what the user wants to improve.
2. Secondly you will extract one or more action points based on the <human_feedback>. Depending on how many ideas, topics or suggestions 
the <human_feedback> contains you will generate from 1 to N action points. Each <human_feedback> review will contain a single action point. 
3. As long the <human_feedback> is not empty, you will always return at least 1 action point, but you will return more action points 
if the feedback touches multiple ideas. 

Here is an example of a reviewed based on the human feedback:
<example_of_human_feedback_action_point>
Review(
    profile="human_feedback",
    location="Article level",
    comment="Add all the points from the article guideline to the article."
)
</example_of_human_feedback_action_point>

...
"""
```

This section instructs the LLM on how to use human feedback:
- Use it to guide the review process and focus on specific rules
- Extract action points from the feedback (1 to N depending on how many ideas are present)
- Always return at least 1 action point if feedback is provided
- Each action point becomes a review with `profile="human_feedback"`
"""

"""
3. **Injecting Human Feedback into the Prompt**

When the reviewer runs, it injects the human feedback into the system prompt:

```python
async def ainvoke(self) -> ArticleReviews | SelectedTextReviews:
    system_prompt = self.system_prompt_template.format(
        human_feedback=self.human_feedback.to_context() if self.human_feedback else "",
        article=self.article.to_context(),
        article_guideline=self.article_guideline.to_context(),
        character_template=self.article_profiles.character.to_context(),
        article_template=self.article_profiles.article.to_context(),
        structure_template=self.article_profiles.structure.to_context(),
        mechanics_template=self.article_profiles.mechanics.to_context(),
        terminology_template=self.article_profiles.terminology.to_context(),
        tonality_template=self.article_profiles.tonality.to_context(),
    )
    ...
```

If `human_feedback` is provided, it's converted to XML context format and injected. Otherwise, an empty string is used.
"""

"""
### 3.3 Example: Using ArticleReviewer with Human Feedback

Let's see a practical example of using the `ArticleReviewer` with human feedback. We'll load our sample article, article guideline, and profiles, then provide human feedback to guide the review process.

First, let's import the necessary components:
"""

from brown.entities.reviews import HumanFeedback
from brown.loaders import (
    MarkdownArticleExampleLoader,
    MarkdownArticleGuidelineLoader,
    MarkdownArticleLoader,
    MarkdownArticleProfilesLoader,
)
from brown.models import SupportedModels, get_model
from brown.nodes.article_reviewer import ArticleReviewer
# Output:
#   [32m2025-11-26 17:26:07.324[0m | [1mINFO    [0m | [36mbrown.config[0m:[36m<module>[0m:[36m10[0m - [1mLoading environment file from `.env`[0m


"""
Now let's load the sample inputs. We'll use the same article and guidelines from the test sample directory as we used in previous lessons:
"""

pretty_print.wrapped("STEP 1: Loading Context", width=100)

# Load guideline
guideline_loader = MarkdownArticleGuidelineLoader(uri=Path("article_guideline.md"))
article_guideline = guideline_loader.load(working_uri=SAMPLE_DIR)

# Load profiles
profiles_input = {
    "article": PROFILES_DIR / "article_profile.md",
    "character": PROFILES_DIR / "character_profiles" / "paul_iusztin.md",
    "mechanics": PROFILES_DIR / "mechanics_profile.md",
    "structure": PROFILES_DIR / "structure_profile.md",
    "terminology": PROFILES_DIR / "terminology_profile.md",
    "tonality": PROFILES_DIR / "tonality_profile.md",
}
profiles_loader = MarkdownArticleProfilesLoader(uri=profiles_input)
profiles = profiles_loader.load()

# Load examples
examples_loader = MarkdownArticleExampleLoader(uri=EXAMPLES_DIR)
article_examples = examples_loader.load()

article_loader = MarkdownArticleLoader(uri="article.md")
article = article_loader.load(working_uri=SAMPLE_DIR)

print(f"✓ Guideline: {len(article_guideline.content):,} characters")
print(f"✓ Article: {len(article.content):,} characters")
print(f"✓ Profiles: {len(profiles_input)} profiles loaded")
print(f"✓ Examples: {len(article_examples.examples)} article examples")
# Output:
#   [93m----------------------------------------------------------------------------------------------------[0m

#     STEP 1: Loading Context

#   [93m----------------------------------------------------------------------------------------------------[0m

#   ✓ Guideline: 6,751 characters

#   ✓ Article: 6,982 characters

#   ✓ Profiles: 6 profiles loaded

#   ✓ Examples: 2 article examples


"""
Here is a reminder on how the article looks like:
"""

pretty_print.wrapped(f"{article.to_context()[:4000]}", title="Article (first 4000 characters)")
# Output:
#   [93m--------------------------------- Article (first 4000 characters) ---------------------------------[0m

#     

#   <article>

#       # Workflows vs. Agents: The Critical Decision Every AI Engineer Faces

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

#   Image 2: Flowchart illustrating an AI agent's dynamic decision-making process driven by an LLM.

#   

#   ## Choosing Your Path

#   

#   The core difference between these two approaches lies in a single trade-off: developer-defined logic versus LLM-driven autonomy [[2]](https://decodingml.substack.com/p/llmops-for-production-agentic-rag), [[3]](https://towardsdatascience.com/a-developers-guide-to-building-scalable-ai-workflows-vs-agents/). Workflows offer high reliability at the cost of flexibility, while agents offer high flexibility at the cost of reliability.

#   

#   https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5e64d5e0-7ef1-4e7f-b441-3bf1fef4ff9a_1276x818.png 

#   Image 3: The trade-off between an agent's level of control and application reliability. (Image by Iusztin, P. from [Exploring the difference between agents and workflows [2]](https://decodingml.substack.com/p/llmops-for-production-agentic-rag))

#   

#   ### When to use LLM workflows

#   

#   W

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Now let's create human feedback and run the article reviewer:
"""

human_feedback = HumanFeedback(
    content="""Make the introduction more engaging and catchy. 
Also, expand on the definition of both workflows and agents from the first section"""
)

# Create the article reviewer
model = get_model(SupportedModels.GOOGLE_GEMINI_25_FLASH)
article_reviewer = ArticleReviewer(
    to_review=article,
    article_guideline=article_guideline,
    model=model,
    article_profiles=profiles,
    human_feedback=human_feedback,
)

print("Running article review with human feedback...")
reviews = await article_reviewer.ainvoke()
print(f"\nGenerated {len(reviews.reviews)} reviews")
# Output:
#   Running article review with human feedback...

#   

#   Generated 19 reviews


"""
Let's examine the reviews, especially focusing on the human feedback reviews:
"""

from utils import pretty_print

# Print human feedback reviews
human_feedback_reviews = [r for r in reviews.reviews if r.profile == "human_feedback"]
pretty_print.wrapped(
    f"Found {len(human_feedback_reviews)} reviews based on human feedback", title="Human Feedback Reviews"
)

for i, review in enumerate(human_feedback_reviews, 1):
    pretty_print.wrapped(
        {"profile": review.profile, "location": review.location, "comment": review.comment},
        title=f"{i}. Human Feedback Review",
    )
# Output:
#   [93m-------------------------------------- Human Feedback Reviews --------------------------------------[0m

#     Found 2 reviews based on human feedback

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------- 1. Human Feedback Review -------------------------------------[0m

#     {

#     "profile": "human_feedback",

#     "location": "Article level",

#     "comment": "Make the introduction more engaging and catchy."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------- 2. Human Feedback Review -------------------------------------[0m

#     {

#     "profile": "human_feedback",

#     "location": "Understanding the Spectrum: From Workflows to Agents - First section",

#     "comment": "Expand on the definition of both workflows and agents from the first section."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Let's also see all the other reviews from the profiles:
"""

profile_types = set(r.profile for r in reviews.reviews)

pretty_print.wrapped(
    f"Generated reviews from {len(profile_types)} different profile types: {', '.join(sorted(profile_types))}",
    title="All Reviews Summary",
)
print()

for profile_type in sorted(profile_types):
    profile_reviews = [r for r in reviews.reviews if r.profile == profile_type]
    pretty_print.wrapped(f"{profile_type.upper()}: {len(profile_reviews)} reviews")
    for i, review in enumerate(profile_reviews[:2], 1):  # Show first 2 of each type
        print(f"  {i}. [{review.location}] {review.comment[:100]}...")
    print()
# Output:
#   [93m--------------------------------------- All Reviews Summary ---------------------------------------[0m

#     Generated reviews from 7 different profile types: article_guideline, article_profile, human_feedback, mechanics_profile, structure_profile, terminology_profile, tonality_profile

#   [93m----------------------------------------------------------------------------------------------------[0m

#   

#   [93m----------------------------------------------------------------------------------------------------[0m

#     ARTICLE_GUIDELINE: 7 reviews

#   [93m----------------------------------------------------------------------------------------------------[0m

#     1. [Introduction: The Critical Decision Every AI Engineer Faces - Article level] The introduction's length is 123 words, exceeding the guideline of 100 words....

#     2. [Understanding the Spectrum: From Workflows to Agents - Second paragraph] The article states 'We will look at their core properties and how they are used, rather than their t...

#   

#   [93m----------------------------------------------------------------------------------------------------[0m

#     ARTICLE_PROFILE: 3 reviews

#   [93m----------------------------------------------------------------------------------------------------[0m

#     1. [Introduction: The Critical Decision Every AI Engineer Faces - First paragraph] The introduction, while stating the problem, is not as engaging and captivating as it could be, as h...

#     2. [Introduction: The Critical Decision Every AI Engineer Faces - First paragraph] The introduction is primarily focused on the 'why' (problem) and then states 'By the end of this les...

#   

#   [93m----------------------------------------------------------------------------------------------------[0m

#     HUMAN_FEEDBACK: 2 reviews

#   [93m----------------------------------------------------------------------------------------------------[0m

#     1. [Article level] Make the introduction more engaging and catchy....

#     2. [Understanding the Spectrum: From Workflows to Agents - First section] Expand on the definition of both workflows and agents from the first section....

#   

#   [93m----------------------------------------------------------------------------------------------------[0m

#     MECHANICS_PROFILE: 1 reviews

#   [93m----------------------------------------------------------------------------------------------------[0m

#     1. [Choosing Your Path - Fourth paragraph] The sentence 'If not designed well, there can be huge security concerns, especially with operations ...

#   

#   [93m----------------------------------------------------------------------------------------------------[0m

#     STRUCTURE_PROFILE: 3 reviews

#   [93m----------------------------------------------------------------------------------------------------[0m

#     1. [Choosing Your Path - Fourth paragraph] The paragraph 'Agents excel at dynamic problem-solving like open-ended research or complex customer ...

#     2. [Choosing Your Path - Fourth paragraph] The sentence 'Agents excel at dynamic problem-solving like open-ended research or complex customer s...

#   

#   [93m----------------------------------------------------------------------------------------------------[0m

#     TERMINOLOGY_PROFILE: 2 reviews

#   [93m----------------------------------------------------------------------------------------------------[0m

#     1. [Introduction: The Critical Decision Every AI Engineer Faces - First paragraph] The phrase 'critical decision that often determines if an AI application will be successful in produ...

#     2. [Choosing Your Path - First paragraph] The sentence 'The core difference between these two approaches lies in a single trade-off: developer...

#   

#   [93m----------------------------------------------------------------------------------------------------[0m

#     TONALITY_PROFILE: 1 reviews

#   [93m----------------------------------------------------------------------------------------------------[0m

#     1. [Introduction: The Critical Decision Every AI Engineer Faces - First paragraph] The opening statement 'When building AI applications, engineers face a critical architectural decisi...

#   


"""
Notice how the reviewer generated reviews from multiple sources:
- Human feedback reviews that directly address your specific requests
- Profile-based reviews (article, structure, mechanics, terminology, tonality) that ensure adherence to the style guidelines

The human feedback reviews always have `profile="human_feedback"` and create action points based on your feedback. These reviews will be used by the article writer to edit the article according to your instructions.
"""

"""
## 4. The Edit Article Workflow

Now that we understand how human feedback integrates with the article reviewer, let's explore the `edit_article` workflow. This workflow reviews and edits an existing article based on human feedback and the expected requirements.

The edit article workflow contains only one loop of the same reviewing-editing logic we already use within the generate article workflow. 

Also, the edit article workflow follows the same clean architecture pattern we've used throughout Brown. It leverages the app layer to orchestrate nodes and entities, keeping the code modular and maintainable.
"""

"""
### 4.1 Building the Edit Article Workflow

The workflow is built using LangGraph's functional API. Here's how it's structured:

Source: `brown.workflows.edit_article`
```python
def build_edit_article_workflow(checkpointer: BaseCheckpointSaver):
    """Create an edit article workflow with checkpointer.

    Args:
        checkpointer: Checkpointer to use for workflow persistence.

    Returns:
        Configured workflow entrypoint
    """

    return entrypoint(checkpointer=checkpointer)(_edit_article_workflow)
```

The `build_edit_article_workflow` function is a factory that creates the workflow with a checkpointer for persistence. It uses LangGraph's `@entrypoint` decorator to wrap the main workflow function.

The workflow expects an `EditArticleInput` typed dictionary:

```python
class EditArticleInput(TypedDict):
    dir_path: Path
    human_feedback: str
```

This input specifies:
- `dir_path`: The directory containing the article and all supporting files (guideline, profiles, research, etc.)
- `human_feedback`: The human feedback string to guide the editing process
"""

"""
### 4.2 The Edit Article Workflow Logic

The main workflow function orchestrates the entire editing process:
```python
async def _edit_article_workflow(inputs: EditArticleInput, config: RunnableConfig) -> str:
    writer = get_stream_writer()

    # Progress: Loading context
    writer(WorkflowProgress(progress=0, message="Loading context").model_dump(mode="json"))
    context = {}
    loaders = build_loaders(app_config)
    for context_name, loader in loaders.items():
        loader = cast(Loader, loader)
        context[context_name] = loader.load(working_uri=inputs["dir_path"])

    human_feedback = HumanFeedback(content=inputs["human_feedback"])
    writer(WorkflowProgress(progress=5, message="Loaded context").model_dump(mode="json"))

    # Progress: Reviewing
    writer(WorkflowProgress(progress=20, message="Reviewing article").model_dump(mode="json"))
    reviews = await generate_reviews(context["article"], human_feedback, context["article_guideline"], context["profiles"])
    writer(WorkflowProgress(progress=40, message="Generated reviews").model_dump(mode="json"))

    # Progress: Editing
    writer(WorkflowProgress(progress=60, message="Editing article").model_dump(mode="json"))
    article = await edit_based_on_reviews(
        context["article_guideline"], context["research"], context["profiles"], context["examples"], reviews
    )
    writer(WorkflowProgress(progress=80, message="Edited article").model_dump(mode="json"))

    # Progress: Complete
    writer(WorkflowProgress(progress=100, message="Article editing completed").model_dump(mode="json"))

    return f"""
Here is the edited article:
{article.to_context()}

Here is what you have to do with the edited article:
- print the edited article to the console for the user to see
- give a quick summary of the changes you made
- always apply the changes to the source file without waiting for the user to explicitly ask you to do so
- in case you haven't changed anything, just say that you haven't changed anything
"""
```

The workflow follows these steps:
1. **Load context**: Use the loader builders to load the article, guideline, profiles, research, and examples from the directory
2. **Create human feedback**: Convert the feedback string into a `HumanFeedback` entity
3. **Generate reviews**: Run the article reviewer with human feedback to generate reviews
4. **Edit based on reviews**: Run the article writer with the reviews to produce an edited article
5. **Return instructions**: Return the edited article along with instructions for the MCP client on what to do next

Notice how steps 3 and 4 are identical to the ones from the writing workflow you learned in lesson 23.
"""

"""
### 4.3 Generating Reviews

The `generate_reviews` task creates reviews by running the `ArticleReviewer` node:
```python
@task(retry_policy=retry_policy)
async def generate_reviews(
    article: Article,
    human_feedback: HumanFeedback,
    article_guideline: ArticleGuideline,
    article_profiles: ArticleProfiles,
) -> ArticleReviews:
    model, _ = build_model(app_config, node="review_article")
    article_reviewer = ArticleReviewer(
        to_review=article,
        article_guideline=article_guideline,
        article_profiles=article_profiles,
        human_feedback=human_feedback,
        model=model,
    )
    reviews = await article_reviewer.ainvoke()

    return cast(ArticleReviews, reviews)
```

This task:
- Builds the model from the app config for the "review_article" node
- Creates an `ArticleReviewer` with the article, guideline, profiles, and human feedback
- Uses LangGraph's `@task` decorator with a retry policy for resilience
"""

"""
### 4.4 Editing Based on Reviews

The `edit_based_on_reviews` task creates an edited article using the `ArticleWriter` node:
```python
@task(retry_policy=retry_policy)
async def edit_based_on_reviews(
    article_guideline: ArticleGuideline,
    research: Research,
    article_profiles: ArticleProfiles,
    article_examples: ArticleExamples,
    reviews: ArticleReviews,
) -> Article:
    model, _ = build_model(app_config, node="edit_article")
    article_writer = ArticleWriter(
        article_guideline=article_guideline,
        research=research,
        article_profiles=article_profiles,
        media_items=MediaItems.build(),
        article_examples=article_examples,
        reviews=reviews,
        model=model,
    )
    article = await article_writer.ainvoke()

    return cast(Article, article)
```

This task:
- Builds the model from the app config for the "edit_article" node
- Creates an `ArticleWriter` with all necessary context and the reviews to address
- Also uses the `@task` decorator with retry policy

Notice how the `ArticleWriter` works in "editing mode" when provided with `reviews`. It uses the same writer node from article generation, but the reviews guide it to make specific changes rather than writing from scratch.
"""

"""
### 4.6 Running the Edit Article Workflow
"""

import uuid

from brown.memory import build_in_memory_checkpointer
from brown.workflows.edit_article import build_edit_article_workflow

async with build_in_memory_checkpointer() as checkpointer:
    print("1. Building workflow...\n")
    workflow = build_edit_article_workflow(checkpointer=checkpointer)

    print("2. Configuring workflow...\n")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"   ✓ Thread ID: {thread_id}")

    print("3. Running workflow...")
    print("   This will take several minutes...\n")

    async for event in workflow.astream(
        {
            "dir_path": SAMPLE_DIR,
            "human_feedback": """
Make the introduction more engaging, catchy and shorter. 
Also, expand on the definition of both workflows and agents from the first section""",
        },
        config=config,
        stream_mode=["custom", "values"],
    ):
        event_type, event_data = event
        if event_type == "custom":
            pretty_print.wrapped(event_data, title="Event")
        elif event_type == "values":
            pretty_print.wrapped(event_data, title="Output")

pretty_print.wrapped("WORKFLOW COMPLETED", width=100)
# Output:
#   1. Building workflow...

#   

#   2. Configuring workflow...

#   

#      ✓ Thread ID: a8309001-6f52-48a0-b2b7-92f73d5f005e

#   3. Running workflow...

#      This will take several minutes...

#   

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 0,

#     "message": "Loading context"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 5,

#     "message": "Loaded context"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 20,

#     "message": "Reviewing article"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 40,

#     "message": "Generated reviews"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 60,

#     "message": "Editing article"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 80,

#     "message": "Edited article"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 100,

#     "message": "Article editing completed"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Output ----------------------------------------------[0m

#     

#   Here is the edited article:

#   

#   <article>

#       # Workflows vs. Agents: The Critical Decision Every AI Engineer Faces

#   ### How to choose between predictable control and autonomous flexibility when building AI applications.

#   

#   Every AI engineer faces a core architectural dilemma: build a predictable, step-by-step workflow, or create an autonomous agent that thinks and decides for itself? This choice impacts development time, costs, reliability, and user experience. It is a decision that shapes the success of an AI application in production.

#   

#   This lesson will help you understand the differences between LLM workflows and AI agents, know when to use each, and learn how to combine their strengths in hybrid approaches.

#   

#   ## Understanding the Spectrum: From Workflows to Agents

#   

#   To make the right choice, you first need to understand what LLM workflows and AI agents are. We will look at their core properties and how they are used, rather than their technical specifics.

#   

#   ### LLM Workflows

#   

#   An LLM workflow is a sequence of tasks orchestrated by developer-written code. It can include LLM calls, but also other operations like reading from a database or calling an API. For example, a workflow might take a user query, retrieve relevant documents from a database, summarize them with an LLM, and then send the summary to the user. The path is determined in advance, resulting in a deterministic or rule-based system. This gives you predictable execution, explicit control over the application's flow, and makes the system easier to test and debug. You know exactly where a failure occurred and how to fix it.

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

#   AI agents are systems where an LLM dynamically decides the sequence of steps, reasoning, and actions to achieve a goal. The path is not predefined. Instead, the agent uses a reasoning process to plan its actions based on the task and the current state of its environment. This process is often modeled on frameworks like ReAct (Reason, Act, Observe).

#   

#   For instance, an agent tasked with booking a flight might first search for available flights, then check prices, and finally present options to the user, adapting its steps if a flight is unavailable or too expensive. This allows agents to be adaptive and capable of handling new or unexpected situations through LLM-driven autonomy. They can select tools, execute actions, evaluate the outcomes, and correct their course until the goal is achieved [[1]](https://www.youtube.com/watch?v=kQxr-uOxw2o&t=1s).

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

#   Image 2: Flowchart illustrating an AI agent's dynamic decision-making process driven by an LLM.

#   

#   ## Choosing Your Path

#   

#   The core difference between these two approaches lies in a single trade-off: developer-defined logic versus LLM-driven autonomy [[2]](https://decodingml.substack.com/p/llmops-for-production-agentic-rag), [[3]](https://towardsdatascience.com/a-developers-guide-to-building-scalable-ai-workflows-vs-agents/). Workflows offer high reliability at the cost of flexibility, while agents offer high flexibility at the cost of reliability.

#   

#   https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5e64d5e0-7ef1-4e7f-b441-3bf1fef4ff9a_1276x818.png 

#   Image 3: The trade-off between an agent's level of control and application reliability. (Image by Iusztin, P. from [Exploring the difference between agents and workflows [2]](https://decodingml.substack.com/p/llmops-for-production-agentic-rag))

#   

#   ### When to use LLM workflows

#   

#   Workflows are ideal for repeatable tasks with defined steps. This includes pipelines for data extraction and transformation from sources like the web, Slack, Zoom calls, Notion, and Google Drive. They are also great for automated report or email generation from multiple data sources, repetitive daily tasks such as sending emails or posting social media updates, and content generation or repurposing, like transforming articles into social media posts. Their strength is predictability, ensuring reliable results, easier debugging, and lower costs by using specialized models. The main weakness is rigidity; they cannot handle unexpected scenarios, and adding features can become complex.

#   

#   ### When to use AI agents

#   

#   Agents excel at dynamic problem-solving. This includes open-ended research and synthesis, such as researching about World War II, dynamic problem-solving like debugging code or complex customer support, and interactive task completion in unfamiliar environments, like booking a flight without specifying exact sites. Their strength is flexibility in handling ambiguity. However, this autonomy makes them more prone to errors. Agents are non-deterministic, so performance, latency, and costs can vary with each call, making them unreliable. They require larger LLMs that generalize better, which are more costly. Agents also often need more LLM calls to understand user intent and take actions, increasing costs per call. If not designed well, there can be huge security concerns, especially with write operations, where an agent could delete data or send inappropriate emails. Ultimately, agents are hard to debug and evaluate.

#   

#   ### Hybrid Approaches

#   

#   Most real-world systems are not purely one or the other. They often blend elements of both, creating a hybrid system. In reality, we have a spectrum, a gradient between LLM workflows and AI agents, where a system adopts what is best from both worlds depending on its use cases. A common pattern is to use a workflow for predictable parts of a task and delegate ambiguous steps to an agent. For example, a system might use a human-in-the-loop workflow, where the agent proposes an action, and a human verifies it before execution.

#   

#   ```mermaid

#   graph TD

#       A["Human Input"] --> B["LLM Call (AI Generation)"]

#       B --> C["Action in Environment"]

#       C --> D["Feedback from Environment"]

#       D --> E{"Human Review/Verification"}

#       E -->|"Approved"| G["Stop/Final Output"]

#       E -->|"Rejected"| F["Continue/Refine"]

#       F --> A

#   ```

#   Image 4: A flowchart illustrating an AI generation and human verification loop with iterative refinement. (Source [Exploring the difference between agents and workflows [2]](https://decodingml.substack.com/p/llmops-for-production-agentic-rag))

#   

#   ## The Challenges of Every AI Engineer

#   

#   Understanding the spectrum from LLM workflows to AI agents is a core part of AI engineering. This choice helps determine if your application will succeed in production. Building AI systems means addressing recurring challenges daily. These include data integration, where you build pipelines to pull information from Slack, web APIs, SQL databases, and data lakes, ensuring only high-quality data reaches your AI system. Another challenge is the cost-performance trap, where sophisticated agents deliver impressive results but cost a lot per user interaction, making them economically unfeasible for many applications. Finally, security concerns arise when autonomous agents with powerful write permissions could send wrong emails, delete critical files, or expose sensitive data.

#   

#   In our next lesson, we will explore context engineering, a skill for building both workflows and agents.

#   

#   ## References

#   

#   1. Bouchard, L-F. (n.d.). *Real agents vs. workflows: The truth behind AI 'agents'*. YouTube. https://www.youtube.com/watch?v=kQxr-uOxw2o&t=1s

#   2. Iusztin, P. (n.d.). *Exploring the difference between agents and workflows*. Decoding AI Magazine. https://decodingml.substack.com/p/llmops-for-production-agentic-rag

#   3. (n.d.). *A developer’s guide to building scalable AI: Workflows vs agents*. Towards Data Science. https://towardsdatascience.com/a-developers-guide-to-building-scalable-ai-workflows-vs-agents/

#   4. Google. (n.d.). *Gemini CLI*. GitHub. https://github.com/google-gemini/gemini-cli/blob/main/README.md

#   </article>

#   

#   

#   Here is what you have to do with the edited article:

#   - print the edited article to the console for the user to see

#   - give a quick summary of the changes you made

#   - always apply the changes to the source file without waiting for the user to explicitly ask you to do so

#   - in case you haven't changed anything, just say that you haven't changed anything

#   

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------------------------------------------------------------------------[0m

#     WORKFLOW COMPLETED

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
As you can see, the workflow's output is the edited article. 

We will explain in more depth the impact of running the review-editing workflow alongside human feedback in the video at the end of the lesson.
"""

"""
### 4.6 The Power of Human Feedback

The edit article workflow demonstrates a key advantage of our architecture:

**We can use a low number of review loops during initial article generation, and further run them dynamically with a human in the loop when necessary, with more human guidance.**

This means:
- Initial generation is faster and cheaper with fewer automatic review iterations
- We don't assume how many iterations we need to have an ideal output, but let you decide
- The workflow runs additional review and editing cycles guided by your feedback
- You can repeat this process until satisfied with the results

This approach balances efficiency with quality, using AI to handle the heavy lifting while keeping you in control of the final output.
"""

"""
## 5. The Edit Selected Text Workflow

While the edit article workflow handles entire article edits, you'll often want to refine just a specific section. The `edit_selected_text` workflow enables precise, focused edits on selected text portions.

The workflow structure is almost identical to `edit_article`, thanks to our clean architecture. The main difference is that it operates on a `SelectedText` entity instead of the full `Article`.
"""

"""
### 5.1 Building the Edit Selected Text Workflow

The workflow builder follows the same pattern:

Source: `brown.workflows.edit_selected_text`
```python
def build_edit_selected_text_workflow(checkpointer: BaseCheckpointSaver):
    """Create an edit selected text workflow with checkpointer.

    Args:
        checkpointer: Checkpointer to use for workflow persistence.

    Returns:
        Configured workflow entrypoint
    """

    return entrypoint(checkpointer=checkpointer)(_edit_selected_text_workflow)
```

The workflow expects an `EditSelectedTextInput` typed dictionary:

```python
class EditSelectedTextInput(TypedDict):
    dir_path: Path
    human_feedback: str
    selected_text: str
    number_line_before_selected_text: int
    number_line_after_selected_text: int
```

This input specifies:
- `dir_path`: The directory containing the article and supporting files
- `human_feedback`: Human feedback to guide the editing
- `selected_text`: The specific text portion to edit
- `number_line_before_selected_text`: The starting line number in the article
- `number_line_after_selected_text`: The ending line number in the article

The line numbers help the workflow locate the selected text within the larger article context.
"""

"""
### 5.2 The Edit Selected Text Workflow Logic

The main workflow function is structurally similar to `edit_article`:
```python
async def _edit_selected_text_workflow(inputs: EditSelectedTextInput, config: RunnableConfig) -> str:
    writer = get_stream_writer()

    # Progress: Loading context
    writer(WorkflowProgress(progress=0, message="Loading context").model_dump(mode="json"))
    context = {}
    loaders = build_loaders(app_config)
    for context_name, loader in loaders.items():
        loader = cast(Loader, loader)
        context[context_name] = loader.load(working_uri=inputs["dir_path"])

    selected_text = SelectedText(
        article=context["article"],
        content=inputs["selected_text"],
        first_line_number=inputs["number_line_before_selected_text"],
        last_line_number=inputs["number_line_after_selected_text"],
    )
    human_feedback = HumanFeedback(content=inputs["human_feedback"])
    writer(WorkflowProgress(progress=5, message="Loaded context").model_dump(mode="json"))

    # Progress: Reviewing
    writer(WorkflowProgress(progress=20, message="Reviewing selected text").model_dump(mode="json"))
    reviews = await generate_reviews(selected_text, human_feedback, context["article_guideline"], context["profiles"])
    writer(WorkflowProgress(progress=40, message="Generated reviews").model_dump(mode="json"))

    # Progress: Editing
    writer(WorkflowProgress(progress=60, message="Editing selected text").model_dump(mode="json"))
    selected_text = await edit_based_on_reviews(
        context["article_guideline"], context["research"], context["profiles"], context["examples"], reviews
    )
    writer(WorkflowProgress(progress=80, message="Edited selected text").model_dump(mode="json"))

    # Progress: Complete
    writer(WorkflowProgress(progress=100, message="Selected text editing completed").model_dump(mode="json"))

    return f"""
Here is the edited selected text:
{selected_text.to_context()}

Here is what you have to do with edited selected text:
- print the edited selected text to the console for the user to see
- give a quick summary of the changes you made
- always apply the changes to the source file without waiting for the user to explicitly ask you to do so
- in case you haven't changed anything, just say that you haven't changed anything
"""
```

The workflow follows these steps:
1. **Load context**: Load the full article and supporting files
2. **Create selected text entity**: Build a `SelectedText` entity that contains the selected portion, the full article for context, and line numbers
3. **Create human feedback**: Convert the feedback string to a `HumanFeedback` entity
4. **Generate reviews**: Review the selected text with human feedback
5. **Edit based on reviews**: Edit the selected text based on the reviews
6. **Return instructions**: Return the edited selected text with instructions
"""

"""
### 5.3 Generating Reviews for Selected Text

The `generate_reviews` task for selected text is nearly identical to the article version:
```python
@task(retry_policy=retry_policy)
async def generate_reviews(
    selected_text: SelectedText,
    human_feedback: HumanFeedback,
    article_guideline: ArticleGuideline,
    article_profiles: ArticleProfiles,
) -> SelectedTextReviews:
    model, _ = build_model(app_config, node="review_selected_text")
    selected_text_reviewer = ArticleReviewer(
        to_review=selected_text,
        human_feedback=human_feedback,
        article_guideline=article_guideline,
        article_profiles=article_profiles,
        model=model,
    )
    reviews = await selected_text_reviewer.ainvoke()

    return cast(SelectedTextReviews, reviews)
```

The key difference is:
- It takes a `SelectedText` instead of `Article`
- It returns `SelectedTextReviews` instead of `ArticleReviews`
- It uses the "review_selected_text" node config

The `ArticleReviewer` node is smart enough to handle both cases. When given a `SelectedText`, it focuses reviews on that portion while using the full article as context.
"""

"""
### 5.4 Editing Selected Text Based on Reviews

The `edit_based_on_reviews` task for selected text also follows the same pattern:
```python
@task(retry_policy=retry_policy)
async def edit_based_on_reviews(
    article_guideline: ArticleGuideline,
    research: Research,
    article_profiles: ArticleProfiles,
    article_examples: ArticleExamples,
    reviews: SelectedTextReviews,
) -> SelectedText:
    model, _ = build_model(app_config, node="edit_selected_text")
    article_writer = ArticleWriter(
        article_guideline=article_guideline,
        research=research,
        article_profiles=article_profiles,
        media_items=MediaItems.build(),
        article_examples=article_examples,
        reviews=reviews,
        model=model,
    )
    edited_selected_text = cast(SelectedText, await article_writer.ainvoke())

    return edited_selected_text
```

This task:
- Takes `SelectedTextReviews` instead of `ArticleReviews`
- Returns `SelectedText` instead of `Article`
- Uses the "edit_selected_text" node config

Again, the `ArticleWriter` node handles both article and selected text editing seamlessly.
"""

"""
### 5.5 Why Edit Selected Text?

The edit selected text workflow is crucial because:

**Most often we don't want to edit the whole article, but just a small section, or apply the human feedback just to a small section.**

This workflow enables:
- Faster and cheaper edits by focusing on specific sections
- More precise changes without affecting other parts of the article
- Iterative refinement of individual paragraphs or sections
- Better control over the editing process

Combined with the edit article workflow, you have complete flexibility to refine content at any granularity you need.
"""

"""
### 5.6 Running the Edit Selected Text Workflow

First, explictly load the selected text that we want to edit from the sample article:
"""

article = MarkdownArticleLoader(uri="article.md").load(working_uri=SAMPLE_DIR)

start_line = 8
end_line = 42
selected_text = "\n".join(article.content.split("\n")[start_line:end_line])
pretty_print.wrapped(selected_text, title="Selected text to edit")
# Output:
#   [93m-------------------------------------- Selected text to edit --------------------------------------[0m

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

#   Image 2: Flowchart illustrating an AI agent's dynamic decision-making process driven by an LLM.

#   

#   ## Choosing Your Path

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Then, call the workflow:
"""

from brown.workflows.edit_selected_text import build_edit_selected_text_workflow

async with build_in_memory_checkpointer() as checkpointer:
    print("1. Building workflow...\n")
    workflow = build_edit_selected_text_workflow(checkpointer=checkpointer)

    print("2. Configuring workflow...\n")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"   ✓ Thread ID: {thread_id}")

    print("3. Running workflow...")
    print("   This will take several minutes...\n")
    async for event in workflow.astream(
        {
            "dir_path": SAMPLE_DIR,
            "human_feedback": "Expand on the definition of both workflows and agents.",
            "selected_text": selected_text,
            "number_line_before_selected_text": start_line,
            "number_line_after_selected_text": end_line,
        },
        config=config,
        stream_mode=["custom", "values"],
    ):
        event_type, event_data = event
        if event_type == "custom":
            pretty_print.wrapped(event_data, title="Event")
        elif event_type == "values":
            pretty_print.wrapped(event_data, title="Output")

pretty_print.wrapped("WORKFLOW COMPLETED", width=100)
# Output:
#   1. Building workflow...

#   

#   2. Configuring workflow...

#   

#      ✓ Thread ID: f281995f-fef6-4e9f-8887-9cedc09c0710

#   3. Running workflow...

#      This will take several minutes...

#   

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 0,

#     "message": "Loading context"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 5,

#     "message": "Loaded context"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 20,

#     "message": "Reviewing selected text"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 40,

#     "message": "Generated reviews"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 60,

#     "message": "Editing selected text"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 80,

#     "message": "Edited selected text"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Event ----------------------------------------------[0m

#     {

#     "progress": 100,

#     "message": "Selected text editing completed"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m---------------------------------------------- Output ----------------------------------------------[0m

#     

#   Here is the edited selected text:

#   

#   <selected_text>

#       

#       <content>To make the right choice, you first need to understand what LLM workflows and AI agents are. We will look at their core properties and how they are used, rather than their technical specifics.

#   

#   ### LLM Workflows

#   

#   An LLM workflow is a sequence of tasks orchestrated by developer-written code. It combines LLM calls with other operations like reading from a database or calling an API. Each step is explicitly defined, much like a recipe. The path is predefined, resulting in a deterministic, rule-based system, similar to classic programming.

#   

#   This offers predictable execution and explicit control over the application's flow. It makes the system easier to test and debug, as you know exactly where a failure occurred and how to fix it.

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

#   AI agents are systems where an LLM dynamically decides the sequence of steps, reasoning, and actions to achieve a goal. The path is not predefined. The agent uses a reasoning process to plan its actions based on the task and environment. This is often modeled on frameworks like ReAct, which cycles through Reason, Act, and Observe.

#   

#   Agents are adaptive and handle new situations through LLM-driven autonomy. They strategize, break down tasks, and plan steps, acting like an intelligent assistant. They select tools, execute actions, evaluate outcomes, and correct their course until the goal is achieved [[1]](https://www.youtube.com/watch?v=kQxr-uOxw2o&t=1s).

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

#   Image 2: Flowchart illustrating an AI agent's dynamic decision-making process driven by an LLM.

#   

#   ## Choosing Your Path</content>

#       <first_line_number>8</first_line_number>

#       <last_line_number>42</last_line_number>

#   </selected_text>

#   

#   

#   Here is what you have to do with edited selected text:

#   - print the edited selected text to the console for the user to see

#   - give a quick summary of the changes you made

#   - always apply the changes to the source file without waiting for the user to explicitly ask you to do so

#   - in case you haven't changed anything, just say that you haven't changed anything

#   

#   [93m----------------------------

[... Content truncated due to length ...]

</details>


## YouTube Video Transcripts

<details>
<summary>[00:00] (Music plays as the screen displays "Y Combinator Presents AI STARTUP SCHOOL" with a geometric pattern in shades of teal and grey)</summary>

[00:00] (Music plays as the screen displays "Y Combinator Presents AI STARTUP SCHOOL" with a geometric pattern in shades of teal and grey)

[00:01] Please welcome former director of AI, Tesla, Andrej Karpathy.

[00:06] (Andrej walks onto a circular stage in front of a large screen, to applause. The screen now shows a black and white photo of Andrej smiling, with the title "Software in the era of AI" and his name "Andrej Karpathy")

[00:10] (Andrej smiles, walks across the stage, waves to the audience, and gets ready to speak. Many in the audience are holding up phones, recording)

[00:17] Wow, a lot of people here. Hello.

[00:23] Um, okay. Yeah, so I'm excited to be here today to talk to you about Software in the era of AI. (The screen changes to a panoramic photo of the Golden Gate Bridge and a lavender field, with the title "Software in the era of AI" and "Andrej Karpathy YC AI Startup School June 18")

[00:30] And I'm told that many of you are students, like bachelors, masters, PhD and so on, and you're about to enter the industry. And I think it's actually like an extremely unique and very interesting time to enter the industry right now.

[00:41] And I think fundamentally the reason for that is that, um, software is changing, uh, again. (The slide changes to a white background with black text: "Software is changing. (again)")

[00:47] And I say again because I actually gave this talk already, um, but the problem is that software keeps changing, so I actually have a lot of material to create new talks. And I think it's changing quite fundamentally. I think roughly speaking, software has not changed much on such a fundamental level for 70 years, and then it's changed, I think about twice quite rapidly in the last few years. And so there's just a huge amount of work to do, a huge amount of software to write and rewrite.

[01:13] So let's take a look at maybe the realm of software. So if we kind of think of this as like the map of software, this is a really cool tool called Map of GitHub. (The slide changes to a dark blue background with scattered blue shapes, labeled "Map of GitHub". Each shape is a cluster of smaller, interconnected blue nodes, representing software repositories)

[01:21] Um, this is kind of like all the software that's written. Uh, these are instructions to the computer for carrying out tasks in the digital space. So if you zoom in here, (The slide zooms into one of the blue shapes, revealing a dense network of tiny interconnected pink and blue nodes, labeled "Phosphoria") these are all different kinds of repositories, and this is all the code that has been written.

[01:32] And a few years ago, I kind of observed that, um, software was kind of changing and there was kind of like a new new type of software around, and I called this software 2.0 at the time. (The slide changes to a white background with "Software 2.0" at the top. Below, "Software 1.0 = code" is next to a screenshot of computer code. "Software 2.0 = weights" is next to a diagram of a neural network)

[01:43] And the idea here was that, uh, Software 1.0 is the code you write for the computer. Software 2.0 are basically neural networks, and in particular, the weights of a neural network. And you're not writing this code directly, you are most you are more kind of like tuning the data sets and then you're running an optimizer to create to create the parameters of this neural net.

[02:00] And I think like at the time, neural nets were kind of seen as like just a different kind of classifier, like a decision tree or something like that. And so I think, uh, it was kind of like, um, I I think this framing was a lot more appropriate. And now actually what we have is kind of like an equivalent of GitHub in the realm of Software 2.0. And I think, uh, the HuggingFace, uh, is basically equivalent of GitHub in Software 2.0. (The slide changes, now showing the "Map of GitHub" (Software 1.0) on the left, and a "HuggingFace Model Atlas" (Software 2.0) on the right. The HuggingFace atlas is a dark background with vibrant, complex, starburst-like clusters of interconnected nodes in pink, green, and yellow)

[02:20] And there's also Model Atlas and you can visualize all the code written there. In case you're curious, by the way, the giant circle, the point in the middle, uh, these are the parameters of Flux, the image generator.

[02:32] And so any time someone tunes a Laura on top of a Flux model, you basically create a Git commit, uh, in this space, and, uh, you create a different kind of a image generator. So basically what we have is Software 1.0 is the computer code that programs a computer. (The slide changes to white with a flowchart for "Software 1.0: computer code -> programs -> computer" and an image of a person using an old computer terminal. It then animates to add "Software 2.0: weights -> programs -> neural net" with a diagram of AlexNet)

[02:46] Software 2.0 are the weights which program neural networks. Uh, and here's an example of AlexNet image recognizer neural network. Now, so far, all of the neural networks that we've been familiar with until recently, were kind of like fixed function computers.

[03:00] Image to categories or something like that. And I think what's changed, and I think it's quite a fundamental change, is that neural networks became programmable with large language models. (The slide animates to add "Software 3.0: prompts -> programs -> LLM" with a flowchart diagram of an LLM architecture)

[03:10] And so I I see this as quite new, unique, it's new kind of a computer, and, uh, so in my mind, it's, uh, worth giving it a new designation of Software 3.0. And basically, your prompts are now programs that program the LLM. And, uh, remarkably, uh, these, uh, prompts are written in English.

[03:30] So it's kind of a very interesting programming language. Um, so maybe, uh, to, uh, summarize the difference, if you're doing sentiment classification, for example, you can imagine, uh, writing some, uh, amount of Python to to basically do sentiment classification, or you can train an neural net, or you can prompt a large language model. (The slide changes to "Example: Sentiment Classification", showing three columns: Software 1.0 (Python code), Software 2.0 (training binary classifier with positive/negative examples), and Software 3.0 (a few-shot prompt for sentiment classification))

[03:49] Uh, so here I'm this is a few shot prompt and you can imagine changing it and programming the computer in a slightly different way. So basically we have Software 1.0, Software 2.0, (The slide changes back to the comparison of GitHub, HuggingFace Model Atlas, now with a new yellow region for "Software 3.0" and arrows pointing outward from it, labeled "LLM prompts, in English")

[03:59] and I think we're seeing I maybe you've seen a lot of GitHub code is not just like code anymore, there's a bunch of like English interspersed with code. And so I think kind of there's a growing category of new kind of code. So not only is it a new programming paradigm, it's also remarkable to me that it's in our native language of English.

[04:14] And so when this blew my mind, uh, a few, uh, I guess years ago now, uh, I tweeted this and, um, I think it captured the attention of a lot of people and this is my currently pinned tweet. (The slide changes to a screenshot of Andrej Karpathy's pinned tweet from Jan 24, 2023: "The hottest new programming language is English" with 1.1K likes, 7K retweets, 44K bookmarks, and 7.4M views)

[04:26] Uh, is that remarkably, we're now programming computers in English. Now, when I was at, uh, Tesla, (The slide changes to "Software is eating the world. Software 2.0 eating Software 1.0" with a diagram of the Tesla Autopilot software stack. On the left, a pink box labeled "1.0 code" has a shrinking blue area labeled "2.0 code" at the bottom. On the right, images from car cameras feed into a complex neural network architecture ending in "Bird's eye view predictions")

[04:33] um, we were working on the, uh, autopilot, and, uh, we were trying to get the car to drive. And I sort of showed this slide at the time where you can imagine that the inputs to the car are on the bottom and they're going through a software stack to produce the steering and acceleration.

[04:47] And I made the observation at the time that there was a ton of C++ code around in the autopilot, which was the Software 1.0 code, and then there was some neural nets in there doing image recognition. And, uh, I kind of observed that over time as we made the autopilot better, basically, the neural network grew in capability and size, and in addition to that, all the C++ code was being deleted and kind of like was, um, and a lot of the kind of capabilities and functionality that was originally written in 1.0 was migrated to 2.0. So as an example, a lot of the stitching up of information across images from the different cameras and across time was done by neural network, and we were able to delete a lot of code.

[05:27] And so the Software 2.0 stack would quite literally ate through the software stack of the autopilot. So I thought this was really remarkable at the time. And I think we're seeing the same thing again, where, uh, basically we have a new kind of software, and it's eating through the stack. We have three completely different programming paradigms. (The slide changes to "A huge amount of Software will be (re-)written." with a diagram of a square representing all software, divided into shrinking "1.0" (red) and growing "2.0" (blue) and "3.0" (yellow) regions, with arrows pointing from 2.0 and 3.0 into 1.0)

[05:45] And I think if you're entering the industry, it's a very good idea to be fluent in all of them, because they all have slight pros and cons, and you may want to program some functionality in 1.0 or 2.0 or 3.0. Are you going to train a neural net? Are you going to just prompt an LLM? Should this be a piece of code that's explicit, et cetera. So we all have to make these decisions and actually potentially, uh, fluidly transition between these paradigms.

[06:07] So, what I want to get into now is, (The slide changes to white with black text: "Part 1 How to think about LLMs")

[06:11] first, I want to in the first part talk about LLMs and how to kind of like think of this new paradigm and the ecosystem and what that looks like. Like what are what is this new computer? What does it look like? And what does the ecosystem look like?

[06:22] Um, I was struck by this quote from Andrew actually, uh, many years ago now, I think. And I think Andrew is going to, uh, be speaking right after me. (The slide changes to white with black text: ""AI is the new electricity" -Andrew Ng")

[06:30] Uh, but he said at the time, AI is the new electricity. And I do think that it, um, kind of captures something very interesting in that LLMs certainly feel like they have properties of utilities right now. So, (The slide changes to white with black text: "LLMs have properties of utilities..." and bullet points. On the right, an image of a large electrical substation at sunset)

[06:42] um, LLM Labs, like OpenAI, Gemini, Anthropic, et cetera, they spend CAPEX to train the LLMs, and this is kind of equivalent to building out a grid. And then there's OPEX to serve that intelligence over APIs to all of us. And this is done through metered access where we pay per million tokens or something like that. And we have a lot of demands that are very utility-like demands out of this API. We demand low latency, high uptime, consistent quality, et cetera.

[07:08] In electricity, you would have a transfer switch, so you can transfer your electricity source from like grid and solar or battery or generator. In LLMs, we have maybe OpenRouter and easily switch between the different types of LLMs that exist. (The slide animates to highlight the bullet points: "- CAPEX to train an LLM (~= to build the grid)", "- OPEX to serve intelligence over increasingly homogeneous API (prompt, image, tools, ...)", "- Metered access ($/1M tokens)", "- Demand for low latency, high uptime, consistent quality (~= demanding consistent voltage from grid)", "- OpenRouter ~= Transfer Switch (grid, solar, battery, generator...)". A graph below the substation image is labeled "LLM Rankings OpenRouter" showing an upward trend)

[07:22] Because the LLMs are software, they don't compete for physical space. So it's okay to have basically like six electricity providers, and you can switch between them, right? Uh, because they don't compete in such a direct way.

[07:31] And I think what's also a little fascinating, and we saw this in the last few days actually, a lot of the LLMs went down, and people were kind of like stuck and unable to work. (The slide animates to add a bullet point: "- Intelligence "brownouts" e.g. when OpenAI goes down.")

[07:40] And I think it's kind of fascinating to me that when the state of the art LLMs go down, it's actually kind of like an intelligence brownout in the world. It's kind of like when the voltage is unreliable in the grid, and, uh, the planet just gets dumber, the more reliance we have on these models, which already is like really dramatic, and I think we'll continue to grow.

[08:00] But LLMs don't only have properties of utilities. I think it's also fair to say that they have some properties of fabs. (The slide changes to white with black text: "LLMs have properties of fabs..." and bullet points. On the right, aerial images of large industrial complexes, possibly chip fabs, and server racks)

[08:05] And the reason for this is that the CAPEX required for building LLMs is actually quite large. Uh, it's not just like building some, uh, power station or something like that, right? Uh, you're investing a huge amount of money, and I think the tech tree, and for the technology, is growing quite rapidly. (The slide animates to highlight bullet points: "- Huge CAPEX", "- Deep tech tree R&D, secrets", "- 4nm process node ~= 10^20 FLOPS cluster")

[08:24] So we're in a world where we have sort of deep tech trees, research and development, secrets that are centralizing inside the LLM Labs. Um, and I but I think the analogy muddies a little bit also because as I mentioned, this is software, and software is a bit less defensible, uh, because it is so malleable. (The slide animates to add bullet points: "- Anyone training on NVIDIA GPUs ~= fabless", "- Google training on TPUs ~= owns fab (e.g. Intel)")

[08:38] And so, um, I think it's just an interesting kind of thing to think about potentially. There's many analogies you can make. Like a 4nm process node maybe is something like a cluster with certain max flops. (The slide animates to add the text: "e.g. xAI Colossus cluster (100K H100 GPUs)")

[08:52] You can think about when you're using when you're using Nvidia GPUs and you're only doing the software, and you're not doing the hardware, that's kind of like the fabless model. Uh, but if you're actually also building your own hardware, and you're training on TPUs if you're Google, that's kind of like the Intel model where you own your fab. So I think there's some analogies here that make sense.

[09:07] But actually I think the analogy that makes the most sense perhaps is that in my mind, LLMs have very strong kind of analogies to operating systems. (The slide changes to white with black text: "LLMs have properties of Operating Systems..." and bullet points. On the right, logos of Windows, MacOS, and various Linux distributions)

[09:16] Uh, in that, this is not just electricity or water. It's not something that comes out of a tap as a commodity. Uh, this is these are now increasingly complex software ecosystems. Right? So, uh, they're not just like simple commodities like electricity. (The slide animates to highlight bullet points: "- LLMs are increasingly complex software ecosystems, not simple commodities like electricity.", "- LLMs are Software. Trivial to copy & paste, manipulate, change, distribute, open source, steal..., not physical infrastructure.")

[09:31] And it's kind of interesting to me that the ecosystem is shaping in a very similar kind of way where you have a few closed source providers, like Windows or Mac OS. And then you have an open source alternative like Linux. And I think for, uh, neural for LLMs as well, we have a kind of a few competing closed source providers. (The slide animates to show a network diagram of LLMs and their interconnections, labelled at the bottom with various LLM logos like OpenAI, Anthropic, Google, and others. It also highlights a bullet point: "- Some amount of switching friction due to different features, performance, style, capabilities etc. per domain.")

[09:48] And then maybe the Llama ecosystem is currently like maybe a close approximation to something that may grow into something like Linux. Again, I think it's still very early because these are just simple LLMs, but we're starting to see that these are going to get a lot more complicated. It's not just about the LLM itself, it's about the tool use and the multimodalties and how all of that works. (The slide animates to highlight "- System/user (prompt) space ~= kernel/user (memory) space")

[10:07] And so when I sort of had this realization a while back, I tried to sketch it out, and it kind of seemed to me like LLMs are kind of like a new operating system, right? So the LLM is a new kind of a computer. (The slide changes to a white background with a diagram labeled "LLM OS". It shows a central "LLM" box connected to "CPU", "RAM (context window)", "Disk", "File system (+embeddings)", "Software 1.0 tools ("classical computer")" like Calculator, Python interpreter, Terminal, and Peripheral devices I/O (Video, Audio), Ethernet, Browser, Other LLMs)

[10:17] It's sitting, it's kind of like the CPU equivalent. Uh, the context windows are kind of like the memory, and then the LLM is orchestrating memory and compute, uh, for problem solving, um, using all of these abilities here. And so definitely, if you look at it, it looks very much like operating system from that perspective. Um, a few more analogies. (The slide changes to white with black text: "You can run an app like VS Code on: - Windows 10, 11 - Mac 10.15 - Linux" and an image showing download options for VS Code on Windows, Linux, and Mac)

[10:40] For example, if you want to download an app, say I go to VS Code and I go to download. You can download VS Code and you can run it on Windows, Linux, or Mac. In the same way, as you can take an LLM app like Cursor, (The slide animates to add text: "Just like you can run an LLM app like Cursor on: - GPT o3 - Claude 4-sonet - Gemini 2.5-pro - DeepSeek" and a screenshot of Cursor's chat interface showing a dropdown to select different LLM models)

[10:52] and you can run it on GPT or Claude or Gemini series, right? There's just a drop down. So it's kind of like similar in that way as well. More analogies that I think strike me is that we're kind of like in this 1960s-ish era, (The slide changes to "1950s - 1970s time-sharing era" with images of old mainframe computers and people using terminals. Text lists characteristics: Centralized, expensive computers => OS runs in the cloud, I/O is streamed, compute is batched over users)

[11:07] where LLM compute is still very expensive for this new kind of a computer. And that forces the LLMs to be centralized in the cloud, and we're all just, uh, sort of thin clients that interact with it over the network. And none of us have full utilization of these computers, and therefore, it makes sense to use time sharing, where we're all just, you know, a dimension of the batch when they're running the computer in the cloud.

[11:31] And this is very much what computers used to look like at during this time. The operating systems were in the cloud, everything was streamed around and there was batching. And so the personal computing revolution hasn't happened yet because it's just not economical, it doesn't make sense, but I think some people are trying. (The slide animates to highlight text about personal computing not happening yet)

[11:47] And it turns out that Mac Minis, for example, are a very good fit for some of the LLMs because it's all if you're doing batch one inference, this is all super memory bound, so this actually works. (The slide changes to "Early hints of Personal Computing v2". Two tweets are displayed: one showing stacked Mac Minis used for Llama inference, the other a similar setup for running DeepSeek)

[11:57] And, uh, I think these are some early indications maybe of personal computing, but this hasn't really happened yet. It's not clear what this looks like, maybe some of you get to invent what what this is, or how it works, or, uh, what this should what this should be.

[12:10] Maybe one more analogy that I'll mention is whenever I talk to ChatGPT or some LLM directly in text, I feel like I'm talking to an operating system through the terminal. (The slide changes to a meme format: "Corporate needs you to find the differences between this picture and this picture." On the left, an old computer terminal. On the right, a ChatGPT interface. Below, the text: "They're the same picture." Other text: "(text) chat ~= terminal, direct/native access to the OS. GUI hasn't been invented yet. (~1970)")

[12:19] Like it just it's it's text, it's direct access to the operating system, and I think a GUI hasn't yet really been invented in like a general way. Like, should ChatGPT have a GUI? What different than just the tech bubbles? Uh, certainly some of the apps that we're going to go into in a bit have GUI, but there's no like GUI across all the tasks that make sense.

[12:42] Um, there are some ways in which LLMs are different from kind of operating systems in some fairly unique way and from early computing. And I wrote about, uh, this one particular property that strikes me as very different. (The slide changes to a meme-like image of three-headed dragon representing "Government", "Corporations", and "Consumer". The "Consumer" head has a goofy expression. Text: "Power to the people: How LLMs flip the script on technology diffusion" and a blog post link.)

[12:54] Uh, this time around. It's that LLMs like flip, they flip the direction of technology diffusion that is usually, uh, present in technology. So for example, with electricity, cryptography, computing, flight, internet, GPS, lots of new transformative technologies that have not been around. Typically, it is the government and corporations that are the first users because it's new and expensive, et cetera, and it only later diffuses to consumer. (The slide animates to show an old diagram of military ballistics next to the "Government" head and a person boiling an egg next to the "Consumer" head labeled "Hi ChatGPT how to boil egg?")

[13:20] Uh, but I feel like LLMs are kind of like flipped around. So maybe with early computers, it was all about ballistics and military use, but with LLMs, it's all about how do you boil an egg or something like that? This is certainly a lot of my use. So it's really fascinating to me that we have a new magical computer, and it's like helping me boil an egg. It's not helping the government do something really crazy like some military ballistics or some special technology.

[13:42] Indeed, corporations and governments are lagging behind the adoption of all of us of all of these technologies. So it's just backwards, and I think it informs maybe some of the uses of how we want to use this technology or like what are some of the first apps and so on. (The slide zooms out to show the full meme again)

[13:56] So in summary so far, LLM Labs, (The slide changes to white with black text: "Part 1 Summary" and bullet points: "LLM labs: - Fab LLMs - LLMs ~= Operating Systems (circa 1960s) - Available via time-sharing, distributed like utility. NEW: Billions of people have sudden access to them! It is our time to program them.")

[14:00] I think accurate language to use, but LLMs are complicated operating systems. There circa 1960s in computing and we're redoing computing all over again. And they're currently available via time sharing and distributed like a utility. What is new and unprecedented is that they're not in the hands of a few governments and corporations, they're in the hands of all of us because we all have a computer and it's all just software. And ChatGPT was beamed down to our computers like to billions of people like instantly and overnight, and this is insane. (The slide animates to show a cartoon Andrej Karpathy on the right, gesturing as if beaming down ChatGPT)

[14:29] Uh, and it's kind of insane to me that this is the case, and now it is our time to enter the industry and program these computers. This is crazy. So I think this is quite remarkable. (Andrej smiles, walks off stage to applause. Music plays again. The screen changes to white with black text: "Part 2 LLM Psychology")

[14:38] Before we program LLMs, we have to kind of like spend some time to think about what these things are. And I especially like to kind of talk about their psychology. So, the way I like to think about LLMs is that they're kind of like people spirits. Um, they are stochastic simulations of people. (The slide changes to a black background with an abstract, glowing blue human figure sitting thoughtfully, intertwined with data streams. Text: "LLMs are "people spirits": stochastic simulations of people. Simulator = autoregressive Transformer" and a Transformer model diagram. Below: "=> They have a kind of emergent "psychology".")

[15:02] Um, and the simulator in this case happens to be an autoregressive Transformer. So Transformer is a neural net. Uh, it's and it just kind of like is goes on the level of tokens. It goes chunk, chunk, chunk, chunk. And there's an almost equal amount of compute for every single chunk. Um, and, um, this simulator, of course, is is just is basically there's some weights involved, and we fit it to all of text that we have on the internet and so on. And you end up with this kind of a simulator. And because it is trained on humans, it's got this emergent psychology that is human-like. So the first thing you'll notice is, of course, (The slide changes to a warm library setting. A young man with glasses sits at a desk, surrounded by stacks of books, looking intently at an open book. Text: "Encyclopedic knowledge/memory, ..." A movie poster for "Rain Man" is also displayed)

[15:32] LLMs have encyclopedic knowledge and memory, and they can remember lots of things, a lot more than any single individual human can because they've read so many things. It's it actually kind of reminds me of this movie Rain Man, which I actually really recommend people watch, it's an amazing movie, I love this movie. Um, and Dustin Hoffman here is an autistic savant, who has almost perfect memory. So he can read it a he can read like a phone book and remember all of the names and, uh, phone numbers. And I kind of feel like LLMs are kind of like very similar. They can remember SHA hashes and lots of different kinds of things very, very easily. So they certainly have superpowers in some set in some respects. But they also have a bunch of, I would say, cognitive deficits. (The slide changes to a photo of a glowing cat hologram next to a young man reading in a library. Text: "Hallucinations". Below, another photo of a stressed student looking at '2+2=5' on a whiteboard. Text: "Jagged intelligence. Famous examples: 9.11 > 9.9, two 'r' in 'strawberry', ...")

[16:11] So they hallucinate quite a bit, um, and they kind of make up stuff and don't have a very good, uh, sort of internal model of self-knowledge, not sufficient at least. And this has gotten better, but not perfect. They display jagged intelligence. So they're going to be superhuman in some problem solving domains. And then they're going to make mistakes that basically no human will make, like, you know, they will insist that 9.11 is greater than 9.9 or there are two Rs in strawberry, these are some famous examples. But basically, there are rough edges that you can trip on. So that's kind of I think, also kind of unique. (The slide changes to a worried young man looking at a paper that says "What did you eat for breakfast?". Text: "Anterograde amnesia. Context windows ~= working memory. No continual learning, no equivalent of "sleep" to consolidate knowledge, insight or expertise into weights.")

[16:42] Um, they also kind of suffer from anterograde amnesia. Um, so, uh, and I think I'm alluding to the fact that if you have a co-worker who joins your organization, this co-worker will over time learn your organization and they will understand and gain like a huge amount of context on the organization, and they go home and they sleep, and they consolidate knowledge and they develop expertise over time. LLMs don't natively do this, and this is not something that has really been solved in the R&D of LLMs, I think. (The slide changes to movie posters for "Memento" and "50 First Dates")

[17:08] Um, and so context windows are really kind of like working memory, and you have to sort of program the working memory quite directly because they don't just kind of like get smarter by, uh, by default. And I think a lot of people get tripped up by the analogies, uh, in this way. In popular culture, I recommend people watch these two movies, uh, Memento and 50 First Dates. (The slide changes to a young man with a stack of books, one labeled "TRUST ME". Text: "Gullibility => Prompt injection risks, e.g. of private data")

[17:29] In both of these movies, the protagonists, their weights are fixed, and their context windows get wiped every single morning, and it's really problematic to go to work or have relationships when this happens, and this happens to LLMs all the time. I guess one more thing I would point to is security kind of related limitations of the use of LLMs. So, for example, LLMs are quite gullible. Uh, they are susceptible to prompt injection risks. They might leak your data, et cetera. And so, um, and there's many other considerations, uh, security related. So, (The slide changes to a collage of images from previous slides, with the text: "Part 2 Summary LLM Psychology Kind of a lossy simulation of a savant with cognitive issues.")

[17:55] So basically, long story short, you have to load your you have to load your That is simultaneously think through this superhuman thing that has a bunch of cognitive deficits and issues. How do we and yet, they are extremely like useful. And so how do we program them, and how do we work around their deficits and enjoy their superhuman powers. (The slide changes to white with black text: "Part 3 Opportunities")

[18:16] So what I want to switch to now is talk about the opportunities of how do we use these models, and what are some of the biggest opportunities. This is not a comprehensive list, just some of the things that I thought were interesting for this talk. The first thing I'm kind of excited about is what I would call partial autonomy apps. (The slide animates to add: "Partial autonomy apps "Copilot" / "Cursor for X"")

[18:30] For example, let's work with the example of coding. You can certainly go to ChatGPT directly, and you can start copy pasting code around, and copy pasting bug reports and stuff around, and getting code and copy pasting everything around. (The slide changes to a ChatGPT interface and an old computer terminal. The ChatGPT chat has a prompt: "Hi ChatGPT can you help me fix a bug? Here is my code: ... When I run it, I get the following error: ...")

[18:42] Why would you why would you do that? Why would you go directly to the operating system? It makes a lot more sense to have an app dedicated for this. And so I think many of you, uh, use Cursor, I do as well. (The slide changes to a screenshot of the Cursor code editor interface, showing code on the left and a chat pane on the right)

[18:55] Uh, and, uh, Cursor is kind of like the thing you want instead. You don't want to just directly go to the ChatGPT. And I think Cursor is a very good example of an early LLM app that has a bunch of properties that I think are, um, useful across all the LLM apps. So in particular, you will notice that we have a traditional interface, (The slide changes to a split screenshot of Cursor. The left side is labeled "Traditional interface" (blue), and the right side "LLM integration" (red). Bullet points on the right explain: "1. Package state into a context window before calling LLM.")

[19:14] that allows a human to go in and do all the work manually, just as before. But in addition to that, we now have this LLM integration that allows us to go in bigger chunks. And so some of the properties of LLM apps that I think are shared and useful, to point out. (The slide animates to add bullet point: "2. Orchestrate and call multiple models (e.g. embedding models, chat models, diff apply models, ...)")

[19:30] Number one, the LLMs basically do a ton of the context management, um, number two, they orchestrate multiple calls to LLMs, right? So in the case of Cursor, there's under the hood embedding models for all your files, the actual chat models, models that apply diffs to the code, and this is all orchestrated for you. (The slide animates to add bullet point: "3. Application-specific GUI")

[19:44] A really big one that, uh, I think also maybe not fully appreciated always is application specific GUI and the importance of it. Um, because you don't just want to talk to the operating system directly in text. Text is very hard to read, interpret, understand, and also like you don't want to take some of these actions natively in text. So it's much better to just see a diff as like red and green change, and you can see what's being added, it's subtracted. It's much easier to just do command Y to accept or command N to reject. I shouldn't have to type it in text, right? So a GUI allows a human to audit the work of these fallible systems and to go faster. I'm going to come back to this point a little bit, uh, later as well. (The slide animates to add bullet point: "4. Autonomy slider: Tab -> Cmd+K -> Cmd+L -> Cmd+I (agent mode)")

[20:23] And the last kind of feature I want to point out is that there's what I call the autonomy slider. So, for example, in Cursor, you can just do top completion, you're mostly in charge, you can select a chunk of code and command K to change just that chunk of code, you can do command L to change the entire file, or you can do command I, which just, you know, letter it do whatever you want in the entire repo. And that's the sort of full autonomy agentic version. And so you are in charge of the autonomy slider, and depending on the complexity of the task at hand, you can, uh, tune the amount of autonomy that you're willing to give up for that task.

[20:57] Maybe to show one more example of a fairly successful LLM app, uh, Perplexity, (The slide changes to a screenshot of Perplexity's search interface, with a search result about "meta buys scale ai". Bullet points for "Example: Anatomy of Perplexity" on the right: "1. Package information into a context window", "2. Orchestrate multiple LLM models", "3. Application-specific GUI for Input/Output UIUX", "4. autonomy slider")

[21:03] um, they it also has very similar features to what I've just pointed out in Cursor. Uh, it packages up a lot of the information, it orchestrates multiple LLMs, it's got a GUI that allows you to audit some of its work. So, for example, it will, uh, cite sources, and you can imagine inspecting them. And it's got an autonomy slider. You can either just do a quick search, or you can do research, or you can do deep research and come back 10 minutes later. So this is all just varying levels of autonomy that you give up to the tool.

[21:28] So I guess my question is, I feel like all software will become partially autonomous. And I'm trying to think through like what does that look like? And for many of you who maintain products and services, how are you going to make your products and services partially autonomous? Can an LLM see all the things that a human can? Can an LLM act in all the ways that a human can? (The slide changes to a split screen. On the left, Adobe Photoshop. On the right, Unreal Engine. Text: "What does all software look like in the partial autonomy world?" Below, questions: "Can an LLM "see" all the things the human can?", "Can an LLM "act" in all the ways a human can?", "How can a human supervise and stay in the loop?")

[21:49] And can humans supervise and stay in the loop of this activity? Because again, these are fallible systems that aren't yet perfect. And what does a diff look like in Photoshop or something like that, you know? And also a lot of the traditional software right now, it has all these switches and all this kind of stuff that's all designed for human. All this has to change and become accessible to LLMs. (The slide changes to a circular diagram with "AI" and "HUMAN" arrows rotating. "Generation" from AI to Human, "Verification" from Human to AI. Text: "Consider the full workflow of partial autonomy UIUX")

[22:08] So, one thing I want to stress with a lot of these LLM apps that I'm not sure gets, uh, as much attention as it should is, um, we're now kind of like cooperating with AIs, and usually they are doing the generation, and we as humans are doing the verification. It is in our interest to make this loop go as fast as possible, so we're getting a lot of work done. (The slide animates to highlight "Verification" and adds text: "1. Make this EASY FAST to win.")

[22:29] There are two major ways that I, uh, think, uh, this can be done. Number one, you can speed up verification a lot, um, and I think GUIs, for example, are extremely important to this because a GUI utilizes your computer vision GPU in all of our head. Reading text is effortful and it's not fun, but looking at stuff is fun, and it's it's just a kind of like a highway to your brain. So I think GUIs are very useful for auditing systems and visual representations in general. (The slide animates to add text: "2. Keep AI 'on a tight leash' to increase the probability of successful verification")

[22:54] And number two, I would say is, we have to keep the AI on the leash. We keep I think a lot of people are getting way over excited with AI agents, and, uh, it's not useful to me to get a diff of 1,000 lines of code to my repo. Like, I have to I'm still the bottleneck, right? Even though that 1,000 lines come out instantly, I have to make sure that this thing is not introducing bugs, is just like and that is doing the correct thing, right? And that there's no security issues and so on. (The slide changes to a cartoon image of a boy holding a robot on a leash. The robot has multiple screens showing code. Text: "Human+AI UIUX for Coding")

[23:33] So I'm I think that, um, yeah, basically you have to sort of like It's in our interest to make the the flow of these two go very, very fast, and we have to somehow keep the AI on the leash because it gets way too over reactive. It's, uh, it's kind of like this. This is how I feel when I do AI assisted coding. (The slide animates to add text: "AI-assisted coding workflows (very rapidly evolving...) - describe the single, neat concrete, incremental change - don't ask for code, ask for approaches - pick an approach, draft code - review / learn / pull up API docs, ask for explanations, ... - wind back, try a different approach - test - git commit - ask for suggestions as what could be implemented next - repeat")

[23:51] If I'm just vibe coding, everything is nice and great, but if I'm actually trying to get work done, it's not so great to have an over reactive, uh, agents doing all this kind of stuff. So, this slide is not very good. I'm sorry. But I guess I'm trying to develop like many of you, some ways of utilizing these agents in my coding workflow, and to do AI assisted coding. And in my own work, I'm always scared to get way too big diffs. (The slide zooms out to show the full slide)

[24:04] I always go in small incremental chunks. I want to make sure that everything is good. I want to spin this loop very, very fast. And, uh, I sort of work on small chunks of single concrete thing. Uh, and so I think, uh, many of you probably are developing similar ways of working with the with LLMs. (The slide changes to a text box with a sample prompt: "I would reject the prompt to give okay results, but has minimum edge cases, poor positive and quality tradeoffs. Here's how you might re-write an inferior prompt so it's for the best case:" followed by a revised prompt structure. Below: "AI-assisted coding for tasks that can't get likely with ideas")

[24:18] Um, I also saw a number of blog posts that try to develop these best practices for working with LLMs. And here's one that I read recently and I thought was quite good. And it kind of discussed some techniques, and some of them have to do with how you keep the AI on the leash. So as an example, if you are prompting, if your prompt is big, then, uh, the AI might not do exactly what you wanted, and in that case, verification will fail. You're going to ask for something else. If a verification fails, then you're going to start spinning. So it makes a lot more sense to spend a bit more time to be more concrete in your prompts, which increases the probability of successful verification and you can move forward. And so I think a lot of us are going to end up finding, um, kind of techniques like this. (The slide changes to two images: "1. App for course creation (for teacher)" showing an open textbook, and "2. App for course serving (for student)" showing a teacher with a student at a computer. Text: "Example: keeping agents on the leash - AI = Education / LLM101h")

[24:55] I think in my own work as well, I'm currently interested in, uh, what education looks like in, um, together with, uh, kind of like now that we have AI, uh, and LLMs, what does education look like? And I think a a large amount of thought for me goes into how we keep AI on the leash. I don't think it just works to go to ChatGPT and be like, hey, teach me physics. I don't think this works because the AI is like gets lost in the woods. (The slide changes to a Tesla interior with the Autonomy slider visible on the screen. Text: "Example: Tesla Autopilot")

[25:18] And so for me, this is actually two separate apps, for example. Uh, there's an app for a teacher that creates courses, and then there's an app that takes courses and serves them to students. And in both cases, we now have a this intermediate artifact of a course that is auditable, and we can make sure it's good. We can make sure it's consistent, and the AI is kept on the leash with respect to a certain syllabus, a certain like, um, progression of projects, and so on. And so this is one way of keeping the AI on a leash, and I think it has a much higher likelihood of working, and the AI is not getting lost in the woods.

[25:49] One more kind of analogy I wanted to sort of allude to is I'm I'm not I'm no stranger to partial autonomy, and I kind of worked on this, I think, for five years at Tesla. And this is also partial autonomy product and shares a lot of the features. Like for example, right there in the instrument panel is the GUI of the autopilot. So it's showing me what the what the neural network sees and so on. And we have the autonomy slider, where over the course of my tenure there, we did more and more autonomous tasks for the user. (The slide changes to a Waymo self-driving car. Text: "2015 - 2025 was the decade of "driving agents" 2013: my first demo drive in a Waymo around Palo Alto (it was perfect).")

[26:17] And maybe the story that I wanted to tell very briefly is, uh, actually the first time I drove a self-driving vehicle was in 2013. And I had a friend who worked at Waymo, and, uh, he offered to give me a drive around Palo Alto. I took this picture using a Google Glass at the time. And many of you are so young that you might not even know what that is. Uh, but, uh, yeah, this was like all the rage at the time. And we got into this car, and we went for about a 30-minute drive around Palo Alto. Highways, uh, streets and so on. And this drive was perfect. There were zero interventions. And this was 2013, which is now 12 years ago. And it's kind of struck me because at the time when I had this perfect drive, this perfect demo, I felt like, wow, self-driving is imminent because this just works. This is incredible. (The slide animates to show a blank space on the text bar)

[27:03] Um, but here we are 12 years later, and we are still working on autonomy. Um, we are still working on driving agents. And even now, we haven't actually like fully solved the problem. Like, you may see Waymos going around, and they look driverless, but, you know, there's still a lot of teleoperation, and a lot of human in the loop of a lot of this driving. (The slide animates to show a cartoon image of Tony Stark with the Iron Man suit, one fully armored and one as a wireframe with a slider in the middle. Text: "THE IRON MAN SUIT Augmentation Agent")

[27:20] So we still haven't even like declared success, but I think it's definitely like going to succeed at this point, but it just took a long time. And so I think like like this is this software is really tricky. I think in the same way that driving is tricky. And so when I see things like, oh, 2025 is the year of agents, I get very concerned. (Andrej holds up a small, black remote, looking concerned)

[27:38] And I kind of feel like, you know, this is the decade of agents. And this is going to be quite some time. We need humans in the loop. We need to do this carefully. This is software. Let's be serious here. (Andrej smiles, slightly embarrassed, as the audience laughs)

[27:50] What the hell? (Andrej smiles, then looks at his remote)

[27:52] I had to follow all these instructions. This was crazy. So I think the last part of my talk, (Andrej smiles again)

[27:56] I guess one more analogy that I always think through is the Iron Man suit. Uh, I think this is I always love Iron Man. I think it's like so, um, correct in a bunch of ways with respect to technology and how it will play out. And what I love about the Iron Man suit is that it's both an augmentation, and Tony Stark can drive it, and it's also an agent, and in some of the movies, the Iron Man suit is quite autonomous and can fly around and find Tony and all this kind of stuff. And so this is the autonomy slider is we can be we can build augmentations, or we can build agents. And we kind of want to do a bit of both. But, (The slide changes back to the Iron Man suit image with a slider moving from "Augmentation" to "Agent")

[28:22] at this stage, I would say, working with fallible LLMs and so on, I would say, you know, it's less Iron Man robots, and more Iron Man suits that you want to build. (The slide changes to a white background with black text: "Building Autonomous Software". Left side has red X: "Iron Man robots", "Flashy demos of autonomous agents", "AGI 2027". Right side has green checkmark: "Iron Man suits", "Partial autonomy products", "Custom GUI and UIUX", "Fast Generation - Verification loop", "Autonomy slider")

[28:34] It's less like building flashy demos of autonomous agents and more building partial autonomy products. And these products have custom GUIs and UIUX, and we're trying to, uh, and this is done so that the generation verification loop with the human is very, very fast. But we are not losing the sight of the fact that it is in principle possible to automate this work. And there should be an autonomy slider in your product. And you should be thinking about how you can slide that autonomy slider and make your product, uh, sort of more autonomous over time. But this is kind of how I think there's lots of opportunities in these kinds of products.

[29:05] I want to now switch gears a little bit and talk about one other dimension that I think is very unique. Not only is there a new type of programming language that allows for autonomy in software, but also, as I mentioned, it's programmed in English, which is this natural interface, and suddenly everyone is a programmer because everyone speaks natural language like English. (The slide changes to white with black text: "Make software highly accessible 🙂")

[29:21] So this is extremely bullish and very interesting to me and also completely unprecedented, I would say. It used to be the case that you need to spend five to 10 years studying something to be able to do something in software. This is not the case anymore. So I don't know if by any chance, anyone has heard of vibe coding. (The slide animates to add: "(Have you heard of vibe coding by any chance?)")

[29:39] Uh, this this is the tweet that's kind of like introduced this. (The slide changes to a screenshot of Andrej Karpathy's tweet about "vibe coding")

[29:41] But I'm told that this is now like a major meme. (Audience laughs and claps) Fun story about this is that I've been on Twitter for like 15 years or something like that at this point, and I still have no clue which tweet will become viral and which tweet like fizzles and no one cares. And I thought that this tweet was going to be the latter. I don't know, it was just like a shower of thoughts, but this became like a total meme. (The slide changes to a Wikipedia page for "Vibe coding")

[30:18] And I really just can't tell, but I guess like it struck a chord and gave a name to something that everyone was feeling but couldn't quite say in words. So now there's a Wikipedia page and everything. (Andrej smiles as the audience claps)

[30:25] So this is like (Audience laughs)

[30:27] This is like a major contribution now or something like that. So, (Audience laughs)

[30:30] Um, so Tom Wolf from HuggingFace shared this beautiful video that I really love. (The slide changes to a tweet from Thomas Wolf showing a video of children coding with LLMs. Children are clapping and excited)

[30:37] These are kids vibe coding. (Children are excitedly using laptops and computers, clapping and smiling. One child sings a tune)

[30:44] And I find that this is such a wholesome video. Like I love this video. Like, how can you look at this video and feel bad about the future? The future is great. (Audience laughs)

[30:52] I think this will end up being like a gateway drug to software development. Um, I'm not a doomer about the future of the generation. And I think yeah, I love this video. So I tried vibe coding a little bit as well because it's so fun. So vibe coding is so great when you want to build something super duper custom that doesn't appear to exist, and you just want to wing it because it's a Saturday or something like that. (The slide changes to a smartphone screen showing a "Vibe Coding iOS app" that tracks calories, with buttons to add or subtract 100 kcal)

[31:13] So, I built this, uh, iOS app, and I don't I can't actually program in Swift, but I was really shocked that I was able to build like a super basic app, and I'm not going to explain it, it's really, uh, dumb. But, uh, I kind of like this was just like a day of work, and this was running on my phone like later that day, and I was like, wow, this is amazing. I didn't have to like read through Swift for like five days or something like that to like get started. (The app shows calorie count changing with button presses)

[31:37] I also vibe coded this app called MenuGen. And this is live, you can try it at menugen.app. And I basically had this problem where I show up at a restaurant, I read through the menu and I have no idea what any of the things are. (The slide changes to a split screen. On the left, a restaurant menu for "Spago Wolfgang Puck". On the right, a grid of food images with descriptions, resembling a digital menu. Text: "Vibe coding MenuGen https://www.menugen.app/")

[31:47] And I need pictures. So, this doesn't exist. So I was like, hey, I'm going to vibe code it. So, um, this is what it looks like. (The slide changes to a video of someone using the MenuGen app on a smartphone. They take a photo of the Spago menu)

[31:56] You go to menugen.app. Um, and, uh, you take a picture of a of a menu, and then MenuGen generates the images. (The app uploads the image, extracts items, and generates individual food photos for each menu item, scrolling through them)

[32:05] And everyone gets $5 in credits for free when you sign up. And therefore, this is a major cost center in my life. So, this is a negative negative revenue app for me right now. I've lost a huge amount of money on MenuGen. (Andrej laughs, audience laughs)

[32:20] Okay. But the fascinating thing about MenuGen for me is that the code of the the vibe the vibe coding part, the code was actually easy part of of vibe coding MenuGen. (The slide changes to a white background with black text: "The code was the easiest part! :O Most of the work was in the browser clicking things." Below, lists of LLM API keys, Flux API keys (sad emojis), Running locally (ez emoji), Vercel deployments (sad emoji), Domain names (sad emoji), Authentication (sad emoji), Payments (sad emoji). A small text box links to "Vibe coding MenuGen" blog post)

[32:30] And most of it actually was when I tried to make it real so that you can actually have authentication and payments and the domain name and a Vercel deployment. This was really hard, and all of this was not code. All of this DevOps stuff was in me in the browser clicking stuff. And this was extreme slog and took another week. So it was really fascinating that I had the MenuGen, um, basically demo working on my laptop in a few hours. And then it took me a week because I was trying to make it real. And the reason for this is this was just really annoying, um, So for example, if you try to add Google login to your web page, I know this is very small but just a huge amount of instructions of this a clerk library telling me how to integrate this. And this is crazy. Like it's telling me, go to this URL, click on this drop down, choose this, go to this and click on that, and it's like telling me what to do. Like a computer is telling me the actions I should be taking, like you do it. Why am I doing this? (Audience laughs)

[33:28] What the hell? (Audience laughs)

[33:32] I had to follow all these instructions. This was crazy. So I think the last part of my talk, therefore, focuses on, can we just build for agents? (The slide changes to white with black text: "Build for agents 🤖")

[33:43] I don't want to do this work. Can agents do this? Thank you. (Audience claps)

[33:47] Okay. So roughly speaking, I think there's a new category of consumer and manipulator of digital information. (The slide changes to white with black text: "There is new category of consumer/manipulator of digital information: 1. Humans (GUIs) 2. Computers (APIs) 3. NEW: Agents <- computers... but human-like")

[33:53] It used to be just humans through GUIs or computers through APIs. And now we have a completely new thing. And agents, they're they're computers, but they are human-like. Kind of, right? They're people spirits. There's people spirits on the internet and they need to interact with our software infrastructure. (Andrej gestures with his hands in a complex way)

[34:07] Like, can we build for them? It's a new thing. So as an example, you can have robots.txt on your domain, and you can instruct, uh, or like advise, I suppose, um, uh, web crawlers on how to behave on your website. (The slide changes to a split screen. On the left, a GitHub repo for "nanogpt". On the right, a stylized markdown file labeled "The /llms.txt file" with proposed standards for LLMs)

[34:18] In the same way, you can have maybe llms.txt file, which is just a simple markdown that's telling LLMs what this domain is about. And this is very readable to a to an LLM. If it had to instead get the HTML of your web page and try to parse it, this is very error prone and difficult, and it will screw it up, and it's not going to work. So we can just directly speak to the LLM, it's worth it. (The slide changes to a split screen. On the left, Vercel documentation. On the right, Stripe documentation mentioning LLMs in their integration workflow)

[34:38] Um, a huge amount of documentation is currently written for people. So you will see things like lists and bold and pictures, and this is not directly accessible by an LLM. So, (The slide animates to highlight "vercle.com/docs/llms.txt" on the Vercel side, and the Stripe documentation is titled "Build on Stripe with LLMs", showing a plain text version of the docs)

[34:50] I see some of the services now are transitioning a lot of their docs to be specifically for LLMs. So Vercel and Stripe as an example are early movers here, uh, but there are a few more that I've seen already. And they offer their documentation in markdown. Markdown is super easy for LLMs to understand. This is great. (The slide changes to a black background with "Manim Mathematical Animation Engine" and a photo of Grant Sanderson (3Blue1Brown) pointing to a complex mathematical animation, with Python code below)

[35:09] Um, maybe one simple example from from, uh, my experience as well. Maybe some of you know three blue one brown. He makes beautiful animation videos on YouTube. (Audience claps)

[35:25] Yeah, I love this library. So that he wrote, uh, Manim. And I wanted to make my own. And, uh, there's extensive documentations on how to use Manim. And, uh, so I didn't want to actually read through it. So I copy pasted the whole thing to an LLM, and I described what I wanted, and it just worked out of the box. Like LLM just vibe coded me an animation exactly what I wanted. And I was like, wow, this is amazing. So if we can make docs legible to LLMs, it's going to unlock a huge amount of, um, kind of use. And, um, I think this is wonderful and should should happen more. (The slide changes to a screenshot of a webpage titled "Introducing Operator", showing an agent performing browsing tasks, with a keyboard and mouse below)

[35:55] The other thing I wanted to point out is that it is absolutely possible that in the future, LLMs will be able to This is not even future, this is today. They'll be able to go around, and they'll be able to click stuff and so on. But I still think it's very worth, uh, basically meeting LLM halfway, LLMs halfway, and making it easier for them to access all this information, uh, because this is still fairly expensive, I would say, to use and, uh, a lot more difficult. (The slide changes back to the "robots.txt" and "/llms.txt" comparison)

[37:57] And so I do think that lots of software, there will be a long tail where it won't like adapt because these are not like life player sort of repositories or digital infrastructure, and we will need these tools. Uh, but I think for everyone else, I think it's very worth kind of like meeting in some middle point. So I'm bullish on both, if that makes sense.

[38:15] So in summary, what an amazing time to get into the industry. (The slide changes to a collage of images from previous slides, with bullet points summarizing the key takeaways from the presentation: "Partial autonomy LLM apps: - Package context - Orchestrate LLM calls - Custom GUI - Autonomy slider", "speed up the full generation-verification flow", and "Build for agents 🤖")

[38:20] We need to rewrite a ton of code, a ton of code will be written by professionals and vibe coders. These LLMs are kind of like utilities, kind of like fabs, but they're kind of especially like operating systems. But it's so early. It's like 1960s of operating systems. And, uh, and I think a lot of the analogies cross over. Um, and these LLMs are kind of like these fallible, uh, you know, people spirits that we have to learn to work with. And in order to do that properly, we need to adjust our infrastructure towards it. So when you're building these LLM apps, I describe some of the ways of working effectively with these LLMs, and some of the tools that make that, uh, kind of possible. And how you can spin this loop very, very quickly, and basically, uh, create partial autonomy products. And then, um, yeah, a lot of code has to also be written for the agents more directly. But in any case, going back to the Iron Man suit analogy, I think what we'll see over the next decade, roughly, is we're going to take the slider from left to right. (The slider in the Iron Man suit animation moves from "Augmentation" to "Agent")

[39:13] And I'm very interesting. It's going to be very interesting to see what that looks like, and I can't wait to build it with all of you. Thank you. (Andrej smiles, bows to applause)

[39:24] (Andrej walks off stage to applause, and the screen returns to the "AI STARTUP SCHOOL" title card)
*The presentation covered Andrej Karpathy's perspective on the evolution of software through Software 1.0 (code), 2.0 (weights/neural networks), and 3.0 (prompts/LLMs), comparing LLMs to utilities, fabs, and operating systems, and highlighting the opportunities and challenges of building partial autonomy applications and infrastructure for a future where everyone can "vibe code" in natural language.*

</details>


## Additional Sources Scraped

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

```python
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

```python
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

```python
from langgraph.types import Command

# Get review from a user (e.g., via a UI)
# In this case, we're using a bool, but this can be any json-serializable value.
human_review = True

for item in workflow.stream(Command(resume=human_review), config):
    print(item)
```

```python
{'workflow': {'essay': 'An essay about topic: cat', 'is_approved': False}}
```

The workflow has been completed and the review has been added to the essay.

## Entrypoint

The [`@entrypoint`](https://reference.langchain.com/python/langgraph/func/#langgraph.func.entrypoint) decorator can be used to create a workflow from a function. It encapsulates workflow logic and manages execution flow, including handling _long-running tasks_ and [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts).

### Definition

An **entrypoint** is defined by decorating a function with the `@entrypoint` decorator.The function **must accept a single positional argument**, which serves as the workflow input. If you need to pass multiple pieces of data, use a dictionary as the input type for the first argument.Decorating a function with an `entrypoint` produces a [`Pregel`](https://reference.langchain.com/python/langgraph/pregel/#langgraph.pregel.Pregel.stream) instance which helps to manage the execution of the workflow (e.g., handles streaming, resumption, and checkpointing).You will usually want to pass a **checkpointer** to the `@entrypoint` decorator to enable persistence and use features like **human-in-the-loop**.

- Sync

- Async

```python
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

```python
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

- Invoke

- Async Invoke

- Stream

- Async Stream

```python
config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}
my_workflow.invoke(some_input, config)  # Wait for the result synchronously
```

### Resuming

Resuming an execution after an [interrupt](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt) can be done by passing a **resume** value to the [`Command`](https://reference.langchain.com/python/langgraph/types/#langgraph.types.Command) primitive.

- Invoke

- Async Invoke

- Stream

- Async Stream

```python
from langgraph.types import Command

config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}

my_workflow.invoke(Command(resume=some_resume_value), config)
```

**Resuming after an error**To resume after an error, run the `entrypoint` with a `None` and the same **thread id** (config).This assumes that the underlying **error** has been resolved and execution can proceed successfully.

- Invoke

- Async Invoke

- Stream

- Async Stream

```python

config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}

my_workflow.invoke(None, config)
```

### Short-term memory

When an `entrypoint` is defined with a `checkpointer`, it stores information between successive invocations on the same **thread id** in [checkpoints](https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints).This allows accessing the state from the previous invocation using the `previous` parameter.By default, the `previous` parameter is the return value of the previous invocation.

```python
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

```python
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

```python
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

- Synchronous Invocation

- Asynchronous Invocation

```python
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

- Incorrect

- Correct

In this example, a side effect (writing to a file) is directly included in the workflow, so it will be executed a second time when resuming the workflow.

```python
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

- Incorrect

- Correct

In this example, the workflow uses the current time to determine which task to execute. This is non-deterministic because the result of the workflow depends on the time at which it is executed.

```python
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
<summary>the-fastmcp-server-fastmcp</summary>

The central piece of a FastMCP application is the `FastMCP` server class. This class acts as the main container for your application’s tools, resources, and prompts, and manages communication with MCP clients.

## Creating a Server

Instantiating a server is straightforward. You typically provide a name for your server, which helps identify it in client applications or logs.

```
from fastmcp import FastMCP

# Create a basic server instance
mcp = FastMCP(name="MyAssistantServer")

# You can also add instructions for how to interact with the server
mcp_with_instructions = FastMCP(
    name="HelpfulAssistant",
    instructions="""
        This server provides data analysis tools.
        Call get_average() to analyze numerical data.
    """,
)
```

The `FastMCP` constructor accepts several arguments:

## FastMCP Constructor Parameters

name

str

default:"FastMCP"

A human-readable name for your server

instructions

str \| None

Description of how to interact with this server. These instructions help clients understand the server’s purpose and available functionality

version

str \| None

Version string for your server. If not provided, defaults to the FastMCP library version

website\_url

str \| None

`New in version: 2.13.0`URL to a website with more information about your server. Displayed in client applications

icons

list\[Icon\] \| None

`New in version: 2.13.0`List of icon representations for your server. Icons help users visually identify your server in client applications. See [Icons](https://gofastmcp.com/servers/icons) for detailed examples

auth

OAuthProvider \| TokenVerifier \| None

Authentication provider for securing HTTP-based transports. See [Authentication](https://gofastmcp.com/servers/auth/authentication) for configuration options

lifespan

AsyncContextManager \| None

An async context manager function for server startup and shutdown logic

tools

list\[Tool \| Callable\] \| None

A list of tools (or functions to convert to tools) to add to the server. In some cases, providing tools programmatically may be more convenient than using the `@mcp.tool` decorator

include\_tags

set\[str\] \| None

Only expose components with at least one matching tag

exclude\_tags

set\[str\] \| None

Hide components with any matching tag

on\_duplicate\_tools

Literal\["error", "warn", "replace"\]

default:"error"

How to handle duplicate tool registrations

on\_duplicate\_resources

Literal\["error", "warn", "replace"\]

default:"warn"

How to handle duplicate resource registrations

on\_duplicate\_prompts

Literal\["error", "warn", "replace"\]

default:"replace"

How to handle duplicate prompt registrations

strict\_input\_validation

bool

default:"False"

`New in version: 2.13.0`Controls how tool input parameters are validated. When `False` (default), FastMCP uses Pydantic’s flexible validation that coerces compatible inputs (e.g., `"10"` → `10` for int parameters). When `True`, uses the MCP SDK’s JSON Schema validation to validate inputs against the exact schema before passing them to your function, rejecting any type mismatches. The default mode improves compatibility with LLM clients while maintaining type safety. See [Input Validation Modes](https://gofastmcp.com/servers/tools#input-validation-modes) for details

include\_fastmcp\_meta

bool

default:"True"

`New in version: 2.11.0`Whether to include FastMCP metadata in component responses. When `True`, component tags and other FastMCP-specific metadata are included in the `_fastmcp` namespace within each component’s `meta` field. When `False`, this metadata is omitted, resulting in cleaner integration with external systems. Can be overridden globally via `FASTMCP_INCLUDE_FASTMCP_META` environment variable

## Components

FastMCP servers expose several types of components to the client:

### Tools

Tools are functions that the client can call to perform actions or access external systems.

```
@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b
```

See [Tools](https://gofastmcp.com/servers/tools) for detailed documentation.

### Resources

Resources expose data sources that the client can read.

```
@mcp.resource("data://config")
def get_config() -> dict:
    """Provides the application configuration."""
    return {"theme": "dark", "version": "1.0"}
```

See [Resources & Templates](https://gofastmcp.com/servers/resources) for detailed documentation.

### Resource Templates

Resource templates are parameterized resources that allow the client to request specific data.

```
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """Retrieves a user's profile by ID."""
    # The {user_id} in the URI is extracted and passed to this function
    return {"id": user_id, "name": f"User {user_id}", "status": "active"}
```

See [Resources & Templates](https://gofastmcp.com/servers/resources) for detailed documentation.

### Prompts

Prompts are reusable message templates for guiding the LLM.

```
@mcp.prompt
def analyze_data(data_points: list[float]) -> str:
    """Creates a prompt asking for analysis of numerical data."""
    formatted_data = ", ".join(str(point) for point in data_points)
    return f"Please analyze these data points: {formatted_data}"
```

See [Prompts](https://gofastmcp.com/servers/prompts) for detailed documentation.

## Tag-Based Filtering

`New in version: 2.8.0`FastMCP supports tag-based filtering to selectively expose components based on configurable include/exclude tag sets. This is useful for creating different views of your server for different environments or users.Components can be tagged when defined using the `tags` parameter:

```
@mcp.tool(tags={"public", "utility"})
def public_tool() -> str:
    return "This tool is public"

@mcp.tool(tags={"internal", "admin"})
def admin_tool() -> str:
    return "This tool is for admins only"
```

The filtering logic works as follows:

-   **Include tags**: If specified, only components with at least one matching tag are exposed
-   **Exclude tags**: Components with any matching tag are filtered out
-   **Precedence**: Exclude tags always take priority over include tags

To ensure a component is never exposed, you can set `enabled=False` on the component itself. To learn more, see the component-specific documentation.

You configure tag-based filtering when creating your server:

```
# Only expose components tagged with "public"
mcp = FastMCP(include_tags={"public"})

# Hide components tagged as "internal" or "deprecated"
mcp = FastMCP(exclude_tags={"internal", "deprecated"})

# Combine both: show admin tools but hide deprecated ones
mcp = FastMCP(include_tags={"admin"}, exclude_tags={"deprecated"})
```

This filtering applies to all component types (tools, resources, resource templates, and prompts) and affects both listing and access.

## Running the Server

FastMCP servers need a transport mechanism to communicate with clients. You typically start your server by calling the `mcp.run()` method on your `FastMCP` instance, often within an `if __name__ == "__main__":` block in your main server script. This pattern ensures compatibility with various MCP clients.

```
# my_server.py
from fastmcp import FastMCP

mcp = FastMCP(name="MyServer")

@mcp.tool
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    # This runs the server, defaulting to STDIO transport
    mcp.run()

    # To use a different transport, e.g., HTTP:
    # mcp.run(transport="http", host="127.0.0.1", port=9000)
```

FastMCP supports several transport options:

-   STDIO (default, for local tools)
-   HTTP (recommended for web services, uses Streamable HTTP protocol)
-   SSE (legacy web transport, deprecated)

The server can also be run using the FastMCP CLI.For detailed information on each transport, how to configure them (host, port, paths), and when to use which, please refer to the [**Running Your FastMCP Server**](https://gofastmcp.com/deployment/running-server) guide.

## Custom Routes

When running your server with HTTP transport, you can add custom web routes alongside your MCP endpoint using the `@custom_route` decorator. This is useful for simple endpoints like health checks that need to be served alongside your MCP server:

```
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

mcp = FastMCP("MyServer")

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

if __name__ == "__main__":
    mcp.run(transport="http")  # Health check at http://localhost:8000/health
```

Custom routes are served alongside your MCP endpoint and are useful for:

-   Health check endpoints for monitoring
-   Simple status or info endpoints
-   Basic webhooks or callbacks

For more complex web applications, consider [mounting your MCP server into a FastAPI or Starlette app](https://gofastmcp.com/deployment/http#integration-with-web-frameworks).

## Composing Servers

`New in version: 2.2.0`FastMCP supports composing multiple servers together using `import_server` (static copy) and `mount` (live link). This allows you to organize large applications into modular components or reuse existing servers.See the [Server Composition](https://gofastmcp.com/servers/composition) guide for full details, best practices, and examples.

```
# Example: Importing a subserver
from fastmcp import FastMCP
import asyncio

main = FastMCP(name="Main")
sub = FastMCP(name="Sub")

@sub.tool
def hello():
    return "hi"

# Mount directly
main.mount(sub, prefix="sub")
```

## Proxying Servers

`New in version: 2.0.0`FastMCP can act as a proxy for any MCP server (local or remote) using `FastMCP.as_proxy`, letting you bridge transports or add a frontend to existing servers. For example, you can expose a remote SSE server locally via stdio, or vice versa.Proxies automatically handle concurrent operations safely by creating fresh sessions for each request when using disconnected clients.See the [Proxying Servers](https://gofastmcp.com/servers/proxy) guide for details and advanced usage.

```
from fastmcp import FastMCP, Client

backend = Client("http://example.com/mcp/sse")
proxy = FastMCP.as_proxy(backend, name="ProxyServer")
# Now use the proxy like any FastMCP server
```

## OpenAPI Integration

`New in version: 2.0.0`FastMCP can automatically generate servers from OpenAPI specifications or existing FastAPI applications using `FastMCP.from_openapi()` and `FastMCP.from_fastapi()`. This allows you to instantly convert existing APIs into MCP servers without manual tool creation.See the [FastAPI Integration](https://gofastmcp.com/integrations/fastapi) and [OpenAPI Integration](https://gofastmcp.com/integrations/openapi) guides for detailed examples and configuration options.

```
import httpx
from fastmcp import FastMCP

# From OpenAPI spec
spec = httpx.get("https://api.example.com/openapi.json").json()
mcp = FastMCP.from_openapi(openapi_spec=spec, client=httpx.AsyncClient())

# From FastAPI app
from fastapi import FastAPI
app = FastAPI()
mcp = FastMCP.from_fastapi(app=app)
```

## Server Configuration

Servers can be configured using a combination of initialization arguments, global settings, and transport-specific settings.

### Server-Specific Configuration

Server-specific settings are passed when creating the `FastMCP` instance and control server behavior:

```
from fastmcp import FastMCP

# Configure server-specific settings
mcp = FastMCP(
    name="ConfiguredServer",
    include_tags={"public", "api"},              # Only expose these tagged components
    exclude_tags={"internal", "deprecated"},     # Hide these tagged components
    on_duplicate_tools="error",                  # Handle duplicate registrations
    on_duplicate_resources="warn",
    on_duplicate_prompts="replace",
    include_fastmcp_meta=False,                  # Disable FastMCP metadata for cleaner integration
)
```

### Global Settings

Global settings affect all FastMCP servers and can be configured via environment variables (prefixed with `FASTMCP_`) or in a `.env` file:

```
import fastmcp

# Access global settings
print(fastmcp.settings.log_level)        # Default: "INFO"
print(fastmcp.settings.mask_error_details)  # Default: False
print(fastmcp.settings.strict_input_validation)  # Default: False
print(fastmcp.settings.include_fastmcp_meta)   # Default: True
```

Common global settings include:

-   **`log_level`**: Logging level (“DEBUG”, “INFO”, “WARNING”, “ERROR”, “CRITICAL”), set with `FASTMCP_LOG_LEVEL`
-   **`mask_error_details`**: Whether to hide detailed error information from clients, set with `FASTMCP_MASK_ERROR_DETAILS`
-   **`strict_input_validation`**: Controls tool input validation mode (default: False for flexible coercion), set with `FASTMCP_STRICT_INPUT_VALIDATION`. See [Input Validation Modes](https://gofastmcp.com/servers/tools#input-validation-modes)
-   **`include_fastmcp_meta`**: Whether to include FastMCP metadata in component responses (default: True), set with `FASTMCP_INCLUDE_FASTMCP_META`
-   **`env_file`**: Path to the environment file to load settings from (default: “.env”), set with `FASTMCP_ENV_FILE`. Useful when your project uses a `.env` file with syntax incompatible with python-dotenv

### Transport-Specific Configuration

Transport settings are provided when running the server and control network behavior:

```
# Configure transport when running
mcp.run(
    transport="http",
    host="0.0.0.0",           # Bind to all interfaces
    port=9000,                # Custom port
    log_level="DEBUG",        # Override global log level
)

# Or for async usage
await mcp.run_async(
    transport="http",
    host="127.0.0.1",
    port=8080,
)
```

### Setting Global Configuration

Global FastMCP settings can be configured via environment variables (prefixed with `FASTMCP_`):

```
# Configure global FastMCP behavior
export FASTMCP_LOG_LEVEL=DEBUG
export FASTMCP_MASK_ERROR_DETAILS=True
export FASTMCP_STRICT_INPUT_VALIDATION=False
export FASTMCP_INCLUDE_FASTMCP_META=False
```

### Custom Tool Serialization

`New in version: 2.2.7`By default, FastMCP serializes tool return values to JSON when they need to be converted to text. You can customize this behavior by providing a `tool_serializer` function when creating your server:

```
import yaml
from fastmcp import FastMCP

# Define a custom serializer that formats dictionaries as YAML
def yaml_serializer(data):
    return yaml.dump(data, sort_keys=False)

# Create a server with the custom serializer
mcp = FastMCP(name="MyServer", tool_serializer=yaml_serializer)

@mcp.tool
def get_config():
    """Returns configuration in YAML format."""
    return {"api_key": "abc123", "debug": True, "rate_limit": 100}
```

The serializer function takes any data object and returns a string representation. This is applied to **all non-string return values** from your tools. Tools that already return strings bypass the serializer.This customization is useful when you want to:

-   Format data in a specific way (like YAML or custom formats)
-   Control specific serialization options (like indentation or sorting)
-   Add metadata or transform data before sending it to clients

If the serializer function raises an exception, the tool will fall back to the default JSON serialization to avoid breaking the server.

</details>

<details>
<summary>use-the-functional-api-docs-by-langchain</summary>

The Functional API allows you to add LangGraph’s key features — persistence, memory, human-in-the-loop, and streaming — to your applications with minimal changes to your existing code.

For conceptual information on the functional API, see [Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api).

## Creating a simple workflow

When defining an `entrypoint`, input is restricted to the first argument of the function. To pass multiple inputs, you can use a dictionary.

```
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    value = inputs["value"]
    another_value = inputs["another_value"]
    ...

my_workflow.invoke({"value": 1, "another_value": 2})
```

Extended example: simple workflow

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

```
from langgraph.types import StreamWriter

@entrypoint(checkpointer=checkpointer)
async def main(inputs: dict, writer: StreamWriter) -> int:
...
```

## Retry policy

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

```
'OK'
```

## Caching Tasks

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

```
main.invoke(None, config=config)
```

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

```
config = {"configurable": {"thread_id": "1"}}

for event in graph.stream("foo", config):
    print(event)
    print("\n")
```

Note that we’ve paused with an [`interrupt`](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt) after `step_1`. The interrupt provides instructions to resume the run. To resume, we issue a [`Command`](https://docs.langchain.com/oss/python/langgraph/interrupts#resuming-interrupts) containing the data expected by the `human_feedback` task.

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

```
config = {
    "configurable": {
        "thread_id": "1"
    }
}
list(graph.get_state_history(config))
```

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

## Integrate with other libraries

- [Add LangGraph’s features to other frameworks using the functional API](https://docs.langchain.com/langsmith/autogen-integration): Add LangGraph features like persistence, memory and streaming to other agent frameworks that do not provide them out of the box.

</details>

<details>
<summary>www-anthropic-com</summary>

We've worked with dozens of teams building LLM agents across industries. Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks.

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.

## What are agents?

"Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:

- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
- **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Below, we will explore both types of agentic systems in detail. In Appendix 1 (“Agents in Practice”), we describe two domains where customers have found particular value in using these kinds of systems.

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

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fd3083d3f40bb2b6f477901cc9a240738d3dd1371-2401x1000.png&w=3840&q=75The augmented LLM

We recommend focusing on two key aspects of the implementation: tailoring these capabilities to your specific use case and ensuring they provide an easy, well-documented interface for your LLM. While there are many ways to implement these augmentations, one approach is through our recently released [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol), which allows developers to integrate with a growing ecosystem of third-party tools with a simple [client implementation](https://modelcontextprotocol.io/tutorials/building-a-client#building-mcp-clients).

For the remainder of this post, we'll assume each LLM call has access to these augmented capabilities.

### Workflow: Prompt chaining

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F7418719e3dab222dccb379b8879e1dc08ad34c78-2401x1000.png&w=3840&q=75The prompt chaining workflow

**When to use this workflow:** This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

**Examples where prompt chaining is useful:**

- Generating Marketing copy, then translating it into a different language.
- Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

### Workflow: Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840&q=75The routing workflow

**When to use this workflow:** Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

**Examples where routing is useful:**

- Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes, prompts, and tools.
- Routing easy/common questions to smaller, cost-efficient models like Claude Haiku 4.5 and hard/unusual questions to more capable models like Claude Sonnet 4.5 to optimize for best performance.

### Workflow: Parallelization

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations:

- **Sectioning**: Breaking a task into independent subtasks run in parallel.
- **Voting:** Running the same task multiple times to get diverse outputs.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840&q=75The parallelization workflow

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

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840&q=75The orchestrator-workers workflow

**When to use this workflow:** This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

**Example where orchestrator-workers is useful:**

- Coding products that make complex changes to multiple files each time.
- Search tasks that involve gathering and analyzing information from multiple sources for possible relevant information.

### Workflow: Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75The evaluator-optimizer workflow

**When to use this workflow:** This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

**Examples where evaluator-optimizer is useful:**

- Literary translation where there are nuances that the translator LLM might not capture initially, but where an evaluator LLM can provide useful critiques.
- Complex search tasks that require multiple rounds of searching and analysis to gather comprehensive information, where the evaluator decides whether further searches are warranted.

### Agents

Agents are emerging in production as LLMs mature in key capabilities—understanding complex inputs, engaging in reasoning and planning, using tools reliably, and recovering from errors. Agents begin their work with either a command from, or interactive discussion with, the human user. Once the task is clear, agents plan and operate independently, potentially returning to the human for further information or judgement. During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress. Agents can then pause for human feedback at checkpoints or when encountering blockers. The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully. We expand on best practices for tool development in Appendix 2 ("Prompt Engineering your Tools").

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png&w=3840&q=75Autonomous agent

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

**Examples where agents are useful:**

The following examples are from our own implementations:

- A coding Agent to resolve [SWE-bench tasks](https://www.anthropic.com/research/swe-bench-sonnet), which involve edits to many files based on a task description;
- Our [“computer use” reference implementation](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo), where Claude uses a computer to accomplish tasks.

https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F4b9a1f4eb63d5962a6e1746ac26bbc857cf3474f-2400x1666.png&w=3840&q=75High-level flow of a coding agent

## Combining and customizing these patterns

These building blocks aren't prescriptive. They're common patterns that developers can shape and combine to fit different use cases. The key to success, as with any LLM features, is measuring performance and iterating on implementations. To repeat: you should consider adding complexity _only_ when it demonstrably improves outcomes.

## Summary

Success in the LLM space isn't about building the most sophisticated system. It's about building the _right_ system for your needs. Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short.

When implementing agents, we try to follow three core principles:

1. Maintain **simplicity** in your agent's design.
2. Prioritize **transparency** by explicitly showing the agent’s planning steps.
3. Carefully craft your agent-computer interface (ACI) through thorough tool **documentation and testing**.

Frameworks can help you get started quickly, but don't hesitate to reduce abstraction layers and build with basic components as you move to production. By following these principles, you can create agents that are not only powerful but also reliable, maintainable, and trusted by their users.

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
