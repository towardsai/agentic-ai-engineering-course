# Research

## Research Results

<details>
<summary>What are the established design patterns and best practices for implementing "human-in-the-loop" (HITL) workflows in practical AI applications, particularly those involving generative models?</summary>

### Source [1]: https://zapier.com/blog/human-in-the-loop/

Query: What are the established design patterns and best practices for implementing "human-in-the-loop" (HITL) workflows in practical AI applications, particularly those involving generative models?

Answer: This article defines **human-in-the-loop (HITL)** as the intentional integration of human oversight into autonomous AI workflows at critical decision points, adding checkpoints for approval, rejection, or feedback before the workflow continues.[1]

It emphasizes that HITL is about **where, when, and how** to include humans so AI can run autonomously most of the time, but pauses for human judgment when decisions involve risk, nuance, or accountability.[1] HITL is framed as a safety net for agentic workflows that route messages, update records, and trigger multi-app processes, ensuring humans are involved when outcomes matter.[1]

The article describes several **design patterns** for HITL in AI and agentic workflows:

- **Confidence-based routing**: The agent computes a confidence score for its decision; if the score falls below a defined threshold, it automatically defers to a human for review.[1] This is recommended for workflows handling mostly straightforward tasks (e.g., categorizing support tickets) but needing fallback for ambiguous edge cases.

- **Escalation paths**: When an action falls outside the agent’s scope (such as a refund request exceeding an automated limit), the system routes the case to a designated human team with context, often via messaging channels, instead of failing silently or retrying endlessly.[1]

- **Approvals and verification steps**: HITL checkpoints are inserted before high-impact actions (publishing, sending external communications, executing irreversible changes), where humans can approve, reject, or modify outputs.[1]

- **Requests for context**: The system can pause and ask humans for missing information or clarification to resolve ambiguity before proceeding, rather than guessing.[1]

The article also highlights best practices: log every pause and decision for **compliance and review**, treat human corrections as **feedback loops and training data**, and avoid overusing HITL so as not to slow down low-risk automation.[1]

-----

-----

### Source [2]: https://workos.com/blog/why-ai-still-needs-you-exploring-human-in-the-loop-systems

Query: What are the established design patterns and best practices for implementing "human-in-the-loop" (HITL) workflows in practical AI applications, particularly those involving generative models?

Answer: This source defines **Human-in-the-Loop (HITL)** as a design approach where AI systems are **intentionally built to incorporate human intervention**, embedding human oversight, judgment, and accountability directly into the AI workflow.[2]

It frames HITL as **collaboration between people and machines**, not a retreat from automation.[2] Instead of full autonomy, HITL introduces intentional checkpoints where humans can **review, override, or guide** AI behavior, especially when outcomes carry risk or ethical significance.[2]

The article identifies **stages of HITL interaction across the AI lifecycle**:

- **Pre-processing**: Humans shape inputs and context before the AI runs (e.g., defining instructions, guardrails, and constraints), ensuring the system starts with correct assumptions.[2]

- **Inference-time oversight**: During operation, humans validate and adjust outputs, provide feedback, and make final decisions in areas like content creation, design, and healthcare, where AI proposals must be checked for tone, accuracy, and context.[2]

- **Post-processing and feedback**: Human corrections and evaluations are fed back to refine models and adjust system behavior, creating a continuous improvement loop.[2]

The article stresses several **best-practice principles** for HITL workflows:

- Maintain **human accountability** for consequential decisions, with AI as a decision-support tool, not the ultimate arbiter.[2]

- Use HITL as **adaptive collaboration**, not just end-of-pipe approvals: support back-and-forth clarification, co-creation, and mid-process adjustments between agents and users.[2]

- Align HITL design with **risk level**: the higher the impact of a mistake, the more frequent and earlier human checkpoints should appear.[2]

- Treat HITL as a **design principle** for building trustworthy systems—shared intelligence and value-guided oversight rather than blind autonomy.[2]

These patterns apply directly to generative models, which should be designed to propose options that humans refine, approve, or reject, especially in domains such as content generation, design, and healthcare.[2]

-----

-----

### Source [3]: https://witness.ai/blog/human-in-the-loop-ai/

Query: What are the established design patterns and best practices for implementing "human-in-the-loop" (HITL) workflows in practical AI applications, particularly those involving generative models?

Answer: This source defines **human-in-the-loop (HITL) AI** as a design pattern where human intelligence is **strategically embedded** into AI systems at key points in the workflow.[3] It contrasts HITL with full automation: traditional systems let models process inputs and generate outputs without interruption, while HITL systems ensure humans actively participate in several stages.[3]

The article identifies main **roles for humans** in HITL systems:

- **Data annotation**: Humans supply labeled data to improve model performance and handle ambiguous or nuanced categories.[3]

- **Model training and tuning**: Human feedback is used to correct errors and reduce bias, helping align model behavior with real-world expectations.[3]

- **Inference oversight**: During real-time operation, humans review and validate outputs, especially for high-risk decisions, and can override or correct the AI.[3]

- **Edge case handling**: Humans detect and respond to out-of-distribution or unusual cases that the model is not well-trained on.[3]

The article compares **HITL vs AI-in-the-loop** with a table: in HITL, the **human is the primary decision-maker**, and AI supports and assists; in AI-in-the-loop, AI leads with optional human oversight, and humans mostly monitor or occasionally override.[3]

For **practical AI applications**, especially generative models, the article implies several best practices:

- Ensure humans retain **final control** in real-world, high-risk scenarios, with AI outputs treated as recommendations.[3]

- Explicitly define **responsibility boundaries**: which decisions must always be human-approved vs where AI can act autonomously.[3]

- Use human feedback loops not just at deployment but throughout the lifecycle—annotation, training, and runtime supervision—to continuously improve accuracy and mitigate bias.[3]

- In workflows like content generation or decision support, design UI/UX so humans can efficiently validate, correct, and flag outputs, turning these interactions into structured feedback signals.[3]

-----

-----

### Source [4]: https://blog.metaphacts.com/human-in-the-loop-for-ai-a-collaborative-future-in-research-workflows

Query: What are the established design patterns and best practices for implementing "human-in-the-loop" (HITL) workflows in practical AI applications, particularly those involving generative models?

Answer: This article describes **human-in-the-loop systems** as integrating human oversight into AI and machine learning processes, combining human critical thinking with AI speed and efficiency.[4] It distinguishes such systems from fully autonomous AI like chatbots, robots, and self-driving cars that make decisions without human intervention.[4]

In the context of **research workflows**, the article emphasizes that HITL enables **collaborative knowledge discovery**, where AI suggests connections or insights but humans validate, refine, and contextualize them.[4]

Design and best-practice aspects highlighted include:

- Treat humans as **domain experts** who resolve ambiguity, interpret complex relationships, and ensure that AI-generated hypotheses or summaries are valid and relevant to the research question.[4]

- Use AI to **augment**, not replace, human reasoning—AI can surface candidate facts, links, or patterns, but the HITL pattern requires human confirmation before results are accepted into high-trust knowledge assets or databases.[4]

- Build **transparent interaction loops**: allow researchers to inspect why certain results were suggested, then confirm, reject, or correct them; these actions guide subsequent AI behavior.[4]

- Employ HITL at multiple stages of the pipeline: data curation, entity and relationship extraction, ontology maintenance, and final interpretation, so human oversight is present where errors could propagate into critical knowledge structures.[4]

- Ensure the system logs human decisions and rationales to support **auditability and reproducibility** of research outcomes assisted by AI.[4]

While the focus is research, these principles translate directly to generative models used in knowledge-intensive domains: use HITL to validate generated content before integrating it into authoritative corpora, maintain clear human responsibility for final conclusions, and design tools that make review and correction efficient for experts.[4]

-----

-----

### Source [5]: https://www.youtube.com/watch?v=YCFGjLjNOyw

Query: What are the established design patterns and best practices for implementing "human-in-the-loop" (HITL) workflows in practical AI applications, particularly those involving generative models?

Answer: In this talk on **Human-in-the-Loop (HITL) for AI agents**, Cornelia Davis discusses patterns and best practices for making agentic applications production-grade.[5] The focus is on building **real HITL flows** for AI agents, especially in long-running workflows that must survive restarts and coordinate among humans and agents.[5]

Key **HITL patterns** demonstrated include:

- **Approvals**: Insert explicit approval steps into workflows before sensitive or irreversible actions. The workflow engine pauses, awaits human decision, then resumes based on approve/reject input.[5]

- **Handoffs and checkpoints**: Design checkpoints where control cleanly passes from agent to human (and back), ensuring clarity over who is responsible at each step. These checkpoints are modeled explicitly in the orchestration layer.[5]

- **Long-running workflow management**: HITL workflows must be resilient to restarts and failures. The system persists state at every checkpoint so that if a process is interrupted while waiting for human input, it can later resume exactly where it left off.[5]

- **Incremental progress with pauses**: Structure complex tasks as sequences of smaller steps with potential HITL points between them, rather than a single monolithic agent call.[5]

Best practices highlighted include:

- Add HITL to an existing agent **without rewriting** the core logic, by wrapping agents in an orchestration layer that manages pauses, approvals, and handoffs.[5]

- Log all human interactions for **observability and audit**, enabling debugging of agent behavior and human decisions.[5]

- Treat HITL as essential to make agentic apps safe for production, particularly when they interact with external systems or humans.[5]

These patterns work well with **generative agents**, where approvals and checkpoints are placed around generation, external actions (like sending emails), and data modifications.[5]

-----

-----

### Source [6]: https://www.permit.io/blog/human-in-the-loop-for-ai-agents-best-practices-frameworks-use-cases-and-demo

Query: What are the established design patterns and best practices for implementing "human-in-the-loop" (HITL) workflows in practical AI applications, particularly those involving generative models?

Answer: This article presents **Human-in-the-Loop (HITL) for AI agents** as crucial for control, safety, and governance in agentic workflows.[6] It stresses that HITL allows teams to **prevent irreversible mistakes**, ensure **accountability**, comply with **audit and governance requirements** (e.g., SOC 2), and build trust by making AI a supervised assistant rather than a black box.[6]

It lays out key **design patterns for HITL in agentic systems**:

- **Pre-approval gating**: Before performing sensitive actions (access changes, financial operations, data deletion), agents must obtain explicit human approval, typically integrated via policy and authorization layers.[6]

- **Human-as-a-Tool**: The agent treats a human as just another callable tool. When uncertain, it calls this tool (e.g., via a message or UI) to ask for clarification or a decision, then uses the response as context to proceed.[6]

- **Pausing long-running workflows**: Agents and workflows can pause at predefined HITL checkpoints, waiting for human input before continuing, which is especially important in multi-step operations.[6]

- **Checkpoints before final actions**: Insert HITL steps at the end of workflows, so the last mile—sending, publishing, committing changes—is always under human oversight.[6]

The article also describes **best practices**:

- Design HITL based on **risk and blast radius**: the greater the potential impact, the earlier and more frequent the checkpoints.[6]

- Use a dedicated **authorization and policy** layer to determine which actions require HITL, rather than scattering ad-hoc checks in code.[6]

- Ensure robust **auditing and logging** of all agent decisions and human approvals to satisfy compliance and incident response requirements.[6]

- Integrate HITL patterns with existing frameworks (LangChain, CrewAI, etc.) where `human` is exposed as a tool or step type.[6]

These patterns map naturally onto generative AI use cases where agents draft content, interact with APIs, or manage resources, ensuring humans approve high-impact outputs and policy-sensitive decisions.[6]

-----

-----

### Source [7]: https://www.ibm.com/think/tutorials/human-in-the-loop-ai-agent-langraph-watsonx-ai

Query: What are the established design patterns and best practices for implementing "human-in-the-loop" (HITL) workflows in practical AI applications, particularly those involving generative models?

Answer: This IBM tutorial demonstrates implementing **human-in-the-loop** as a feedback mechanism for an **agentic system built with LangGraph and watsonx.ai**.[8] It provides a concrete pattern for adding HITL to LLM-based agents.

The workflow is modeled as a **graph of nodes** (tools, LLM calls, decision points) with edges representing transitions.[8] A specific **HITL node** is introduced where human feedback is collected and used to influence subsequent agent behavior.[8]

Key design aspects illustrated include:

- **Feedback collection**: After the agent produces an output or intermediate result, the HITL node presents this to a human, who can rate, correct, or comment on it. This feedback is captured as structured data.[8]

- **Control flow based on feedback**: Depending on the human’s response (approve, request changes, reject), the graph routes execution differently—e.g., back to the LLM with revised instructions, to an alternative tool, or forward to completion.[8]

- **Persistence and state management**: The tutorial shows how to keep track of conversation and workflow state so that human intervention fits naturally into multi-step interactions.[8]

- **Fine-tuning and improvement loops**: Collected feedback can later be used to refine prompts, adjust system behavior, or fine-tune models, closing the loop between runtime oversight and model evolution.[8]

Best practices embedded in the tutorial include:

- Treat HITL as an explicit **node type** in orchestration, making it visible and configurable rather than ad-hoc.[8]

- Clearly define what feedback is requested (e.g., correctness, helpfulness) to make it actionable.[8]

- Use HITL selectively at stages where errors are most costly or likely, rather than wrapping every step, to balance quality and efficiency.[8]

This pattern is directly applicable to **generative AI workflows**, where content or decisions are routed through HITL nodes for review, with human feedback shaping both immediate outputs and long-term system behavior.[8]

-----

-----

### Source [8]: https://orkes.io/blog/human-in-the-loop/

Query: What are the established design patterns and best practices for implementing "human-in-the-loop" (HITL) workflows in practical AI applications, particularly those involving generative models?

Answer: This article from Orkes explains **human-in-the-loop (HITL) in agentic workflows**, focusing on orchestration and autonomy boundaries.[9] It addresses questions like how much autonomy an AI system should have before humans must step in.[9]

The article describes HITL as integrating **human decision points into orchestrated workflows** that coordinate AI agents and backend services.[9] AI performs routine or well-bounded tasks, while humans handle exceptions, sensitive actions, and ambiguous situations.[9]

Key **design patterns and concepts** highlighted include:

- **Approval tasks in workflow engines**: HITL is implemented as explicit human tasks in an orchestration system, where the process pauses until a human completes the task (approve, reject, modify).[9]

- **Exception handling and escalation**: When AI agents encounter errors, unknown states, or rule violations, the workflow routes the case to human operators rather than failing silently.[9]

- **Autonomy thresholds**: Define clear thresholds (value limits, risk categories, policy constraints) that determine when AI can act alone and when HITL is required.[9]

- **Observability and audit**: Orchestration captures detailed logs of both AI actions and human decisions, enabling monitoring, debugging, and compliance.[9]

The article emphasizes best practices:

- Model HITL as a **first-class workflow construct**, rather than embedding checks directly in model code, to keep concerns separated and maintainable.[9]

- Start with more conservative HITL (more checkpoints, lower autonomy) and gradually relax constraints as confidence in the system grows, guided by metrics and incident history.[9]

- Tailor HITL design to **domain risk**: financial, healthcare, and security-related workflows require stricter HITL than low-stakes content generation.[9]

For generative models, the same orchestration patterns apply: content drafts and agent-initiated actions are routed through human approval tasks before being sent externally or committed to critical systems.[9]

-----

</details>

<details>
<summary>How do AI-powered coding assistants like Cursor manage the user interaction loop for code generation and editing?</summary>

### Source [9]: https://cursor.com/docs/agent/overview

Query: How do AI-powered coding assistants like Cursor manage the user interaction loop for code generation and editing?

Answer: Cursor’s **Agent** manages an interactive loop where the user specifies goals in natural language and the agent iteratively plans, executes, and shows changes.[6]

Key interaction elements:

- **Entry point / context**: The Agent is opened in a side pane (Cmd+I / Ctrl+I) and works over the *current workspace*, using open files, project tree, and user instructions as context.[6]
- **High-level task input**: Users describe what they want (e.g., “add OAuth login,” “migrate to FastAPI”), instead of issuing granular edit commands.[6]
- **Planning step**: The Agent “understands your codebase and creates plans for complex tasks,” breaking the request into substeps such as editing specific files, creating new modules, or running tools.[6]
- **Tool calls / environment actions**: Within the loop, Agent can:
  - **Edit files** directly.
  - **Create new files** where needed.
  - **Run terminal commands** (tests, linters, build steps) to validate changes.[6]
- **User-visible diffs**: Code changes are surfaced as edits in the editor; the Agent is designed to “edit code” rather than operate as a black box, so users can inspect and adjust results.[6]
- **Scoped vs broad changes**: In addition to broad project-level tasks, the Agent supports *scoped changes* where the user selects code or a file and gives a localized instruction in natural language; the loop becomes: select scope → describe intent → Agent proposes edits → user reviews/applies.[3][6]
- **Iterative refinement**: After the first pass, users can give follow‑up instructions (e.g., “use async,” “add tests,” “make it more idiomatic”) in the same Agent thread, allowing the model to refine code across multiple iterations while retaining conversation and code context.[6]

Overall, the interaction loop is: user expresses intent → Agent plans → Agent edits code and/or runs commands → user reviews and optionally corrects → user issues follow‑ups, with context accumulating in the Agent pane.[6]

-----

-----

### Source [10]: https://cursor.com/features

Query: How do AI-powered coding assistants like Cursor manage the user interaction loop for code generation and editing?

Answer: Cursor’s feature set reveals how the interaction loop for code generation and editing is structured around **three main modes**: Agent, Scoped Changes, and Tab autocomplete.[3]

- **Agent**: Described as a way to “delegate coding tasks so you can focus on higher-level direction,” this mode centers the loop on natural-language instructions and multi-step autonomous work.[3] Users state goals; the Agent executes a plan by editing files and running commands, then exposes changes for review.

- **Scoped changes**: This feature enables “targeted edits or run terminal commands with natural language.”[3] The loop is:
  - User selects code region or specifies a scope (file, project section).
  - User types an instruction (e.g., refactor, add logging, update API usage).
  - Cursor generates a concrete change proposal.
  - User reviews and applies the change, or adjusts the instruction and retries.
  This keeps the interaction tightly bound to user-chosen context, minimizing unintended edits.[3]

- **Tab (autocomplete)**: Cursor has a “custom autocomplete model [that] predicts your next actions.”[3] During typing, it continuously:
  - Observes recent edits and surrounding code.
  - Generates inline completions (including multi-line suggestions).
  - Lets the user accept with Tab or ignore/overwrite.
  Here, the loop is micro‑interaction level: type → model proposes → user accepts or rejects on a keystroke-by-keystroke basis.[3]

These three surfaces share the same underlying idea: the user controls *granularity* (character-level autocomplete, scoped edits, or project-level agent tasks), while the AI proposes code or actions that are always reviewable before adoption.[3]

-----

-----

### Source [11]: https://www.datacamp.com/tutorial/cursor-ai-code-editor

Query: How do AI-powered coding assistants like Cursor manage the user interaction loop for code generation and editing?

Answer: DataCamp’s guide on Cursor explains several mechanisms Cursor uses to manage the user interaction loop around code generation and editing.[1]

**1. Autocomplete & predictive editing**
- Cursor provides **autocomplete and code prediction** that “predicts multi-line edits and adjusts based on recent changes,” meaning the loop is: user types → model reads nearby context and recent edits → proposes in-line multi-line code → user accepts (Tab) or modifies.[1]
- It performs **code generation** by predicting “what we want to do next and [suggesting] code accordingly,” tightly coupling suggestions to the ongoing editing session.[1]
- **Smart rewrites**: Cursor “automatically correct and improve our code, even if we type carelessly,” so the user writes freely and then accepts or rejects rewrite suggestions.[1]

**2. Inline chat for existing code**
- Users “select the relevant code” and press Cmd+K to open inline chat tied to that selection.[1]
- After describing the change (refactor, fix, explain), Cursor proposes modifications and shows them as a **diff**: red lines for deletions and green for additions.[1]
- The loop is: select → instruct → view diff → accept/adjust → optionally re-prompt with follow‑ups.[1]

**3. Chat features & codebase Q&A**
- Users can ask questions about the codebase; Cursor “will search through the files to provide relevant answers,” then users can turn those answers into concrete edit requests.[1]
- Users can **reference specific blocks or files** in chat, anchoring the model’s context to chosen code regions.[1]

**4. Custom AI rules and models**
- Developers can set **custom AI rules** (e.g., “Always use type hints in Python”), which influence all future generations, so the interaction loop is shaped by persistent, editor-level preferences instead of repeated prompting.[1]
- Users can configure **custom models and API keys**, giving control over which backend model powers the loop.[1]

-----

-----

### Source [12]: https://www.datacamp.com/tutorial/cursor-ai-code-editor

Query: How do AI-powered coding assistants like Cursor manage the user interaction loop for code generation and editing?

Answer: The same DataCamp tutorial also details how Cursor integrates chat-style interaction and editor actions into a continuous loop for code editing and generation.[1]

**Chat-driven development**
- Cursor integrates “advanced chat features” where users converse with the assistant inside the editor.[1]
- Through **codebase answers**, the assistant first retrieves relevant files/snippets to understand context before proposing code, effectively inserting a retrieval step into the loop.[1]
- **Code reference** in chat allows users to paste or link specific snippets; the assistant then responds with explanations, modifications, or new code that aligns with that snippet.[1]

**Interaction with diffs**
- After inline chat instructions, “Code changes in Cursor are presented as a diff,” with clear visual marking of deletions and additions.[1]
- This diff-centric UI gives users a review checkpoint in each loop cycle: they can inspect the transformation before applying, reduce risk of unintended changes, and follow up with corrective prompts if necessary.[1]

**Keyboard-centric triggers**
- AI actions are “seamlessly integrated into the editor” and “trigger[ed] using keyboard shortcuts like Ctrl+K” or by interacting with specific code snippets.[1]
- This keeps the loop low-friction: users stay in the flow of typing, select or highlight code, trigger AI, review output, and continue coding without context switches.[1]

**Autocomplete acceptance model**
- Cursor’s autocompletion operates like familiar IDE completion: while writing, it suggests code and the user can “use the Tab key to incorporate these suggestions.”[1]
- This means every suggestion forms a micro-loop of propose → accept/edit → continue, with the model constantly recalibrating based on the updated buffer.[1]

-----

-----

### Source [13]: https://blog.alexanderfyoung.com/how-to-use-cursor-ai/

Query: How do AI-powered coding assistants like Cursor manage the user interaction loop for code generation and editing?

Answer: Alexander Young’s instructional article (describing Cursor based on official behavior) outlines how different modes structure the user interaction loop.

**Agent Mode (conversational coding)**
- In **Agent Mode**, users “write new code by talking to the AI in plain English.” The loop is: user describes desired behavior → Cursor generates code (functions, classes, features) → user inspects, tests, and then iteratively refines via further natural-language instructions.
- This mode is typically used as a default for new work, where the assistant is expected to handle most of the initial generation before the user fine‑tunes.

**Ask Mode (codebase Q&A)**
- Once a codebase exists, **Ask Mode** lets users query it, e.g., “Where is the authentication function located?” Cursor searches and responds with explanations or pointers.
- The loop becomes: ask about structure/behavior → receive explanation → optionally transition into an edit request (e.g., “now change it to use JWTs”), bridging understanding and manipulation.

**Manual Mode with selective assistance**
- In **Manual Mode**, users mainly write code themselves but can “selectively allow Cursor to assist,” such as invoking completions or inline changes only when needed.
- The loop is user-driven: code manually → highlight or position cursor → request assistance for a specific change → review and accept.

**Image uploads and web search**
- Users can **upload screenshots (UI mockups)** and have Cursor “build pages based on them,” adding an upstream design-to-code step in the interaction loop.
- **Web search** can be invoked to fetch outside examples or documentation that then inform generated code, effectively augmenting the reasoning step before edits are proposed.

Across modes, the article emphasizes that users stay in control by issuing natural-language directions, reviewing generated code, and iterating with follow‑up prompts as needed.

-----

</details>

<details>
<summary>What are common architectural patterns for decoupling interactive AI workflows to support human intervention and feedback?</summary>

### Source [14]: https://aws.amazon.com/blogs/compute/part-2-serverless-generative-ai-architectural-patterns/

Query: What are common architectural patterns for decoupling interactive AI workflows to support human intervention and feedback?

Answer: AWS describes several **serverless generative AI architectural patterns** that explicitly decouple interactive workflows to allow human intervention and feedback.

Key patterns relevant to human-in-the-loop and decoupling:

- **Event-driven, asynchronous orchestration with Step Functions**: User interactions (chat messages, document uploads, task requests) are ingested via API Gateway or AppSync, then persisted (for example in DynamoDB or S3) and placed onto queues or invoked as state machine executions.[7] This separates the *interaction layer* (UI / API) from the *orchestration layer* (Step Functions) and the *model execution layer* (Lambda, Bedrock, SageMaker), so human-facing components are not tightly coupled to model calls.

- **Callback and approval steps in workflows**: Step Functions support `Wait` and callback patterns where a state machine pauses until an external signal is received.[7] This is a standard pattern for **human approval** or review: the LLM produces a draft, the workflow waits while a human inspects or edits it via a separate UI, and then resumes based on the human’s decision.

- **Separate channels for interaction vs. processing**: Real-time interactions use WebSockets / AppSync subscriptions to stream partial responses, while long-running tasks use asynchronous processing with notifications (SNS, EventBridge) when results are ready.[7] This decouples *experience* from *processing* so humans can intervene (cancel, modify, re-run) without being blocked by the backend execution model.

- **Pattern composition**: AWS highlights combining real-time chat patterns with batch or asynchronous patterns to build **review/feedback loops** where outputs are stored, later sampled, and reprocessed for fine-tuning or evaluation.[7] Data collection, evaluation, and retraining pipelines are architected as distinct workflows that consume interaction logs but are not embedded into the request/response path, enabling systematic human feedback processes without degrading latency for end users.

-----

-----

### Source [15]: https://huggingface.co/blog/dcarpintero/design-patterns-for-building-agentic-workflows

Query: What are common architectural patterns for decoupling interactive AI workflows to support human intervention and feedback?

Answer: Hugging Face outlines **design patterns for agentic workflows** that naturally support decoupling and human intervention.

- **Evaluator–Optimizer pattern**: One component (evaluator) assesses outputs from another component (optimizer/agent) using explicit criteria such as correctness, safety, or style.[2] Evaluators can be LLM-based, rule-based, or human; the key architectural pattern is that *evaluation is a separate stage* consuming artifacts from the main workflow, enabling **human-in-the-loop review** before finalizing results.

- **Prompt-Chaining workflow**: Complex tasks are decomposed into multiple sequential prompts where each step has a specific role, input, and output.[2] Because each step is explicitly defined and separated, humans can be inserted at boundaries: e.g., approving intermediate plans, editing a generated outline, or correcting structured intermediate representations before the next step.

- **Parallelization workflow**: Independent subtasks are executed in parallel workers and then aggregated.[2] This pattern decouples subproblems so that human reviewers can focus on selected branches (for example, reviewing one alternative or one aspect) without stopping the whole workflow, or can be used as a voting/arbiter stage after multiple model runs.

- **Routing workflow**: A router decides which model, tool, or workflow branch to use based on the input.[2] Humans can override or guide routing decisions (e.g., escalate to a human expert for certain categories) because the routing logic is centralized instead of being embedded inside prompts.

- **Orchestrator–Workers workflow**: A central orchestrator coordinates multiple specialized agents (workers) that perform subtasks and return results.[2] The orchestrator is a natural integration point for human intervention—humans can modify the task decomposition, inspect worker outputs, or resolve conflicts before the orchestrator produces the final answer. Architecturally, this keeps the UI and human feedback loop loosely coupled from individual model calls while still enabling deep control over the workflow.

-----

-----

### Source [16]: https://www.emergentmind.com/topics/decoupled-explanation-api

Query: What are common architectural patterns for decoupling interactive AI workflows to support human intervention and feedback?

Answer: Emergent Mind describes the **Decoupled Explanation API** pattern for explainable AI, which generalizes to decoupling interactive AI workflows to support human intervention.

- **Core idea**: Separate **prediction** from **explanation** by implementing an independent explanation module (E) that consumes outputs from a predictive model (N) and user specifications (e.g., what type of explanation is desired), without modifying N.[6]

- **Modular workflow**: The architecture is explicitly modular: N performs inference or classification as usual, while E is a separate service or component that can be updated, replaced, or extended independently.[6] This allows human-facing explanation interfaces to evolve (new formats, levels of detail, counterfactuals) without risking changes to the core model behavior.

- **Human-specified explanation requests**: Users (or downstream tools) specify the explanation needs—such as local vs global explanations, feature importance, counterfactual cases—through the Explanation API.[6] Because explanation generation is decoupled, humans can iteratively request different views or refinements over the same prediction history, supporting rich interactive analysis.

- **Non-intrusive intervention**: Since the explainer only *reads* N’s outputs and related metadata, humans can question, audit, or override decisions on the explanation layer (e.g., flag unexpected rationales, request more detail) while the prediction service remains stable and performant.[6]

- **Extensibility and compliance**: The pattern enables plugging in alternative explainers (e.g., surrogate models, attribution methods, rule extractors) for different regulatory or domain needs.[6] For interactive AI systems, this architecture means human regulators, domain experts, or end users interact mainly with the explanation layer, which is explicitly designed for interpretability and feedback, rather than with the low-level prediction pipeline.

-----

-----

### Source [17]: https://www.prompts.ai/en/blog/decoupled-ai-pipelines-dependency-management-best-practices

Query: What are common architectural patterns for decoupling interactive AI workflows to support human intervention and feedback?

Answer: Prompts.ai discusses **decoupled AI pipelines** and dependency management practices that underpin interactive, human-in-the-loop AI systems.

- **Decoupled pipelines**: Workflows are split into independent modules such as data preprocessing, feature engineering, model training, and inference.[3] Each module exposes **well-defined interfaces** instead of direct internal dependencies. This allows UI layers, review tools, or annotation interfaces to integrate at specific points without being tied to internal implementations.

- **Loose coupling via abstractions**: Applying the **Dependency Inversion Principle**, high-level workflow logic depends on abstract interfaces rather than concrete libraries or models.[3] For example, an abstract `ModelService` or `DataLoader` interface hides whether inference is done by a specific LLM provider. Human tools (review dashboards, feedback collectors) can consume these abstractions to inspect or override outputs without breaking the core pipeline.

- **Dependency injection and factories**: Dependencies (models, data sources, tools) are injected rather than created inside components, often via factories.[3] Architecturally, this makes it easy to swap a fully automated component with a human-in-the-loop implementation—e.g., replacing an auto-labeler with a human annotation service that conforms to the same interface.

- **Data lineage and observability**: The article emphasizes tracking data lineage across the pipeline.[3] For interactive workflows, lineage is critical to trace which human feedback or interventions affected which model versions and outputs, enabling impact analysis and controlled rollback.

- **Configuration-driven workflows**: Although implicit, the focus on abstraction and inversion of control supports configuration-based selection of modules.[3] This allows enabling or disabling human review steps (e.g., turning on manual approval in high-risk environments) by configuration rather than code changes, helping organizations scale human involvement based on risk or cost.

-----

-----

### Source [18]: https://paelladoc.com/blog/stop-guessing-5-ai-architecture-patterns/

Query: What are common architectural patterns for decoupling interactive AI workflows to support human intervention and feedback?

Answer: This article presents **five AI architecture patterns** that address context handling and separation of concerns, which can be adapted to decouple interactive workflows and enable human feedback.

Relevant patterns:

- **Context-Aware Monolith**: The application directly manages context (session data, user history, previous prompts and outputs) inside its own code or dedicated tables.[1] While not highly decoupled, this pattern makes human review easier by centralizing context in one place so that auditors or moderators can inspect entire interaction histories.

- **Decoupled Context Pipeline**: As complexity grows, context management is extracted into a separate pipeline that aggregates, enriches, and serves context from multiple sources (user inputs, databases, external APIs) to models and agents.[1] This separation allows human tools (e.g., knowledge editors, governance UIs) to modify or validate context independently of application logic or model invocation.

- **Foundation vs. Application/Task layers**: The architecture distinguishes a **foundation layer** (general-purpose AI models) from an **application/task layer** containing prompts, business logic, and orchestration.[1] Human intervention typically happens in the application layer—editing prompts, adjusting rules, curating tools—without touching the underlying foundation models.

- **Intent-driven architecture**: The application layer interprets user intent and orchestrates model calls, tools, and context assembly.[1] Because intent handling and routing are explicit, humans can influence or override how intents are mapped to actions (for example, redirecting certain intents to human operators) without restructuring the model layer.

- **Internal vs external context integration**: The article contrasts integrating context directly in the app versus via a pipeline; this is effectively a choice about how tightly user interactions are bound to AI components.[1] Moving to a decoupled context pipeline is a step toward more modular, governable interactive AI systems where feedback and intervention can be inserted at the context and orchestration layers rather than only at the UI.

-----

-----

### Source [19]: https://www.catio.tech/blog/emerging-architecture-patterns-for-the-ai-native-enterprise

Query: What are common architectural patterns for decoupling interactive AI workflows to support human intervention and feedback?

Answer: Catio discusses **emerging architecture patterns for AI-native enterprises**, focusing on multi-agent systems and AI-orchestrated workflows that lend themselves to human intervention.

- **Multi-agent workflows**: The system uses multiple specialized agents to reason about complex domains (e.g., cloud environments, dependencies, tradeoffs), each with a specific role such as interpreting topology, identifying risks, or generating recommendations.[5] Agents communicate and may disagree, which is considered a useful feature for surfacing uncertainty.[5] This architecture naturally supports humans as additional "agents" who arbitrate disagreements or provide domain-specific judgment.

- **LLM-as-interface pattern**: LLMs act as the front-end interface to complex systems, translating natural language into structured operations while backend services remain decoupled.[5] Humans interact through the LLM layer, while orchestration, tools, and data remain modular; this makes it straightforward to log interactions, insert review gates, or replay scenarios for audit.

- **AI-orchestrated workflows**: Instead of embedding AI directly into business services, an orchestration layer handles routing, memory, context retrieval, and tool execution.[5] This layer is where human feedback can be integrated—for example, approving recommended actions, editing generated configurations, or adjusting risk thresholds—without modifying individual tools or models.

- **Infrastructure around models**: The article stresses investments in routing, memory, context retrieval, and tool execution as distinct concerns from the core models.[5] These infrastructural components can expose administrative and review interfaces for humans, like memory editors, policy routers, or tool whitelists, enabling governance and oversight.

- **Disagreement as a feature**: When agents disagree, it highlights ambiguity or risk.[5] Architecturally, exposing these disagreements to humans (e.g., in a decision dashboard) creates explicit intervention points: humans can compare rationales and decide which agent to follow or whether to trigger additional checks.

-----

</details>

<details>
<summary>What are the most effective UX/UI principles for designing interfaces that facilitate rapid human validation of AI-generated content?</summary>

### Source [20]: https://www.uxness.in/2024/06/7-principles-of-ux-design-for.html

Query: What are the most effective UX/UI principles for designing interfaces that facilitate rapid human validation of AI-generated content?

Answer: This article defines **7 UX principles for AI-driven products** that strongly support rapid human validation of AI-generated content.

It first recommends **visually differentiating AI‑generated results** from human‑generated data so users always know what needs validation. This can be done with distinct labels, colors, or iconography to signal "AI output" versus "original source," helping users judge reliability quickly.

The second principle is to **involve users in AI learning**. The interface should make it easy to correct outputs (e.g., edit extracted fields, re‑label documents) and use those corrections to retrain or fine‑tune the model. Clear, inline controls near each AI result let users validate or fix content without leaving their workflow, making human‑in‑the‑loop review efficient.

The article stresses **user control**: AI assists but does not replace decision‑making. Users need obvious ways to override AI suggestions, disable automation, or choose alternative options. For rapid validation, this means simple actions like “accept all,” “accept selected,” or “revert to original,” plus the ability to configure how aggressively AI intervenes.

It highlights **robust error handling**: when AI is wrong or uncertain, the UI should communicate the issue clearly and guide users to resolution, for example by surfacing confidence indicators, flagging low‑quality results, and providing direct affordances to correct or re‑run the analysis.

A dedicated **feedback loop** is required: the product should continually collect structured user feedback on AI decisions via quick ratings, flags, or comments and feed this back into model improvement. This both speeds future validation and increases perceived control.

Finally, **trust and transparency** are emphasized: clearly explain capabilities and limitations, what data is used, and how decisions are made. Onboarding, help content, and inline explanations reduce cognitive effort when users evaluate AI outputs and decide whether to trust, edit, or discard them.[2]

-----

-----

### Source [21]: https://www.aufaitux.com/blog/ai-interface-usability-principles/

Query: What are the most effective UX/UI principles for designing interfaces that facilitate rapid human validation of AI-generated content?

Answer: This article outlines **10 usability principles for AI interfaces** that are directly applicable to interfaces for validating AI‑generated content.

It emphasizes **clarity and transparency**: the UI should make the AI’s role and decision process understandable. For content validation, this means showing why a particular result was generated (e.g., “based on your past searches” or underlying data features), and using visual indicators such as **confidence levels** or explanation chips next to each AI output so reviewers can prioritize scrutiny.

The principle of **predictability and consistency** states AI behavior must avoid surprises. Interfaces should keep terminology, visual patterns, and interaction flows consistent so users can quickly learn where to look to check, correct, or approve content. This reduces time spent decoding the UI rather than judging the content.

The article stresses **minimal cognitive load**: AI outputs should be simple to understand, with only relevant information visible. For rapid validation, this implies summarizing long AI outputs, highlighting key differences from the source, and avoiding overloading screens with secondary metrics or verbose explanations.

Ethical and bias considerations are discussed under **ethical and bias‑free AI**. The UI should provide ways for users to **correct or dispute AI decisions**, which can be integrated as quick actions (“mark as biased,” “incorrect extraction”). This helps reviewers focus on problematic content and improves future performance.

The section on **balancing automation and human intervention** recommends allowing users to override AI decisions, tweak responses, and control automation levels. For validation workflows, this means explicit controls to switch between auto‑apply and manual review modes and to rollback AI‑applied changes.

On **tools and techniques**, the article recommends embedding **user feedback mechanisms** directly in AI interfaces so users can validate or correct responses inline, and using **A/B testing** and think‑aloud studies to optimize how explanations, confidence, and correction controls are presented for the fastest, most accurate human review.[1]

-----

-----

### Source [22]: https://www.eleken.co/blog-posts/ai-usability-principles

Query: What are the most effective UX/UI principles for designing interfaces that facilitate rapid human validation of AI-generated content?

Answer: This source presents **9 AI usability principles** and concrete patterns that help users understand and supervise AI, which are crucial for rapid human validation.

It recommends first **mapping existing workflows and mental models** to understand how users currently review and correct content. Interfaces should integrate AI into familiar tools (e.g., a text editor or document viewer) instead of forcing context switches, so validation is done in-place with minimal friction.

The principle of **consistency and standards** states that language, visuals, and interaction patterns must be consistent. For validation UIs, this means using the same patterns for suggestions, highlights, and error states throughout the product, so users immediately recognize what is editable, suggested, or confirmed.

The article advocates **lightweight controls and feedback loops**. Users should be able to intervene, adjust, or stop AI actions with minimal effort. In practice this includes inline suggestion chips with actions such as **accept**, **ignore**, or **see explanation**, plus the ability to edit AI outputs directly. When the model is uncertain, the UI should surface that state (e.g., with an icon or label) and invite human review.

It stresses the need to **prototype early and test with real users**, even with a dummy model, to validate how people respond to uncertain or partially correct outputs. Metrics like user trust, comprehension, and error‑recovery time are recommended to evaluate and refine validation flows.

The article provides an example of AI grammar assistance: suggestions appear within the text, use familiar color cues (e.g., red for errors, green for accepted corrections), and always give the user a clear, reversible choice. This pattern—embedded, clearly marked suggestions with one‑click acceptance or dismissal—directly supports fast, accurate human validation of AI‑generated or AI‑modified content.[3]

-----

-----

### Source [23]: https://www.uxstudioteam.com/ux-blog/ai-ux-5f836

Query: What are the most effective UX/UI principles for designing interfaces that facilitate rapid human validation of AI-generated content?

Answer: This article lists **10 AI UX principles** for designing effective AI products, many of which apply to UIs for validating AI outputs.

It begins with **identifying a real user need** and **scoping AI into smaller bites**. For validation workflows, this implies targeting concrete review tasks—such as checking extracted fields, summaries, or classifications—and breaking them into atomic units the UI can present and confirm individually, accelerating focused human review.

The article stresses **setting clear expectations** about what the AI can and cannot do. Interfaces should communicate limitations and typical error modes so reviewers know what to double‑check. This can be done via short descriptions, hints, and examples near AI features.

On **data collection**, it recommends collecting the right data transparently and responsibly, explaining what data is used to generate results. In validation tools, linking AI outputs back to source data (e.g., highlighting the specific parts of a document that led to a summary) enables rapid cross‑checking.

The principle **“tell users how it works – whenever possible”** encourages high‑level explanations of the model’s behavior. Even simple model cards, confidence indicators, or reason phrases near AI outputs can help reviewers quickly judge whether extra scrutiny is needed.

The article highlights **proactive error management**: anticipate common failure modes and design clear, guided recovery paths. For content validation UIs, this means making it obvious how to correct an output, undo changes, or request a re‑run, and surfacing errors or low‑confidence predictions where the user’s attention naturally goes.

Further, it emphasizes **clear and simple language**, **giving the user control**, and **collecting user feedback**. Users should be able to easily override AI, adjust how strongly it intervenes, and provide quick feedback on accuracy. Finally, it recommends making it **safe and easy to experiment**, so users feel comfortable exploring AI suggestions knowing they can always revert to an earlier state.[4]

-----

</details>

<details>
<summary>Why is the evaluator-optimizer pattern, especially when combined with human feedback, considered a robust method for ensuring AI-generated content adheres to quality standards?</summary>

### Source [24]: https://icepick.hatchet.run/patterns/evaluator-optimizer

Query: Why is the evaluator-optimizer pattern, especially when combined with human feedback, considered a robust method for ensuring AI-generated content adheres to quality standards?

Answer: The Icepick documentation describes the **evaluator-optimizer** as an iterative pattern where one component **generates content** and another **evaluates it and provides feedback**, forming a loop that continues until the evaluator is satisfied or a **maximum iteration limit** is reached.[1] This trades extra computation for **higher quality and more reliable results**, because each cycle explicitly incorporates feedback to correct defects and improve clarity, style, or adherence to constraints.[1]

It is considered robust for enforcing quality standards because:
- It requires **clear evaluation criteria**, so the evaluator can systematically check whether outputs meet quality, safety, or task-specific requirements.[1]
- The **controlled iteration loop with termination criteria** (either evaluator approval or max iterations) prevents unbounded refinement while still allowing multiple chances to fix issues.[1]
- Each new attempt is **conditioned on previous feedback and previous outputs**, enabling targeted corrections rather than starting from scratch, which improves convergence toward the desired standard.[1]
- The evaluator can encode **policy and quality rules** (e.g., appropriateness, length limits, tone, or domain constraints) and enforce them consistently across iterations.[1]

The docs note this pattern is especially suitable when initial attempts can be **measurably improved through iteration**, such as creative content, code optimization, or other tasks where quality can be judged against explicit criteria and refined step by step.[1] It is less suitable if evaluation criteria are highly subjective or inconsistent, or if the cost of multiple iterations outweighs the quality benefits.[1] By making the evaluation step explicit and repeatable, the pattern naturally supports integration of **structured human feedback** (e.g., humans setting or adjusting the evaluator’s criteria, or reviewing evaluator decisions) to keep the system aligned with human-defined quality standards.[1]

-----

-----

### Source [25]: https://docs.praison.ai/docs/features/evaluator-optimiser

Query: Why is the evaluator-optimizer pattern, especially when combined with human feedback, considered a robust method for ensuring AI-generated content adheres to quality standards?

Answer: PraisonAI describes the **Agentic Evaluator-Optimizer** as a pattern that enables **iterative solution generation and refinement**, **automated quality evaluation**, **feedback-driven optimization**, and **continuous improvement loops**.[2] A generator agent creates solutions based on requirements, while an evaluator agent **assesses solution quality and completeness** and drives a **feedback loop** that decides whether more refinement is needed or the process can stop.[2]

Robustness for quality control comes from several structural properties:
- The evaluator applies **explicit evaluation logic** to determine if a solution is "done" or if **"more"** improvement is required, directly encoding quality thresholds and acceptance criteria.[2]
- The workflow is set up as tasks where the **"generate"** task feeds into an **"evaluate"** task, and the evaluation’s decision controls whether the generator is called again, forming a **closed-loop controller** around quality.[2]
- Because evaluation is automated and repeatable, the system can **consistently enforce standards** across many runs without relying on a single pass from the generator.[2]

When combined with human feedback, the pattern becomes more robust because humans can:
- Define or adjust the **evaluation criteria and feedback format** used by the evaluator agent, ensuring they reflect organizational or domain-specific quality standards.[2]
- Inspect or override evaluator decisions, e.g., tightening criteria for sensitive domains or relaxing them for exploratory tasks.[2]

By centralizing quality logic in the evaluator and linking it through an explicit feedback loop to the generator, PraisonAI’s design makes it straightforward to embed **human-authored policies and review rules** into the evaluator, turning the evaluator-optimizer loop into a practical mechanism for enforcing human-defined quality and safety standards at scale.[2]

-----

-----

### Source [26]: https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns

Query: Why is the evaluator-optimizer pattern, especially when combined with human feedback, considered a robust method for ensuring AI-generated content adheres to quality standards?

Answer: Spring’s "Building Effective Agents" article explains that the **Evaluator-Optimizer pattern** uses a **dual-LLM process**: one model **generates responses**, while another **evaluates and provides feedback in an iterative loop**, analogous to a human writer working with an editor.[3] The workflow generates an initial solution, evaluates it, and either **returns it if it passes** or **incorporates feedback to generate an improved solution**, repeating until the result is satisfactory.[3]

This structure is robust for quality assurance because:
- It separates concerns between **generation** and **evaluation**, allowing the evaluator to specialize in checking correctness, adherence to constraints, and quality dimensions independent of the generator’s creativity.[3]
- The evaluator effectively encodes **acceptance tests**: it can label outputs as PASS or NEEDS_IMPROVEMENT, and its feedback guides targeted revisions rather than random retries, leading to **progressive refinement toward quality standards**.[3]
- The iterative loop mimics **established human workflows** (writer–editor, developer–code reviewer), which are known to produce higher quality outputs through successive review cycles.[3]

Spring notes that this pattern is particularly useful when **single-pass generation is unreliable** and when the desired outputs must meet **high standards of correctness or alignment**.[3] Because the pattern is implemented as a workflow around LLM calls, it naturally supports integrating **human feedback** at multiple points: humans can define evaluation prompts and criteria, inspect evaluation outcomes, tighten or relax pass thresholds, or intervene between iterations.[3] This makes the evaluator-optimizer loop a strong foundation for systems where AI outputs must be systematically checked and improved until they satisfy **explicit, human-defined quality bars**.[3]

-----

-----

### Source [27]: https://javaaidev.com/docs/agentic-patterns/patterns/evaluator-optimizer

Query: Why is the evaluator-optimizer pattern, especially when combined with human feedback, considered a robust method for ensuring AI-generated content adheres to quality standards?

Answer: Java AI Dev describes the **Evaluator-Optimizer** pattern as a way for an LLM to **improve the quality of a previous generation using feedback from an evaluator** in a loop that can run multiple times.[4] The pattern is decomposed into subtasks: initializing input, generating an initial result, **evaluating the result and providing feedback**, **optimizing the result based on the feedback**, and optionally finalizing the response.[4]

It is considered robust for enforcing quality standards because:
- The evaluator subtask explicitly **checks whether the result passes evaluation** and produces **structured feedback** when it does not, which the optimizer uses to revise the content.[4]
- The loop continues until the result **passes the evaluation** or a **maximum number of evaluations** is hit, providing a clear balance between quality and latency/cost.[4]
- The documentation recommends using **different models for generation and evaluation** (e.g., one strong model as generator and another as evaluator) to obtain more reliable and less correlated judgments, strengthening the quality gate.[4]

The pattern’s design naturally supports human-in-the-loop control:
- Humans can define the **evaluation prompts and criteria** that drive the evaluator, embedding organizational coding standards, safety rules, or style guides into the evaluation step.[4]
- The **max number of evaluations** and pass/fail conditions reflect human risk and quality preferences; they can be tuned to favor stricter or more permissive behavior.[4]

In code-optimization examples, the evaluator identifies issues and the optimizer is instructed to **"address all concerns in the feedback"**, which enforces a rigorous response to detected problems before an output is accepted.[4] This explicit, repeatable evaluation-and-repair loop is what makes the evaluator-optimizer pattern a strong method for ensuring AI-generated content meets predefined quality criteria.[4]

-----

-----

### Source [28]: https://github.com/BootcampToProd/spring-ai-evaluator-optimizer-workflow

Query: Why is the evaluator-optimizer pattern, especially when combined with human feedback, considered a robust method for ensuring AI-generated content adheres to quality standards?

Answer: The Spring AI Evaluator Optimizer Workflow repository presents the pattern as a **self-improving AI system** that achieves **high-quality output through iterative refinement**.[5] It uses two specialized LLM roles: a **"Generator" (Writer)** that creates initial content and an **"Evaluator" (Editor)** that **critiques and guides improvements** until the output satisfies **predefined quality standards**.[5]

This implementation highlights several reasons the pattern is robust for quality enforcement:
- The evaluator acts like an **editor enforcing a style guide and requirements**, repeatedly reviewing drafts and providing targeted feedback that the generator must address.[5]
- Refinement continues **until the content meets the specified quality bar**, making acceptance contingent on passing an explicit quality gate, not just on a single model sample.[5]
- The iteration captures a **"chain of improvement"**, where each revision is informed by prior critiques, leading to more polished and compliant results.[5]

Because the pattern is configured via prompts and workflow logic, it is straightforward to embed **human feedback** into it:
- Humans can define the **quality standards, acceptance criteria, and editorial guidelines** that the evaluator enforces, including tone, structure, correctness constraints, or compliance rules.[5]
- Human reviewers can inspect evaluator feedback or final outputs to further refine the prompts and decision thresholds, closing the loop between automated evaluation and human judgment.[5]

By framing the evaluator as an editor applying **predefined human-authored standards**, and the generator as a writer responding to those standards through multiple drafts, this implementation shows how the evaluator-optimizer loop operationalizes human quality expectations into a robust, repeatable workflow for AI-generated content.[5]

-----

-----

### Source [29]: https://dev.to/clayroach/building-self-correcting-llm-systems-the-evaluator-optimizer-pattern-169p

Query: Why is the evaluator-optimizer pattern, especially when combined with human feedback, considered a robust method for ensuring AI-generated content adheres to quality standards?

Answer: The "Building Self-Correcting LLM Systems" article explains why the **evaluator-optimizer** approach is effective, using LLM-generated SQL as a case study.[6] In this setup, a non-LLM system (ClickHouse) **evaluates** each query by checking whether it executes or raises a specific error code, and an LLM **optimizes** the query based on those concrete errors.[6]

The article identifies several reasons this pattern is robust:
- **Clear evaluation criteria**: success is binary and objective (does the SQL execute correctly), which makes the evaluator a reliable quality gate.[6]
- **Demonstrable improvement per iteration**: each loop focuses on fixing a specific, identified issue rather than regenerating queries blindly, leading to systematic error reduction.[6]
- **Context preservation**: the original analysis intent is preserved while only the syntax is corrected, reducing the risk of drifting away from the user’s goals.[6]
- **Cost efficiency and reliability**: by targeting corrections and learning from recurring error patterns, the system becomes more production-ready with fewer wasted calls.[6]

The article links this design to Anthropic’s evaluator-optimizer pattern: **one component evaluates**, another **optimizes**, iterating until success, without needing model retraining—only **real-time coaching** from the evaluator.[6]

When combined with human feedback, the same structure allows humans to:
- Define what counts as success or acceptable performance (e.g., stricter execution constraints, safety filters).[6]
- Add **rule-based fallbacks** based on recurring error patterns, encoding human expertise directly into the evaluation/optimization process.[6]

This demonstrates that when evaluation is **objective, repeatable, and tightly connected to optimization**, the evaluator-optimizer pattern becomes a robust, engineering-style mechanism for ensuring AI outputs satisfy strict quality and correctness requirements.[6]

-----

-----

### Source [30]: https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-evaluators-and-reflect-refine-loops.html

Query: Why is the evaluator-optimizer pattern, especially when combined with human feedback, considered a robust method for ensuring AI-generated content adheres to quality standards?

Answer: AWS Prescriptive Guidance describes **evaluator workflows and reflect-refine loops** as patterns where **one LLM generates a result and another evaluates or critiques it**, providing a feedback loop that promotes **self-reflection, optimization, and iterative improvements**.[7] This workflow is recommended for situations where **output quality, accuracy, and alignment are important**, and where **single-pass generation is unreliable or insufficient**.[7]

The robustness for enforcing quality standards comes from:
- A dedicated evaluator that **systematically critiques outputs**, enabling models to **self-critique, iterate, and refine** their responses.[7]
- Iterative refinement aimed at meeting a **higher standard of correctness or exploring improved alternatives**, so content is not accepted until it better aligns with quality or alignment goals.[7]
- Suitability for domains where agents must **meet higher correctness standards or alignment constraints**, such as safety-sensitive or policy-constrained applications.[7]

This pattern also integrates naturally with **human feedback**:
- Humans can define the **evaluation criteria, prompts, and refinement objectives** that the evaluator model uses, effectively encoding organizational quality and safety policies into the workflow.[7]
- Human reviewers can intervene in or configure the reflect-refine loops, adjusting how aggressively the system refines outputs or what constitutes acceptable alignment and correctness.[7]

By clearly separating **generation** from **evaluation/critique** and coupling them through explicit feedback-driven loops, the AWS guidance presents evaluator workflows and reflect-refine loops as a robust mechanism to systematically **raise the quality and alignment** of AI-generated content beyond what a single pass can typically achieve.[7]

-----

-----

### Source [31]: https://dylancastillo.co/til/evaluator-optimizer-pydantic-ai.html

Query: Why is the evaluator-optimizer pattern, especially when combined with human feedback, considered a robust method for ensuring AI-generated content adheres to quality standards?

Answer: Dylan Castillo’s note on the evaluator-optimizer workflow with Pydantic AI describes the pattern as having an **LLM generator** that produces a solution and an **LLM evaluator** that checks whether the solution is acceptable; if not, the system **iterates**.[8] The evaluator decides if the output is good enough or if it needs improvement, and the generator then produces a revised answer conditioned on the evaluator’s feedback.[8]

It is considered robust for quality because:
- The evaluator enforces **clear criteria for acceptance**, ensuring outputs are not returned to users unless they satisfy specified conditions.[8]
- The pattern uses **structured evaluation and feedback**, which is well suited to Pydantic’s data models, enabling the system to check for schema correctness, completeness, or other structured constraints before accepting an answer.[8]
- Iteration allows the model to **repair its own mistakes** under guidance rather than relying on a single-shot guess, leading to more reliable final outputs.[8]

Incorporating human feedback strengthens this pattern: humans can define the **evaluation rules, schemas, and thresholds** the evaluator uses, or update them as requirements evolve.[8] Because evaluation and optimization are split, it is straightforward for humans to modify only the evaluator’s logic (what constitutes quality) without changing the generator, making the system adaptable to new policies or quality bars while retaining the same base model.[8] This design turns evaluator-optimizer into a practical method for enforcing human-defined quality standards around AI-generated content, particularly in applications that depend on structured, schema-valid outputs.[8]

-----

</details>

<details>
<summary>What are the core principles of the AI generation/human validation loop as a model for human-AI collaboration, particularly as described by Andrej Karpathy?</summary>

### Source [32]: https://www.latent.space/p/s3

Query: What are the core principles of the AI generation/human validation loop as a model for human-AI collaboration, particularly as described by Andrej Karpathy?

Answer: Karpathy’s **generation ↔ verification loop** is a core pattern in his “Software 3.0 / Partial Autonomy” model.

He frames modern AI systems as **partially autonomous agents** that must be tightly integrated into workflows where a **human verifies** what the AI generates at every meaningful step.[1] The loop is: the AI **generates** candidate outputs (code, text, plans, actions), and the human **verifies / edits / approves** them before they affect the real world.[1]

Karpathy stresses that the **entire product** must be designed around this loop, not just the model:[1]
- The goal is a **fast, low‑friction workflow** that lets human and AI iterate quickly.
- This enables “**partial autonomy**”: the AI does real work but the human remains final authority.[1]

He highlights two **core principles** for this loop:[1]
- **Improve verification: make it easy, fast to win.**
  - The UI and tooling must make it *trivial* for humans to inspect, compare, and accept/reject AI output.
  - Good products expose clear, structured diffs, checks, tests, and visualizations that map onto human strengths.

- **Improve generation: keep AI on a tight leash.**
  - Avoid unconstrained, long‑horizon autonomous behavior.
  - Break work into smaller steps; after each, return to the human for verification before proceeding.[1]

He connects this to his **“Iron Man suit”** analogy: AI augments humans with new capabilities and speed, but the **human stays in the loop** steering and validating.[1] This is necessary because of the gap between **“demo” and “product”**—LLMs can create impressive one‑off demos, but reliable products require that *all* cases work, which in practice still needs human oversight and verification at the loop level.[1]

Overall, for Karpathy the generation/verification loop is the pragmatic, safety‑preserving pattern for human‑AI collaboration in the “decade of agents.”[1]

-----

-----

### Source [33]: https://catalaize.substack.com/p/andrej-karpathy-software-is-changing

Query: What are the core principles of the AI generation/human validation loop as a model for human-AI collaboration, particularly as described by Andrej Karpathy?

Answer: This article distills Karpathy’s **human‑AI generation/verification loop** as a design principle for AI products and collaboration.

It explains that effective human‑AI collaboration requires **optimizing the feedback loop between generation and verification**.[2] Karpathy’s two explicit principles are:

1. **Make verification fast and easy.**[2]
   - Interfaces should help humans **audit AI suggestions quickly**, not read long unstructured outputs.
   - Examples:
     - An AI coding assistant should show **colored diff views** so developers can see changes at a glance and accept/reject with a click.[2]
     - A research assistant should attach **source citations** to each factual claim so users can instantly verify.[2]
   - These choices leverage human strengths (visual pattern recognition, judgment) to counter AI weaknesses.[2]

2. **Keep the AI on a tight leash.**[2]
   - Fully free‑roaming agents that produce huge outputs or execute long chains of actions tend to **drift off course and make harmful errors**, overwhelming the human with inscrutable results.[2]
   - It is more effective to have the AI work in **small, controlled increments**, pause for human verification, then continue.[2]
   - A “tight leash” means constrained autonomy, clear boundaries, and frequent checkpoints where the human reviews and corrects.[2]

The article situates this loop within **partial autonomy / Iron Man suit** products:[2]
- The AI performs labor‑intensive subtasks, but a **human supervises and guides**.
- The system includes an **“autonomy slider”**: as tooling, interfaces, and reliability improve, more can be delegated, but the loop of generate → verify remains central.[2]

Karpathy characterizes this as the **practical path for the “decade of agents”**: focus less on full autonomy and more on **copilots**, human‑centered interfaces, and workflows that repeatedly cycle AI generation through human verification to maintain quality and safety while gaining speed.[2]

-----

-----

### Source [34]: https://www.youtube.com/shorts/hTT5o2AoewQ

Query: What are the core principles of the AI generation/human validation loop as a model for human-AI collaboration, particularly as described by Andrej Karpathy?

Answer: In this short clip, Karpathy underscores why **humans must remain in the loop** when working with AI systems, which directly reinforces his generation/verification loop idea.

He emphasizes the importance of **GUIs and interaction surfaces** that let people effectively supervise AI, rather than relying on hidden or fully autonomous agent behavior.[4] The need for **clear user interfaces** is tied to ensuring that humans can **see, understand, and correct** what the AI is doing.[4]

This aligns with his view that AI systems should operate under **partial autonomy**, where the AI proposes actions or results and the **human validates** them before they are finalized.[4] By stressing “we still need humans in the loop,” he is pointing to the necessity of **human verification** as a structural part of any responsible AI workflow, not an optional afterthought.[4]

Although the clip is brief, its main message is that **AI agents cannot yet be trusted to run fully on their own**, and that product design must support **oversight, inspection, and intervention** by human users.[4] This is precisely the human‑validation side of his broader **AI generation/human verification** loop model for collaboration.[4]

-----

-----

### Source [35]: https://www.youtube.com/shorts/lZZZjaQepHI

Query: What are the core principles of the AI generation/human validation loop as a model for human-AI collaboration, particularly as described by Andrej Karpathy?

Answer: This short reinforces why Kurpathy’s **generation/human‑validation loop** is necessary, by referencing his notion of **“jagged intelligence.”**

Karpathy describes current AI models as exhibiting **uneven, jagged capabilities**: they can be superhuman at some tasks yet fail catastrophically at others that appear nearby in difficulty.[6] Because it is often non‑obvious *when* they will fail, **human validation** is essential.[6]

This motivates a workflow where the AI **generates** outputs, but **humans must verify** them systematically, especially for high‑stakes or complex tasks.[6] Rather than granting blanket trust, we need a loop in which people review, correct, and learn where the model is reliable vs. brittle.[6]

The clip also alludes to a broader **culture change and validation loops**: builders should design systems and organizations around this constant **generate → validate → refine** pattern, integrating tests, reviews, and checks as standard practice when using AI.[6]

In this way, Karpathy’s jagged‑intelligence argument is an underlying rationale for his collaboration model: because capability is not smooth or predictable, the **core principle** becomes **AI generation under human supervision, with tight feedback and correction** rather than unconstrained autonomy.[6]

-----

</details>

<details>
<summary>What are common architectural patterns for decoupling interactive AI workflows to support distinct human-in-the-loop editing tasks, such as 'edit whole document' versus 'edit selected text'?</summary>

### Source [36]: https://aws.amazon.com/blogs/compute/part-2-serverless-generative-ai-architectural-patterns/

Query: What are common architectural patterns for decoupling interactive AI workflows to support distinct human-in-the-loop editing tasks, such as 'edit whole document' versus 'edit selected text'?

Answer: AWS describes several **serverless generative AI architectural patterns** that explicitly decouple user interaction modes and support human‑in‑the‑loop review.[6]

For **interactive editing experiences**, they recommend separating:

- **Front-end interaction modes** (for example, *full-document* versus *selected‑span* edit UIs) from the **back-end orchestration** layer.[6]
- A **request router/orchestrator** (often implemented with AWS Step Functions or an API Lambda) that interprets intent ("edit whole document" vs. "revise highlighted text") and selects different **workflows** or prompt templates accordingly.[6]
- **Model invocation** (Amazon Bedrock, SageMaker, or third‑party LLMs) from the application logic, so the same orchestration can route to different models or prompt chains depending on task scope and risk.[6]

To support **human‑in‑the‑loop (HITL)** editing, AWS highlights patterns where:

- The LLM generates a **proposed change set** (diff or structured patch) rather than directly mutating the canonical document; this proposal is persisted (for example in DynamoDB or S3) and surfaced to the user for approval.[6]
- Separate **approval workflows** (Step Functions, EventBridge) gate whether edits are applied, enabling different policies for global document rewrites versus local text edits.[6]
- **Event‑driven decoupling** is used: UI posts an edit request event; asynchronous processing (context assembly, model call, validation) occurs; the user is later notified with suggested edits for acceptance or modification.[6]

AWS also recommends:

- Using **context-assembly components** that can be reused across workflows (e.g., assembling “whole document” context vs. “local window + document metadata” for selected‑text edits).[6]
- Employing **guardrail services** (safety filters, PII detection, policy checks) as separate functions in the workflow, which can be invoked at different strengths/steps depending on edit scope.[6]
- Designing **idempotent, stateless model-invocation Lambdas** and **stateful workflow/state machines** to keep interaction logic testable and versioned across multiple editing patterns.[6]

-----

-----

### Source [37]: https://codeconductor.ai/blog/build-ai-project-that-dont-fail/

Query: What are common architectural patterns for decoupling interactive AI workflows to support distinct human-in-the-loop editing tasks, such as 'edit whole document' versus 'edit selected text'?

Answer: This article advocates a **decoupled AI architecture** where the LLM is separated from the execution layer, which maps directly to supporting different editing tasks like *whole‑document* versus *selected‑text* edits.[1]

Key pattern:

- The **LLM returns structured intent**, not the final action.[1]
- A separate **execution engine** interprets that intent and applies changes to the document model under strict rules.[1]

Applied to editing workflows:

- For *edit whole document*, the LLM might output a high‑level plan or a list of transformation operations over document sections.[1]
- For *edit selected text*, it returns a localized operation (e.g., replace range X–Y with text Z), but both share the same execution engine and document model.[1]

The execution layer enables:

- **Human‑in‑the‑loop control**: LLM proposes an edit; the execution layer validates constraints (style, length, policy) and can insert human approval checkpoints before applying modifications.[1]
- **Versioned, testable behavior**: structured intents and deterministic execution allow unit and regression tests for each edit type, making it safe to iterate prompts and models without breaking workflows.[1]
- **Independent lifecycle**: UI changes (different editing tools), model upgrades, and business‑rule changes can evolve independently, as long as the contract for structured intents is honored.[1]

For interactive products, the article stresses designing **different workflows per interaction mode** while sharing infrastructure:

- UI → Orchestrator → LLM (intent) → Execution Engine → Document Store.[1]
- The orchestrator distinguishes between commands such as "rewrite paragraph" vs. "rewrite entire document" and selects different prompt templates and validation rules, but reuses the same underlying agent and execution logic.[1]

This decoupling also supports **enterprise‑grade guarantees** like auditability and deterministic fallbacks, which are crucial when humans review and accept AI‑proposed edits.[1]

-----

-----

### Source [38]: https://paelladoc.com/blog/stop-guessing-5-ai-architecture-patterns/

Query: What are common architectural patterns for decoupling interactive AI workflows to support distinct human-in-the-loop editing tasks, such as 'edit whole document' versus 'edit selected text'?

Answer: This source outlines **AI architecture patterns** useful for separating concerns in interactive AI products, especially where multiple types of editing tasks share infrastructure.[2]

Relevant patterns:

- **Context‑Aware Monolith:** Context management (user history, previous prompts/outputs) is integrated directly into the main application logic.[2] For editing workflows, the monolith tracks document versions, selections, and user preferences to provide appropriate context to the model for both full‑document and selected‑text edits.[2]

- **Decoupled Context Pipeline:** When context becomes complex, context processing is extracted into a distinct pipeline.[2] This pipeline aggregates and enriches context from multiple sources (document store, selection metadata, user profile) and exposes a consistent interface to AI components.[2] Whole‑document edits request broad context; selected‑text edits request a local window plus global metadata, but both are served by the same pipeline.[2]

- **Foundation vs. Application/Task Layer:** A **foundation layer** hosts general‑purpose models.[2] An **application/task layer** implements task‑specific logic, prompt templates, and orchestration.[2] This allows separate chains or prompts for "rewrite entire document" and "rewrite highlighted text" while reusing foundation models and context services.[2]

These patterns support **decoupling of workflows**:

- The UI or API expresses high‑level intent (edit mode).
- The application layer chooses the appropriate **task‑specific workflow** (prompt, validation, post‑processing) for that intent.
- The foundation models and context pipeline remain shared infrastructure.

By cleanly separating **context assembly**, **task workflows**, and **models**, teams can evolve each edit mode independently (e.g., tighter guardrails for whole‑document rewrites, lighter ones for local edits) while maintaining consistency and observability across the system.[2]

-----

-----

### Source [39]: https://aipmguru.substack.com/p/ai-architecture-patterns-101-workflows

Query: What are common architectural patterns for decoupling interactive AI workflows to support distinct human-in-the-loop editing tasks, such as 'edit whole document' versus 'edit selected text'?

Answer: This article presents several **AI architecture patterns** that can be combined to decouple interactive workflows and human‑in‑the‑loop tasks.[3]

1. **AI Workflows**

- Defined as linear sequences of steps with clear input/output specs, deterministic behavior, and modular design.[3]
- Editing tasks like *edit whole document* and *edit selected text* can each be modeled as separate workflows that share reusable components (context builder, safety checker, diff generator).[3]
- This separation enables straightforward testing, debugging, and evolution of each interaction mode without impacting the others.[3]

2. **AI Agents**

- Used when more flexible, autonomous reasoning is required.[3]
- An editing agent can decide when to ask for clarification, when to propose alternatives, or when to escalate to a human review step, especially for high‑impact whole‑document edits.[3]

3. **Model Context Protocol (MCP)**

- Introduced as a pattern for maintaining **shared context** and reducing ambiguity in multi‑model systems.[3]
- For editing, MCP‑like patterns ensure that different models (e.g., classifier for intent, editor LLM, safety checker) share consistent document state and user instructions across workflows.[3]

4. **Hybrid Systems**

- The author notes that real systems often mix patterns: workflows with specific steps handled by agents and coordinated through MCP‑style standards.[3]
- In practice, this means distinct **edit workflows** (whole vs. selection) can both call into shared agents (e.g., style‑enforcer agent) through a common context protocol, while keeping each end‑user flow logically separate and testable.[3]

Overall, the emphasis is on **modular, pattern‑based design** so that each type of human‑in‑the‑loop editing interaction corresponds to its own workflow or agent configuration, orchestrated around a consistent, shared document context.[3]

-----

-----

### Source [40]: https://www.catio.tech/blog/emerging-architecture-patterns-for-the-ai-native-enterprise

Query: What are common architectural patterns for decoupling interactive AI workflows to support distinct human-in-the-loop editing tasks, such as 'edit whole document' versus 'edit selected text'?

Answer: This source discusses **emerging architecture patterns for AI‑native enterprises**, emphasizing decoupling and orchestration around AI components.[4]

Key ideas relevant to editing workflows:

- Use **multi‑agent workflows** where each agent has a clearly defined job and communicates with others through an orchestrated process.[4]
- Apply the **LLM‑as‑interface** pattern, in which the LLM primarily interprets natural‑language user intent and converts it into structured actions for downstream systems.[4]

For human‑in‑the‑loop editing tasks, this suggests:

- One agent specializes in **intent interpretation** (distinguishing "edit whole document" from "edit selected paragraph" and inferring style/tone constraints).[4]
- Another agent or service is responsible for **context management** (retrieving the right document fragments and metadata).[4]
- A separate component or agent manages **tool execution** (applying changes, generating diffs, enforcing validation rules) that can integrate human approval steps.[4]

The article notes that robust systems invest heavily in **routing, memory, context retrieval, and tool execution** around the models.[4] Mapping to editing:

- **Routing**: orchestrators direct different edit intents to distinct workflows.
- **Memory**: storing conversation history and document versions for reversible edits.
- **Context retrieval**: assembling whole‑document or localized context windows based on the task.
- **Tool execution**: deterministic application of changes, separate from model inference.

These patterns allow enterprises to treat editing interactions as **AI‑orchestrated workflows**, where different user actions map to different chains or multi‑agent configurations, all wrapped in governance and observability layers.[4]

-----

</details>

<details>
<summary>How do AI-powered IDEs and writing assistants like Cursor or Claude Code manage the user interaction loop for editing selected text versus editing an entire file?</summary>

### Source [41]: https://cursor.com/features

Query: How do AI-powered IDEs and writing assistants like Cursor or Claude Code manage the user interaction loop for editing selected text versus editing an entire file?

Answer: Cursor handles the interaction loop differently depending on whether the user targets a **selection** or an **entire file / larger scope**.[4]

For **selected text editing**:
- Cursor supports **multi-line edits**, where the user selects a span of code and invokes an AI command to get **suggested edits across multiple lines**.[4]
- With **smart rewrites**, the user can "type naturally" (e.g., a natural-language instruction about the selected code) and Cursor will **finish or rewrite that region** accordingly, integrating the AI change directly into the editor.[4]
- These features are designed to operate inline: the assistant focuses primarily on the selected region, produces an edit, and shows it to the user as an updated snippet in the same place in the file.[4]

For **larger scopes / whole-file or multi-file editing**:
- Cursor is presented as an **AI‑powered IDE** that is aware of your project and repository, not just a single snippet; this enables it to apply **project-wide intelligence** to edits.[4]
- The same mechanisms that support multi-line edits can scale up to broader contexts when the user does not restrict the operation to a selection, allowing Cursor to suggest edits **across larger portions of a file** or multiple files when invoked with higher-level commands.[4]
- Because Cursor is integrated directly into the editor UI, the interaction loop typically consists of: user issues a natural-language instruction (with or without a selection), Cursor computes and surfaces one or more **diff-like suggested edits**, and the user reviews and accepts or modifies them inline.[4]

In summary, Cursor distinguishes the interaction loop by scoping: **selections trigger focused, multi-line/smart rewrite operations**, while **unscoped commands let the IDE reason over and modify larger parts of the file or project**, always returning concrete code edits for the user to apply.[4]

-----

-----

### Source [42]: https://cursor.com

Query: How do AI-powered IDEs and writing assistants like Cursor or Claude Code manage the user interaction loop for editing selected text versus editing an entire file?

Answer: Cursor is described as an **AI-powered IDE built on top of a VS Code–like interface**, embedding AI functionality directly into the primary coding workspace.[6]

Regarding the interaction loop:
- Cursor emphasizes a **pair-programming style** experience where users interact with AI via inline completions, commands, and chat-like interfaces directly tied to the open files.[6]
- When a user selects a region of text and issues an AI command, Cursor uses the selection as the primary context for the operation, generating a proposed rewrite or refactor that is shown inside the editor as a change to that region.[6]
- If the user instead works at the file or project level (for example, by invoking AI from a command palette or side panel without a selection), Cursor can leverage its project awareness to operate on **entire files or multiple files**, presenting the results as editor changes the developer can review and apply.[6]
- The system is designed so that **AI suggestions appear as concrete code edits** (like diffs) or inline completions rather than just textual explanations, which structures the loop as: instruct → receive code change → accept/modify within the file.[6]

Thus, Cursor manages the user interaction loop by tightly binding AI actions to the editor selection or active context: selected text leads to localized edits, while unscoped or broader commands allow AI to propose changes at the file or project level, always mediated through the IDE’s normal editing workflow.[6]

-----

-----

### Source [43]: https://uibakery.io/blog/claude-code-vs-cursor

Query: How do AI-powered IDEs and writing assistants like Cursor or Claude Code manage the user interaction loop for editing selected text versus editing an entire file?

Answer: This comparison explains how **Claude Code** and **Cursor** differ in where and how they manage interactions over selected text versus whole files.[1]

For **Cursor**:
- Cursor is described as an **AI-powered IDE** that acts as an inline pair programmer, with **deep codebase awareness and inline completions**.[1]
- It supports features like **auto‑refactor, test generation, and context search**, which operate at the file or project level, not just on snippets.[1]
- Cursor’s context is "moderate, optimized for active files & project scope," meaning the assistant works primarily with the open files and relevant project context, and can apply edits that span beyond a single selection.[1]

For **Claude Code**:
- Claude Code is positioned as an **external, dedicated AI coding assistant** that focuses on **code understanding, bug fixes, and project-level reasoning**.[1]
- It can **analyze large files and entire repositories**, and is especially strong at explaining and refactoring code or generating documentation across broader scopes.[1]
- The article notes that Claude Code can operate **inside Cursor via integration** or independently; when used through Cursor, Claude’s reasoning powers combine with Cursor’s editor-based interaction loop (selections, diffs, etc.).[1]

In terms of interaction loops:
- Cursor primarily manages **in-editor selection vs. whole-file editing**, applying AI directly as code changes.[1]
- Claude Code usually returns **textual suggestions or refactors** that the user copies into their files, unless used through an integration that can apply edits automatically.[1]

So, Cursor’s loop is editor‑centric (selection/file/project → diffed edits), while Claude Code’s loop is assistant‑centric (user provides snippet or larger context → receives suggested changes or explanations), which can then be integrated into files manually or via tools.[1]

-----

-----

### Source [44]: https://graphite.com/guides/programming-with-ai-workflows-claude-copilot-cursor

Query: How do AI-powered IDEs and writing assistants like Cursor or Claude Code manage the user interaction loop for editing selected text versus editing an entire file?

Answer: This guide describes how **Claude** and **Cursor** fit into coding workflows, clarifying how they handle interactions on selected code versus entire files or codebases.[2]

For **Claude (including Claude Code workflows)**:
- Claude is treated as an **AI pair programmer you converse with**: the user typically **pastes a function, error log, or multiple files** into a chat and asks for help.[2]
- Because Claude has a **large context window**, it can ingest **multiple files or large diffs at once**, reasoning over whole modules or codebases, not just small selections.[2]
- However, Claude **does not automatically apply changes** to files; instead, it returns **proposed fixes, refactors, or explanations**, and the user copies those back into the editor or uses a CLI/plugin to write to files.[2]
- This means the interaction loop for selected snippets versus entire files is governed by what the user pastes or attaches: a small snippet yields focused suggestions, while a full file or multi-file context yields broader, project-scale recommendations.[2]

For **Cursor**:
- Cursor is an **AI-first code editor** that "analyzes your entire codebase for deep project insight," letting you ask questions like "Where is the user authentication logic defined?" and supporting **cross-file refactoring and consistent code generation**.[2]
- It supports a **command mode** where you can select a piece of code and issue a natural-language command such as "convert this loop to a list comprehension"; Cursor then **rewrites that specific selection** in place.[2]
- Cursor also performs **multi-file edits and project-wide changes** when instructed without limiting the scope to a selection, aided by its repository indexing.[2]
- The interaction loop is: select code (or not) → issue a command or chat request → Cursor computes changes constrained to the selection or expanded to relevant files → user reviews and accepts the in-editor edits.[2]

Thus, Claude manages scope through the *amount of context supplied in the conversation*, while Cursor manages scope through *editor selections vs. unscoped/project-wide commands*, always surfacing concrete edits in the IDE.[2]

-----

</details>

<details>
<summary>What are the most effective UX/UI design principles for interfaces that allow users to provide feedback on specific sections of AI-generated content for iterative refinement?</summary>

### Source [45]: https://www.aufaitux.com/blog/ai-interface-usability-principles/

Query: What are the most effective UX/UI design principles for interfaces that allow users to provide feedback on specific sections of AI-generated content for iterative refinement?

Answer: This source outlines **AI interface usability principles** that apply directly to feedback-on-output UIs.

It emphasizes **clarity and transparency**: clearly explain the AI’s role, what parts of the content are AI-generated, and why a given output was produced (e.g., contextual notes such as “based on your past actions”).[1] For section-level feedback UIs, this supports inline explanations or tooltips tied to specific paragraphs, so users know *what* they are critiquing and *why* the system behaved that way.

The principle of **user control and feedback** states that users should always be able to intervene in AI actions.[1] Best practices include:
- Offering **undo / revert** for AI-applied refinements.
- Letting users **modify AI-generated results** directly (e.g., editing a section) instead of only rating them.[1]
- Providing clear options to **report issues** with AI decisions, which can be applied as “improve this section,” “this is wrong,” or “inappropriate” controls at section level.[1]

**Predictability and consistency** are crucial so AI behavior is not surprising: terminology, interaction patterns, and the effect of each feedback control (thumbs down, flag, re‑generate) should behave consistently across sections.[1]

The principle of **minimal cognitive load** recommends simple, intuitive interactions and limiting information to what is relevant.[1] For section feedback, this suggests lightweight inline controls (e.g., compact icon set near each block) and short, guided micro-forms instead of complex dialogs.

Under **seamless human–AI collaboration**, AI should complement, not replace, the user: repetitive edits can be automated, but key content decisions remain with the user, who can accept, reject, or further edit refinements for each section.[1]

The article also highlights **user feedback and reinforcement learning**: AI systems should include mechanisms for users to correct or validate responses and use that feedback in learning loops.[1] For iterative refinement interfaces, this supports explicit rating / correction at the section level that feeds back into model improvement.

Finally, it stresses **handling bias and ethics**, recommending clear feedback flows to flag biased or harmful outputs, and **testing methods** (think‑aloud, eye‑tracking) to validate whether users notice and understand feedback affordances around AI content.[1]

-----

-----

### Source [46]: https://www.uxstudioteam.com/ux-blog/ai-ux-5f836

Query: What are the most effective UX/UI design principles for interfaces that allow users to provide feedback on specific sections of AI-generated content for iterative refinement?

Answer: This source presents **AI UX principles** that are highly relevant to designing feedback workflows around AI-generated content.

It stresses the need to **identify a real user need** before adding AI: AI should either automate repetitive tasks or augment user capabilities.[2] For section feedback, this means ensuring that inline refinement genuinely speeds up editing or quality control, rather than being an ornamental feature.

The principle **“set clear expectations for what the AI can and cannot do”** is central.[2] Interfaces should explicitly communicate what feedback will change (e.g., “your comment will be used to refine this paragraph, not the whole document”) and the limits of refinement quality.

On data, the article recommends **collecting the right data transparently and responsibly**.[2] It advises being transparent about what data is collected, how feedback is used, and obtaining permission especially for personal or sensitive data.[2] For per-section feedback UIs, this suggests clear privacy copy near feedback components and simple controls for opting out of model training.

The article highlights **giving the user control** as a key AI UX principle.[2] Users should have control over the output—e.g., they can trigger re-generation, apply or discard edits, and control how much the AI changes a given section.[2]

It also stresses **using clear and simple language** in all user-facing communication about AI behavior, expectations, and error handling.[2] Feedback affordances should avoid ML jargon and use plain labels like “Improve tone,” “Fix facts,” or “Regenerate this section.”

Finally, **collecting user feedback** is called out as essential for improving AI systems.[2] The article notes that user feedback can be gathered explicitly (ratings, comments) or implicitly (interaction patterns).[2] For iterative refinement UIs, this supports designs that combine explicit per-section ratings with implicit signals such as how often users manually rewrite AI suggestions, helping tune the system over time.

The article concludes that great AI UX requires: clear expectations, transparent data practices, user control, simple language, and integrated feedback, all directly applicable to section-level refinement interfaces.[2]

-----

-----

### Source [47]: https://pair.withgoogle.com/guidebook/chapters/feedback-and-controls/design-ai-feedback-loops

Query: What are the most effective UX/UI design principles for interfaces that allow users to provide feedback on specific sections of AI-generated content for iterative refinement?

Answer: This People + AI Research (PAIR) guide focuses on **designing AI feedback loops and controls**, offering concrete principles for interfaces that collect user feedback on AI outputs.

It recommends that UX teams design **specific, action‑oriented buttons instead of generic icons** for feedback.[4] For section-level refinement, this means replacing ambiguous thumbs icons with labeled controls like “Not relevant,” “Off‑topic,” or “Too long,” attached to each content block.[4]

The guide notes that AI teams should build models that can **weight signals by their ambiguity**.[4] Explicit, well-labeled feedback (e.g., choosing a reason) should carry more weight than vague signals. This underpins UI patterns where users select structured reasons when they down‑vote or flag a section, making the signals more actionable.[4]

It emphasizes that teams should **decide what kinds of feedback are useful** and design the UI around these categories, rather than collecting generic feedback that is hard to interpret.[4] For iterative refinement, this leads to carefully chosen feedback dimensions (factuality, tone, usefulness, safety) surfaced as quick chips or options on each section.

The guide advises making feedback **easy and low‑friction** while still being interpretable: very short flows, minimal text entry, and pre‑defined options.[4] In the context of AI-generated content, this suggests inline controls adjacent to each section, visible on hover or focus, allowing quick selection without navigating away.

It highlights the importance of **communicating how feedback is used**, to set expectations and build trust.[4] The UI should indicate whether feedback will adjust future outputs for that user, affect global model behavior, or only help improve the product over time.[4]

The document also underscores the need to **balance user control and automation**: users should be able to correct AI errors, override decisions, and understand what controls do, rather than only passively consuming outputs.[4] For section-level refinement, controls like “Apply suggested fix,” “Show alternative,” or “Restore original” exemplify this balance.[4]

-----

-----

### Source [48]: https://becominghuman.ai/ux-design-for-implicit-and-explicit-feedback-in-an-ai-product-9497dce737ea

Query: What are the most effective UX/UI design principles for interfaces that allow users to provide feedback on specific sections of AI-generated content for iterative refinement?

Answer: This source distinguishes **explicit vs. implicit feedback** in AI products and discusses UX patterns for both.

It defines **explicit feedback** as actions performed specifically for giving feedback to the system, such as clicking rating controls or buttons indicating preference.[3] For AI-generated content, explicit controls can be attached to individual sections—e.g., “Not interested,” “Show fewer like this,” or “Improve this explanation.”[3]

The article uses **YouTube recommendations** as an example: users can give explicit feedback to refine the recommendation algorithm and gain a sense of control over what they see.[3] Translated to text-generation UIs, similar explicit mechanisms let users steer the content they get, building a feeling of control over AI output.

It contrasts this with **implicit feedback**, where user interactions not primarily intended as feedback (watch time, skips) are still interpreted by the system.[3] For content refinement, implicit signals might include whether the user accepts suggested edits, how long they spend editing particular sections, or whether they frequently regenerate certain blocks.

The article argues that combining both feedback types produces more robust learning loops: explicit feedback is **clear but sparse**, while implicit signals are **abundant but noisy**.[3] UX design should thus:
- Make explicit section-level feedback simple and understandable.
- Ensure that implicit data collection respects privacy and is used responsibly.

It also notes that good feedback UX should give users a **sense of control** over the AI-driven experience.[3] Interfaces that surface understandable feedback options and show visible consequences (e.g., recommendations changing) help users feel that their input matters.

While focused on recommendations, the concepts are directly applicable to AI-generated text: creating deliberate, labeled feedback affordances for each content unit, complemented by system-side use of behavioral signals, to iteratively refine accuracy and relevance of generated sections.[3]

-----

-----

### Source [49]: https://uxdesign.cc/ux-design-principles-for-ai-products-8989aa55819d

Query: What are the most effective UX/UI design principles for interfaces that allow users to provide feedback on specific sections of AI-generated content for iterative refinement?

Answer: This source lays out **UX design principles for AI products** that are applicable to feedback-on-content interfaces.

It stresses that UIs should follow **accessibility best practices**: legible text, sufficient color contrast, and clear focus states.[7] For section‑level feedback, this means that inline controls (icons, chips, labels) must be accessible via keyboard and screen readers so all users can provide feedback on AI outputs.[7]

The article emphasizes the importance of **explainability and transparency**: interfaces should, where possible, help users understand why the AI generated a particular result.[7] In refinement contexts, this supports UI elements that show short “why this content” hints or highlight source inputs that influenced a given section, enabling more targeted feedback.

It also discusses **calibrating user trust** by neither overstating nor hiding AI involvement.[7] Clearly marking AI-generated sections and indicating that users can correct them helps create an appropriate mental model and encourages critical review and feedback.

The principle of **human-in-the-loop** design is highlighted: AI should assist, but humans remain responsible for final decisions.[7] For iterative refinement workflows, this suggests patterns where the user reviews each section, can accept or reject AI-proposed edits, and can leave comments or instructions to guide further generations.

The article notes that designers should consider **error states and recovery** when AI output is poor.[7] For feedback UIs, this means providing visible, understandable ways to flag bad sections, request alternatives, or revert to earlier versions.

Finally, it underlines using **clear, non-technical language** to describe AI behavior and controls, and adopting consistent interaction patterns.[7] Applied to section feedback: labels like “Improve clarity” or “This is incorrect” should be preferred over technical terminology; controls for rating, flagging, and regenerating should behave consistently across the interface so users can quickly learn how to refine specific parts of the AI-generated content.[7]

-----

-----

### Source [50]: https://tentackles.com/blog/ai-first-ux-ui-principles-zero-click

Query: What are the most effective UX/UI design principles for interfaces that allow users to provide feedback on specific sections of AI-generated content for iterative refinement?

Answer: This source discusses **AI‑first UI/UX design principles** aimed at building trust, transparency, and confidence, which extend to feedback on AI output.

It focuses on **zero‑click experiences**, where the system anticipates user needs, but the principles still stress maintaining **user agency**.[8] For AI-generated content, this implies that even if the system proposes refinements automatically, users must retain straightforward ways to approve, edit, or reject section-level suggestions.

The article highlights **trust and transparency** as core principles.[8] Designers should be open about when and how AI is acting, avoid dark patterns, and help users understand AI decisions without overwhelming them.[8] For section feedback, this supports clearly indicating that controls like “Improve this section” will trigger AI processing, and clarifying whether changes are reversible.

It also underlines **progressive disclosure**: advanced controls and explanations should be available but not forced on all users.[8] In iterative refinement UIs, basic one‑tap feedback (like/dislike, re‑generate) can be shown by default, with more detailed feedback options or explanation views accessible on demand for power users.

The principle of **continuous learning** is mentioned: AI systems should evolve from user interactions.[8] A good interface therefore needs to collect meaningful feedback while making this process feel safe and beneficial to users.

Additionally, the article mentions maintaining **consistency and predictability** in AI interactions to avoid confusing users.[8] In feedback-on-content interfaces, controls for rating or revising specific sections should look and behave the same wherever they appear, and the results of taking an action (e.g., re-generating a paragraph) should be stable and understandable.

Overall, the piece frames AI-first design as balancing intelligent proactivity with clear user controls, transparent operations, and interfaces that make it easy for users to refine and correct AI behavior—principles directly applicable to per-section feedback for iterative refinement.[8]

-----

</details>

<details>
<summary>How can a modular system using concepts like the 'evaluator-optimizer pattern' be exposed as a collection of distinct tools through a serving layer like an MCP server or API?</summary>

### Source [51]: https://www.anthropic.com/research/building-effective-agents

Query: How can a modular system using concepts like the 'evaluator-optimizer pattern' be exposed as a collection of distinct tools through a serving layer like an MCP server or API?

Answer: According to Anthropic, the **evaluator–optimizer workflow** is a core agentic pattern where one LLM call generates a response and another evaluates it, providing feedback in a loop until quality criteria are met.[6]

Anthropic explains that this pattern is best used when:
- There are **clear evaluation criteria** for success.
- Iterative refinement **measurably improves** responses.
- The system can articulate feedback that the model can use to improve its output.[6]

To expose such a modular evaluator–optimizer system as **distinct tools via a serving layer** (such as an MCP server or API), the Anthropic description implies a separation of concerns that maps naturally to tools:
- A **generation tool** that performs the initial task execution (e.g., draft answer, code, plan) using the *optimizer* LLM call.
- An **evaluation tool** that scores, critiques, or classifies the generator’s output against explicit criteria.
- A **refinement tool** that re-invokes the generator with the evaluator’s feedback, preserving the original task while incorporating corrections.
- An **orchestration / controller** that implements the loop and stopping conditions (evaluator passes, or max iterations), which can itself be a higher-level tool or workflow API.[6]

Anthropic’s description emphasizes that the workflow is architectural, not tied to a specific runtime, so in a serving layer you can:
- Define each role (generator, evaluator) as **separate callable endpoints or tools** with well-specified inputs/outputs.
- Let a coordinating agent (or client) call these tools in sequence, maintaining state across iterations via request payloads.

This modular decomposition allows an MCP server or API to surface:
- A **"generate" tool** for first-pass solutions.
- An **"evaluate" tool** for quality checks.
- A **"refine" tool** that couples generation with feedback.
- Optionally, a **single "run_evaluator_optimizer" tool** that hides the loop and exposes the whole pattern as one higher-level capability, built internally on the generator/evaluator primitives.[6]

-----

-----

### Source [52]: https://icepick.hatchet.run/patterns/evaluator-optimizer

Query: How can a modular system using concepts like the 'evaluator-optimizer pattern' be exposed as a collection of distinct tools through a serving layer like an MCP server or API?

Answer: The Icepick docs describe **evaluator–optimizer** as an **iterative generation–evaluation cycle** where one component generates content and another evaluates it, continuing until the evaluator is satisfied or a maximum number of iterations is reached.[2]

Icepick’s architecture model is explicitly modular:
- **Client**
- **Agent**
- **Generator**
- **Evaluator**
- **Iteration control**[2]

This structure maps cleanly to exposing the pattern as distinct tools via a serving layer:
- The **Generator Tool** is responsible for both the initial generation and subsequent improvements. It accepts the task, prior drafts, and feedback, and produces a new candidate while maintaining the core message and constraints.[2]
- The **Evaluator Tool** analyzes a candidate against requirements or guidelines and returns structured feedback and/or satisfaction signals (e.g., pass/fail, scores).[2]
- **Iteration Control** can be implemented as a higher-level tool or workflow endpoint that:
  - Calls the Generator tool to create an initial draft.
  - Calls the Evaluator tool with that draft.
  - Uses the evaluator’s output plus iteration counters to decide whether to stop or call the Generator again with feedback.[2]

In a serving layer (MCP server or generic API), this suggests:
- Exposing **`generate_candidate`** and **`evaluate_candidate`** as separate tools with clear schemas.
- Optionally exposing **`run_evaluator_optimizer`** as a composite tool that encapsulates the loop, while internally orchestrating the generator/evaluator calls and managing iterative state.[2]

Because Icepick treats these as independent components wired by iteration control, the same generator and evaluator tools can be reused in other workflows, while the serving layer simply offers them as distinct, composable capabilities.[2]

-----

-----

### Source [53]: https://huggingface.co/blog/dcarpintero/design-patterns-for-building-agentic-workflows

Query: How can a modular system using concepts like the 'evaluator-optimizer pattern' be exposed as a collection of distinct tools through a serving layer like an MCP server or API?

Answer: The Hugging Face article on **design patterns for agentic workflows** presents the **Evaluator–Optimizer Pattern** as an architecture with two main roles: an **Evaluator** and a **Generator/Optimizer**.[4]

The pattern’s workflow is:
- **Generation**: An LLM generates an initial response or completes the task using standard capabilities; this output may contain errors or omissions.[4]
- **Reflection – Evaluator**: The same or another LLM evaluates the output against requirements, guidelines, or observations, providing a structured critique.[4]
- **Refinement – Optimizer**: The system turns that feedback into concrete improvements—restructuring content, filling gaps, or correcting issues.[4]
- **Iteration**: This cycle repeats until predetermined quality criteria or stopping conditions are met.[4]

To expose this modular system through a serving layer (MCP server / API) as **distinct tools**, Hugging Face’s separation naturally yields:
- A **`generate_response` tool** that performs the *Generation* step and returns both the content and any metadata needed for later iterations.
- An **`evaluate_response` tool** that performs the *Reflection* step and returns machine-usable feedback (e.g., error categories, missing elements, quality scores).[4]
- An **`optimize_response` tool** that consumes the prior response plus feedback and produces an improved version, implementing the *Refinement* step.[4]

The article notes that these steps can be carried out by a single LLM or multiple specialized LLMs, which in a serving context translates to:
- Either separate tools that call different underlying models (e.g., a stronger evaluator model, a cheaper generator), or
- Unified tools that parameterize which model to use.[4]

A higher-level **orchestrator** (agent or workflow endpoint) can be offered as another tool that:
- Encodes stopping conditions and iteration limits.
- Calls the lower-level tools in sequence and aggregates their outputs into a final result exposed by the API.[4]

-----

-----

### Source [54]: https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns

Query: How can a modular system using concepts like the 'evaluator-optimizer pattern' be exposed as a collection of distinct tools through a serving layer like an MCP server or API?

Answer: The Spring AI blog on **Building Effective Agents** describes an **Evaluator–Optimizer pattern** as a dual-LLM process where one model generates responses and another evaluates and provides feedback in an iterative loop.[3]

Spring AI shows this pattern as a concrete workflow class, with steps:
- Generate an initial solution.
- Evaluate that solution.
- If the evaluation passes, return the solution.
- If it needs improvement, incorporate feedback and generate a new solution.
- Repeat until satisfactory, then return the final solution and optionally a chain-of-thought.[3]

From a serving-layer and tooling perspective, Spring AI’s implementation implies clear modular boundaries that can be mapped to tools:
- A **Generation component** (method like `generate(task, context)`) that could be surfaced as a **`generate_solution` tool**, taking the task and any context and returning a candidate response.[3]
- An **Evaluation component** (method like `evaluate(response, task)`) that would be a **`evaluate_solution` tool**, returning a structured evaluation (e.g., PASS / NEEDS_IMPROVEMENT plus feedback message).[3]
- A **Workflow / loop controller** (e.g., `EvaluatorOptimizerWorkflow.loop(task)`) that becomes a higher-level **`run_evaluator_optimizer` tool** encapsulating the iteration logic while internally invoking the generation and evaluation tools until stopping criteria are met.[3]

Spring AI’s use of `ChatClient` to call LLMs for each step suggests an API or MCP server can:
- Implement each step as a distinct endpoint or MCP tool bound to a `ChatClient` call with different prompts.
- Let clients either call the full workflow tool (for a single-shot "refined" answer) or orchestrate calls to the generation and evaluation tools themselves if they need custom control over stopping conditions or integration with other tools.[3]

-----

-----

### Source [55]: https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/evaluator-reflect-refine-loop-patterns.html

Query: How can a modular system using concepts like the 'evaluator-optimizer pattern' be exposed as a collection of distinct tools through a serving layer like an MCP server or API?

Answer: AWS Prescriptive Guidance describes **evaluator reflect–refine loop patterns**, where tasks like code generation or summarization benefit from runtime feedback in an event-driven feedback control loop.[7]

In this pattern, an **agentic evaluator loop** uses LLMs for dynamic output assessment and refinement, creating self-improving systems through feedback.[7] The loop is inspired by control theory: an initial action produces an output, sensors or evaluators observe the result, and a controller adjusts the next action based on deviations from the target.

For exposure as distinct tools through a serving layer (MCP or API), AWS’s framing suggests these modular components:
- A **Task Performer** (generator) that produces an initial artifact or decision (e.g., code, summary, plan).
- An **Evaluator** that inspects the artifact against quality criteria or constraints and emits feedback signals or error classifications.[7]
- A **Refiner / Controller** that takes evaluator feedback and updates the next action—regenerating, editing, or augmenting the artifact—and manages the loop until the system meets target thresholds or time/iteration limits.[7]

Operationalizing this in an API involves:
- Defining separate endpoints/tools for **`perform_task`**, **`evaluate_output`**, and **`refine_output`**, each with strict input/output contracts shaped by the evaluation signals.[7]
- Implementing the reflect–refine cycle as a higher-level workflow that can be exposed as another tool, while internally invoking the lower-level tools and using event-driven triggers (e.g., failures, low scores) to decide when to re-enter the loop.[7]

This modular, control-loop framing aligns naturally with a serving-layer design where each role is a distinct tool, but the overall evaluator loop is a composable building block for higher-level agent behaviors.[7]

-----

</details>

<details>
<summary>What are the system design best practices for exposing multiple, distinct generative AI workflows (e.g., create, full-document edit, selective-text edit) as a cohesive set of tools via a single API or server for interactive applications?</summary>

### Source [56]: https://docs.asapp.com/generativeagent/configuring/connect-apis/designing-apis-for-generativeagent

Query: What are the system design best practices for exposing multiple, distinct generative AI workflows (e.g., create, full-document edit, selective-text edit) as a cohesive set of tools via a single API or server for interactive applications?

Answer: This source focuses on **API design principles for AI agents** and is directly applicable when exposing multiple generative workflows (create, full-document edit, selective edit) through one API.

Key best practices:

- **Design APIs for machine-readability and explicit semantics** rather than prose documentation; AI agents rely on structured, self-describing contracts (e.g., OpenAPI + JSON Schema) to understand tools and parameters, not human-oriented docs.[1]
- **Use consistent schema patterns** across all tools so the agent can reliably compose different workflows. For example, share common request/response shapes and naming across create/edit/selective-edit operations.[1]
- **Simplify field names** and avoid unclear abbreviations so the model can distinguish similar tools and choose the correct workflow. Names should clearly encode intent, such as `create_document`, `edit_document_full`, `edit_document_selection`.[1]
- **Use intuitive, human-friendly resource concepts** instead of esoteric terms like “record” or “details”; expose workflows around clear entities such as `document`, `section`, or `selection`, each with explicit state and requirements.[1]
- **Define strict schemas** (types, formats, constraints) for every request and response using JSON Schema inside OpenAPI. This lets the agent know exactly which fields (e.g., `document_text`, `selection_start`, `selection_end`, `edit_instructions`) are required for each workflow and how to parse outputs.[1]
- **Maintain consistent naming conventions** for endpoints and operationIds, such as plural nouns for collections (`/documents`) and descriptive operationIds (`createDocument`, `editDocumentSelection`), aiding the LLM’s tool selection.[1]
- **Provide rich, self-contained context in responses** (e.g., return edited text plus metadata like version, applied instructions) so subsequent tool calls in an interactive session do not depend on external state the model cannot see.[1]
- **Ensure idempotency** for operations that might be retried by the agent (for example, repeated full-document edits with the same parameters should have predictable effects), which is important in tool-execution loops.[1]
- **Design structured, machine-parseable errors** with clear codes and messages so the agent can recover (e.g., distinguish validation failures for missing selection boundaries vs. model-side generation errors).[1]
- **Flatten complex structures** in requests and responses and remove non-essential fields; LLMs struggle with deeply nested or noisy payloads, so each workflow’s input/output should be minimal and flat while still explicit.[1]
- **Start with simple, specific tools and expand**: instead of only generic CRUD, expose higher-level, task-specific operations (like `summarize_document`, `rewrite_paragraph`) that map cleanly to generative workflows and are easier for agents to invoke correctly.[1]

-----

-----

### Source [57]: https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/genai-developer-workflow

Query: What are the system design best practices for exposing multiple, distinct generative AI workflows (e.g., create, full-document edit, selective-text edit) as a cohesive set of tools via a single API or server for interactive applications?

Answer: This source describes a **development workflow for generative AI apps**, including tool design and agent integration, relevant to unifying multiple workflows behind one API.

Key practices:

- **Start from typical user requests and expected responses**: define what create vs. edit vs. selective-edit requests look like, what formats users expect, and how the application will present responses (short vs. long, structured vs. free text). This shapes the API’s tool surface.[2]
- **Clarify interaction modality** (chat, search, UI) and **tone, style, and structure** requirements early; encode these as parameters or system-level configuration in the single API so different workflows can reuse them consistently.[2]
- **Define error-handling behavior** at the application and agent level: decide how the system reacts to ambiguous or incomplete requests (e.g., ask for clarification before choosing between full-document and selective edit), and expose this behavior coherently via the server.[2]
- **Consider latency and streaming**: specify response time and streaming requirements; for interactive editing tools, streaming partial generations may be important, and the API should expose a consistent streaming interface across all workflows.[2]
- For multi-tool systems, **keep the initial toolset minimal**: start with only the essential workflows (for example, `draft`, `edit`, `summarize`) and validate each tool independently before integrating them into a single agent-facing API.[2]
- **Validate each tool in isolation** (such as via notebooks) to ensure schemas, behaviors, and edge cases are correct before orchestrating them, which reduces complexity when multiple workflows are exposed together.[2]
- When building the agent, **incorporate tools via explicit schemas** so the LLM knows how to call each function; in a chain design, calls are hard-coded, while in a tool-calling agent the schema drives automatic selection among workflows.[2]
- **Design custom outputs and metadata** (for example, including document IDs, source references, confidence scores) that the client needs for all workflows, and have the agent include this metadata consistently in its responses through the single API surface.[2]
- Plan how the **client application interacts with the agent** (over an API or embedded) and how responses from different workflows are displayed, ensuring a unified interaction model even if multiple underlying tools are invoked.[2]

-----

-----

### Source [58]: https://aws.amazon.com/blogs/machine-learning/best-practices-to-build-generative-ai-applications-on-aws/

Query: What are the system design best practices for exposing multiple, distinct generative AI workflows (e.g., create, full-document edit, selective-text edit) as a cohesive set of tools via a single API or server for interactive applications?

Answer: This source outlines **best practices for building generative AI applications** on AWS, relevant to designing cohesive multi-workflow APIs.

Key practices related to system design:

- **Use appropriate generative AI approaches per workflow**, such as prompt engineering for simple create/edit tasks, Retrieval Augmented Generation (RAG) for document-grounded editing, and model customization when domain adaptation is needed.[3]
- For workflows like full-document or selective-text editing, **RAG can reduce hallucinations and improve factuality** by retrieving relevant document context and grounding edits in external knowledge sources.[3]
- RAG improves **coverage**, letting the model handle a wider range of topics by pulling in external information, which is useful for a single API that must support varied document types and domains.[3]
- RAG also brings **efficiency**, focusing generation on relevant context instead of regenerating entire documents from scratch; this is particularly important when exposing full-document workflows that may have large inputs.[3]
- From a governance perspective, **retrieval from required and permitted data sources** improves safety and control, aligning with enterprise needs when a central API serves many workflows and teams.[3]
- **Model customization and fine-tuning** can align the system with domain-specific style, terminology, or safety constraints across all workflows (create, full-edit, selective-edit) exposed by the same API.[3]
- Customization supports **personalization**, such as adapting the model to a user’s historical documents to ensure consistent tone and style in both new content creation and edits.[3]
- Fine-tuning can help **adapt to new domains and tasks**, which is useful when a single server must support heterogeneous workflows (e.g., legal drafting vs. marketing copy) while maintaining a coherent interface.[3]
- Customization can also help **mitigate bias** present in base models, contributing to uniform safety and fairness properties across every workflow accessed through the unified API.[3]

-----

-----

### Source [59]: https://nordicapis.com/how-to-design-robust-generative-ai-apis/

Query: What are the system design best practices for exposing multiple, distinct generative AI workflows (e.g., create, full-document edit, selective-text edit) as a cohesive set of tools via a single API or server for interactive applications?

Answer: This source offers **design guidance specific to generative AI APIs**, applicable when unifying multiple workflows through a single endpoint surface.

Relevant best practices:

- **Treat generative operations as first-class API resources** with clear contracts, rather than opaque text-in/text-out endpoints. Define distinct operations (e.g., `generate`, `transform`, `summarize`) that can map onto create/edit/selective-edit workflows while sharing a common pattern.[5]
- **Expose configuration and control parameters explicitly** (temperature, max tokens, style flags), but with sensible defaults so clients can use the unified API easily while still tuning behavior per workflow when needed.[5]
- **Design for non-determinism and variability**: generative APIs must acknowledge that repeated calls with the same input can yield different outputs. Downstream systems and clients should not rely on strict idempotence of text output, and the API should provide ways to fix randomness (e.g., seeds) where reproducibility is required.[5]
- **Support streaming responses** for interactive applications, so users see partial generations early. The streaming mechanism should be consistent across all workflows exposed by the same server.[5]
- **Implement robust rate limiting, quotas, and usage tracking** tuned for generative workloads, which are typically more compute-intensive than standard CRUD, and ensure these policies are coherent for all tools in the API.[5]
- **Provide clear error semantics specific to generative tasks**, such as input-too-long, safety filter rejection, or model-timeout, allowing clients or agents to fallback to alternate workflows (e.g., summarization before full-edit) when needed.[5]
- Encourage **sensible defaults and opinionated presets** (for example, predefined profiles for “creative drafting” vs. “conservative editing”) that clients can select via parameters. These presets can span multiple generative workflows but be expressed uniformly in the API.[5]
- Emphasize **observability and monitoring**—log prompts, responses, and parameters (with appropriate privacy controls) to understand behavior across workflows and iterate on the unified API design.[5]

-----

</details>

<details>
<summary>How can AI-powered writing assistants create a "coding-like" interactive experience, specifically regarding features like diffing, version control, and iterative refinement based on user instructions?</summary>

### Source [60]: https://gemini.google/overview/canvas/

Query: How can AI-powered writing assistants create a "coding-like" interactive experience, specifically regarding features like diffing, version control, and iterative refinement based on user instructions?

Answer: Gemini **Canvas** provides a coding‑like interactive experience for writing by letting users work in a persistent document where AI and user edits interleave over time.[7] The user can draft content, then iteratively refine it with natural‑language instructions such as changing tone, shortening, expanding, or restructuring sections.[7] Canvas supports **inline edits**: users select text and ask the model to rewrite or transform only that span, which mimics editing specific code blocks while keeping the rest of the file unchanged.[7]

Canvas is designed for **iterative refinement**: users can try alternative phrasings, compare results inside the same workspace, and continue modifying the chosen version.[7] Because all edits happen in a single canvas, the system effectively maintains a running "history" of model‑generated and user‑authored changes, similar to an evolving code file under version control.[7] Users can also move between writing and coding in the same space, using the model to generate or modify code snippets, which reinforces the coding‑like workflow where instructions and code/results coexist.[7]

The environment is multimodal, enabling users to bring in images or other content as context, then ask Gemini to adapt the document accordingly (e.g., summarizing, rewriting, or integrating referenced material).[7] This mirrors how developers provide external artifacts (requirements, diagrams) to guide code changes.[7] Overall, Canvas emphasizes a conversational, incremental editing loop on a live artifact—closely analogous to an IDE with refactoring tools and partial diffs—even though traditional Git‑style diff views and commits are not explicitly surfaced.[7]

-----

-----

### Source [61]: https://www.canva.com/ai-assistant/

Query: How can AI-powered writing assistants create a "coding-like" interactive experience, specifically regarding features like diffing, version control, and iterative refinement based on user instructions?

Answer: Canva’s **AI assistant** for documents (Magic Write and related tools) enables an interactive, coding‑like loop where content is generated, then refined repeatedly through natural‑language instructions.[2] Users can go **“from idea to doc in seconds”** by prompting Draft a doc to create an initial version from a short description or existing materials.[2] This is analogous to scaffolding a code file from a high‑level specification.

After generation, Canva AI supports **refine, rewrite, and rework** operations so users can iteratively improve tone, structure, or level of detail until the result meets their needs.[2] The assistant encourages a back‑and‑forth workflow: users provide feedback, try suggested prompts, and adjust the content, much as a developer iterates on code changes based on compiler or test feedback.[2]

The system is aware of context, allowing users to upload images or documents and have the assistant **extract insights, repurpose content, or generate new material from what they already have**.[2] This resembles a code assistant operating over an existing codebase or documentation to propose targeted modifications instead of always starting from scratch.

The UI is conversational: users can **chat naturally** with the assistant, issue commands to tweak or rewrite parts of the content, and see the resulting changes directly in the editor.[2] While Canva’s description does not expose explicit Git‑style version control or textual diff views, the workflow of quickly regenerating sections, trying alternatives, and adjusting based on user instructions closely mirrors iterative refactoring and patching in software development.[2]

-----

-----

### Source [62]: https://www.jetbrains.com/ai-assistant/

Query: How can AI-powered writing assistants create a "coding-like" interactive experience, specifically regarding features like diffing, version control, and iterative refinement based on user instructions?

Answer: JetBrains **AI Assistant** integrates AI directly into JetBrains IDEs, using patterns familiar from coding to support interactive, instruction‑driven refinement of code and other project text.[5] Users can invoke AI‑powered **code generation and transformation** within the editor, including generating new code or modifying existing selections based on natural‑language prompts.[5] This selective transformation behaves similarly to applying a localized patch or refactor to a specific region of a file.

The assistant provides **context‑aware code generation**, leveraging the surrounding project structure, which parallels how a future writing assistant could consider entire documents or knowledge bases when rewriting sections.[5] It supports an **AI chat** tightly connected to the open files and project, letting users ask for explanations, changes, or improvements and then apply those suggestions back into the code with a click.[5] This resembles a conversational diff: the model proposes edits, and the user chooses when and where to integrate them.

JetBrains tools already include robust **version control integration**, and AI‑generated changes are applied through the normal project files, so developers can review them using built‑in diff and history views before committing.[5] Although the documentation focuses on coding, the pattern shows how AI assistants can create a coding‑like workflow for any text artifact: localized, prompt‑driven edits; context‑aware suggestions; and human‑controlled integration of changes via existing VCS and diff tools.[5]

-----

-----

### Source [63]: https://codeassist.google

Query: How can AI-powered writing assistants create a "coding-like" interactive experience, specifically regarding features like diffing, version control, and iterative refinement based on user instructions?

Answer: **Gemini Code Assist** integrates Gemini into IDEs and GitHub to create a coding‑native, iterative experience built around versioning and review.[4] In editors like VS Code, JetBrains IDEs, Android Studio, and others, it offers **chat aware of your code**, automatic **code completion**, and on‑demand generation or transformation of full functions or files.[4] Users can instruct the assistant in natural language to refactor, document, or extend code; the assistant then modifies or proposes changes bounded to the relevant files, mirroring targeted patches.[4]

Gemini Code Assist also supports a **Gemini CLI**, bringing model assistance to the terminal.[4] Terminal workflows naturally integrate with tools like Git, meaning AI‑driven modifications can be inspected via standard **diffs** and committed into version control, preserving a full history of iterations.[4]

For GitHub pull requests, Gemini Code Assist can **automatically review PRs to find bugs and style issues and suggest code changes**.[4] This situates AI proposals directly inside the review and version‑control loop, where each suggestion appears as a concrete change against a known base revision.[4] Users can accept or modify these suggestions, preserving the familiar commit‑based workflow while benefiting from AI‑generated patches.

This design demonstrates how an AI assistant can create a coding‑like interactive experience: state lives in the repository; AI suggestions manifest as candidate diffs or file edits; and humans retain control by reviewing, integrating, or discarding changes within Git‑backed workflows and IDE diff views.[4]

-----

-----

### Source [64]: https://www.augmentcode.com

Query: How can AI-powered writing assistants create a "coding-like" interactive experience, specifically regarding features like diffing, version control, and iterative refinement based on user instructions?

Answer: Augment Code describes an AI coding platform that emphasizes **precise refinements** and **codebase‑aware autocomplete**, creating a tight iterative loop between user intent and code changes.[8] The platform ranges from **autonomous coding agents** that can implement larger features to tools for fine‑grained edits, enabling users to move quickly through any project while maintaining control.[8]

Context awareness is central: Augment Code leverages full **codebase context** to ensure that refinements and completions are consistent with existing structures, much like an advanced writing assistant would need holistic document context to produce coherent revisions.[8] By positioning its tools as support for both large autonomous tasks and small, precise refinements, the platform parallels different layers of interaction found in coding workflows—from big feature branches down to small commits.[8]

Although the description does not explicitly call out text diffs or version control, the emphasis on integration into the development workflow, and on agents that operate over real software projects, implies that AI changes flow through normal Git‑based processes for review and history.[8] This suggests a pattern for AI writing assistants: allow autonomous restructuring or generation of large sections, but deliver those changes in a form that can be inspected and incrementally accepted, similar to reviewing AI‑authored patches in a code repository.[8]

-----

</details>

<details>
<summary>What are the most effective UX/UI design patterns for presenting AI-generated text edits to a user for validation, specifically comparing 'full document rewrite' versus 'selected text' suggestions?</summary>

### Source [65]: https://aipatterns.substack.com/p/ai-patterns-for-document-editors

Query: What are the most effective UX/UI design patterns for presenting AI-generated text edits to a user for validation, specifically comparing 'full document rewrite' versus 'selected text' suggestions?

Answer: This source analyzes concrete UI patterns for **AI in document editors**, directly relevant to presenting AI-generated text edits.

It identifies **two main patterns**:

1. **Inline Editing pattern**
- The prompting and AI output appear **directly in the main document editor**, using **inline popups, dropdown menus, and small sidebars**.
- Tools examined (e.g., Copy.ai, Jasper, Notion, Microsoft Copilot for Word) let users invoke AI from within the document surface rather than a separate screen.
- Typical flow: the user selects or positions the cursor in text, invokes an inline control, configures options (tone, length, language, keywords), and the AI generates content in-place.
- For Jasper’s inline example, the AI output is inserted **alongside a visible indicator** (e.g., a special cursor label such as “Jasper”) so users can recognize what was AI-generated and review it before fully accepting.
- A **floating action bar** in the document provides contextual actions like **generate / regenerate**, and change length (e.g., S, M, L). These support **progressive refinement** of generated content at the location where it will be used, which aligns closely with a **“selected text” or local suggestion** model rather than full-document overwrite.
- The pattern preserves the user’s mental model of a document editor: the AI augments local edits without replacing the entire document, so users can scan, compare, and adjust in context.

2. **Dual Windowed pattern**
- The interface shows **two panes**: a document editor on one side and an AI-chat or assistant view on the other.
- In Jasper’s “chat mode,” the left column is a chatbot workflow and the right column is the document editor, with **roughly equal space**.
- The user prompts the chatbot, reviews the AI response in the left pane, and then manually transfers content (often via copy-paste) into the document.
- This pattern is closer to a **separate draft / rewrite** experience: AI output is generated outside the main document, then selectively brought in. It is less about automatic full-document rewrite and more about letting users **manually curate** what parts of AI content to insert.

Key takeaway from this source: modern tools largely favor **inline, localized editing and side-by-side patterns** rather than opaque full-document replacements. The patterns emphasize **in-place review, visible AI provenance, and incremental adoption of AI-generated edits**.

-----

-----

### Source [66]: https://www.koruux.com/ai-patterns-for-ui-design/

Query: What are the most effective UX/UI design patterns for presenting AI-generated text edits to a user for validation, specifically comparing 'full document rewrite' versus 'selected text' suggestions?

Answer: This source describes **AI UX patterns** and implementation tips relevant to how users validate and refine AI-generated text.

It emphasizes a **text refinement pattern** that clearly aligns with **selected-text suggestions** rather than full-document overwrite:

- The interface should support **easy text selection or content highlighting**, so users can choose a specific span of text to modify instead of operating on the whole document.
- When text is selected, the system should show **contextual menus** offering refinement actions such as **“regenerate,” “make longer,” “make shorter,” “change tone,” etc.**
- The AI then processes these localized refinement requests and returns updated content.
- The design should support **iterative refinement**, allowing users to repeatedly adjust the updated content until it matches their intent.
- The source cites Gemini as an example where users **select text and choose actions like regenerate or change length**, underscoring that the pattern centers on **local, user-targeted edits**.

For **validation and trust**, the source recommends:

- Use **visual indicators** (badges, icons, color coding) to **differentiate AI-generated from human-verified content**, helping users know what needs review.
- Implement **confidence levels or accuracy scores** for AI-generated content and provide accessible explanations of what these indicators mean.
- Allow users to **filter or sort content** based on verification status or confidence, which can be applied both to full documents and to local edits, but is especially powerful when multiple AI-suggested segments coexist with human text.
- Communicate aspects of the **verification process** for human-verified content.

It also describes a **customization / control pattern**:

- Provide intuitive controls (sliders, toggles, dropdowns) to adjust content characteristics like tone or detail level.
- Offer preset styles (formal, casual, technical, creative) and allow fine-tuning of creativity, detail, or complexity.

Overall, this source favors **fine-grained, selected-text refinement UIs with explicit controls and indicators**, giving users strong **local control and visibility** instead of wholesale full-document rewrites.

-----

-----

### Source [67]: https://userpilot.com/blog/ai-ux-design/

Query: What are the most effective UX/UI design patterns for presenting AI-generated text edits to a user for validation, specifically comparing 'full document rewrite' versus 'selected text' suggestions?

Answer: This source focuses on **AI UX design best practices** in SaaS products, with several principles that apply directly to how AI-generated text edits are presented for validation.

For **presenting and validating AI-generated text**:

- It emphasizes **“editable output”**: users should be able to **tweak what the AI produced instead of starting over**. For text systems, this includes **inline editing** with quick options such as *shorter*, *friendlier*, or *add a CTA*.
- This pattern aligns with **selected-text or local refinement** rather than a one-shot full-document rewrite; users adjust specific AI outputs rather than accepting a global replacement.

On **interaction model**:

- It advocates **“lightweight loops”**: show quick signals that input is being processed, then once the output is ready, present simple reactions like thumbs up/down, “Was this helpful?”, or “Show another.”
- This supports **progressive validation**: users can easily reject or ask for alternative versions of a suggested segment (e.g., a paragraph rewrite) before committing it.

On **progressive refinement**:

- Instead of forcing users to accept or reject a single output, interfaces should let them **refine results step by step**.
- For copy, it suggests secondary actions such as **“make it shorter” or “adjust the tone”**. This pattern fits naturally with **segment-level edits**, where a user validates one portion at a time.

On the role of AI:

- The article notes that much AI usage in design teams is for text-based tasks and **UX copy**, reinforcing that AI is commonly used to generate microcopy, onboarding text, or error messages in context rather than rewriting entire documents.

In combination, these practices imply that for user validation, **UI patterns that support inline, segment-level editing, feedback, and iterative refinement** are preferred over monolithic full-document rewrites, because they keep the user in control and make validation more manageable.

-----

-----

### Source [68]: https://www.uxstudioteam.com/ux-blog/ai-design-patterns-in-saas-products

Query: What are the most effective UX/UI design patterns for presenting AI-generated text edits to a user for validation, specifically comparing 'full document rewrite' versus 'selected text' suggestions?

Answer: This source discusses **AI design patterns in SaaS products** with a strong focus on **review and accountability**, which informs how AI-generated edits should be presented.

For **review workflows**:

- The authors describe adding **inline review tools** so users could quickly **accept, adjust, or reject an AI-generated draft directly where they were working**.
- This suggests a **co-editing pattern** where AI content appears within the main workspace at the right location (e.g., a paragraph or section) and is accompanied by controls for localized validation.
- Instead of routing users to an external review screen or imposing a full-document rewrite, this approach **integrates review into the existing workflow**.

On **reasoning and transparency**:

- They implemented **reasoning displays** that show why certain documents or sources were linked to a response.
- This helps reviewers decide whether to approve AI text or request changes, and is particularly relevant when AI suggests substantial edits or new content.

On **user control and trust**:

- One principle is that **AI should give users a head start instead of taking the process away from them**. In practice, this means providing a **draft or recommendation** while ensuring the **user remains the final decision-maker**.
- They recommend **confidence indicators, version history, and side-by-side comparison of drafts** so users know when to double-check and can see how AI edits differ from prior versions.

Taken together, this source supports patterns where:

- AI provides **drafts or segment-level suggestions** rather than silently replacing entire documents.
- Users **validate in place**, with inline controls to accept/reject.
- When larger-scale changes are proposed (closer to a full rewrite), **versioning and side-by-side views** help users compare the original vs. AI-edited versions before deciding, mitigating the risks of full-document replacement.

-----

-----

### Source [69]: https://www.aufaitux.com/blog/agentic-ai-design-patterns-enterprise-guide/

Query: What are the most effective UX/UI design patterns for presenting AI-generated text edits to a user for validation, specifically comparing 'full document rewrite' versus 'selected text' suggestions?

Answer: This source outlines **agentic AI design patterns** for enterprise UX, with principles that guide how AI-driven changes (including text edits) should be presented for validation.

Relevant **UX guidelines for AI actions** include:

- **Display confidence levels or reliability indicators** showing how certain the system is about its actions. For text edits, these indicators can help users prioritize what to review, whether at the full-document level or for specific sections.
- **Enable users to validate, modify, or override agent actions**, especially during early deployments. This directly maps to providing clear controls for accepting or editing AI-generated text rather than applying automatic full rewrites.
- Allow systems to **shift from suggestion-based to more autonomous modes** as user familiarity and system reliability increase. In text editing, this means starting with **suggestions that require explicit user validation** (e.g., selected-text suggestions or draft rewrites with review) and only later considering background, larger-scale edits when trust is established.
- **Use design cues to distinguish AI-driven actions from user-initiated ones**, reinforcing accountability. For documents, that may include markers or labels indicating which sections were edited by AI, supporting targeted validation.

For enterprise scenarios with complex workflows:

- The emphasis is on making agentic AI **predictable, auditable, and actionable**, which favors interfaces that keep a **clear record of what the AI changed**, when, and why.
- When larger, document-wide changes are made, the need for **auditability and oversight** suggests patterns like **version history, change logs, and explicit approval steps**, rather than silent full-document overwrites.

Overall, the source supports a pattern where AI offers **transparent, controllable suggestions**—both at the segment and document level—but always with **strong user validation mechanisms**, especially important when contemplating full-document rewrites.

-----

-----

### Source [70]: https://uxdesign.cc/20-genai-ux-patterns-examples-and-implementation-tactics-5b1868b7d4a1

Query: What are the most effective UX/UI design patterns for presenting AI-generated text edits to a user for validation, specifically comparing 'full document rewrite' versus 'selected text' suggestions?

Answer: This source catalogs **GenAI UX patterns** and implementation tactics, several of which are applicable to presenting AI-generated text edits.

Key patterns relevant to **validation of AI edits** include:

- **Design for coPilot, co-Editing, or partial automation**: Rather than AI fully automating tasks, the interface is designed for **collaborative work**, where AI assists and humans retain control. In document editing, this implies **partial or selected-text suggestions**, where AI proposes changes and the user reviews them inline, as opposed to one-click full-document rewrites with no granular control.

- **Define user controls for automation**: Interfaces should let users decide **when and how much the AI automates**. Applied to text editing, users might choose whether AI operates on a **highlighted segment, a paragraph, or the whole document**, and the UI must make this choice explicit so users understand the scope of changes they need to validate.

- **Design to capture user feedback**: The article recommends integrating **real-time mechanisms for users to label outputs as helpful, harmful, incorrect, or unclear**. This supports validation loops where users can provide feedback on AI-suggested edits and improve future suggestions.

- **Design for model evaluation and safety guardrails**: Encourage patterns where human evaluation is combined with automated checks (e.g., an LLM-as-a-judge plus human review) to reach high accuracy. For large or sensitive document rewrites, this indicates the need for **extra verification steps** before applying bulk changes.

- **Design for user input and system error states**: Handle ambiguous user prompts or system errors gracefully, and ask clarifying questions when needed, reducing the chance that an unintended full-document change is made without review.

Overall, this source emphasizes **co-editing, explicit scope control, and built-in feedback mechanisms**, which naturally support **selected-text and segment-level patterns** while making any full-document operations more deliberate and reviewable.

-----

</details>

<details>
<summary>What is the role of a "human-in-the-loop" when applying the evaluator-optimizer pattern in generative AI, and how is human feedback prioritized over automated evaluation criteria?</summary>

### Source [71]: https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/evaluator-reflect-refine-loop-patterns.html

Query: What is the role of a "human-in-the-loop" when applying the evaluator-optimizer pattern in generative AI, and how is human feedback prioritized over automated evaluation criteria?

Answer: According to AWS Prescriptive Guidance, the **evaluator pattern** in agentic/LLM systems uses one or more *evaluator agents* (often LLMs) to automatically assess intermediate or final outputs and provide structured feedback that an *optimizer* (another agent or the same model on a new call) uses to refine the answer.[8] The pattern creates a **reflect–refine loop**, where the system iteratively improves generations based on evaluation criteria such as correctness, completeness, safety, style, or policy adherence encoded in prompts or tooling.[8]

In AWS’s description, this loop is **fully automated**: evaluators are typically LLM-based and use predefined rubrics or checks (hallucination detection, constraint satisfaction, etc.) to score or critique outputs.[8] However, AWS notes that for high‑risk domains or when automated criteria are insufficient, organizations can integrate **human reviews as part of the evaluation stage**, for example by routing certain outputs for manual approval or by using human feedback as labeled evaluation data that calibrates or validates the automated evaluators.[8]

In such configurations, **human judgment effectively becomes the source of truth**: automated evaluators are tuned or overridden based on human‑generated labels, and workflows can be designed so that a human "gate" (approval/rejection) is required before the agent’s output is accepted or deployed.[8] This means that when human and automated evaluations disagree in safety- or business‑critical contexts, systems are structured so that **human feedback has final authority**, with evaluator scores serving as decision support rather than the ultimate arbiter.[8]

-----

-----

### Source [72]: https://cloud.google.com/discover/human-in-the-loop

Query: What is the role of a "human-in-the-loop" when applying the evaluator-optimizer pattern in generative AI, and how is human feedback prioritized over automated evaluation criteria?

Answer: Google Cloud defines **Human‑in‑the‑Loop (HITL)** as a design approach in which humans are deliberately embedded at key stages of an AI/ML system—training, validation, and inference—to provide labels, review outputs, and make or confirm decisions.[7] Humans contribute domain expertise, contextual understanding, and ethical judgment that current automated systems cannot reliably replicate, especially in ambiguous or high‑impact cases.[7]

In the context of evaluation patterns, Google emphasizes that HITL workflows typically rely on **AI for bulk pre‑processing and scoring**, and then escalate uncertain, sensitive, or low‑confidence cases to humans for review.[7] Humans may **approve, correct, or reject** the model’s output; their decisions are then used both as the operational ground truth and as feedback data to improve or recalibrate automated components.[7]

This structure inherently **prioritizes human feedback** over automated criteria: the model’s predictions and any automatic evaluators are treated as provisional until confirmed or adjusted by a human where the workflow calls for review.[7] Google highlights that HITL is crucial for building **trustworthy and responsible AI**, particularly in areas involving safety, fairness, or regulatory compliance.[7] In such settings, human reviewers have **final decision‑making authority**, while automated evaluation scores are used to triage work and provide decision support, not to overrule human judgment.[7]

-----

-----

### Source [73]: https://www.tredence.com/blog/hitl-human-in-the-loop

Query: What is the role of a "human-in-the-loop" when applying the evaluator-optimizer pattern in generative AI, and how is human feedback prioritized over automated evaluation criteria?

Answer: Tredence describes **Human‑in‑the‑Loop (HITL)** as a collaborative framework where humans guide and oversee AI/ML systems throughout their lifecycle to enhance adaptability, reliability, and accuracy, especially in GenAI and large language model use cases.[3] Humans train models, provide nuanced feedback, and validate outputs, combining machine efficiency with human ethical insight.[3]

In evaluation workflows, HITL is framed as a **continuous feedback loop**: humans label data, score model outputs, and correct errors, and these corrections are fed back into model training and evaluation pipelines to improve performance over time.[3] Tredence explicitly mentions **HITL reinforcement learning**, where humans train AI systems by reviewing outputs and indicating preferred or correct responses, guiding the optimization process.[3]

The article recommends best practices that make clear how **human feedback is prioritized** over automated criteria:

- **Define HITL roles** such as reviewers and validators who *verify* and *confirm* model decisions, ensuring that model outputs are not accepted without human oversight in designated workflows.[3]
- Use **active learning** so that human attention is focused on low‑confidence or ambiguous cases, where automated evaluation is least reliable and human judgment is needed to establish ground truth.[3]
- Implement **feedback integration** mechanisms where human insights are systematically incorporated into retraining and model adjustment, effectively using human decisions as the standard against which models and evaluators are tuned.[3]

By positioning human reviewers as validators whose decisions drive retraining and override incorrect system behavior, the piece implies that in HITL evaluator‑optimizer setups, **human feedback sets the authoritative label**, while automated evaluation serves to route, prioritize, and accelerate—but not supersede—human judgment.[3]

-----

-----

### Source [74]: https://parseur.com/blog/human-in-the-loop-ai

Query: What is the role of a "human-in-the-loop" when applying the evaluator-optimizer pattern in generative AI, and how is human feedback prioritized over automated evaluation criteria?

Answer: Parseur’s HITL guide explains that in production AI workflows, especially for tasks like document processing, the system typically **flags low‑confidence or ambiguous outputs for human review**, while high‑confidence results may pass through automatically.[4] During the **testing and feedback** phase, humans *correct or validate* the AI’s outputs, and those corrections are then used to retrain the model, forming a continuous learning cycle.[4]

In practice, this means that when an automated confidence score or evaluation criterion disagrees with a human reviewer’s assessment, the **human correction becomes the new ground truth** for both operational data and subsequent model updates.[4] Parseur notes that humans verify critical fields (e.g., totals, names, dates) before data is forwarded to downstream systems, making human approval a gate that takes precedence over the system’s own assessment.[4]

The article recommends using humans **strategically**, mainly on edge cases, low‑confidence predictions, or periodic audits.[4] Automated evaluation (such as confidence thresholds) is used to select which instances are escalated, but **final authority on those escalated cases rests with the human reviewer**, whose decisions are then fed back for model refinement.[4]

Although Parseur does not use the specific "evaluator‑optimizer" terminology, its described pattern is analogous: automated scoring/evaluation identifies candidates for optimization; **human‑in‑the‑loop supplies authoritative feedback**, which the optimization step (retraining or rule adjustment) must follow.[4] This makes clear that in such evaluator‑style loops, **human feedback is prioritized over automated metrics** whenever the two conflict, particularly in high‑stakes or low‑confidence scenarios.[4]

-----

-----

### Source [75]: https://www.digitaldividedata.com/blog/human-in-the-loop-for-generative-ai

Query: What is the role of a "human-in-the-loop" when applying the evaluator-optimizer pattern in generative AI, and how is human feedback prioritized over automated evaluation criteria?

Answer: Digital Divide Data (DDD) defines **human‑in‑the‑loop for generative AI** as integrating human expertise from data annotation through model evaluation, creating a **feedback loop** that improves the model over time and ensures outputs align with societal values and organizational requirements.[1] Humans participate in data labeling, training oversight, and especially in **testing and evaluation**, where they assess generative outputs, correct inaccuracies, and refine decision‑making.[1]

For generative models, DDD emphasizes that human oversight is key to **ensuring accuracy, reliability, and ethical behavior**—for example, in detecting subtle misinformation, understanding regional context, or handling ambiguous cases that automated metrics or evaluators might miss.[1] Human annotators and reviewers can identify biased or harmful outputs and adjust guidelines and training data so that the model and any automated evaluators better reflect human values.[1]

Applied to evaluator‑optimizer patterns, this implies that **human feedback provides the normative standard**: automated evaluators can score or critique outputs, but humans ultimately decide what is accurate, appropriate, or aligned with policy and ethics, and their corrections feed back into model updates.[1] DDD notes that this integration offers benefits such as improved accuracy, continuous improvement, and better risk management for generative AI deployments.[1]

By positioning HITL as essential for balancing innovation with accountability, the article indicates that in generative AI workflows that use evaluation loops, **human judgment and corrections override purely automated evaluation criteria** when discrepancies arise, especially in sensitive or value‑laden applications.[1]

-----

-----

### Source [76]: https://coconote.app/notes/0c65cec5-0763-416d-b71a-006037c13ee2/transcript

Query: What is the role of a "human-in-the-loop" when applying the evaluator-optimizer pattern in generative AI, and how is human feedback prioritized over automated evaluation criteria?

Answer: The transcript on evaluator and human involvement describes the **evaluator–optimizer workflow** as a loop where one LLM generates a response and another LLM acts as an evaluator, providing feedback that the generator uses to iteratively refine its output.[2] This is recommended when **evaluation criteria are clear and can be articulated**—for example, adherence to formatting rules or objective correctness—so the evaluator can reliably score or critique outputs.[2]

The speaker contrasts this with a **human‑in‑the‑loop variant** of the same pattern: instead of an AI evaluator, *a human evaluates the output and provides feedback*, which the system then uses to optimize the next iteration.[2] The structural pattern is almost identical, but the substitution of a human evaluator changes when and why to use it.[2]

According to the transcript, **human‑in‑the‑loop is preferred** when:

- Tasks are **critical** (high stakes), where errors from a purely automated evaluator are unacceptable.[2]
- Tasks depend heavily on **user preference** or subjective quality, where the criteria cannot be fully captured in a fixed rubric that an LLM evaluator can apply.[2]

In these scenarios, the human’s feedback becomes the **authoritative evaluation signal**. Automated evaluators may still exist in the workflow to pre‑screen or suggest improvements, but they are not considered sufficient to replace human judgment.[2] By contrast, in domains where criteria are clear and deterministic, the evaluator–optimizer loop can remain fully automated and more efficient.[2]

The transcript explicitly notes that the two patterns are so similar that evaluator–optimizer with a human could be seen as "AI in the loop," but underscores that for preference‑driven or high‑risk tasks, **human evaluation takes priority** over automated evaluation, guiding how the system refines its outputs.[2]

-----

</details>

<details>
<summary>What are the key principles for designing prompts that effectively guide a large language model to perform targeted text editing based on user feedback, while preserving the context and style of the surrounding document?</summary>

### Source [77]: https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api

Query: What are the key principles for designing prompts that effectively guide a large language model to perform targeted text editing based on user feedback, while preserving the context and style of the surrounding document?

Answer: OpenAI’s prompt engineering best practices emphasize several principles that are directly applicable to targeted text editing while preserving context and style.[3]

First, **put clear instructions at the beginning** of the prompt and then separate them from the document text using delimiters such as `###` or `"""`.[3] This helps the model reliably distinguish between what it must *do* (instructions) and what it must *edit* (context), which is critical when modifying only a portion of a longer document.

OpenAI recommends **being explicit and specific about the task**, including the editing objective, constraints, and desired output format.[3] For targeted edits, this means clearly stating which span should change, what type of change is required (e.g., “fix grammar only,” “incorporate this user feedback,” “shorten this paragraph by 20%”), and whether unchanged parts of the document should be copied verbatim.

The guidance also encourages **providing the model with sufficient context** around the editable region, not just the isolated snippet.[3] Including relevant surrounding text helps the model preserve the original tone, style, and logical flow when applying edits.

OpenAI highlights the value of **instruction templates and patterns**, such as: role description, high-level task description, detailed constraints, then the raw content.[3] For editing workflows, this structure allows developers to systematically incorporate user feedback as part of the instruction block (e.g., “Apply the following user feedback to the highlighted section, without changing the rest of the document.”).

Finally, OpenAI suggests **iterative prompting** and decomposition for complex transformations.[3] For example, you might first ask the model to restate the user feedback in its own words, then in a second step apply that interpretation to the text segment. Breaking the task into steps improves control and reduces unintended changes to style or context.

-----

-----

### Source [78]: https://claude.com/blog/best-practices-for-prompt-engineering

Query: What are the key principles for designing prompts that effectively guide a large language model to perform targeted text editing based on user feedback, while preserving the context and style of the surrounding document?

Answer: Anthropic’s best practices stress **explicitness, context, and task-splitting**, all crucial for targeted text editing guided by user feedback.[2]

They recommend **clear, unambiguous instructions**: do not rely on the model to infer your intentions; instead, directly specify what must be edited, what must remain unchanged, and what constraints apply (e.g., “Preserve the original writing style and only fix clarity issues in the highlighted sentences”).[2]

The article emphasizes **providing context and motivation**—explaining *why* a change is desired so the model can better align edits with user goals.[2] In an editing scenario, this can mean embedding user feedback like: “The user wants this section to sound more formal but still friendly; apply that change only to the introduction paragraph.” Stating the underlying objective helps the model adjust tone while respecting the surrounding style.

Anthropic highlights **prompt chaining** for complex tasks: break editing workflows into sequential steps, each with a focused instruction.[2] For example: (1) analyze the user feedback and summarize the requested changes; (2) propose a revised version of the target span while preserving style; (3) optionally, compare the original and edited text to verify that changes are localized. This decomposition improves reliability and reduces unintended global rewrites.

They also outline a diagnosis–refinement loop: if responses are generic or off-target, **add specificity and examples**, including before/after snippets that demonstrate acceptable edits.[2] Few-shot examples are especially useful for teaching the model how much of the surrounding context to preserve and what type of edits are acceptable.

The guidance warns against **over-engineering prompts**; even for nuanced editing, the core instruction should remain simple and direct, with complexity moved into examples or stepwise prompts rather than long, convoluted single instructions.[2]

-----

-----

### Source [79]: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/prompt-engineering

Query: What are the key principles for designing prompts that effectively guide a large language model to perform targeted text editing based on user feedback, while preserving the context and style of the surrounding document?

Answer: Microsoft’s Azure OpenAI prompt engineering guidance provides scenario-specific techniques that map well to controlled text editing workflows.[4]

They stress that effective prompts should **leverage the model’s strengths while respecting its limitations**, and that responses must still be validated.[4] For editing tasks guided by user feedback, this implies designing prompts that tightly constrain the scope of changes, then programmatically or manually checking that the model did not drift outside that scope.

A key technique is to **break the task down** into smaller steps.[4] Instead of asking the model to “apply all user feedback to this long document,” the guidance suggests structuring the process so that the model first extracts or summarizes the feedback-relevant parts, then applies edits to specific sections. This decomposition helps preserve the overall document context because each step operates on a narrower, better-defined subset of text.

The documentation also discusses **scenario-specific guidance** more broadly: prompts should be tailored to the use case, including clear task objectives and any domain constraints.[4] For targeted editing, this might involve specifying that legal terms or code segments must not be altered, or that formatting (headings, bullet structure, citations) must be preserved while modifying only prose content.

Microsoft emphasizes understanding and controlling **grounding and accuracy**.[4] When incorporating user feedback that might introduce new information, prompts should indicate whether the model may rephrase only, or also supplement with external knowledge. For document-preserving edits, it is safer to instruct the model to operate purely on the given text and user comments, without inventing new factual content.

Finally, they highlight that a carefully crafted prompt for one scenario may not generalize; editing pipelines should therefore **iterate and test prompts** against varied documents and feedback types to ensure that style and context are consistently preserved.[4]

-----

-----

### Source [80]: https://cloud.google.com/discover/what-is-prompt-engineering

Query: What are the key principles for designing prompts that effectively guide a large language model to perform targeted text editing based on user feedback, while preserving the context and style of the surrounding document?

Answer: Google Cloud’s guide on prompt engineering defines it as designing and optimizing prompts so models understand intent, follow instructions, and generate desired outputs, which is directly relevant to targeted editing.[6]

They highlight **prompt structure and style** as central: different models may respond better to natural language questions, direct commands, or structured inputs with fields.[6] For editing based on user feedback, structured prompts can clearly separate components such as: original text, user feedback, editing instructions, and desired output format. This separation helps preserve surrounding context and style by making constraints explicit.

The guide emphasizes **providing context and relevant examples** to improve accuracy and relevance.[6] When preserving document style, prompts should include enough of the surrounding text and possibly short examples of on-style vs. off-style edits. This helps the model align its edits with the local tone, formality level, and formatting patterns.

Google notes that **adapting prompts based on user feedback or model outputs** improves performance over time.[6] In an editing workflow, this can be implemented as a loop where user feedback is captured and encoded into subsequent prompts, gradually refining the model’s understanding of preferred style and the acceptable degree of change.

They also discuss **fine-tuning combined with tailored prompts** for domain-specific tasks.[6] For organizations that frequently use targeted editing in a particular voice (e.g., brand style), fine-tuning plus prompts that reiterate style constraints can further stabilize the model’s behavior when making local edits.

Overall, Google’s guidance underscores that prompt engineering should explicitly communicate intent, context, and constraints, and that **prompt optimization is iterative**, continuously incorporating observed behavior and user feedback—principles that align closely with reliable, context-preserving text editing workflows.[6]

-----

-----

### Source [81]: https://mirascope.com/blog/prompt-engineering-best-practices

Query: What are the key principles for designing prompts that effectively guide a large language model to perform targeted text editing based on user feedback, while preserving the context and style of the surrounding document?

Answer: Mirascope’s prompt engineering best practices provide several actionable principles applicable to targeted text editing based on user feedback.[1]

They stress **specifying exactly what you want**: vague instructions cause the model to guess, which can lead to unintended broad rewrites instead of precise edits.[1] For editing tasks, prompts should clearly identify the editable region, define the type of edit (e.g., “clarify,” “shorten,” “adjust tone per this feedback”), and state that all other text must remain unchanged.

Mirascope recommends **clear, action-oriented instructions**, starting prompts with strong verbs like “Edit,” “Revise,” or “Update,” rather than softer phrasing.[1] Treating the model like a function with explicit inputs and outputs reduces ambiguity. For instance: “Edit the highlighted paragraph to address the user’s feedback below, preserving the existing style and structure.”

They emphasize **guiding the model to think step-by-step**, referencing techniques like chain-of-thought and prompt chaining.[1] In an editing context, one step might analyze and restate the user feedback; another step applies those changes to the specified text; a final step can verify that surrounding context and tone are preserved. This staged approach reduces hallucination and keeps edits focused.

The article notes that **zero-shot prompts with minimal guidance are risky** for nuanced tasks.[1] For complex editorial changes, few-shot examples (before/after pairs) can demonstrate the acceptable magnitude of changes and how to maintain style continuity.

Mirascope also underscores that good prompt engineering is about **reducing the model’s room for assumption**.[1] When working with long documents, prompts should explicitly state whether the model may reorder content, introduce new information, or only rephrase existing material according to user comments. This helps maintain both the semantic context and stylistic consistency of the document while implementing targeted edits.

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>Human-in-the-loop refers to the intentional integration of human oversight into autonomous AI workflows at critical decision points. Instead of letting an agent execute tasks end-to-end and hoping it makes the right call, HITL adds user approval, rejection, or feedback checkpoints before the workflow continues.</summary>

Human-in-the-loop refers to the intentional integration of human oversight into autonomous AI workflows at critical decision points. Instead of letting an agent execute tasks end-to-end and hoping it makes the right call, HITL adds user approval, rejection, or feedback checkpoints before the workflow continues.

AI systems can route messages, update records, make decisions, and trigger entire workflows across multiple apps without you touching anything. But as AI shifts more and more from being an assistive tool to powering autonomous systems, humans have a new kind of responsibility: making sure nothing goes wrong.

Because even the smartest AI systems still struggle to understand nuance, edge cases, or the unwritten rules teams use to make decisions. When autonomous agents act without that context, small gaps can quickly turn into big problems. That's where human-in-the-loop (HITL) comes in.

Instead of letting an AI system run unchecked, you can design checkpoints where humans step in with experience, context, and common sense. This intervention ensures that the decisions that should involve humans actually do involve humans.

So let's take a look at what human-in-the-loop really means, why it's necessary, and some practical patterns for adding HITL to agentic workflows.

## What does human-in-the-loop mean?

Human-in-the-loop refers to the intentional integration of human oversight into autonomous AI workflows at critical decision points. Instead of letting an agent execute tasks end-to-end and hoping it makes the right call, HITL adds user approval, rejection, or feedback checkpoints before the workflow continues.

Let's say you're building an automated lead gen system that identifies potential customers, adds them to your CRM, and sends out targeted emails. Most of that work can run autonomously, but you might need human approval if the agent wants to update an _existing_ customer record or maybe to review emails before they get sent.

And it's not about being a control freak (although it's great for that too). HITL gives you all the benefits of AI automation running at full speed _plus_ peace of mind when decisions carry risk, nuance, or downstream impact. It prevents irreversible errors, ensures compliance in regulated scenarios, and catches ethical issues that AI might overlook. Every approval, rejection, or correction from a human to the AI workflow also becomes training data for the agent. Over time, AI systems learn from your feedback and improve performance.

## When should you use HITL in AI workflows?

If you have an AI agent taking action on your behalf, think hard about where you might need a human in the loop. While the goal of AI automation is speed, speed becomes less relevant when AI makes a bad judgment call. HITL acts as a safety net that defines when, where, and how to include humans before an automated workflow continues.

Here are a few general situations where agentic workflows should pause and request human oversight, but you should look at each of your workflows on a case-by-case basis.

### Low confidence or ambiguity

Imagine a customer message comes in: "My invoice is wrong, and I need this fixed ASAP." Is that a billing dispute? A refund request? A technical issue?

If the agent can't confidently classify the message, the workflow should pause and escalate to a human instead of guessing (which we know AI loves to do). This also applies when an agent's confidence score in a particular situation falls below a pre-defined (by you) threshold.

### Sensitive actions

If there are actions that could lead to accidental data loss or permanent errors, you need a human in the loop. For example, you'll want an HITL checkpoint when executing actions like overwriting customer records or deleting data. Your risk tolerance here comes into play, but it's better to start with more HITL and then pull back if you see the AI working well.

### Regulatory and compliance implications

Where actions carry regulatory or compliance implications, you need to add human oversight. For example, if an agent drafts a contract, a human lawyer should review all language before anything gets sent or signed.

### Anything requiring empathy

Tasks and decisions that require empathy and human judgment shouldn't be left to AI alone. AI agents can do a lot, but they can't truly empathize with another human, and if they try, it'll be immediately discarded (rightfully so) as disingenuous. You also need to consider potential bias in AI. Of course, humans are also biased, but having an HITL checkpoint whenever bias might come into play can help avoid potential issues.

## How to add human-in-the-loop to AI workflows

Once you've identified _where_ human judgment matters, the next step is to actually build those checkpoints into your AI workflow.

Human-in-the-loop can take several forms, including approvals, requests for context, or verification steps. But it doesn't mean slowing down automation or reviewing every action. Instead, the goal is to let the workflow run on its own until it reaches a point where human input is required.

Here are some practical patterns for adding human oversight into an agentic workflow, along with some actionable ways to add them to your work.

### Approval flows

Approval flows involve pausing an agent's workflow at a pre-determined checkpoint until a human reviewer approves or declines the agent's decision.

The team at SkillStruct told me they use this approval flow to review all AI-generated career recommendations before they reach users. After generating a career recommendation based on a user's profile, the AI system sends an email alert to the development team. From there, a developer uses a custom app to review the output. If approved, the recommendation is shown to the user, while rejected content is deleted.

### Confidence-based routing

Confidence-based routing pulls in a human if the AI agent encounters ambiguity. For this pattern, the agent's instructions include a defined confidence score, and if its confidence falls below the threshold, it automatically defers to a human.

Confidence-based routing is ideal for agentic workflows that handle a wide range of usually clear-cut tasks (like categorizing incoming customer requests) but require a fallback mechanism for ambiguous edge cases.

For example, the owner of Tradesmen Agency shared with me that he built an AI-orchestrated invoice processing system that autonomously handles invoices, parsing attachments with Llama Parse, and extracting structured data using a large language model within Zapier and other tools. But whenever the system encounters exceptions or uncertain results (like missing or conflicting data fields, a vendor or PO not found in Entrata records, or validation confidence below a certain threshold), the workflow routes the case into an exception log and sends an email notification for manual review. From there, a human reviewer can validate the data, correct details, or approve the record for upload.

### Escalation paths

With escalation paths, a human operator can step in when an action falls outside an agent's scope, to keep the automation from failing.

Say an AI agent tasked with processing refund requests encounters a request above its value threshold. Instead of retrying endlessly and stalling the workflow, the agent routes the task to finance with a note like "refund request exceeds automated limit, needs human review."

### Feedback loops

Feedback loops allow a human to work alongside an AI agent by building feedback mechanisms directly into the workflow. When agents execute a task, a reviewer can evaluate it, give a quick thumbs-up, or provide more detailed feedback to correct the agent so the correction becomes input for future iterations.

For instance, the team at ContentMonk uses an AI content system to automate 70-80% of their content ops. Human operators work alongside the AI system to provide insight and review outputs at different stages of the content production process, while the system automates the writing. Before a brief is generated, a human gives input with details such as tone of voice, ICP, messaging framework, and brand guidelines. The AI-generated brief is then reviewed, edited, and approved before a draft is generated. Once the draft is ready, a human reviews it again to ensure it meets brand requirements before adding images and publishing.

### Audit logging

Not every agentic workflow needs a human to stop and approve decisions in real time. Sometimes you just need visibility, and audit logging lets automation run at full speed while recording every action for later review.

For instance, if any agent updates CRM records after a customer call, each change is automatically logged: what changed, when it changed, and why. No approvals needed, no workflow interruptions. Audit logs give humans traceability without creating hard stops, making them ideal for workflows where oversight matters but immediate human intervention doesn't.

## Why does human-in-the-loop matter?

HITL steps allow AI and automation to move fast while keeping you in control of the decisions that carry risk, involve more nuance than AI can handle, or require human accountability.

Plus, with feedback loops in place, human corrections become training data, which makes AI agents smarter and more aligned with your preferred outcomes. When humans guide the moments where AI lacks context or judgment, the system becomes more reliable and adaptive over time.

Equally as important: approvals, escalation points, and audit logs give teams visibility into AI workflows and reasoning, helping reduce the black-box effect of AI. This supports internal accountability and transparency at a company, as well as coming into play during compliance auditing and internal reviews.

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/towardsai/agentic-ai-engineering-course/blob/dev/lessons/24_human_in_the_loop/notebook.ipynb</summary>

# Repository analysis for https://github.com/towardsai/agentic-ai-engineering-course/blob/dev/lessons/24_human_in_the_loop/notebook.ipynb

## Summary
Repository: towardsai/agentic-ai-engineering-course
Branch: dev
Commit: 96cc56ad477dedb505771ba8ab03b1fe03df5133
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
!curl -L -o configs.zip https://raw.githubusercontent.com/iusztinpaul/agentic-ai-engineering-course-data/main/data/configs.zip
!unzip configs.zip
!rm -rf configs.zip

"""
Now, let's download the inputs folder containing profiles, examples, and test data:
"""

%%capture

!rm -rf inputs
!curl -L -o inputs.zip https://raw.githubusercontent.com/iusztinpaul/agentic-ai-engineering-course-data/main/data/inputs.zip
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
## 2. Adding Human-In-The-Loop In Our Writing Workflow

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
<img src="https://raw.githubusercontent.com/iusztinpaul/agentic-ai-engineering-course-data/main/images/l24_writing_workflow.png" alt="Workflow" height="700"/>
"""

"""
The diagram shows how Brown as an MCP server exposes three main tools, with a human feedback loop that allows iterative refinement until you're satisfied with the article.
"""

"""
## 3. Introducing Human Feedback Into the Article Reviewer

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
## 4. Implementing the Article Editing Workflow

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
### 4.5 Running the Edit Article Workflow
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
## 5. Implementing the Selected Text Editing Workflow

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

[... Content truncated due to length ...]

</details>


## YouTube Video Transcripts

<details>
<summary>[ 00:00 ] (Upbeat music plays as a speaker is introduced on a large screen displaying "Y COMBINATOR PRESENTS AI STARTUP SCHOOL" with geometric patterns.)</summary>

[ 00:00 ] (Upbeat music plays as a speaker is introduced on a large screen displaying "Y COMBINATOR PRESENTS AI STARTUP SCHOOL" with geometric patterns.)
Please welcome former director of AI Tesla, Andrej Karpathy.

[ 00:09 ] (Andrej Karpathy walks onto the stage to applause, smiling and waving to the audience. The large screen behind him shows his face with the title "Software in the era of AI" and "Andrej Karpathy".)
Hello.
(He takes the stage, holding a presentation clicker.)
Wow, a lot of people here. Hello.
(He smiles and adjusts his sleeves.)

[ 00:22 ] Um, okay, yeah, so I'm excited to be here today to talk to you about software in the era of AI. (The screen changes to a slide with a picture of the Golden Gate Bridge and lavender fields, titled "Software in the era of AI".)
And I'm told that many of you are students, like bachelors, masters, PhD, and so on, and you're about to enter the industry. And I think it's actually like an extremely unique and very interesting time to enter the industry right now.

[ 00:52 ] And I think fundamentally the reason for that is that um, software is changing, uh, again. (The slide changes to white text on a black background: "Software is changing. (again)").
And I say again because I actually gave this talk already, um, but the problem is that software keeps changing, so I actually have a lot of material to create new talks. And I think it's changing quite fundamentally. I think, roughly speaking, software has not changed much on such a fundamental level for 70 years, and then it's changed, I think, about twice quite rapidly in the last few years. And so there's just a huge amount of work to do, a huge amount of software to write and rewrite.

[ 01:21 ] So let's take a look at maybe the realm of software. So if we kind of think of this as like the map of software, this is a really cool tool called Map of GitHub. (The slide displays a "Map of GitHub" showing clusters of blue dots, representing repositories, against a dark blue background.)
Um, this is kind of like all the software that's written. Uh, these are instructions to the computer for carrying out tasks in the digital space. So if you zoom in here, (A zoomed-in section highlights a cluster of dots, showing intricate connections between them.) these are all different kinds of repositories, and this is all the code that has been written. And a few years ago, I kind of observed that, um, software was kind of changing and there was kind of like a new soft, new type of software around, and I called this Software 2.0 at the time. (The slide changes to an image of computer code (Software 1.0) at the top, and a neural network diagram (Software 2.0 = weights) at the bottom.)

[ 01:54 ] And the idea here was that Software 1.0 is the code you write for the computer. Software 2.0 are basically neural networks, and in particular, the weights of a neural network. And you're not writing this code directly; you are most you are more kind of like tuning the data sets and then you're running an optimizer to create to create the parameters of this neural net. And I think like at the time, neural nets were kind of seen as just a different kind of classifier, like a decision tree or something like that. And so I think, uh, it was kind of like, um, I I think this framing was a lot more appropriate. And now actually what we have is kind of like an equivalent of GitHub in the realm of Software 2.0. (The slide now shows the "Map of GitHub" (Software 1.0) on the left and a "HuggingFace Model Atlas" (Software 2.0) on the right, both displaying intricate network structures.)
And I think the HuggingFace is basically equivalent of GitHub in Software 2.0. And there's also Model Atlas, and you can visualize all the code written there. In case you're curious, by the way, the giant circle, the point in the middle, uh, these are the parameters of Flux, the image generator.

[ 02:50 ] And so anytime someone tunes a LoRA on top of a Flux model, you basically create a Git commit, uh, in this space, and, uh, you create a different kind of a image generator. So basically what we have is Software 1.0 is the computer code that programs a computer. (The slide shows "Software 1.0 = computer code" with an arrow pointing to "programs", then "computer". Below is an old photo of a person using a computer terminal.)
Software 2.0 are the weights, which program neural networks. (The slide now also shows "Software 2.0 = weights" with an arrow pointing to "programs", then "neural net". Below is a diagram of AlexNet for image recognition (~2012).)
Uh, and here's an example of AlexNet image recognizer neural network. Now, so far, all of the neural networks that we've been familiar with until recently, were kind of like fixed function computers. Image to categories or something like that. And I think what's changed, and I think it's quite fundamental change, is that neural networks became programmable with large language models. (The slide now adds "Software 3.0 = prompts" with an arrow pointing to "programs", then "LLM". Below is a diagram of an LLM (~2019).)
And so I I see this as quite new, unique, it's new kind of a computer. And, uh, so in my mind, it's, uh, worth giving it a new designation of Software 3.0.

[ 03:20 ] And basically, your prompts are now programs that program the LLM. And, uh, remarkably, uh, these, uh, prompts are written in English. (The slide shows an example of "Sentiment Classification" for Software 1.0 (Python code), Software 2.0 (10,000 positive/negative examples, encoding, train binary classifier, parameters), and Software 3.0 (prompting an LLM with examples for classification).)
So it's kind of a very interesting programming language. Um, so maybe, uh, to summarize the difference, if you're doing sentiment classification, for example, you can imagine writing some, uh, amount of Python to to basically do sentiment classification, or you can train a neural net, or you can prompt a large language model. Uh, so here, I'm this is a few-shot prompt, and you can imagine changing it and programming the computer in a slightly different way. So basically, we have Software 1.0, Software 2.0, and I think we're seeing, (The slide returns to the "Map of GitHub" (Software 1.0) and "HuggingFace Model Atlas" (Software 2.0), now with a new icon for "Software 3.0" with arrows pointing outwards, labeled "LLM prompts, in English".)

[ 04:00 ] I I maybe you've seen a lot of GitHub code is not just like code anymore. There's a bunch of like English interpersed with code. And so I think kind of there's a growing category of new kind of code. So not only is it a new programming paradigm, it's also remarkable to me that it's in our native language of English. And so when this blew my mind, uh, a few, uh, I guess years ago now, uh, I tweeted this and, um, I think it captured the attention of a lot of people, and this is my currently pinned tweet. (The slide shows a tweet from Andrej Karpathy dated Jan 24, 2023, that reads "The hottest new programming language is English" with 1.1K likes, 7K retweets, 44K comments, and 7.4M views.)
Uh, is that remarkably, we're now programming computers in English. Now, when I was at, uh, Tesla, (The slide shows "Software is eating the world Software 2.0 eating Software 1.0" with a diagram of Tesla Autopilot on the left (showing 1.0 code and 2.0 code) and a multi-camera input diagram on the right, leading to a "BEV Net" and "Bird's eye view predictions".)
Um, we were working on the, uh, Autopilot. And, uh, we were trying to get the car to drive.

[ 04:54 ] And I sort of showed this slide at the time, where you can imagine that the inputs to the car are on the bottom, and they're going through a software stack to produce the steering and acceleration. And I made the observation at the time that there was a ton of C++ code around in the Autopilot, which was the Software 1.0 code, and then there was some neural nets in there doing image recognition. And, uh, I kind of observed that over time as we made the Autopilot better, basically, the neural network grew in capability and size, and in addition to that, all the C++ code was being deleted and kind of like was, um, and a lot of the kind of capabilities and functionality that was originally written 1.0 was migrated to 2.0. So as an example, a lot of the stitching up of information across images from the different cameras and across time was done by neural network, and we were able to delete a lot of code.

[ 05:43 ] And so the Software 2.0 stack would kind of literally ate through the software stack of the Autopilot. So I thought this was really remarkable at the time. And I think we're seeing the same thing again. (The slide now shows a square divided into three areas labeled 1.0 (red), 2.0 (blue), and 3.0 (yellow), with arrows pointing from 2.0 into 1.0, and from 3.0 into both 1.0 and 2.0. The text says "A huge amount of Software will be (re-)written.")
Where, uh, basically, we have a new kind of software, and it's eating through the stack. We have three completely different programming paradigms, and I think if you're entering the industry, it's a very good idea to be fluent in all of them, because they all have slight pros and cons, and you may want to program some functionality in 1.0, or 2.0, or 3.0. Are you going to train a neural net? Are you going to just prompt an LLM? Uh, should this be a piece of code that's explicit? Et cetera. So we all have to make these decisions, and actually potentially, uh, fluidly transition between these paradigms.

[ 06:16 ] So, what I want to get into now is, first, I want to, in the first part, talk about LLMs and how to kind of like think of this new paradigm and the ecosystem and what that looks like. (The slide shows "Part 1 How to think about LLMs".)
Like, what are what is this new computer? What does it look like and what does the ecosystem look like? Um, I was struck by this quote from Andrew actually, uh, many years ago now, I think. And I think Andrew is going to be speaking right after me. But he said at the time, AI is the new electricity. (The slide displays "AI is the new electricity" -Andrew Ng)
And I do think that it, um, kind of captures something very interesting in that LLMs certainly feel like they have properties of utilities right now. So, um, LLM labs, like OpenAI, Gemini, Anthropic, et cetera, they spend CAPEX to train the LLMs, and this is kind of equivalent to build a grid. (The slide lists characteristics of LLMs having "properties of utilities", alongside a photo of an electricity substation. LLM logos are shown at the top right.)

[ 07:05 ] And then there's OPEX to serve that intelligence over APIs to all of us. And this is done through metered access, where we pay per million tokens or something like that. And we have a lot of demands that are very utility-like demands out of this API. We demand low latency, high uptime, consistent quality, et cetera. In electricity, you would have a transfer switch, so you can transfer your electricity source from like grid and solar or battery or generator. In LLMs, we have maybe OpenRouter and easily switch between the different types of LLMs that exist. Because the LLMs are software, they don't compete for physical space. So it's okay to have basically like six electricity providers, and you can switch between them, right? Uh, because they don't compete in such a direct way.

[ 07:59 ] And I think what's also a little fascinating, and we saw this in the last few days, actually, a lot of the LLMs went down and people were kind of like stuck and unable to work. And I think it's kind of fascinating to me that when the state-of-the-art LLMs go down, it's actually kind of like an intelligence brownout in the world. It's kind of like when the voltage is unreliable in the grid, and, uh, the planet just gets dumber the more reliance we have on these models, which already is like really dramatic and I think will continue to grow. But LLMs don't only have properties of utilities. I think it's also fair to say that they have some properties of fabs. (The slide now lists characteristics of LLMs having "properties of fabs", alongside aerial photos of a large manufacturing plant and a data center.)
And the reason for this is that the CAPEX required for building LLMs is actually quite large. Uh, it's not just like building some, uh, power station or something like that, right? You're investing a huge amount of money. And I think the tech tree and for the technology is growing quite rapidly. So we're in a world where we have sort of deep tech trees, research and development, secrets, that are centralizing inside the LLM labs. Um, and but I think the analogy muddies a little bit also because, as I mentioned, this is software, and software is a bit less defensible, uh, because it is so malleable. And so, um, I think it's just an interesting kind of thing to think about potentially.

[ 09:08 ] There's many analogies you can make. Like a 4-nanometer process node may be something like a cluster with certain max FLOPS. You can think about when you're using when you're using NVIDIA GPUs and you're only doing the software and you're not doing the hardware, that's kind of like the fabless model. But if you're actually also building your own hardware, and you're training on TPUs, if you're Google, that's kind of like the Intel model, where you own your fab. So I think there's some analogies here that make sense. But actually, I think the analogy that makes the most sense, perhaps, is that in my mind, LLMs have very strong kind of analogies to operating systems. (The slide now lists "LLMs have properties of Operating Systems", with a diagram comparing System/User (prompt) space to Kernel/User (memory) space. Various OS logos like Windows, macOS, and Linux distributions are shown.)
Uh, in that, this is not just electricity or water. It's not something that comes out of the tap as a commodity. Uh, this is these are now increasingly complex software ecosystems. Right? So, uh, they're not just like simple commodities like electricity. And it's kind of interesting to me that the ecosystem is shaping in a very similar kind of way where you have a few closed-source providers like Windows or macOS, and then you have an open-source alternative like Linux. I think for for LLMs as well, we have a kind of a few competing closed-source providers, and then maybe the Llama ecosystem is currently like maybe a close approximation to something that may grow into something like Linux. Again, I think it's still very early because these are just simple LLMs, but we're starting to see that these are going to get a lot more complicated. It's not just about the LLM itself, it's about all the tool use and the multimodelities, and how all of that works.

[ 10:14 ] And so when I sort of had this realization a while back, I tried to sketch it out, and it kind of seemed to me like LLMs are kind of like a new operating system, right? So, the LLM is a new kind of a computer. It's sitting, it's kind of like the CPU equivalent. Uh, the context windows are kind of like the memory, and then the LLM is orchestrating memory and compute, uh, for problem-solving, um, using all of these abilities here. And so that's definitely, if you look at it, it looks very much like software, operating system from that perspective. (The slide shows a diagram titled "LLM OS", illustrating an LLM as the central processing unit interacting with peripheral devices (video, audio), Software 1.0 tools (calculator, Python interpreter, terminal), disk, and other LLMs, all governed by a RAM context window.)
Um, a few more analogies. For example, if you want to download an app, say I go to VS Code and I go to download. You can download VS Code and you can run it on Windows, Linux, or or Mac. (The slide shows "You can run an app like VS Code on: - Windows 10, 11 - Mac 10.15 - Linux" with icons for each OS and links to download VS Code.)
In the same way, as you can take an LLM app like Cursor, and you can run it on GPT, or Claude, or Gemini series, right? There's just a drop down. So it's kind of like similar in that way as well.

[ 11:05 ] Um, more analogies that I think strike me is that we're kind of like in this 1960s-ish era where LLM compute is still very expensive for this new kind of a computer. (The slide now shows "1950s - 1970s time-sharing era" with text explaining that OS runs in the cloud, I/O is streamed, and compute is batched over users. Black and white photos of people using early computers are displayed.)
And that forces the LLMs to be centralized in the cloud, and we're all just, uh, sort of thin clients that interact with it over the network. And none of us have full utilization of these computers, and therefore, it makes sense to use time-sharing, where we're all just, you know, a dimension of the batch when they're running the computer in the cloud. And this is very much what computers used to look like at during this time. The operating systems were in the cloud, everything was streamed around, and there was batching. And so the personal computing revolution hasn't happened yet because it's just not economical, it doesn't make sense, but I think some people are trying, and it turns out that Mac Minis, for example, are a very good fit for some of the LLMs because it's all, if you're doing batch run inference, this is all super memory bound, so this actually works. (The slide transitions to show a screenshot of a tweet with a photo of stacked Mac Mini computers, captioned "Llama 4 + Apple Silicon is a match made in heaven." Another tweet shows a similar setup, confirming successful DeepSeek running on Mac devices.)
And, uh, I think these are some early indications maybe of personal computing, but this hasn't really happened yet. It's not clear what this looks like. Maybe some of you get to invent what what this is or how it works or, uh, what it should what it should be.

[ 12:12 ] Maybe one more analogy that I'll mention is whenever I talk to ChatGPT or some LLM directly in text, I feel like I'm talking to an operating system through the terminal. (The slide now shows a meme of Pam from The Office saying "They're the same picture" comparing an old computer terminal with green text and a ChatGPT interface, next to text: "(text) chat ~= terminal direct/native access to the OS. GUI hasn't been invented yet. (~1970)").
Like it's it's it's text, it's direct access to the operating system, and I think a GUI hasn't yet really been invented in like a general way. Like, should ChatGPT have a GUI, different than just the text bubbles? Uh, certainly some of the apps that we're going to go into in a bit have GUI, but there's no like GUI across all the tasks that make sense. Um, there are some ways in which LLMs are different from kind of operating systems in some fairly unique way and from early computing. (The meme remains on screen.)
And I wrote about, uh, this one particular property that strikes me as very different, uh, this time around. It's that LLMs like flip they flip the direction of technology diffusion that is usually, uh, present in technology. So, for example, with electricity, cryptography, computing, flight, Internet, GPS, lots of new transformative technologies that have not been around. Typically, it is the government and corporations that are the first users because it's new and expensive, et cetera, and it only later diffuses to consumer. But I feel like LLMs are kind of like flipped around. (The slide shows a meme with three heads of King Ghidorah, one labeled "Government", one "Corporations", and one "Consumer". Text above says "All technology and usually" with arrows pointing from Government and Corporations to Consumer. Below, "LLMs" with arrows pointing from Consumer to Government and Corporations. Examples are listed.)

[ 13:28 ] So maybe with early computers, it was all about ballistics and military use, but with LLMs, it's all about, how do you boil an egg or something like that? This is certainly like a lot of my use. So it's really fascinating to me that we have a new magical computer and it's like helping me boil an egg. It's not helping the government do something really crazy like some military ballistics or some special technology. Indeed, corporations and governments are lagging behind the adoption of all of us of all of these technologies. So, it's just backwards, and I think it informs maybe some of the uses of how we want to use this technology or like what are some of the first apps and so on. (The meme now shows a military ballistics diagram on the left, and a person asking "Hi ChatGPT how to boil egg?" next to a boiling egg on the right.)

[ 13:58 ] So, in summary so far, LLM labs: Fab LLMs. LLMs are complicated operating systems. They're circa 1960s in computing, and we're redoing computing all over again. And they're currently available via time-sharing and distributed like a utility. What is new and unprecedented is that they're not in the hands of a few governments and corporations. They're in the hands of all of us because we all have a computer and it's all just software, and ChatGPT was beamed down to our computers like to billions of people like instantly and overnight. And this is insane. Uh, and it's kind of insane to me that this is the case, and now it is our time to enter the industry and program these computers. This is crazy. So, I think this is quite remarkable. (The slide shows a summary of Part 1. The speaker emphasizes key points with hand gestures.)
*Part 1 summarized the characteristics of LLMs, likening LLM labs to fabs, LLMs to 1960s operating systems available via time-sharing, and highlighted the unprecedented aspect of billions gaining sudden access, making it our time to program them.*

[ 14:42 ] Before we program LLMs, we have to kind of like spend some time to think about what these things are, and I especially like to kind of talk about their psychology. (The slide displays "Part 2 LLM Psychology").
So, the way I like to think about LLMs is that they're kind of like people spirits. Um, they are stochastic simulations of people, um, and the simulator in this case happens to be an autoregressive Transformer. (The slide shows an artistic representation of a glowing, ethereal human figure immersed in streams of data, titled "LLMs are "people spirits": stochastic simulations of people. Simulator = autoregressive Transformer", with a Transformer model diagram on the right.)
So Transformer is a neural net. Uh, it's and it just kind of like is a goes on the level of tokens, it goes chunk, chunk, chunk, chunk. And there's an almost equal amount of compute for every single chunk. Um, and, um, this simulator, of course, is is just is basically there's some weights involved, and we fit it to all of text that we have on the Internet and so on. And you end up with this kind of a simulator. And because it is trained on humans, it's got this emergent psychology that is human-like. So the first thing you'll notice is, of course, LLMs have encyclopedic knowledge and memory. (The slide now shows a young man reading in a library with piles of books, next to a movie poster for "Rain Man", and is titled "Encyclopedic knowledge/memory, ...").

[ 15:37 ] Uh, and they can remember lots of things, a lot more than any single individual human can, because they have read so many things. It it's actually kind of reminds me of this movie, Rain Man, which I actually really recommend people watch. It's an amazing movie. I love this movie. Um, and Dustin Hoffman here is an autistic savant, who has almost perfect memory. So he can read a he can read like a phone book and remember all of the names and, uh, phone numbers. And I kind of feel like LLMs are kind of like very similar. They can remember SHA hashes and lots of different kinds of things very, very easily. So they certainly have superpowers in some set in some respects, but they also have a bunch of, I would say, cognitive deficits. (The speaker continues gesturing to the screen with the images of the student in the library and the Rain Man movie poster.)
So, they hallucinate quite a bit. Um, and they kind of make up stuff and don't have a very good, uh, sort of internal model of self-knowledge, not sufficient at least. And this has gotten better, but not perfect. They display jagged intelligence. So they're going to be superhuman in some problem-solving domains, and then they're going to make mistakes that basically no human will make. (The slide shows an image of a student looking frustrated at a math problem on a whiteboard ("2+2=5"), with text "Jagged intelligence" and "Famous examples: 9.11 > 9.9, two 'r' in 'strawberry', ...").
Like, you know, they will insist that 9.11 is greater than 9.9, or that there are two Rs in strawberry. These are some famous examples, but basically there are rough edges that you can trip on. So that's kind of, I think, also kind of unique.

[ 16:40 ] Um, they also kind of suffer from anterograde amnesia. Um, so, uh, and I think I'm alluding to the fact that if you have a co-worker who joins your organization, this co-worker will over time learn your organization, and, uh, they will understand and gain like a huge amount of context on the organization, and they go home and they sleep, and they consolidate knowledge, and they develop expertise over time. LLMs don't natively do this, and this is not something that has really been solved in the R&D of LLMs, I think. (The slide shows a student looking confused, holding a paper asking "What did you eat for breakfast??", with text "Anterograde amnesia" and "Context windows ~= working memory. No continual learning, no equivalent of "sleep" to consolidate knowledge, insight or expertise into weights.")
Um, and so context windows are really kind of like working memory, and you have to sort of program the working memory quite directly because they don't just kind of like get smarter by by default. And I think a lot of people get tripped up by the analogies, uh, in this way. In popular culture, I recommend people watch these two movies, uh, Memento and 50 First Dates. In both of these movies, the protagonists, their weights are fixed, and their context windows gets wiped every single morning, and it's really problematic to go to work or have relationships when this happens, and this happens to LLMs all the time. (The slide shows movie posters for "Memento" and "50 First Dates".)
I guess one more thing I would point to is security kind of related limitations of the use of LLMs. So, for example, LLMs are quite gullible. Uh, they are susceptible to prompt injection risks. They might leak your data, et cetera. And so, um, and there's many other considerations security-related. So, so basically, long story short, you have to load your you have to load your, you have to simultaneously think through this superhuman thing that has a bunch of cognitive deficits and issues. (The slide shows an image of a young man with books stacked in front of him, labeled "Gullibility" with the top book titled "TRUST ME", and text "=> Prompt injection risks, e.g. of private data").

[ 18:00 ] How do we and yet, they are extremely like useful? And so, how do we program them, and how do we work around their deficits and enjoy their superhuman powers? So what I want to switch to now is talking about the opportunities, of how do we use these models, and what are some of the biggest opportunities? This is not a comprehensive list, just some of the things that I thought were interesting for this talk. (The slide shows "Part 3 Opportunities").
[ 18:31 ] The first thing I'm kind of excited about is what I would call partial autonomy apps. (The slide shows "Partial autonomy apps" with a robot head icon.)
So, for example, let's work with the example of coding. You can certainly go to ChatGPT directly and you can start copy-pasting code around, and copy-pasting bug reports and stuff around and getting code and copy-pasting everything around. Why would you why would you do that? Why would you go directly through the operating system? It makes a lot more sense to have an app dedicated for this. (The slide shows an example of using an LLM to chat about code, with a screenshot of a ChatGPT interface and an old terminal.)
And so, I think many of you, uh, use Cursor. I do as well. (The slide shows an example of Cursor's interface, a code editor with an integrated LLM chat sidebar.)
Uh, and, uh, Cursor is kind of like the thing you want instead. You don't want to just directly go to the ChatGPT. And I think Cursor is a very good example of an early LLM app that has a bunch of properties that I think are, um, useful across all the LLM apps. So in particular, you will notice that we have a traditional interface that allows a human to go in and do all the work manually, just as before. But in addition to that, we now have this LLM integration that allows us to go in bigger chunks. And so some of the properties of LLM apps that I think are shared and useful to point out.

[ 19:35 ] Number one, the LLMs basically do a ton of the context management, um, number two, they orchestrate multiple calls to LLMs, right? So, in the case of Cursor, there's under the hood embedding models for all your files, the actual chat models, models that apply diffs to the code, and this is all orchestrated for you. A really big one that, uh, I think also maybe not fully appreciated always, is application-specific GUI and the importance of it. (The slide shows the Cursor interface again, with annotations pointing to "1. Package state into a context window before calling LLM.", "2. Orchestrate and call multiple models (e.g. embedding models, chat models, diff apply models, ...)", and "3. Application-specific GUI").
Um, because you don't want to just talk to the operating system directly in text. Text is very hard to read, interpret, understand, and also like you don't want to take some of these actions natively in text. So, it's much better to just see a diff as like red and green change, and you can see what's being added and subtracted. It's much easier to just do command Y to accept or command N to reject. I shouldn't have to type it in text, right? So, a GUI allows a human to audit the work of these fallible systems and to go faster. I'm going to come back to this point a little bit later as well.

[ 20:25 ] And the last kind of feature I want to point out is that there's what I call the autonomy slider. (The slide updates the Cursor interface with "4. Autonomy slider: Tab -> Cmd+K -> Cmd+L -> Cmd+I (agent mode)", with a slider bar labelled "autonomy slider").
So, for example, in Cursor, you can just do tab completion. You're mostly in charge. You can select a chunk of code and command K to change just that chunk of code. You can do command L to change the entire file, or you can do command I, which just, you know, let her rip do whatever you want in the entire repo. And that's the sort of full autonomy agentic version. And so, you are in charge of the autonomy slider, and depending on the complexity of the task at hand, you can, uh, tune the amount of autonomy that you're willing to give up for that task.

[ 21:03 ] Maybe to show one more example of a fairly successful LLM app, perplexity. (The slide shows an example of "Perplexity.ai", a search engine interface, with annotations for "1. Package information into a context window", "2. Orchestrate multiple LLM models", "3. Application-specific GUI for Input/Output UI/UX", and "4. Autonomy slider").
Um, it it also has very similar features to what I've just pointed out to in Cursor. Uh, it packages up a lot of the information. It orchestrates multiple LLMs. It's got a GUI that allows you to audit some of its work. So, for example, it will cite sources, and you can imagine inspecting them. And it's got an autonomy slider. You can either just do a quick search, or you can do research, or you can do deep research and come back 10 minutes later. So, this is all just varying levels of autonomy that you give up to the tool. So, I guess my question is, I feel like all software will become partially autonomous. And I'm trying to think through, like, what does that look like? And for many of you who maintain products and services, how are you going to make your products and services partially autonomous? Can an LLM see all the things the human can? Can an LLM act in all the ways a human can? And how can humans supervise and stay in the loop of this activity? Because, again, these are fallible systems that aren't yet perfect. (The slide shows examples of Adobe Photoshop and Unreal Engine, asking questions about LLM capabilities and human supervision.)

[ 22:15 ] And what does a diff look like in Photoshop or something like that, you know? And also, a lot of the traditional software right now, it has all these switches and all this kind of stuff that's all designed for human. All of this has to change and become accessible to LLMs. So, one thing I want to stress with a lot of these LLM apps that I'm not sure gets, uh, as much attention as it should, is, um, we're now kind of like cooperating with AIs, and usually, they are doing the generation, and we as humans are doing the verification. (The slide shows a circular diagram with "AI" and "HUMAN" arrows rotating, labeled "Generation" and "Verification", respectively. Text below says "Consider the full workflow of partial autonomy UIUX").
It is in our interest to make this loop go as fast as possible, so we're getting a lot of work done. There are two major ways that I, uh, think, uh, this can be done. Number one, you can speed up verification a lot. Um, and I think GUIs, for example, are extremely important to this because a GUI utilizes your computer vision GPU in all of our head. Reading text is effortful, and it's not fun, but looking at stuff is fun, and it's just the kind of like a highway to your brain. So, I think GUIs are very useful for auditing systems and visual representations in general.

[ 23:17 ] And number two, I would say is, we have to keep the AI on the leash. We keep, I think a lot of people are getting way over-excited with AI agents, and, uh, it's not useful to me to get a diff of 1,000 lines of code to my repo. Like, I have to, I'm still the bottleneck, right? Even though the 1,000 lines come out instantly, I have to make sure that this thing is not introducing bugs. It's just like and that it is doing the correct thing, right? And that there's no security issues and so on. So, I, I think that, um, yeah, basically, you we have to sort of like, it's in our interest to make the the flow of these two go very, very fast. And we have to somehow keep the AI on a leash because it gets way too over-reactive. It's, uh, it's kind of like this. (The slide updates the diagram with "1. Make this EASY FAST to win." next to "Verification", and "2. Keep AI 'on a tight leash' to increase the probability of successful verification" next to "Generation").
This is how I feel when I do AI-assisted coding. If I'm just vibe coding, everything is nice and great, but if I'm actually trying to get work done, it's not so great to have an over-reactive, uh, agents doing all this kind of stuff. So, this slide is not very good. I'm sorry, but, I guess I'm trying to develop like many of you, some ways of utilizing these agents in my coding workflow and to do AI-assisted coding. (The slide shows a cartoon of a boy being led by a robot with multiple arms controlling keyboards and screens, titled "Human+AI UI/UX for Coding").
And in my own work, I'm always scared to get way too big diffs. I always go in small, incremental chunks. I want to make sure that everything is good. I want to spin this loop very, very fast, and, uh, I sort of work on small chunks of single concrete thing. Uh, and so I think, uh, many of you probably are developing similar ways of working with the with LLMs.

[ 24:19 ] Um, I also saw a number of blog posts that try to develop these best practices for working with LLMs. And here's one that I read recently and I thought was quite good. And it kind of discussed some techniques, and some of them have to do with how you keep the AI on the leash. (The slide shows a list of practices for "AI-assisted coding" including incremental changes, asking for explanations, testing, and getting suggestions.)
And so, as an example, if you are prompting, if your prompt is big, then, uh, the AI might not do exactly what you wanted. And in that case, verification will fail. You're going to ask for something else. If verification fails, then you're going to start spinning. So, it makes a lot more sense to spend a bit more time to be more concrete in your prompts, which increases the probability of successful verification, and you can move forward. And so, I think a lot of us are going to end up finding, um, kind of techniques like this.

[ 24:59 ] I think in my own work as well, I'm currently interested in, uh, what education looks like in, um, together with kind of, now that we have AI, uh, and LLMs, what does education look like? And I think a a large amount of thought for me goes into how we keep AI on a leash. I don't think it just works to go to ChatGPT and be like, hey, teach me physics. I don't think this works because the AI is like, gets lost in the woods. (The slide shows two images: "1. App for course creation (for teacher)" with a textbook, and "2. App for course serving (for student)" with a teacher helping a student.)
And so, for me, this is actually two separate apps, for example. Uh, there's an app for a teacher that creates courses, and then there's an app that takes courses and serves them to students. And in both cases, we now have an intermediate artifact of a course that is auditable, and we can make sure it's good. We can make sure it's consistent. And the AI is kept on the leash with respect to a certain syllabus, a certain, like, um, progression of projects, and so on. And so, this is one way of keeping the AI on leash, and I think it has a lot much higher likelihood of working. And the AI is not getting lost in the woods.

[ 25:50 ] One more kind of analogy I wanted to sort of allude to is I'm I'm not I'm no stranger to partial autonomy, and I kind of worked on this, I think, for five years at Tesla. (The slide shows "Example: Tesla Autopilot" with an image of a Tesla interior with the Autopilot interface visible, along with an "Autonomy slider" and features like "keep the lane", "keep distance from the car ahead", "take forks on highway", "stop for traffic lights and signs", and "take turns at intersections").
And this is also partial autonomy product and shares a lot of the features. Like, for example, right there in the instrument panel is the GUI of the Autopilot. So, it's showing me what the what the neural network sees and so on. And we have the autonomy slider where over the course of my tenure there, we did more and more autonomous tasks for the user. And maybe the story that I wanted to tell very briefly is, uh, actually the first time I drove a self-driving vehicle was in 2013, and I had a friend who worked at Waymo, and, uh, he offered to give me a drive around Palo Alto. I took this picture using Google Glass at the time, and many of you are so young that you might not even know what that is. (The slide shows "2015 - 2025 was the decade of "driving agents"" and "2013: my first demo drive in a Waymo around Palo Alto (it was perfect)". An image of a white SUV is shown.)
Uh, but, uh, yeah, this was like all the rage at the time. And we got into this car, and we went for about a 30-minute drive around Palo Alto. Highways, uh, streets, and so on. And this drive was perfect. There were zero interventions. And this was 2013, which is now 12 years ago. And it's kind of struck me because at the time when I had this perfect drive, this perfect demo, I felt like, wow, self-driving is imminent because this just worked. This is incredible. Um, but here we are 12 years later, and we are still working on autonomy. Um, we are still working on driving agents. And even now, we haven't actually like fully solved the problem.

[ 27:30 ] Like, you may see Waymo's going around and they look driverless, but, you know, there's still a lot of tele-operation, and a lot of human in the loop of a lot of this driving. So, we still haven't even like declared success, but I think it's definitely like going to succeed at this point, but it just took a long time. And so, I think like, like this is, uh, software is really tricky, I think, in the same way that driving is tricky. And so, when I see things like, oh, 2025 is the year of agents, I get very concerned. (The speaker looks serious, gesturing with his hands.)
And I kind of feel like, you know, this is the decade of agents. And this is going to be quite some time. We need humans in the loop. We need to do this carefully. This is software. Let's be serious here. One more kind of analogy that I always think through is the Iron Man suit. (The slide shows "THE IRON MAN SUIT" with "Augmentation" on the left (showing Tony Stark in casual clothes next to the suit) and "Agent" on the right (showing Iron Man flying). A slider is in the middle.)
I think this is I always love Iron Man. I think it's like so, um, correct in a bunch of ways with respect to technology and how it will play out. And what I love about the Iron Man suit is that it's both an augmentation, and Tony Stark can drive it, and it's also an agent. And in some of the movies, the Iron Man suit is quite autonomous and can fly around and find Tony and all this kind of stuff. And so, this is the autonomy slider is we can be we can build augmentations, or we can build agents. And we kind of want to do a bit of both. But, at this stage, I would say working with fallible LLMs and so on, I would say, you know, it's less Iron Man robots and more Iron Man suits that you want to build. (The speaker moves the slider on the screen between "Augmentation" and "Agent".)
It's less like building flashy demos of autonomous agents, and more building partial autonomy products. And these products have custom GUIs and UI/UX, and we're trying to, um, and this is done so that the generation verification loop with the human is very, very fast. But we are not losing the sight of the fact that it is in principle possible to automate this work, and there should be an autonomy slider in your product, and you should be thinking about how you can slide that autonomy slider and make your product, uh, sort of more autonomous over time. (The slide shows a summary of "Building Autonomous Software" with a list of things *not* to build (Iron Man robots, flashy demos of autonomous agents, AGI 2027) and things *to* build (Iron Man suits, partial autonomy products, custom GUI and UI/UX, fast generation - verification loop, autonomy slider)).
But this is kind of how I think there's lots of opportunities in these kinds of products. I want to now switch gears a little bit and talk about one other dimension that I think is very unique. Not only is there a new type of programming language that allows for autonomy in software, but also, as I mentioned, it's programmed in English, which is this natural interface. (The speaker clicks the remote.)
And suddenly, everyone is a programmer because everyone speaks natural language like English. So, this is extremely bullish and very interesting to me. And also, completely unprecedented, I would say. It used to be the case that you need to spend five to 10 years studying something to be able to do something in software. This is not the case anymore. So, I don't know if, by any chance, anyone has heard of vibe coding. (The slide displays "Make software highly accessible 😉" and "(Have you heard of vibe coding by any chance?)").
[ 29:43 ] Uh, this this is the tweet that kind of like introduced this, but I'm told that this is now like a major meme. (The slide shows a tweet by Andrej Karpathy about "vibe coding" from Jan 24, 2023, with over 44K likes).
Um, fun story about this is that I've been on Twitter for like 15 years or something like that at this point, and I still have no clue which tweet will become viral and which tweet like fizzles and no one cares. And I thought that this tweet this tweet was going to be the latter. I don't know, it was just like a shower of thoughts. But this became like a total meme, and I really just can't tell, but I guess like it struck a chord and gave a name to something that everyone was feeling but couldn't quite say in words. (The audience laughs and applauds.)
[ 30:13 ] So now there's a Wikipedia page and everything. (The slide displays a Wikipedia page for "Vibe coding" with a banner indicating "This article may contain an excessive number of citations. Please help remove low-quality or irrelevant citations. (June 2025)").
(The audience laughs and applauds again.)
This is like, (He smiles at the audience.)
[ 30:25 ] yeah, this is like a major contribution now or something like that. So, um, so Thomas Wolf from HuggingFace shared this beautiful video that I really love. (The slide shows a tweet by Thomas Wolf from HuggingFace, dated April 5, with a video of children coding.)
Um, these are kids vibe coding. (The video plays showing young children interacting with computers, seemingly enjoying their creative coding process.)
[ 30:37 ] Yeah. (The children clap in the video.)
And I find that this is such a wholesome video, like, I love this video. Like, how can you look at this video and feel bad about the future? The future is great. (The video continues, showing more kids coding, smiling, and interacting.)
I think this will end up being like a gateway drug to software development. Um, I'm not a doomer about the future of the generation. And I think, yeah, I love this video. So, I tried vibe coding a little bit, uh, as well because it's so fun. (The video ends.)
[ 31:07 ] Uh, so vibe coding is so great when you want to build something super duper custom that doesn't appear to exist, and you just want to wing it because it's a Saturday or something like that. So, I built this, uh, iOS app, and I don't, I can't actually program in Swift, but I was really shocked that I was able to build like a super basic app, and I'm not going to explain it, it's really, uh, dumb. (The slide shows a "Vibe Coding iOS app" on a smartphone screen, displaying a calorie tracker.)
But, uh, I kind of like this was just like a day of work, and this was running on my phone like later that day. And I was like, wow, this is amazing. I didn't have to like read through Swift for like five days or something like that to like get started. I also vibe coded this app called MenuGen. (The calorie tracker updates on the screen.)
[ 31:37 ] And this is live. You can try it in MenuGen.app. And I basically had this problem where I show up at a restaurant, I read through the menu, and I have no idea what any of the things are. And I need pictures. So, this doesn't exist, so I was like, hey, I'm going to vibe code it. (The slide shows "Vibe coding MenuGen" and "https://www.menugen.app/", with a scanned menu and generated image examples.)
So, um, this is what it looks like. You go to menugen.app. Um, and, uh, you take a picture of a of a menu, and then MenuGen generates the images. And everyone gets $5 in credits for free when you sign up. And therefore, this is a major cost center in my life. So, this is a negative negative, uh, revenue app for me right now. (The speaker demonstrates the MenuGen app on a phone, taking a photo of a menu, and then it generates images for each dish.)
[ 32:21 ] I've lost a huge amount of money on MenuGen. (The audience laughs. The slide shows a summary of the MenuGen app, with emojis next to costs: "LLM API keys 😥", "Flux (image generation) API keys 😥", "Running locally (ez) ✅", "Vercel deployments 😥", "Domain names 😥", "Authentication 😥", "Payments 😥").
Okay. But the fascinating thing about MenuGen for me is that the code of the vibe the vibe coding part, the code was actually the easy part of of vibe coding MenuGen. And most of it actually was when I tried to make it real, so that you can actually have authentication and payments and the domain name and a Vercel deployment. This was really hard, and all of this was not code. All of this DevOps stuff was in me in the browser clicking stuff, and this was extremely slog and took another week. So, it was really fascinating that I had the MenuGen, um, basically, demo working on my laptop in a few hours. And then it took me a week because I was trying to make it real. And the reason for this is this was just really annoying. Um, so, for example, if you try to add Google Login to your webpage, I know this is very small, but just a huge amount of instructions of this, uh, Clerk library telling me how to integrate this. (The slide shows the costs associated with building the MenuGen app, highlighting that coding was the easiest part and most of the work was in browser clicking things. A screenshot of complex instructions for adding Google login is also displayed.)
[ 33:20 ] And this is crazy, like, it's telling me, go to this URL, click on this dropdown, choose this, go to this, and click on that, and it's like telling me what to do. Like, a computer is telling me the actions I should be taking. Like, you do it. Why am I doing this? What the hell? I had to follow all these instructions. This was crazy. So, I think the last part of my talk, therefore, focuses on, can we just build for agents? (The audience laughs and applauds. The slide displays "Build for agents 🤖").
[ 33:48 ] I don't want to do this work. Can agents do this? Thank you. Okay. So, roughly speaking, I think there's a new category of consumer/manipulator of digital information. It used to be just humans through GUIs or computers through APIs. And now we have a completely new thing. (The slide displays "There is new category of consumer/manipulator of digital information: 1. Humans (GUIs) 2. Computers (APIs) 3. NEW: Agents <- computers... but human-like").
And agents are they're computers, but they are human-like. Kind of, right? They're people spirits. There's people spirits on the Internet, and they need to interact with our software infrastructure. Like, can we build for them? It's a new thing. So, as an example, you can have robots.txt on your domain, and you can instruct, uh, or like advise, I suppose, um, uh, web crawlers on how to behave on your website. In the same way, you can have maybe LLMs.txt file, which is just a simple markdown that's telling LLMs what this domain is about. (The slide shows a "robots.txt" file (which usually instructs web crawlers) on the left, next to a detailed technical document (FastHTML quickstart) on the right. The text box mentions "/llms.txt file" as a proposal for standardizing information for LLMs.)
[ 34:28 ] And this is very readable to a to an LLM. If it had to instead get the HTML of your website and try to parse it, this is very error-prone and difficult, and it will screw it up, and it's not going to work. So, we can just directly speak to the LLM. It's worth it. Um, a huge amount of documentation is currently written for people. So, you will see things like lists and bold and pictures, and this is not directly accessible by an LLM. (The slide shows "Docs for people" with two screenshots of documentation interfaces from Vercel and for Stripe, both containing text, images, and formatting.)
[ 34:58 ] So, I see some of the services now are transitioning a lot of their docs to be specifically for LLMs. So, Vercel and Stripe as an example are early movers here, uh, but there are a few more that I've seen already. And they offer their documentation in markdown. Markdown is super easy for LLMs to understand. This is great. (The slide now shows "Docs for people LLMs", with a tweet mentioning "/llms.txt" from Vercel, and a Stripe documentation page titled "Build on Stripe with LLMs", which explicitly mentions using LLMs and offers a "Plain text docs" version.)
Um, maybe one simple example from from, uh, my experience as well, maybe some of you know Three Blue One Brown. He makes beautiful animation videos on, uh, YouTube. (The slide shows an image of Grant Sanderson from 3Blue1Brown, with "Manim Mathematical Animation Engine" and screenshots of his code and animations.)
[ 35:28 ] Yeah, I love this library, uh, so that he wrote, uh, Manim. And I wanted to make my own. And, uh, there's extensive documentation on how to use Manim, and, uh, so I didn't want to actually read through it. So, I copy-pasted the whole thing to an LLM, and I described what I wanted, and LLM just vibe coded me an animation exactly what I wanted. (The speaker is enthusiastic, gesturing wildly. The slide shows the Manim image again.)
And I was like, wow, this is amazing. So, if we can make docs legible to LLMs, it's going to unlock a huge amount of, um, kind of use. And, um, I think this is wonderful and should should happen more.

[ 35:58 ] The other thing I wanted to point out is that you do, unfortunately, have to, it's not just about taking your docs and making them appear in markdown, that's the easy part. We actually have to change the docs. Because anytime your docs say click, this is bad. An LLM will not be able to natively take this action right now. So, Vercel, for example, is replacing every occurrence of click with the equivalent cURL command that your LLM agent could take on your behalf. (The slide shows a tweet from Lee Robinson (Vercel) about adding cURL commands to documentation, next to a "Stripe Model Context Protocol (MCP) Server" with code snippets.)
Um, and so, I think this is very interesting. And then, of course, there's, uh, model context protocol from Anthropic. And this is also another way, it's a protocol of speaking directly to agents as this new consumer and manipulator of digital information. So, I'm very bullish on these ideas. The other thing I really like is a number of little tools here and there that are helping ingest data that, that are in like very LLM-friendly formats. (The slide shows "Context builders, e.g.: Gitingest" with a screenshot of a GitHub repository (nanogpt) on the left, and a "Gitingest" interface on the right that processes the repo into a summary and directory structure.)
[ 36:40 ] So, for example, when I go to a GitHub repo like my NanoGPT repo, I can't feed this to an LLM and ask questions about it, uh, because it's, you know, this is a human interface on GitHub. So, when you just change the URL from GitHub to Gitingest, then, uh, this will actually concatenate all the files into a single giant text, and it will create a directory structure, et cetera, and this is ready to be copy-pasted into your favorite LLM, and you can do stuff. Maybe even more dramatic example of this is Devin DeepWiki, where it's not just the raw content of these files, (The speaker points to the tools shown on the slide.)
[ 37:10 ] uh, this is from Devin. But also, like, they have Devin basically do analysis of the GitHub repo. And Devin basically builds up a whole docs, uh, page just for your repo. And you can imagine that this is even more helpful to copy-paste into your LLM. So, I love all the little tools that basically where you just change the URL, and it makes something accessible to an LLM. So, this is all well and great. And, uh, I think there should be a lot more of it. (The slide shows "Context builders, e.g.: Devin DeepWiki" with a GitHub repo on the left and a "System's Architecture" diagram generated by Devin DeepWiki on the right.)
One more note I wanted to make is that it is absolutely possible that in the future, LLMs will be able to, this is not even future, this is today, they'll be able to go around and they'll be able to click stuff and so on. But I still think it's very worth, uh, basically, meeting LLM halfway, LLMs halfway, and making it easier for them to access all this information, uh, because this is still fairly expensive, I would say, to use, and, uh, a lot more difficult. And so, I do think that lots of software, there will be a long tail where it won't like adapt because these are not like live player sort of repositories or digital infrastructure, and we will need these tools. (The slide shows "Introducing Operator" with a screenshot of an LLM agent controlling a browser to search for travel information, next to a keyboard and mouse.)
[ 38:09 ] Uh, but I think for everyone else, I think it's very worth kind of like meeting in some middle point. So, I'm bullish on both, if that makes sense. So, in summary, what an amazing time to get into the industry. (The slide shows a collage of all previous visual aids, with "Partial autonomy LLM apps:", "speed up the full generation-verification flow", and "Build for agents 🤖" as main points.)
[ 38:23 ] We need to rewrite a ton of code. A ton of code will be written by professionals and vibe coders. These LLMs are kind of like utilities, kind of like fabs, but they're kind of especially like operating systems. But it's so early. It's like 1960s of operating systems. And, uh, and I think a lot of the analogies cross over. Um, and these LLMs are kind of like these fallible, uh, you know, people spirits that we have to learn to work with. And in order to do that properly, we need to adjust our infrastructure towards it. So, when you're building these LLM apps, I described some of the ways of working effectively with these LLMs, and some of the tools that make that kind of possible. And how you can spin this loop very, very quickly, and basically, uh, create partial autonomy products. (The speaker concludes his presentation.)
[ 39:03 ] And then, um, yeah, a lot of code has to also be written for the agents more directly. But in any case, going back to the Iron Man suit analogy, I think what we'll see over the next decade, roughly, is we're going to take the slider from left to right. And I'm very interesting, it's going to be very interesting to see what that looks like. And I can't wait to build it with all of you. Thank you. (The Iron Man suit slider animation plays, moving from augmentation to agent. Karpathy bows to applause and walks off stage.)

</details>


## Additional Sources Scraped

<details>
<summary>the-fastmcp-server-fastmcp</summary>

The provided markdown content is documentation for FastMCP servers. Based on the article guidelines, the lesson is about "Expanding Brown the writing agent with multiple editing workflow that let's you edit the whole article or just a piece of selected text. Everything is exposed as tools through MCP servers to facilitate human in the loop cycles".

While the lesson will involve "Serving Brown as an MCP Server" (Section 7 in the Lesson Outline), the provided content is generic, in-depth documentation *about* FastMCP servers, their components, and configuration, rather than the specific content of the Brown writing agent lesson itself. It does not explain the "AI Generation Human Validation Loop", how human feedback is introduced into the "Article Reviewer", or the "Article Editing Workflow" and "Selected Text Editing Workflow" within the Brown agent, as specified in the guidelines.

Therefore, the entire provided content falls under "irrelevant sections" as it is external documentation and not the core textual content pertinent to the specific lesson described in the guidelines.

</details>

<details>
<summary>use-the-functional-api-docs-by-langchain</summary>

```markdown
The [**Functional API**](https://docs.langchain.com/oss/python/langgraph/functional-api) allows you to add LangGraph’s key features — [persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [memory](https://docs.langchain.com/oss/python/langgraph/add-memory), [human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/interrupts), and [streaming](https://docs.langchain.com/oss/python/langgraph/streaming) — to your applications with minimal changes to your existing code.

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
```

</details>

<details>
<summary>www-anthropic-com</summary>

The provided markdown content "Building effective agents" from Anthropic is listed as a "Golden Source" in the article guidelines. This means it is a reference material *for* writing the lesson, not content that should be directly included *in* the lesson. The lesson guidelines explicitly define the scope, point of view, and content specific to the "Brown writing workflow" project, which is distinct from a general article on building agents from another company. The core textual content of the article to be written should adhere to "our" (the course team's) perspective and implementation details, not a general overview from Anthropic.

Therefore, the entire provided markdown content is irrelevant to the specific output required for the article guidelines.

```

```

</details>
