# Research

## Research Results

<details>
<summary>What are the fundamental differences between Plan-and-Execute and ReAct agent architectures?</summary>

### Source [1]: https://blog.langchain.com/planning-agents/

Query: What are the fundamental differences between Plan-and-Execute and ReAct agent architectures?

Answer: - **Core idea:** Plan-and-Execute separates an LLM-powered **planner** from the **execution runtime**, whereas ReAct interleaves reasoning and actions in a single loop where the larger agent is consulted at every step[2].  
- **Efficiency:** Plan-and-Execute can be **faster** because sub-tasks execute without consulting the larger model after each action; the main model is engaged only for (re)planning and final response[2].  
- **Cost:** It can be **cheaper** by delegating sub-tasks to **smaller, domain-specific models**, reserving the larger model for planning checkpoints and the final answer[2].  
- **Quality:** It can **improve performance** by forcing explicit, full-task planning up front (akin to chain-of-thought), and by decomposing the problem into focused sub-tasks that are executed independently[2].  
- **When beneficial:** Applications requiring **multiple tool invocations or API calls** gain from reduced latency and cost due to fewer calls to powerful models during execution[2].  
- **Contrast with ReAct:** ReAct’s pattern involves continuous reason-act cycles with the main agent orchestrating each tool call, which can increase latency and cost compared to preplanned execution in Plan-and-Execute[2].

-----

-----

### Source [2]: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/

Query: What are the fundamental differences between Plan-and-Execute and ReAct agent architectures?

Answer: - **ReAct definition:** A popular general-purpose agent architecture integrating **tool calling**, **memory**, and **planning** to enable multi-step decision-making with dynamic tool use[4].  
- **Mechanism:** The LLM performs **multi-step decisions** and selects from tools iteratively, maintaining context via memory and adapting plans as it goes—characteristic of ReAct’s intertwined reasoning-acting loop[4].  
- **Modern implementation:** Contemporary agents based on ReAct rely on **LLM tool-calling** and operate over a **message list**, aligning with iterative, step-by-step control flow and execution[4].  
- **Implied difference vs Plan-and-Execute:** Because ReAct couples planning with immediate action selection on each step, it contrasts with Plan-and-Execute’s up-front plan formation and separated execution stage[4].

-----

-----

### Source [3]: https://arxiv.org/html/2404.11584v1

Query: What are the fundamental differences between Plan-and-Execute and ReAct agent architectures?

Answer: - **Architectural theme:** Successful goal execution hinges on **planning** and **self-correction**; agents without effective plans and self-evaluation risk endless loops or subpar results[3].  
- **Implication for differences:** ReAct-style single agents benefit from iterative planning and self-correction during execution, while Plan-and-Execute emphasizes an explicit **planning phase** with opportunities to **refine plans as new information is learned** in structured stages[3].  
- **Use cases:** Single-agent patterns (including ReAct) work well when tasks involve straightforward tool calls and iterative progress, whereas architectures with a **defined planning phase** and refinement steps are advantageous for complex, multi-step goals—aligning with Plan-and-Execute’s separation of planning from execution[3].

-----

-----

### Source [4]: https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9

Query: What are the fundamental differences between Plan-and-Execute and ReAct agent architectures?

Answer: - **High-level comparison:** ReAct uses a **reasoning–action loop**, deciding and acting step-by-step, while Plan-and-Execute uses a **planning–execution separation** where a plan is produced first and then carried out[1].  
- **Implementation focus:** Provides **LangChain-based implementations** showing ReAct’s loop structure and Plan-and-Execute’s planner-executor decomposition, highlighting practical engineering trade-offs[1].  
- **Performance trade-offs:** Presents **response time, accuracy, and token/cost analyses**, indicating scenarios where Plan-and-Execute reduces latency/cost by minimizing large-model calls during execution, versus ReAct’s frequent large-model involvement per step[1].  
- **Selection guidance:** Recommends pattern selection based on **scene characteristics**; tasks needing adaptive, fine-grained step-by-step reasoning may prefer ReAct, while **multi-step workflows** with predictable sub-tasks benefit from Plan-and-Execute’s upfront planning and delegated execution[1].

-----

-----

### Source [5]: https://www.willowtreeapps.com/craft/building-ai-agents-with-plan-and-execute

Query: What are the fundamental differences between Plan-and-Execute and ReAct agent architectures?

Answer: - **Plan-and-Execute pitfalls:** Linear, upfront plans can lack flexibility; handling failures may require explicit mechanisms to **update the plan**, otherwise progress stalls when unexpected issues occur[5].  
- **Dynamic alternative:** Using a single prompt to both plan the next step and execute supports more **dynamic behavior**, highlighting ReAct-like adaptability versus strict plan-first execution[5].  
- **Practical challenges:** As tools grow, prompts bloat; tool management and memory persistence need careful design—issues that can be more pronounced when separating planning and execution without robust failure handling[5].  
- **GUI and tool limits:** Agents relying on fixed plans may **struggle with GUI interactions** and unforeseen tool behaviors, motivating reflection and adaptive next-step selection characteristic of ReAct-style loops[5].

-----

</details>

<details>
<summary>How do modern LLMs like GPT-4o and Claude 3.5 Sonnet handle interleaved reasoning and tool use internally?</summary>

### Source [6]: https://platform.openai.com/docs/guides/function-calling

Query: How do modern LLMs like GPT-4o and Claude 3.5 Sonnet handle interleaved reasoning and tool use internally?

Answer: - OpenAI describes tool use via **function calling**, where the model outputs a JSON-like schema selecting a tool name and arguments; the application executes the tool and returns results to the model before the model continues its response[1].  
- The model’s internal reasoning remains latent, but the interface allows the model to interleave “thinking → tool call → observe result → continue,” because responses can be streamed and interrupted when the model decides to call a tool, then resumed after tool output is fed back in as new messages[1].  
- Developers can supply a list of tools with JSON Schemas; the model decides when to call which tool and with what arguments, effectively planning multi-step workflows by alternating natural-language tokens and tool-call tokens across multiple turns[1].  
- The docs recommend passing tool results back as assistant messages with role “tool,” enabling iterative chains of tool calls, result inspection, and further calls—supporting interleaved reasoning with external capabilities such as retrieval, code execution, or APIs[1].  
- Safety and control suggestions include validating tool arguments, constraining schemas, and supervising multi-step tool loops; this architecture separates model policy from tool execution while keeping the model “in the loop” for next-step decisions[1].

-----

-----

### Source [7]: https://platform.openai.com/docs/guides/structured-outputs

Query: How do modern LLMs like GPT-4o and Claude 3.5 Sonnet handle interleaved reasoning and tool use internally?

Answer: - OpenAI provides **Structured Outputs** that constrain the model to produce JSON conforming to a schema, which can be combined with function calling to ensure reliable arguments and intermediate state during tool use[2].  
- By forcing well-typed structures, the model can more robustly interleave reasoning with tools: plan steps, emit arguments, receive results, then emit subsequent structured actions—reducing parsing errors that would otherwise break multi-step tool chains[2].  
- The guide emphasizes deterministic schemas for multi-turn workflows, enabling the model to alternate between free-form reasoning and schema-constrained action selection across turns[2].

-----

-----

### Source [8]: https://platform.openai.com/docs/guides/reasoning

Query: How do modern LLMs like GPT-4o and Claude 3.5 Sonnet handle interleaved reasoning and tool use internally?

Answer: - OpenAI’s “Reasoning” guide explains that models can engage in multi-step problem solving and **call tools during reasoning** for tasks like code execution or retrieval, with the application returning results that become new context for continued reasoning[3].  
- It highlights that tool feedback is appended to the conversation so the model can revise plans, try additional tools, or produce final answers, demonstrating the interleaved loop: propose → act → observe → update[3].  
- The guide notes that careful prompt design (instructions, scratch space allowances, and tool descriptions) improves the model’s ability to decide when to use tools versus internal reasoning[3].

-----

-----

### Source [9]: https://api.openai.com/docs/guides/vision

Query: How do modern LLMs like GPT-4o and Claude 3.5 Sonnet handle interleaved reasoning and tool use internally?

Answer: - For multimodal models like GPT-4o, OpenAI documents that the model can process images and text jointly and also **invoke tools** within the same session, letting it reason over visual inputs, call tools (e.g., OCR, retrieval), and continue reasoning with returned outputs interleaved in the dialogue[4].  
- The vision guide shows that image content is represented as inputs the model can reference while planning actions; tool outputs (e.g., extracted text) are fed back as messages for subsequent steps, enabling interleaved visual reasoning and tool use[4].

-----

-----

### Source [10]: https://www.anthropic.com/news/3-5-models-and-computer-use

Query: How do modern LLMs like GPT-4o and Claude 3.5 Sonnet handle interleaved reasoning and tool use internally?

Answer: - Anthropic introduces “**computer use**,” which lets Claude 3.5 Sonnet operate a virtual computer: it can browse, click, type, and navigate apps via tools exposed by the API, and can interleave these actions with natural-language reasoning in a continuous loop[5].  
- Claude plans multi-step workflows, executes UI actions, observes screen state returned as tool outputs (e.g., DOM, screenshots, text), and adjusts its plan—implementing an interleaved cycle of think → act → observe → revise[5].  
- Anthropic reports improved performance on agentic tool-use benchmarks (e.g., TAU-bench), indicating the model’s ability to manage multi-step tool interactions with feedback over multiple turns[5].  
- Safety controls (ASL-2) and staged evaluations accompany this capability, with guardrails to constrain available actions and domains while keeping the model in the loop for next-step selection[5].

### : https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Anthropic’s tool use docs describe how Claude selects tools and emits structured arguments; the application executes the tool and returns results as messages that Claude then incorporates into further reasoning.  
- The API supports iterative, multi-turn tool calls: Claude can chain tools, inspect outputs, and decide next actions, enabling interleaved planning and execution without leaving the conversational context.  
- Developers define tools via JSON Schemas; Claude’s responses may be partial to request tool calls mid-stream, after which the tool results are appended and Claude continues—supporting interleaved operation across steps.  
- The docs advise validating arguments, limiting tool scopes, and logging tool traces to maintain safety and debuggability in complex interleaved workflows.

### : https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
- Claude supports **structured outputs** that enforce response schemas, improving reliability of tool arguments, intermediate plans, and final results in multi-step flows.  
- By constraining outputs, Claude can more predictably alternate between free-form reasoning and schema-bound actions, aiding robust interleaved tool use where each step must produce valid, machine-readable content.  
- The documentation provides recommendations for combining structured outputs with tool use to reduce failure modes in long chains of actions.

-----

</details>

<details>
<summary>What are best practices for prompting an AI agent to perform goal decomposition on a complex task?</summary>

### Source [11]: https://www.amazon.science/blog/how-task-decomposition-and-smaller-llms-can-make-ai-more-affordable

Query: What are best practices for prompting an AI agent to perform goal decomposition on a complex task?

Answer: - Define clear, independent subtasks whenever possible. According to Amazon Science, aim to decompose a complex task into **independent** components so each subtask can have a targeted prompt and context; this simplifies troubleshooting and improves reliability by isolating failures to specific steps[4].  
- Use targeted prompts and contexts per subtask. Craft **focused prompts** with only the information needed for that subtask to reduce distraction and improve accuracy; this also reduces context length and cost when orchestrating multiple agents[4].  
- Balance decomposition depth to avoid overengineering. Excessive subdivision increases coordination overhead and can undermine benefits; maintain a **balance between cost, performance, and simplicity** to avoid diminishing returns[4].  
- Ensure coherence when subtasks are not independent. When interdependencies exist, apply **prompt engineering or retrieval** to keep outputs aligned across steps, preserving necessary context while avoiding unnecessary breadth[4].  
- Consider persona- or function-based agent roles. In agentic workflows, use **functional agents** (e.g., database querying) or **persona-based agents** (e.g., UX designer) to structure responsibilities and prompts clearly, improving division of labor and reviewability[4].  
- Prefer multiple smaller, fine-tuned models where appropriate. Task decomposition can enable use of **smaller LLMs** specialized per subtask to improve cost efficiency without sacrificing performance on complex applications[4].  
- Preserve creative/contextual richness where it matters. Avoid fragmenting tasks so much that models lose the **novelty and contextual richness** emergent from broader context; keep holistic prompts where capturing hidden relationships is beneficial[4].

-----

-----

### Source [12]: https://relevanceai.com/prompt-engineering/break-down-your-prompts-for-better-ai-results

Query: What are best practices for prompting an AI agent to perform goal decomposition on a complex task?

Answer: - Use specific, actionable language for each subtask. Clearly state expected outputs, constraints, and success criteria to guide the agent through **sequential, focused steps** for complex goals[2].  
- Maintain consistent terminology and context cues. Ensure **consistent vocabulary** and pass forward essential details between steps to prevent drift and incoherence when integrating sub-outputs[2].  
- Balance decomposition levels. Avoid **over-decomposition**; too many components create unnecessary complexity, redundancy, and integration challenges that can degrade results[2].  
- Manage interdependencies explicitly. Identify dependencies across subtasks and plan **information handoffs** so each step has just the context it needs without repetition[2].  
- Sequence tasks logically. Frame complex requests as a **stepwise plan** (e.g., analysis → design → execution → synthesis), enabling better reasoning and comprehensive coverage of the goal[2].  
- Plan for integration. Anticipate the **integration step** and specify how to combine outputs (schemas, formats, acceptance checks) to ensure smooth assembly into a final deliverable[2].  
- Weigh time vs. benefit. Recognize that detailed decomposition requires time; apply it **strategically** where accuracy and control matter most[2].

-----

-----

### Source [13]: https://silicondales.com/ai/decomposed-prompting/

Query: What are best practices for prompting an AI agent to perform goal decomposition on a complex task?

Answer: - Break complex prompts into smaller, sequential sub-prompts. Decomposed prompting improves LLM performance by guiding **multi-step reasoning** and reducing failure on complex instructions[3].  
- Provide step-by-step guidance instead of only asking for a final answer. Treat the prompt like a **turn-by-turn route**, specifying intermediate outputs to scaffold reasoning and improve accuracy in tasks like complex QA and code generation[3].  
- Use structured sub-prompts to simplify problem decomposition. A **structured sequence** clarifies expectations at each step, making it easier for the model to handle complexity and for you to monitor progress and quality[3].  
- Apply in agentic/orchestrated systems. Decomposed prompting is a **cornerstone of modern AI agents and LLM orchestration**, enabling specialized roles and clearer task boundaries for better results[3].

-----

</details>

<details>
<summary>What are common techniques for implementing self-correction loops in AI agents?</summary>

### Source [14]: https://relevanceai.com/prompt-engineering/learn-to-use-critic-prompting-for-self-correction-in-ai-responses

Query: What are common techniques for implementing self-correction loops in AI agents?

Answer: - Describes **CRITIC prompting** as a structured self-correction loop where an AI generates an initial response, performs critical analysis, and then refines the output before delivery.  
- Outlines a three-phase pipeline:  
  - **Response Generation:** initial content creation, context analysis, and basic error checks.  
  - **Critical Analysis:** deep evaluation of content, consistency checks, and accuracy verification.  
  - **Refinement Implementation:** targeted error correction, optimization, and quality enhancement.  
- Emphasizes integrating multi-layer review steps to systematically detect and correct issues such as inconsistencies or tone/brand misalignment in real-world applications (e.g., marketing copy).  
- Notes engineering considerations: ensure the self-correction pipeline does not introduce excessive computational overhead while maintaining response speed and accuracy, implying practical limits and the need for careful system design.

-----

-----

### Source [15]: https://galileo.ai/blog/self-evaluation-ai-agents-performance-reasoning-reflection

Query: What are common techniques for implementing self-correction loops in AI agents?

Answer: - Focuses on **self-evaluation** via error identification mechanisms that monitor outputs in real time for inconsistencies, implausibilities, or errors, acting as internal verification layers.  
- Recommends a **layered detection approach**:  
  - **Self-consistency:** generate multiple reasoning paths and compare results to flag inconsistencies and select the most reliable answer.  
  - **Retrieval-augmented verification:** automatically cross-reference claims against trusted sources using indexing and embedding search for factual validation.  
  - **Entropy/probability monitoring:** track token-level uncertainty patterns correlated with hallucinations.  
  - **Domain-specific verifiers:** specialized modules for math/numerics to parse and recompute results for independent validation.  
- Suggests prioritizing high-precision verification in critical domains (medical, financial, legal) and implementing multi-stage reasoning processes to enable effective self-reflection and quality control.

-----

-----

### Source [16]: https://www.lionbridge.com/blog/ai-training/ai-self-correction/

Query: What are common techniques for implementing self-correction loops in AI agents?

Answer: - Lists practical, prompt-based self-correction techniques:  
  - **Accuracy-focused prompts:** prepend instructions emphasizing accuracy, factuality, careful reasoning, and admitting uncertainty.  
  - **Expert persona prompts:** configure the agent to act as a detailed domain expert with explicit best-practice guidance to reduce hallucinations; test prompts on known tasks to iteratively refine.  
- Highlights best practices: make expert personas specific (avoid generic roles), include widely accepted procedures, and consider multiple persona variants for different task types to preempt errors.  
- Frames these as low-friction methods to nudge models toward self-checking behavior within the prompt stack, complementing other programmatic verification loops.

-----

-----

### Source [17]: https://dev.to/louis-sanna/self-correcting-ai-agents-how-to-build-ai-that-learns-from-its-mistakes-39f1

Query: What are common techniques for implementing self-correction loops in AI agents?

Answer: - Presents a simple self-correction loop with three core techniques:  
  - **Error detection:** identify wrong outputs via exceptions, error codes, failing test cases, or performance thresholds.  
  - **Reflection:** analyze what went wrong by logging errors, tracking failed API calls, and re-evaluating response quality.  
  - **Retry logic:** attempt improved strategies (switch APIs, optimize algorithms, use backups) after reflection.  
- Provides an example (optimizing a Fibonacci implementation) to illustrate iterative improvement where the agent autonomously retries with better methods.  
- Suggests feeding error logs back into the model to improve future performance, operationalizing a closed feedback loop that combines telemetry with adaptive prompting or policy updates.

-----

-----

### Source [18]: https://spiralscout.com/blog/self-modifying-ai-software-development

Query: What are common techniques for implementing self-correction loops in AI agents?

Answer: - Describes **self-modifying AI agents** that maintain an internal model of a codebase and update it through feedback loops when errors are corrected by users.  
- When a wrong answer is flagged, the agent runs internal workflows (e.g., vector DB lookups, dependency graph rebuilding) to understand the mistake and adjust its internal model and reasoning processes.  
- Emphasizes that the agent does more than store corrections: it can rescan databases, rebuild dependency graphs, and adjust “cognitive pathways,” enabling persistent learning tailored to the project.  
- Frames this as a continuous adaptation loop where user feedback directly informs the agent’s knowledge representation and future decision-making, improving effectiveness over time.

-----

</details>

<details>
<summary>How is agentic planning and reasoning applied in large-scale, deep research systems?</summary>

### Source [19]: https://www.glean.com/blog/a-complete-guide-to-agentic-reasoning

Query: How is agentic planning and reasoning applied in large-scale, deep research systems?

Answer: Agentic reasoning enables large-scale research systems to not only retrieve information but also to plan, act, evaluate, and iteratively improve toward goals using reasoning loops.[1] In practice, agentic systems fuse large language models with external tools and specialized agents—such as web search agents for real-time information gathering, code execution agents for analysis or simulation, and mind map/graph agents for structuring complex domains—allowing end-to-end workflows across open-ended problems.[1] For deep research, this means dynamically scoping a problem, decomposing it into sub-questions, sourcing fresh evidence across internal and external corpora, running computational checks, organizing knowledge graphs of claims/evidence, and converging on decisions grounded in the latest context.[1] The agent can adapt plans in real time as it observes intermediate outcomes, using cyclical reasoning to refine hypotheses, select tools, and correct course—contrasting with static pipelines.[1] A representative example is complex medical literature triage: the agent searches current studies, simulates outcomes based on patient parameters, maps factors and relations, and synthesizes a recommendation—integrating research, reasoning, and action into a single coherent process.[1]

-----

-----

### Source [20]: https://www.glean.com/blog/agentic-reasoning-future-ai

Query: How is agentic planning and reasoning applied in large-scale, deep research systems?

Answer: In enterprise-scale knowledge work, agentic reasoning is applied through multi-step plans where each step is executed by specialized tools and sub-agents (search, reasoning, data analysis, employee/expert search), with built-in self-reflection to optimize the path to the goal.[5] For deep research-like tasks (e.g., resolving a complex support ticket), the system must infer the core problem, retrieve targeted documentation, synthesize cross-source evidence, and draft a tailored response—requiring decomposition, evidence routing, and iterative evaluation.[5] Glean reports early research showing a 24% increase in response and action relevance with its agentic architecture, indicating measurable gains from structured planning and reflective loops at scale.[5] The architecture illustrates how agentic planning operationalizes: break down objectives, orchestrate tool use per step, monitor outcomes, and adapt plans—supporting large-scale, high-variance research workflows where context quality and tool sequencing determine end results.[5]

-----

-----

### Source [21]: https://www.ayadata.ai/how-ai-agents-actually-think-planning-reasoning-and-why-it-matters-for-enterprise-ai/

Query: How is agentic planning and reasoning applied in large-scale, deep research systems?

Answer: Reasoning models strengthen the planning core of agentic systems—deciding what to do next and why—addressing a common weakness in enterprise deployments handling complex, multi-step research tasks.[2] Enhanced planning includes better task decomposition, error handling and recovery, sophisticated tool selection and sequencing, and adaptation to changing circumstances during execution—capabilities essential for long-running, deep research pipelines.[2] Strategic thinking improvements include evaluating alternative approaches before acting, prioritizing goals, and understanding dependencies and constraints—enabling robust orchestration across data sources, analyses, and validation passes.[2] The piece outlines multiple approaches relevant to large-scale research: conditional-logic agents for high-reliability segments; goal-based planning using search to find optimal action sequences; and iterative ReAct loops where the agent reasons, acts, observes, and updates its plan—powerful for exploratory literature reviews or open-ended investigations but requiring safeguards against looping.[2]

-----

-----

### Source [22]: https://blog.kore.ai/what-is-agentic-reasoning

Query: How is agentic planning and reasoning applied in large-scale, deep research systems?

Answer: Agentic reasoning is centered on a reasoning engine that processes information, evaluates options, and executes decisions autonomously—integrating algorithms, contextual awareness, and real-time adaptability to achieve specified goals without direct human control.[3] Core principles relevant to deep research include autonomous problem-solving (independently decomposing complex objectives and executing tasks end-to-end) and adaptability with contextual awareness (interpreting nuanced instructions and adjusting actions to evolving evidence).[3] For large-scale systems, Kore.ai emphasizes scalability through modular design and pre-built integrations, allowing the architecture to extend across diverse workflows—key for research environments spanning multiple domains and tools.[3] Human-in-the-loop mechanisms are embedded for critical decisions, balancing autonomy with oversight—vital in high-stakes research to validate findings, resolve ambiguities, and maintain trust while still benefiting from agentic speed and coverage.[3]

-----

-----

### Source [23]: https://www.lyzr.ai/blog/agentic-reasoning/

Query: How is agentic planning and reasoning applied in large-scale, deep research systems?

Answer: Applied agentic reasoning distributes work across specialized agents—web-search for live evidence gathering, coding for computation/simulation, and mind-map agents for structuring insights—mirroring human delegation in research programs.[4] In operation, these agents build and refine context collaboratively, performing fact-checking, computing, and iterative refinement, culminating in explainable decisions rather than single-pass outputs.[4] The outlined pattern maps directly to deep research: systematic evidence retrieval, quantitative modeling via code execution, and knowledge structuring via graphs or maps to capture relationships among variables, claims, and sources—supporting traceability and synthesis at scale.[4] This multi-agent composition underscores how planning determines agent roles and sequencing, while reasoning governs when to branch, revisit assumptions, or escalate for human review—practices essential to reliable large-scale research workflows.[4]

-----

</details>

<details>
<summary>What are the architectural differences between modern reasoning models that use interleaved "thinking" streams (like Claude 3.5) versus explicit ReAct-style loops for agentic tasks?</summary>

### Source [24]: https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html

Query: What are the architectural differences between modern reasoning models that use interleaved "thinking" streams (like Claude 3.5) versus explicit ReAct-style loops for agentic tasks?

Answer: - Modern Claude models support an explicit “extended thinking” mode where you allocate a separate token budget for the model’s internal reasoning before it produces a final answer, exposing varying levels of transparency into step-by-step thoughts.[1]
- Extended thinking can be enabled per request and includes controls like max thinking tokens and interactions with context-window limits and costs, indicating an architectural path where a single model interleaves internal “thinking” content with final outputs under a governed budget.[1]
- Streaming behavior differs when thinking is enabled: the system may emit larger chunks interleaved with token-by-token delivery, especially for thinking content, reflecting a batched-generation pipeline for internal reasoning streams distinct from standard answer tokens.[1]
- The feature works across multiple Claude versions (e.g., Claude 3.7 Sonnet, Claude Sonnet 4, Claude Opus 4), but with noted differences in “thinking across model versions,” suggesting model-specific implementations of the thinking stream and its exposure/controls.[1]
- Tool use integrates with extended thinking, implying that the model’s internal chain-of-thought-like content can guide function calls within the same interaction loop without requiring an external, explicit ReAct controller; the “thinking” stream is an internal first-class channel coordinated by the API.[1]

-----

-----

### Source [25]: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude

Query: What are the architectural differences between modern reasoning models that use interleaved "thinking" streams (like Claude 3.5) versus explicit ReAct-style loops for agentic tasks?

Answer: - Claude 3.7 Sonnet is “the first Claude model to offer extended thinking—the ability to solve complex problems with careful, step-by-step reasoning,” and lets developers choose “standard thinking” for speed or “extended thinking” for advanced reasoning within the same single model interface.[3]
- The single-model design supports agentic tasks like “agentic coding,” “customer-facing agents,” and “computer use,” implying that planning, tool selection, and error correction occur within the model’s integrated reasoning mode rather than an externally orchestrated ReAct loop.[3]
- Positioning emphasizes that extended thinking improves instruction following and tool selection for complex workflows, aligning with an architecture where the model’s internal interleaved reasoning guides multi-step actions natively, contrasted with ReAct which typically relies on an outer loop to alternate between Thought, Action, and Observation.[3]

-----

-----

### Source [26]: https://docs.anthropic.com/en/docs/about-claude/models/overview

Query: What are the architectural differences between modern reasoning models that use interleaved "thinking" streams (like Claude 3.5) versus explicit ReAct-style loops for agentic tasks?

Answer: - Anthropic’s latest models (e.g., Claude Opus 4.1, Claude Sonnet 4) are described as having “superior reasoning capabilities,” with large context windows and tool support, fitting an architecture where richer internal reasoning is embedded into base model behavior and exposed via productized features like extended thinking.[4]
- The documentation highlights tools and the Model Context Protocol (MCP), indicating official support for tool use within the same conversational interface, which pairs naturally with internal thinking streams to plan and invoke tools without necessarily requiring an explicit ReAct-style external loop.[4]

-----

-----

### Source [27]: https://www.anthropic.com/news/claude-3-5-sonnet

Query: What are the architectural differences between modern reasoning models that use interleaved "thinking" streams (like Claude 3.5) versus explicit ReAct-style loops for agentic tasks?

Answer: - The announcement frames Claude 3.5 Sonnet’s advances in reasoning (including vision) as step-change improvements, consistent with a trajectory where internal, integrated reasoning supports complex multimodal tasks without requiring external ReAct scaffolding for many use cases.[2]
- “Artifacts” enable a collaborative workspace alongside conversation, which complements an interleaved-thinking architecture by letting the model generate, refine, and present intermediate work products in-session, instead of depending on an external Thought/Action/Observation loop to structure intermediate outputs.[2]

-----

-----

### Source [59]: https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html

Query: What are the architectural differences between modern reasoning models that use interleaved "thinking" streams (like Claude 3.5) versus explicit ReAct-style loops for agentic tasks?

Answer: - Modern Claude models support an explicit “extended thinking” mode that runs an internal reasoning process with a configurable token budget before emitting the final answer, providing varying transparency into step-by-step thoughts.[1]
- Developers enable thinking mode and set a maximum internal reasoning token budget, indicating a built-in capability for budgeted, interleaved internal “thinking” separate from final output.[1]
- Streaming behavior with thinking enabled can produce alternating bursts and token-by-token delivery, reflecting that the model interleaves internal reasoning content and externalized output in batched segments during generation.[1]
- The feature integrates with tool use: “Extended thinking with tool use” is supported, implying the model’s internal chain-of-thought can proceed across tool calls without forcing an external loop controller to alternate explicit “Thought/Act/Observe” turns.[1]
- The docs distinguish behavior across model versions (e.g., Claude 3.7 vs Claude 4), suggesting architectural and API differences in how internal thinking is handled and surfaced, rather than relying on a fixed, explicit ReAct loop structure.[1]
- There are considerations for prompt and “thinking block caching,” indicating the system can cache parts of the internal reasoning segments to optimize repeated or multi-step tasks without explicit outer orchestration loops.[1]
- Token and cost management are first-class: limits on max tokens and context window with extended thinking, plus cost considerations, show built-in support for long, interleaved internal reasoning within a single model invocation, contrasted with external ReAct loops that typically incur multiple round-trips and token contexts across steps.[1]

-----

-----

### Source [60]: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude

Query: What are the architectural differences between modern reasoning models that use interleaved "thinking" streams (like Claude 3.5) versus explicit ReAct-style loops for agentic tasks?

Answer: - Claude 3.7 Sonnet is described as the first Claude model to “offer extended thinking—the ability to solve complex problems with careful, step-by-step reasoning,” and lets users choose between standard thinking and extended thinking within a single model, rather than mandating an external ReAct controller.[2]
- It is optimized for agentic coding, customer-facing agents, computer use, and complex workflows, indicating the internal extended thinking is intended to coordinate planning, tool selection, and error correction inside one invocation, not just across explicit tool loops.[2]
- The guidance positions extended thinking as native to the model, allowing a balance of speed vs. quality without switching to an explicit multi-turn Thought/Act/Observe orchestration, which is typical of ReAct-style loops.[2]

-----

-----

### Source [61]: https://docs.anthropic.com/en/docs/about-claude/models/overview

Query: What are the architectural differences between modern reasoning models that use interleaved "thinking" streams (like Claude 3.5) versus explicit ReAct-style loops for agentic tasks?

Answer: - Anthropic’s model overview highlights “superior reasoning capabilities” for models like Opus 4.1 and Sonnet 4, framing reasoning as an inherent capability of the model class, rather than something that necessarily requires an external ReAct loop to function.[3]
- The page emphasizes large context windows and advanced reasoning, consistent with designs where the model conducts substantial internal planning and reasoning in a single session, complementing rather than depending on explicit external ReAct cycles.[3]

-----

-----

### Source [62]: https://www.anthropic.com/news/claude-3-5-sonnet

Query: What are the architectural differences between modern reasoning models that use interleaved "thinking" streams (like Claude 3.5) versus explicit ReAct-style loops for agentic tasks?

Answer: - Claude 3.5 Sonnet advances “intelligence” and visual reasoning, and introduces “Artifacts” for interactive creation, reflecting a product direction where the model can sustain complex reasoning and generation workflows in one environment, not solely through explicit ReAct loops.[4]
- The positioning suggests that complex, multi-step tasks (e.g., code, documents, designs) are managed within a collaborative workspace tied to the model’s ongoing internal reasoning, compatible with interleaved thinking streams rather than strict external Thought/Act/Observe cycles.[4]

-----

</details>

<details>
<summary>What are the most effective prompting strategies to induce robust self-correction and goal decomposition in AI agents when faced with conflicting information or unexpected tool outputs?</summary>

### Source [28]: https://arxiv.org/html/2401.14043v3

Query: What are the most effective prompting strategies to induce robust self-correction and goal decomposition in AI agents when faced with conflicting information or unexpected tool outputs?

Answer: The survey proposes a goal-oriented taxonomy of prompting that explicitly targets two capabilities: **goal decomposition** and **self-correction via sub-goal evaluation**.[1] It reports that, compared with standard input–output prompting, goal decomposition using Chain-of-Thought (CoT) yields a 22.6% average improvement on arithmetic reasoning, while self-refine-style sub-goal evaluation improves by 21.1%, and valuable sub-goal selection via self-consistency improves by 32.5% in arithmetic reasoning, indicating robust gains from decomposition and evaluation stages.[1] The paper emphasizes that the effectiveness of these strategies is task-dependent: goal decomposition can outperform sub-goal selection in symbolic reasoning, suggesting that when faced with conflicting information, explicit decomposition helps isolate contradictions at the subtask level.[1] It also highlights “stage synergy”: combining decomposition with action execution (Program-of-Thought, PoT) further improves CoT by 14.7% in arithmetic reasoning, while methods integrating goal decomposition, action selection, and sub-goal evaluation (e.g., planning frameworks like SayCan variants) vastly outperform using sub-goal evaluation alone, implying multi-stage prompting pipelines are more robust to unexpected tool outputs through iterative evaluation and replanning.[1] The authors note comparison challenges due to prompt template and decoding differences, underscoring the need for unified evaluation, but their aggregated results support practices such as: decomposing goals into sub-goals; selecting among diverse reasoning paths (self-consistency) to reduce brittleness; and introducing explicit sub-goal evaluation or self-refinement loops to detect and correct errors that arise from tools or conflicting evidence.[1]

-----

-----

### Source [29]: https://www.human-i-t.org/beginner-guide-prompt-engineering/

Query: What are the most effective prompting strategies to induce robust self-correction and goal decomposition in AI agents when faced with conflicting information or unexpected tool outputs?

Answer: The guide presents “Chain of Thought Factored Decomposition Prompt Engineering” as a practical strategy to break complex tasks into sequential components, prompting the model first to elicit missing or conflicting information, then to propose solutions, and finally to execute resolution steps.[5] Its worked example demonstrates a three-phase scaffold: (1) ask targeted questions to clarify constraints and surface inconsistencies, (2) generate alternatives conditioned on clarified constraints, and (3) outline an actionable plan, which together function as a lightweight goal decomposition and self-correction loop.[5] This structure encourages the model to pause and interrogate assumptions before committing to outputs, a pattern that mitigates error propagation when tool outputs are unexpected or input signals conflict. By explicitly instructing the model to separate understanding, solution generation, and execution, the approach operationalizes decomposition and embeds checkpoints where the model can reconcile discrepancies, revise intermediate conclusions, and align final actions to the clarified goal.[5] The emphasis on stepwise prompting and targeted questioning offers a template for agent prompts: require an initial “information gap and conflict check,” compel conditional decision-making based on newly gathered details, and conclude with a concrete, verifiable plan, thereby inducing robust self-correction through structured intermediate reasoning stages.[5]

-----

-----

### Source [63]: https://arxiv.org/html/2401.14043v3

Query: What are the most effective prompting strategies to induce robust self-correction and goal decomposition in AI agents when faced with conflicting information or unexpected tool outputs?

Answer: The paper proposes a goal-oriented taxonomy of prompting methods for LLMs, organizing techniques by stages such as goal decomposition, action execution, sub-goal evaluation, and valuable sub-goal selection, and evaluates their effects across tasks.[1] It reports that, versus standard input–output prompting, goal decomposition (e.g., Chain-of-Thought) improves arithmetic reasoning by 22.6%, sub-goal evaluation (e.g., Self-Refine) by 21.1%, and valuable sub-goal selection (e.g., Self-Consistency) by 32.5%, indicating that multi-step reasoning and sampling diverse reasoning paths can yield more robust answers.[1] The paper notes that the best strategy depends on task characteristics: goal decomposition outperforms sub-goal selection in symbolic reasoning, whereas valuable sub-goal selection yields the largest gains in arithmetic reasoning.[1] Combining stages further boosts performance; Program-of-Thought (combining CoT with tool/action execution) adds another 14.7% over CoT in arithmetic reasoning, and multi-stage planners like SayPlan (goal decomposition + action selection + sub-goal evaluation) outperform single-stage evaluators (LLM-Planner) by 73.3% in planning excutability, suggesting stage synergy is effective when confronting complex or noisy tool use.[1] The authors also emphasize challenges comparing methods due to variations in templates, in-context examples, and decoding strategies, underscoring the importance of unified evaluation when deploying self-correction and decomposition workflows.[1]

-----

-----

### Source [64]: https://www.human-i-t.org/beginner-guide-prompt-engineering/

Query: What are the most effective prompting strategies to induce robust self-correction and goal decomposition in AI agents when faced with conflicting information or unexpected tool outputs?

Answer: This guide describes Chain-of-Thought factored decomposition as a structured prompting technique that breaks complex tasks into sequential subcomponents, encouraging stepwise reasoning and targeted information gathering before solution synthesis.[5] It illustrates correct application with a prompt that explicitly instructs the model to first elicit missing or conflicting details via targeted questions, then propose alternatives based on gathered information, and finally execute a resolution step, demonstrating how decomposition prompts can manage uncertainty and reduce error propagation when initial inputs are incomplete or conflicting.[5] The example highlights operational scaffolding: require the model to (1) identify and clarify the core issue, (2) branch into options conditioned on clarified evidence, and (3) produce an action plan, which is a practical pattern for inducing self-correction and robust goal decomposition in agent workflows facing unexpected outputs.[5]

-----

</details>

<details>
<summary>How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?</summary>

### Source [30]: https://arxiv.org/pdf/2503.13657

Query: How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?

Answer: The MAST taxonomy for multi-agent LLM systems distinguishes failures arising from system specification/design, coordination/execution, and post-execution evaluation, emphasizing that many breakdowns stem from structural choices rather than core model limits[1]. For agents that only call tools sequentially without explicit reasoning, typical failures align with specification issues (ambiguous prompts, poorly defined roles, inadequate tool permissions) and execution-time mis-coordination, because such agents lack planning states to detect or recover from upstream design defects[1]. MAST highlights fine-grained modes (defined in its appendices) such as ambiguous task decomposition and role-scoping errors; in a purely sequential caller, these manifest as silent misapplication of tools, brittle linear scripts, and inability to adjust workflows mid-task[1]. By contrast, reasoning agents introduce additional failure surface in the planning/reflection layer: incorrect chains of thought, flawed intermediate assumptions, or hallucinated subgoals can misguide otherwise correct tool sequences, making failures originate earlier (pre-execution planning) and propagate through execution[1]. The framework’s case studies show that fixes require redesign (clearer specifications, better orchestration, robust escalation), implying that non-reasoning sequential agents predominantly fail due to rigid pipelines and lack of adaptive control, whereas reasoning agents fail due to erroneous internal plans despite having richer control loops[1].

-----

-----

### Source [31]: https://arxiv.org/html/2508.04691

Query: How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?

Answer: An analysis of coordinated agent teams (MARS) documents persistent failure modes even with extensive knowledge bases and recovery protocols: hierarchical role misalignment, tool access violations, delayed or missing handling of failure reports, noncompliance with workflows, and false completion reporting[2]. For non-reasoning agents that merely call tools in order, these issues appear as linear execution proceeding despite permission mismatches and misassigned tasks, with no timely detection or escalation because there is no explicit reasoning step to validate state or reconsider plans[2]. Reasoning agents, intended to mitigate such issues, still exhibit these failures when structural coordination is weak; however, they add distinct failure pathways: misinterpretation of workflow logic, overconfident reflection leading to bypassing or claiming completion, and coordination breakdowns despite available guidance[2]. The study concludes the bottleneck is structural (communication/oversight), not information scarcity: non-reasoning agents fail by silently violating constraints and skipping recovery, while reasoning agents can compound errors by rationalizing or reinterpreting instructions, causing protocol noncompliance and false reporting under the guise of “reasoned” progress[2].

-----

-----

### Source [32]: https://huyenchip.com/2025/01/07/agents.html

Query: How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?

Answer: Documented agent failure modes include planning errors, tool execution mistakes, and reflection-induced illusions of completion[3]. A non-reasoning sequential tool caller primarily fails at the tool layer: invalid tool selection, wrong parameters, and unvalidated outputs, because it lacks an explicit planning/reflection loop to check plan validity or revise steps[3]. Metrics such as “out of all tool calls, how many are valid,” “how often are invalid tools called,” and “how often are valid tools called with invalid parameters” characterize these execution-centric failures[3]. In contrast, a reasoning agent introduces planning-specific failures: generating invalid or incomplete plans, requiring many attempts to produce a valid plan, and reflection errors where the agent insists a multi-step task is done when it is not (e.g., partial assignments treated as complete)[3]. Thus, non-reasoning agents skew toward detectable tool-call validity errors and brittle linearity, while reasoning agents add harder-to-detect cognitive errors (faulty plans, deceptive self-evaluation), necessitating evaluations that separately measure plan validity and the accuracy of reflective self-assessment[3].

-----

-----

### Source [33]: https://www.codemotion.com/magazine/ai-ml/ai-agents-reasoning-paradox/

Query: How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?

Answer: Reasoning (e.g., chain-of-thought) can improve complex tasks but creates a “paradox”: multi-step inference chains are vulnerable to single-step defects that cascade into wrong conclusions[4]. A sequential tool caller without internal reasoning typically fails locally—at the point of a bad tool call—leading to straightforward, step-local errors and limited propagation because there is no deep inference state to corrupt[4]. A reasoning agent, however, can suffer cumulative error: an early incorrect assumption in its reasoning chain misguides subsequent steps, yielding confidently wrong decisions across the task[4]. The article’s sequential reasoning example shows how an error in an intermediate function leads to a globally wrong outcome, illustrating that reasoning agents add failure modes like compounding inference mistakes and brittle dependency on intermediate correctness, whereas non-reasoning sequential callers primarily face linear execution errors without complex epistemic drift[4].

-----

-----

### Source [34]: https://galileo.ai/blog/why-most-ai-agents-fail-and-how-to-fix-them

Query: How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?

Answer: Common agent pitfalls include infinite looping, unclear termination criteria, and inadequate monitoring[5]. For non-reasoning sequential tool callers, looping often arises from rigid control loops that repeat calls without adaptive planning, making “clear termination conditions” and external monitors critical[5]. Reasoning agents may loop for different reasons: flawed plans or reflections that repeatedly revise or re-justify steps without converging, indicating that enhanced reasoning can paradoxically increase loop risk unless bounded by strong stopping rules[5]. The guidance implies contrasting mitigations: for non-reasoning agents, enforce strict termination guards and tool-level validations; for reasoning agents, combine those with planning-quality checks and constraints on reflection cycles to prevent justification loops and ensure progress toward multi-step task completion[5].

-----

-----

### Source [45]: https://arxiv.org/pdf/2503.13657

Query: How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?

Answer: According to this taxonomy of failures in multi-agent LLM systems (MAST), failures are grouped into specification issues, interaction/coordination issues, and limitations of individual agents, mapped to pre-execution, execution, and post-execution phases[1]. For a non-reasoning agent that only calls tools sequentially, failures disproportionately arise from specification issues and execution-time misuses of tools because the agent lacks internal deliberation to reinterpret or recover from ambiguous prompts—e.g., ambiguous or incomplete task specifications directly propagate into incorrect tool calls without self-correction[1]. Such agents also exhibit tool-use failures during execution (e.g., calling the wrong tool, violating tool constraints) that go undetected post-execution due to limited reflective checks[1]. In contrast, a reasoning agent introduces additional interaction failures: reasoning steps can amplify misalignment, produce overconfident but wrong plans, or create cascading errors during execution that originate from flawed deliberation even when tools are available[1]. MAST case studies emphasize that many failures stem from system design rather than just LLM capability—implying that purely sequential, non-reasoning pipelines fail by faithfully executing flawed specifications, while reasoning agents can fail by inventing flawed plans or miscoordinating roles despite adequate specifications[1]. The framework highlights the need for structural redesigns (e.g., clearer role definitions, validation checkpoints) to mitigate both failure classes[1].

-----

-----

### Source [46]: https://openreview.net/pdf?id=wM521FqPvI

Query: How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?

Answer: This grounded-theory study (MASFT) identifies 18 fine-grained failure modes across pre-execution, execution, and post-execution, including specification ambiguity/misalignment and coordination breakdowns[2]. Non-reasoning, sequential tool callers are especially vulnerable to “Specification Ambiguity and Misalignment”: incomplete or inconsistent instructions cause direct execution failures because there is no intermediate planning to reconcile gaps or seek clarifications[2]. They also suffer from execution failures like incorrect parameterization or ordering of tool calls, with limited post-execution verification to catch errors[2]. Reasoning agents, while capable of planning and disambiguation, introduce additional failure modes: erroneous or inconsistent plans, hallucinated intermediate assumptions, and misinterpretations that propagate through multi-step reasoning chains, leading to confident but incorrect multi-step execution[2]. MASFT’s mapping to phases clarifies that sequential agents fail “fast” at execution due to rigid adherence, whereas reasoning agents can fail earlier (planning) and then compound errors during execution and even post-execution reporting if reflective summaries rationalize mistakes[2]. The study’s examples show supervisors failing to seek needed clarifications—an archetype of reasoning/coordination failure that would differ from a non-reasoning pipeline which simply proceeds and fails silently[2].

-----

-----

### Source [47]: https://arxiv.org/html/2508.04691v1

Query: How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?

Answer: This analysis of coordination failures in multi-agent reasoning systems reports persistent failures even with detailed knowledge bases: hierarchical role misalignment (8/10 traces), tool access violations (10/10), lack of timely handling of failure reports (10/10), noncompliance with prescribed workflows (4/10), and bypassing or false completion reporting (2/10)[3]. It notes that non-reasoning models can show fewer failure patterns in the studied scenario, not due to better problem-solving but because simplified execution reduces the surface for coordination and reflective failure modes[3]. For non-reasoning sequential tool callers, predominant failures include tool access violations and rigid noncompliance with workflows when contingencies arise—there is no reasoning to detect, escalate, or recover from errors; failures remain unhandled and propagate to completion[3]. Reasoning agents, in contrast, introduce coordination-specific failure modes: managers performing subordinate tasks, delegating incorrectly, or rationalizing completion without actual work—failures arising from flawed internal reasoning and role interpretation rather than mere tool misuse[3]. Even with extensive failure recovery protocols, reasoning teams failed to detect and escalate issues, indicating structural limitations in reasoning-mediated coordination; by comparison, a purely sequential pipeline would typically fail by violating permissions or ordering without attempting remediation[3].

-----

-----

### Source [48]: https://huyenchip.com/2025/01/07/agents.html

Query: How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?

Answer: This overview of agent failure modes emphasizes that agents add unique failures from planning, tool execution, and efficiency, and that evaluating agents requires identifying and measuring the frequency of each mode[4]. For non-reasoning sequential tool callers, key failures cluster around tool execution: wrong tool choice, incorrect parameters, and brittle sequential dependencies that create single points of failure with minimal detect-and-recover capability[4]. Without planning or reflection, such agents underperform on complex, multi-step tasks requiring decomposition, error handling, or adaptive branching; their errors are often easier to spot as localized tool failures but harder to correct mid-run[4]. Reasoning agents add planning-related failures: incorrect task decomposition, overlong or inefficient chains, and cumulative error propagation across steps; they can also incur efficiency failures (unnecessary steps, tool overuse) stemming from suboptimal reasoning traces[4]. The piece underscores that as task complexity increases, the number of potential failure points grows—implying that reasoning, while necessary for complex tasks, expands the failure surface to include planning and coordination mistakes beyond the execution errors seen in purely sequential pipelines[4].

-----

-----

### Source [49]: https://www.codemotion.com/magazine/ai-ml/ai-agents-reasoning-paradox/

Query: How do the failure modes of a non-reasoning agent that only calls tools sequentially differ from a reasoning agent when executing a complex, multi-step task?

Answer: This article discusses the “paradox of reasoning,” noting that chain-of-thought can introduce cumulative error: a single mistaken intermediate inference can derail the entire multi-step process[5]. Applied to complex tasks, a reasoning agent may fail through error accumulation across steps, overconfidence in flawed intermediate conclusions, or getting sidetracked by digressions; these are distinct from non-reasoning sequential failures that typically arise from direct tool misapplication or rigid step order without adaptive correction[5]. The provided example illustrates failure in sequential reasoning where an incorrect simulation function biases downstream conclusions, exemplifying how reasoning adds a compounding failure mode beyond simple execution mistakes[5].

-----

</details>

<details>
<summary>What is the historical context and evolution from chain-of-thought (CoT) prompting to the development of the ReAct framework for AI agents?</summary>

### Source [35]: https://arxiv.org/abs/2201.11903

Query: What is the historical context and evolution from chain-of-thought (CoT) prompting to the development of the ReAct framework for AI agents?

Answer: The paper “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models” introduced chain-of-thought (CoT) as prompting that elicits a series of intermediate reasoning steps before the final answer, showing large gains on multi-step reasoning tasks.[5] It demonstrated that CoT works reliably in sufficiently large models (e.g., PaLM-scale, GPT-3+ scale) and improves performance on arithmetic word problems, commonsense reasoning, and symbolic reasoning benchmarks when the model is prompted to “think step by step.”[5] The authors evaluated few-shot CoT with exemplars that include rationales and highlighted that explicitly prompting for reasoning can transform LLM performance without gradient updates, contrasting with direct-answer prompting.[5] This work set the foundation for later techniques such as self-consistency (sampling multiple CoTs and aggregating) and motivated research on reasoning traces, prompting methods, and instruction design for complex tasks, establishing CoT as a core paradigm for structured reasoning in LLMs that later informed planning-and-reasoning workflows in agent frameworks.[5]

-----

-----

### Source [36]: https://en.wikipedia.org/wiki/Prompt_engineering

Query: What is the historical context and evolution from chain-of-thought (CoT) prompting to the development of the ReAct framework for AI agents?

Answer: Prompt engineering describes CoT as a technique that allows LLMs to solve problems through a series of intermediate steps, rather than outputting a final answer directly, aligning with Google Research’s formulation.[4] It documents automation around CoT, including “auto-CoT,” which selects diverse questions via clustering and has an LLM produce zero-shot CoT demonstrations to build few-shot exemplars—showing how CoT evolved from manual exemplars to automated generation of reasoning traces.[4] It also situates CoT within broader prompt design and optimization workflows (e.g., automatic prompt engineer), indicating a progression from handcrafted reasoning prompts to algorithmically searched or generated instructions, which influenced later agent methods that interleave reasoning with actions by improving the robustness and diversity of reasoning prompts used in complex tasks.[4]

-----

-----

### Source [37]: https://orq.ai/blog/what-is-chain-of-thought-prompting

Query: What is the historical context and evolution from chain-of-thought (CoT) prompting to the development of the ReAct framework for AI agents?

Answer: This overview explains CoT as structured, stepwise reasoning that improves accuracy and interpretability by having models articulate intermediate steps for complex problems.[1] It provides historical context: CoT emerged as researchers sought to enhance the reasoning ability of LLMs like GPT and PaLM; prompting the model to state intermediate steps significantly improved benchmark performance and interpretability.[1] The evolution includes follow-on techniques like self-consistency that refine how models navigate uncertainty, indicating a trajectory from basic CoT to ensemble-style reasoning that samples multiple chains and aggregates outcomes.[1] The article emphasizes CoT’s role in product development and real-world applications, framing it as a foundational capability upon which more advanced agent paradigms—those that combine reasoning with external actions or tools—were built.[1]

-----

-----

### Source [38]: https://ramp.com/blog/chain-of-thought-prompting

Query: What is the historical context and evolution from chain-of-thought (CoT) prompting to the development of the ReAct framework for AI agents?

Answer: This piece narrates the origin and diffusion of CoT: it gained prominence in 2022 via the Google Brain paper by Jason Wei and Denny Zhou, which showed explicit reasoning instructions markedly improve performance on tasks like arithmetic and logical inference, especially in large models (PaLM, GPT-3+ scale).[2] It characterizes CoT as “show your work” prompting that yields more accurate and trustworthy results and notes its rapid adoption across research and enterprise use cases beyond math, including data analysis, strategy evaluation, hypothesis generation, coding, and education—marking the transition from niche research to a general reasoning interface for LLM applications.[2] This broadening set the stage for agent frameworks that integrate CoT-style reasoning with decision and action loops, where transparent intermediate steps assist planning and tool use.[2]

-----

-----

### Source [39]: https://www.gigaspaces.com/blog/chain-of-thought-prompting-and-explainable-ai

Query: What is the historical context and evolution from chain-of-thought (CoT) prompting to the development of the ReAct framework for AI agents?

Answer: This article frames CoT as a significant advancement for LLM reasoning and explainability: instead of only final answers, CoT prompts models to articulate step-by-step logic, improving performance on tasks demanding logic, calculation, and decision-making.[3] It highlights zero-shot CoT—using generic instructions like “explain step by step” without task-specific exemplars—showing that sufficiently capable models can be induced to perform multi-step reasoning via prompt design alone.[3] By emphasizing planning and sequential reasoning, it connects CoT to human-like cognitive processes and transparency, a conceptual bridge toward agent frameworks that must both plan and act; the practice of verbalizing internal reasoning underpins later methods that interleave reasoning with environment interactions (as in ReAct-style agents) to improve task success and interpretability.[3]

-----

-----

### Source [50]: https://arxiv.org/abs/2201.11903

Query: What is the historical context and evolution from chain-of-thought (CoT) prompting to the development of the ReAct framework for AI agents?

Answer: The paper “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models” (Wei et al., 2022) formally introduced **chain-of-thought (CoT) prompting** as a method to elicit step-by-step intermediate reasoning from LLMs, showing substantial gains on multi-step reasoning tasks.[1] The authors demonstrated that prompting models with exemplars that include detailed reasoning enables them to solve complex problems more accurately than giving only final-answer exemplars, especially on arithmetic, commonsense, and symbolic reasoning benchmarks.[1] The work established that larger models (e.g., PaLM) benefited more from CoT and that CoT could be combined with few-shot prompting to generalize across tasks.[1] This paper set the historical foundation for later techniques like self-consistency and tool-augmented reasoning by centering the idea that explicit intermediate reasoning improves both accuracy and interpretability, catalyzing an evolution from silent inference to visible, stepwise thought processes in LLMs.[1]

-----

-----

### Source [51]: https://en.wikipedia.org/wiki/Prompt_engineering

Query: What is the historical context and evolution from chain-of-thought (CoT) prompting to the development of the ReAct framework for AI agents?

Answer: The prompt engineering article situates **CoT** within a broader toolkit of prompting strategies and automation methods.[2] It attributes CoT to Google Research as a technique for solving problems via a series of intermediate steps rather than direct answers, aligning with the 2022 Wei et al. paper.[2] It also describes downstream evolutions such as “auto-CoT,” where LLMs generate diverse CoT demonstrations automatically by clustering questions and producing zero-shot CoT on representative items; these demonstrations are then used for few-shot prompts to improve generalization.[2] This historical arc shows progression from manual few-shot CoT exemplars to automated generation of reasoning traces, reflecting an increasing emphasis on scalable, systematic reasoning elicitation that set the stage for frameworks integrating reasoning with action (e.g., planning and tool use) characteristic of agentic methods like ReAct.[2]

-----

-----

### Source [52]: https://orq.ai/blog/what-is-chain-of-thought-prompting

Query: What is the historical context and evolution from chain-of-thought (CoT) prompting to the development of the ReAct framework for AI agents?

Answer: This guide contextualizes **chain-of-thought prompting** as a transformative approach that emerged to enhance LLM reasoning by articulating intermediate steps, improving both accuracy and interpretability across tasks such as arithmetic and logical problem-solving.[3] It outlines the historical development from early applications in GPT/PaLM-era models to refinements like **self-consistency**, which samples multiple reasoning paths to improve reliability, indicating a maturation from single-trace explanations to ensembles of thoughts for robustness.[3] The piece frames CoT as foundational in LLM product development and research, emphasizing its role in benchmarks and real-world applications, and highlighting its influence on later agentic paradigms where stepwise reasoning informs decisions and actions—an important precursor to frameworks like ReAct that blend reasoning with environment interaction.[3]

-----

-----

### Source [53]: https://www.gigaspaces.com/blog/chain-of-thought-prompting-and-explainable-ai

Query: What is the historical context and evolution from chain-of-thought (CoT) prompting to the development of the ReAct framework for AI agents?

Answer: This overview explains CoT as prompting LLMs to produce step-by-step reasoning, improving performance on tasks requiring logic, calculation, and decision-making by breaking problems into sequential components.[4] It notes adoption across major models (e.g., Gemini, OpenAI) and details mechanisms like zero-shot CoT where instructions alone elicit multi-step reasoning without specific exemplars.[4] By emphasizing transparency and structured deliberation, it charts the trajectory from direct-answer prompting to explicit reasoning processes, which provided the conceptual substrate for agent frameworks that not only think but also act—bridging from “explain your reasoning” to “reason-and-act” cycles as epitomized by ReAct.[4]

-----

</details>

<details>
<summary>In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?</summary>

### Source [40]: https://blog.langchain.com/planning-agents/

Query: In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?

Answer: Large-scale deep-research systems balance ReAct and Plan-and-Execute by separating a heavyweight “planner” from an execution runtime and only invoking the planner at re-planning checkpoints and for the final synthesis, which reduces latency and cost compared to looping a single LLM after every tool call[3]. They route multi-step tool usage to cheaper or domain-specific models during execution, reserving top-tier models for planning and summaries, yielding cost savings over ReAct while improving completion quality by forcing explicit multi-step plans up front[3]. This pattern is preferred when tasks require multiple API calls or tools, because fewer high-cost LLM invocations are needed and execution can proceed autonomously between plan checkpoints; explicit planning also boosts task completion rates and quality for complex workflows[3]. In practice, teams implement variants (e.g., hierarchical planners, periodic re-plans) to retain some adaptivity of ReAct without per-action calls, striking a middle ground between responsiveness and efficiency for long-horizon research tasks[3].

-----

-----

### Source [41]: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/

Query: In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?

Answer: ReAct integrates tool-calling, memory, and planning in a single loop, enabling dynamic multi-step behavior with flexible tool use and stateful context, which is valuable when research directions shift based on intermediate findings[5]. However, because the LLM reasons after each step, pure ReAct can incur high token and time costs. To balance trade-offs, production systems commonly hybridize: they keep ReAct’s memory and tool-calling for adaptivity on uncertain subproblems, while constraining it within plan-and-execute scaffolds (e.g., run ReAct locally inside a subtask or within bounded iterations), thereby limiting per-action LLM calls but preserving responsiveness when evidence diverges from the initial plan[5]. This modularization lets orchestration frameworks switch between planned execution for predictable pipelines and reactive loops for exploratory branches, aligning with deep-research needs where some paths are deterministic (data collection/ETL) and others are open-ended (hypothesis refinement)[5].

-----

-----

### Source [42]: https://www.willowtreeapps.com/craft/building-ai-agents-with-plan-and-execute

Query: In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?

Answer: Plan-and-Execute systems in practice establish explicit tools, memory, and a loop: the agent first formulates a stepwise plan from the goal and available tools, then executes each step in order while updating memory with tool calls, parameters, and results[4]. Memory acts as progress tracking and informs subsequent planning rounds, enabling checkpointed re-plans instead of continuous per-action reasoning[4]. This structure is well-suited for web-scale research tasks (e.g., browsing and extraction) because execution proceeds autonomously across steps like open_browser → navigate → extract, while the planner intervenes only to adjust the plan or to assemble final outputs[4]. By isolating planning from execution and persisting structured traces, teams can bound iteration counts, enforce tool usage policies, and audit decisions—key controls for enterprise research agents balancing ReAct-like flexibility with plan-first efficiency[4].

-----

-----

### Source [43]: https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9

Query: In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?

Answer: Operational trade-offs often drive a hybrid choice: ReAct offers faster response time and medium token use, whereas Plan-and-Execute is slower per task but achieves higher completion accuracy on complex problems, with higher token usage and API calls[1]. Example cost figures with GPT-4-class models show ReAct around 2k–3k tokens and 3–5 calls versus Plan-and-Execute around 3k–4.5k tokens and 5–8 calls, with corresponding cost ranges, underscoring that planning can be more expensive but more reliable for complex analysis[1]. Teams mitigate this by front-loading planning for structure and delegating sub-steps to lighter tools/models; they reserve reactive loops for ambiguous steps where upfront plans are brittle, thus balancing total cost, latency, and reliability[1]. Case studies in data analysis illustrate that when tasks decompose cleanly, plan-and-execute yields better structured reports; when tasks are exploratory, reactive loops reduce overplanning overhead[1].

-----

-----

### Source [44]: https://www.nutrient.io/blog/rewoo-vs-react-choosing-right-agent-architecture/

Query: In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?

Answer: Deep-research workflows contend with ReAct’s variability: each think–act–observe cycle re-prompts with growing histories, so when answers aren’t found early, token costs and delays escalate; performance depends heavily on tool access and model ability[2]. For high-volume document processing and report generation, upfront planning (as in ReWOO/plan-then-execute) is advantageous: it structures extraction steps and minimizes repeated long-context prompts, stabilizing runtime and cost[2]. In contrast, interactive or underspecified research tasks benefit from ReAct’s tight feedback loop, which adapts goals at runtime based on observations[2]. Production systems therefore often combine them: use planned execution for deterministic ingestion and extraction phases, and gate into bounded ReAct loops when encountering ambiguous findings or missing data, thereby containing worst-case token growth while retaining adaptability needed for scientific or market analysis reviews[2].

-----

-----

### Source [54]: https://blog.langchain.com/planning-agents/

Query: In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?

Answer: - Large-scale systems favor a separation between a **planner** LLM and an **executor/runtime**, because it reduces repeated consultation of a large model after every action, a key inefficiency in pure ReAct loops[4].  
- Plan-and-Execute improves multi-step workflows: once a plan is produced, sub-tasks can be executed without repeatedly invoking the main agent, often using lighter or domain-specific models for sub-steps, lowering latency and cost at scale[4].  
- Cost control: heavyweight LLMs are reserved for planning/replanning and final synthesis, while cheaper models/tools perform sub-tasks—this is cited as a primary way to outperform ReAct on cost for complex jobs[4].  
- Quality/throughput: forcing an explicit plan increases completion rates and quality on complex tasks by ensuring global reasoning up front and enabling decomposition into focused steps, which scales better for market analysis or literature review pipelines[4].  
- Practical balance in production: systems adopt a Plan-and-Execute backbone with periodic “replanning” gates, enabling partial reactivity when new information arises, rather than fully reactive step-by-step loops throughout[4].

-----

-----

### Source [55]: https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9

Query: In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?

Answer: - Empirical trade-offs: Plan-and-Execute showed higher task completion accuracy (reported 92%) than ReAct (85%) on complex tasks, but with slower response time and higher token usage, reflecting the planning overhead often acceptable in deep research contexts[1].  
- Cost/performance profile: ReAct averaged 2k–3k tokens and 3–5 API calls; Plan-and-Execute 3k–4.5k tokens and 5–8 calls, implying higher per-task cost but better complex-task handling—commonly justified for high-stakes analyses[1].  
- Practical pattern: Use ReAct for quicker, exploratory steps or when tasks are simple; switch to Plan-and-Execute for structured, multi-step research where robustness and completeness matter (e.g., multi-dataset synthesis), often combining both within the same system[1].  
- Case design: In data analysis, Plan-and-Execute integrates explicit tools and an execution loop over a precomputed plan; in production research agents, this translates to predictable pipelines, auditability of steps, and easier scaling across documents/datasets[1].

-----

-----

### Source [56]: https://serjhenrique.com/react-wese-plan-and-execute-and-chatdb-architectures-applied-to-question-answer-database-use-case/

Query: In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?

Answer: - Hybridization pattern: Split into Plan, Execute (often via an internal ReAct for a step), and Replan—this balances the ReAct flexibility within steps with the global structure and safety of Plan-and-Execute[2].  
- Replan gates ask: “Do we have enough info to answer?” and “Do we need to replan?”—these checkpoints reduce unnecessary execution and allow dynamic adjustment when new evidence emerges, a common need in deep research workflows[2].  
- Execution strategy in practice: Take the first available step from the plan, add dependency context, run a localized ReAct-style resolution for that step, store results, then reassess—supporting iterative evidence accumulation without full reactive thrashing[2].  
- Benefit for large-scale reviews: This design short-circuits overly long plans when sufficient evidence is reached, and it localizes reactive loops to constrained subproblems, controlling token/cost variance while remaining adaptive[2].

-----

-----

### Source [57]: https://www.nutrient.io/blog/rewoo-vs-react-choosing-right-agent-architecture/

Query: In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?

Answer: - Architectural framing: ReAct offers think–act–observe loops suited to dynamic, ill-defined tasks; ReWOO/plan-then-execute suits structured processing like batch document extraction and reporting, typical in scientific and market research pipelines[3].  
- Variance control: ReAct can become slow and costly due to repeated prompts with full history; upfront planning mitigates this by reducing the number of high-cost deliberation steps and stabilizing latency across large batches[3].  
- Operational balance: Production agents often apply ReAct during discovery/interactive phases and switch to plan-first execution for large-scale processing, to ensure predictable throughput and budgets[3].  
- Tooling implications: With many tools or weak signal tasks, ReAct’s loop count grows; plan-first methods constrain tool usage to a deterministic sequence, improving efficiency and reliability in multi-document or multi-API workflows[3].

-----

-----

### Source [58]: https://www.willowtreeapps.com/craft/building-ai-agents-with-plan-and-execute

Query: In practice, how do large-scale "deep research" agentic systems, like those used for market analysis or scientific review, balance the trade-offs between ReAct and Plan-and-Execute architectures?

Answer: - System components: Plan-and-Execute agents rely on explicit **tools** (e.g., browser automation, link parsing) and **memory** (short- and long-term) to track progress and inform future steps—critical for deep research agents to avoid redundant work and to maintain context across large jobs[5].  
- Loop design: The agent receives an instruction, creates a plan based on available tools, then executes steps, leveraging memory to inform subsequent planning—mirroring how market/science review systems enforce structure while remaining adaptive via stored breadcrumbs[5].  
- Practical balance: Central planning defines scope and order; execution uses instrumented tools and memory to progress efficiently; replanning occurs if memory indicates blockers or new findings—this orchestrates a middle ground between rigid plans and fully reactive loops[5].  
- Scaling insight: By formalizing tool use and memory within the plan–execute loop, teams can parallelize step execution, audit actions, and bound token usage—key practices to scale deep research pipelines predictably[5]

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>Artificial intelligence is transforming how we work, streamlining tasks, improving decisions, and enhancing productivity. A major leap forward in this evolution is the rise of agentic reasoning: an approach that enables AI systems to plan, execute, and adapt like dynamic problem-solvers.</summary>

Artificial intelligence is transforming how we work, streamlining tasks, improving decisions, and enhancing productivity. A major leap forward in this evolution is the rise of agentic reasoning: an approach that enables AI systems to plan, execute, and adapt like dynamic problem-solvers.

Agentic reasoning combines large language models (LLMs) with powerful tools like web search, code execution, and structured memory. This synergy allows AI to break down complex problems, gather and analyze information, and respond in context — all while learning and refining its approach. The result is a new class of intelligent systems that can support everything from research and diagnostics to customer support and software development.

## What is agentic reasoning?

Agentic reasoning is an advanced AI capability that allows machines to **plan, act, evaluate, and improve** in pursuit of specific goals. Unlike traditional systems that follow fixed instructions or respond to patterns, agentic AI uses reasoning loops to make decisions in real time based on context.

At its core, agentic reasoning fuses large language models with external tools and agents that extend its capabilities. These tools may include:

- **Web search agents** to gather real-time, relevant information from internal or external sources
- **Code execution agents** to perform calculations, simulations, or data analysis
- **Mind map or graph agents** to structure and visualize relationships between key concepts and facts

This toolkit allows agentic systems to handle open-ended, multifaceted problems that require context-aware decisions and adaptive workflows.

For example, an AI agent helping diagnose a rare disease might search for the latest research, simulate treatment options based on patient data, map risk factors, and generate a personalized recommendation. That kind of synthesis — research, reasoning, and action — is what sets agentic reasoning apart from more reactive AI systems.

## How does agentic reasoning work?

Agentic systems operate through a continuous “ **think-act-observe**” loop that mirrors human problem-solving. This loop enables AI to plan, carry out actions, evaluate results, and refine its approach as it learns.

### Key components

At the core of every agentic system is an LLM. The LLM interprets user inputs, breaks down complex tasks, and communicates with external tools to carry out the work.

To extend its reasoning capabilities, the AI is paired with:

- **Information retrieval agents** that pull current, context-relevant insights from across data sources
- **Computational agents** that handle tasks like quantitative modeling or logic-based calculations
- **Conceptual mapping agents** that help organize and visualize ideas, dependencies, or flows

Together, these components form a modular system that can tackle more than just static Q&A — it can operate with intent.

### The reasoning loop

1. **Task decomposition**: The AI breaks a larger goal into smaller subtasks.
2. **Delegation**: It assigns subtasks to the most relevant tools or agents.
3. **Observation**: The AI reviews the output of each step, interpreting results and adjusting course as needed.
4. **Synthesis**: It integrates outputs into a broader solution or recommendation.
5. **Adaptation**: Based on results, the AI refines its approach and continues the loop.


This iterative loop allows agentic systems to move beyond simple prompt-response mechanics and into territory where decision-making, learning, and long-term planning are possible.

</details>

<details>
<summary>The expanding use of generative-AI applications has increased the demand for accurate, cost-effective large language models (LLMs). LLMs’ costs vary significantly based on their size, typically measured by the number of parameters: switching to the next smaller size often results in a 70%–90% cost savings. However, simply using smaller, lighter-weight LLMs is not always a viable option due to their diminished capabilities compared to state-of-the-art "frontier LLMs."</summary>

The expanding use of generative-AI applications has increased the demand for accurate, cost-effective large language models (LLMs). LLMs’ costs vary significantly based on their size, typically measured by the number of parameters: switching to the next smaller size often results in a 70%–90% cost savings. However, simply using smaller, lighter-weight LLMs is not always a viable option due to their diminished capabilities compared to state-of-the-art "frontier LLMs."

While reduction in parameter size usually diminishes performance, evidence suggests that smaller LLMs, when specialized to perform tasks like question-answering or text summarization, can match the performance of larger, unmodified frontier LLMs on those same tasks. This opens the possibility of balancing cost and performance by breaking complex tasks into smaller, manageable subtasks. Such _task decomposition_ enables the use of cost-effective, smaller, more-specialized task- or domain-adapted LLMs while providing control, increasing troubleshooting capability, and potentially reducing hallucinations.

However, this approach comes with trade-offs: while it can lead to significant cost savings, it also increases system complexity, potentially offsetting some of the initial benefits. This blog post explores the balance between cost, performance, and system complexity in task decomposition for LLMs.

As an example, we'll consider the case of using task decomposition to generate a personalized website, demonstrating potential cost savings and performance gains. However, we'll also highlight the potential pitfalls of overengineering, where excessive decomposition can lead to diminishing returns or even undermine the intended benefits.

## I. Task decomposition

Ideally, a task would be decomposed into subtasks that are independent of each other. That allows for the creation of targeted prompts and contexts for each subtask, which makes troubleshooting easier by isolating failures to specific subtasks, rather than requiring analysis of a single, large, black-box process.

Sometimes, however, decomposition into independent subtasks isn’t possible. In those cases, prompt engineering or information retrieval may be necessary to ensure coherence between subtasks. However, overengineering should be avoided, as it can unnecessarily complicate workflows. It also runs the risk of sacrificing the novelty and contextual richness that LLMs can provide by capturing hidden relationships within the complete context of the original task.

But we’ll address these points later. First, let us provide an example where the task of personalized website generation is decomposed into an _agentic workflow._ The _agents_ in an agentic workflow might be functional agents, which perform specific tasks (e.g., database query), or persona-based agents that mimic human roles in an organization (e.g., UX designer). In this post, I'll focus on the persona-based approach.

### A simple example: Creating a personalized website

In our scenario, a business wants to create a website builder that generates tailored web experiences for individual visitors, without human supervision. Generative AI's creativity and ability to work under uncertainty make it suitable for this task. However, it is crucial to control the workflow, ensuring adherence to company policies, best practices, and design guidelines and managing cost and performance.

Examples of web pages produced with generative AI.

This example is based on [an agentic-workflow solution](https://aws.amazon.com/blogs/machine-learning/reinvent-personalization-with-generative-ai-on-amazon-bedrock-using-task-decomposition-for-agentic-workflows/) we published on the Amazon Web Services (AWS) Machine Learning Blog. For that solution, we divided the overall process into subtasks of a type ordinarily assigned to human agents, such as the personalizer (UX/UI designer/product manager), artist (visual-art creator), and website builder (front-end developer).

Generating a personalized website using a single large LLM _(top)_ versus decomposing the task using smaller LLMs _(bottom)_.

The personalizer agent aims to provide tailored experiences for website visitors by considering both their profiles and the company's policies, offerings, and design approaches. This is an average-sized text-to-text LLM with some reasoning skills. The agent also incorporates retrieval-augmented generation (RAG) to leverage vetted "company research".

Here’s a sample prompt for the personalizer:

_You are an AI UI/UX designer tasked with creating a visually appealing website. Keep in mind the industry pain points \[specify relevant pain points — RAG retrieved\] to ensure a tailored experience for your customer \[provide customer profile — JSON to natural language\]. In your response, provide two sections: a website description for front-end developers and visual elements for the artists to follow. You should follow the design guidelines \[include relevant design guidelines\]._

The artist agent's role is to reflect the visual-elements description in a well-defined image, whether it's a background image or an icon. Text-to-image prompts are more straightforward, starting with "Create an \[extracted from personalizer response\]."

The final agent is the front-end developer, whose sole responsibility is to create the front-end website artifacts. Here, you can include your design systems, code snippets, or other relevant information. In our simple case, we used this prompt:

_You are an experienced front-end web developer tasked with creating an accessible, \[specify the website's purpose\] website while adhering to the specified guidelines \[include relevant guidelines\]. Carefully read the 'Website Description' \[response from personalizer\] provided by the UI/UX designer AI and generate the required HTML, CSS, and JavaScript code to build the described website. Ensure that \[include specific requirements\]._

Here, you can continue the approach with a quality assurance (QA) agent or perform a final pass to see if there are discrepancies.

## II. The big trade-off and the trap of overengineering

Task decomposition typically introduces additional components (new LLMs, orchestrators), increasing complexity and adding overhead. While smaller LLMs may offer faster performance, the increased complexity can lead to higher latency. Thus, task decomposition should be evaluated within the broader context.

Let's represent the task complexity as _O(n)_, where _n_ is the task size. With a single LLM, complexity grows linearly with task size. On the other hand, in parallel task decomposition with _k_ subtasks and _k_ smaller language models, the initial decomposition has a constant complexity — _O(1)_. Each of the _k_ language models processes its assigned subtask independently, with a complexity of _O(n/k)_, assuming an even distribution.

After processing, the results from the _k_ language models need coordination and integration. This step's complexity is _O(km)_, where fully pairwise coordination gives _m_ = 2, but in reality, 1 < _m_ ≤ 2.

Therefore, the overall complexity of using multiple language models with task decomposition can be expressed as

_Ok-LLMs = O(1)_ + _k (O(n/k))_ + _O(km)_ → _O(n)_ + _O(km)_

While the single-language-model approach has a complexity of _O(n)_, the multiple-language-model approach introduces an additional term, _O(km)_, due to coordination and integration overhead, with 1 < _m_ ≤ 2.

For small _k_ values and pairwise connectivity, the _O(km)_ overhead is negligible compared to _O(n)_, indicating the potential benefit of the multiple-language-model approach. However, as _k_ and _m_ grow, the _O(km)_ overhead becomes significant, potentially diminishing the gains of task decomposition. The optimal approach depends on the task, the available resources, and the trade-off between performance gains and coordination overhead. Improving technologies will reduce _m_, lowering the complexity of using multiple LLMs.

### A mental model for cost and complexity

A helpful mental model for deciding whether to use task decomposition is to consider the estimated total cost of ownership (TCO) of your application. As your user base grows, infrastructure cost becomes dominant, and optimization methods like task decomposition can reduce TCO, despite the upfront engineering and science costs. For smaller applications, a simpler approach, such as selecting a large model, may be more appropriate and cost effective.

A mental model to help decide the question of complexity versus simplicity.

### Overengineering versus novelty and simplicity

Task decomposition and the creation of agentic workflows with smaller LLMs can come at the cost of the novelty and creativity that larger, more powerful models often display. By “manually” breaking tasks into subtasks and relying on specialized models, the overall system may fail to capture the serendipitous connections and novel insights that can emerge from a more holistic approach. Additionally, the process of crafting intricate prompts to fit specific subtasks can result in overly complex and convoluted prompts, which may contribute to reduced accuracy and increased hallucinations.

Task decomposition using multiple, smaller, fine-tuned LLMs offers a promising approach to improving cost efficiency for complex AI applications, potentially providing substantial infrastructure cost savings compared to using a single, large, frontier model. However, care must be taken to avoid overengineering, as excessive decomposition can increase complexity and coordination overhead to the point of diminishing returns. Striking the right balance between cost, performance, simplicity, and retaining AI creativity will be key to unlocking the full potential of this promising approach.

</details>

<details>
<summary>We’re releasing three agent architectures in LangGraph showcasing the “plan-and-execute” style agent design. These agents promise a number of improvements over traditional Reasoning and Action (ReAct)-style agents.</summary>

We’re releasing three agent architectures in LangGraph showcasing the “plan-and-execute” style agent design. These agents promise a number of improvements over traditional Reasoning and Action (ReAct)-style agents.

⏰ First of all, they can execute multi-step workflow _**faster**,_ since the larger agent doesn’t need to be consulted after each action. Each sub-task can be performed without an additional LLM call (or with a call to a lighter-weight LLM).

💸 Second, they offer **cost savings** over ReAct agents. If LLM calls are used for sub-tasks, they typically can be made to smaller, domain-specific models. The larger model then is only called for (re-)planning steps and to generate the final response.

🏆 Third, they can **perform better** overall (in terms of task completions rate and quality) by forcing the planner to explicitly “think through” all the steps required to accomplish the entire task. Generating the full reasoning steps is a tried-and-true prompting technique to improve outcomes. Subdividing the problem also permits more focused task execution.

## Background

Over the past year, language model-powered agents and state machines have emerged as a promising design pattern for creating flexible and effective ai-powered products.

At their core, agents use LLMs as general-purpose problem-solvers, connecting them with external resources to answer questions or accomplish tasks.

LLM agents typically have the following main steps:

1. Propose action: the LLM generates text to respond directly to a user or to pass to a function.
2. Execute action: your code invokes other software to do things like query a database or call an API.
3. Observe: react to the response of the tool call by either calling another function or responding to the user.

The [ReAct](https://arxiv.org/abs/2210.03629?ref=blog.langchain.com) agent is a great prototypical design for this, as it prompts the language model using a repeated thought, act, observation loop:

```
Thought: I should call Search() to see the current score of the game.
Act: Search("What is the current score of game X?")
Observation: The current score is 24-21
... (repeat N times)
```

A typical ReAct-style agent trajectory.

This takes advantage of [Chain-of-thought](https://arxiv.org/abs/2201.11903?ref=blog.langchain.com) prompting to make a single action choice per step. While this can be effect for simple tasks, it has a couple main downsides:

1. It requires an LLM call for each tool invocation.
2. The LLM only plans for 1 sub-problem at a time. This may lead to sub-optimal trajectories, since it isn't forced to "reason" about the whole task.

One way to overcome these two shortcomings is through an explicit planning step. Below are two such designs we have implemented in LangGraph.

## **Plan-And-Execute**https://blog.langchain.com/content/images/2024/02/plan-and-execute.pngPlan-and-execute Agent

Based loosely on Wang, et. al.’s paper on [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091?ref=blog.langchain.com), and Yohei Nakajima’s [BabyAGI](https://github.com/yoheinakajima/babyagi?ref=blog.langchain.com) project, this simple architecture is emblematic of the planning agent architecture. It consists of two basic components:

1. A **planner**, which prompts an LLM to generate a multi-step plan to complete a large task.
2. **Executor**(s), which accept the user query and a step in the plan and invoke 1 or more tools to complete that task.

Once execution is completed, the agent is called again with a re-planning prompt, letting it decide whether to finish with a response or whether to generate a follow-up plan (if the first plan didn’t have the desired effect).

This agent design lets us avoid having to call the large planner LLM for each tool invocation. It still is restricted by serial tool calling and uses an LLM for each task since it doesn't support variable assignment.

## Reasoning WithOut Observations

In [ReWOO](https://arxiv.org/abs/2305.18323?ref=blog.langchain.com), Xu, et. al, propose an agent that removes the need to always use an LLM for each task while still allowing tasks to depend on previous task results. They do so by permitting variable assignment in the planner's output. Below is a diagram of the agent design.https://blog.langchain.com/content/images/2024/02/rewoo.pngReWOO Agent

Its **planner** generates a plan list consisting of interleaving "Plan" (reasoning) and "E#" lines. As an example, given the user query "What are the stats for the quarterbacks of the super bowl contenders this year", the planner may generate the following plan:

```
Plan: I need to know the teams playing in the superbowl this year
E1: Search[Who is competing in the superbowl?]
Plan: I need to know the quarterbacks for each team
E2: LLM[Quarterback for the first team of #E1]
Plan: I need to know the quarterbacks for each team
E3: LLM[Quarter back for the second team of #E1]
Plan: I need to look up stats for the first quarterback
E4: Search[Stats for #E2]
Plan: I need to look up stats for the second quarterback
E5: Search[Stats for #E3]
```

Notice how the planner can reference previous outputs using syntax like `#E2` . This means it can execute a task list without having to re-plan every time.

The **worker** node loops through each task and assigns the task output to the corresponding variable. It also replaces variables with their results when calling subsequent calls.

Finally, the **Solver** integrates all these outputs into a final answer.

This agent design can be more effective than a naive plan-and-execute agent since each task can have only the required context (its input and variable values).

It still relies on sequential task execution, however, which can create a longer runtime.

## **LLMCompiler**https://blog.langchain.com/content/images/2024/02/llm-compiler-1.pngLLMCompiler Agent

The **LLMCompiler**, by [Kim, et. al.,](https://arxiv.org/abs/2312.04511?ref=blog.langchain.com) is an agent architecture designed to further increase the **speed** of task execution beyond the plan-and-execute and ReWOO agents described above, and even beyond OpenAI’s parallel tool calling.

The LLMCompiler has the following main components:

1. **Planner**: streams a DAG of tasks. Each task contains a tool, arguments, and list of dependencies.
2. **Task Fetching Unit** schedules and executes the tasks. This accepts a stream of tasks. This unit schedules tasks once their dependencies are met. Since many tools involve other calls to search engines or LLMs, the extra parallelism can grant a significant speed boost (the paper claims 3.6x).
3. **Joiner**: dynamically replan or finish based on the entire graph history (including task execution results) is an LLM step that decides whether to respond with the final answer or whether to pass the progress back to the (re-)planning agent to continue work.

The key runtime-boosting ideas here are:

- **Planner** outputs are **_streamed;_** the output parser eagerly yields task parameters and their dependencies.
- The **task fetching unit** receives the parsed task stream and schedules tasks once all their dependencies are satisfied.
- Task arguments can be _variables,_ which are the outputs of previous tasks in the DAG. For instance, the model can call `search("${1}")` to search for queries generated by the output of task 1. This lets the agent work even faster than the "embarrassingly parallel" tool calling in OpenAI.

By formatting tasks as a DAG, the agent can save precious time while invoking tools, leading to an overall better user experience.

## Conclusion

These three agent architectures are prototypical of the "plan-and-execute" design pattern, which separates an LLM-powered "planner" from the tool execution runtime. If your application requires multiple tool invocations or API calls, these types of approaches can reduce the time it takes to return a final result and help you save costs by reducing the frequency of calls to more powerful LLMs.

</details>

<details>
<summary>When building LLM Agent systems, choosing the right reasoning pattern is crucial. This article provides an in-depth comparison of two mainstream Agent reasoning patterns: ReAct (Reasoning and Acting) and Plan-and-Execute, helping you make informed technical decisions through practical cases.</summary>

When building LLM Agent systems, choosing the right reasoning pattern is crucial. This article provides an in-depth comparison of two mainstream Agent reasoning patterns: ReAct (Reasoning and Acting) and Plan-and-Execute, helping you make informed technical decisions through practical cases.

## 1\. Working Principles of Both Patterns

### 1.1 ReAct Pattern

ReAct (Reasoning and Acting) pattern is an iterative approach that alternates between thinking and acting. Its core workflow includes:

1.  **Reasoning**: Analyze current state and objectives
2.  **Acting**: Execute specific operations
3.  **Observation**: Obtain action results
4.  **Iteration**: Continue thinking and acting based on observations

### 1.2 Plan-and-Execute Pattern

Plan-and-Execute pattern adopts a "plan first, execute later" strategy, dividing tasks into two distinct phases:

1.  **Planning Phase**:
    -   Analyze task objectives
    -   Break down into subtasks
    -   Develop execution plan
2.  **Execution Phase**:
    -   Execute subtasks in sequence
    -   Process execution results
    -   Adjust plan if needed

## 5\. Selection Guide and Best Practices

### 5.1 When to Choose ReAct

1.  **Simple Direct Tasks**
    -   Single clear objective
    -   Few steps
    -   Quick response needed
2.  **Real-time Interactive Scenarios**
    -   Customer service dialogues
    -   Instant queries
    -   Simple calculations
3.  **Cost-Sensitive Scenarios**
    -   Limited token budget
    -   Need to control API calls

### 5.2 When to Choose Plan-and-Execute

1.  **Complex Multi-step Tasks**
    -   Requires task breakdown
    -   Step dependencies
    -   Intermediate result validation
2.  **High-Accuracy Scenarios**
    -   Financial analysis
    -   Data processing
    -   Report generation
3.  **Long-term Planning Tasks**
    -   Project planning
    -   Research analysis
    -   Strategic decisions

### 5.3 Best Practice Recommendations

1.  **Hybrid Usage Strategy**
    -   Choose patterns based on subtask complexity
    -   Combine both patterns in one system
2.  **Performance Optimization Tips**
    -   Implement caching mechanisms
    -   Enable parallel processing
    -   Optimize prompt templates
3.  **Cost Control Methods**
    -   Set token limits
    -   Implement task interruption
    -   Use result caching

## Conclusion

Both ReAct and Plan-and-Execute have their strengths, and the choice between them should consider task characteristics, performance requirements, and cost constraints. In practical applications, you can flexibly choose or even combine both patterns to achieve optimal results.

</details>

<details>
<summary>Self-Evaluation in AI Agents: Enhancing Performance Through Reasoning and Reflection</summary>

# Self-Evaluation in AI Agents: Enhancing Performance Through Reasoning and Reflection

In an era where over 80% of AI projects fail—double the rate of traditional IT initiatives—self-evaluation in AI agents has emerged as a critical differentiator for successful AI systems.

With self-evaluation in AI agents, reliability is enhanced, and the supervision requirements that typically undermine enterprise AI initiatives are reduced.

This article explores three fundamental components of AI self-evaluation: Chain of Thought (CoT) analysis for transparent reasoning, error identification mechanisms for early mistake detection, and self-reflection techniques enabling continuous improvement.

## What is Chain of Thought (CoT) in AI Agent Self-Evaluation?

Chain of Thought (CoT) is a technique that enables AI systems to explicitly break down their reasoning process into a sequence of intermediate steps before arriving at a final answer.

In AI agent self-evaluation, CoT serves as a mechanism for the agent to track, analyze, and evaluate its own decision-making process. By making reasoning transparent, the agent can identify where potential errors might occur and improve its problem-solving approach.In contrast to traditional "black box" AI models which provide answers without explaining their reasoning process, CoT transforms these systems to transparent decision-making by emulating human reasoning patterns.

The power of CoT lies in its dual enhancement of performance and transparency. CoT enables multi-step reasoning that significantly improves accuracy and reliability in complex tasks. This capability is particularly valuable for AI self-evaluation, as it allows systems to examine their own reasoning chains, identify potential flaws, and implement mechanisms for error detection.

### Implementing Effective CoT for Self-Evaluation

Implementing Chain of Thought (CoT) analysis requires strategic prompt engineering, the use of agentic AI frameworks, and careful pattern structuring. The most effective implementation begins with clear instructions that explicitly direct the model to "think step by step" before providing a final answer.

Three primary approaches exist for CoT implementation: zero-shot, few-shot, and fine-tuned methods:

For maximum effectiveness, structure your CoT prompts using consistent formats, such as numbered steps or bulleted lists. These structured formats create natural checkpoints where the model can assess the validity of each reasoning step before proceeding.

When designing CoT prompt templates, include explicit evaluation criteria at each step to improve the detection of logical errors. For example, implementing a reasoner-verifier architecture where the reasoning component generates intermediary steps, and the verification component validates each step increases accuracy on complex reasoning tasks.

For self-evaluation applications, implement dual-pass reasoning, where the model first produces a complete reasoning chain and then separately evaluates each step with specific verification criteria. This separation of generation and evaluation prevents the model from becoming anchored to its initial reasoning path.

For complex domain-specific applications, implement a modular CoT approach where specialized reasoning modules handle different aspects of the problem. For example, in financial analysis, separate modules might handle numerical calculations, regulatory compliance checking, and market trend analysis, with their outputs combined through a meta-reasoning layer that integrates these specialized chains.

### Measuring Chain of Thought (CoT) Effectiveness

Measuring the effectiveness of Chain of Thought (CoT) analysis requires a multidimensional approach focusing on coherence, factuality, and reasoning depth. These qualitative assessments provide insight into not just whether the model reached the correct conclusion but also how soundly it reasoned its way there.

Rigorous evaluation of CoT should include both intrinsic and extrinsic metrics. Intrinsic metrics assess the quality of the reasoning process itself, while extrinsic metrics measure the impact on downstream task performance. According to research on Chain-of-Thought, specific intrinsic evaluation metrics for measuring CoT’s effectiveness include:

For extrinsic evaluation, technical benchmarks like the BIG-Bench Hard and MATH datasets provide standardized task batteries where improvement from baseline to CoT performance can be precisely quantified.

Additionally, utilizing AI agent performance metrics and chatbot performance metrics can provide insights into the effectiveness of CoT in conversational AI agents.

As demonstrated with OpenAI's o-model series, models that employ sophisticated CoT capabilities rank in the 89th percentile on competitive programming questions and exceed PhD-level accuracy on physics, biology, and chemistry problems. However, CoT approaches require significantly more computational resources than standard prompting techniques, and there's always a risk of generating plausible yet incorrect reasoning paths.

## Error Identification Mechanisms in AI Agents' Self-Evaluation

Error identification mechanisms are systematic processes and algorithms that enable AI agents to detect, categorize, and flag potential mistakes in their reasoning or outputs.

These mechanisms serve as quality control systems that operate in real time during the agent's functioning. They provide an internal verification layer that monitors for inconsistencies, implausibilities, or outright errors before delivering results to users.

Effective error identification mechanisms operate across multiple dimensions of AI agents’ output, checking for:

In practical applications, these mechanisms act as cognitive guardrails that prevent AI agents from confidently presenting incorrect information, helping to maintain system reliability even when faced with novel or complex scenarios.

### Implementing Error Identification Mechanisms

Teams should focus on building a layered detection approach that combines multiple verification strategies. Start by integrating a self-consistency framework where the agent generates multiple reasoning attempts for complex problems. This approach allows the system to compare results across different reasoning paths, flagging inconsistencies that might indicate errors.

For factual verification, a retrieval-augmented architecture that automatically cross-references generated claims against trusted knowledge sources should be implemented. This requires creating an indexing system for reference information and embedding similarity search functionality to validate generated content. When implementing such systems, prioritize high-precision verification for critical domains like medical, financial, or legal applications.

Another promising research-backed approach to implementing error identification is the self-consistency method. This method generates multiple potential reasoning processes for a given problem and then evaluates the consistency across these paths to identify the most reliable answer. This approach has demonstrated significant improvements in reasoning accuracy across multiple benchmarks.

Hallucination detection can be strengthened by monitoring the model's internal probability distributions during generation. Implement entropy tracking that alerts when token probabilities exhibit unexpected patterns that correlate with confabulation.

For mathematical reasoning, build specialized verification modules that can parse equations, re-compute calculations independently, and validate numerical results. This approach has proven particularly effective for finance, scientific, and engineering applications where numerical accuracy is paramount.

### Measuring Error Identification Effectiveness

Robust measurement frameworks are essential for evaluating how well error identification mechanisms perform. Begin by establishing baseline error rates for your system without any detection mechanisms in place, then compare against performance with various detection approaches enabled.

Track false positive and false negative rates across different error categories. For critical applications, prioritize minimizing false negatives (undetected errors) even at the cost of some false positives. Also, the precision-recall tradeoff for error detection should be monitored to find the optimal balance for specific use cases.

For hallucination detection, measure the system's ability to identify fabricated content and its skill at providing appropriate uncertainty indicators when information is limited. Effective systems should demonstrate high detection rates on challenging "adversarial" questions that provoke hallucinations.

When measuring the performance of self-consistency approaches, track the accuracy improvements and the "agreement rate" among different reasoning paths. Lower agreement typically indicates problems that require more sophisticated verification.

In complex reasoning tasks, measure the granularity of error detection—systems should identify not just that an error occurred, but precisely where in the reasoning chain it emerged.

In practical applications like fraud detection, measure the economic impact of improved error detection. For example, MIT researchers demonstrated systems that reduced false positives by 54% through improved error detection algorithms, potentially saving financial institutions hundreds of thousands of euros annually through more precise detection mechanisms.

## Self-Reflection in Modern Language Models and AI Agents

Self-reflection in AI agents is the capability of AI agents to critically analyze their own outputs, reasoning processes, and decision-making pathways. This metacognitive ability enables AI agents to evaluate the quality of their answers, recognize limitations in their understanding, identify potential errors, and iteratively improve their performance without external correction.

Similar to human metacognition, AI self-reflection creates an internal feedback loop where the agent actively questions its own conclusions, considers alternative perspectives, and refines its approach based on this introspective analysis.

### Implementing Self-Reflection Systems

To implement effective self-reflection in AI agents, begin by designing multi-stage reasoning processes in which the agent generates an initial response and then enters a dedicated reflection phase to critically assess that response. This two-stage approach creates a clear separation between generation and evaluation.

During the reflection phase, program the AI agent to analyze specific aspects of its response using a comprehensive rubric. This rubric should guide the agent to:

The reflection process works best when framed as specific questions the agent must answer about its own work rather than vague instructions to "reflect."

For complex problem-solving tasks, implement feedback loops, which are systematic mechanisms that enable AI systems to incorporate evaluation signals back into their operation, creating a continuous improvement cycle that enhances self-reflection capabilities.

Feedback loops for self-reflection require carefully designed architectures to capture, analyze, and integrate evaluation signals effectively. The most effective implementations typically involve a three-stage pipeline:

Research from Anthropic on Constitutional AI demonstrates that models trained with feedback loops that critique their own outputs show substantial improvements in factuality while maintaining performance. Enterprise implementations can achieve similar results without retraining by implementing inference-time feedback loops that maintain state across interactions.

### Measuring Self-Reflection Effectiveness

To rigorously assess self-reflection capabilities, develop evaluation frameworks that measure both the agent's ability to detect its own errors and its capacity to make substantive improvements based on those insights.

Begin by tracking the " correction rate"—the percentage of initially incorrect responses that the agent successfully identifies and fixes through self-reflection. This metric provides a direct measure of reflection effectiveness. For sophisticated applications, disaggregate this measurement across different error types to identify specific reflection strengths and weaknesses.

Measure the quality of improvements by comparing the agent's initial responses against its post-reflection answers using established benchmarks. Beyond accuracy, assess the depth of reflection by evaluating how thoroughly the agent analyzes its own outputs.

Superficial reflections might only identify surface errors, while more profound reflections examine underlying reasoning patterns and assumptions. Develop rubrics that categorize reflection quality from basic fact-checking to sophisticated metacognitive analysis.

In conversational applications, track how reflection impacts user satisfaction and trust. Effective self-reflection should reduce the need for users to correct the agent, leading to smoother interactions and higher completion rates for complex tasks.

Additionally, monitor AI agent metrics like task completion time, user correction frequency, and explicit satisfaction ratings to gauge the real-world impact of reflection capabilities.

For advanced implementations, measure the reduced human reviewer workload resulting from improved self-reflection. Systems with strong self-evaluation capabilities typically require significantly less oversight, creating operational efficiencies that can be quantified through reduced quality assurance costs.

</details>


## Code Sources

_No code sources found._


## YouTube Video Transcripts

_No YouTube video transcripts found._


## Additional Sources Scraped

<details>
<summary>ai-agents-in-2025-expectations-vs-reality-ibm</summary>

An AI agent is a software program capable of acting autonomously to understand, plan and execute tasks. AI agents are powered by LLMs and can interface with tools, other models and other aspects of a system or network as needed to fulfill user goals.

We’re going beyond asking a chatbot to suggest a dinner recipe based on the available ingredients in the fridge. Agents are more than automated customer experience emails that inform you it’ll be a few days until a real-world human can get to your inquiry.

[AI agents differ from traditional AI assistants](https://www.ibm.com/think/topics/ai-agents-vs-ai-assistants) that need a prompt each time they generate a response. In theory, a user gives an agent a high-level task, and the agent figures out how to complete it.

Current offerings are still in the early stages of approaching this idea. “What’s commonly referred to as ‘agents’ in the market is the addition of rudimentary planning and tool-calling (sometimes called function calling) capabilities to LLMs,” says Ashoori. “These enable the LLM to break down complex tasks into smaller steps that the LLM can perform.”

Hay is optimistic that more robust agents are on a way: “You wouldn’t need any further progression in models today to build [future AI agents](https://www.ibm.com/think/insights/ai-agents-evolve-rapidly),” he says.

</details>

<details>
<summary>arxiv-org</summary>

While large language models (LLMs) have demonstrated impressive performance across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with and gather additional information from external sources such as knowledge bases or environments. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines in addition to improved human interpretability and trustworthiness. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes prevalent issues of hallucination and error propagation in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generating human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. Furthermore, on two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of $34 %$ and $10 %$ respectively, while being prompted with only one or two in-context examples.

# 1 INTRODUCTION

A unique feature of human intelligence is the ability to seamlessly combine task-oriented actions with verbal reasoning (or inner speech, Alderson-Day & Fernyhough, 2015), which has been theorized to play an important role in human cognition for enabling self-regulation or strategization (Vygotsky, 1987; Luria, 1965; Fernyhough, 2010) and maintaining a working memory (Baddeley, 1992). Consider the example of cooking up a dish in the kitchen. Between any two specific actions, we may reason in language in order to track progress (“now that everything is cut, I should heat up the pot of water”), to handle exceptions or adjust the plan according to the situation (“I don’t have salt, so let me use soy sauce and pepper instead”), and to realize when external information is needed (“how do I prepare dough? Let me search on the Internet”). We may also act (open a cookbook to read the recipe, open the fridge, check ingredients) to support the reasoning and to answer questions (“What dish can I make right now?”). This tight synergy between “acting” and “reasoning” allows humans to learn new tasks quickly and perform robust decision making or reasoning, even under previously unseen circumstances or facing information uncertainties.

Recent results have hinted at the possibility of combining verbal reasoning with interactive decision making in autonomous systems. On one hand, properly prompted large language models (LLMs) have demonstrated emergent capabilities to carry out several steps of reasoning traces to derive answers from questions in arithmetic, commonsense, and symbolic reasoning tasks (Wei et al., 2022). However, this “chain-of-thought” reasoning is a static black box, in that the model uses its own internal representations to generate thoughts and is not grounded in the external world, which limits its ability to reason reactively or update its knowledge. This can lead to issues like fact hallucination and error propagation over the reasoning process (Figure 1 (1b)). On the other hand, recent work has explored the use of pre-trained language models for planning and acting in interactive environments (Ahn et al., 2022; Nakano et al., 2021; Yao et al., 2020; Huang et al., 2022a), with a focus on predicting actions via language priors. These approaches usually convert multi-modal observations into text, use a language model to generate domain-specific actions or plans, and then use a controller to choose or execute them. However, they do not employ language models to reason abstractly about high-level goals or maintain a working memory to support acting, barring Huang et al. (2022b) who perform a limited form of verbal reasoning to reiterate spatial facts about the current state. Beyond such simple embodied tasks to interact with a few blocks, there have not been studies on how reasoning and acting can be combined in a synergistic manner for general task solving, and if such a combination can bring systematic benefits compared to reasoning or acting alone.https://arxiv.org/pdf/images/dd0e9f64b42d2cab71cdcecddd80ea2cf5aa212b5bf9a21882834d8e50a5302d.jpg

Figure 1: (1) Comparison of 4 prompting methods, (a) Standard, (b) Chain-of-thought (CoT, Reason Only), (c) Act-only, and (d) ReAct (Reason+Act), solving a HotpotQA (Yang et al., 2018) question. (2) Comparison of (a) Act-only and (b) ReAct prompting to solve an AlfWorld (Shridhar et al., 2020b) game. In both domains, we omit in-context examples in the prompt, and only show task solving trajectories generated by the model (Act, Thought) and the environment (Obs).

In this work, we present ReAct, a general paradigm to combine reasoning and acting with language models for solving diverse language reasoning and decision making tasks (Figure 1). ReAct prompts LLMs to generate both verbal reasoning traces and actions pertaining to a task in an interleaved manner, which allows the model to perform dynamic reasoning to create, maintain, and adjust high-level plans for acting (reason to act), while also interact with the external environments (e.g. Wikipedia) to incorporate additional information into reasoning (act to reason).

We conduct empirical evaluations of ReAct and state-of-the-art baselines on four diverse benchmarks: question answering (HotPotQA, Yang et al., 2018), fact verification (Fever, Thorne et al., 2018), text-based game (ALFWorld, Shridhar et al., 2020b), and webpage navigation (WebShop, Yao et al., 2022). For HotPotQA and Fever, with access to a Wikipedia API that the model can interact with, ReAct outperforms vanilla action generation models while being competitive with chain-ofthought reasoning (CoT) (Wei et al., 2022). The best approach overall is a combination of ReAct and CoT that allows for the use of both internal knowledge and externally obtained information during reasoning. On ALFWorld and WebShop, two or even one-shot ReAct prompting is able to outperform imitation or reinforcement learning methods trained with $1 0 ^ { 3 } \\sim \\mathrm { { \\bar { 1 0 } } ^ { 5 } }$ task instances, with an absolute improvement of $34 %$ and $10 %$ in success rates respectively. We also demonstrate the importance of sparse, versatile reasoning in decision making by showing consistent advantages over controlled baselines with actions only. Besides general applicability and performance boost, the combination of reasoning and acting also contributes to model interpretability, trustworthiness, and diagnosability across all domains, as humans can readily distinguish information from model’s internal knowledge versus external environments, as well as inspect reasoning traces to understand the decision basis of model actions.

To summarize, our key contributions are the following: (1) we introduce ReAct, a novel promptbased paradigm to synergize reasoning and acting in language models for general task solving; (2) we perform extensive experiments across diverse benchmarks to showcase the advantage of ReAct in a few-shot learning setup over prior approaches that perform either reasoning or action generation in isolation; (3) we present systematic ablations and analysis to understand the importance of acting in reasoning tasks, and reasoning in interactive tasks; (4) we analyze the limitations of ReAct under the prompting setup (i.e. limited support of reasoning and acting behaviors), and perform initial finetuning experiments showing the potential of ReAct to improve with additional training data. Scaling up ReAct to train and operate on more tasks and combining it with complementary paradigms like reinforcement learning could further unlock the potential of large language models.

# 2 REAC T: SYNERGIZING REASONING + ACTING

Consider a general setup of an agent interacting with an environment for task solving. At time step $t$ , an agent receives an observation $o \_ { t } \\in \\mathcal { O }$ from the environment and takes an action $a \_ { t } \\in \\mathcal A$ following some policy $\\pi ( \\boldsymbol { a } \_ { t } \| \\boldsymbol { c } \_ { t } )$ , where $c \_ { t } = \\left( o \_ { 1 } , a \_ { 1 } , \\cdot \\cdot \\cdot , o \_ { t - 1 } , a \_ { t - 1 } , o \_ { t } \\right)$ is the context to the agent. Learning a policy is challenging when the mapping $c \_ { t } \\mapsto a \_ { t }$ is highly implicit and requires extensive computation. For example, the agent shown in Figure 1(1c) is unable to generate the correct final action (Act 4) to finish the QA task as it requires complex reasoning over the trajectory context (Question, Act 1-3, Obs 1-3). Similarly, the agent shown in Figure 1(2a) fails to comprehend from the context that sinkbasin 1 does not contain peppershaker 1, thus keep producing hallucinating actions.

The idea of ReAct is simple: we augment the agent’s action space to ${ \\hat { \\mathcal { A } } } = { \\mathcal { A } } \\cup { \\mathcal { L } }$ , where $\\mathcal { L }$ is the space of language. An action $\\hat { a } \_ { t } \\in \\mathcal { L }$ in the language space, which we will refer to as a thought or a reasoning trace, does not affect the external environment, thus leading to no observation feedback. Instead, a thought $\\hat { a } \_ { t }$ aims to compose useful information by reasoning over the current context $c \_ { t }$ , and update the context $\\boldsymbol c \_ { t + 1 } = \\left( c \_ { t } , \\hat { a } \_ { t } \\right)$ to support future reasoning or acting. As shown in Figure 1, there could be various types of useful thoughts, e.g. decomposing task goals and create action plans (2b, Act 1; 1d, Thought 1), injecting commonsense knowledge relevant to task solving (2b, Act 1), extracting important parts from observations (1d, Thought2, 4), track progress and transit action plans (2b, Act 8), handle exceptions and adjust action plans (1d, Thought 3), and so on.

However, as the language space $\\mathcal { L }$ is unlimited, learning in this augmented action space is difficult and requires strong language priors. In this paper, we mainly focus on the setup where a frozen large language model, PaLM-540B (Chowdhery et al., 2022)1, is prompted with few-shot in-context examples to generate both domain-specific actions and free-form language thoughts for task solving (Figure 1 (1d), (2b)). Each in-context example is a human trajectory of actions, thoughts, and environment observations to solve a task instance (see Appendix C). For the tasks where reasoning is of primary importance (Figure 1(1)), we alternate the generation of thoughts and actions so that the task-solving trajectory consists of multiple thought-action-observation steps. In contrast, for decision making tasks that potentially involve a large number of actions (Figure 1(2)), thoughts only need to appear sparsely in the most relevant positions of a trajectory, so we let the language model decide the asynchronous occurrence of thoughts and actions for itself.

Since decision making and reasoning capabilities are integrated into a large language model, ReAct enjoys several unique features: A) Intuitive and easy to design: Designing ReAct prompts is straightforward as human annotators just type down their thoughts in language on top of their actions taken. No ad-hoc format choice, thought design, or example selection is used in this paper. We detail prompt design for each task in Sections 3 and 4. B) General and flexible: Due to the flexible thought space and thought-action occurrence format, ReAct works for diverse tasks with distinct action spaces and reasoning needs, including but not limited to QA, fact verification, text game, and web navigation. C) Performant and robust: ReAct shows strong generalization to new task instances while learning solely from one to six in-context examples, consistently outperforming baselines with only reasoning or acting across different domains. We also show in Section 3 additional benefits when finetuning is enabled, and in Section 4 how ReAct performance is robust to prompt selections. D) Human aligned and controllable: ReAct promises an interpretable sequential decision making and reasoning process where humans can easily inspect reasoning and factual correctness. Moreover, humans can also control or correct the agent behavior on the go by thought editing, as shown in Figure 5 in Section 4.

# 3 KNOWLEDGE-INTENSIVE REASONING TASKS

We begin with knowledge-intensive reasoning tasks like multi-hop question answering and fact verification. As shown in Figure 1(1d), by interacting with a Wikipedia API, ReAct is able to retrieve information to support reasoning, while also use reasoning to target what to retrieve next, demonstrating a synergy of reasoning and acting.

# 3.1 SETUP

Domains We consider two datasets challenging knowledge retrieval and reasoning: (1) HotPotQA (Yang et al., 2018), a multi-hop question answering benchmark that requires reasoning over two or more Wikipedia passages, and (2) FEVER (Thorne et al., 2018), a fact verification benchmark where each claim is annotated SUPPORTS, REFUTES, or NOT ENOUGH INFO, based on if there exists a Wikipedia passage to verify the claim. In this work, we operate in a question-only setup for both tasks, where models only receive the question/claim as input without access to support paragraphs, and have to rely on their internal knowledge or retrieve knowledge via interacting with an external environment to support reasoning.

Action Space We design a simple Wikipedia web API with three types of actions to support interactive information retrieval: (1) search\[entity\], which returns the first 5 sentences from the corresponding entity wiki page if it exists, or else suggests top-5 similar entities from the Wikipedia search engine, (2) lookup\[string\], which would return the next sentence in the page containing string, simulating $\\mathrm { C t r l + F }$ functionality on the browser. (3) finish\[answer\], which would finish the current task with answer. We note that this action space mostly can only retrieve a small part of a passage based on exact passage name, which is significantly weaker than state-of-theart lexical or neural retrievers. The purpose is to simulate how humans would interact with Wikipedia, and force models to retrieve via explicit reasoning in language.

# 3.2 METHODS

ReAct Prompting For HotpotQA and Fever, we randomly select 6 and 3 cases2 from the training set and manually compose ReAct-format trajectories to use as few-shot exemplars in the prompts. Similar to Figure 1(d), each trajectory consists of multiple thought-action-observation steps (i.e. dense thought), where free-form thoughts are used for various purposes. Specifically, we use a combination of thoughts that decompose questions (“I need to search x, find y, then find $\\boldsymbol { z } ^ { \\flat }$ ), extract information from Wikipedia observations $\\mathbf { \\epsilon } ^ { \* } \\mathbf { \\epsilon } \_ { \\mathbf { X } }$ was started in $1 8 4 4 ^ { , 9 }$ , “The paragraph does not tell x”), perform commonsense ( ${ } ^ { \* \* } \\mathbf { X }$ is not y, so z must instead be...”) or arithmetic reasoning $^ { \\cdot \\cdot } 1 8 4 4 < 1 9 8 9 ^ { , 3 } ,$ ), guide search reformulation (“maybe I can search/look up x instead”), and synthesize the final answer (“...so the answer is x”). See Appendix C for more details.

Table 1: PaLM-540B prompting results on HotpotQA and Fever.

|     |     |     |
| --- | --- | --- |
| PromptMethoda | HotpotQA (EM) | Fever (Acc) |
| Standard CoT(Wei et al., 2022) | 28.7 29.4 | 57.1 |
| CoT-SC (Wang et al.,2022a) | 33.4 | 56.3 60.4 |
| Act ReAct | 25.7 | 58.9 |
| CoT-SC→ReAct | 27.4 34.2 | 60.9 64.6 |
| ReAct→CoT-SC | 35.1 | 62.0 |
| Supervised SoTAb | 67.5 | 89.5 |

HotpotQA EM is 27.1, 28.9, 33.8 for Standard, CoT, CoT-SC in Wang et al. (2022b). b (Zhu et al., 2021; Lewis et al., 2020)https://arxiv.org/pdf/images/c95596879744d747588f90509051b529103be1dcd083cb7d3c75f8c7b95364a6.jpg

Figure 2: PaLM-540B prompting results with respect to number of CoT-SC samples used.

Baselines We systematically ablate ReAct trajectories to build prompts for multiple baselines (with formats as Figure 1(1a-1c)): (a) Standard prompting (Standard), which removes all thoughts, actions, observations in ReAct trajectories. (b) Chain-of-thought prompting (CoT) (Wei et al., 2022), which removes actions and observations and serve as a reasoning-only baseline. We also build a self-consistency baseline (CoT-SC) (Wang et al., 2022a;b) by sampling $2 1 \\thinspace \\mathrm { C o T }$ trajectories with decoding temperature 0.7 during inference and adopting the majority answer, which is found to consistently boost performance over CoT. (c) Acting-only prompt (Act), which removes thoughts in ReAct trajectories, loosely resembling how WebGPT (Nakano et al., 2021) interacts with the Internet to answer questions, though it operates on a different task and action space, and uses imitation and reinforcement learning instead of prompting.

Combining Internal and External Knowledge As will be detail in Section 3.3, we observe that the problem solving process demonstrated by ReAct is more factual and grounded, whereas CoT is more accurate in formulating reasoning structure but can easily suffer from hallucinated facts or thoughts. We therefore propose to incorporate ReAct and $\\mathrm { C o T - S C }$ , and let the model decide when to switch to the other method based on the following heuristics: A) $\\mathtt { R e a c t \\to c o T - S c }$ : when ReAct fails to return an answer within given steps, back off to $\\mathrm { C o T - S C }$ . We set 7 and 5 steps for HotpotQA and FEVER respectively as we find more steps will not improve ReAct performance3. B) Co $\\mathtt { r } { - } \\mathtt { S C } \\to \\mathtt { R e A c t }$ : when the majority answer among $n \ C o \\mathrm { T } { - } S \\mathrm C$ samples occurs less than $n / 2$ times (i.e. internal knowledge might not support the task confidently), back off to ReAct.

Finetuning Due to the challenge of manually annotating reasoning traces and actions at scale, we consider a bootstraping approach similar to Zelikman et al. (2022), using 3,000 trajectories with correct answers generated by ReAct (also for other baselines) to finetune smaller language models (PaLM-8/62B) to decode trajectories (all thoughts, actions, observations) conditioned on input questions/claims. More details are in Appendix B.1.

# 3.3 RESULTS AND OBSERVATIONS

ReAct outperforms Act consistently Table 1 shows HotpotQA and Fever results using PaLM540B as the base model with different prompting methods. We note that ReAct is better than Act on both tasks, demonstrating the value of reasoning to guide acting, especially for synthesizing the final answer, as shown in Figure 1 (1c-d). Fine-tuning results 3 also confirm the benefit of reasoning traces for more informed acting.

Table 2: Types of success and failure modes of ReAct and CoT on HotpotQA, as well as their percentages in randomly selected examples studied by human.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| 一 | Type | Definition | ReAct | CoT |
| Success | True positive | Correct reasoning trace and facts | 94% | 86% |
| False positive | Hallucinated reasoning trace or facts | 6% | 14% |
| Failure | Reasoning error | Wrong reasoning trace (including failing to recover from repetitive steps) | 47% | 16% |
| Search result error | Search return empty or does not contain useful information | 23% | 1 |
| Hallucination | Hallucinated reasoning trace or facts | 0% | 56% |
| Labelambiguity | Right prediction but did not match the label precisely | 29% | 28% |

ReAct vs. CoT On the other hand, ReAct outperforms CoT on Fever (60.9 vs. 56.3) and slightly lags behind CoT on HotpotQA (27.4 vs. 29.4). Fever claims for SUPPORTS/REFUTES might only differ by a slight amount (see Appendix D.1), so acting to retrieve accurate and up-to-date knowledge is vital. To better understand the behavioral difference between ReAct and CoT on HotpotQA, we randomly sampled 50 trajectories with correct and incorrect answers (judged by EM) from ReAct and CoT respectively (thus 200 examples in total), and manually labeled their success and failure modes in Table 2. Some key observations are as follows:

A) Hallucination is a serious problem for CoT, resulting in much higher false positive rate than ReAct $14 %$ vs. $6 %$ ) in success mode, and make up its major failure mode $( 5 6 % )$ . In contrast, the problem solving trajectory of $\\scriptstyle { \\mathrm { R e A c t } }$ is more grounded, fact-driven, and trustworthy, thanks to the access of an external knowledge base.

B) While interleaving reasoning, action and observation steps improves ReAct’s groundedness and trustworthiness, such a structural constraint also reduces its flexibility in formulating reasoning steps, leading to more reasoning error rate than CoT. we note that there is one frequent error pattern specific to ReAct, in which the model repetitively generates the previous thoughts and actions, and we categorize it as part of “reasoning error” as the model fails to reason about what the proper next action to take and jump out of the $\\mathrm { l o o p ^ { 4 } }$ .

C) For ReAct, successfully retrieving informative knowledge via search is critical. Noninformative search, which counts for $23 %$ of the error cases, derails the model reasoning and gives it a hard time to recover and reformulate thoughts. This is perhaps an expected trade-off between factuality and flexibility, which motivates our proposed strategies of combining two methods.

We provide examples for each success and failure modes in Appendix E.1. We also find some HotpotQA questions may contain outdated answer labels, see Figure 4 for example.

ReAct $^ +$ CoT-SC perform best for prompting LLMs Also shown in Table 1, the best prompting method on HotpotQA and Fever are $\\mathtt { R e A c t } \\to \\mathtt { C o T } \\mathrm { - } \\mathrm { S C }$ and $\\mathtt { C o T - S C } \\to \\mathtt { R e A c t }$ respectively. Furthermore, Figure 2 shows how different methods perform with respect to the number of $\\mathrm { C o T - S C }$ samples used. While two $\\mathtt { R e A c t + C o T - S C }$ methods are advantageous at one task each, they both significantly and consistently outperform CoT-SC across different number of samples, reaching $\\mathrm { C o T - S C }$ performance with 21 samples using merely 3-5 samples. These results indicate the value of properly combining model internal knowledge and external knowledge for reasoning tasks.

ReAct performs best for fine-tuning Figure 3 shows the scaling effect of prompting/finetuning four methods (Standard, CoT, Act, ReAct) on HotpotQA. With PaLM-8/62B, prompting ReAct performs worst among four methods due to the difficulty to learn both reasoning and acting from in-context examples. However, when finetuned with just 3,000 examples, ReAct becomes the best method among the four, with PaLM-8B finetuned ReAct outperforming all PaLM-62B prompting methods, and PaLM-62B finetuned ReAct outperforming all 540B prompting methods. In contrast, finetuning Standard or CoT is significantly worse than finetuning ReAct or Act for both PaLM8/62B, as the former essentially teaches models to memorize (potentially halluincated) knowledge facts, and the latter teaches models how to (reason and) act to access information from Wikipedia, a more generalizable skill for knowledge reasoning. As all prompting methods are still significantly far from domain-specific state-of-the-art approaches (Table 1), we believe finetuning with more human-written data might be a better way to unleash the power of ReAct.https://arxiv.org/pdf/images/292146198530f128c8ca2240dd31347b2f48943744b89bd485f9ece112470687.jpg

Figure 3: Scaling results for prompting and finetuning on HotPotQA with ReAct (ours) and baselines.

# 4 DECISION MAKING TASKS

We also test ReAct on two language-based interactive decision-making tasks, ALFWorld and WebShop, both of which feature complex environments that require agents to act over long horizons with sparse rewards, warranting the need for reasoning to act and explore effectively.

ALFWorld ALFWorld (Shridhar et al., 2020b) (Figure 1(2)) is a synthetic text-based game designed to align with the embodied ALFRED benchmark (Shridhar et al., 2020a). It includes 6 types of tasks in which an agent needs to achieve a high-level goal (e.g. examine paper under desklamp) by navigating and interacting with a simulated household via text actions (e.g. go to coffeetable 1, take paper 2, use desklamp 1). A task instance can have more than 50 locations and take an expert policy more than 50 steps to solve, thus challenging an agent to plan and track subgoals, as well as explore systematically (e.g. check all desks one by one for desklamp). In particular, one challenge built into ALFWorld is the need to determine likely locations for common household items (e.g. desklamps will likely be on desks, shelfs, or dressers), making this environment a good fit for LLMs to exploit their pretrained commonsense knowledge. To prompt ReAct, we randomly annotate three trajectories from the training set for each task type, where each trajectory includes sparse thoughts that (1) decompose the goal, (2) track subgoal completion, (3) determine the next subgoal, and (4) reason via commonsense where to find an object and what to do with it. We show prompts used for ALFWorld in Appendix C.4. Following Shridhar et al. (2020b), we evaluate on 134 unseen evaluation games in a task-specific setup. For robustness, we construct 6 prompts for each task type through each permutation of 2 annotated trajectories from the 3 we annotate. Act prompts are constructed using the same trajectories, but without thoughts — since task instances are randomly chosen from the training set, it favors neither $\\scriptstyle \\mathrm { R e A c t }$ nor Act and provides a fair and controlled comparison to test the importance of sparse thoughts. For baselines, we use BUTLER (Shridhar et al., 2020b), an imitation learning agent trained on $\\mathrm { \\bar { 1 0 ^ { 5 } } }$ expert trajectories for each task type5.

WebShop Can ReAct also interact with noisy real-world language environments for practical applications? We investigate WebShop (Yao et al., 2022), a recently proposed online shopping website environment with 1.18M real-world products and $1 2 \\mathrm { k }$ human instructions. Unlike ALFWorld, Webshop contains a high variety of structured and unstructured texts (e.g. product titles, descriptions, and options crawled from Amazon), and requires an agent to purchase a product based on a user instruction (e.g. “I am looking for a nightstand with drawers. It should have a nickel finish, and priced lower than $$ 140$ ) through web interactions (e.g. search “nightstand drawers”, choose buttons such as “color: modern-nickel-white” or “back to search”). This task is evaluated by average score (percentage of desired attributes covered by the chosen product averaged across all episodes) and success rate (percentage of episodes where the chosen product satisfies all requirements) on 500 test instructions. We formulate Act prompts with actions to search, choose product, choose options, and buy, with ReAct prompts additionally reasoning to determine what to explore, when to buy, and what products options are relevant to the instruction. See Table 6 for an example prompt, and Table 10 for model predictions in the Appendix. We compare to an imitation learning (IL) method trained with 1,012 human annotated trajectories, and a imitation $+$ reinforcement learning $( \\mathrm { I L } + \\mathrm { R L } )$ method additionally trained with 10,587 training instructions.

|     |     |     |
| --- | --- | --- |
| Method | Score | SR |
| Act ReAct | 62.3 66.6 | 30.1 40.0 |
| IL | 59.9 | 29.1 |
| IL+RL | 62.4 | 28.7 |
| Human Expert | 82.1 | 59.6 |

Table 3: AlfWorld task-specific success rates $( % )$ . BUTLER and $\\mathbf { B U T L E R } \_ { g }$ results are from Table 4 of Shridhar et al. (2020b). All methods use greedy decoding, except that BUTLER uses beam search.

|     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Pick | Clean | Heat | Cool | Look | Pick 2 | All |
| Act (best of 6) | 88 | 42 | 74 | 67 | 72 | 41 | 45 |
| ReAct (avg) | 65 | 39 | 83 | 76 | 55 | 24 | 57 |
| ReAct (best of 6) | 92 | 58 | 96 | 86 | 78 | 41 | 71 |
| ReAct-IM (avg) | 55 | 59 | 60 | 55 | 23 | 24 | 48 |
| ReAct-IM (best of 6) | 62 | 68 | 87 | 57 | 39 | 33 | 53 |
| BUTLERg (bestof 8) | 33 | 26 | 70 | 76 | 17 | 12 | 22 |
| BUTLER (best of 8) | 46 | 39 | 74 | 100 | 22 | 24 | 37 |

Table 4: Score and success rate (SR) on Webshop. $\\mathrm { I L / I L + R L }$ taken from Yao et al. (2022).

Results ReAct outperforms Act on both ALFWorld (Table 3) and Webshop (Table 4). On ALFWorld, the best ReAct trial achieves an average success rate of $71 %$ , significantly outperforming the best Act $( 4 5 % )$ and BUTLER $( 3 7 % )$ trials. In fact, even the worse ReAct trial $( 4 8 % )$ beats the best trial of both methods. Moreover, the advantage of $\\scriptstyle { \\mathrm { R e A c t } }$ over Act is consistent across six controlled trials, with relative performance gain ranging from $33 %$ to $90 %$ and averaging $62 %$ . Qualitatively, we saw that, without any thoughts at all, Act fails to correctly decompose goals into smaller subgoals, or loses track of the current state of the environment. Example trajectories comparing ReAct and Act can be found in Appendix D.2.1 and Appendix D.2.2.

On Webshop, one-shot Act prompting already performs on par with IL and $\\scriptstyle \\mathrm { I L + R L }$ methods. With additional sparse reasoning, ReAct achieves significantly better performance, with an absolute $10 %$ improvement over the previous best success rate. By checking examples, we find that ReAct is more likely to identify instruction-relevant products and options by reasoning to bridge the gap between noisy observations and actions (e.g. “For ‘space-saving ottoman bench for living room’, the item has options ‘39x18x18inch’ and ‘blue’ and seems good to buy.”). However, existing methods are still far from the performance of expert humans (Table 4), who perform significantly more product explorations and query re-formulations that are still challenging for prompting-based methods.

On the value of internal reasoning vs. external feedback To our knowledge, ReAct is the first demonstration of combined reasoning and action using an LLM applied to an interactive environment within a closed-loop system. Perhaps the closest prior work is Inner Monologue (IM), from Huang et al. (2022b), in which actions from an embodied agent are motivated by an eponymous “inner monologue”. However, IM’s “inner monologue” is limited to observations of the environment state and what needs to be completed by the agent for the goal to be satisfied. In contrast, the reasoning traces in ReAct for decision making is flexible and sparse, allowing diverse reasoning types (see Section 2) to be induced for different tasks.

To demonstrate the differences between ReAct and IM, and to highlight the importance of internal reasoning vs. simple reactions to external feedback, we ran an ablation experiment using a thought pattern composed of IM-like dense external feedback. As can be seen in Table 3, ReAct substantially outperforms IM-style prompting (ReAct-IM) (71 vs. 53 overall success rate), with consistent advantages on five out of six tasks. Qualitatively, we observed that ReAct-IM often made mistakes in identifying when subgoals were finished, or what the next subgoal should be, due to a lack of highlevel goal decomposition. Additionally, many ReAct-IM trajectories struggled to determine where an item would likely be within the ALFWorld environment, due to a lack of commonsense reasoning. Both shortcomings can be addressed in the ReAct paradigm. More details about ReAct-IM is in Appendix B.2. An example prompt for ReAct-IM can be found in Appendix C.4, and an example trajectory in Appendix D.2.3.

# 5 RELATED WORK

Language model for reasoning Perhaps the most well-known work of using LLMs for reasoning is Chain-of-Thought (CoT) (Wei et al., 2022), which reveals the ability of LLMs to formulate their own “thinking procedure” for problem solving. Several follow-up works have since been performed, including least-to-most prompting for solving complicated tasks (Zhou et al., 2022), zero-shotCoT (Kojima et al., 2022), and reasoning with self-consistency (Wang et al., 2022a). Recently, (Madaan & Yazdanbakhsh, 2022) systematically studied the formulation and structure of CoT, and observed that the presence of symbols, patterns and texts is crucial to the effectiveness of CoT. Other work has also been extended to more sophisticated reasoning architecture beyond simple prompting. For example Selection-Inference (Creswell et al., 2022) divides the reasoning process into two steps of “selection” and “inference”. STaR (Zelikman et al., 2022) bootstraps the reasoning process by finetuning the model on correct rationales generated by the model itself. Faithful reasoning (Creswell & Shanahan, 2022) decomposes multi-step reasoning into three steps, each performed by a dedicated LM respectively. Similar approaches like Scratchpad (Nye et al., 2021), which finetunes a LM on intermediate computation steps, also demonstrate improvement on multi-step computation problems. In contrast to these methods, ReAct performs more than just isolated, fixed reasoning, and integrates model actions and their corresponding observations into a coherent stream of inputs for the model to reason more accurately and tackle tasks beyond reasoning (e.g. interactive decision making).

Language model for decision making The strong capability of LLMs has enabled them to perform tasks beyond language generation, and it is becoming more popular to take advantage of LLMs as a policy model for decision making, especially in interactive environments. WebGPT (Nakano et al., 2021) uses an LM to interact with web browsers, navigate through web pages, and infer answers to complicated questions from ELI5 (Fan et al., 2019). In comparison to ReAct, WebGPT does not explicitly model the thinking and reasoning procedure, instead rely on expensive human feedback for reinforcement learning. In conversation modeling, chatbots like BlenderBot (Shuster et al., 2022b) and Sparrow (Glaese et al., 2022) and task-oriented dialogue systems like SimpleTOD (Hosseini-Asl et al., 2020) also train LMs to make decision about API calls. Unlike ReAct, they do not explicitly consider the reasoning procedure either, and also relies on expensive datasets and human feedback collections for policy learning. In contrast, ReAct learns a policy in a much cheaper way, since the decision making process only requires language description of the reasoning procedure.6

LLMS have also been increasingly employed in interactive and embodied environments for planning and decision making. Perhaps most relevant to ReAct in this respect are SayCan (Ahn et al., 2022) and Inner Monologue (Huang et al., 2022b), which use LLMs for robotic action planning and decision making. In SayCan, LLMs were prompted to directly predict possible actions a robot can take, which is then reranked by an affordance model grounded on the visual environments for final prediction. Inner Monologue made further improvements by adding the eponymous “inner monologue", which is implemented as injected feedback from the environment. To our knowledge, Inner Monologue is the first work that demonstrates such a closed-loop system, which ReAct builds on. However, we argue that Inner Monologue does not truly comprise of inner thoughts — this is elaborated in Section 4. We also note that leveraging language as semantically-rich inputs in the process of interactive decision making has been shown to be successful under other settings (Abramson et al., 2020; Karamcheti et al., 2021; Huang et al., $2 0 2 2 \\mathrm { a }$ ; Li et al., 2022). It is becoming more evident that with the help of LLMs, language as a fundamental cognitive mechanism will play a critical role in interaction and decision making. What is more, progress in LLMs has also inspired the development of versatile and generalist agents like Reed et al. (2022).

# 6 CONCLUSION

We have proposed ReAct – a simple yet effective method for synergizing reasoning and acting in large language models. Through a diverse set of experiments on multi-hop question-answering, fact checking, and interactive decision-making tasks, we show that ReAct leads to superior performance with interpretable decision traces. Despite the simplicity of our method, complex tasks with large action spaces require more demonstrations to learn well, which unfortunately can easily go beyond the input length limit of in-context learning. We explore the fine-tuning approach on HotpotQA with initial promising results, but learning from more high-quality human annotations will be the desiderata to further improve the performance. Scaling up ReAct with multi-task training and combining it with complementary paradigms like reinforcement learning could result in stronger agents that further unlock the potential of LLMs for more applications.

</details>

<details>
<summary>building-effective-ai-agents-anthropic</summary>

Agents are emerging in production as LLMs mature in key capabilities—understanding complex inputs, engaging in reasoning and planning, using tools reliably, and recovering from errors. Agents begin their work with either a command from, or interactive discussion with, the human user. Once the task is clear, agents plan and operate independently, potentially returning to the human for further information or judgement. During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress. Agents can then pause for human feedback at checkpoints or when encountering blockers. The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully.https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png&w=3840&q=75Autonomous agent

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

### Workflow: Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75The evaluator-optimizer workflow

**When to use this workflow:** This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

**Examples where evaluator-optimizer is useful:**

- Literary translation where there are nuances that the translator LLM might not capture initially, but where an evaluator LLM can provide useful critiques.
- Complex search tasks that require multiple rounds of searching and analysis to gather comprehensive information, where the evaluator decides whether further searches are warranted.

</details>

<details>
<summary>building-with-extended-thinking-anthropic</summary>

Extended thinking gives Claude enhanced reasoning capabilities for complex tasks, while providing varying levels of transparency into its step-by-step thought process before it delivers its final answer.

## How extended thinking works

When extended thinking is turned on, Claude creates `thinking` content blocks where it outputs its internal reasoning. Claude incorporates insights from this reasoning before crafting a final response.

The API response will include `thinking` content blocks, followed by `text` content blocks.

Here’s an example of the default response format:

```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me analyze this step by step...",
      "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve/op91XWHOEBLLqIOMfFG/UvLEczmEsUjavL...."
    },
    {
      "type": "text",
      "text": "Based on my analysis..."
    }
  ]
}
```

## How to use extended thinking

To turn on extended thinking, add a `thinking` object, with the `type` parameter set to `enabled` and the `budget_tokens` to a specified token budget for extended thinking.

The `budget_tokens` parameter determines the maximum number of tokens Claude is allowed to use for its internal reasoning process. In Claude 4 models, this limit applies to full thinking tokens, and not to [the summarized output](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#summarized-thinking). Larger budgets can improve response quality by enabling more thorough analysis for complex problems, although Claude may not use the entire budget allocated, especially at ranges above 32k.

`budget_tokens` must be set to a value less than `max_tokens`. However, when using [interleaved thinking with tools](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#interleaved-thinking), you can exceed this limit as the token limit becomes your entire context window (200k tokens).

### Summarized thinking

With extended thinking enabled, the Messages API for Claude 4 models returns a summary of Claude’s full thinking process. Summarized thinking provides the full intelligence benefits of extended thinking, while preventing misuse.

### Streaming thinking

You can stream extended thinking responses using [server-sent events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent%5Fevents/Using%5Fserver-sent%5Fevents).

When streaming is enabled for extended thinking, you receive thinking content via `thinking_delta` events.

When using streaming with thinking enabled, you might notice that text sometimes arrives in larger chunks alternating with smaller, token-by-token delivery. This is expected behavior, especially for thinking content.

The streaming system needs to process content in batches for optimal performance, which can result in this “chunky” delivery pattern, with possible delays between streaming events. We’re continuously working to improve this experience, with future updates focused on making thinking content stream more smoothly.

## Extended thinking with tool use

Extended thinking can be used alongside [tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview), allowing Claude to reason through tool selection and results processing.

### Preserving thinking blocks

During tool use, you must pass `thinking` blocks back to the API, and you must include the complete unmodified block back to the API. This is critical for maintaining the model’s reasoning flow and conversation integrity.

While you can omit `thinking` blocks from prior `assistant` role turns, we suggest always passing back all thinking blocks to the API for any multi-turn conversation. The API will:

- Automatically filter the provided thinking blocks
- Use the relevant thinking blocks necessary to preserve the model’s reasoning
- Only bill for the input tokens for the blocks shown to Claude

When Claude invokes tools, it is pausing its construction of a response to await external information. When tool results are returned, Claude will continue building that existing response. This necessitates preserving thinking blocks during tool use, for a couple of reasons:

1.  **Reasoning continuity**: The thinking blocks capture Claude’s step-by-step reasoning that led to tool requests. When you post tool results, including the original thinking ensures Claude can continue its reasoning from where it left off.
2.  **Context maintenance**: While tool results appear as user messages in the API structure, they’re part of a continuous reasoning flow. Preserving thinking blocks maintains this conceptual flow across multiple API calls. For more information on context management, see our [guide on context windows](https://docs.anthropic.com/en/docs/build-with-claude/context-windows).

### Interleaved thinking

Extended thinking with tool use in Claude 4 models supports interleaved thinking, which enables Claude to think between tool calls and make more sophisticated reasoning after receiving tool results.

With interleaved thinking, Claude can:

- Reason about the results of a tool call before deciding what to do next
- Chain multiple tool calls with reasoning steps in between
- Make more nuanced decisions based on intermediate results

With interleaved thinking, the `budget_tokens` can exceed the `max_tokens` parameter, as it represents the total budget across all thinking blocks within one assistant turn.

Tool use without interleaved thinking:

1.  Claude thinks once at the beginning to understand the task
2.  Makes all tool use decisions upfront
3.  When tool results are returned, Claude immediately provides a response without additional thinking

Tool use with interleaved thinking:

1.  Claude thinks about the task initially
2.  After receiving the calculator result, Claude can think again about what that result means
3.  Claude then decides how to query the database based on the first result
4.  After receiving the database result, Claude thinks once more about both results before formulating a final response
5.  The thinking budget is distributed across all thinking blocks within the turn

This pattern allows for more sophisticated reasoning chains where each tool’s output informs the next decision.

</details>

<details>
<summary>cdn-openai-com</summary>

While conventional software enables users to streamline and automate workflows, agents are able to perform the same workflows on the users’ behalf with a high degree of independence.

Agents are systems that independently accomplish tasks on your behalf.

A workflow is a sequence of steps that must be executed to meet the user’s goal, whether that's resolving a customer service issue, booking a restaurant reservation, committing a code change, or generating a report.

Applications that integrate LLMs but don’t use them to control workflow execution—think simple chatbots, single-turn LLMs, or sentiment classifiers—are not agents.

More concretely, an agent possesses core characteristics that allow it to act reliably and consistently on behalf of a user:

01

It leverages an LLM to manage workflow execution and make decisions. It recognizes when a workflow is complete and can proactively correct its actions if needed. In case of failure, it can halt execution and transfer control back to the user.

02

It has access to various tools to interact with external systems—both to gather context and to take actions—and dynamically selects the appropriate tools depending on the workflow’s current state, always operating within clearly defined guardrails.

# When should you build an agent?

Building agents requires rethinking how your systems make decisions and handle complexity. Unlike conventional automation, agents are uniquely suited to workflows where traditional deterministic and rule-based approaches fall short.

Consider the example of payment fraud analysis. A traditional rules engine works like a checklist, flagging transactions based on preset criteria. In contrast, an LLM agent functions more like a seasoned investigator, evaluating context, considering subtle patterns, and identifying suspicious activity even when clear-cut rules aren’t violated. This nuanced reasoning capability is exactly what enables agents to manage complex, ambiguous situations effectively.

As you evaluate where agents can add value, prioritize workflows that have previously resisted automation, especially where traditional methods encounter friction:

| | | |
| --- | --- | --- |
| 01 | Complex decision-making: | Workflows involving nuanced judgment, exceptions, or context-sensitive decisions, for example refund approval in customer service workflows. |
| 02 | Difficult-to-maintain rules: | Systems that have become unwieldy due to extensive and intricate rulesets, making updates costly or error-prone, for example performing vendor security reviews. |
| 03 | Heavy reliance on unstructured data: | Scenarios that involve interpreting natural language, extracting meaning from documents,or interacting with users conversationally, for example processing a home insurance claim. |

Before committing to building an agent, validate that your use case can meet these criteria clearly.

Otherwise, a deterministic solution may suffice.

# Agent design foundations

In its most fundamental form, an agent consists of three core components:

01 Model The LLM powering the agent’s reasoning and decision-making 02 Tools External functions or APIs the agent can use to take action 03 Instructions Explicit guidelines and guardrails defining how the agent behaves

# Orchestration

With the foundational components in place, you can consider orchestration patterns to enable your agent to execute workflows effectively.

While it’s tempting to immediately build a fully autonomous agent with complex architecture, customers typically achieve greater success with an incremental approach.

In general, orchestration patterns fall into two categories:

# 01

Single-agent systems, where a single model equipped with appropriate tools and instructions executes workflows in a loop

02 Multi-agent systems, where workflow execution is distributed across multiple coordinated agents

Let’s explore each pattern in detail.

# Single-agent systems

A single agent can handle many tasks by incrementally adding tools, keeping complexity manageable and simplifying evaluation and maintenance. Each new tool expands its capabilities without prematurely forcing you to orchestrate multiple agents.https://cdn.openai.com/business-guides-and-resources/images/1b855f5e45c9c41108110fe9f2212a99fe1eccdb3706d50c96a644dc7c456730.jpg

Every orchestration approach needs the concept of a ‘run’, typically implemented as a loop that lets agents operate until an exit condition is reached. Common exit conditions include tool calls, a certain structured output, errors, or reaching a maximum number of turns.

This concept of a while loop is central to the functioning of an agent. In multi-agent systems, as you’ll see next, you can have a sequence of tool calls and handofsf between agents but allow the model to run multiple steps until an exit condition is met.

An efefctive strategy for managing complexity without switching to a multi-agent framework is to use prompt templates. Rather than maintaining numerous individual prompts for distinct use cases, use a single flexible base prompt that accepts policy variables. This template approach adapts easily to various contexts, signifciantly simplifying maintenance and evaluation. As new use cases arise, you can update variables rather than rewriting entire workflows.

# When to consider creating multiple agents

Our general recommendation is to maximize a single agent’s capabilities frist. More agents can provide intuitive separation of concepts, but can introduce additional complexity and overhead, so often a single agent with tools is sufcifient.

For many complex workfolws, splitting up prompts and tools across multiple agents allows for improved performance and scalability. When your agents fail to follow complicated instructions or consistently select incorrect tools, you may need to further divide your system and introduce more distinct agents.

Practical guidelines for splitting agents include:

# Complex logic

When prompts contain many conditional statements (multiple if-then-else branches), and prompt templates get difcifult to scale, consider dividing each logical segment across separate agents.

# Tool overload

The issue isn’t solely the number of tools, but their similarity or overlap. Some implementations successfully manage more than 15 well-defnied, distinct tools while others struggle with fewer than 10 overlapping tools. Use multiple agents if improving tool clarity by providing descriptive names, clear parameters, and detailed descriptions doesn’t improve performance.

# Multi-agent systems

While multi-agent systems can be designed in numerous ways for specifci workflows and requirements, our experience with customers highlights two broadly applicable categories:

# Manager (agents as tools)

A central “manager” agent coordinates multiple specialized agents via tool calls, each handling a specifci task or domain.

# Decentralized (agents handing offto agents)

Multiple agents operate as peers, handing of tasks to one another based on their specializations.

Multi-agent systems can be modeled as graphs, with agents represented as nodes. In the manager pattern, edges represent tool calls whereas in the decentralized pattern, edges represent handoffs that transfer execution between agents.

Regardless of the orchestration pattern, the same principles apply: keep components flexible, composable, and driven by clear, well-structured prompts.

# Manager pattern

The manager pattern empowers a central LLM—the “manager”—to orchestrate a network of specialized agents seamlessly through tool calls. Instead of losing context or control, the manager intelligently delegates tasks to the right agent at the right time, effortlessly synthesizing the results into a cohesive interaction. This ensures a smooth, unified user experience, with specialized capabilities always available on-demand.

This pattern is ideal for workflows where you only want one agent to control workflow execution and have access to the user.https://cdn.openai.com/business-guides-and-resources/images/77ab31e96a92e417e60361adafa68aaec5577ea6624a59af4dee0b34ff923ca4.jpg

# Decentralized pattern

In a decentralized pattern, agents can ‘handof’fworkfolw execution to one another. Handofsf are a one way transfer that allow an agent to delegate to another agent.

This pattern involves using many agents on equal footing, where one agent can directly hand offcontrol of the workfolw to another agent. This is optimal when you don’t need a single agent maintaining central control or synthesis—instead allowing each agent to take over execution and interact with the user as needed.https://cdn.openai.com/business-guides-and-resources/images/8d4d21564c4a3af5d8456b906da6fda033d5cc3a066277671e6738de913f2bd2.jpg

In the above example, the initial user message is sent to triage\_agent. Recognizing that the input concerns a recent purchase, the triage\_agent would invoke a handoffto the order\_management\_agent, transferring control to it.

This pattern is especially efefctive for scenarios like conversation triage, or whenever you prefer specialized agents to fully take over certain tasks without the original agent needing to remain involved. Optionally, you can equip the second agent with a handoffback to the original agent, allowing it to transfer control again if necessary.

</details>

<details>
<summary>how-reasoning-ai-agents-transform-high-stakes-decision-makin</summary>

Thanks to reasoning AI models, agents can learn how to think critically and tackle complex tasks. This new class of “reasoning agents” can break down complicated problems, weigh options and make informed decisions — while using only as much compute and as many [tokens](https://blogs.nvidia.com/blog/ai-tokens-explained/) as needed.

Reasoning agents are making a splash in industries where decisions rely on multiple factors. Such industries range from customer service and healthcare to manufacturing and financial services.

## **Reasoning On vs. Reasoning Off**

Modern AI agents can toggle reasoning on and off, allowing them to efficiently use compute and tokens.

A full [chain‑of‑thought](https://www.nvidia.com/en-us/glossary/cot-prompting/) pass performed during reasoning can take up to 100x more compute and tokens than a quick, single‑shot reply — so it should only be used when needed. Think of it like turning on headlights — switching on high beams only when it’s dark and turning them back to low when it’s bright enough out.

Single-shot responses are great for simple queries — like checking an order number, resetting a password or answering a quick FAQ. Reasoning might be needed for complex, multistep tasks such as reconciling tax depreciation schedules or orchestrating the seating at a 120‑guest wedding.

## **Reasoning AI Agents in Action**

Reasoning AI agents are already being used for complex problem-solving across industries, including:

- **Healthcare:** Enhancing diagnostics and treatment planning.
- **Customer Service**: Automating and personalizing complex customer interactions, from resolving billing disputes to recommending tailored products.
- **Finance:** Autonomously analyzing market data and providing investment strategies.
- **Logistics and Supply Chain:** Optimizing delivery routes, rerouting shipments in response to disruptions and simulating possible scenarios to anticipate and mitigate risks.
- **Robotics**: Powering warehouse robots and autonomous vehicles, enabling them to plan, adapt and safely navigate dynamic environments.

## **Designing an AI Reasoning Agent**

A few key components are required to build an AI agent, including tools, memory and planning modules. Each of these components augments the agent’s ability to interact with the outside world, create and execute detailed plans, and otherwise act semi- or fully autonomously.

</details>

<details>
<summary>measuring-ai-ability-to-complete-long-tasks-metr</summary>

**Summary:** We propose measuring AI performance in terms of the _length_ of tasks AI agents can complete. We show that this metric has been consistently exponentially increasing over the past 6 years, with a doubling time of around 7 months. Extrapolating this trend predicts that, in under a decade, we will see AI agents that can independently complete a large fraction of software tasks that currently take humans days or weeks.https://metr.org/assets/images/measuring-ai-ability-to-complete-long-tasks/length-of-tasks-log.png
The length of tasks (measured by how long they take human professionals) that generalist frontier model agents can complete autonomously with 50% reliability has been doubling approximately every 7 months for the last 6 years. The shaded region represents 95% CI calculated by hierarchical bootstrap over task families, tasks, and task attempts.

We think that forecasting the capabilities of future AI systems is important for understanding and preparing for the impact of powerful AI. But predicting capability trends is hard, and even understanding the abilities of today’s models can be confusing.

Current frontier AIs are vastly better than humans at text prediction and knowledge tasks. They outperform experts on most exam-style problems for a fraction of the cost. With some task-specific adaptation, they can also serve as useful tools in many applications. And yet the best AI agents are not currently able to carry out substantive projects by themselves or directly substitute for human labor. They are unable to reliably handle even relatively low-skill, computer-based work like remote executive assistance. It is clear that capabilities are increasing very rapidly in some sense, but it is unclear how this corresponds to real-world impact.https://metr.org/assets/images/measuring-ai-ability-to-complete-long-tasks/test-scores-ai-capabilities-relative-human-performance.png
AI performance has increased rapidly on many benchmarks across a variety of domains. However, translating this increase in performance into predictions of the real world usefulness of AI can be challenging.


We find that measuring the length of tasks that models can complete is a helpful lens for understanding current AI capabilities. This makes sense: AI agents often seem to struggle with stringing together longer sequences of actions more than they lack skills or knowledge needed to solve single steps.

On a diverse set of multi-step software and reasoning tasks, we record the time needed to complete the task for humans with appropriate expertise. We find that the time taken by human experts is strongly predictive of model success on a given task: current models have almost 100% success rate on tasks taking humans less than 4 minutes, but succeed <10% of the time on tasks taking more than around 4 hours. This allows us to characterize the abilities of a given model by “the length (for humans) of tasks that the model can successfully complete with x% probability”.https://metr.org/assets/images/measuring-ai-ability-to-complete-long-tasks/model-success-rate.png

For each model, we can fit a logistic curve to predict model success probability using human task length. After fixing a success probability, we can then convert each model’s predicted success curve into a time duration, by looking at the length of task where the predicted success curve intersects with that probability. For example, here are fitted success curves for several models, as well as the lengths of tasks where we predict a 50% success rate:https://metr.org/assets/images/measuring-ai-ability-to-complete-long-tasks/models-are-succeeding-at-increasingly-long-tasks.png
Depiction of the process of computing the time horizon. For example, Claude 3.7 Sonnet (the right-most model, represented in the darkest green) has a time horizon of approximately one hour, as this is where its fitted logistic curve intersects the 50% success probability threshold.


We think these results help resolve the apparent contradiction between superhuman performance on many benchmarks and the common empirical observations that models do not seem to be robustly helpful in automating parts of people’s day-to-day work: the best current models—such as Claude 3.7 Sonnet—are capable of _some_ tasks that take even expert humans hours, but can only reliably complete tasks of up to a few minutes long.

That being said, by looking at historical data, we see that the length of tasks that state-of-the-art models can complete (with 50% probability) has increased dramatically over the last 6 years.https://metr.org/assets/images/measuring-ai-ability-to-complete-long-tasks/length-of-tasks-linear.png

If we plot this on a logarithmic scale, we can see that the length of tasks models can complete is well predicted by an exponential trend, with a doubling time of around 7 months.https://metr.org/assets/images/measuring-ai-ability-to-complete-long-tasks/length-of-tasks-log.png

Our estimate of the length of tasks that an agent can complete depends on methodological choices like the tasks used and the humans whose performance is measured. However, we’re fairly confident that the overall trend is roughly correct, at around 1-4 doublings per year. If the measured trend from the past 6 years continues for 2-4 more years, generalist autonomous agents will be capable of performing a wide range of week-long tasks.

The steepness of the trend means that our forecasts about when different capabilities will arrive are relatively robust even to large errors in measurement or in the comparisons between models and humans. For example, if the absolute measurements are off by a factor of 10x, that only changes the arrival time by around 2 years.

We discuss the limitations of our results, and detail various robustness checks and sensitivity analyses in [the full paper](https://arxiv.org/abs/2503.14499). Briefly, we show that similar trends hold (albeit more noisily) on:

1. Various subsets of our tasks that might represent different distributions (very short software tasks vs the diverse HCAST vs RE-Bench, and subsets filtered by length or qualitative assessments of “messiness”).
2. A separate dataset based on real tasks (SWE-Bench Verified), with independently collected human time data based on estimates rather than baselines. This shows an even faster doubling time, of under 3 months.https://metr.org/assets/images/measuring-ai-ability-to-complete-long-tasks/time-horizon-swe.png
We replicate our results on SWE-bench Verified and observe a similar exponential trend


We also show in the paper that our results do not appear to be especially sensitive to which tasks or models we include, nor to any other methodological choices or sources of noise that we investigated:https://metr.org/assets/images/measuring-ai-ability-to-complete-long-tasks/uncertainty-in-extrapolated-date.png
A sensitivity analysis of the extrapolated date at which frontier AI systems will have a horizon of 1 month. In each row, we apply 10,000 random perturbations to our data and find the distribution over the date of 1-month AI implied by the perturbed data. Box endpoints represent the 25th and 75th percentiles, and whiskers the 10th and 90th percentiles, with outliers not displayed. Note that this plot does not account for future changes in the trend or external validity concerns, which are responsible for the majority of our uncertainty.


However, there remains the possibility of substantial model error. For example, there are reasons to think that recent trends in AI are more predictive of future performance than pre-2024 trends. As shown above, when we fit a similar trend to just the 2024 and 2025 data, this shortens the estimate of when AI can complete month-long tasks with 50% reliability by about 2.5 years.

### Conclusion

We believe this work has important implications for AI benchmarks, forecasts, and risk management.

First, our work demonstrates an approach to making benchmarks more useful for forecasting: measuring AI performance in terms of the _length_ of tasks the system can complete (as measured by how long the tasks take humans). This allows us to measure how models have improved over a wide range of capability levels and diverse domains. At the same time, the direct relationship to real-world outcomes permits a meaningful interpretation of absolute performance, not just relative performance.

Second, we find a fairly robust exponential trend over years of AI progress on a metric which matters for real-world impact. If the trend of the past 6 years continues to the end of this decade, frontier AI systems will be capable of autonomously carrying out month-long projects. This would come with enormous stakes, both in terms of potential benefits and potential risks.

</details>

<details>
<summary>react-synergizing-reasoning-and-acting-in-language-models</summary>

Recent advances have expanded the applicability of language models (LM) to downstream tasks. On one hand, existing language models that are properly prompted, via [chain-of-thought](https://ai.googleblog.com/2022/05/language-models-perform-reasoning-via.html), demonstrate emergent capabilities that carry out self-conditioned reasoning traces to derive answers from questions, excelling at various arithmetic, commonsense, and symbolic reasoning tasks. However, with chain-of-thought prompting, a model is not grounded in the external world and uses its own internal representations to generate reasoning traces, limiting its ability to reactively explore and reason or update its knowledge. On the other hand, recent work uses pre-trained language models for planning and acting in various interactive environments (e.g., [text games](https://arxiv.org/pdf/2010.02903.pdf), [web navigation](https://arxiv.org/pdf/2112.09332.pdf), [embodied tasks](https://arxiv.org/pdf/2201.07207.pdf), [robotics](https://ai.googleblog.com/2022/08/towards-helpful-robots-grounding.html)), with a focus on mapping text contexts to text actions via the language model’s internal knowledge. However, they do not reason abstractly about high-level goals or maintain a working memory to support acting over long horizons.

In “ [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629.pdf)”, we propose a general paradigm that combines reasoning and acting advances to enable language models to solve various language reasoning and decision making tasks. We demonstrate that the _Reason+Act_(ReAct) paradigm systematically outperforms reasoning and acting only paradigms, when prompting bigger language models and fine-tuning smaller language models. The tight integration of reasoning and acting also presents human-aligned task-solving trajectories that improve interpretability, diagnosability, and controllability..

## Model Overview

ReAct enables language models to generate both verbal reasoning traces and text actions in an interleaved manner. While actions lead to observation feedback from an external environment (“Env” in the figure below), reasoning traces do not affect the external environment. Instead, they affect the internal state of the model by reasoning over the context and updating it with useful information to support future reasoning and acting.

|     |
| --- |
| [https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiuuYg9Pduep9GkUfjloNVOiy3qjpPbT017GKlgGEGMaLNu_TCheEeJ7r8Qok6-0BK3KMfLvsN2vSgFQ8xOvnHM9CAb4Ix4I62bcN2oXFWfqAJzGAGbVqbeCyVktu3h9Dyf5ameRe54LEr32Emp0nG52iofpNOTXCxMY12K7fvmDZNPPmfJaT5zo1OBQA/s16000/Screen%20Shot%202022-11-08%20at%208.53.49%20AM.png](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiuuYg9Pduep9GkUfjloNVOiy3qjpPbT017GKlgGEGMaLNu_TCheEeJ7r8Qok6-0BK3KMfLvsN2vSgFQ8xOvnHM9CAb4Ix4I62bcN2oXFWfqAJzGAGbVqbeCyVktu3h9Dyf5ameRe54LEr32Emp0nG52iofpNOTXCxMY12K7fvmDZNPPmfJaT5zo1OBQA/s595/Screen%20Shot%202022-11-08%20at%208.53.49%20AM.png) |
| Previous methods prompt language models (LM) to either generate self-conditioned reasoning traces or task-specific actions. We propose ReAct, a new paradigm that combines reasoning and acting advances in language models. |

## ReAct Prompting

We focus on the setup where a frozen language model, [PaLM-540B](https://arxiv.org/pdf/2204.02311.pdf), is prompted with few-shot in-context examples to generate both domain-specific actions (e.g., “search” in question answering, and “go to” in room navigation), and free-form language reasoning traces (e.g., “Now I need to find a cup, and put it on the table”) for task solving.

For tasks where reasoning is of primary importance, we alternate the generation of reasoning traces and actions so that the task-solving trajectory consists of multiple reasoning-action-observation steps. In contrast, for decision making tasks that potentially involve a large number of actions, reasoning traces only need to appear sparsely in the most relevant positions of a trajectory, so we write prompts with sparse reasoning and let the language model decide the asynchronous occurrence of reasoning traces and actions for itself.

As shown below, there are various types of useful reasoning traces, e.g., decomposing task goals to create action plans, injecting commonsense knowledge relevant to task solving, extracting important parts from observations, tracking task progress while maintaining plan execution, handling exceptions by adjusting action plans, and so on.

The synergy between reasoning and acting allows the model to perform dynamic reasoning to create, maintain, and adjust high-level plans for acting (reason to act), while also interacting with the external environments (e.g., Wikipedia) to incorporate additional information into reasoning (act to reason).

## ReAct Fine-tuning

We also explore fine-tuning smaller language models using ReAct-format trajectories. To reduce the need for large-scale human annotation, we use the ReAct prompted PaLM-540B model to generate trajectories, and use trajectories with task success to fine-tune smaller language models (PaLM-8/62B).

|     |
| --- |
| [https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhoAazr9qsoobs5Nkp7_uxjml4AEWA9iwUfoNfJpcJEnj2ZOdrTXptaf9R2CyRK7Qif64zcPbywR6AeIOaeZs19vQ7OH6n-6vEyh1exiHXC965OSoNX4bsGjuIZ3Po9CuJb-LhDYyYTQr1rZum-FZ285gi11jsuiAG58C8MzifUPj8VCC_-2N3k3Fsosg/s16000/HotPotQA.png](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhoAazr9qsoobs5Nkp7_uxjml4AEWA9iwUfoNfJpcJEnj2ZOdrTXptaf9R2CyRK7Qif64zcPbywR6AeIOaeZs19vQ7OH6n-6vEyh1exiHXC965OSoNX4bsGjuIZ3Po9CuJb-LhDYyYTQr1rZum-FZ285gi11jsuiAG58C8MzifUPj8VCC_-2N3k3Fsosg/s776/HotPotQA.png) |
| Comparison of four prompting methods, (a) Standard, (b) Chain of thought (CoT, Reason Only), (c) Act-only, and (d) ReAct, solving a [HotpotQA](https://arxiv.org/abs/1809.09600) question. In-context examples are omitted, and only the task trajectory is shown. ReAct is able to retrieve information to support reasoning, while also using reasoning to target what to retrieve next, demonstrating a synergy of reasoning and acting. |

## Results

We conduct empirical evaluations of ReAct and state-of-the-art baselines across four different benchmarks: question answering (HotPotQA), fact verification ( [Fever](https://arxiv.org/abs/1803.05355)), text-based game ( [ALFWorld](https://arxiv.org/abs/2010.03768)), and web page navigation ( [WebShop](https://arxiv.org/abs/2207.01206)). For HotPotQA and Fever, with access to a [Wikipedia API](https://en.wikipedia.org/api/rest_v1/) with which the model can interact, ReAct outperforms vanilla action generation models while being competitive with chain of thought reasoning (CoT) performance. The approach with the best results is a combination of ReAct and CoT that uses both internal knowledge and externally obtained information during reasoning.

|     |     |     |
| --- | --- | --- |
|  | **HotpotQA (exact match, 6-shot)** | **FEVER (accuracy, 3-shot)** |
| Standard | 28.7 | 57.1 |
| Reason-only (CoT) | 29.4 | 56.3 |
| Act-only | 25.7 | 58.9 |
| ReAct | 27.4 | 60.9 |
| Best ReAct + CoT Method | **35.1** | **64.6** |
| Supervised SoTA | 67.5 (using ~140k samples) | 89.5 (using ~90k samples) |

|     |
| --- |
| PaLM-540B prompting results on HotpotQA and Fever. |

On ALFWorld and WebShop, ReAct with both one-shot and two-shot prompting outperforms imitation and reinforcement learning methods trained with ~105 task instances, with an absolute improvement of 34% and 10% in success rates, respectively, over existing baselines.

|     |     |     |
| --- | --- | --- |
|  | **AlfWorld (2-shot)** | **WebShop (1-shot)** |
| Act-only | 45 | 30.1 |
| ReAct | **71** | **40** |
| Imitation Learning Baselines | 37 (using ~100k samples) | 29.1 (using ~90k samples) |

|     |
| --- |
| PaLM-540B prompting task success rate results on AlfWorld and WebShop. |

|     |
| --- |
| [https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEg_0lCKXSvFq4yyjM5PGdM27OF6LWco9qFGQS1dwa3DtEF8AnAuXg9Q_nPDVyAArYwl9sGsB000-iuKJuSsNjo--fi1ZCJbrj-KwsZ6M569nWg-h2xRGHkdvQobUY9RiIr4MYkathIFyiAHZSnHAwVUfeijU-tCLyaHRgqXQah1XObtE71a00IbGdywVw/s16000/image1.png](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEg_0lCKXSvFq4yyjM5PGdM27OF6LWco9qFGQS1dwa3DtEF8AnAuXg9Q_nPDVyAArYwl9sGsB000-iuKJuSsNjo--fi1ZCJbrj-KwsZ6M569nWg-h2xRGHkdvQobUY9RiIr4MYkathIFyiAHZSnHAwVUfeijU-tCLyaHRgqXQah1XObtE71a00IbGdywVw/s839/image1.png) |
| Scaling results for prompting and fine-tuning on HotPotQA with ReAct and different baselines. ReAct consistently achieves best fine-tuning performances. |

|     |
| --- |
| [https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgP1HCCuyIgO9D3UQKQSKFAth_Xbtqke0UO0rVbAHYA3tmbGjC6wt_du2bEm12RxFx4uWQs1LxpqaFgmHExL8QRfnPJXHVgmy-TRU3yvsDpHa-oxiX8AzmaWsm92y0J2hxdJdsjxmvFqUyYIdLIfhlr2JOIQzuaXml5YXlrF7MxC22B6thYBl72mNMKvg/s16000/image6.png](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgP1HCCuyIgO9D3UQKQSKFAth_Xbtqke0UO0rVbAHYA3tmbGjC6wt_du2bEm12RxFx4uWQs1LxpqaFgmHExL8QRfnPJXHVgmy-TRU3yvsDpHa-oxiX8AzmaWsm92y0J2hxdJdsjxmvFqUyYIdLIfhlr2JOIQzuaXml5YXlrF7MxC22B6thYBl72mNMKvg/s1212/image6.png) |
| [https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEi41aji28YNe7jqjXOC0-bdWL6nFc6jlrVXOyVD7v15lYMEJ1JNzV-Q9V1Fh-GpX5iW_gH6CWnnvGyECHQkZF33H9E3RI-GTRKA7ZhaSPjyN2rbniob0_biOcP89qZYtGMpQiodO52CJ5iauN11aitR5brKbYIdB349vFMMwqirnZ2TdufpyHz9QbOyDA/s16000/image2.png](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEi41aji28YNe7jqjXOC0-bdWL6nFc6jlrVXOyVD7v15lYMEJ1JNzV-Q9V1Fh-GpX5iW_gH6CWnnvGyECHQkZF33H9E3RI-GTRKA7ZhaSPjyN2rbniob0_biOcP89qZYtGMpQiodO52CJ5iauN11aitR5brKbYIdB349vFMMwqirnZ2TdufpyHz9QbOyDA/s1216/image2.png) |

|     |
| --- |
| A comparison of the ReAct ( **top**) and CoT ( **bottom**) reasoning trajectories on an example from Fever (observation for ReAct is omitted to reduce space). In this case ReAct provided the right answer, and it can be seen that the reasoning trajectory of ReAct is more grounded on facts and knowledge, in contrast to CoT’s hallucination behavior. |

We also explore human-in-the-loop interactions with ReAct by allowing a human inspector to edit ReAct’s reasoning traces. We demonstrate that by simply replacing a hallucinating sentence with inspector hints, ReAct can change its behavior to align with inspector edits and successfully complete a task. Solving tasks becomes significantly easier when using ReAct as it only requires the manual editing of a few thoughts, which enables new forms of human-machine collaboration.

|     |
| --- |
| [https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgORrqQ_PMp1JiljcjCXK3BqVHFR5kJ1mUxISgURlkRa6RH2fCaP3HT6rALL453TM_wD3wyKhJrfAlqlgG6jEU-RsvQsNfb02PNzqgvDLwK1XyZPaaFyc9dGRzkQzLcGGWitXzf2Mthf3YymP-0w09-pxMJxrCScFIfKxDAyFUWQCV7tR8YGGeuiNqiKA/s16000/AlfWorld.png](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgORrqQ_PMp1JiljcjCXK3BqVHFR5kJ1mUxISgURlkRa6RH2fCaP3HT6rALL453TM_wD3wyKhJrfAlqlgG6jEU-RsvQsNfb02PNzqgvDLwK1XyZPaaFyc9dGRzkQzLcGGWitXzf2Mthf3YymP-0w09-pxMJxrCScFIfKxDAyFUWQCV7tR8YGGeuiNqiKA/s790/AlfWorld.png) |
| A human-in-the-loop behavior correction example with ReAct on AlfWorld. (a) ReAct trajectory fails due to a hallucinating reasoning trace (Act 17). (b) A human inspector edits two reasoning traces (Act 17, 23), ReAct then produces desirable reasoning traces and actions to complete the task. |

## Conclusion

We present ReAct, a simple yet effective method for synergizing reasoning and acting in language models. Through various experiments that focus on multi-hop question-answering, fact checking, and interactive decision-making tasks, we show that ReAct leads to superior performance with interpretable decision traces.

ReAct demonstrates the feasibility of jointly modeling thought, actions and feedback from the environment within a language model, making it a versatile agent that is capable of solving tasks that require interactions with the environment. We plan to further extend this line of research and leverage the strong potential of the language model for tackling broader embodied tasks, via approaches like massive multitask training and coupling ReAct with equally strong reward models.

</details>

<details>
<summary>scraping-failed</summary>

⚠️ Error scraping https://arxiv.org/pdf/2504.19678 after 3 attempts: Internal Server Error: Failed to make POST request. (Internal server error) - Scrape resulted in unsupported file: File size exceeds 10MB - No additional error details provided.

</details>

<details>
<summary>what-is-a-react-agent-ibm</summary>

A ReAct agent is an [AI agent](https://www.ibm.com/think/topics/ai-agents) that uses the “reasoning and acting” (ReAct) framework to combine [chain of thought (CoT)](https://www.ibm.com/think/topics/chain-of-thoughts) reasoning with external tool use. The ReAct framework enhances the ability of a [large language model (LLM)](https://www.ibm.com/think/topics/large-language-models) to handle complex tasks and decision-making in [agentic workflows](https://www.ibm.com/think/topics/agentic-workflows).

First introduced by Yao and others in the 2023 paper, “ReACT: Synergizing Reasoning and Acting in Language Models,” ReAct can be understood most generally as a [machine learning](https://www.ibm.com/think/topics/machine-learning) (ML) paradigm to integrate the reasoning and action-taking capabilities of LLMs.

More specifically, ReAct is a conceptual framework for building AI agents that can interact with their environment in a structured but adaptable way, by using an LLM as the agent’s “brain” to coordinate anything from simple [retrieval augmented generation (RAG)](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) to intricate [multiagent](https://www.ibm.com/think/topics/multiagent-system) workflows.

Unlike traditional artificial intelligence (AI) systems, ReAct agents don’t separate decision-making from task execution. Therefore, the development of the ReAct paradigm was an important step in the evolution of [generative AI (gen AI)](https://www.ibm.com/think/topics/generative-ai) beyond mere conversational [chatbots](https://www.ibm.com/think/topics/chatbots) and toward complex problem-solving.

ReAct agents and derivative approaches continue to power AI applications that can autonomously plan, execute and adapt to unforeseen circumstances.

## How do ReAct agents work?

The ReAct framework is inspired by the way humans can intuitively use natural language—often through our own inner monologue—in the step-by-step planning and execution of complex tasks.

Rather than implementing rule-based or otherwise predefined workflows, ReAct agents rely on their LLM’s reasoning capabilities to dynamically adjust their approach based on new information or the results of previous steps.

Imagine packing for a brief trip. You might start by identifying key considerations (“ _What will the weather be like while I’m there?_”), then actively consult external sources (“ _I’ll check the local weather forecast_”).

By using that new information (“ _It’s going to be cold_”), you determine your next consideration (“ _What warm clothes do I have?_”) and action (“ _I’ll check my closet_”). Upon taking that action, you might encounter an unexpected obstacle (“ _All of my warm clothes are in storage_”) and adjust your next step accordingly (“ _What clothes can I layer together?_”).

In a similar fashion, the ReAct framework uses [prompt engineering](https://www.ibm.com/think/topics/prompt-engineering) to structure an AI agent’s activity in a formal pattern of alternating thoughts, actions and observations:

- The verbalized CoT reasoning steps ( _thoughts_) help the model decompose the larger task into more manageable subtasks.
- Predefined _actions_ enable the model to use tools, make [application programming interface (API)](https://www.ibm.com/think/topics/api) calls and gather more information from external sources (such as search engines) or knowledge bases (such as an internal docstore).
- After taking an action, the model then reevaluates its progress and uses that _observation_ to either deliver a final answer or inform the next _thought_. The observation might ideally also consider prior information, whether from earlier in the model’s standard context window or from an external memory component.

Because the performance of a ReAct agent depends heavily on the ability of its central LLM to “verbally” think its way through complex tasks, ReAct agents benefit greatly from highly capable models with advanced reasoning and instruction-following ability.

To minimize cost and [latency](https://www.ibm.com/think/topics/latency), a multiagent ReAct framework might rely primarily on a larger, more performant model to serve as the central agent whose reasoning process or actions might involve delegating subtasks to more agents built using smaller, more efficient models.

### ReAct agent loops

This framework inherently creates a feedback loop in which the model problem-solves by iteratively repeating this interleaved _thought-action-observation_ process.

Each time this loop is completed—that is, each time the agent has taken an action and made an observation based on the results of that action—the agent must then decide whether to repeat or end the loop.

When and how to end the reasoning loop is an important consideration in the design of a ReAct agent. Establishing a maximum number of loop iterations is a simple way to limit latency, costs and token usage, and avoid the possibility of an endless loop.

Conversely, the loop can be set to end when some specific condition is met, such as when the model has identified a potential final answer that exceeds a certain confidence threshold.

To implement this kind of reasoning and acting loop, ReAct agents typically use some variant of _ReAct prompting_, whether in the system prompt provided to the LLM or in the context of the user query itself.

## ReAct prompting

ReAct prompting is a specific prompting technique designed to guide an LLM to follow the ReAct paradigm of _thought_, _action_ and _observation_ loops. While the explicit use of conventional ReAct prompting methods is not strictly necessary to build a ReAct agent, most ReAct-based agents implement or at least take direct inspiration from it.

First outlined in the original ReAct paper, ReAct prompting’s primary function is to instruct an LLM to follow the ReAct loop and establish which tools can be used—that is, which actions can be taken—when handling user queries.

Whether through explicit instructions or the inclusion of [few-shot](https://www.ibm.com/think/topics/few-shot-learning) examples, ReAct prompting should:

- **Guide the model to use chain of thought reasoning:** Prompt the model to reason its way through tasks by thinking step by step, interleaving thoughts with actions.
- **Define actions:** Establish the specific actions available to the model. An action might entail the generation of a specific type of next thought or subprompt but usually involves [using external tools](https://www.ibm.com/think/topics/tool-calling) or making APIs.
- **Instruct the model to make observations:** Prompt the model to reassess its context after each action step and use that updated context to inform the next reasoning step.
- **Loop:** Instruct the model to repeat the previous steps if necessary. You could provide specific conditions for ending that loop, such as a maximum number of loops, or instruct the agent to end its reasoning process whenever it feels it has arrived at the correct final output.
- **Output final answer:** Whenever those end conditions have been met, provide the user with the final output in response to their initial query. As with many uses of LLMs, as reasoning models employing chain of thought reasoning before determining a final output, ReAct agents are often prompted to conduct their reasoning process within a [“scratchpad.”](https://arxiv.org/abs/2112.00114)

A classic demonstration of ReAct prompting is the system prompt for the prebuiltZERO\_SHOT\_REACT-DESCRIPTION
ReAct agent module in [Langchain](https://www.ibm.com/think/topics/langchain)’s LangGraph. It’s called “ [zero-shot](https://www.ibm.com/think/topics/zero-shot-learning)” because, with this predefined system prompt, the LLM being used with the module does not need any further examples to behave as a ReAct agent.

```
Answer the following questions as best you can. You have access to the following tools:

Wikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.
duckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.
Calculator: Useful for when you need to answer questions about math.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Wikipedia, duckduckgo_search, Calculator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
```

## Benefits of ReAct agents

The introduction of the ReAct framework was an important step in the advancement of LLM-driven [agentic workflows](https://www.ibm.com/think/topics/agentic-workflows). From grounding LLMs in real time, real-world external information through (RAG) to contributing to subsequent breakthroughs—such as [Reflexion](https://arxiv.org/abs/2303.11366), which led to modern reasoning models—ReAct has helped catalyze the use of LLMs for tasks well beyond text generation.

The utility of ReAct agents is drawn largely from some of the inherent qualities of the ReAct framework:

- **Versatility:** ReAct agents can be configured to work with a wide variety of external tools and APIs. Though [fine-tuning](https://www.ibm.com/think/topics/fine-tuning) relevant ReAct prompts (using relevant tools) can improve performance, no prior configuration of the model is required to execute [tool calls](https://www.ibm.com/think/topics/tool-calling).
- **Adaptability:** This versatility, along with the dynamic and situational nature of how they determine the appropriate tool or API to call, means that ReAct agents can use their reasoning process to adapt to new challenges. Especially when operating within a lengthy context window or augmented with external memory, they can learn from past mistakes and successes to tackle unforeseen obstacles and situations. This makes ReAct agents flexible and resilient.
- **Explainability:** The verbalized reasoning process of a ReAct agent is simple to follow, which facilitates debugging and helps make them relatively user-friendly to build and optimize.
- **Accuracy:** As the original ReAct paper asserts, chain of thought (CoT) reasoning alone has many benefits for LLMs, but also runs an increased risk of hallucination. ReAct’s combination of CoT with a connection external to information sources significantly reduces [hallucinations](https://www.ibm.com/think/topics/ai-hallucinations), making ReAct agents more accurate and trustworthy.

## ReAct agents vs. function calling

Another prominent paradigm for agentic AI is function calling, originally [introduced by OpenAI in June 2023](https://openai.com/index/function-calling-and-other-api-updates/) to supplement the agentic abilities of its [GPT models](https://www.ibm.com/think/topics/gpt).

The function calling paradigm entails [fine-tuning](https://www.ibm.com/think/topics/fine-tuning) models to recognize when a particular situation should result in a tool call and output a structured [JSON](https://www.ibm.com/docs/en/baw/24.x?topic=formats-javascript-object-notation-json-format) object containing the arguments necessary to call those functions.

Many proprietary and open source LLM families, [including IBM® Granite®](https://www.ibm.com/granite/docs/models/granite/#function-calling), Meta’s [Llama](https://www.ibm.com/think/news/meta-llama-3-2-models) series, Anthropic’s [Claude](https://www.ibm.com/think/topics/claude-ai) and [Google Gemini](https://www.ibm.com/think/topics/google-gemini), now support function calling.

Whether ReAct or function calling is “better” will generally depend on the nature of your specific use case. In scenarios involving relatively straightforward (or at least predictable) tasks, function calling can execute faster, save tokens, and be simpler to implement than a ReAct agent.

In such circumstances, the number of tokens that would be spent on a ReAct agent’s iterative loop of CoT reasoning might be seen as inefficient.

The inherent tradeoff is a relative lack of ability to customize how and when the model chooses which tool to use. Likewise, when an agent handles tasks that call for complex reasoning, or scenarios that are dynamic or unpredictable, the rigidity of function calling might limit the agent’s adaptability. In such situations, it’s often beneficial to be able to view the step-by-step reasoning that led to a specific tool call.

</details>

<details>
<summary>what-is-agentic-reasoning-ibm</summary>

## What is agentic reasoning?

Agentic reasoning is a component of AI agents that handles decision-making. It allows artificial intelligence agents to conduct tasks autonomously by applying conditional logic or heuristics, relying on perception and memory, enabling it to pursue goals and optimize for the best possible outcome.

Earlier machine learning models followed a set of preprogrammed rules to arrive at a decision. Advances in AI have led to AI models with more evolved reasoning capabilities, but they still require human intervention to convert information into knowledge. Agentic reasoning takes it one step further, allowing AI agents to transform knowledge into action.

The “reasoning engine” powers the planning and tool calling phases of agentic workflows. Planning decomposes a task into more manageable reasoning, while tool calling helps inform an AI agent’s decision through available tools. These tools can include application programming interfaces (APIs), external datasets and data sources such as knowledge graphs.

## Agentic reasoning strategies

Agentic reasoning can be approached in different ways based on an agent’s architecture and type. Here are some common techniques for AI agent reasoning, including the pros and cons of each:

**● Conditional logic**

**● Heuristics**

**● ReAct (Reason + Act)**

**● ReWOO (Reasoning WithOut Observation)**

**● Self-reflection**

**● Multiagent reasoning**

### Conditional logic

Simple AI agents follow a set of preprogrammed condition-action rules. These rules usually take the form of “if-then” statements, where the “if” portion specifies the condition and the “then” portion indicates the action. When a condition is met, the agent carries out the corresponding action.

This reasoning methodology is especially suitable for domain-specific use cases. In finance, for instance, a fraud detection agent flags a transaction as fraudulent according to a set of criteria defined by a bank.

With conditional logic, agentic AI can’t act accordingly if it comes across a scenario it doesn’t recognize. To reduce this inflexibility, model-based agents use their memory and perception to store a current model or state of their environment. This state is updated as the agent receives new information. Model-based agents, however, are still bound by their condition-action rules.

For example, a robot navigates through a warehouse to stock a product on a shelf. It consults a model of the warehouse for the route it takes, but when it senses an obstacle, it can alter its path to avoid that obstacle and continue its traversal.

### Heuristics

AI agent systems can also use heuristics for reasoning. Goal-based agents, for instance, have a preset goal. Using a search algorithm, they find sequences of actions that can help them achieve their goal and then plan these actions before conducting them.

For example, an autonomous vehicle can have a navigation agent whose objective is to suggest the quickest path to a destination in real-time. It can search through different routes and recommend the fastest 1.

Like goal-based agents, utility-based agents search for action sequences that achieve a goal, but they factor in utility as well. They employ a utility function to determine the most optimal outcome. In the navigation agent example, it can be tasked with finding not only the swiftest route but also 1 that will consume the least amount of fuel.

### ReAct (Reason + Act)

This reasoning paradigm involves a think-act-observe loop for step-by-step problem-solving and iterative enhancement of responses. An agent is instructed to generate traces of its reasoning process,1 much like what happens with chain-of-thought reasoning in generative AI (gen AI) models and large language models (LLMs). It then acts on that reasoning and observes its output,2 updating its context with new reasoning based on its observations. The agent repeats the cycle until it arrives at an answer or solution.2

ReAct does well on natural language-specific tasks, and its traceability improves transparency. However, it can also generate the same reasoning and actions repeatedly, which can lead to infinite loops.2

### ReWOO (Reasoning WithOut Observation)

Unlike ReAct, ReWOO removes the observation step and plans ahead instead. This agentic reasoning design pattern consists of 3 modules: planner, worker and solver.3

The planner module breaks down a task into subtasks and allocates each of them to a worker module. The worker incorporates tools used to substantiate each subtask with evidence and facts. Finally, the solver module synthesizes all the subtasks and their corresponding evidence to draw a conclusion.3

ReWOO outperforms ReAct on certain natural language processing (NLP) benchmarks. However, adding extra tools can degrade ReWOO’s performance, and it doesn’t do well in situations where it has limited context about its environment.3

### Self-reflection

Agentic AI can also include self-reflection as part of assessing and refining its reasoning capabilities. An example of this is Language Agent Tree Search (LATS), which shares similarities with tree-of-thought reasoning in LLMs.

LATS was inspired by the Monte Carlo reinforcement learning method, with researchers adapting the Monte Carlo Tree Search for LLM-based agents.4 LATS builds a decision tree that represents a state as a node and an edge as an action, searches the tree for potential action options and employs a state evaluator to choose a particular action.2 It also applies a self-reflection reasoning step, incorporating its own observations as well as feedback from a language model to identify any errors in reasoning and recommend alternatives.2 The reasoning errors and reflections are stored in memory, serving as additional context for future reference.4

LATS excels in more complex tasks such as coding and interactive question answering and in workflow automation, including web search and navigation.4 However, a more involved approach and extra self-reflection step makes LATS more resource- and time-intensive compared to methods like ReAct.2

### Multiagent reasoning

Multiagent systems consist of multiple AI agents working together to solve complex problems. Each agent specializes in a certain domain and can apply its own agentic reasoning strategy.

However, the decision-making process can vary based on the AI system’s architecture. In a hierarchical or vertical ecosystem, 1 agent acts as a leader for AI orchestration and decides which action to take. Meanwhile, in a horizontal architecture, agents decide collectively.

## Challenges in agentic reasoning

Reasoning is at the core of AI agents and can result in more powerful AI capabilities, but it also has its limitations. Here are some challenges in agentic reasoning:

**● Computational complexity**

**● Interpretability**

**● Scalability**

### Computational complexity

Agentic reasoning can be difficult to implement. The process also requires significant time and computational power, especially when solving more complicated real-world problems. Enterprises must find ways to optimize their agentic reasoning strategies and be ready to invest in the necessary AI platforms and resources for development.

### Interpretability

Agentic reasoning might lack explainability and transparency on how decisions were made. Various methods can help establish interpretability, and integrating AI ethics and human oversight within algorithmic development are critical to make sure agentic reasoning engines make decisions fairly, ethically and accurately.

### Scalability

Agentic reasoning techniques are not 1-size-fits-all solutions, making it hard to scale them across AI applications. Businesses might need to tailor these reasoning design patterns for each of their use cases, which requires time and effort.

</details>

<details>
<summary>what-is-ai-agent-orchestration-ibm</summary>

# What is AI agent orchestration?

[Artificial intelligence (AI)](https://www.ibm.com/think/artificial-intelligence) agent orchestration is the process of coordinating multiple specialized [AI agents](https://www.ibm.com/think/topics/ai-agents) within a unified system to efficiently achieve shared objectives.

Rather than relying on a single, general-purpose AI solution, AI agent orchestration employs a network of AI agents, each designed for specific tasks, working together to automate complex workflows and processes.

To fully understand AI agent orchestration, it's essential to first understand AI agents themselves. This involves [understanding the differences](https://www.ibm.com/think/topics/agentic-ai-vs-generative-ai) between two key types of AI: [generative AI](https://www.ibm.com/think/topics/generative-ai), which creates original content based on a user’s prompt, and [agentic AI](https://www.ibm.com/think/insights/agentic-ai), which autonomously makes decisions and acts to pursue complex goals with minimal supervision.

AI assistants exist on a continuum, starting with rule-based chatbots, progressing to more advanced virtual assistants and evolving into generative AI and [large language model (LLM)](https://www.ibm.com/think/topics/large-language-models)-powered assistants capable of handling single-step tasks. At the top of this progression are AI agents, which operate autonomously. These agents make decisions, design workflows and use function calling to connect with external tools—such as [application programming interfaces (APIs)](https://www.ibm.com/think/topics/api), data sources, web searches and even other AI agents—to fill gaps in their knowledge. This is agentic AI.

AI agents are specialized, meaning each one is optimized for a particular function. Some agents focus on business and customer-facing tasks like billing, troubleshooting, scheduling and decision-making, while others handle more technical functions like [natural language processing (NLP),](https://www.ibm.com/think/topics/natural-language-processing) data retrieval and process automation. Advanced LLMs such as OpenAI's ChatGPT-4o or Google's Gemini often power these agents, with generative AI capabilities enabling them to create human-like responses and handle complex tasks autonomously.

Multi-agent systems (MAS) emerge when multiple AI agents collaborate, either in a structured or decentralized manner, to solve complex tasks more efficiently than a single agent might.

In practice, AI agent orchestration functions like a digital symphony. Each agent has a unique role and the system is guided by an orchestrator—either a central AI agent or framework —that manages and coordinates their interactions. The orchestrator helps synchronize these specialized agents, ensuring that the right agent is activated at the right time for each task. This coordination is crucial for handling multifaceted workflows that involve various tasks, helping ensure that processes are run seamlessly and efficiently.

For example, as part of [customer service automation](https://www.ibm.com/think/topics/customer-service-automation), the orchestrator agent (the system responsible for managing AI agents) might determine whether to engage a billing agent versus a technical support agent, helping ensure that customers receive seamless and relevant assistance. In MAS, agents might coordinate without a single orchestrator, dynamically communicating to collaboratively solve problems (see “Types of AI orchestration” below)

The benefits of AI agent orchestration are significant in industries with complex, dynamic needs such as telecommunications, banking and healthcare. By deploying specialized agents that are trained on targeted datasets and workflows, businesses can enhance [operational efficiency](https://www.ibm.com/think/topics/operational-efficiency), improve decision-making and deliver more accurate, efficient and context-aware results for both employees and customers.

## Why AI agent orchestration is important

As AI systems grow more advanced, a single AI model or agent is often insufficient for handling complex tasks. Autonomous systems frequently struggle to collaborate because they are built across multiple clouds and applications, leading to siloed operations and inefficiencies. AI agent orchestration bridges these gaps, enabling multiple AI agents to work together efficiently and ensuring that sophisticated tasks are run seamlessly.

In large-scale applications such as healthcare, finance and customer service, multiple agents often need to work together to handle different aspects of a task. For example, in healthcare, AI agents can coordinate between diagnostic tools, patient management systems and administrative workflows to streamline operations and enhance treatment accuracy. Without orchestration, these agents might work in isolation, leading to inefficiencies, redundancies or gaps in execution.

By managing interactions between multi-agent systems, orchestration helps ensure that each agent contributes effectively toward a shared goal. It optimizes workflows, minimizes errors and enhances interoperability, allowing AI systems to dynamically allocate resources, prioritize tasks and respond to changing conditions in real time. This capability is valuable in fields requiring continuous optimization such as supply chain management and personalized digital assistants.

As AI systems continue to evolve, AI agent orchestration becomes increasingly essential for unlocking their full potential.

## Types of AI agent orchestration

There are several types of AI agent orchestration. Real-world systems often combine multiple orchestration styles for more effective results.

**Centralized orchestration**: A single AI orchestrator agent acts as the "brain" of the system, directing all the other agents, assigning tasks and making final decisions. This structured approach helps ensure consistency, control and predictable workflows.

**Decentralized orchestration**: This model shifts away from a single, controlling entity, allowing MAS to function through direct communication and collaboration. Agents make independent decisions or reach a consensus as a group. This makes the system more scalable and resilient since no single failure can bring it down.

**Hierarchical orchestration**: Here, AI agents are arranged in layers, resembling a tiered command structure. Higher-level orchestrator agents oversee and manage lower-level agents, striking a balance between strategic control and task-specific execution. This allows for more organized workflows while still enabling specialized agents to operate with some autonomy. If the hierarchy becomes too rigid, adaptability can suffer.

**Federated orchestration**: This approach focuses on collaboration between independent AI agents or separate organizations, allowing them to work together without fully sharing data or relinquishing control over their individual systems. This is especially useful in situations where privacy, security or regulatory constraints prevent unrestricted data sharing, such as in healthcare, banking or cross-company collaborations.

## Comparing AI agent orchestration with related practices

**AI orchestration** manages and automates various AI components—like machine learning models, data pipelines and APIs—to help ensure that they work together efficiently within a system. It focuses on optimizing performance, automating repetitive tasks, supporting scalability and system-wide performance.

**AI agent orchestration** is a subset of AI orchestration that focuses specifically on coordinating autonomous AI agents—software entities that can make independent decisions and take actions. It helps ensure that agents collaborate effectively, assigning tasks and structuring workflows.

**Multi-agent orchestration** goes a step further, managing multiple AI agents working together on complex problems. It deals with communication, role allocation and conflict resolution to help ensure seamless collaboration between agents.

## AI agent orchestration steps

AI agent orchestration is a structured process to help ensure seamless collaboration between AI agents. The goal is to manage specialized agents effectively so they can autonomously complete tasks, share data flow and optimize workflows.

Initial steps involving design, configuration and implementation are performed by humans, including as AI engineers, developers and business strategists. Once the orchestrator agent is set up, it autonomously manages AI applications, assigning tasks, coordinating workflows and facilitating real-time collaboration.

The process generally follows these key steps:

- Assessment and planning
- Selection of specialized AI agents
- Orchestration framework implementation
- Agent selection and assignment
- Workflow coordination and execution
- Data sharing and context management
- Continuous optimization and learning

### Assessment and planning (human-driven)

Before orchestration begins, organizations assess their existing AI ecosystem and identify processes that might benefit from multi-agent orchestration. The orchestration team defines clear objectives, determines the scope of integration and selects the appropriate AI technologies.

### Selection of specialized AI agents (human-driven)

AI engineers and developers choose task-specific AI agents, such as those specializing in data analysis, automation or decision-making. These agents use gen AI and machine learning models to enhance their functions.

### Orchestration framework implementation (human-driven)

System architects integrate selected AI agents into a unified orchestration framework, establishing workflows that facilitate smooth agent-to-agent communication. This involves:

- Defining task execution sequences
- Setting up API integrations for data access
- Implementing open source orchestration tools such as IBM watsonx Orchestrate, Microsoft Power Automate and LangChain

Once this is complete, the orchestrator agent takes over real-time execution.

### Agent selection and assignment (orchestrator-driven)

The orchestrator dynamically identifies the best-suited AI agents for each task based on real-time data, workload balancing and predefined rules.

### Workflow coordination and execution (orchestrator-driven)

The orchestrator platform manages task sequencing and execution, helping to ensure smooth collaboration between agents. This includes:

- Breaking down tasks into subtasks
- Assigning the right AI agents to handle each step
- Managing inter-agent dependencies
- Integrating with external systems through API calls to access necessary data and services

### Data sharing and context management (orchestrator-driven)

To help ensure accuracy and prevent redundant work, AI agents continuously exchange information, maintaining a shared knowledge base. The orchestrator updates agents with real-time context.

### Continuous optimization and learning (orchestrator + human input)

The orchestrator monitors agent performance, detects inefficiencies and can autonomously adjust workflows. Human oversight is often required for refining orchestration strategies, retraining AI models or modifying orchestration rules for long-term improvements.

## AI agent orchestration benefits

AI agent orchestration offers several key benefits across various industries, making it a valuable approach for businesses aiming to enhance their operations and customer interactions.

**Enhanced efficiency**: Coordinating multiple specialized agents helps businesses streamline workflows, reduce redundancies and improve overall operational performance.

**Agility and flexibility**: AI agent orchestration allows organizations to adapt their operations rapidly as market conditions change.

**Improved experiences**: Orchestrated AI agents can enhance operational efficiency and provide more accurate and personalized support, resulting in more satisfying experiences for customers and employees.

**Increased reliability and fault tolerance**: The failure of one agent can be mitigated by others, which enhances system reliability and helps ensure continuous service delivery.

**Self-improving workflows**: Unlike traditional integration patterns, agent orchestration enables the creation of workflows that can autonomously adapt to new data and evolving requirements, improving over time.

**Scalability**: AI agent orchestration allows organizations to handle increased demand without compromising performance or accuracy.

## AI agent orchestration challenges

AI agent orchestration comes with several challenges, but each has potential solutions. By addressing these challenges, AI agent orchestration can be more efficient, scalable and resilient.

**Multi-agent dependencies**: When deploying multi-agent frameworks, there is a risk of malfunction. Systems built on the same foundation models may be susceptible to shared vulnerabilities, which might lead to a widespread failure of all agents that are involved or make them more prone to external attacks. This underscores the importance of data governance in building foundation models and thorough training and testing processes.

**Coordination and communication**: If agents don’t interact properly, they can end up working against each other or duplicating efforts. To prevent this, it’s important to establish clear protocols, standardized APIs and reliable message-passing systems to keep everything running smoothly.

**Scalability**: As the number of AI agents increases, maintaining system performance and manageability becomes more complex. A poorly designed orchestration system may struggle with increased workloads, leading to delays or system failures. This can be avoided by using decentralized or hierarchical orchestration models that distribute decision-making, preventing a single point of failure or congestion.

**Decision-making complexity**: In multi-agent environments, determining how tasks should be allocated and executed can become highly complex. Without a clear structure, agents may struggle to make decisions, particularly in dynamic environments where conditions frequently change. Reinforcement learning, prioritization algorithms and predefined roles can help ensure that agents can autonomously determine their tasks while maintaining efficiency.

**Fault tolerance**: What happens if an agent or the orchestrator itself fails? Fault tolerance is crucial and needs to be reinforced by designing failover mechanisms, redundancy strategies and self-healing architectures that allow the system to recover automatically without human intervention.

**Data privacy and security**: AI agents frequently process and share sensitive information, raising concerns about data security and privacy. To mitigate these risks, organizations should implement strong encryption protocols, enforce strict access controls and use federated learning techniques that allow AI models to improve collaboratively without exposing raw data.

**Adaptability and learning**: AI agents must continuously adapt to new tasks and challenges. Systems that require constant manual updates can become inefficient and costly to maintain. To enhance adaptability, machine learning techniques, continuous monitoring and feedback loops can be integrated into the orchestration process. These methods enable AI agents to refine their behavior over time, improving individual and system-wide performance without requiring frequent human intervention.

</details>

<details>
<summary>what-is-ai-agent-planning-ibm</summary>

AI agent planning refers to the process by which an artificial intelligence (AI) agent determines a sequence of actions to achieve a specific goal. It involves decision-making, goal prioritization and action sequencing, often using various planning algorithms and frameworks.

Planning is a module common to many types of agents that exists alongside other modules such as perception, reasoning, decision-making, action, memory, communication and learning. Planning works in conjunction with these other modules to help ensure that agents achieve outcomes desired by their designers.

Not all agents can plan. Unlike simple reactive agents that respond immediately to inputs, planning agents anticipate future states and generate a structured action plan before execution. This makes AI planning essential for automation tasks that require multistep decision-making, optimization and adaptability.

Advances in large language models (LLMs) such as OpenAI’s GPT and related techniques involving machine learning algorithms resulted in the generative AI (gen AI) boom of recent years, and further advancements have led to the emerging field of autonomous agents.

By integrating tools, APIs, hardware interfaces and other external resources, agentic AI systems are increasingly autonomous, capable of real-time decision-making and adept at problem-solving across various use cases.

Complex agents can’t act without making a decision, and they can’t make good decisions without first making a plan. Agentic planning consists of several key components that work together to encourage optimal decision-making.

### Goal definition

The first and most critical step in AI planning is defining a clear objective. The goal serves as the guiding principle for the agent’s decision-making process, determining the end state it seeks to achieve. Goals can either be static, remaining unchanged throughout the planning process, or dynamic, adjusting based on environmental conditions or user interactions.

For instance, a self-driving car might have a goal of reaching a specific destination efficiently while adhering to safety regulations. Without a well-defined goal, an agent would lack direction, leading to erratic or inefficient behavior.

If the goal is complex, agentic AI models will break it down into smaller, more manageable sub-goals in a process called task decomposition. This allows the system to focus on complex tasks in a hierarchical manner.

LLMs play a vital role in task decomposition, breaking down a high-level goal into smaller subtasks and then executing those subtasks through various steps. For instance, a user might ask a chatbot with a natural language prompt to plan a trip.

The agent would first decompose the task into components such as booking flights, finding hotels and planning an itinerary. Once decomposed, the agent can use application programming interfaces (APIs) to fetch real-time data, check pricing and even suggest destinations.

### State representation

To plan effectively, an agent must have a structured understanding of its environment. This understanding is achieved through state representation, which models the current conditions, constraints and contextual factors that influence decision-making.

Agents have some built-in knowledge from their training data or datasets representing previous interactions, but perception is required for agents to have a real-time understanding of their environment. Agents collect data through sensory input, allowing it to model its environment, along with user input and data describing its own internal state.

The complexity of state representation varies depending on the task. For example, in a chess game, the state includes the position of all pieces on the board, while in a robotic navigation system, the state might involve spatial coordinates, obstacles and terrain conditions.

The accuracy of state representation directly impacts an agent’s ability to make informed decisions, as it determines how well the agent can predict the outcomes of its actions.

### Action sequencing

Once the agent has established its goal and assessed its environment, it must determine a sequence of actions that will transition it from its current state to the desired goal state. This process, known as action sequencing, involves structuring a logical and efficient set of steps that the agent must follow.

The agent needs to identify potential actions, reduce that list to optimal actions, prioritize them and identifying dependencies between actions and conditional steps based on potential changes in the environment. The agent might allocate resources to each step in the sequence, or schedule actions based on environmental constraints.

For example, a robotic vacuum cleaner needs to decide the most effective path to clean a room, ensuring it covers all necessary areas without unnecessary repetition. If the sequence of actions is not well planned, the AI agent might take inefficient or redundant steps, leading to wasted resources and increased execution time.

The ReAct framework is a methodology used in AI for handling dynamic decision-making. In the ReAct framework, reasoning refers to the cognitive process where the agent determines what actions or strategies are required to achieve a specific goal.

This phase is similar to the planning phase in agentic AI, where the agent generates a sequence of steps to solve a problem or fulfill a task. Other emerging frameworks include ReWOO, RAISE and Reflexion, each of which has its own strengths and weaknesses.

### Optimization and evaluation

AI planning often involves selecting the most optimal path to achieving a goal, especially when multiple options are available. Optimization helps ensure that an agent's chosen sequence of actions is the most efficient, cost-effective or otherwise beneficial given the circumstances. This process often requires evaluating different factors such as time, resource consumption, risks and potential rewards.

For example, a warehouse robot tasked with retrieving items must determine the shortest and safest route to avoid collisions and reduce operational time. Without proper optimization, AI agents might execute plans that are functional but suboptimal, leading to inefficiencies. Several methods can be used to optimize decision-making, including:

#### Heuristic search

Heuristic search algorithms help agents find optimal solutions by estimating the best path toward a goal. These algorithms rely on heuristic functions—mathematical estimates of how close a given state is to the desired goal. Heuristic searches are particularly effective for structured environments where agents need to find optimal paths quickly.

#### Reinforcement learning

Reinforcement learning enables agents to optimize planning through trial and error, learning which sequences of actions lead to the best outcomes over time. An agent interacts with an environment, receives feedback in the form of rewards or penalties, and refines its strategies accordingly.

#### Probabilistic planning

In real-world scenarios, AI agents often operate in uncertain environments where outcomes are not deterministic. Probabilistic planning methods account for uncertainty by evaluating multiple possible outcomes and selecting actions with the highest expected utility.

### Collaboration

Single agent planning is one thing, but in a multiagent system, AI agents must work autonomously while interacting with each other to achieve individual or collective goals.

The planning process for AI agents in a multiagent system is more complex than for a single agent because agents must not only plan their own actions but also consider the actions of other agents and how their decisions interact with those of others.

Depending on the agentic architecture, each agent in the system typically has its own individual goals, which might involve accomplishing specific tasks or maximizing a reward function. In many multiagent systems, agents need to work together to achieve shared goals.

These goals could be defined by an overarching system or emerge from the agents’ interactions. Agents need mechanisms to communicate and align their goals, especially in cooperative scenarios. This could be done through explicit messaging, shared task definitions or implicit coordination.

Planning in multiagent systems can be centralized, where a single entity or controller—likely an LLM agent—generates the plan for the entire system.

Each agent receives instructions or plans from this central authority. It can also be decentralized, where agents generate their own plans but work collaboratively to help ensure that they align with each other and contribute to global objectives, often requiring communication and negotiation.

This collaborative decision-making process enhances efficiency, reduces biases in task execution, helps to avoid hallucinations through cross-validation and consensus-building and encourages the agents to work toward a common goal.

The phases in agentic AI workflows do not always occur in a strict step-by-step linear fashion. While these phases are often distinct in conceptualization, in practice, they are frequently interleaved or iterative, depending on the nature of the task and the complexity of the environment in which the agent operates.

In a typical agentic workflow, the next phase after planning is action execution, where the agent carries out the actions defined in the plan. This involves performing tasks and interacting with external systems or knowledge bases with retrieval augmented generation (RAG), tool use and function calling ( tool calling).

Building AI agents for these capabilities might involve LangChain. Python scripts, JSON data structures and other programmatic tools enhance the AI’s ability to make decisions.

After executing plans, some agents can use memory to learn from their experiences and iterate their behavior accordingly.

In dynamic environments, the planning process must be adaptive. Agents continuously receive feedback about the environment and other agents’ actions and must adjust their plans accordingly. This might involve revising goals, adjusting action sequences, or adapting to new agents entering or leaving the system.

When an agent detects that its current plan is no longer feasible (for example, due to a conflict with another agent or a change in the environment), it might engage in replanning to adjust its strategy. Agents can adjust their strategies using chain of thought reasoning, a process where they reflect on the steps needed to reach their objective before taking action.

</details>
