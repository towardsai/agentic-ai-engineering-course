## Global Context of the Lesson

### What We Are Planning to Share

This article introduces two of the foundational ingredients of agentic behavior: planning and reasoning. We discuss why LLMs inherently lack default planning capabilities, necessitating an "orchestrating agent" structure. We explore historically important yet still relevant planning/reasoning strategies like ReAct and Plan-and-Execute, explaining their value in structuring agent thought processes, even as modern models (like o3/o4-mini) internalize some of these abilities. We also cover how agents decompose goals and can self-correct.

### Why We Think It's Valuable

Understanding how to imbue LLMs with planning and reasoning capabilities is very important for AI engineers aiming to build autonomous agents that can tackle complex, multi-step tasks. While advanced models are increasingly capable, grasping these fundamental patterns provides deeper insight into agent design, debugging, and the evolution of AI, allowing engineers to build more robust and intelligent systems.

### Expected Length of the Lesson

**1800-2000 words** (without the titles and references), where we assume that 200-250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

100% theory - 0% real-world examples

## Achoring the Lesson in the Course

### Details About the Course

This piece is part of a broader course on AI agents and LLM workflows. The course consists of 3 parts, each with multiple lessons.

Thus, it's essential to always anchor this piece into the broader course, understanding where the reader is in its journey. You will be careful to consider the following:
- The points of view.
- To not reintroduce concepts already taught in the previous lesson.
- To be careful when talking about concepts introduced only in future lessons.
- To always reference previous and future lessons when discussing topics outside the piece's scope.

### Lesson Scope

This is lesson 7 (from part 1) of the course on AI Agents.

### Point of View

The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use 'we,' 'our,' and 'us' to refer to the team who creates the course, and 'you' or 'your' to address the reader. Avoid singular first person and don't use 'we' to refer to the student.

Examples of correct point of view:
- Instead of "Before we can choose between workflows and agents, we need a clear understanding of what they are." word it as "To choose between workflows and agents, you need a clear understanding of what they are."

### Who Is the Intended Audience

Aspiring AI engineers who are learning about planning and reasoning for agents for the first time.

### Concepts Introduced in Previous Lessons

In previous lessons of the course, we introduced the following concepts:

**Part 1:**

- **Lesson 1 - AI Engineering & Agent Landscape**: Understanding the role, the stack, and why agents matter now
- **Lesson 2 - Workflows vs. Agents**: Grasping the crucial difference between predefined logic and LLM-driven autonomy
- **Lesson 3 - Context Engineering**: The art of managing information flow to LLMs
- **Lesson 4 - Structured Outputs**: Ensuring reliable data extraction from LLM responses
- **Lesson 5 - Basic Workflow Ingredients**: Implementing chaining, routing, parallel and the orchestrator-worker patterns
- **Lesson 6 - Agent Tools & Function Calling**: Giving your LLM the ability to take action

As this is only the 7th lesson of the course, we haven't introduced too many concepts. At this point, the reader knows what an LLM is and a few high-level ideas about the LLM workflows and AI agents landscape.

### Concepts That Will Be Introduced in Future Lessons

In future lessons of the course, we will introduce the following concepts:

**Part 1:**

- **Lesson 8 - Implementing ReAct**: Building a reasoning agent from scratch
- **Lesson 9 - Agent Memory & Knowledge**: Short-term vs. long-term memory (procedural, episodic, semantic)
- **Lesson 10 - RAG Deep Dive**: Advanced retrieval techniques for knowledge-augmented agents
- **Lesson 11 - Multimodal Processing**: Working with documents, images, and complex data

**Part 2:**

- MCP
- Developing the research agent and the writing agent

**Part 3:**

- Making the research and writing agents ready for production
- Monitoring
- Evaluations

### Anchoring the Reader in the Educational Journey

Within the course we are teaching the reader multiple topics and concepts. Thus, understanding where the reader is in its educational journey is critical for this piece. You have to use only previously introduced concepts, while being reluctant about using concepts that haven't been introduced yet.

When discussing the concepts introduced in previous lessons listed in the `Concepts Introduced in Previous Lessons` section, avoid reintroducing them to the reader. Especially don't reintroduce acronyms. Use them as if the reader already knows what they are.

Avoid using all the concepts that haven't been introduced in previous lessons listed in the `Concepts That Will Be Introduced in Future Lessons` subsection. Whenever another concept requires references to these banned concepts, instead of directly using them, use intuitive analogies or explanations that are more general and easier to understand, as you would explain them to a 7-year-old. For example:
- If the "tools" concept wasn't introduced yet and you have to talk about agents, refer to them as "actions".
- If the "routing" concept wasn't introduced yet and you have to talk about it, refer to it as "guiding the workflow between multiple decisions".
You can use the concepts that haven't been introduced in previous lessons listed in the `Concepts That Will Be Introduced in Future Lessons` subsection, only if we explicitly specify them. Still, even in that case, as the reader doesn't know how that concept works, you are just allowed to use the term, while keeping the explanation extremely high-level and intuitive, as if you were explaining it to a 7-year-old.
Whenever you use a concept from the `Concepts That Will Be Introduced in Future Lessons` subsection, explicitly specify in what lesson it will be explained in more detail, leveraging the particulars from the subsection. If not explicitly specified in the subsection, simply state that we will cover it in future lessons without providing a concrete lesson number.

In all use cases avoid using acronyms that aren't explicitly stated in the guidelines. Rather use other more accessible synonyms or descriptions that are easier to understand by non-experts.

## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end-to-end lesson:

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

## Lesson Outline 

1. What a Non-Reasoning Model Does And Why It Fails on Complex Tasks
2. Teaching Models to “Think” Chain-of-Thought and Its Limits
3. Separating Planning from Answering Foundations of ReAct and Plan-and-Execute
4. ReAct in Depth Loop, Evolving Example, Pros and Cons
5. Plan-and-Execute in Depth Plan, Execution, Pros and Cons
6. Where This Shows Up in Practice Deep Research–Style Systems
7. Modern Reasoning Models Thinking vs Answer Streams and Interleaved Thinking
8. Advanced Agent Capabilities Enabled by Planning Goal Decomposition and Self-Correction

## Section 1 - What a Non-Reasoning Model Does And Why It Fails on Complex Tasks

- Use the recurring example of a "Technical Research Assistant Agent" to frame the problem. The agent must produce a comprehensive technical report on "Latest developments in edge AI deployment" (find recent papers, summarize findings, identify trends and gaps, write a structured report).
- Show how a non-reasoning model behaves: it immediately “answers” without first drafting a plan. It treats the entire task as a single response generation. It may call the right tools in the correct order, without correcting on mistakes or unforeseen results.
    - Consequences for complex tasks:
        - Superficial and weak outputs.
        - No iteration on partial results; it does not analyze its own output to fix problems.
        - No explicit breakdown of sub-goals, so it misses important steps (e.g., verification, cross-source comparison).
- Connect back to prior lessons (briefly, without re-teaching): workflows and structured outputs gave us modularity and reliability where the process is predictable (no need to adjust for unforeseen results); tools enabled actions. Yet, without explicit reasoning and planning, outputs still drift for complex tasks where adaptation is key (which are the tasks more suitable for agents).
- Transition to Section 2: To address this, we first “teach” the model to produce a reasoning trace, i.e. thinking before answering.

- **Section length:** 250-350 words

## Section 2 - Teaching Models to “Think” Chain-of-Thought and Its Limits

- Explain the idea: like humans “talk to themselves,” we can ask an LLM to write a reasoning trace (thinking tokens) before the final answer. This adds planning power and enables iterations on partial solutions.
- Provide a simple chain-of-thought example for the same research task:
    - Prompt idea: “Before answering, think step by step about how you will research and verify sources on edge AI deployment. Then provide the final report.”
    - Expected behavior: the model drafts a high-level plan (search → read → compare → synthesize) and reasons about verification.
- Clarify the limitation:
    - The plan and the answer appear in the same text, which is confusing to parse and hard to control.
    - The model may only write an initial plan but not execute an iterative loop to refine, verify, or correct.
- Transition to Section 3: To gain structure and control, we separate planning/reasoning from answering/action — the foundation of ReAct and Plan-and-Execute.

- **Section length:** 250-350 words

## Section 3 - Separating Planning from Answering Foundations of ReAct and Plan-and-Execute

- Present the core idea: ask the model to (1) plan/reason and (2) produce the answer or call a tool as two distinct phases or interleaved steps.
- Why this helps:
    - Clear separation yields control and interpretability.
    - Enables iterative loops where observations can update the plan.
    - Allows different handling for reasoning traces vs. final outputs.
- Positioning:
    - ReAct interleaves Thought → Action → Observation in a loop.
    - Plan-and-Execute separates a Planning phase from an Execution phase.
- Transition to Section 4: First, we go deep into ReAct using the evolving research-assistant example.

- **Section length:** 150-250 words

## Section 4 - ReAct in Depth Loop, Evolving Example, Pros and Cons

- Historical context and motivation: ReAct emerged to bridge free-form reasoning (chain-of-thought) and action-only paradigms by interleaving both with environment feedback.
- Explain the loop clearly and provide a diagram deliverable:
    - Deliverable: Mermaid diagram of the ReAct loop (Thought → Action → Observation → Thought … → Final Answer).
- Evolving example (continue the "Technical Research Assistant Agent"):
    - Thought: “I need recent, trustworthy sources on edge AI deployment.”
    - Action: search("latest developments in edge AI deployment 2024 site:arxiv.org OR site:nature.com")
    - Observation: returns a list of candidate papers.
    - Thought: “Select 3 highly cited, 1 industry report; check publication year and venue.”
    - Action: fetch_and_extract(paper_urls)
    - Observation: extracted abstracts and metadata.
    - Thought: “Summarize per source and compare claims about adoption rates; flag conflicts.”
    - Action: summarize_and_compare(extractions)
    - Observation: Paper A says 40% adoption; Paper B says 25%.
    - Thought: “Conflict detected; find a third-party market analysis to adjudicate.”
    - Action: search("edge AI deployment adoption market analysis 2024")
    - Observation: credible report found.
    - Thought: “Resolve conflict using the report; finalize trends and gaps.”
    - Final Answer: structured report with citations and resolved statistics.
- Pros:
    - High interpretability; natural error recovery via observations.
    - Stays on track for exploratory tasks; supports incremental verification.
- Cons:
    - Potentially slower; requires tooling and loop control.
    - Can be less predictable; needs guardrails and good system prompts.
- Transition to Section 5: For tasks with known structure, Plan-and-Execute can be more efficient and predictable.

- **Section length:** 500-600 words

## Section 5 - Plan-and-Execute in Depth Plan, Execution, Pros and Cons

- Core concept: produce a plan upfront; then execute sequentially or in controlled parallel, updating the plan only when needed.
- Provide a diagram deliverable:
    - Deliverable: Mermaid diagram with distinct Planning and Execution phases.
- Evolving example (same agent):
    - Planning phase output:
        1) Define scope and success criteria. 2) Search across academic and industry sources.
        3) Select top N sources by relevance and quality. 4) Summarize each source.
        5) Compare findings; identify trends and gaps. 6) Draft outline.
        7) Write the report; include citations and a methodology note.
    - Execution phase:
        - Step through the plan: search → select → extract → summarize → compare → outline → write.
        - Plan refinement triggers: missing data, conflicts, or low-quality sources.
- Pros:
    - Upfront structure improves efficiency and reliability for well-defined tasks.
    - Easier to bound cost/time; clearer stage ownership.
- Cons:
    - Less flexible for highly exploratory problems; may need frequent re-planning.
    - Risk of rigid adherence to an imperfect initial plan.
- Transition to Section 6: These ideas power real systems like Deep Research, which operationalize iterative planning and verification at scale.

- **Section length:** 450-550 words

## Section 6 - Where This Shows Up in Practice Deep Research–Style Systems

- Describe how research-style systems (e.g., Deep Research) operationalize planning/reasoning:
    - Long-horizon tasks decomposed into sub-goals.
    - Iterative search → read → compare → verify → write cycles.
    - Explicit prompts and policies to reduce hallucinations and enforce verification.
- Connect to the recurring example: the agent performs many micro-cycles of searching, extracting, verifying stats (e.g., adoption rates), and updating its notes before writing.
- Tie back to patterns:
    - Many such systems are ReAct-like loops with strong prompts and tools.
    - Others are closer to Plan-and-Execute with periodic re-planning.
- Transition to Section 7: Modern reasoning models make some of this behavior more implicit by generating “thinking” and “answer” streams.

- **Section length:** 200-300 words

## Section 7 - Modern Reasoning Models Thinking vs Answer Streams and Interleaved Thinking

- Explain how modern reasoning models are trained to integrate planning directly:
    - Two streams: a private "thinking" stream and a public "answer" stream.
    - Some models “think first,” then perform tool calls and produce the final output.
    - Others support interleaved thinking: after each tool call or written chunk, they emit new thinking tokens to update the plan.
- Implications for system design:
    - You may rely less on explicit loops, but you still need guardrails, prompts, and state to ensure reliability.
    - Separation of reasoning and answering remains useful for debugging and control, even with strong implicit planning.
- Transition to Section 8: With planning and reasoning in place, agents unlock advanced capabilities like goal decomposition and self-correction.

- **Section length:** 250-350 words

## Section 8 - Advanced Agent Capabilities Enabled by Planning Goal Decomposition and Self-Correction

- Goal decomposition:
    - Break tasks into sub-goals and sub-sub-goals; choose order and tactics.
    - In ReAct-style agents, decomposition often occurs implicitly during Thought steps; prompts can nudge this behavior.
- Self-correction:
    - Detect failures, contradictions, or low-confidence results; update the plan.
    - Continue the recurring example: conflicting adoption rates (40% vs 25%). The agent inserts a “verification” sub-goal, finds a market analysis, reconciles numbers, and revises the report.
    - Techniques: re-prompt with error info, alternative actions, re-evaluate plan, ask for clarification.
- Why patterns still matter with strong models:
    - They improve debuggability (trace Thought/Action/Observation), consistency (explicit control loops), and pedagogy (shared mental model of agent thinking).
- Connect to next lessons:
    - Next (Lesson 8), we implement ReAct from scratch.
    - Soon after: memory systems (Lesson 9), knowledge-augmented retrieval (Lesson 10), multimodal processing (Lesson 11).

- **Section length:** 450-550 words

## Article Code

This is a theory-only lesson, so there is no code associated with it.

## Golden Sources

- [Agentic Reasoning - IBM](https://www.ibm.com/think/topics/agentic-reasoning)
- [AI Agent Orchestration - IBM](https://www.ibm.com/think/topics/ai-agent-orchestration)
- [ReAct Agent - IBM](https://www.ibm.com/think/topics/react-agent)
- [Building effective agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629)

## Other Sources

- [AI Agents in 2025: Expectations vs Reality - IBM](https://www.ibm.com/think/insights/ai-agents-2025-expectations-vs-reality)
- [Reasoning AI Agents Transform Decision Making - NVIDIA](https://blogs.nvidia.com/blog/reasoning-ai-agents-decision-making/)
- [From LLM Reasoning to Autonomous AI Agents - arXiv](https://arxiv.org/pdf/2504.19678)
- [A practical guide to building agents - OpenAI](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- [AI Agent Planning - IBM](https://www.ibm.com/think/topics/ai-agent-planning)
- [ReAct - Google](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models)
- [Measuring AI Ability to Complete Long Tasks - METR](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)
- [Interleaved Thinking for Reasoning LLMs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#interleaved-thinking)