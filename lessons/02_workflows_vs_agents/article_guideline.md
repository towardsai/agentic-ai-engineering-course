## Global Context

- **What I’m planning to share:** This article will define and clearly contrast LLM Workflows (predefined, orchestrated steps) with Agentic Systems (dynamic, LLM-directed processes), drawing on distinctions like those from Anthropic. We'll explore use cases where each approach is most effective, emphasizing the core difference: developer-defined logic versus LLM-driven autonomy. The necessity of orchestration for both will be discussed. Furthermore, we'll analyze the design and capabilities of prominent, state-of-the-art agent examples (as of 2025, such as Deep Research agents, Devin, and Codex-like systems), deconstructing their operational mechanisms (e.g., planning, tool use loops, memory) and highlighting common patterns and challenges.
- **Why I think it’s valuable:** For an AI Engineer, choosing the right architectural approach—workflow or agent—is a critical early decision that impacts complexity, flexibility, and suitability for the AI project or product. Understanding this distinction, along with the mechanics of cutting-edge agents, provides both a conceptual framework and a practical inspiration and intuition for designing sophisticated AI solutions.
- **Who is the intended audience:** Engineers learning for the first time about the specifics of LLM workflows, AI agents and how they are different.
- **Article length:** 3000 words (equal to a 15-minute read for someone who reads 200 words per minute)


## Outline

1. Defining the Spectrum: LLM Workflows vs. Agentic Systems
2. Choosing Your Path: Use Cases and Considerations
3. State-of-the-Art Agent Examples (2025) - A Closer Look
4. Deconstructing Advanced Agent Mechanisms
5. Common Patterns and Enduring Challenges in Agent Design**

### Section 1: Defining the Spectrum: LLM Workflows vs. Agentic Systems

- **LLM Workflows:**
	- Definition: Systems where a sequence of tasks, potentially involving LLM calls, is largely predefined and orchestrated by developer-written code.
	- Characteristics: Deterministic or rule-based paths, predictable execution, explicit control flow.
	- Analogy: A well-defined assembly line.
	- Reference: Briefly mention Anthropic's distinctions if they provide a clear framework.
	- Attach an image with a simple LLM Workflow.
- **Agentic Systems:**
	- Definition: Systems where an LLM (or multiple LLMs) plays a central role in dynamically deciding the sequence of steps, reasoning, and actions to achieve a goal.
	- Characteristics: Adaptive, capable of handling novelty, LLM-driven autonomy in decision-making and execution path.
	- Analogy: A skilled human expert tackling an unfamiliar problem.
	- Attach an image of how a simple Agentic System looks.
- **The Core Difference:** Emphasize developer-defined logic (workflows) versus LLM-driven autonomy in reasoning and action selection (agents).
- **The Role of Orchestration:** Explain that both workflows and agents require an orchestration layer, but their nature differs. In workflows, it executes a defined plan; in agents, it facilitates the LLM's dynamic planning and execution.
- **Section length:** 600 words


### Section 2: Choosing Your Path: Use Cases and Considerations
- **When to Use LLM Workflows:**
	- Examples: Structured data extraction and transformation, automated report generation from templates, content summarization followed by translation, form processing and content generation, such as articles or blogs (where the structure is well-defined and requires minimal human feedback).
	- Strengths: Predictability, reliability for well-defined tasks, easier debugging of fixed paths, potentially lower operational cost if simpler models can be used for sub-tasks.
- **When to Use Agentic Systems:**
	- Examples: Open-ended research and synthesis, dynamic problem-solving (e.g., debugging code, complex customer support), interactive task completion in unfamiliar environments, and creative content generation requiring iterative refinement (where the structure isn't well defined and needs more human feedback).
	- Strengths: Adaptability to new situations, flexibility to handle ambiguity and complexity as the steps are dynamically decided and potential for emergent solutions.
- **Hybrid Approaches:** Most real-world systems might blend elements of both. Thus, in reality, we have a spectrum, a gradient between LLM Workflows and Agentic Systems, where a system adopts what's best from both worlds depending on its use cases.
- Highlight that when building an application you usually have an "autonomy slider" where you decide how much control you have to give to the user. As you go more manual you usually use an LLM workflow together with a human that verifies intermediate steps. As you go more automatic you give more control to the agent with less human in the loop steps. Use the Cursor (CMD+K, CMD+L, CMD+I) and Perplexity (search, research, deep research) examples from the "Andrej Karpathy: Software Is Changing (Again)" resource.
- The ultimate goal is to speed up the AI generation <-> Human verification loop, which is often done through good workflows/agentic architecture and well designed UI/UX platforms (e.g., Cursor for coding).
- Attach an image showing the gradient between the two worlds. 
- **Section length:** 600 words


### Section 3: State-of-the-Art Agent Examples (2025) - A Closer Look
- Introduce a selection of prominent, SOTA agent examples, such as Deep Research agents (from OpenAI or Perplexity), Coding agents (Codex, Anthropic CLI, Cursor, or Windsurf), OpenAI's Operator concept, or other relevant 2025 examples.
- For each agent:
	- Briefly describe the problem it solves and its functionality. 
	- Highlight what makes it an agent or workflow based on the definitions in Section 1.
	- Discuss its potential impact or novelty.
-  **Section length:** 300 words (keep the section short as we will dig in more details in the next section)

### Section 4: Deconstructing Advanced Agent Mechanisms
- **Codex (or a similar advanced code agent):**
	- Explain its likely operational loop:
		1. **Goal Understanding & Planning:** Decomposing the high-level coding task.
		2. **Environment Interaction:** Reading files, understanding project structure.
		3. **Code Generation/Modification:** Writing or editing code.
		4. **Execution & Testing:** Running the code, using a linter, running tests.
		5. **Debugging & Iteration:** Analyzing errors, modifying the plan, and trying again.
	- Tools used: File system access, code interpreter, web search (for documentation/solutions), version control.
	- Create a simple mermaid diagram supporting the idea.
- **Deep Research Agents (e.g., for scientific discovery or market research):**
	- Explain the iterative multi-step process:
		1. **Query Formulation & Planning:** Defining research questions, planning search strategy.
		2. **Iterative Search:** Using search engine tools, accessing databases.
		3. **Information Extraction & Filtering:** Identifying relevant information from sources.
		4. **Synthesis & Analysis:** Combining information, identifying patterns, drawing conclusions.
		5. **Citation & Reporting:** Generating a structured output with proper sourcing.
	* Create a simple mermaid diagram supporting the idea.
- **OpenAI's Operator (or similar computer control agents):**
	- Explain the concept: Agents designed to operate computer applications or websites via GUIs or OS-level commands.
	- Key mechanisms: Vision capabilities (to understand screens), action mapping (mouse clicks, keyboard inputs), planning to achieve user goals within applications.
	- Challenges: Robustness to UI changes, interpreting visual information accurately.
	- Create a simple mermaid diagram supporting the idea.
-  **Section length:** 500 words

### Section 5: Common Patterns and Enduring Challenges in Agent Design**
* Set the scene in the world of Agentic Systems by presenting a brief summary of a common architecture of an AI agent. It will include common architectural patterns, such as:
	- Planning (e.g., task decomposition, goal setting).
	- Tool Use (as a fundamental way to interact and act).
	- Memory (short-term for context, long-term for learning/knowledge).
	- Iterative Refinement / Self-Correction loops.
- Draw one or more Mermaid diagrams illustrating how the architectural patterns from above work together and form an agent 
- The goal of this section is not for people to fully understand what an AI agent is, but just to build an intuition on what it takes to create one. In future articles, we will dig into all the necessary details. 
- Conclude the section (and article) with some sentences on common issues encountered while building AI agents that require careful design choices, such as:
	- Reliability and consistency of LLM reasoning are compromised as the errors from each decision compound. 
	- Handling long-context and maintaining coherence over many steps.
	- Effective error detection and recovery.
        - Scalability and cost of complex agent operations.
        - Security implications of autonomous agents with powerful tools.
        - Evaluation of open-ended agent performance.
- **Section length:** 1000 words

## References

[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
[Real Agents vs. Workflows: The Truth Behind AI 'Agents'](https://www.youtube.com/watch?v=kQxr-uOxw2o&t=1s)
[Exploring the difference between agents and workflows](https://decodingml.substack.com/p/llmops-for-production-agentic-rag)
[Building Production-Ready RAG Applications: Jerry Liu](https://www.youtube.com/watch?v=TRjq7t2Ms5I)
[Andrej Karpathy: Software Is Changing (Again)](https://www.youtube.com/watch?v=LCEmiRjPEtQ)

