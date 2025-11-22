## Outline

1. Introduction: The Critical Decision Every AI Engineer Faces
2. Understanding the Spectrum: From Workflows to Agents
3. Choosing Your Path
4. Exploring Common Patterns
5. Zooming In on Our Favorite Examples
6. The Challenges of Every AI Engineer

## Section 1 - Introduction: The Critical Decision Every AI Engineer Faces

- **The Problem:** When building AI applications, engineers face a critical architectural decision early in their development process. Should they create a predictable, step-by-step workflow where they control every action, or should they build an autonomous agent that can think and decide for itself? This is one of the key decisions that will impact everything from the product such as development time and costs to reliability and user experience.
- **Why This Decision Matters:** Choose the wrong approach and you might end up with:
  - An overly rigid system that breaks when users deviate from expected patterns or developers try to add new features
  - An unpredictable agent that works brilliantly 80% of the time but fails catastrophically when it matters most
  - Months of development time wasted rebuilding the entire architecture
  - Frustrated users who can't rely on the AI application
  - Frustrated executives who cannot affort to keep the AI agent running as the costs are too high relative to the profits
- Make a quick reference to the real-world where in 2024-2025 billion-dollar AI startups succeed or fail based primarily on this architectural decision. The successful companies, teams and AI engineers know when to use workflows versus agents, and more importantly, how to combine both approaches effectively.
- **Quick walkthrough of what we'll learn by the end of this lesson:** Take the core ideas of what we'll learn in the lesson from the `What We Are Planning to Share` subsection and highlight the importance and existence of the lesson from `Why We Think It's Valuable` subsection of the `Global Context of the Lesson` section.
- **Section length:** 300 words

## Section 2 - Understanding the Spectrum: From Workflows to Agents

- In this section we want to take a brief look at what LLM workflows and AI agents are. At this point we don't focus on the technical specifics of each, but rather on their properties and how they are used.
- On **LLM workflows** we care about:
	- Definition: A sequence of tasks involving LLM calls or other operations such as reading/writing data to a database or file system. It is largely predefined and orchestrated by developer-written code.
	- Characteristics: The steps are defined in advance, resulting in deterministic or rule-based paths with predictable execution and explicit control flow. 
	- Analogy: A factory assembly line.
	- Concepts we will learn in future lessons: chaining, routing, orchestrator-worker patterns
	- Attach an image from the research with a simple LLM Workflow.
- On **AI agents** we care about:
	- Definition: Systems where an LLM (or multiple LLMs) plays a central role in dynamically deciding (planning) the sequence of steps, reasoning, and actions to achieve a goal. The steps are not defined in advance, but are dynamically planned based on the task and current state of the environment.
	- Characteristics: Adaptive, capable of handling novelty, LLM-driven autonomy in decision-making and execution path.
	- Analogy: A skilled human expert tackling an unfamiliar problem adapting on the moment after each "Eurika" moment
	- Concepts we will learn in future lessons: tools, memory, and ReAct agents
	- Attach an image from the research of how a simple Agentic System looks.
- **The Role of Orchestration:** Explain that both workflows and agents require an orchestration layer, but their nature differs. In workflows, it executes a defined plan; in agents, it facilitates the LLM's dynamic planning and execution.
- **Section length:** 400 words

## Section 3: Choosing Your Path

- In the previous section we defined the LLM workflows and AI agents independently, now we want to explore their core differences: Developer-defined logic vs LLM-driven autonomy in reasoning and action selection.
- Attach an image from the research showing the gradient between LLM workflows and AI agents.
- **When to use LLM workflows:**
	- Examples where the structure is well-defined:
		- Pipelines for data extraction and transformation from sources such as the web, messaging tools like Slack, video calls from Zoom, project management tools like Notion, and cloud storage tools like Google Drive
		- Automated report or emails generation from multiple data sources
		- Understanding project requirements and creating or updating tasks in Notion project management tools
		- Document summarization followed by translation
		- Repetitive daily tasks: Sending emails, posting social media updates, responding to messages
		- Content generation or repurposing, such as transforming articles into social media posts 
	- Strengths: Predictability, reliability for well-defined tasks, easier debugging of fixed paths, potentially lower operational costs as we can leverage simpler and smaller models specialized in given sub-tasks. Because the workflows are predictable, the costs and latency are more predictable. Ultimately, because we can leverage smaller models, the infrastructure overhead is smaller.
	- Weaknesses: Potentially more development time required as each step is manually engineered. The user experience is rigid as it cannot handle unexpected scenarios. Adding new features can get complex when the application grows, similar to developing standard software tools.
	- Usually preferred in enterprises or regulated fiels as they require predictable programs that work all the time. For example, in finance, when a financial advisor asks for a financial report, it should contain the right information all the time, as it has a direct impact on people's money and life. Another domain where workflows are preferred is in the health space, where for AI tools to be used in production, they require working with high-accuracy all the time, as they have a direct impact on people's lives.
	- Ideal for MVPs requiring rapid deployment by hardcoding features
	- Best for scenarios where cost per request matters more than sophisticated reasoning (thousands of requests per minute)
- **When to use AI agents:**
	- Examples: 
		- Open-ended research and synthesis (e.g., researching about WW2)
		- Dynamic problem-solving (e.g., debugging code, complex customer support)
		- Interactive task completion in unfamiliar environments (e.g., booking a flight, where we don't specify the exact sites to use)
	- Strengths: Adaptability to new situations and the flexibility to handle ambiguity and complexity as the steps are dynamically decided.
	- Weaknesses: The system is more prone to errors. As the agent is non-deterministic, the performance, latency and costs can vary with each call of the agent, making agents often unreliable. As agents require LLMs that can generalize better, which are bigger, hence more costly, adopting an agentic solution usually ends up being more costly. AI agents usually require more LLM calls to understand the user intent and take various actions, which can result again in bigger costs per call. If not designed well, there can be huge security concerns, especially on write operations, where it can delete all our data or send inappropriate emails. Ultimately, a huge disadvantage of AI agents is that they are hard to debug and evaluate.
	- Funny story on the current issues with agents: People had their code deleted by Replit Agent or Claude Code and making jokes about it as "Anyway I wanted to start a new project."
- **Hybrid Approaches:** Most real-world systems blend elements of both approaches. Thus, in reality, we have a spectrum, a gradient between LLM workflows and AI agents, where a system adopts what's best from both worlds depending on its use cases.
- Highlight that when building an application you usually have an "autonomy slider" where you decide how much control to give to the LLM versus the user. As you go more manually, you usually use an LLM workflow together with a human that verifies intermediate steps. As you go more automatically, you give more control to the agent with fewer human-in-the-loop steps. Use the Cursor (CMD+K, CMD+L, CMD+I) and Perplexity (search, research, deep research) examples from the Andrej Karpathy "Software Is Changing (Again)" resource.
- The ultimate goal is to speed up the AI generation <-> Human verification loop, which is often achieved through good workflows/agentic architecture and well-designed UI/UX platforms (e.g., Cursor for coding).
- Generate a mermaid to illustrate the AI generation and human verification loop
- **Section length:** 500 words

## Section 4: Exploring Common Patterns

- To introduce the reader to the AI Engineering world, we will present the most common patterns used to build AI agents and LLM workflows. Explain them as if this is the first time the reader hears about them.
- LLM workflows:
	- **Chaining and routing** to automate together multiple LLM calls. As a first automation step, it helps gluing together multiple LLM calls and deciding between multiple appropriate options. (Draw a mermaid diagram)
	- **Orchestrator-worker** to understand the user intent, dynamically plan and call multiple actions, and synthesize them into a final answer. It allows the AI program to dynamically decide what actions to take based on the given task, making a smooth transition between the workflows and agentic world. (Draw a mermaid diagram)
	- **Evaluator-optimizer loop** used to auto-correct the results from an LLM based on automated feedback. LLM outputs can drastically be improved by providing feedback on what they did wrong. This pattern automates that by having an "LLM reviewer" that analyzes the output from the LLM who generated the answer, creates an error report (also known as a reflextion), and passes it back to the generator to auto-correct itself. Example: A human writer refines a document based on feedback. (Draw a mermaid diagram)
- Core components of a ReAct AI agent:
	- Explain at a high-level that the pattern is used to automatically decide what action to take, interpret the output of the action, and repeat until the given task is completed. This is the core of a ReAct agent.
	- LLM to take actions and interpret outputs from tools
	- tools to take actions within the external environment (more on tools in Lesson 6)
	- short-term memory: The working memory of the agent (make a comparison to how RAM works for computers)
	- long-term memory: This is used to access factual data about the external world (such as public websites from the internet or private data from a company's databases) and remember user preferences (more on memory in Lesson 9)
	- Make a reference that this is how the ReAct pattern works, which we will explain in a lot of detail in Lessons 7 and 8. Highlight that almost all modern agents from the industry use the ReAct pattern as it has shown the most potential.
	- Draw a mermaid diagram showing the dynamics between the core components of an AI agent illustrating the ReAct pattern at a high level
- The goal of this section is not for readers to fully understand how these patterns work, but just to build an intuition on various LLM workflows and AI agents patterns that they will learn during the course. Specify that in future lessons, they will dig into all the necessary details of each pattern.

- **Section length:** 550 words

## Section 5: Zooming In on Our Favorite Examples

- To better anchor the reader in the world of LLM workflows and AI agents we want to introduce some concrete examples, from a simple workflow (e.g., Google Workspace document summarization), to a single agent system (Gemini CLI code assistant) to a more advanced hybrid solution (Perplexity's Deep Research agent). 
- The reader is not yet aware of any complex topics and knows about LLM workflows or AI agents only what was explained previously in this lesson, so keep this section high-level without any fancy acronyms. Explain everything as if speaking to a 7-year-old.

- [Start with a simple LLM workflow example] **Document summarization and analysis workflow by Gemini in Google Workspace**:
	- **Problem:** When working in teams and looking for the right document, it can transform into a time-consuming process, because many documents are large, hence it's hard for us to understand which document contains the right information. Thus, a quick, embedded summarization can guide us and our search strategies. 
	- Create a suggestive mermaid diagram highlighting that this is a workflow.
	- Explain how such a workflow would look. Explain that this is a pure and simple workflow as a chain with multiple LLM calls:
		- read document
		- summarize using an LLM call
		- extract key points using another LLM call
		- save results to a database
		- show results to user
- [Continue with an AI agent example] **Gemini CLI coding assistant:** 
	- **Problem:** Writing code is a time-consuming process. You have to read boring documentation or outdated blogs. When working on new code bases, understanding it is a slow process. When working with a new programming language, to write high-quality code, you first need a bootcamp on it before writing any industry-level code. That's where a coding assistant can help you speed up writing code on existing and new code bases.
	- To build up the intuition on agents, we present at a very high and intuitive level how the Gemini CLI tool leverages the ReAct (Reason and Act) agent architecture to implement a single-agent system for coding. 
	- This is how Gemini CLI works based on our latest research from August 2025. Also, specify that `gemini-cli` is open-sourced on GitHub. Thus, on this particular example, we can be more accurate and specific on the actual implementation.
	- Use cases: 
		- Writing code from scratch, without requiring any coding experience (known as vibe coding).
		- Assisting an engineer to write code faster by writing only specific functions or classes.
		- Support for writing documentation.
		- Helping us quickly understand how new code bases work.
	- Implemented in TypeScript.
	- Similar tools: Cursor, Windsurf, Claude Code, Warp
	- To keep the examples light and intuitive, use parallels such as:
		- tools as actions
		- context as state or working memory
	- Present at a high level how Gemini CLI uses the ReAct pattern to implement a coding assistant:
		1. **Context Gathering**: The system loads the directory structure, available tools, and the conversation history, known as context, into the state.
		2. **LLM Reasoning**: The Gemini model analyzes the user input against the current context to understand what actions it requires to take to adapt the code as requested by the user.
		3. **Human in the Loop**: Before taking any actions it validates the execution plan with the user.
		4. **Tool Execution**: The selected actions, known as tools, are executed. The tools can be things such as file operations to read the current state of the code, web requests to documentation and ultimately generating the code. Then the agent processes the tool outputs, adds the results into the conversation context to reference it in future iterations.
		5. **Evaluation**: The agent dynamically evaluates whether the generated code is correct or not by running or compiling the code.
		6. **Loop Decision**: The agent determines if the task is completed or it should repeat steps 2 to 5 by planning and executing more tools.
	- Create a mermaid diagram showing how the operational loop works.
	- More tool examples: 
		- File system access: 
			- grep functions to read specific functions or classes from the codebase
			- listing the directory structure of the codebase or module
		- Coding:
			- code interpreting
			- generating code diffs
			- executing the generated code for dynamic validation
		- web search:
			- documentation
			- blogs
			- solutions
		- version control, such as git, to automatically commit your code to GitHub or GitLab

- [Wrap-up with a hybrid example between workflows and agents] **Perplexity deep research (e.g., for scientific, financial, social research):**
	- **Problem:** Researching a brand new topic is a scary thing to do. Most of the time we don't know where to start. What is the right blog, paper, YouTube video or course to start reading? Also, for more trivial questions, most of the time, we don't have the time to dig into too many resources. That's why having a research assistant that quickly scans the internet into a report can provide a huge boost in your learning process on scientific, financial, social topics.
	- To build up the intuition on LLM workflows and AI agents hybrid systems we will present how the Perplexity's Deep Research agent at an intuitive and very high level.
	- Perplexity's Deep Research agent is a hybrid system that combines ReAct reasoning with LLM workflow patterns to do autonomous research at expert level. Unlike single-agent approaches, like the Gemini CLI one, this system uses multiple specialized agents that are orchestrated in parallel by workflows, performing dozens of searches across hundreds of sources to synthesize comprehensive research reports within 2-4 minutes.
	- This is how Perplexity Deep Research agent works based on our latest research from August 2025. Also, specify that the solution is closed-source. Thus, everything that we write here is an assumption based on what we could find on the open internet or other people's speculations. Still, it's an amazing use case to understand how hybrid systems work.
	- Here is an oversimplified version of how Perplexity's Deep Research agent could work:
		1. **Research Planning & Decomposition**: The orchestrator analyzes the research question and decomposes it into a set of targeted sub-questions. Highlight how the orchestrator leverages the orchestrator-worker pattern to deploy multiple research agents with different sub-questions.
		2. **Parallel Information Gathering**: For each sub-question, to optimize and move faster in the search space, we run in parallel specialized search agents that leverages tools such as web searches and document retrieval to gather as much information as possible for that specific question. As the research agents are isolated between each other, the input tokens are smaller, helping the LLM to stay focused.
		3. **Analysis & Synthesis**: After gathering bulks of sources, each agent validates and scores each source using strategies such as domain credibility or relevance scoring relative to the query. Then, each source is ranked based on its importance. Ultimately, the top K sources are summarized into a final report.
		4. **Iterative Refinement & Gap Analysis:** The orchestrator gathers the information from all the agents which ran in parallel and tries to identify knowledge gaps relative to the research requested by the user. Based on any potential knowledge gaps it generates follow-up queries by repeating steps 1 and 3 until all the knowledge gaps are filled or, to avoid infinite loops, a max number of steps is reached.
		5. **Report Generation:** The orchestrator takes the results from all the AI agents and generates a final report with inline citations.
	- Create a mermaid diagram showing how the iterative multi-step process works.
	- Highlight how the deep research agent operates as a hybrid between workflows and agents combining structured planning with dynamic adaptation: The workflow uses the orchestrator-worker pattern to dynamically reason, supervise and call in parallel multiple agents specialized in researching only a targeted sub-query until all the user requested research topics are fulfilled.

-  **Section length:** 900 words (without counting the mermaid diagram code)

## Section 6 - Conclusion: The Challenges of Every AI Engineer

- **The Reality of AI Engineering:** Now that you understand the spectrum from LLM workflows to AI agents, it's important to recognize that every AI Engineer—whether working at a startup or a Fortune 500 company faces these same fundamental challenges whenever it has to design a new AI application. These are one of the core decisions that determine whether your AI application succeeds in production or fails spectacularly.
- To set the scene for future lessons and patterns we will learn, present some daily challenges every AI engineer battles:
	- **Reliability Issues:** Your agent works perfectly in demos but becomes unpredictable with real users. LLM reasoning failures can compound through multi-step processes, leading to unexpected and costly outcomes.
	- **Context Limits:** Systems struggle to maintain coherence across long conversations, gradually losing track of their purpose. Ensuring consistent output quality across different agent specializations presents a continuous challenge.
	- **Data Integration:** Building pipelines to pull information from Slack, web APIs, SQL databases, and data lakes while ensuring only high-quality data is passed to your AI system (garbage-in, garbage-out principle).
	- **Cost-Performance Trap:** Sophisticated agents deliver impressive results but cost a fortune per user interaction, making them economically unfeasible for many applications.
	- **Security Concerns:** Autonomous agents with powerful write permissions could send wrong emails, delete critical files, or expose sensitive data.
- **The Good News:** These challenges are solvable. In upcoming lessons, we'll cover patterns for building reliable products through specialized evaluation and monitoring pipelines, strategies for building hybrid systems, and ways to keep costs and latency under control.
- **Your Path Forward:** By the end of this course, you'll have the knowledge to architect AI systems that are not only powerful but also robust, efficient, and safe. You'll know when to use workflows versus agents and how to build effective hybrid systems that work in the real world.
- To transition from this lesson to the next, specify what we will learn in future lessons. First mention what we will learn in next lesson, which is Lesson 3. Next leverage the concepts listed in subsection `Concepts That Will Be Introduced in Future Lessons` to make slight references to other topics we will learn during this course. To stay focused, specify only the ones that are present in this current lesson.
- **Section length:** 350 words

## Golden Sources

- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Real Agents vs. Workflows: The Truth Behind AI 'Agents'](https://www.youtube.com/watch?v=kQxr-uOxw2o&t=1s)
- [Exploring the difference between agents and workflows](https://decodingml.substack.com/p/llmops-for-production-agentic-rag)
- [A Developer’s Guide to Building Scalable AI: Workflows vs Agents](https://towardsdatascience.com/a-developers-guide-to-building-scalable-ai-workflows-vs-agents/)
- [Gemini CLI README.md](https://github.com/google-gemini/gemini-cli/blob/main/README.md)
